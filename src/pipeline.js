'use strict';
/**
 * Async job runner. Real progress only — no fake stages, no fake "done".
 *
 * Stages: queued -> analysing -> scripting -> awaiting-approval
 *         -> tts -> rendering -> done | failed
 *
 * runAnalysis(jobId):  analyse video frames with Gemini, then generate the
 *   Burmese script. Stops at awaiting-approval so a human can edit/approve.
 * runGenerate(jobId):  TTS per approved scene, then FFmpeg render.
 *   Any failure marks the job failed with the real error message.
 */
const fs = require('fs');
const path = require('path');
const store = require('./store');
const gemini = require('./gemini');
const edgeTts = require('./edgeTts');
const render = require('./render');

function setStage(jobId, status, progress, extra = {}) {
  return store.updateJob(jobId, { status, progress, ...extra });
}

function fail(jobId, err) {
  const message = err && err.message ? err.message : String(err);
  console.error(`[job ${jobId}] FAILED:`, message);
  return store.updateJob(jobId, { status: 'failed', error: message });
}

async function runAnalysis(jobId) {
  const job = store.getJob(jobId);
  if (!job) throw new Error(`Job ${jobId} not found`);
  const dir = store.jobDir(jobId);
  try {
    setStage(jobId, 'analysing', 10);
    const framesDir = path.join(dir, 'frames');
    const summaries = await gemini.analyzeVideo(job.sourcePath, framesDir);
    store.updateJob(jobId, { sceneSummaries: summaries });
    setStage(jobId, 'scripting', 30);

    const script = await gemini.generateScript({
      sceneSummaries: summaries,
      lengthMinutes: job.lengthMinutes,
      category: job.category,
      fileName: job.originalName,
    });
    // Give each scene a real source window; Gemini already returns sourceStart/End.
    const scenes = script.scenes.map((s, i) => ({
      sceneNumber: s.sceneNumber || i + 1,
      sourceStartSec: s.sourceStartSec ?? 0,
      sourceEndSec: s.sourceEndSec ?? 0,
      startSec: s.sourceStartSec ?? 0,
      endSec: s.sourceEndSec ?? 0,
      narrationMy: s.narrationMy || '',
      audioPath: null,
      audioDurationSec: null,
    }));
    setStage(jobId, 'awaiting-approval', 40, {
      title: script.title,
      scenes,
      error: null,
    });
    console.log(`[job ${jobId}] script ready: ${scenes.length} scenes, awaiting approval`);
    return store.getJob(jobId);
  } catch (err) {
    return fail(jobId, err);
  }
}

async function runGenerate(jobId) {
  const job = store.getJob(jobId);
  if (!job) throw new Error(`Job ${jobId} not found`);
  if (!job.scenes || job.scenes.length === 0) {
    return fail(jobId, new Error('No approved script scenes to generate from. Approve the script first.'));
  }
  const dir = store.jobDir(jobId);
  const audioDir = path.join(dir, 'audio');
  fs.mkdirSync(audioDir, { recursive: true });

  try {
    // --- TTS ---
    // TTS_ENGINE: 'gemini' (default — batched, ONE call per recap;
    // TTS_BATCH=0 restores one-call-per-scene), 'edge' (Microsoft Edge
    // TTS, my-MM-ThihaNeural male Burmese voice, no quota), 'auto'
    // (Gemini first, Edge fallback when Gemini's daily quota is gone).
    // The free tier counts REQUESTS (10/day/model), not minutes, so a
    // 5-min Gemini recap costs 1 batched request instead of 7.
    const total = job.scenes.length;
    const engine = process.env.TTS_ENGINE || 'gemini';
    let scenesWithAudio;
    if (engine === 'edge') {
      setStage(jobId, 'tts', 45);
      scenesWithAudio = await edgeTts.generateAllSceneAudioEdge(job.scenes, audioDir);
      store.updateJob(jobId, {
        scenes: scenesWithAudio,
        audioDurations: Object.fromEntries(
          scenesWithAudio.map((s) => [s.sceneNumber, s.audioDurationSec])
        ),
      });
      scenesWithAudio.forEach((s, i) =>
        console.log(`[job ${jobId}] TTS scene ${i + 1}/${total}: ${s.audioDurationSec.toFixed(1)}s (edge)`)
      );
      setStage(jobId, 'tts', 70);
    } else if (process.env.TTS_BATCH === '0') {
      scenesWithAudio = [];
      for (let i = 0; i < total; i++) {
        const scene = job.scenes[i];
        const progress = 40 + Math.round((i / total) * 30); // 40-70%
        setStage(jobId, 'tts', progress);
        const outMp3 = path.join(audioDir, `scene-${String(i + 1).padStart(2, '0')}.mp3`);
        // Throws if no API key / no audio — job fails honestly, never fakes it.
        const { durationSec } = await gemini.generateSceneAudio(scene.narrationMy, outMp3);
        scenesWithAudio.push({ ...scene, audioPath: outMp3, audioDurationSec: durationSec });
        store.updateJob(jobId, {
          scenes: scenesWithAudio.concat(job.scenes.slice(i + 1)),
          audioDurations: { ...job.audioDurations, [scene.sceneNumber]: durationSec },
        });
        console.log(`[job ${jobId}] TTS scene ${i + 1}/${total}: ${durationSec.toFixed(1)}s`);
      }
    } else {
      setStage(jobId, 'tts', 45);
      try {
        scenesWithAudio = await gemini.generateRecapAudio(job.scenes, audioDir);
      } catch (err) {
        if (engine === 'auto' && gemini.isDailyQuotaError(err)) {
          console.log(`[job ${jobId}] Gemini daily TTS quota exhausted — falling back to Edge TTS`);
          scenesWithAudio = await edgeTts.generateAllSceneAudioEdge(job.scenes, audioDir);
        } else {
          throw err;
        }
      }
      store.updateJob(jobId, {
        scenes: scenesWithAudio,
        audioDurations: Object.fromEntries(
          scenesWithAudio.map((s) => [s.sceneNumber, s.audioDurationSec])
        ),
      });
      scenesWithAudio.forEach((s, i) =>
        console.log(`[job ${jobId}] TTS scene ${i + 1}/${total}: ${s.audioDurationSec.toFixed(1)}s`)
      );
      setStage(jobId, 'tts', 70);
    }

    // --- Render ---
    setStage(jobId, 'rendering', 75);
    const out720 = path.join(dir, 'final_720p.mp4');
    const out480 = path.join(dir, 'final_480p.mp4');
    const { durationSec } = await render.renderRecap({
      sourceVideo: job.sourcePath,
      scenes: scenesWithAudio,
      workDir: dir,
      out720,
      out480,
    });
    setStage(jobId, 'rendering', 95);

    // Free the original upload after a successful render (copyright + storage).
    // Keep it if KEEP_SOURCE=1 (local debugging).
    if (process.env.KEEP_SOURCE !== '1') {
      try { fs.unlinkSync(job.sourcePath); } catch (_) { /* already gone */ }
    }

    const done = setStage(jobId, 'done', 100, {
      scenes: scenesWithAudio,
      finalVideoUrl: `/api/jobs/${jobId}/video`,
      finalVideo720p: 'final_720p.mp4',
      finalVideo480p: 'final_480p.mp4',
      error: null,
    });
    console.log(`[job ${jobId}] done: final recap ${durationSec.toFixed(1)}s`);
    return done;
  } catch (err) {
    return fail(jobId, err);
  }
}

module.exports = { runAnalysis, runGenerate, setStage, fail };
