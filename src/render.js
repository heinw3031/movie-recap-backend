'use strict';
/**
 * Real FFmpeg rendering.
 *
 * What was fake in the Android app (MainViewModel.startVoiceAndSyncGeneration):
 *   delay(550ms) x 5 stages, then isMediaGenerated = true. No audio, no video.
 * What this module does:
 *   renderRecap() cuts a REAL clip from the source video for every scene,
 *   stretches/shrinks it only within 0.9x-1.1x so the clip lasts as long as
 *   the scene's real TTS audio, mixes that audio in, concats all scenes,
 *   and exports a real 720p MP4 (plus a 480p version) you can play.
 *
 * If a source clip is far shorter than its audio even at 0.9x, the clip is
 * extended forward into the following footage (continuous, no repeat) so
 * the narration never plays over a looped repeat of the same shots.
 * If it is far longer, it is trimmed to the audio length at natural speed.
 * Looping was removed 2026-11-16: job 25a6d122 proved it breaks sync —
 * scene 1 (45s window, 57s narration) replayed its footage while the
 * narration moved on, and 150s windows only showed their first ~47s while
 * the narration described the whole stretch. Scenes are now one beat
 * (30-60s) with narration sized to the beat, so the gentle 0.9x-1.1x
 * tempo path is the normal case and extend/trim are only safety nets.
 */
const fs = require('fs');
const path = require('path');
const { execFile } = require('child_process');
const { promisify } = require('util');

const execFileAsync = promisify(execFile);

async function ffprobeDuration(filePath) {
  const { stdout } = await execFileAsync('ffprobe', [
    '-v', 'error', '-show_entries', 'format=duration',
    '-of', 'default=noprint_wrappers=1:nokey=1', filePath,
  ]);
  const d = parseFloat(stdout.trim());
  if (!Number.isFinite(d) || d <= 0) throw new Error(`ffprobe could not read duration of ${filePath}`);
  return d;
}

async function ffprobeStreams(filePath) {
  const { stdout } = await execFileAsync('ffprobe', [
    '-v', 'error', '-show_entries', 'stream=codec_type,codec_name,width,height',
    '-of', 'json', filePath,
  ]);
  return JSON.parse(stdout).streams || [];
}

function clampSpeed(raw) {
  // Spec: speed adjust only 0.9x-1.1x.
  return Math.min(1.1, Math.max(0.9, raw));
}

/**
 * Cut one clip from the source video.
 * cutClip(input, startSec, durationSec, outPath)
 * Re-encodes to 720p/30fps/AAC so concat later is consistent.
 */
async function cutClip(input, startSec, durationSec, outPath) {
  if (durationSec <= 0) throw new Error('cutClip: duration must be > 0');
  fs.mkdirSync(path.dirname(outPath), { recursive: true });
  await execFileAsync('ffmpeg', [
    '-y', '-ss', String(startSec), '-i', input,
    '-t', String(durationSec),
    '-vf', 'scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30',
    '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23',
    '-c:a', 'aac', '-b:a', '128k',
    outPath,
  ]);
  return outPath;
}

/**
 * Render one scene segment: source clip + its TTS audio, same length.
 * Returns the segment path.
 */
async function renderSceneSegment({ sourceVideo, sourceStartSec, sourceEndSec, audioPath, audioDurationSec, outPath }) {
  const sourceDur = Math.max(0.1, sourceEndSec - sourceStartSec);
  // How the clip length is matched to the narration — NEVER loop:
  // - ratio in 0.9-1.1 (the normal beat-matched case): gentle tempo change
  //   so the clip lasts exactly as long as the audio.
  // - window much longer than the audio: natural speed, show the first
  //   audioDurationSec seconds of the window.
  // - window much shorter than the audio: natural speed, extend forward
  //   into the following footage (continuous, no repeat) to fill the audio.
  const ratio = sourceDur / audioDurationSec;
  let speed = 1;
  let videoLen = audioDurationSec;
  if (ratio >= 0.9 && ratio <= 1.1) {
    speed = clampSpeed(ratio);
    videoLen = sourceDur / speed;
  }

  const args = ['-y'];
  args.push('-ss', String(sourceStartSec), '-i', sourceVideo, '-i', audioPath);
  const filter = [
    '[0:v]trim=0:' + videoLen.toFixed(3) + ',setpts=PTS-STARTPTS,scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30[v]',
    '[1:a]atempo=' + speed.toFixed(3) + '[a]',
  ].join(';');
  args.push(
    '-filter_complex', filter,
    '-map', '[v]', '-map', '[a]',
    '-t', String(audioDurationSec),
    '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23',
    '-c:a', 'aac', '-b:a', '128k',
    '-shortest',
    outPath
  );
  fs.mkdirSync(path.dirname(outPath), { recursive: true });
  await execFileAsync('ffmpeg', args);
  return outPath;
}

/**
 * Fast path: cut one video-only clip with stream copy (no re-encode).
 * -ss before -i seeks to the nearest keyframe, so a boundary can be off by
 * ~1s — acceptable for a recap, and far faster than decoding + x264.
 */
async function renderSceneSegmentFast({ sourceVideo, sourceStartSec, audioDurationSec, outPath }) {
  if (audioDurationSec <= 0) throw new Error('renderSceneSegmentFast: duration must be > 0');
  fs.mkdirSync(path.dirname(outPath), { recursive: true });
  await execFileAsync('ffmpeg', [
    '-y', '-ss', String(sourceStartSec), '-i', sourceVideo,
    '-t', String(audioDurationSec),
    '-c:v', 'copy', '-an',
    outPath,
  ]);
  return outPath;
}

/**
 * Fast recap render: NO video re-encode at all.
 * Cuts each scene's video with stream copy, concats the segments with stream
 * copy, concats the narration MP3s (one quick AAC encode), and muxes them.
 * The 480p version still needs one scale pass (same as the slow path).
 *
 * Eligibility is strict — returns null (caller falls back to renderRecap)
 * unless EVERY scene qualifies:
 * - source video is h264 at 720p height, so -c copy segments form a
 *   standard MP4 (theora/vp9 sources and non-720p sources need the slow path
 *   to normalise codec + resolution);
 * - every scene's source segment lasts 0.9x-1.1x its narration and is at
 *   least as long as the narration (no speed change, no looping possible
 *   without re-encoding).
 * Any failure also returns null via the caller's try/catch — never throws.
 */
async function renderRecapFast({ sourceVideo, scenes, workDir, out720, out480 }) {
  const streams = await ffprobeStreams(sourceVideo);
  const v = streams.find((s) => s.codec_type === 'video');
  if (!v || v.codec_name !== 'h264' || v.height !== 720) return null;
  for (const s of scenes) {
    if (!s.audioPath || !fs.existsSync(s.audioPath)) return null;
    const audioDur = s.audioDurationSec || (await ffprobeDuration(s.audioPath));
    const sourceDur = Math.max(0.1, (s.sourceEndSec ?? 0) - (s.sourceStartSec ?? 0));
    const speed = sourceDur / audioDur;
    if (speed < 0.9 || speed > 1.1 || sourceDur < audioDur * 0.98) return null;
  }

  const segDir = path.join(workDir, 'segments-fast');
  fs.mkdirSync(segDir, { recursive: true });
  const segmentPaths = [];
  for (let i = 0; i < scenes.length; i++) {
    const s = scenes[i];
    const audioDur = s.audioDurationSec || (await ffprobeDuration(s.audioPath));
    const seg = path.join(segDir, `scene-${String(i + 1).padStart(2, '0')}.mp4`);
    await renderSceneSegmentFast({
      sourceVideo,
      sourceStartSec: s.sourceStartSec ?? 0,
      audioDurationSec: audioDur,
      outPath: seg,
    });
    segmentPaths.push(seg);
  }

  const vlist = path.join(workDir, 'concat-fast-video.txt');
  fs.writeFileSync(vlist, segmentPaths.map((p) => `file '${p.replace(/'/g, "'\\''")}'`).join('\n'));
  const videoOnly = path.join(workDir, 'video-only-fast.mp4');
  await execFileAsync('ffmpeg', [
    '-y', '-f', 'concat', '-safe', '0', '-i', vlist,
    '-c', 'copy',
    videoOnly,
  ]);

  const alist = path.join(workDir, 'concat-fast-audio.txt');
  fs.writeFileSync(alist, scenes.map((s) => `file '${s.audioPath.replace(/'/g, "'\\''")}'`).join('\n'));
  const narration = path.join(workDir, 'narration-fast.m4a');
  await execFileAsync('ffmpeg', [
    '-y', '-f', 'concat', '-safe', '0', '-i', alist,
    '-c:a', 'aac', '-b:a', '128k',
    narration,
  ]);

  fs.mkdirSync(path.dirname(out720), { recursive: true });
  await execFileAsync('ffmpeg', [
    '-y', '-i', videoOnly, '-i', narration,
    '-c:v', 'copy', '-c:a', 'copy',
    '-shortest',
    out720,
  ]);

  if (out480) {
    fs.mkdirSync(path.dirname(out480), { recursive: true });
    await execFileAsync('ffmpeg', [
      '-y', '-i', out720,
      '-vf', 'scale=854:480',
      '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23',
      '-c:a', 'copy',
      out480,
    ]);
  }

  const durationSec = await ffprobeDuration(out720);
  return { out720, out480: out480 || null, durationSec, fast: true };
}

/**
 * Render the full recap.
 * scenes: [{ sourceStartSec, sourceEndSec, audioPath, audioDurationSec }]
 * Returns { out720, out480, durationSec }.
 */
async function renderRecap({ sourceVideo, scenes, workDir, out720, out480 }) {
  if (!scenes || scenes.length === 0) throw new Error('renderRecap: no scenes with audio.');
  // Fast path first (RENDER_FAST=0 disables it): stream-copy cuts + mux,
  // no video re-encode — the slow re-encode below is the fallback.
  if (process.env.RENDER_FAST !== '0') {
    try {
      const fast = await renderRecapFast({ sourceVideo, scenes, workDir, out720, out480 });
      if (fast) {
        console.log('[render] fast path used: no video re-encode');
        return fast;
      }
      console.log('[render] fast path not eligible — using full re-encode');
    } catch (e) {
      console.log(`[render] fast path failed (${String(e.message).slice(0, 160)}) — falling back to full re-encode`);
    }
  }
  const segDir = path.join(workDir, 'segments');
  fs.mkdirSync(segDir, { recursive: true });

  const segmentPaths = [];
  for (let i = 0; i < scenes.length; i++) {
    const s = scenes[i];
    if (!s.audioPath || !fs.existsSync(s.audioPath)) {
      throw new Error(`renderRecap: scene ${i + 1} has no audio file — refusing to render a silent recap.`);
    }
    const audioDur = s.audioDurationSec || (await ffprobeDuration(s.audioPath));
    const seg = path.join(segDir, `scene-${String(i + 1).padStart(2, '0')}.mp4`);
    await renderSceneSegment({
      sourceVideo,
      sourceStartSec: s.sourceStartSec ?? 0,
      sourceEndSec: s.sourceEndSec ?? (s.sourceStartSec ?? 0) + audioDur,
      audioPath: s.audioPath,
      audioDurationSec: audioDur,
      outPath: seg,
    });
    segmentPaths.push(seg);
  }

  const listPath = path.join(workDir, 'concat.txt');
  fs.writeFileSync(listPath, segmentPaths.map((p) => `file '${p.replace(/'/g, "'\\''")}'`).join('\n'));

  fs.mkdirSync(path.dirname(out720), { recursive: true });
  await execFileAsync('ffmpeg', [
    '-y', '-f', 'concat', '-safe', '0', '-i', listPath,
    '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23',
    '-c:a', 'aac', '-b:a', '128k',
    out720,
  ]);

  if (out480) {
    fs.mkdirSync(path.dirname(out480), { recursive: true });
    await execFileAsync('ffmpeg', [
      '-y', '-i', out720,
      '-vf', 'scale=854:480',
      '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23',
      '-c:a', 'copy',
      out480,
    ]);
  }

  const durationSec = await ffprobeDuration(out720);
  return { out720, out480: out480 || null, durationSec };
}

module.exports = { ffprobeDuration, ffprobeStreams, cutClip, renderSceneSegment, renderRecap, renderRecapFast, clampSpeed };
