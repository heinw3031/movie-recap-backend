'use strict';
/**
 * Real Gemini calls for the Myanmar Movie Recap backend.
 *
 * What was fake in the Android app (RecapScriptRepository.kt):
 *   prompt used ONLY fileName + category + length — the video was never watched,
 *   so a generic name like "movie.mp4" made Gemini invent the story.
 * What this module does instead:
 *   analyseVideo() extracts REAL frames from the uploaded video (1 frame / 30s)
 *   and sends those frames to Gemini 3.8 Flash, so the summary comes from
 *   what is actually on screen. The filename is passed only as a label.
 *
 * generateSceneAudio() calls Gemini TTS for real. If there is no API key,
 * or Gemini returns no audio, it THROWS — it never returns a fake "done".
 *
 * No API key is stored in code. Set GEMINI_API_KEY in the environment.
 */
const fs = require('fs');
const path = require('path');
const { execFile } = require('child_process');
const { promisify } = require('util');

const execFileAsync = promisify(execFile);

// 2026-11-15: Gemini free tier is strict — gemini-3.8-flash returns 503
// "high demand" when overloaded, and gemini-3.8-flash-tts allows only
// 3 requests/minute (429 RESOURCE_EXHAUSTED, "retry in 48s"). The pipeline
// used to fail the whole job on the first such error. Now every Gemini
// call retries with backoff, and TTS calls are spaced out so the quota
// is never hit in the first place.
function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function retryDelayFromError(err) {
  const m = String((err && err.message) || '').match(/retry in ([\d.]+)s/i);
  if (m) return Math.ceil(parseFloat(m[1]) * 1000);
  return null;
}

function isDailyQuotaError(err) {
  const msg = String((err && err.message) || '');
  return /per day|daily|GenerateRequestsPerDay/i.test(msg);
}

function isRetryableError(err) {
  // A daily-quota 429 is NOT retryable: no wait fixes it until tomorrow's
  // reset, so retrying would only burn minutes and hide the real cause.
  if (isDailyQuotaError(err)) return false;
  const msg = String((err && err.message) || '');
  return (
    /429|503|UNAVAILABLE|RESOURCE_EXHAUSTED|fetch failed|ECONNRESET|ETIMEDOUT/i.test(msg)
  );
}

async function withRetry(fn, what) {
  const delays = [15000, 30000, 60000, 60000, 90000];
  let lastErr;
  for (let attempt = 0; attempt <= delays.length; attempt++) {
    try {
      return await fn();
    } catch (err) {
      lastErr = err;
      if (!isRetryableError(err) || attempt === delays.length) throw err;
      const hinted = retryDelayFromError(err);
      const wait = Math.min(Math.max(hinted || delays[attempt], delays[attempt]), 120000);
      console.log(`[gemini] ${what} failed (attempt ${attempt + 1}): ${String(err.message).slice(0, 120)} — retrying in ${Math.round(wait / 1000)}s`);
      await sleep(wait);
    }
  }
  throw lastErr;
}

// Free-tier TTS quota is 3 requests/minute per model. Space TTS calls
// at least 21s apart so a 7-scene (or 38-scene) script never trips it.
let lastTtsAt = 0;
async function throttleTts() {
  const gapMs = 21000;
  const wait = lastTtsAt + gapMs - Date.now();
  if (wait > 0) await sleep(wait);
  lastTtsAt = Date.now();
}

// 2026-11-15: gemini-2.5-flash returns 404 "no longer available to new
// users" for this API key, and gemini-2.5-flash-preview-tts is the same
// generation — both replaced with the 3.8 generation (verified live in
// GET /v1beta/models on 2026-11-15: gemini-3.8-flash and
// gemini-3.8-flash-tts both exist). Override via env if Gemini retires them.
// 2026-11-15: the free-tier DAILY TTS quota is 10 requests/day PER MODEL
// (verified: job bd6b52c2 died at scene 7 with "GenerateRequestsPerDayPerModel-FreeTier").
// One 5-min recap needs 7 per-scene calls, so the old path allowed ~1 recap/day.
// Two fixes live here: (a) rotate across both surviving 3.8 TTS models
// (flash-tts + flash-lite-tts, same schema and voices), and (b) batch a whole
// recap into ONE call (see generateRecapAudio) — one call returns ~11 min of
// audio, so a 5-min recap costs 1 request instead of 7.
// SCRIPT_MODEL: the script call is text-only. gemini-3.8-flash text is 20 RPD,
// which caps recaps at 10/day even with TTS fixed; gemini-3.5-flash-lite is
// 500 RPD. Default stays on the proven 3.8-flash until lite script quality
// is tested — switch with SCRIPT_MODEL=gemini-3.5-flash-lite.
const ANALYSIS_MODEL = process.env.ANALYSIS_MODEL || 'gemini-3.8-flash';
const SCRIPT_MODEL = process.env.SCRIPT_MODEL || ANALYSIS_MODEL;
const TTS_MODELS = (process.env.TTS_MODELS || process.env.TTS_MODEL || 'gemini-3.8-flash-tts,gemini-3.8-flash-lite-tts')
  .split(',')
  .map((s) => s.trim())
  .filter(Boolean);
const TTS_MODEL = TTS_MODELS[0];
// Models whose DAILY quota is exhausted (reset at midnight Pacific).
// A daily-quota 429 is never retried — waiting cannot fix it — the call
// fails over to the next model immediately instead.
const exhaustedTtsModels = new Set();
// Burmese TTS voice. 'Kore' is a documented prebuilt voice for the TTS model;
// change via TTS_VOICE env if Gemini adds a my-MM-specific voice.
const DEFAULT_TTS_VOICE = process.env.TTS_VOICE || 'Kore';

function requireApiKey() {
  const key = process.env.GEMINI_API_KEY;
  if (!key || !key.trim()) {
    throw new Error(
      'GEMINI_API_KEY is not set. Get a free key in Google AI Studio (Get API key), ' +
        'then set GEMINI_API_KEY in .env or in the environment. ' +
        'No script, audio or analysis is faked when the key is missing — the job fails with this error.'
    );
  }
  return key;
}

function getClient() {
  // Lazy require so `node --check` / render-only tests do not need the SDK installed.
  const { GoogleGenAI } = require('@google/genai');
  return new GoogleGenAI({ apiKey: requireApiKey() });
}

function stripJsonFences(text) {
  if (!text) return text;
  return text
    .replace(/^```(?:json)?\s*/i, '')
    .replace(/\s*```\s*$/, '')
    .trim();
}

function parseJsonResponse(text, what) {
  const cleaned = stripJsonFences(text);
  try {
    return JSON.parse(cleaned);
  } catch (e) {
    const start = cleaned.indexOf('{');
    const end = cleaned.lastIndexOf('}');
    if (start >= 0 && end > start) {
      return JSON.parse(cleaned.slice(start, end + 1));
    }
    throw new Error(`Gemini returned invalid JSON for ${what}: ${e.message}. Raw: ${cleaned.slice(0, 500)}`);
  }
}

async function ffprobeDuration(filePath) {
  const { stdout } = await execFileAsync('ffprobe', [
    '-v', 'error', '-show_entries', 'format=duration',
    '-of', 'default=noprint_wrappers=1:nokey=1', filePath,
  ]);
  const d = parseFloat(stdout.trim());
  if (!Number.isFinite(d) || d <= 0) throw new Error(`ffprobe could not read duration of ${filePath}`);
  return d;
}

/** Extract 1 frame every `intervalSec` seconds. Returns [{ path, atSec }]. */
async function extractFrames(videoPath, outDir, intervalSec = 30) {
  fs.mkdirSync(outDir, { recursive: true });
  const duration = await ffprobeDuration(videoPath);
  // fps=1/30 alone emits NO frame for videos shorter than 30s (and drops the
  // first frame of longer videos) — round=up makes the t=0 frame emit.
  // format=yuvj420p is required for the MJPEG encoder in ffmpeg 8.x, which
  // rejects non-full-range YUV ("Non full-range YUV is non-standard").
  await execFileAsync('ffmpeg', [
    '-y', '-i', videoPath,
    '-vf', `fps=1/${intervalSec}:round=up,format=yuvj420p`,
    '-q:v', '3',
    path.join(outDir, 'frame_%04d.jpg'),
  ]);
  const files = fs.readdirSync(outDir).filter((f) => f.endsWith('.jpg')).sort();
  return files.map((f, i) => ({ path: path.join(outDir, f), atSec: i * intervalSec, duration }));
}

/**
 * Analyse the video for real: send extracted frames to Gemini 3.8 Flash.
 * Returns [{ startSec, endSec, description }] covering the video.
 * Never invents a story from the filename alone — if no frames could be
 * extracted, it throws instead of guessing.
 */
async function analyzeVideo(videoPath, framesDir) {
  const ai = getClient();
  const frames = await extractFrames(videoPath, framesDir, 30);
  if (frames.length === 0) {
    throw new Error('No frames could be extracted from the uploaded video — cannot analyse it. The video may be corrupt or in an unsupported format.');
  }
  const duration = frames[0].duration;

  const parts = [];
  for (const frame of frames) {
    parts.push({
      inlineData: {
        mimeType: 'image/jpeg',
        data: fs.readFileSync(frame.path).toString('base64'),
      },
    });
    parts.push({ text: `Frame at ${frame.atSec}s of ${Math.round(duration)}s.` });
  }
  parts.push({
    text: [
      'These are real frames, in time order, from the uploaded video.',
      'Do NOT invent a plot from the filename. Describe only what you can see in these frames.',
      'Return JSON only, no markdown, in this exact shape:',
      '{"scenes":[{"startSec":0,"endSec":90,"description":"what happens in this stretch, in English, 1-2 sentences"}]}',
      'Cover the whole video from 0s to the end, in chronological order. Merge similar stretches; aim for 6-15 scenes.',
      `Video duration is ${Math.round(duration)} seconds.`,
    ].join('\n'),
  });

  const response = await withRetry(
    () =>
      ai.models.generateContent({
        model: ANALYSIS_MODEL,
        contents: [{ parts }],
      }),
    'video analysis'
  );
  const text = response.text;
  if (!text) throw new Error('Gemini analysis returned no text.');
  const parsed = parseJsonResponse(text, 'video analysis');
  const scenes = Array.isArray(parsed.scenes) ? parsed.scenes : [];
  if (scenes.length === 0) throw new Error('Gemini analysis returned no scenes.');
  return scenes.map((s) => ({
    startSec: Number(s.startSec) || 0,
    endSec: Number(s.endSec) || 0,
    description: String(s.description || '').trim(),
  }));
}

const SCENE_COUNT_BY_MINUTES = { 5: 7, 10: 13, 15: 20, 30: 38 };

/**
 * Generate a Burmese scene-by-scene recap script from REAL scene summaries.
 * Returns { title, scenes: [{ sceneNumber, startSec, endSec, sourceStartSec, sourceEndSec, narrationMy }] }
 * sourceStartSec/sourceEndSec are the timestamps in the ORIGINAL video to cut
 * the clip from — this is what makes the later render match the narration.
 */
async function generateScript({ sceneSummaries, lengthMinutes, category, fileName }) {
  const ai = getClient();
  if (![5, 10, 15, 30].includes(lengthMinutes)) {
    throw new Error(`Unsupported recap length ${lengthMinutes}. Use 5, 10, 15 or 30 minutes.`);
  }
  const sceneCount = SCENE_COUNT_BY_MINUTES[lengthMinutes];
  const summaryText = sceneSummaries
    .map((s) => `${s.startSec}s-${s.endSec}s: ${s.description}`)
    .join('\n');

  const prompt = [
    'You are a professional Myanmar (Burmese) movie-recap narrator for YouTube/Facebook.',
    'Write a gripping, natural Burmese-language recap script, divided into exactly ' + sceneCount + ' scenes,',
    `for a ${lengthMinutes}-minute recap video. Genre: ${category}.`,
    'Base the story ONLY on these real scene summaries from the video — do not invent characters, twists or an ending that is not in the summaries:',
    summaryText,
    'Requirements (return JSON only, no markdown):',
    '{"title":"catchy Burmese title","scenes":[{"sceneNumber":1,"sourceStartSec":0,"sourceEndSec":90,"narrationMy":"..."}]}',
    '- "scenes": exactly ' + sceneCount + ' items, in story order.',
    '- "sourceStartSec"/"sourceEndSec" must be real timestamps from the summaries above, so the clip cut from the source video matches this scene.',
    '- "narrationMy": natural, expressive Burmese Unicode, 2-3 SHORT sentences per scene, suitable for voiceover. Short sentences help Burmese TTS pronounce English names and numbers. Write English movie/character names in a simple phonetic way where needed.',
    '- The file was uploaded as "' + (fileName || 'video.mp4') + '" — use it only as a label, never as the source of the plot.',
  ].join('\n');

  const response = await withRetry(
    () =>
      ai.models.generateContent({
        model: SCRIPT_MODEL,
        contents: prompt,
        config: { responseMimeType: 'application/json' },
      }),
    'script generation'
  );
  const text = response.text;
  if (!text) throw new Error('Gemini script generation returned no text.');
  const parsed = parseJsonResponse(text, 'Burmese script');
  const scenes = Array.isArray(parsed.scenes) ? parsed.scenes : [];
  if (scenes.length === 0) throw new Error('Gemini script generation returned no scenes.');
  return {
    title: String(parsed.title || '').trim() || 'မြန်မာ Recap',
    scenes: scenes.map((s, i) => ({
      sceneNumber: Number(s.sceneNumber) || i + 1,
      startSec: Number(s.sourceStartSec) || 0,
      endSec: Number(s.sourceEndSec) || 0,
      sourceStartSec: Number(s.sourceStartSec) || 0,
      sourceEndSec: Number(s.sourceEndSec) || 0,
      narrationMy: String(s.narrationMy || '').trim(),
    })),
  };
}

/** Write a WAV header for 16-bit mono PCM at 24 kHz (Gemini TTS output format). */
function writePcmAsWav(pcmBuffer, outPath) {
  const sampleRate = 24000;
  const numChannels = 1;
  const bitsPerSample = 16;
  const byteRate = (sampleRate * numChannels * bitsPerSample) / 8;
  const blockAlign = (numChannels * bitsPerSample) / 8;
  const header = Buffer.alloc(44);
  header.write('RIFF', 0);
  header.writeUInt32LE(36 + pcmBuffer.length, 4);
  header.write('WAVE', 8);
  header.write('fmt ', 12);
  header.writeUInt32LE(16, 16);
  header.writeUInt16LE(1, 20); // PCM
  header.writeUInt16LE(numChannels, 22);
  header.writeUInt32LE(sampleRate, 24);
  header.writeUInt32LE(byteRate, 28);
  header.writeUInt16LE(blockAlign, 32);
  header.writeUInt16LE(bitsPerSample, 34);
  header.write('data', 36);
  header.writeUInt32LE(pcmBuffer.length, 40);
  fs.writeFileSync(outPath, Buffer.concat([header, pcmBuffer]));
}

/**
 * Write audio bytes to a WAV file. gemini-3.8 TTS returns a complete RIFF
 * WAV over generateContent; older models returned headerless PCM. Writing a
 * second header on top of a complete WAV corrupts it silently — detect it
 * and write the bytes as-is.
 */
function audioBufferToWav(buffer, outPath) {
  if (buffer.length >= 4 && buffer.toString('ascii', 0, 4) === 'RIFF') {
    fs.writeFileSync(outPath, buffer);
    return;
  }
  writePcmAsWav(buffer, outPath);
}

/**
 * One Gemini TTS call, rotating across TTS_MODELS. A model whose DAILY
 * quota is exhausted is skipped for the rest of the process (a daily 429
 * is never retried — waiting cannot fix it until tomorrow's reset).
 * Returns the raw audio bytes. Throws when every model is exhausted or
 * the call fails after retries.
 */
async function generateTtsAudioBuffer(text, voiceName, what) {
  const ai = getClient();
  let lastErr;
  for (const model of TTS_MODELS) {
    if (exhaustedTtsModels.has(model)) continue;
    await throttleTts();
    try {
      const response = await withRetry(
        () =>
          ai.models.generateContent({
            model,
            contents: [{ parts: [{ text }] }],
            config: {
              responseModalities: ['AUDIO'],
              speechConfig: {
                voiceConfig: { prebuiltVoiceConfig: { voiceName } },
              },
            },
          }),
        `${what} (${model})`
      );
      const parts =
        (response.candidates &&
          response.candidates[0] &&
          response.candidates[0].content &&
          response.candidates[0].content.parts) ||
        [];
      const audioPart = parts.find((p) => p.inlineData && p.inlineData.data);
      if (!audioPart) {
        throw new Error(
          'Gemini TTS returned no audio. Check your GEMINI_API_KEY, ' +
            'the TTS model quota, and that the text is not empty. The job is marked failed — audio is never faked.'
        );
      }
      const buffer = Buffer.from(audioPart.inlineData.data, 'base64');
      if (buffer.length === 0) throw new Error('Gemini TTS returned empty audio data.');
      return buffer;
    } catch (err) {
      lastErr = err;
      if (isDailyQuotaError(err)) {
        exhaustedTtsModels.add(model);
        console.log(`[gemini] ${model} daily TTS quota exhausted — failing over to the next model.`);
        continue;
      }
      throw err;
    }
  }
  throw lastErr || new Error('Gemini TTS failed on every model (daily quota exhausted for the day).');
}

/**
 * Generate REAL Burmese speech for one scene with Gemini TTS.
 * Returns { path (MP3), durationSec }.
 * Throws a clear error when there is no API key or no audio is returned —
 * it never pretends to have generated audio.
 */
async function generateSceneAudio(textMy, outMp3Path, voiceName = DEFAULT_TTS_VOICE) {
  if (!textMy || !textMy.trim()) throw new Error('generateSceneAudio: empty narration text.');
  const buffer = await generateTtsAudioBuffer(textMy, voiceName, 'scene TTS');
  const wavPath = outMp3Path.replace(/\.mp3$/i, '.wav');
  audioBufferToWav(buffer, wavPath);
  await execFileAsync('ffmpeg', ['-y', '-i', wavPath, '-codec:a', 'libmp3lame', '-q:a', '4', outMp3Path]);
  fs.unlinkSync(wavPath);
  const durationSec = await ffprobeDuration(outMp3Path);
  return { path: outMp3Path, durationSec };
}

/** Per-scene TTS for a list of scenes. Returns scenes with audioPath + audioDurationSec. */
async function generateAllSceneAudio(scenes, audioDir, voiceName = DEFAULT_TTS_VOICE) {
  fs.mkdirSync(audioDir, { recursive: true });
  const out = [];
  for (let i = 0; i < scenes.length; i++) {
    const scene = scenes[i];
    const num = scene.sceneNumber || i + 1;
    const mp3 = path.join(audioDir, `scene_${num}.mp3`);
    const { durationSec } = await generateSceneAudio(scene.narrationMy, mp3, voiceName);
    out.push({ ...scene, audioPath: mp3, audioDurationSec: durationSec });
  }
  return out;
}

/**
 * Split one MP3 into N segments at its longest silences.
 * Returns [{ path, durationSec }] or null when the split is not trustworthy
 * (too few silences, or a segment shorter than 0.5s) — the caller then
 * falls back to per-scene TTS instead of shipping a bad split.
 */
async function splitAudioBySilence(wholeMp3, scenes, workDir) {
  const sceneCount = scenes.length;
  const totalSec = await ffprobeDuration(wholeMp3);
  if (!totalSec || totalSec <= 0) return null;
  let stderr = '';
  try {
    const res = await execFileAsync('ffmpeg', [
      '-i', wholeMp3,
      '-af', 'silencedetect=noise=-30dB:d=1.0',
      '-f', 'null', '-',
    ]);
    stderr = res.stderr || '';
  } catch (e) {
    stderr = (e && e.stderr) || '';
  }
  const silences = [];
  let pendingStart = null;
  for (const line of stderr.split('\n')) {
    const s = line.match(/silence_start:\s*([\d.]+)/);
    if (s) {
      pendingStart = parseFloat(s[1]);
      continue;
    }
    const e = line.match(/silence_end:\s*([\d.]+)\s*\|\s*silence_duration:\s*([\d.]+)/);
    if (e && pendingStart !== null) {
      silences.push({ start: pendingStart, end: parseFloat(e[1]), duration: parseFloat(e[2]) });
      pendingStart = null;
    }
  }
  const needed = sceneCount - 1;
  if (silences.length < needed) return null;
  // Take the N-1 LONGEST silences as the scene boundaries (natural
  // sentence pauses inside a scene are ~0.2-0.5s; the inserted
  // <long pause> between scenes is ~1.2s+, so the longest ones win).
  const boundaries = silences
    .slice()
    .sort((a, b) => b.duration - a.duration)
    .slice(0, needed)
    .sort((a, b) => a.start - b.start)
    .map((s) => (s.start + s.end) / 2);
  const cuts = [0, ...boundaries, totalSec];
  const segments = [];
  for (let i = 0; i < sceneCount; i++) {
    const num = scenes[i].sceneNumber || i + 1;
    const out = path.join(workDir, `scene_${num}.mp3`);
    // -ss/-to BEFORE -i are input options: absolute timestamps in the file.
    await execFileAsync('ffmpeg', [
      '-y', '-ss', String(cuts[i]), '-to', String(cuts[i + 1]),
      '-i', wholeMp3, '-c', 'copy', out,
    ]);
    const durationSec = await ffprobeDuration(out);
    if (!durationSec || durationSec < 0.5) return null;
    segments.push({ path: out, durationSec });
  }
  return segments;
}

/**
 * Batched TTS: one Gemini call for a whole chunk of scenes, then split the
 * audio back into per-scene clips. This is the fix for the 10/day free-tier
 * ceiling — the quota counts REQUESTS, not minutes, and one call returns
 * ~11 min of audio, so a 5-min recap costs 1 request instead of 7.
 * Scenes are joined with <long pause> (a documented Gemini TTS inline tag)
 * so the split can find the boundaries; if the split fails, only that
 * chunk falls back to per-scene calls — a bad split never fails the job.
 * Returns scenes with audioPath + audioDurationSec.
 */
async function generateRecapAudio(scenes, audioDir, voiceName = DEFAULT_TTS_VOICE) {
  fs.mkdirSync(audioDir, { recursive: true });
  // Group scenes so each chunk is at most ~9 min of estimated audio
  // (one call returns ~11 min max). ~20 Burmese chars ≈ 1s of speech is a
  // conservative estimate from the proven per-scene jobs.
  const MAX_CHUNK_SEC = 540;
  const chunks = [];
  let cur = [];
  let curSec = 0;
  for (const scene of scenes) {
    const est = Math.max(3, (scene.narrationMy || '').length / 20);
    if (cur.length > 0 && curSec + est > MAX_CHUNK_SEC) {
      chunks.push(cur);
      cur = [];
      curSec = 0;
    }
    cur.push(scene);
    curSec += est;
  }
  if (cur.length > 0) chunks.push(cur);

  const out = [];
  for (let c = 0; c < chunks.length; c++) {
    const chunk = chunks[c];
    const joined = chunk.map((s) => s.narrationMy).join('\n<long pause>\n');
    const wholeMp3 = path.join(audioDir, `whole_${c}.mp3`);
    const wavPath = path.join(audioDir, `whole_${c}.wav`);
    const buffer = await generateTtsAudioBuffer(
      joined,
      voiceName,
      `batched TTS chunk ${c + 1}/${chunks.length}`
    );
    audioBufferToWav(buffer, wavPath);
    await execFileAsync('ffmpeg', ['-y', '-i', wavPath, '-codec:a', 'libmp3lame', '-q:a', '4', wholeMp3]);
    fs.unlinkSync(wavPath);
    const segments = await splitAudioBySilence(wholeMp3, chunk, audioDir);
    if (!segments) {
      console.log(`[gemini] batched split failed for chunk ${c + 1} — falling back to per-scene TTS for its ${chunk.length} scenes`);
      const fallback = await generateAllSceneAudio(chunk, audioDir, voiceName);
      out.push(...fallback);
    } else {
      chunk.forEach((scene, i) =>
        out.push({ ...scene, audioPath: segments[i].path, audioDurationSec: segments[i].durationSec })
      );
    }
    try {
      fs.unlinkSync(wholeMp3);
    } catch (_) {}
  }
  return out;
}

module.exports = {
  ANALYSIS_MODEL,
  SCRIPT_MODEL,
  TTS_MODEL,
  TTS_MODELS,
  requireApiKey,
  ffprobeDuration,
  extractFrames,
  analyzeVideo,
  generateScript,
  generateSceneAudio,
  generateRecapAudio,
  splitAudioBySilence,
  SCENE_COUNT_BY_MINUTES,
};
