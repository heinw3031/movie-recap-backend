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
// ANALYSIS_MODE: 'frames' (default — extract 1 frame/30s, send as images)
// or 'files' (Gemini Files API — upload the whole video once, Gemini watches
// AND listens, so dialogue-driven scenes are no longer missed, and there is
// no frame-extraction step at all). 'files' falls back to 'frames' on error.
const ANALYSIS_MODE = process.env.ANALYSIS_MODE || 'frames';
// MOMENT_CUT: '1' = moment-level cutting (user direction 2026-11-16 20:07:
// "if the narration says the boy and the girl walk on the bridge, the video
// MUST show them walking at that moment"). Each scene's narration is split
// into one segment per sentence, each segment cut from the exact moment its
// sentence describes — narration moment = picture moment, cut by cut.
// Default off: scene-level rendering (current proven path) is unchanged.
// When on, script returns segments, TTS splits audio per segment (two-level
// silence split, scene first then sentence), and render cuts per segment.
// Any failure at any level falls back to the scene-level path — a bad
// moment split never fails the job.
const MOMENT_CUT = process.env.MOMENT_CUT === '1';
// VERIFY_MOMENTS: '0' = off, anything else (including unset) = ON.
// Moment verification (2026-11-17): the analysis model's moments are
// sometimes mis-grounded — the timestamp does not show what the moment
// claims (job 6b70028f, 3.8-flash analysis: @98s "holographic cables and
// power conduits pulse with light" but the source frame at 98s is a
// sniper-scope view of a flying war robot over rooftops; @432s "Celia
// releases Thom, her projection dissolves" but the frame is Barley aiming
// a rifle on a rooftop; 3.6-flash did the same: @259s "officer puts hand
// on Thom's shoulder" over an empty street). The renderer is proven
// correct (output frames are pixel-identical to the source at the scene's
// timestamps) and the narration faithfully follows the moments, so the
// fault is the moments themselves — verification anchors the narration to
// moments PROVEN to be in the footage. A focused yes/no about one frame
// is far easier for the model than generating correct timestamps in the
// first place. A moment that fails is DROPPED (its sentence is dropped
// too, by the existing skip rule); a moment whose event is visible nearby
// is MOVED to the correct timestamp. Default ON: the live test needs it.
// Set VERIFY_MOMENTS=0 in Railway to switch it off without a code
// redeploy. Any verification failure falls back to the unverified moments
// — a bad verification never fails the job.
// Quota: up to MOMENTS_PER_VERIFY_CALL moments are batched into ONE
// generateContent call (one inline image per moment + its claim), grouped
// by script chunk (40 beats). A 5-min test (~50 beats, ~100 moments)
// costs ~3 verification calls; a 30-min test (~500 beats, ~700 moments)
// costs ~18 on top of ~13 script + ~13 expansion calls, so a long recap
// may exceed one project's 20/day — switch it off for long recaps if
// quota is the binding constraint.
const VERIFY_MOMENTS = process.env.VERIFY_MOMENTS !== '0';
const MOMENTS_PER_VERIFY_CALL = 40;
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
async function detectFirstNonBlackSec(videoPath) {
  // Best-effort: find where the opening fade-from-black ends, so the first
  // beat starts on real picture instead of black frames (Tears of Steel
  // fades in from black for ~9s; the lite analysis started its first beat
  // at 0s and the recap opened on 9s of black). Returns null on any error.
  try {
    const { stdout, stderr } = await execFileAsync(
      'ffmpeg',
      ['-i', videoPath, '-vf', 'blackdetect=d=0.1:pix_th=0.10', '-f', 'null', '-'],
      { timeout: 60000 }
    );
    const log = `${stdout}\n${stderr}`;
    const m = log.match(/black_start:0(?:\.0+)?\s+black_end:([0-9.]+)/);
    if (m) return Math.ceil(parseFloat(m[1]));
  } catch (_) { /* fall through */ }
  return null;
}

async function analyzeVideo(videoPath, framesDir, firstNonBlackSec = null) {
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
      'Describe ONLY what is literally visible — what people and robots actually DO, in plain words. Never invent motion, weather, lighting or an action that is not on screen: a robot standing still is NOT "powering up", a bright room is NOT "a smoggy dawn skyline".',
      'Return JSON only, no markdown, in this exact shape:',
      '{"beats":[{"startSec":0,"endSec":12,"description":"what this beat is about, in English, 1-2 sentences","moments":[{"atSec":2,"what":"one short clause: what is seen/heard at this moment"},{"atSec":8,"what":"..."}]}]}',
      'BEATS, not chapters: split the video into short beats of 8-15 seconds each (about one beat per 10 seconds). Each beat is ONE micro-event — one thing happening, one place, one action — never a summary of several seconds, never several actions in one beat. One beat = one narration sentence.',
      'Inside EVERY beat, list 1-2 key moments in time order with their approximate timestamp (atSec) and one short clause each (what is actually seen or heard at that exact moment). Moments are the ground truth the narration will be timed against — be precise, never invent a moment.',
      'LITERAL ONLY: describe ONLY what is literally visible/audible in the frames at that moment — a still object, a cutaway or a memory/flashback is just that. Never invent motion, weather, time of day, or a power-up/awakening that is not actually on screen. If a beat cross-cuts between places, say so and list what each cut actually shows.',
      'Skip pure black screens, fade-ins, fade-outs and end credits entirely — no beat may be mostly black, and the first beat starts when the picture actually begins.' + (firstNonBlackSec !== null ? ` The video fades in from black; the first real picture is at about ${firstNonBlackSec}s — the first beat must start at or after ${firstNonBlackSec}s.` : ''),
      'Cover the whole video from the first real picture to the end, in chronological order, beats back-to-back with no gaps.',
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
  const scenes = Array.isArray(parsed.beats) ? parsed.beats : parsed.scenes;
  if (!Array.isArray(scenes) || scenes.length === 0) throw new Error('Gemini analysis returned no scenes.');
  return scenes.map((s) => ({
    startSec: Number(s.startSec) || 0,
    endSec: Number(s.endSec) || 0,
    description: String(s.description || '').trim(),
    moments: (Array.isArray(s.moments) ? s.moments : [])
      .map((m) => ({ atSec: Number(m.atSec) || 0, what: String(m.what || '').trim() }))
      .filter((m) => m.what),
  }));
}

const SCENE_COUNT_BY_MINUTES = { 5: 7, 10: 13, 15: 20, 30: 38, 60: 76 };

function mimeTypeForVideo(filePath) {
  const ext = path.extname(filePath).toLowerCase();
  return (
    {
      '.mp4': 'video/mp4',
      '.mov': 'video/quicktime',
      '.webm': 'video/webm',
      '.ogv': 'video/ogg',
      '.ogg': 'video/ogg',
      '.mkv': 'video/x-matroska',
    }[ext] || 'video/mp4'
  );
}

async function waitForFileActive(ai, name, timeoutMs = 120000) {
  const started = Date.now();
  for (;;) {
    const f = await ai.files.get({ name });
    if (f.state === 'ACTIVE') return f;
    if (f.state === 'FAILED') {
      throw new Error(`Gemini Files API: uploaded file entered FAILED state (${name}).`);
    }
    if (Date.now() - started > timeoutMs) {
      throw new Error('Gemini Files API: timed out waiting for the uploaded video to become ACTIVE.');
    }
    await sleep(5000);
  }
}

/**
 * Analyse with the Gemini Files API: upload the whole video once and let
 * Gemini watch AND listen to it (picture + dialogue/audio). One call for
 * the whole video, no frame extraction, and dialogue-driven twists are no
 * longer missed. The uploaded file is deleted afterwards. Throws on failure
 * — the caller (analyzeVideoAuto) falls back to frame analysis.
 */
async function analyzeVideoWithFiles(videoPath, firstNonBlackSec = null) {
  const ai = getClient();
  const mimeType = mimeTypeForVideo(videoPath);
  const duration = await ffprobeDuration(videoPath);
  const file = await withRetry(
    () => ai.files.upload({ file: videoPath, config: { mimeType } }),
    'video upload (Files API)'
  );
  try {
    await waitForFileActive(ai, file.name);
    const response = await withRetry(
      () =>
        ai.models.generateContent({
          model: ANALYSIS_MODEL,
          contents: [
            {
              parts: [
                { fileData: { mimeType, fileUri: file.uri } },
                {
                  text: [
                    'Watch AND listen to this video (picture and sound/dialogue).',
                    'Do NOT invent a plot from the filename. Describe only what you can see and hear.',
                    'Describe ONLY what is literally visible and audible — what people and robots actually DO, in plain words. Never invent motion, weather, lighting or an action that is not on screen: a robot standing still is NOT "powering up", a bright room is NOT "a smoggy dawn skyline".',
                    'Return JSON only, no markdown, in this exact shape:',
                    '{"beats":[{"startSec":0,"endSec":12,"description":"what this beat is about, in English, 1-2 sentences","moments":[{"atSec":2,"what":"one short clause: what is seen/heard at this moment"},{"atSec":8,"what":"..."}]}]}',
                    'BEATS, not chapters: split the video into short beats of 8-15 seconds each (about one beat per 10 seconds). Each beat is ONE micro-event — one thing happening, one place, one action — never a summary of several seconds, never several actions in one beat. One beat = one narration sentence.',
                    'Inside EVERY beat, list 1-2 key moments in time order with their approximate timestamp (atSec) and one short clause each (what is actually seen or heard at that exact moment). Moments are the ground truth the narration will be timed against — be precise, never invent a moment.',
                    'LITERAL ONLY: describe ONLY what is literally visible/audible at that moment — a still object, a cutaway or a memory/flashback is just that. Never invent motion, weather, time of day, or a power-up/awakening that is not actually on screen. If a beat cross-cuts between places, say so and list what each cut actually shows.',
                    'Skip pure black screens, fade-ins, fade-outs and end credits entirely — no beat may be mostly black, and the first beat starts when the picture actually begins.' + (firstNonBlackSec !== null ? ` The video fades in from black; the first real picture is at about ${firstNonBlackSec}s — the first beat must start at or after ${firstNonBlackSec}s.` : ''),
                    'Cover the whole video from the first real picture to the end, in chronological order, beats back-to-back with no gaps.',
                    `Video duration is ${Math.round(duration)} seconds.`,
                  ].join('\n'),
                },
              ],
            },
          ],
        }),
      'video analysis (Files API)'
    );
    const text = response.text;
    if (!text) throw new Error('Gemini Files API analysis returned no text.');
    const parsed = parseJsonResponse(text, 'video analysis (Files API)');
    const scenes = Array.isArray(parsed.beats) ? parsed.beats : parsed.scenes;
    if (!Array.isArray(scenes) || scenes.length === 0) throw new Error('Gemini Files API analysis returned no scenes.');
    return scenes.map((s) => ({
      startSec: Number(s.startSec) || 0,
      endSec: Number(s.endSec) || 0,
      description: String(s.description || '').trim(),
      moments: (Array.isArray(s.moments) ? s.moments : [])
        .map((m) => ({ atSec: Number(m.atSec) || 0, what: String(m.what || '').trim() }))
        .filter((m) => m.what),
    }));
  } finally {
    try {
      await ai.files.delete({ name: file.name });
    } catch (_) {}
  }
}

/** Analyse dispatch: Files API when ANALYSIS_MODE=files (with frame fallback), else frames. */
async function analyzeVideoAuto(videoPath, framesDir) {
  const firstNonBlackSec = await detectFirstNonBlackSec(videoPath);
  if (firstNonBlackSec !== null) console.log(`[analysis] first real picture at ~${firstNonBlackSec}s (opening fade from black)`);
  if (ANALYSIS_MODE === 'files') {
    try {
      return await analyzeVideoWithFiles(videoPath, firstNonBlackSec);
    } catch (err) {
      console.log(`[analysis] Files API failed (${String(err.message).slice(0, 160)}) — falling back to frame analysis`);
      return analyzeVideo(videoPath, framesDir, firstNonBlackSec);
    }
  }
  return analyzeVideo(videoPath, framesDir, firstNonBlackSec);
}

/**
 * Extract the single frame at `atSec` from a video, downscaled to 320px
 * wide (small enough to send 40 of them in one vision call, big enough to
 * recognise what is on screen). Returns the JPEG path. Input `-ss` seeks
 * to the nearest keyframe then decodes forward, so the returned frame is
 * the frame AT atSec, not just the nearest keyframe.
 */
async function extractMomentFrame(videoPath, atSec, outPath) {
  fs.mkdirSync(path.dirname(outPath), { recursive: true });
  await execFileAsync('ffmpeg', [
    '-y', '-ss', String(atSec), '-i', videoPath,
    '-vframes', '1',
    '-vf', 'scale=320:-1,format=yuvj420p',
    '-q:v', '3',
    outPath,
  ]);
  if (!fs.existsSync(outPath)) {
    throw new Error(`extractMomentFrame: no frame at ${atSec}s in ${videoPath}`);
  }
  return outPath;
}

/**
 * Ask Gemini (vision, ANALYSIS_MODEL) to verify one group of moments.
 * `group` is [{ id, atSec, what, framePath }] — one inline image per
 * moment, each followed by its claim, so the model can attribute every
 * verdict to the right frame. Returns the verdicts array (possibly empty
 * on a bad response — the caller treats missing verdicts as unverified).
 * One call per group, never one call per moment.
 */
async function verifyMomentGroupWithModel(ai, group) {
  const parts = [];
  for (const m of group) {
    parts.push({
      inlineData: {
        mimeType: 'image/jpeg',
        data: fs.readFileSync(m.framePath).toString('base64'),
      },
    });
    parts.push({ text: `Moment ${m.id} @${m.atSec}s: "${m.what}"` });
  }
  parts.push({
    text: [
      'These are real frames from one video, each followed by a claim about what that frame shows.',
      'For EACH moment, look at its frame and judge ONLY that frame against its claim.',
      'Return JSON only, no markdown, in this exact shape:',
      '{"verdicts":[{"id":0,"confirmed":true},{"id":1,"confirmed":false,"actual":"a sniper-scope view of a flying war robot over rooftops","correctedAtSec":null}]}',
      'Rules:',
      '- "confirmed": true ONLY if the frame CLEARLY shows the claimed event or action. Be strict: a similar place, a similar person, or the same scene a few seconds off does NOT count.',
      '- If not confirmed, "actual" is one short clause describing what the frame ACTUALLY shows (in English, what is literally visible).',
      '- If the claimed event IS visible at a different timestamp nearby (within a few seconds), set "correctedAtSec" to that timestamp (a number, in seconds). Otherwise set it to null. Never guess a timestamp you cannot see.',
      '- If you are unsure, say confirmed:false with actual describing the frame — never confirm on a guess.',
      '- Return one verdict per moment id, in the same order as the moments above. Do not add moments, do not drop any.',
    ].join('\n'),
  });
  const response = await withRetry(
    () =>
      ai.models.generateContent({
        model: ANALYSIS_MODEL,
        contents: [{ parts }],
        config: { responseMimeType: 'application/json' },
      }),
    'moment verification'
  );
  const text = response.text;
  if (!text) return [];
  const parsed = parseJsonResponse(text, 'moment verification');
  if (Array.isArray(parsed.verdicts)) return parsed.verdicts;
  if (Array.isArray(parsed.results)) return parsed.results;
  return [];
}

/**
 * Apply verification verdicts to a set of beats (pure — no I/O, testable).
 * `moments` is the flat list [{ id, beatIndex, momentIndex, atSec, what }]
 * built by verifyMoments; `verdicts` is the model's verdict array.
 * Returns { beats, stats, events } where beats is a NEW array (the
 * original is untouched):
 *  - confirmed: the moment is kept as-is.
 *  - rejected with a correctedAtSec inside its own beat (±5s, clamped
 *    into the beat): the moment is MOVED to the corrected timestamp and
 *    keeps its `what` (the event description was right, the timestamp was
 *    wrong).
 *  - rejected without a usable correction: the moment is DROPPED.
 *  - no verdict for a moment: the moment is KEPT (unverified — a bad or
 *    incomplete verification must never gut the script).
 * A correction outside its own beat is treated as no correction (dropped):
 * moving a moment across beats would shuffle the story, and the event is
 * usually already covered by the neighbouring beat's moments.
 * `events` lists the dropped/corrected moments (for the console log).
 */
function applyVerdicts(beats, moments, verdicts) {
  const byId = new Map();
  for (const v of Array.isArray(verdicts) ? verdicts : []) {
    const id = Number(v && v.id);
    if (Number.isFinite(id) && !byId.has(id)) byId.set(id, v);
  }
  const stats = { total: moments.length, verified: 0, corrected: 0, dropped: 0, unverified: 0 };
  const events = [];
  const correctedAt = new Map(); // moment id -> new atSec
  const droppedIds = new Set();
  for (const m of moments) {
    const v = byId.get(m.id);
    if (!v) {
      stats.unverified++;
      continue;
    }
    if (v.confirmed === true) {
      stats.verified++;
      continue;
    }
    const beat = beats[m.beatIndex];
    const corr = Number(v.correctedAtSec);
    const inBeat =
      beat &&
      Number.isFinite(corr) &&
      corr >= 0 &&
      corr >= beat.startSec - 5 &&
      corr <= beat.endSec + 5 &&
      corr !== m.atSec;
    if (inBeat) {
      const clamped = Math.max(beat.startSec, Math.min(beat.endSec, corr));
      correctedAt.set(m.id, clamped);
      stats.corrected++;
      events.push({ id: m.id, atSec: m.atSec, what: m.what, outcome: 'corrected', actual: String(v.actual || ''), correctedAtSec: clamped });
    } else {
      droppedIds.add(m.id);
      stats.dropped++;
      events.push({ id: m.id, atSec: m.atSec, what: m.what, outcome: 'dropped', actual: String(v.actual || ''), correctedAtSec: null });
    }
  }
  // Rebuild each beat's moments from the flat list, so a dropped moment
  // disappears and a corrected moment moves — the beat's other fields
  // (description, range) are untouched. Kept moments are re-sorted into
  // time order: a corrected moment can otherwise sit after a later one.
  const byBeatMoment = new Map();
  for (const m of moments) byBeatMoment.set(`${m.beatIndex}:${m.momentIndex}`, m);
  const newBeats = beats.map((b, bi) => {
    const kept = [];
    (b.moments || []).forEach((m, mi) => {
      const flat = byBeatMoment.get(`${bi}:${mi}`);
      if (!flat) {
        kept.push(m);
        return;
      }
      if (droppedIds.has(flat.id)) return;
      if (correctedAt.has(flat.id)) {
        kept.push({ atSec: correctedAt.get(flat.id), what: String(m.what || '').trim() });
      } else {
        kept.push(m);
      }
    });
    kept.sort((a, b) => a.atSec - b.atSec);
    return { ...b, moments: kept };
  });
  return { beats: newBeats, stats, events };
}

/**
 * Verify every analysis moment against its ACTUAL video frame, and return
 * beats whose moments are only the verified ones.
 *
 * Flow: collect all moments (global ids, beat order) -> group them by
 * script chunk (40 beats) and split into <= MOMENTS_PER_VERIFY_CALL per
 * model call -> for each group, extract the real frame at each moment's
 * timestamp and ask Gemini (vision) to confirm the claim -> apply the
 * verdicts (confirm / correct / drop / keep-if-no-verdict).
 *
 * Returns { beats, stats } where stats is
 * { total, verified, corrected, dropped, unverified, calls, enabled,
 *   status } and status is 'verified' | 'skipped' | 'failed'.
 * On ANY failure (frame extraction, model call, bad JSON, unexpected
 * error) the ORIGINAL beats are returned and the job continues with
 * unverified moments — verification never fails the job and never drops
 * a moment it could not check.
 *
 * opts (all optional, for tests):
 *  - enabled: default VERIFY_MOMENTS (env).
 *  - maxMomentsPerCall: default MOMENTS_PER_VERIFY_CALL.
 *  - chunkBeats: default SCRIPT_CHUNK_BEATS (grouping aligns with the
 *    script chunks so one chunk's moments are verified together).
 *  - frameProvider(videoPath, atSec, outPath): default
 *    extractMomentFrame. Tests inject a stub that writes a fake frame.
 *  - modelFn(group): default the real Gemini call. Tests inject a mock
 *    that returns verdicts without calling Gemini. The group passed to
 *    modelFn is [{ id, atSec, what, framePath, beatIndex }].
 */
async function verifyMoments(beats, videoPath, workDir, opts = {}) {
  const enabled = opts.enabled !== undefined ? opts.enabled : VERIFY_MOMENTS;
  const maxPerCall = opts.maxMomentsPerCall || MOMENTS_PER_VERIFY_CALL;
  const chunkBeats = opts.chunkBeats || SCRIPT_CHUNK_BEATS;
  const frameProvider = opts.frameProvider || extractMomentFrame;
  const modelFn = opts.modelFn || null;
  const baseStats = { total: 0, verified: 0, corrected: 0, dropped: 0, unverified: 0, calls: 0, enabled, status: 'skipped' };
  if (!enabled) return { beats, stats: baseStats };
  if (!Array.isArray(beats) || beats.length === 0) return { beats, stats: baseStats };

  // Flat moment list, in beat order, with global ids.
  const moments = [];
  beats.forEach((b, bi) => {
    (b.moments || []).forEach((m, mi) => {
      const what = String(m.what || '').trim();
      if (!what) return;
      moments.push({ id: moments.length, beatIndex: bi, momentIndex: mi, atSec: Number(m.atSec) || 0, what });
    });
  });
  if (moments.length === 0) {
    return { beats, stats: { ...baseStats, status: 'verified' } };
  }
  baseStats.total = moments.length;

  // Group moments: never split one beat across calls, never cross a
  // script-chunk boundary (40 beats — the same split as
  // splitBeatsIntoChunks), at most maxPerCall per call.
  const groups = [];
  let cur = [];
  let curChunk = 0;
  const byBeat = new Map();
  for (const m of moments) {
    if (!byBeat.has(m.beatIndex)) byBeat.set(m.beatIndex, []);
    byBeat.get(m.beatIndex).push(m);
  }
  beats.forEach((b, bi) => {
    const bMoments = byBeat.get(bi) || [];
    if (bMoments.length === 0) return;
    const bChunk = Math.floor(bi / chunkBeats);
    if (cur.length > 0 && (bChunk !== curChunk || cur.length + bMoments.length > maxPerCall)) {
      groups.push(cur);
      cur = [];
    }
    curChunk = bChunk;
    cur.push(...bMoments);
  });
  if (cur.length > 0) groups.push(cur);

  const frameDir = workDir;
  let allVerdicts = [];
  let calls = 0;
  let failedGroups = 0;
  try {
    fs.mkdirSync(frameDir, { recursive: true });
    for (let gi = 0; gi < groups.length; gi++) {
      const group = groups[gi];
      // Extract the real frame at each moment's timestamp. A moment whose
      // frame cannot be extracted is left unverified (kept) — never
      // dropped on an ffmpeg error.
      const withFrames = [];
      for (const m of group) {
        const outPath = path.join(frameDir, `m${m.id}.jpg`);
        try {
          await frameProvider(videoPath, m.atSec, outPath);
          withFrames.push({ ...m, framePath: outPath });
        } catch (e) {
          console.log(`[verify] frame extraction failed for moment ${m.id} @${m.atSec}s (${String(e.message).slice(0, 100)}) — keeping it unverified`);
        }
      }
      if (withFrames.length === 0) {
        failedGroups++;
        continue;
      }
      try {
        calls++;
        let verdicts;
        if (modelFn) {
          verdicts = await modelFn(withFrames);
        } else {
          verdicts = await verifyMomentGroupWithModel(getClient(), withFrames);
        }
        if (Array.isArray(verdicts)) allVerdicts.push(...verdicts);
      } catch (e) {
        failedGroups++;
        console.log(`[verify] group ${gi + 1}/${groups.length} failed (${String(e.message).slice(0, 120)}) — keeping its ${withFrames.length} moment(s) unverified`);
      }
    }
    const { beats: newBeats, stats, events } = applyVerdicts(beats, moments, allVerdicts);
    stats.calls = calls;
    stats.enabled = enabled;
    stats.status = failedGroups === groups.length && groups.length > 0 ? 'failed' : 'verified';
    for (const ev of events) {
      console.log(`[verify] moment ${ev.id} @${ev.atSec}s ${ev.outcome}${ev.outcome === 'corrected' ? ` -> ${ev.correctedAtSec}s` : ''}: "${ev.what}" — frame shows: ${ev.actual || '(not reported)'}`);
    }
    console.log(`[verify] ${stats.verified} verified, ${stats.corrected} corrected, ${stats.dropped} dropped, ${stats.unverified} unverified, ${calls} call(s), status ${stats.status}`);
    return { beats: newBeats, stats };
  } catch (err) {
    console.log(`[verify] verification failed (${String(err.message).slice(0, 160)}) — continuing with unverified moments`);
    return { beats, stats: { ...baseStats, unverified: moments.length, status: 'failed' } };
  } finally {
    try {
      fs.rmSync(frameDir, { recursive: true, force: true });
    } catch (_) {}
  }
}

/**
 * The style model: a REAL narration from Recappian Myanmar 2.0 (Real Steel
 * recap, first ~3 min, auto-captions cleaned). The user caught that my
 * hand-written rules produced WRITTEN prose — long dense sentences stuffed
 * with adjectives ("dim flickering light, dark cold lab") — while the real
 * narrator uses SHORT sentences, one or two facts each, plain everyday
 * words, and never describes appearances. This example IS the style: Gemini
 * copies its voice instead of following my rules. (2026-11-16: user said
 * "Gemini ကို script ရေးခိုင်း" — let Gemini write the script.)
 */
const REAL_STYLE_EXAMPLE = [
  'ချာလီက အကြွေးပတ်လည်ဝိုင်းနေတဲ့ စက်ရုပ်ဖိုက်တာ လောင်းကစားသမားတစ်ယောက်ဖြစ်ပါတယ်။',
  'သူများတွေက စက်ရုပ်အချင်းချင်း ချနေချိန်မှာ သူ့စက်ရုပ်က နှစ်သိုးနဲ့ချရင် တစ်စီဖြစ်သွားပါတယ်။',
  'အရင်အကြွေးတွေတောင်မကျေသေးခင် စက်ရုပ်ကပါ သုံးစားမရတော့တာကြောင့် ချာလီက လောင်ကြီးမပေးဘဲ လစ်ထွက်လာပါတယ်။',
  'တစ်ခါလဲ သူခိုးထောင်ပြီး သတင်းဆိုးတစ်ခုက ထပ်ရောက်လာပြန်ပါတယ်။',
  'သူနဲ့ကွဲနေတာကြာပြီဖြစ်တဲ့ မိန်းမက ဆုံးသွားပြီဖြစ်တာကြောင့် အသက်၁၁နှစ်အရွယ် သားကို အဖေဖြစ်တဲ့ ချာလီက အုပ်ထိန်းခွင့်ယူရမှာဖြစ်ပါတယ်။',
  'ကလေးနာမည်က မက်စ်ဖြစ်ပြီး သူ့ရဲ့အဒေါ်က သူ့ကိုမွေးစားချင်နေပါတယ်။',
  'ဒါပေမဲ့ အမေက သေတန်းစာမချန်ခဲ့တဲ့အတွက် တရားဥပဒေအရ အဖေဖြစ်တဲ့ ချာလီပဲ စောင့်ရှောက်ခွင့်ရမှာဖြစ်ပါတယ်။',
  'အဲ့ဒါကိုသိတဲ့ ချာလီက ဒေါ်လာ၁သိန်းပေးရင် အုပ်ထိန်းခွင့်ပေးမယ်လို့ ပြောပါတယ်။',
  'မိန်းမ အလိုက်ချင်နေတဲ့ မှာလဲ သဘောတူလိုက်ပြီး လောလောဆယ်တော့ ခရီးသွားဖို့ရှိတာကြောင့် ဧည့်ရပ်တွေမှာ ကလေးကို ခဏကြည့်ပေးထားဖို့ ပြောပါတယ်။',
  'ချာလီကလဲ လက်ခံလိုက်ပါတယ်။ အိမ်မရှိတဲ့ ချာလီက သွေးရည်းစားဘဲမှာ နေပါတယ်။',
  'မာဘင်က ပိုက်ဆံပေးတာကို မက်သိသွားတော့ ကိုယ့်သားကိုယ်တောင် ရောင်းစားတဲ့အဖေဆိုပြီး မုန်နေပါတယ်။',
  'ဂရုမစိုက်ဘဲ သူလိုချင်တဲ့ စက်ရုပ်ကို ဝယ်လိုက်ပါတယ်။ စက်ရုပ်နာမည်က နွိုင်စီဝိုင်းဖြစ်ပြီး လွန်ခဲ့တဲ့ ငါးနှစ်က ချန်ပီယံ စက်ရုပ်ဖြစ်ပါတယ်။',
  'မက်ကလဲ စက်ရုပ်လက်ဝှေ့ပွဲတွေကို ကြိုက်တာကြောင့် နွိုင်စီဝိုင်းရဲ့ နိုင်ပွဲတွေအားလုံးကို သိနေပါတယ်။',
  'ချာလီနဲ့ မက်တို့က စက်ရုပ်လက်ဝှေ့ပွဲကို သွားဖို့ ပြင်ဆင်ပါတယ်။ မက်ကလဲ မခေါ်တဲ့ကြားက အတင်းလိုက်သွားပါတယ်။',
  'ပွဲကို ရောက်တော့ ချာလီက ပွဲသေးသေးတွေ မထိုးပဲ နာမည်ကြီး စက်ရုပ်တွေနဲ့ပဲ ထိုးချင်နေပါတယ်။',
  'မက်က ပွဲသေးသေးတွေကနေ တဆင့်ခြင်း တက်သွားဖို့ ပြောပေမယ့် လုပ်ချင်ရာစုလုပ်တဲ့ ချာလီကတော့ အကြမ်းဆုံးကောင်နဲ့ ပွဲတည်လိုက်ပါတယ်။',
  'အဲ့လိုနဲ့ ပွဲစတော့ နွိုင်စီဝိုင်းက အသာစီးရသလိုရှိပေမယ့် ချာလီက အသံစနစ်တွေကို ကြိုကြည့်မလာတာကြောင့် အခွင့်အရေးအရ လဲ မထိုးမိဘဲ တုံ့ဆိုင်းတုံ့ဆိုင်းဖြစ်နေပါတယ်။',
  'အဲ့ဒါကြောင့်ပဲ တစ်ဖက်စက်ရုပ်က နွိုင်စီဝိုင်းကို ပယ်ပယ်နယ်နဲ့ ပြန်ဆော်ပါတယ်။',
  'တရားမဝင် လက်ဝှေ့ပွဲဖြစ်လို့ စည်းမျဉ်းတွေလဲ မရှိတာကြောင့် လက်တစ်ဖက်ပါ ပြတ်သွားပါတယ်။',
  'ချာလီကတော့ အခုချိန်ထိတောင် ဘာလုပ်ခံရမှန်းမသိဘဲ တိုင်ပတ်နေပါတယ်။',
  'နောက်ဆုံးမှာ တော့ နွိုင်စီဝိုင်း ခေါင်းပျက်ပြီး ချာလီတို့ ရှုံးသွားပါတယ်။',
  'အမှိုက်သက်သာဖြစ်သွားတဲ့ စက်ရုပ်မှာ အသုံးဝင်သေးတာဆိုလို့ အသံနဲ့ထိန်းချုပ်လို့ရတဲ့ စနစ်တစ်ခုပဲ ကျန်ပါတော့တယ်။',
  'နောက်ထပ် စက်ရုပ်မဝယ်နိုင်တော့တဲ့ ချာလီက သူ့စက်ရုပ်ကို ပြင်ဖို့ စက်ရုပ်သင်္ချိုင်းမှာ အသုံးဝင်နိုင်မယ့် စက်ရုပ်အစိတ်အပိုင်းတွေ သွားရှာပါတယ်။',
  'ကိုယ့်ကြောင့် ခိုးဝက်ဟိုးဟို တီးတိလိုက်ရှာရင်း မက်က ကမ်းမပေါ်ကနေ ပြုတ်ကျသွားပါတယ်။',
  'ကံကောင်းပါပဲ စက်ရုပ်လက်တစ်ဖက်နဲ့ ညီပြီး တန်းလှမ်းဖြစ်နေချိန် ချာလီက ဆွဲတင်လိုက်ပါတယ်။',
  'မက်က သူ့ကိုဆွဲထားတဲ့ စက်ရုပ်လက်ကနေ တူးကြည့်တော့ စက်ရုပ်တစ်ကိုယ်လုံး ရှိနေတာကို တွေ့လိုက်ရပါတယ်။',
].join('\n');

/**
 * Beat-lock helpers (2026-11-17, YouTube recap study: one sentence = one
 * story beat, ~5-10s; the picture always shows that same beat, 22/22
 * samples). The script model ignores prompt-only structural instructions
 * (job b27e7aa3: whole-beat ranges, ~12s narrations for 45-90s ranges,
 * zero segments), so alignment is enforced in CODE from the analysis
 * moments — the timestamped ground truth — not by asking the model.
 */

/** Split Burmese narration into sentences (terminator ။, also ! and ?). */
function splitSentences(text) {
  const parts = String(text || '').match(/[^။!?]+(?:[။!?]|$)/g) || [];
  return parts.map((p) => p.trim()).filter(Boolean);
}

/**
 * Normalise a list of per-sentence segments: clamp each window inside the
 * scene's clip [clipStart, clipEnd], sort in time order, de-overlap, and
 * size each window to its sentence (~15 chars/sec, min 4s) so the window
 * is never shorter than the sentence it carries. A window shorter than its
 * sentence is what makes the renderer show the NEXT moment early.
 * Returns the normalised list (possibly empty) — the caller decides the
 * minimum length that counts as moment-cutting.
 */
function normaliseSegments(raw, clipStart, clipEnd) {
  const list = (Array.isArray(raw) ? raw : [])
    .map((g) => ({
      text: String((g && g.text) || '').trim(),
      sourceStartSec: Number((g && g.sourceStartSec) || 0),
      sourceEndSec: Number((g && g.sourceEndSec) || 0),
    }))
    .filter((g) => g.text && g.sourceEndSec > g.sourceStartSec)
    .sort((a, b) => a.sourceStartSec - b.sourceStartSec);
  const norm = [];
  for (const g of list) {
    let a = Math.max(g.sourceStartSec, clipStart);
    let b = Math.min(g.sourceEndSec, clipEnd);
    if (b <= a) {
      a = clipStart;
      b = clipEnd;
    }
    if (b <= a) continue;
    const estSec = Math.max(4, g.text.length / 15);
    if (b - a < estSec) b = Math.min(clipEnd, a + estSec);
    if (b <= a) continue;
    if (norm.length > 0 && a < norm[norm.length - 1].sourceEndSec) {
      a = norm[norm.length - 1].sourceEndSec;
      if (b <= a) continue;
    }
    norm.push({ text: g.text, sourceStartSec: a, sourceEndSec: b });
  }
  return norm;
}

/**
 * Build a scene's per-sentence segments from its analysis beat's moments.
 * Sentence i is assigned to moment i, in time order; window =
 * [moment.atSec, next moment's atSec or clip end], extended to the
 * sentence's spoken length. Moments outside the scene's clip are ignored
 * (the narration describes only what is inside the clip). Sentences with
 * no moment are DROPPED by the caller (see generateScriptChunk) — the
 * returned segments' texts ARE the kept narration. Returns null when
 * moment-cutting is impossible for this scene (fewer than 2 sentences or
 * fewer than 2 moments inside the clip) — the scene then renders
 * scene-level. One segment = the scene itself, so it is not kept (an
 * extra split boundary would only add failure modes for zero benefit).
 */
function buildSegmentsFromMoments(sentences, beat, clipStart, clipEnd) {
  if (!MOMENT_CUT || !Array.isArray(sentences) || sentences.length < 2) return null;
  if (!beat || !Array.isArray(beat.moments)) return null;
  const moments = beat.moments
    .map((m) => ({ atSec: Number(m.atSec) || 0, what: String(m.what || '').trim() }))
    .filter((m) => m.what && m.atSec >= clipStart && m.atSec < clipEnd)
    .sort((a, b) => a.atSec - b.atSec);
  if (moments.length < 2) return null;
  const kept = sentences.slice(0, moments.length);
  if (kept.length < 2) return null;
  const raw = kept.map((text, i) => ({
    text,
    sourceStartSec: moments[i].atSec,
    sourceEndSec: i + 1 < moments.length ? moments[i + 1].atSec : clipEnd,
  }));
  const norm = normaliseSegments(raw, clipStart, clipEnd);
  return norm.length > 1 ? norm : null;
}

/**
 * The model's own segments, used only as a hint when the beat has no
 * usable moments to build from. Same normalisation as the code-built path.
 * Returns null unless at least 2 valid segments survive.
 */
function normaliseModelSegments(modelSegments, clipStart, clipEnd) {
  if (!MOMENT_CUT || !Array.isArray(modelSegments) || modelSegments.length === 0) return null;
  const norm = normaliseSegments(modelSegments, clipStart, clipEnd);
  return norm.length > 1 ? norm : null;
}

/**
 * Drift flag for the job record: a scene whose narration is under half its
 * clip range is at risk of showing the wrong picture (the renderer trims
 * the clip to the narration, so late moments never appear). Moment-cut
 * scenes are aligned per segment, so the scene-level comparison does not
 * apply to them. Spoken Burmese runs ~20 chars/sec.
 */
function computeDriftRisk(narrationMy, startSec, endSec, hasSegments) {
  if (hasSegments) return false;
  const rangeLen = Math.max(1, endSec - startSec);
  return (narrationMy || '').length / 20 < 0.5 * rangeLen;
}

/** True when a scene carries usable per-sentence segments. */
function hasMomentSegments(scene) {
  return MOMENT_CUT && Array.isArray(scene.segments) && scene.segments.length > 1;
}

/**
 * TTS text units for one scene: its sentences when it is a moment scene,
 * its whole narration otherwise. generateRecapAudio joins every scene's
 * units with <long pause> in ONE call — moment mode is per-scene, so a
 * scene without segments no longer disables moment-cutting for the chunk.
 */
function chunkTtsUnits(scene) {
  if (hasMomentSegments(scene)) {
    return scene.segments.map((g) => String(g.text || '').trim()).filter(Boolean);
  }
  const text = String(scene.narrationMy || '').trim();
  return text ? [text] : [];
}

/**
 * Generate a Burmese scene-by-scene recap script from REAL scene summaries.
 * Returns { title, scenes: [{ sceneNumber, startSec, endSec, sourceStartSec, sourceEndSec, narrationMy }] }
 * sourceStartSec/sourceEndSec are the timestamps in the ORIGINAL video to cut
 * the clip from — this is what makes the later render match the narration.
 */
// Script chunking (2026-11-17, beat-lock fix): a 30-min recap of an 82-min
// film means ~200-300 short beats — too many for ONE script call (the
// model drops beats and the JSON gets huge). Beats are split into groups
// of ~40 and the script is generated per group, then concatenated in
// order. A 5-min recap (~40-60 beats) is one or two groups; the path is
// unchanged apart from the split. Beats are NEVER made long again to
// solve scaling — that reintroduces the drift the short beats fixed.
const SCRIPT_CHUNK_BEATS = 40;

/** Split beats into consecutive groups of at most `chunkSize` (pure, testable). */
function splitBeatsIntoChunks(sceneSummaries, chunkSize = SCRIPT_CHUNK_BEATS) {
  const chunks = [];
  for (let i = 0; i < sceneSummaries.length; i += chunkSize) {
    chunks.push(sceneSummaries.slice(i, i + chunkSize));
  }
  return chunks;
}

/**
 * Generate the script for ONE chunk of beats (<= SCRIPT_CHUNK_BEATS).
 * targetSecPerScene is computed from the TOTAL beat count by the wrapper,
 * so the concatenated recap still lasts about lengthMinutes.
 */
async function generateScriptChunk({ sceneSummaries, lengthMinutes, category, fileName, targetSecPerScene, chunkIndex, totalChunks }) {
  const ai = getClient();
  // ONE SCENE PER BEAT — every analysed beat becomes a scene. The model
  // kept "picking" beats and dropping turning points (the orange laser grid
  // 190-235s was dropped twice, in the lite run AND the 3.8-flash run), so
  // selection is no longer the model's choice. Scene count = beat count.
  const sceneCount = sceneSummaries.length;
  const summaryText = sceneSummaries
    .map((s) => {
      const moments = (s.moments || []).map((m) => `  ${m.atSec}s: ${m.what}`).join('\n');
      return `${s.startSec}s-${s.endSec}s: ${s.description}${moments ? '\n' + moments : ''}`;
    })
    .join('\n');

  const prompt = [
    'You are the narrator of Recappian Myanmar 2.0 — the most popular Burmese recap channel (883K subs).',
    'Write the script the way the real narrator below writes it. Study this REAL narration first — its voice IS the style. Do not follow any other style description; copy this example:',
    REAL_STYLE_EXAMPLE,
    'How it sounds — do exactly this: SHORT sentences, one or two facts per sentence, then the next sentence. Plain everyday spoken words (သုံးစားမရတော့တာ, လစ်ထွက်လာပါတယ်, တိုင်ပတ်နေပါတယ်, အမှိုက်သက်သာ, ဂရုမစိုက်ဘဲ, အတင်းလိုက်သွားပါတယ်). NEVER pile up adjectives, NEVER describe how things look — no "dim flickering light", no "dark cold lab", no "blazing blue energy", no "meticulously examining". Just say WHAT HAPPENS, then what happens next. Chain sentences with ဒါပေမဲ့ / အဲ့ဒါကြောင့် / အဲ့လိုနဲ့ / နောက်ဆုံး / နောက်ထပ် / ကံကောင်းပါပဲ / တစ်ခါလဲ. You are telling the story to a friend — NOT explaining it, NOT teaching it, NOT describing a picture.',
    'Write a Burmese-language recap script, divided into exactly ' + sceneCount + ' scenes,',
    `for a ${lengthMinutes}-minute recap video. Genre: ${category}.`,
    'Base the story ONLY on these real beats from the video — do not invent characters, twists or an ending that is not in the beats. Each beat is one short moment (8-15s) with exact timestamps:',
    summaryText,
    ...(totalChunks > 1
      ? ['This is part ' + (chunkIndex + 1) + ' of ' + totalChunks + ' of one ' + lengthMinutes + '-minute recap — the beats below are one consecutive stretch of the full story. Write ONLY these beats, in story order; do not add an opening or a closing for the whole recap (the first part opens it, the last part closes it).']
      : []),
    'Requirements (return JSON only, no markdown):',
    '{"title":"catchy Burmese title","scenes":[{"sceneNumber":1,"sourceStartSec":0,"sourceEndSec":45,"narrationMy":"..."}]}',
    (MOMENT_CUT
      ? '- MOMENT CUTTING — each scene ALSO returns "segments": one segment per sentence of its narration, in time order: {"segments":[{"text":"...","sourceStartSec":80,"sourceEndSec":95}]}. Each segment is ONE short sentence (1-2 facts) and its window is the EXACT moment that sentence describes: it MUST start at that moment\'s timestamp from the beats (e.g. the bridge walk starts when they step onto the bridge, not at the beat start), and end where the next sentence\'s moment starts (or at the beat end for the last one). Window length must roughly match the sentence when spoken (~15 Burmese characters = 1 second, minimum 4s). NEVER give two sentences the same window, NEVER let a segment\'s window show a different moment than its sentence describes. If the narration says the boy and the girl walk on the bridge, that segment\'s window MUST show them walking on the bridge. "narrationMy" must be exactly the segments\' texts joined in order.'
      : ''),
    '- "scenes": exactly ' + sceneCount + ' items, in story order.',
    '- ONE SCENE PER BEAT: use EVERY beat below — one scene per beat, in story order. NEVER skip a beat, NEVER drop a beat because it looks dull, NEVER merge two beats into one scene, never split a beat, never widen a window. Every beat is a scene; the scene count is ' + sceneCount + ' because there are ' + sceneCount + ' beats.',
    '- CLIP RANGE, NOT THE WHOLE BEAT: "sourceStartSec"/"sourceEndSec" must be the exact timestamps of the moments this scene\'s narration actually describes — a contiguous range INSIDE its beat. If the beat is 50-95s but the narration is only about the brain examination at 80-95s, the scene\'s window is 80-95s, NOT 50-95s. The viewer only ever sees this range while the narration plays, so the words and the picture show the SAME thing at the SAME time.',
    '- ONE SENTENCE = ONE BEAT. Never narrate a story point whose footage is not inside this scene\'s window — SKIP it instead of describing it over other footage. If the narrated action has no footage (a decision, a phone call), narrate it over the footage of the event it causes, and only if that footage is inside the window.',
    '- "title": a one-sentence spoiler premise in the channels\' style — "when [shocking thing] happened, then [twist]" — ending in …တဲ့အခါ / …တဲ့အချိန်မှာ / …တော့.',
    '- "narrationMy": pure SPOKEN Burmese. Rules, in order of importance:',
    '  1. ZERO literary forms — never write သည်, များ, ၏, ၌, သောအခါက. Use တယ်, တွေ, ရဲ့ (or drop it), မှာ, တုံးက. AND ZERO ဗျာ / ဗျ / ခင်ဗျာ — the narrator never speaks politely to an elder. ဗျာ makes the voice sound like an old man (ဘိုးတော်) and is FORBIDDEN. Narrate in the ပါတယ် register (e.g. ဖြစ်ပါတယ်, ပြောပါတယ်, လုပ်လိုက်ပါတယ်). Every scene\'s narration must END with ပါတယ် — never end with ပါတော့တယ် or တယ်.',
    '  2. SHORT sentences — one or two facts per sentence, like the real example above. NEVER write one long dense sentence packed with 4-7 events and adjectives. If a scene needs 10 seconds of narration, write MORE short sentences — do NOT make the sentences longer.',
    '  3. Open scene 1 cold, straight into the story — no greeting, no channel intro, no hook question. Start with the premise and run.',
    '  4. NEVER address the viewer directly — no နော် / ပါဦး / လေး to the viewer, no questions to the viewer, no "keep watching" lines. Just tell the story.',
    '  5. Say WHAT HAPPENS, never describe how it looks. Name the action and the reaction, in plain words — one vivid verb per beat. Never write a descriptive passage about light, colour, atmosphere or a place. Never explain why something matters; just say what happened next.',
    '  6. No forced jokes. If a moment is funny, say it straight — do not add a joke where the story has none.',
    '  7. Transliterate ALL foreign character/movie names into Burmese script (Charlie → ချာလီ, Max → မက်, Noisy Boy → နွိုင်စီဝိုင်း). NEVER leave English names in the narration.',
    '  8. Paraphrase dialogue as reported speech with …ဆိုပြီး / …လို့ (e.g. "ကိုယ့်သားကိုယ်တောင်ရောင်းစားတဲ့အဖေဆိုပြီး မုန်နေပါတယ်"). Quote at most one short punchline in the whole recap.',
    '  9. Each scene\'s narration must describe ONLY what is actually visible and happening inside its own clip range (sourceStartSec–sourceEndSec) — and IN TIME ORDER: walk through the range\'s moments from first to last, one clause per moment, exactly as they appear on screen. Never jump ahead to a later moment, never invent, never repeat another scene\'s event. If the range shows a bridge argument, say ONLY the bridge argument; if it shows a lab, say ONLY the lab.',
    '  10. LENGTH: each scene\'s narration must last about as long as its clip range — target about ' + targetSecPerScene + ' seconds per scene, so the whole recap lasts about ' + lengthMinutes + ' minutes. Roughly 15 Burmese characters = 1 spoken second. Fill the range with MORE short sentences, in time order, adding more moments, reactions and comments until the scene is full. NEVER make sentences longer to fill time. A narration much shorter than its range means the moments at the END of the range are never seen or heard — too short is a failure.',
    '  11. End the LAST scene with the story\'s closing beat. No subscribe call, no closing line to the viewer — the story ends when the story ends.',
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
  // Clamp every scene's clip range inside the single beat it overlaps most.
  // The model sometimes returns a window spanning several beats (e.g.
  // 105-255s) — then the render only shows the first ~47s while the
  // narration describes the whole 150s, so words and picture diverge.
  // Clamping guarantees the clip stays inside one beat; the prompt asks the
  // model to narrow the range to the moments it actually narrates.
  const snapped = scenes.map((s, i) => {
    const wStart = Number(s.sourceStartSec) || 0;
    const wEnd = Number(s.sourceEndSec) || 0;
    let best = null;
    let bestOverlap = -1;
    for (const b of sceneSummaries) {
      const overlap = Math.max(0, Math.min(wEnd, b.endSec) - Math.max(wStart, b.startSec));
      if (overlap > bestOverlap) {
        bestOverlap = overlap;
        best = b;
      }
    }
    let startSec = wStart;
    let endSec = wEnd;
    if (best) {
      startSec = Math.max(wStart, best.startSec);
      endSec = Math.min(wEnd, best.endSec);
      if (endSec <= startSec) {
        startSec = best.startSec;
        endSec = best.endSec;
      }
    }
    // BEAT LOCK (2026-11-17, YouTube recap study): segments are built in
    // CODE from the analysis beat's timestamped moments (ground truth) —
    // the model (3.6-flash, job b27e7aa3) returned zero segments, so the
    // render fell back to scene-level and the late moments never appeared.
    // Sentence i is assigned to moment i, in time order (the prompt
    // already requires time order). A sentence with no moment is DROPPED
    // from the narration — narrating less, correctly, beats narrating more
    // over the wrong picture (the real recaps' skip rule). The model's own
    // segments are used only as a hint, when the beat has no usable moments.
    const narrationRaw = String(s.narrationMy || '').trim();
    const sentences = splitSentences(narrationRaw);
    let segments = buildSegmentsFromMoments(sentences, best, startSec, endSec);
    let narrationMy = narrationRaw;
    if (segments) {
      narrationMy = segments.map((g) => g.text).join(' ');
    } else {
      const hint = normaliseModelSegments(s.segments, startSec, endSec);
      if (hint) {
        segments = hint;
        narrationMy = hint.map((g) => g.text).join(' ');
      }
    }
    return {
      sceneNumber: Number(s.sceneNumber) || i + 1,
      startSec,
      endSec,
      sourceStartSec: startSec,
      sourceEndSec: endSec,
      narrationMy,
      ...(segments ? { segments } : {}),
    };
  });
  // Length enforcement: a narration much shorter than its clip range means
  // the range's late moments are never seen or heard (job 6ace4f81 scene 2:
  // 29s of narration over a 45s range — the brain it described sat in the
  // trimmed-off tail and never appeared). Spoken Burmese runs ~20 chars/sec;
  // expand any scene under 80% of its range with ONE extra call.
  const short = snapped.filter((s) => {
    // Moment-cut scenes are exempt: their windows are already sized to
    // their sentences by the normalisation above, so expanding the text
    // would break the sentence↔window alignment the segments carry.
    if (s.segments) return false;
    const rangeLen = Math.max(1, s.endSec - s.startSec);
    return s.narrationMy.length / 20 < 0.8 * rangeLen;
  });
  if (short.length > 0) {
    const expandPrompt = [
      'You are the same Myanmar recap narrator. Expand ONLY these scenes so each narration lasts about its clip range when spoken (about 15 Burmese characters per second). Keep the same clip range and the same style: SHORT sentences, one or two facts per sentence, plain everyday spoken words — never pile up adjectives, never describe how things look, just say what happens next; no ဗျာ/ဗျ ever, no literary သည်/များ/၏/၌, ပါတယ် endings, transliterated names, in time order over the range\'s moments, and describe ONLY what happens inside the range. Add MORE short sentences to fill the time — never make the sentences longer.',
      'Return JSON only, no markdown: {"scenes":[{"sceneNumber":1,"narrationMy":"..."}]}',
      ...short.map(
        (s) =>
          `Scene ${s.sceneNumber}: clip ${s.startSec}s-${s.endSec}s (target ~${Math.round(s.endSec - s.startSec)}s spoken). Current narration (${Math.round(s.narrationMy.length / 20)}s): ${s.narrationMy}`
      ),
    ].join('\n');
    try {
      const expResponse = await withRetry(
        () =>
          ai.models.generateContent({
            model: SCRIPT_MODEL,
            contents: expandPrompt,
            config: { responseMimeType: 'application/json' },
          }),
        'script expansion'
      );
      const expParsed = parseJsonResponse(expResponse.text || '', 'script expansion');
      const expanded = Array.isArray(expParsed.scenes) ? expParsed.scenes : [];
      for (const e of expanded) {
        const target = snapped.find((s) => s.sceneNumber === (Number(e.sceneNumber) || 0));
        if (target && String(e.narrationMy || '').trim()) {
          target.narrationMy = String(e.narrationMy).trim();
        }
      }
      console.log(`[script] expanded ${expanded.length} short scene(s) to fill their clip ranges`);
    } catch (err) {
      console.log(`[script] expansion failed (${String(err.message).slice(0, 120)}) — keeping original narrations`);
    }
  }
  // Drift flag for the job record: a scene whose narration is under half
  // its clip range is at risk of showing the wrong picture (the renderer
  // trims the clip to the narration, so late moments never appear).
  // Moment-cut scenes are aligned per segment, so they are never flagged.
  // Computed AFTER expansion, so an expanded narration is judged on its
  // final length.
  for (const s of snapped) {
    s.driftRisk = computeDriftRisk(s.narrationMy, s.startSec, s.endSec, Array.isArray(s.segments) && s.segments.length > 1);
  }
  return {
    title: String(parsed.title || '').trim() || 'မြန်မာ Recap',
    scenes: snapped,
  };
}

/**
 * Generate the full Burmese script, chunked by beats (see
 * SCRIPT_CHUNK_BEATS). One scene per beat, scenes renumbered globally,
 * title from the first chunk.
 */
async function generateScript({ sceneSummaries, lengthMinutes, category, fileName }) {
  if (![5, 10, 15, 30, 60].includes(lengthMinutes)) {
    throw new Error(`Unsupported recap length ${lengthMinutes}. Use 5, 10, 15, 30 or 60 minutes.`);
  }
  if (!Array.isArray(sceneSummaries) || sceneSummaries.length === 0) {
    throw new Error('generateScript: no scene summaries — cannot write a script from nothing.');
  }
  // Target narration length: the finished recap must actually last about
  // lengthMinutes — each scene's narration should fill its share. Computed
  // from the TOTAL beat count so chunking does not change the per-scene target.
  const targetSecPerScene = Math.round((lengthMinutes * 60) / sceneSummaries.length);
  const chunks = splitBeatsIntoChunks(sceneSummaries);
  let title = 'မြန်မာ Recap';
  const allScenes = [];
  for (let c = 0; c < chunks.length; c++) {
    const part = await generateScriptChunk({
      sceneSummaries: chunks[c],
      lengthMinutes,
      category,
      fileName,
      targetSecPerScene,
      chunkIndex: c,
      totalChunks: chunks.length,
    });
    if (c === 0 && part.title) title = part.title;
    for (const s of part.scenes) {
      allScenes.push({ ...s, sceneNumber: allScenes.length + 1 });
    }
  }
  return { title, scenes: allScenes };
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
            // NOTE: do NOT prepend a "Say it like..." style direction to the
            // text — gemini-3.8-flash-tts reads it aloud instead of treating
            // it as an instruction (job 1dea9983 spoke the English direction
            // for ~25s before the Burmese narration). The narration text
            // itself already carries the spoken style; send it plain.
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
 * Split one MP3 into N segments at the silences between units (scenes, or
 * sentences inside one scene for moment-level cutting).
 * Returns [{ path, durationSec }] or null when the split is not trustworthy
 * (too few silences, or a segment shorter than 0.5s) — the caller then
 * falls back to per-scene TTS instead of shipping a bad split.
 *
 * opts.toleranceSec: how far a detected silence may sit from its expected
 * boundary (default max(3, 20% of the file) — the proven scene-level value;
 * sentence-level splits pass a tighter value, since a scene slice is short
 * and its boundaries are close together). opts.outPrefix names the pieces.
 * Why the first version failed on real audio (job 55046916, 2026-11-15):
 * it required silences of >=1.0s and picked the N-1 LONGEST ones. The
 * <long pause> tag's real duration is model-controlled and unproven — if it
 * renders as ~0.5s, nothing is detected and the split returns null.
 * This version (a) detects shorter pauses (d=0.5), and (b) anchors each
 * boundary to where it SHOULD be: scene audio length is roughly
 * proportional to narration length (~20 Burmese chars per second, scaled
 * to the real total), so for each expected boundary it takes the NEAREST
 * detected silence within tolerance. A long pause inside a scene can no
 * longer be mistaken for a scene boundary, and every decision is logged
 * so the next failure is diagnosable from the Railway logs.
 */
async function splitAudioBySilence(wholeMp3, units, workDir, opts = {}) {
  const unitCount = units.length;
  const unitText = (u) => String(u.text || u.narrationMy || '').trim();
  const totalSec = await ffprobeDuration(wholeMp3);
  if (!totalSec || totalSec <= 0) return null;
  let stderr = '';
  try {
    const res = await execFileAsync('ffmpeg', [
      '-i', wholeMp3,
      '-af', 'silencedetect=noise=-35dB:d=0.5',
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
  console.log(
    `[gemini] silencedetect: ${silences.length} silences >=0.5s in ${totalSec.toFixed(1)}s audio: ` +
      silences.map((s) => `${s.start.toFixed(1)}-${s.end.toFixed(1)}s`).join(', ')
  );
  const needed = unitCount - 1;
  if (silences.length < needed) {
    console.log(`[gemini] split failed: need ${needed} boundaries, found ${silences.length} silences`);
    return null;
  }
  // Expected boundary times from narration lengths, scaled to the real total.
  const est = units.map((u) => Math.max(3, unitText(u).length / 20));
  const estTotal = est.reduce((a, b) => a + b, 0);
  const scale = totalSec / estTotal;
  const expected = [];
  let cum = 0;
  for (let i = 0; i < unitCount - 1; i++) {
    cum += est[i] * scale;
    expected.push(cum);
  }
  const used = new Set();
  const boundaries = [];
  for (const exp of expected) {
    let best = null;
    let bestDist = Infinity;
    for (let i = 0; i < silences.length; i++) {
      if (used.has(i)) continue;
      const mid = (silences[i].start + silences[i].end) / 2;
      const dist = Math.abs(mid - exp);
      if (dist < bestDist) {
        bestDist = dist;
        best = i;
      }
    }
    const tolerance = opts.toleranceSec ?? Math.max(3, 0.2 * totalSec);
    if (best === null || bestDist > tolerance) {
      console.log(
        `[gemini] split failed: no silence near expected boundary ${exp.toFixed(1)}s (nearest ${bestDist.toFixed(1)}s away, tolerance ${tolerance.toFixed(1)}s)`
      );
      return null;
    }
    used.add(best);
    boundaries.push((silences[best].start + silences[best].end) / 2);
  }
  boundaries.sort((a, b) => a - b);
  console.log(
    `[gemini] split boundaries: expected [${expected.map((x) => x.toFixed(1)).join(', ')}] -> chosen [${boundaries.map((x) => x.toFixed(1)).join(', ')}]`
  );
  const cuts = [0, ...boundaries, totalSec];
  const segments = [];
  for (let i = 0; i < unitCount; i++) {
    const out = path.join(workDir, `${opts.outPrefix || 'scene'}_${String(i + 1).padStart(2, '0')}.mp3`);
    // -ss/-to BEFORE -i are input options: absolute timestamps in the file.
    await execFileAsync('ffmpeg', [
      '-y', '-ss', String(cuts[i]), '-to', String(cuts[i + 1]),
      '-i', wholeMp3, '-c', 'copy', out,
    ]);
    const durationSec = await ffprobeDuration(out);
    if (!durationSec || durationSec < 0.5) {
      console.log(`[gemini] split failed: segment ${i + 1} duration ${durationSec} too short`);
      return null;
    }
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
    // Moment-level cutting: the TTS text is every unit (one <long pause>
    // between each), still ONE Gemini call per chunk — quota cost unchanged.
    // The audio is then split in two levels: scenes first (proven), then
    // sentences inside each scene slice (short file, tight tolerance).
    // Moment mode is PER-SCENE, not all-or-nothing for the whole chunk:
    // a scene with valid segments contributes its sentences, a scene
    // without contributes its whole narration — one bad scene used to
    // disable moment-cutting for the entire recap. Either level failing
    // drops that scene back to scene-level audio — a bad moment split
    // never fails the job.
    const joined = chunk
      .flatMap((s) => chunkTtsUnits(s))
      .join('\n<long pause>\n');
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
    const segments = await splitAudioBySilence(wholeMp3, chunk, audioDir, {
      outPrefix: `chunk${c}-scene`,
    });
    if (!segments) {
      console.log(`[gemini] batched split failed for chunk ${c + 1} — falling back to per-scene TTS for its ${chunk.length} scenes`);
      const fallback = await generateAllSceneAudio(chunk, audioDir, voiceName);
      // Fallback audio is scene-level: drop segments so render uses it.
      out.push(...fallback.map((s) => {
        const { segments: _dropped, ...rest } = s;
        return rest;
      }));
    } else {
      // Per-scene moment mode: each scene with valid segments is split
      // into its sentences; a scene without segments keeps scene-level
      // audio. A failed sentence split drops only that scene.
      for (let i = 0; i < chunk.length; i++) {
        const scene = chunk[i];
        const slice = segments[i];
        if (!hasMomentSegments(scene)) {
          out.push({ ...scene, audioPath: slice.path, audioDurationSec: slice.durationSec });
          continue;
        }
        const segSplit = await splitAudioBySilence(slice.path, scene.segments, audioDir, {
          toleranceSec: Math.max(1.5, 0.08 * slice.durationSec),
          outPrefix: `chunk${c}-s${i + 1}-seg`,
        });
        if (!segSplit) {
          console.log(`[gemini] sentence split failed for scene ${scene.sceneNumber} — keeping scene-level audio for that scene`);
          const { segments: _dropped, ...rest } = scene;
          out.push({ ...rest, audioPath: slice.path, audioDurationSec: slice.durationSec });
        } else {
          out.push({
            ...scene,
            audioPath: slice.path,
            audioDurationSec: slice.durationSec,
            segments: scene.segments.map((g, j) => ({
              ...g,
              audioPath: segSplit[j].path,
              audioDurationSec: segSplit[j].durationSec,
            })),
          });
        }
      }
    }
    try {
      fs.unlinkSync(wholeMp3);
    } catch (_) {}
  }
  return out;
}

module.exports = {
  ANALYSIS_MODEL,
  MOMENT_CUT,
  VERIFY_MOMENTS,
  MOMENTS_PER_VERIFY_CALL,
  SCRIPT_MODEL,
  SCRIPT_CHUNK_BEATS,
  TTS_MODEL,
  TTS_MODELS,
  isDailyQuotaError,
  requireApiKey,
  ffprobeDuration,
  extractFrames,
  extractMomentFrame,
  analyzeVideo,
  analyzeVideoWithFiles,
  analyzeVideoAuto,
  verifyMoments,
  applyVerdicts,
  generateScript,
  generateScriptChunk,
  splitBeatsIntoChunks,
  splitSentences,
  normaliseSegments,
  buildSegmentsFromMoments,
  normaliseModelSegments,
  computeDriftRisk,
  hasMomentSegments,
  chunkTtsUnits,
  generateSceneAudio,
  generateRecapAudio,
  splitAudioBySilence,
  SCENE_COUNT_BY_MINUTES,
};
