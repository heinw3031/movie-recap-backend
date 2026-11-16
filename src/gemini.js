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
      '{"beats":[{"startSec":0,"endSec":45,"description":"what this beat is about, in English, 1-2 sentences","moments":[{"atSec":5,"what":"one short clause: what is seen/heard at this moment"},{"atSec":20,"what":"..."},{"atSec":38,"what":"..."}]}]}',
      'BEATS, not chapters: split the video into short beats of 30-60 seconds each (about one beat per 45 seconds). Each beat is ONE continuous moment — one place, one action — never a summary of several minutes.',
      'Inside EVERY beat, list 2-4 key moments in time order with their approximate timestamp (atSec) and one short clause each (what is actually seen or heard at that exact moment). Moments are the ground truth the narration will be timed against — be precise, never invent a moment.',
      'Skip pure black screens, fade-ins, fade-outs and end credits entirely — no beat may be mostly black, and the first beat starts when the picture actually begins.',
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
async function analyzeVideoWithFiles(videoPath) {
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
                    'Return JSON only, no markdown, in this exact shape:',
                    '{"beats":[{"startSec":0,"endSec":45,"description":"what this beat is about, in English, 1-2 sentences","moments":[{"atSec":5,"what":"one short clause: what is seen/heard at this moment"},{"atSec":20,"what":"..."},{"atSec":38,"what":"..."}]}]}',
                    'BEATS, not chapters: split the video into short beats of 30-60 seconds each (about one beat per 45 seconds). Each beat is ONE continuous moment — one place, one action — never a summary of several minutes.',
                    'Inside EVERY beat, list 2-4 key moments in time order with their approximate timestamp (atSec) and one short clause each (what is actually seen or heard at that exact moment). Moments are the ground truth the narration will be timed against — be precise, never invent a moment.',
                    'Skip pure black screens, fade-ins, fade-outs and end credits entirely — no beat may be mostly black, and the first beat starts when the picture actually begins.',
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
  if (ANALYSIS_MODE === 'files') {
    try {
      return await analyzeVideoWithFiles(videoPath);
    } catch (err) {
      console.log(`[analysis] Files API failed (${String(err.message).slice(0, 160)}) — falling back to frame analysis`);
      return analyzeVideo(videoPath, framesDir);
    }
  }
  return analyzeVideo(videoPath, framesDir);
}

/**
 * Generate a Burmese scene-by-scene recap script from REAL scene summaries.
 * Returns { title, scenes: [{ sceneNumber, startSec, endSec, sourceStartSec, sourceEndSec, narrationMy }] }
 * sourceStartSec/sourceEndSec are the timestamps in the ORIGINAL video to cut
 * the clip from — this is what makes the later render match the narration.
 */
async function generateScript({ sceneSummaries, lengthMinutes, category, fileName }) {
  const ai = getClient();
  if (![5, 10, 15, 30, 60].includes(lengthMinutes)) {
    throw new Error(`Unsupported recap length ${lengthMinutes}. Use 5, 10, 15, 30 or 60 minutes.`);
  }
  const sceneCount = SCENE_COUNT_BY_MINUTES[lengthMinutes];
  const summaryText = sceneSummaries
    .map((s) => {
      const moments = (s.moments || []).map((m) => `  ${m.atSec}s: ${m.what}`).join('\n');
      return `${s.startSec}s-${s.endSec}s: ${s.description}${moments ? '\n' + moments : ''}`;
    })
    .join('\n');

  // Target narration length: the finished recap must actually last about
  // lengthMinutes — each scene's narration should fill its share.
  const targetSecPerScene = Math.round((lengthMinutes * 60) / sceneCount);
  const prompt = [
    'You are the narrator of Recappian Myanmar 2.0 — the most popular Burmese recap channel (883K subs). Fast, dense, matter-of-fact storytelling. NOT a funny friend chatting, NOT a newsreader, and NEVER an old man (ဘိုးတော်) talking politely.',
    'Write a gripping Burmese-language recap script, divided into exactly ' + sceneCount + ' scenes,',
    `for a ${lengthMinutes}-minute recap video. Genre: ${category}.`,
    'Base the story ONLY on these real beats from the video — do not invent characters, twists or an ending that is not in the beats. Each beat is one short moment (30-60s) with exact timestamps:',
    summaryText,
    'Requirements (return JSON only, no markdown):',
    '{"title":"catchy Burmese title","scenes":[{"sceneNumber":1,"sourceStartSec":0,"sourceEndSec":45,"narrationMy":"..."}]}',
    '- "scenes": exactly ' + sceneCount + ' items, in story order.',
    '- ONE SCENE PER BEAT: pick ' + sceneCount + ' beats in story order (skip the dull ones, keep the turning points). Each scene uses ONE beat only — never merge two beats into one scene, never split a beat, never widen a window. Cover EVERY turning point — never skip a beat whose moments start a new phase of the story (e.g. a reconstruction starting, a new weapon appearing, the hero entering the simulation).',
    '- CLIP RANGE, NOT THE WHOLE BEAT: "sourceStartSec"/"sourceEndSec" must be the exact timestamps of the moments this scene\'s narration actually describes — a contiguous range INSIDE its beat. If the beat is 50-95s but the narration is only about the brain examination at 80-95s, the scene\'s window is 80-95s, NOT 50-95s. The viewer only ever sees this range while the narration plays, so the words and the picture show the SAME thing at the SAME time.',
    '- "title": a one-sentence spoiler premise in the channels\' style — "when [shocking thing] happened, then [twist]" — ending in …တဲ့အခါ / …တဲ့အချိန်မှာ / …တော့.',
    '- "narrationMy": pure SPOKEN Burmese. Rules, in order of importance:',
    '  1. ZERO literary forms — never write သည်, များ, ၏, ၌, သောအခါက. Use တယ်, တွေ, ရဲ့ (or drop it), မှာ, တုံးက. AND ZERO ဗျာ / ဗျ / ခင်ဗျာ — the narrator never speaks politely to an elder. ဗျာ makes the voice sound like an old man (ဘိုးတော်) and is FORBIDDEN. Narrate in the ပါတယ် register (e.g. ဖြစ်ပါတယ်, ပြောပါတယ်, လုပ်လိုက်ပါတယ်).',
    '  2. LONG, DENSE sentences — pack 4-7 events into one flowing sentence, chained with …ပြီး / …တော့ / …ကြောင့် / ဒါပေမဲ့ / အဲ့ဒါကြောင့် / အဲ့လိုနဲ့ / နောက်ဆုံး. NEVER write short choppy one-event clauses — real recaps tell fast and dense, one sentence covers a whole chain of events.',
    '  3. Open scene 1 cold, straight into the story — no greeting, no channel intro, no hook question. Start with the premise and run.',
    '  4. NEVER address the viewer directly — no နော် / ပါဦး / လေး to the viewer, no questions to the viewer, no "keep watching" lines. Just tell the story.',
    '  5. Dramatise the turning point: name the weakness, the counter-move, the decisive action — one vivid verb per beat. Describe feelings through reactions ("froze", "the crowd erupted", "kept getting back up"), never interior monologue.',
    '  6. No forced jokes. If a moment is funny, say it straight — do not add a joke where the story has none.',
    '  7. Transliterate ALL foreign character/movie names into Burmese script (Charlie → ချာလီ, Max → မက်, Noisy Boy → နွိုင်စီဝိုင်း). NEVER leave English names in the narration.',
    '  8. Paraphrase dialogue as reported speech with …ဆိုပြီး / …လို့ (e.g. "ကိုယ့်သားကိုယ်တောင်ရောင်းစားတဲ့အဖေဆိုပြီး မုန်နေပါတယ်"). Quote at most one short punchline in the whole recap.',
    '  9. Each scene\'s narration must describe ONLY what is actually visible and happening inside its own clip range (sourceStartSec–sourceEndSec) — and IN TIME ORDER: walk through the range\'s moments from first to last, one clause per moment, exactly as they appear on screen. Never jump ahead to a later moment, never invent, never repeat another scene\'s event. If the range shows a bridge argument, say ONLY the bridge argument; if it shows a lab, say ONLY the lab.',
    '  10. LENGTH: each scene\'s narration must last about as long as its clip range (range seconds x 20 Burmese characters of short clauses) — target about ' + targetSecPerScene + ' seconds per scene, so the whole recap lasts about ' + lengthMinutes + ' minutes. A narration much shorter than its range means the moments at the END of the range are never seen or heard — fill the range in order, adding more moments, reactions and comments until the scene is full. Too short is a failure.',
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
    return {
      sceneNumber: Number(s.sceneNumber) || i + 1,
      startSec,
      endSec,
      sourceStartSec: startSec,
      sourceEndSec: endSec,
      narrationMy: String(s.narrationMy || '').trim(),
    };
  });
  // Length enforcement: a narration much shorter than its clip range means
  // the range's late moments are never seen or heard (job 6ace4f81 scene 2:
  // 29s of narration over a 45s range — the brain it described sat in the
  // trimmed-off tail and never appeared). Spoken Burmese runs ~20 chars/sec;
  // expand any scene under 80% of its range with ONE extra call.
  const short = snapped.filter((s) => {
    const rangeLen = Math.max(1, s.endSec - s.startSec);
    return s.narrationMy.length / 20 < 0.8 * rangeLen;
  });
  if (short.length > 0) {
    const expandPrompt = [
      'You are the same Myanmar recap narrator. Expand ONLY these scenes so each narration lasts about its clip range when spoken (about 20 Burmese characters per second). Keep the same clip range and the same style: no ဗျာ/ဗျ ever, no literary သည်/များ/၏/၌, long dense sentences (4-7 events per sentence, chained with ပြီး/တော့/ကြောင့်/ဒါပေမဲ့/အဲ့ဒါကြောင့်), ပါတယ် endings, transliterated names, in time order over the range\'s moments, and describe ONLY what happens inside the range.',
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
  return {
    title: String(parsed.title || '').trim() || 'မြန်မာ Recap',
    scenes: snapped,
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
 * Split one MP3 into N segments at the silences between scenes.
 * Returns [{ path, durationSec }] or null when the split is not trustworthy
 * (too few silences, or a segment shorter than 0.5s) — the caller then
 * falls back to per-scene TTS instead of shipping a bad split.
 *
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
async function splitAudioBySilence(wholeMp3, scenes, workDir) {
  const sceneCount = scenes.length;
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
  const needed = sceneCount - 1;
  if (silences.length < needed) {
    console.log(`[gemini] split failed: need ${needed} boundaries, found ${silences.length} silences`);
    return null;
  }
  // Expected boundary times from narration lengths, scaled to the real total.
  const est = scenes.map((s) => Math.max(3, (s.narrationMy || '').length / 20));
  const estTotal = est.reduce((a, b) => a + b, 0);
  const scale = totalSec / estTotal;
  const expected = [];
  let cum = 0;
  for (let i = 0; i < sceneCount - 1; i++) {
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
    const tolerance = Math.max(3, 0.2 * totalSec);
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
  for (let i = 0; i < sceneCount; i++) {
    const num = scenes[i].sceneNumber || i + 1;
    const out = path.join(workDir, `scene_${num}.mp3`);
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
  isDailyQuotaError,
  requireApiKey,
  ffprobeDuration,
  extractFrames,
  analyzeVideo,
  analyzeVideoWithFiles,
  analyzeVideoAuto,
  generateScript,
  generateSceneAudio,
  generateRecapAudio,
  splitAudioBySilence,
  SCENE_COUNT_BY_MINUTES,
};
