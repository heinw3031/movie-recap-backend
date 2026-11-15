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

// 2026-11-15: gemini-2.5-flash returns 404 "no longer available to new
// users" for this API key, and gemini-2.5-flash-preview-tts is the same
// generation — both replaced with the 3.8 generation (verified live in
// GET /v1beta/models on 2026-11-15: gemini-3.8-flash and
// gemini-3.8-flash-tts both exist). Override via env if Gemini retires them.
const ANALYSIS_MODEL = process.env.ANALYSIS_MODEL || 'gemini-3.8-flash';
const TTS_MODEL = process.env.TTS_MODEL || 'gemini-3.8-flash-tts';
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

  const response = await ai.models.generateContent({
    model: ANALYSIS_MODEL,
    contents: [{ parts }],
  });
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

  const response = await ai.models.generateContent({
    model: ANALYSIS_MODEL,
    contents: prompt,
    config: { responseMimeType: 'application/json' },
  });
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
 * Generate REAL Burmese speech for one scene with Gemini TTS.
 * Returns { path (MP3), durationSec }.
 * Throws a clear error when there is no API key or no audio is returned —
 * it never pretends to have generated audio.
 */
async function generateSceneAudio(textMy, outMp3Path, voiceName = DEFAULT_TTS_VOICE) {
  const ai = getClient();
  if (!textMy || !textMy.trim()) throw new Error('generateSceneAudio: empty narration text.');
  const response = await ai.models.generateContent({
    model: TTS_MODEL,
    contents: [{ parts: [{ text: textMy }] }],
    config: {
      responseModalities: ['AUDIO'],
      speechConfig: {
        voiceConfig: { prebuiltVoiceConfig: { voiceName } },
      },
    },
  });
  const parts = (response.candidates && response.candidates[0] && response.candidates[0].content && response.candidates[0].content.parts) || [];
  const audioPart = parts.find((p) => p.inlineData && p.inlineData.data);
  if (!audioPart) {
    throw new Error(
      'Gemini TTS returned no audio for this scene. Check your GEMINI_API_KEY, ' +
        'the TTS model quota, and that the text is not empty. The job is marked failed — audio is never faked.'
    );
  }
  const pcm = Buffer.from(audioPart.inlineData.data, 'base64');
  if (pcm.length === 0) throw new Error('Gemini TTS returned empty audio data.');
  const wavPath = outMp3Path.replace(/\.mp3$/i, '.wav');
  writePcmAsWav(pcm, wavPath);
  await execFileAsync('ffmpeg', ['-y', '-i', wavPath, '-codec:a', 'libmp3lame', '-q:a', '4', outMp3Path]);
  fs.unlinkSync(wavPath);
  const durationSec = await ffprobeDuration(outMp3Path);
  return { path: outMp3Path, durationSec };
}

module.exports = {
  ANALYSIS_MODEL,
  TTS_MODEL,
  requireApiKey,
  ffprobeDuration,
  extractFrames,
  analyzeVideo,
  generateScript,
  generateSceneAudio,
  SCENE_COUNT_BY_MINUTES,
};
