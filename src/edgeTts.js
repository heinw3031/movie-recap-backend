'use strict';
/**
 * Microsoft Edge TTS (edge-tts) — free, no API key, no account, no card.
 *
 * Why this exists: Gemini's free TTS voices are general multilingual
 * voices and the user found the default ('Kore') too obviously AI-sounding.
 * Edge TTS has NATIVE Burmese neural voices — my-MM-NilarNeural (female)
 * and my-MM-ThihaNeural (male) — trained on Burmese, so pronunciation is
 * better, and Thiha is the masculine voice the user asked for
 * (ယောင်္ကျားဆန်ပြီး သြဇာညောင်းတဲ့ အသံ).
 *
 * Limits: 3,000 chars per request (scenes are ~200 chars, so one call
 * per scene); no published daily quota, but the endpoint is unofficial —
 * Microsoft can throttle or block it, so Gemini stays the default engine
 * and Edge is used when TTS_ENGINE=edge, or as the fallback in auto mode.
 *
 * Requires the `edge-tts` CLI (see Dockerfile: python3 + pip install
 * edge-tts). Output is MP3 directly — no WAV conversion needed.
 * Throws on failure; it never returns fake audio.
 */
const fs = require('fs');
const path = require('path');
const { execFile } = require('child_process');
const { promisify } = require('util');

const execFileAsync = promisify(execFile);
const gemini = require('./gemini');

const EDGE_VOICE = process.env.EDGE_TTS_VOICE || 'my-MM-ThihaNeural';
const EDGE_RATE = process.env.EDGE_TTS_RATE || ''; // e.g. '-5%'
const EDGE_PITCH = process.env.EDGE_TTS_PITCH || ''; // e.g. '-5Hz'
const MAX_CHARS_PER_CALL = 3000;

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function chunkText(text) {
  if (text.length <= MAX_CHARS_PER_CALL) return [text];
  const parts = text.split(/(?<=[။.])\s+/);
  const chunks = [];
  let cur = '';
  for (const p of parts) {
    if (cur && (cur + ' ' + p).length > MAX_CHARS_PER_CALL) {
      chunks.push(cur);
      cur = p;
    } else {
      cur = cur ? cur + ' ' + p : p;
    }
  }
  if (cur) chunks.push(cur);
  return chunks;
}

async function runEdgeTts(text, outMp3, voiceName) {
  const args = ['--voice', voiceName, '--text', text, '--write-media', outMp3];
  if (EDGE_RATE) args.push('--rate', EDGE_RATE);
  if (EDGE_PITCH) args.push('--pitch', EDGE_PITCH);
  let lastErr;
  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      await execFileAsync('edge-tts', args, { timeout: 120000 });
      if (fs.existsSync(outMp3) && fs.statSync(outMp3).size > 0) return;
      throw new Error('edge-tts produced no output file');
    } catch (err) {
      lastErr = err;
      console.log(`[edge-tts] attempt ${attempt + 1} failed: ${String(err.message).slice(0, 120)}`);
      await sleep(5000 * (attempt + 1));
    }
  }
  throw lastErr || new Error('edge-tts failed after 3 attempts');
}

/**
 * Generate REAL Burmese speech for one scene with Edge TTS.
 * Returns { path (MP3), durationSec }.
 */
async function generateSceneAudioEdge(textMy, outMp3Path, voiceName = EDGE_VOICE) {
  if (!textMy || !textMy.trim()) throw new Error('generateSceneAudioEdge: empty narration text.');
  const chunks = chunkText(textMy);
  if (chunks.length === 1) {
    await runEdgeTts(chunks[0], outMp3Path, voiceName);
  } else {
    const parts = [];
    for (let i = 0; i < chunks.length; i++) {
      const part = outMp3Path.replace(/\.mp3$/i, `_part${i}.mp3`);
      await runEdgeTts(chunks[i], part, voiceName);
      parts.push(part);
    }
    const listPath = outMp3Path.replace(/\.mp3$/i, '_parts.txt');
    fs.writeFileSync(listPath, parts.map((p) => `file '${p}'`).join('\n'));
    await execFileAsync('ffmpeg', ['-y', '-f', 'concat', '-safe', '0', '-i', listPath, '-c', 'copy', outMp3Path]);
    for (const p of parts) {
      try {
        fs.unlinkSync(p);
      } catch (_) {}
    }
    try {
      fs.unlinkSync(listPath);
    } catch (_) {}
  }
  const durationSec = await gemini.ffprobeDuration(outMp3Path);
  return { path: outMp3Path, durationSec };
}

/** Per-scene Edge TTS for a list of scenes. Returns scenes with audioPath + audioDurationSec. */
async function generateAllSceneAudioEdge(scenes, audioDir, voiceName = EDGE_VOICE) {
  fs.mkdirSync(audioDir, { recursive: true });
  const out = [];
  for (let i = 0; i < scenes.length; i++) {
    const scene = scenes[i];
    const num = scene.sceneNumber || i + 1;
    const mp3 = path.join(audioDir, `scene_${num}.mp3`);
    const { durationSec } = await generateSceneAudioEdge(scene.narrationMy, mp3, voiceName);
    out.push({ ...scene, audioPath: mp3, audioDurationSec: durationSec });
  }
  return out;
}

module.exports = { EDGE_VOICE, generateSceneAudioEdge, generateAllSceneAudioEdge };
