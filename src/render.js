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
 * looped to fill the audio length (a visible repeat, not a silent gap).
 * If it is far longer, it is trimmed to the audio length.
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
  // Speed needed so the clip lasts as long as the audio: speed = sourceDur / audioDur.
  // Clamped to 0.9-1.1. If the clip is still too short after slowing to 0.9x,
  // loop the source segment to fill.
  const speed = clampSpeed(sourceDur / audioDurationSec);
  const adjustedDur = sourceDur / speed;
  const loop = adjustedDur < audioDurationSec * 0.98;

  const args = ['-y'];
  if (loop) args.push('-stream_loop', '-1');
  args.push('-ss', String(sourceStartSec), '-i', sourceVideo, '-i', audioPath);
  const filter = [
    '[0:v]trim=0:' + adjustedDur.toFixed(3) + ',setpts=PTS-STARTPTS,scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30[v]',
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
 * Render the full recap.
 * scenes: [{ sourceStartSec, sourceEndSec, audioPath, audioDurationSec }]
 * Returns { out720, out480, durationSec }.
 */
async function renderRecap({ sourceVideo, scenes, workDir, out720, out480 }) {
  if (!scenes || scenes.length === 0) throw new Error('renderRecap: no scenes with audio.');
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

module.exports = { ffprobeDuration, ffprobeStreams, cutClip, renderSceneSegment, renderRecap, clampSpeed };
