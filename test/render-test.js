'use strict';
/**
 * Render test — NO Gemini calls, NO API key needed.
 *
 * What it proves: the FFmpeg render path in src/render.js is real.
 * It downloads a short Google demo video (the same public sample bucket the
 * Android app used as placeholder content), generates 2 short tone-audio
 * MP3s with ffmpeg as stand-ins for TTS audio — TEST ONLY, clearly labelled,
 * NOT real Burmese TTS — cuts 2 clips, renders a final MP4, and verifies
 * with ffprobe that it has video + audio streams and duration > 0.
 *
 * Run: npm test
 * Output: test/output/final_720p.mp4 (+ final_480p.mp4) and test/TEST_RESULT.md
 */
const fs = require('fs');
const path = require('path');
const { execFile } = require('child_process');
const { promisify } = require('util');

const execFileAsync = promisify(execFile);
const { renderRecap, ffprobeStreams, ffprobeDuration } = require('../src/render');

const TEST_DIR = path.join(__dirname, 'output');
const SOURCE_CANDIDATES = [
  // Preferred: the same Google public sample bucket the Android app used.
  // NOTE (2026-11-14): this bucket returned HTTP 403 AccessDenied from this
  // VM, so the test falls through to the next source. The fallback is
  // documented in TEST_RESULT.md — the render itself is unchanged.
  'https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4',
  'https://storage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4',
  // Working mirror of the same Big Buck Bunny open movie (10s, 720p).
  'https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/720/Big_Buck_Bunny_720_10s_1MB.mp4',
];
const SOURCE_PATH = path.join(__dirname, 'test-source.mp4');

async function tryDownload(url, dest) {
  try {
    await execFileAsync('curl', ['-sL', '-o', dest, url]);
    if (!fs.existsSync(dest) || fs.statSync(dest).size < 100000) return false;
    await ffprobeDuration(dest); // throws if it is not a real video (e.g. an XML error page)
    return true;
  } catch (_) {
    return false;
  }
}

async function ensureSourceVideo() {
  for (const url of SOURCE_CANDIDATES) {
    console.log(`Trying source: ${url}`);
    if (await tryDownload(url, SOURCE_PATH)) {
      console.log(`Downloaded source from ${url}`);
      return { source: url, generatedLocally: false };
    }
    console.log('  -> failed (403 / not a video), trying next source...');
  }
  // Last resort: generate a 15s test video locally with ffmpeg. Still a
  // real video file, and the render path under test is identical.
  console.log('All downloads failed — generating a local test video with ffmpeg (testsrc).');
  await execFileAsync('ffmpeg', ['-y', '-f', 'lavfi', '-i', 'testsrc=duration=15:size=1280x720:rate=30',
    '-f', 'lavfi', '-i', 'sine=frequency=220:duration=15',
    '-c:v', 'libx264', '-preset', 'veryfast', '-c:a', 'aac', '-shortest', SOURCE_PATH]);
  await ffprobeDuration(SOURCE_PATH);
  return { source: 'ffmpeg testsrc (generated locally — downloads were blocked)', generatedLocally: true };
}

async function run(cmd, args) {
  await execFileAsync(cmd, args, { stdio: 'inherit' });
}

async function main() {
  fs.mkdirSync(TEST_DIR, { recursive: true });
  console.log('TEST ONLY: getting a short source video...');
  // Remove any stale failed download (e.g. a 397-byte XML error page).
  for (const stale of [path.join(__dirname, 'ForBiggerBlazes.mp4')]) {
    try { fs.unlinkSync(stale); } catch (_) {}
  }
  const srcInfo = await ensureSourceVideo();
  const sourceDur = await ffprobeDuration(SOURCE_PATH);
  console.log(`Source: ${SOURCE_PATH} (${sourceDur.toFixed(1)}s) from ${srcInfo.source}`);

  // TEST ONLY tone audio — stand-ins for real Gemini TTS MP3s.
  // These are sine tones, NOT Burmese speech. Real TTS is produced by
  // src/gemini.js generateSceneAudio() and needs GEMINI_API_KEY.
  const audio1 = path.join(TEST_DIR, 'TEST-ONLY-tone-scene1.mp3');
  const audio2 = path.join(TEST_DIR, 'TEST-ONLY-tone-scene2.mp3');
  await run('ffmpeg', ['-y', '-f', 'lavfi', '-i', 'sine=frequency=440:duration=3', '-codec:a', 'libmp3lame', '-q:a', '4', audio1]);
  await run('ffmpeg', ['-y', '-f', 'lavfi', '-i', 'sine=frequency=660:duration=4', '-codec:a', 'libmp3lame', '-q:a', '4', audio2]);
  const d1 = await ffprobeDuration(audio1);
  const d2 = await ffprobeDuration(audio2);
  console.log(`TEST ONLY audio: scene1 ${d1.toFixed(2)}s, scene2 ${d2.toFixed(2)}s (sine tones, not TTS)`);

  const out720 = path.join(TEST_DIR, 'final_720p.mp4');
  const out480 = path.join(TEST_DIR, 'final_480p.mp4');
  const result = await renderRecap({
    sourceVideo: SOURCE_PATH,
    scenes: [
      { sourceStartSec: 0, sourceEndSec: 5, audioPath: audio1, audioDurationSec: d1 },
      { sourceStartSec: 5, sourceEndSec: 10, audioPath: audio2, audioDurationSec: d2 },
    ],
    workDir: TEST_DIR,
    out720,
    out480,
  });

  const streams = await ffprobeStreams(out720);
  const hasVideo = streams.some((s) => s.codec_type === 'video');
  const hasAudio = streams.some((s) => s.codec_type === 'audio');
  const streams480 = await ffprobeStreams(out480);

  const report = [
    '# Render test result',
    '',
    'Date: ' + new Date().toISOString(),
    'Test: `npm test` (test/render-test.js) — NO Gemini calls, NO API key used.',
    '',
    '## What was tested',
    '- Source: ' + srcInfo.source + ' (' + sourceDur.toFixed(1) + 's)',
    '- Audio: TEST ONLY sine-tone MP3s generated with ffmpeg (440 Hz / 3s, 660 Hz / 4s) as stand-ins for TTS. These are NOT real Burmese TTS.',
    '- Render: src/render.js renderRecap() — 2 scenes, clip cut to match audio duration, concat, 720p + 480p export.',
    '',
    '## Result',
    `- Output 720p: test/output/final_720p.mp4`,
    `- Output 480p: test/output/final_480p.mp4`,
    `- Duration: ${result.durationSec.toFixed(2)}s`,
    `- Streams (720p): ${JSON.stringify(streams)}`,
    `- Streams (480p): ${JSON.stringify(streams480)}`,
    `- Has video stream: ${hasVideo}`,
    `- Has audio stream: ${hasAudio}`,
    `- PASS: ${hasVideo && hasAudio && result.durationSec > 0 ? 'YES — real MP4 with video + audio' : 'NO'}`,
    '',
    '## What this does NOT prove',
    '- Gemini analysis, Burmese script generation, and Gemini TTS are NOT tested here — they need GEMINI_API_KEY.',
    '- A full 5-minute recap (7 scenes, real TTS) is not tested here — only the FFmpeg render path with 2 scenes.',
    '',
  ].join('\n');
  fs.writeFileSync(path.join(__dirname, 'TEST_RESULT.md'), report);
  console.log(report);
  if (!hasVideo || !hasAudio || result.durationSec <= 0) {
    throw new Error('Render test FAILED: output is missing video/audio or has zero duration.');
  }
  console.log('Render test PASSED.');
}

main().catch((e) => { console.error(e); process.exit(1); });
