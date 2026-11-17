'use strict';
/**
 * Myanmar Movie Recap — real backend (Express).
 *
 * Endpoints (spec: ~/workspace/movie-recap-real-backend.md):
 *   POST /api/jobs              multipart video upload -> { jobId, status: "queued" }
 *   GET  /api/jobs/:id          status + progress + scenes + final video URL
 *   POST /api/jobs/:id/script   approve / edit scenes { scenes: [...], title? }
 *   POST /api/jobs/:id/generate start real TTS + render (only after approval)
 *   GET  /api/jobs/:id/video    stream final MP4 (?quality=480|720, default 720)
 *   GET  /health                { ok: true }
 *
 * No fake progress: /generate returns 202 and the job runs async; the app
 * must poll GET /api/jobs/:id. A job that fails reports its real error.
 */
require('dotenv').config();
const express = require('express');
const multer = require('multer');
const fs = require('fs');
const path = require('path');

const store = require('./src/store');
const pipeline = require('./src/pipeline');
const gemini = require('./src/gemini');

const PORT = Number(process.env.PORT || 8080);
const MAX_UPLOAD_MB = Number(process.env.MAX_UPLOAD_MB || 500);
const VALID_LENGTHS = [5, 10, 15, 30, 60];

const app = express();
// CORS: the JSON API (approve script, generate) is called cross-origin from
// the test helper page and the user's tools. No cookies/credentials are
// involved — the API key never leaves the server — so allow any origin.
// Without this, browsers block POST /api/jobs/:id/script and /generate
// (preflight fails) and only curl/console could approve a script.
app.use((req, res, next) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  if (req.method === 'OPTIONS') return res.sendStatus(204);
  next();
});
app.use(express.json({ limit: '1mb' }));

const upload = multer({
  dest: path.join(__dirname, 'data', 'uploads'),
  limits: { fileSize: MAX_UPLOAD_MB * 1024 * 1024 },
});

app.get('/health', (_req, res) => res.json({ ok: true, service: 'myanmar-movie-recap-backend' }));

app.get('/api/jobs', (_req, res) => res.json({ jobs: store.listJobs() }));

app.post('/api/jobs', upload.single('video'), async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ error: 'No video file uploaded. Send multipart field "video".' });
    const lengthMinutes = Number(req.body.lengthMinutes || 10);
    if (!VALID_LENGTHS.includes(lengthMinutes)) {
      fs.unlinkSync(req.file.path);
      return res.status(400).json({ error: `lengthMinutes must be one of ${VALID_LENGTHS.join(', ')} (minutes). Got ${req.body.lengthMinutes}.` });
    }
    const category = String(req.body.category || 'Action').trim() || 'Action';
    const ext = path.extname(req.file.originalname || '').toLowerCase() || '.mp4';
    const jobDir = path.join(__dirname, 'data', 'jobs');
    fs.mkdirSync(jobDir, { recursive: true });

    // Create the job first so we have an id / directory for the source file.
    const job = store.createJob({
      fileName: req.file.originalname || `video${ext}`,
      originalName: req.file.originalname || `video${ext}`,
      category,
      lengthMinutes,
      sourcePath: req.file.path, // moved below once the dir exists
    });
    const dest = path.join(store.jobDir(job.jobId), `source${ext}`);
    fs.renameSync(req.file.path, dest);
    store.updateJob(job.jobId, { sourcePath: dest });

    // Start analysis+scripting in the background (does not block the response).
    // Errors are recorded on the job itself as status=failed + error.
    pipeline.runAnalysis(job.jobId).catch((e) => console.error(e));

    return res.status(202).json({ jobId: job.jobId, status: 'queued', lengthMinutes, category });
  } catch (err) {
    console.error(err);
    return res.status(500).json({ error: err.message });
  }
});

app.get('/api/jobs/:id', (req, res) => {
  const job = store.getJob(req.params.id);
  if (!job) return res.status(404).json({ error: 'Job not found.' });
  return res.json(store.publicJob(job));
});

app.post('/api/jobs/:id/script', (req, res) => {
  const job = store.getJob(req.params.id);
  if (!job) return res.status(404).json({ error: 'Job not found.' });
  if (job.status !== 'awaiting-approval') {
    return res.status(409).json({ error: `Script cannot be approved yet — job is ${job.status}. Wait for the script draft (awaiting-approval).` });
  }
  const scenes = req.body && Array.isArray(req.body.scenes) ? req.body.scenes : null;
  if (!scenes || scenes.length === 0) {
    return res.status(400).json({ error: 'Body must contain { scenes: [...] } with at least one scene.' });
  }
  for (const s of scenes) {
    if (!s.narrationMy || typeof s.narrationMy !== 'string') {
      return res.status(400).json({ error: 'Every scene needs narrationMy (Burmese narration text).' });
    }
    if (!Number.isFinite(Number(s.sourceStartSec)) || !Number.isFinite(Number(s.sourceEndSec))) {
      return res.status(400).json({ error: 'Every scene needs numeric sourceStartSec and sourceEndSec (timestamps in the source video).' });
    }
  }
  const updated = store.updateJob(job.jobId, {
    title: typeof req.body.title === 'string' ? req.body.title : job.title,
    // Keep the code-built per-sentence segments through approval — dropping
    // them here used to silently disable moment-cutting for the whole job.
    // Segments survive ONLY when they still match the approved narration:
    // if the user edited the text, the segment texts no longer join to it
    // and the stale segments are dropped (the scene then renders
    // scene-level, shrunk to its first moment). driftRisk is recomputed
    // from the approved narration and range.
    scenes: scenes.map((s, i) => {
      const narrationMy = String(s.narrationMy);
      const startSec = Number(s.sourceStartSec);
      const endSec = Number(s.sourceEndSec);
      let segments = null;
      if (Array.isArray(s.segments) && s.segments.length > 1) {
        const norm = s.segments
          .map((g) => ({
            text: String((g && g.text) || '').trim(),
            sourceStartSec: Number((g && g.sourceStartSec) || 0),
            sourceEndSec: Number((g && g.sourceEndSec) || 0),
          }))
          .filter((g) => g.text && g.sourceEndSec > g.sourceStartSec);
        const joined = norm.map((g) => g.text).join(' ');
        if (norm.length > 1 && joined.replace(/\s+/g, ' ') === narrationMy.replace(/\s+/g, ' ')) {
          segments = norm;
        }
      }
      return {
        sceneNumber: Number(s.sceneNumber) || i + 1,
        sourceStartSec: startSec,
        sourceEndSec: endSec,
        startSec,
        endSec,
        narrationMy,
        ...(segments ? { segments } : {}),
        driftRisk: gemini.computeDriftRisk(narrationMy, startSec, endSec, Boolean(segments)),
        audioPath: null,
        audioDurationSec: null,
      };
    }),
  });
  return res.json({ jobId: job.jobId, status: updated.status, scenes: updated.scenes.length, message: 'Script approved. Call POST /api/jobs/:id/generate to start TTS + render.' });
});

app.post('/api/jobs/:id/generate', (req, res) => {
  const job = store.getJob(req.params.id);
  if (!job) return res.status(404).json({ error: 'Job not found.' });
  if (job.status !== 'awaiting-approval') {
    return res.status(409).json({ error: `Cannot generate — job is ${job.status}. The script must be approved first (awaiting-approval).` });
  }
  if (!process.env.GEMINI_API_KEY) {
    return res.status(500).json({ error: 'GEMINI_API_KEY is not set on the server. Set it in .env / Cloud Run secrets — audio is never faked.' });
  }
  store.updateJob(job.jobId, { status: 'tts', progress: 40, error: null });
  pipeline.runGenerate(job.jobId).catch((e) => console.error(e));
  return res.status(202).json({ jobId: job.jobId, status: 'tts', message: 'TTS + render started. Poll GET /api/jobs/:id for progress.' });
});

app.get('/api/jobs/:id/video', (req, res) => {
  const job = store.getJob(req.params.id);
  if (!job) return res.status(404).json({ error: 'Job not found.' });
  if (job.status !== 'done') {
    return res.status(409).json({ error: `Final video is not ready — job is ${job.status} (${job.progress}%).` });
  }
  const quality = req.query.quality === '480' ? '480' : '720';
  const file = path.join(store.jobDir(job.jobId), quality === '480' ? 'final_480p.mp4' : 'final_720p.mp4');
  if (!fs.existsSync(file)) return res.status(404).json({ error: 'Final video file not found on disk.' });
  res.setHeader('Content-Type', 'video/mp4');
  res.setHeader('Accept-Ranges', 'bytes');
  return res.sendFile(file);
});

// Multer / upload errors (e.g. file too large) — return JSON, not an HTML page.
app.use((err, _req, res, _next) => {
  if (err && err.code === 'LIMIT_FILE_SIZE') {
    return res.status(413).json({ error: `Video is too large. Max upload is ${MAX_UPLOAD_MB} MB for this server.` });
  }
  console.error(err);
  return res.status(500).json({ error: err && err.message ? err.message : 'Server error.' });
});

if (require.main === module) {
  app.listen(PORT, () => {
    console.log(`Myanmar Movie Recap backend listening on port ${PORT}`);
    console.log(`GEMINI_API_KEY set: ${Boolean(process.env.GEMINI_API_KEY)} (script/TTS will fail honestly if missing)`);
    console.log(`Max upload: ${MAX_UPLOAD_MB} MB`);
  });
}

module.exports = app;
