'use strict';
/**
 * Job store: in-memory Map mirrored to disk as JSON.
 * Layout per job, under ./data/jobs/<jobId>/ :
 *   job.json        - the job record (status, progress, scenes, error, ...)
 *   source.mp4      - uploaded source video (deleted after render, see pipeline)
 *   frames/         - frames extracted for Gemini analysis
 *   audio/scene-01.mp3 ...
 *   segments/scene-01.mp4 ...
 *   final_720p.mp4 / final_480p.mp4
 */
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

const DATA_ROOT = path.join(__dirname, '..', 'data', 'jobs');
fs.mkdirSync(DATA_ROOT, { recursive: true });

const jobs = new Map();

function jobDir(id) {
  return path.join(DATA_ROOT, id);
}

function publicJob(job) {
  if (!job) return null;
  return {
    jobId: job.jobId,
    status: job.status, // queued | analysing | scripting | awaiting-approval | tts | rendering | done | failed
    stage: job.status,
    progress: job.progress,
    error: job.error || null,
    fileName: job.fileName,
    originalName: job.originalName,
    category: job.category,
    lengthMinutes: job.lengthMinutes,
    title: job.title || null,
    scenes: job.scenes || [],
    sceneSummaries: job.sceneSummaries || [],
    sourceDurationSec: job.sourceDurationSec ?? null,
    audioDurations: job.audioDurations || {},
    finalVideoUrl: job.finalVideoUrl || null,
    finalVideo720p: job.finalVideo720p || null,
    finalVideo480p: job.finalVideo480p || null,
    createdAt: job.createdAt,
    updatedAt: job.updatedAt,
  };
}

function saveJob(job) {
  job.updatedAt = new Date().toISOString();
  fs.mkdirSync(jobDir(job.jobId), { recursive: true });
  fs.writeFileSync(
    path.join(jobDir(job.jobId), 'job.json'),
    JSON.stringify(job, null, 2)
  );
  jobs.set(job.jobId, job);
  return job;
}

function createJob({ fileName, originalName, category, lengthMinutes, sourcePath }) {
  const jobId = crypto.randomUUID();
  const job = {
    jobId,
    status: 'queued',
    progress: 0,
    error: null,
    fileName,
    originalName: originalName || fileName,
    category: category || 'Action',
    lengthMinutes,
    title: null,
    scenes: [],
    sceneSummaries: [],
    sourceDurationSec: null,
    audioDurations: {},
    finalVideoUrl: null,
    finalVideo720p: null,
    finalVideo480p: null,
    sourcePath, // internal only — never returned by publicJob()
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
  };
  return saveJob(job);
}

function getJob(id) {
  if (jobs.has(id)) return jobs.get(id);
  const file = path.join(jobDir(id), 'job.json');
  if (!fs.existsSync(file)) return null;
  const job = JSON.parse(fs.readFileSync(file, 'utf8'));
  jobs.set(id, job);
  return job;
}

function updateJob(id, patch) {
  const job = getJob(id);
  if (!job) return null;
  Object.assign(job, patch);
  return saveJob(job);
}

function listJobs() {
  return fs
    .readdirSync(DATA_ROOT, { withFileTypes: true })
    .filter((d) => d.isDirectory())
    .map((d) => getJob(d.name))
    .filter(Boolean)
    .sort((a, b) => (b.createdAt || '').localeCompare(a.createdAt || ''))
    .map(publicJob);
}

module.exports = { DATA_ROOT, jobDir, publicJob, createJob, getJob, updateJob, saveJob, listJobs };
