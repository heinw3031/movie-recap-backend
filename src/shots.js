'use strict';
/**
 * Shot-boundary + semantic clip matching — movie-narrator step 9
 * (`match_clips`) STRUCTURE, our own Node implementation. No AGPL code
 * was copied; the ideas (real shot boundaries, greedy assignment with
 * reuse penalty + diversity, speed clamp, match_summary with a degraded
 * flag) are re-implemented here from the plan in
 * ~/workspace/research_notes/movie-narrator-adaptation/plan.md.
 *
 * Why this exists (2026-11-18): the core defect is grounding — Gemini
 * analysis mis-grounds most moments (live job 12fe9ad6: 0/96 verified),
 * so narration often does not match the picture. The script's scene
 * windows are LLM-guessed time ranges. This module replaces the guess
 * with evidence:
 *   1. detectShots — FFmpeg `select='gt(scene,0.4)'` gives REAL cut
 *      points (no new dependency; FFmpeg is already on Railway).
 *   2. captionShotFrames (in src/gemini.js) — Gemini 3.8-flash, itself a
 *      VLM, captions one frame per shot midpoint, one generateContent
 *      call per <=40 frames (the verifyMoments batching pattern).
 *   3. assignClips — greedy per-beat assignment: semantic text overlap
 *      between the beat's English description and the shot caption,
 *      plus a story-position anchor bonus and a duration-fit term;
 *      reuse penalty (-0.15) and a hard cap (never the same shot 2x in
 *      a 3-beat window); playback speed clamped to [0.85, 1.25].
 *   4. match_summary with degraded_reason — the job record says
 *      honestly when matching is weak (too few captions / all
 *      heuristic / low confidence) instead of shipping a mismatched
 *      video and pretending it matched.
 *
 * Safety: every step degrades, never fails the job. If shot detection
 * or captioning fails, matchScenesToShots returns null and the pipeline
 * keeps the LLM-guessed windows. A beat whose best semantic score is
 * below LOW_SCORE falls back to the heuristic (nearest shot to its
 * story anchor) for that beat alone.
 *
 * Quota: captioning costs ceil(captionedShots / 40) Gemini calls.
 * Shots are sampled down to SHOT_MATCH_MAX_SHOTS (default 120) before
 * captioning, so a 30-min source (~300-400 shots) costs 3 captioning
 * calls, not 10. The call count is logged and stored in the summary.
 *
 * Env: SHOT_MATCH='0' disables the whole step; SHOT_MATCH_MAX_SHOTS
 * caps captioned shots (default 120); SHOT_DETECT_THRESHOLD sets the
 * scene-change threshold (default 0.4).
 */
const fs = require('fs');
const path = require('path');
const { execFile } = require('child_process');
const { promisify } = require('util');

const execFileAsync = promisify(execFile);

const SHOT_MATCH_ENABLED = process.env.SHOT_MATCH !== '0';
const SHOT_MATCH_MAX_SHOTS = Math.max(
  10,
  parseInt(process.env.SHOT_MATCH_MAX_SHOTS || '120', 10) || 120
);
const SHOT_DETECT_THRESHOLD = (() => {
  const t = parseFloat(process.env.SHOT_DETECT_THRESHOLD || '0.4');
  return Number.isFinite(t) && t > 0 && t < 1 ? t : 0.4;
})();
// movie-narrator's low-score threshold: a segment scoring below this
// falls back to heuristic matching for that segment.
const LOW_SCORE = 0.25;
// Greedy reuse rules (movie-narrator match_clips structure): a shot used
// in the last 3 beats costs -0.15; a shot already used 2x in the last 3
// beats is excluded entirely (diversity post-processing).
const REUSE_PENALTY = 0.15;
const REUSE_WINDOW = 3;
const REUSE_MAX_IN_WINDOW = 2;
// Playback speed clamp, per the plan (renderer uses 0.9-1.1 for gentle
// tempo; the matcher allows a slightly wider band when CHOOSING a shot,
// and flags beats whose raw ratio fell outside it).
const SPEED_MIN = 0.85;
const SPEED_MAX = 1.25;
// Score weights: semantic text overlap dominates; the story-position
// anchor and the duration fit break ties and keep cuts sane.
const W_TEXT = 0.55;
const W_ANCHOR = 0.25;
const W_DURATION = 0.2;
// Degraded threshold (movie-narrator's honesty guard: >70% fake
// captions forces heuristic matching — mirrored here as <30% captioned).
const MIN_CAPTION_COVERAGE = 0.3;

function clampSpeed(raw) {
  return Math.min(SPEED_MAX, Math.max(SPEED_MIN, raw));
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

/**
 * Parse FFmpeg showinfo stderr into sorted, de-duplicated shot-boundary
 * timestamps (seconds). Pure — no I/O, unit-tested. showinfo prints one
 * line per selected frame containing `pts_time:<seconds>`; config lines
 * carry no pts_time and are ignored.
 */
function parseShowinfoTimestamps(stderr) {
  const times = [];
  const re = /pts_time:([\d.]+)/g;
  let m;
  while ((m = re.exec(String(stderr || ''))) !== null) {
    const t = parseFloat(m[1]);
    if (Number.isFinite(t) && t >= 0) times.push(t);
  }
  times.sort((a, b) => a - b);
  const deduped = [];
  for (const t of times) {
    if (deduped.length === 0 || Math.abs(t - deduped[deduped.length - 1]) > 0.05) deduped.push(t);
  }
  return deduped;
}

/**
 * Detect real shot boundaries with FFmpeg scene-change detection.
 * Returns [{ index, startSec, endSec, midSec, durationSec }] covering
 * the whole video. The first frame has no predecessor, so `select`
 * never emits t=0 — the first shot always starts at 0. The last shot
 * ends at the video duration (ffprobe). When no boundary is found the
 * whole video is one shot (a static / single-take source). Throws on
 * ffmpeg failure (missing file, undecodable) — the caller treats a
 * throw as "fall back to LLM windows".
 */
async function detectShots(videoPath, opts = {}) {
  const threshold = opts.threshold || SHOT_DETECT_THRESHOLD;
  const duration = await ffprobeDuration(videoPath);
  let stderr = '';
  try {
    const result = await execFileAsync('ffmpeg', [
      '-v', 'info', '-i', videoPath,
      '-vf', `select='gt(scene,${threshold})',showinfo`,
      '-f', 'null', '-',
    ]);
    stderr = result.stderr || '';
  } catch (err) {
    // ffmpeg exits non-zero on -f null with some inputs but still prints
    // showinfo; use whatever it printed, and only throw if nothing came
    // out at all (real failure, not a quirk).
    stderr = (err && err.stderr) || '';
    if (!/pts_time:/.test(stderr)) throw err;
  }
  const boundaries = parseShowinfoTimestamps(stderr);
  const starts = [0, ...boundaries.filter((t) => t > 0.05 && t < duration - 0.05)];
  const shots = starts.map((start, i) => {
    const end = i + 1 < starts.length ? starts[i + 1] : duration;
    return {
      index: i,
      startSec: start,
      endSec: end,
      midSec: (start + end) / 2,
      durationSec: Math.max(0.05, end - start),
    };
  });
  return shots;
}

/**
 * Evenly sample shots down to maxShots, always keeping the first and
 * last shot (story endpoints matter). Pure — unit-tested.
 */
function sampleShots(shots, maxShots) {
  const list = Array.isArray(shots) ? shots : [];
  const max = Math.max(1, maxShots || SHOT_MATCH_MAX_SHOTS);
  if (list.length <= max) return list.slice();
  if (max === 1) return [list[0]];
  const out = [];
  for (let i = 0; i < max; i++) {
    const idx = Math.round((i * (list.length - 1)) / (max - 1));
    if (out.length === 0 || out[out.length - 1].index !== list[idx].index) out.push(list[idx]);
  }
  return out;
}

// --- Semantic scoring (pure, unit-tested) ---

const STOPWORDS = new Set(
  ('a an the and or but of to in on at for with by from as is are was were be been being ' +
    'it its this that these those he she they we you i his her their our your my me him them us ' +
    'who whom what when where why how which there here then than so such no not only also just ' +
    'very more most some any all each every do does did done can could will would should may might ' +
    'has have had having about into over under again once while during before after above below ' +
    'between through out up down off').split(' ')
);

/**
 * Content tokens: lowercase, Latin alphanumerics plus whitespace-split
 * tokens for other scripts, stopwords removed, naive plural strip
 * ("rooftops" matches "rooftop"). Burmese narration has no reliable
 * word boundaries, so beat text should be the English beat description
 * when one exists (matchScenesToShots prefers it) — token overlap on
 * Burmese text degrades to whole-string tokens and scores ~0, which
 * the anchor + duration terms then carry.
 */
function tokenize(text) {
  const raw = String(text || '')
    .toLowerCase()
    .replace(/[^a-z0-9\u1000-\u109f]+/g, ' ')
    .split(' ')
    .map((t) => t.trim())
    .filter(Boolean);
  const out = new Set();
  for (let t of raw) {
    if (STOPWORDS.has(t)) continue;
    if (t.length > 3 && t.endsWith('s') && !t.endsWith('ss')) t = t.slice(0, -1);
    if (t) out.add(t);
  }
  return out;
}

/**
 * Overlap coefficient: |A∩B| / min(|A|,|B|). Chosen over Jaccard because
 * a shot caption is one short clause (~6-10 content words) while a beat
 * description is 1-2 sentences (~10-25 words) — Jaccard punishes the
 * length mismatch even on a true match. 1.0 = the shorter text's words
 * are all in the longer one; 0 = nothing shared. Empty inputs score 0.
 */
function textOverlap(aText, bText) {
  const a = tokenize(aText);
  const b = tokenize(bText);
  if (a.size === 0 || b.size === 0) return 0;
  let shared = 0;
  for (const t of a) if (b.has(t)) shared++;
  return shared / Math.min(a.size, b.size);
}

function durationFit(shotDur, beatDur) {
  if (!(beatDur > 0) || !(shotDur > 0)) return 0.5;
  const ratio = shotDur / beatDur;
  if (ratio >= SPEED_MIN && ratio <= SPEED_MAX) return 1;
  return Math.max(0, 1 - Math.abs(ratio - 1) * 1.5);
}

function anchorScore(shotMidSec, anchorSec, videoDurationSec) {
  if (!(videoDurationSec > 0)) return 0.5;
  const d = Math.abs(shotMidSec - anchorSec) / videoDurationSec;
  return 1 - Math.min(1, d);
}

/**
 * Score one (beat, shot) pair. Returns { score, textScore,
 * anchorScore, durationScore, speed, speedClamped }. The reuse penalty
 * is applied by assignClips (it needs assignment history), not here.
 */
function scoreShot(beat, shot, videoDurationSec) {
  const textScore = textOverlap(beat.text || '', shot.caption || '');
  const aScore = anchorScore(shot.midSec, beat.anchorSec || 0, videoDurationSec);
  const dScore = durationFit(shot.durationSec, beat.durationSec || 0);
  const score = W_TEXT * textScore + W_ANCHOR * aScore + W_DURATION * dScore;
  const rawSpeed = (shot.durationSec || 0) / Math.max(0.1, beat.durationSec || 0);
  return {
    score,
    textScore,
    anchorScore: aScore,
    durationScore: dScore,
    speed: clampSpeed(rawSpeed),
    speedClamped: rawSpeed < SPEED_MIN || rawSpeed > SPEED_MAX,
  };
}

/**
 * Heuristic fallback for one beat (movie-narrator priority: beat anchor
 * > act bucket > uniform): the shot whose midpoint is nearest the
 * beat's story anchor. Pure — unit-tested.
 */
function heuristicShot(beat, shots) {
  if (!Array.isArray(shots) || shots.length === 0) return null;
  const anchor = beat.anchorSec || 0;
  let best = shots[0];
  let bestD = Math.abs(shots[0].midSec - anchor);
  for (const s of shots) {
    const d = Math.abs(s.midSec - anchor);
    if (d < bestD) {
      bestD = d;
      best = s;
    }
  }
  return best;
}

/**
 * Greedy assignment of beats to shots. Pure — no I/O, unit-tested.
 * beats: [{ text, anchorSec, durationSec, startSec, endSec }] in story
 *   order. `text` is the English beat description when available.
 * shots: [{ index, startSec, endSec, midSec, durationSec, caption }]
 *   — caption '' means uncaptioned (scores on anchor + duration only).
 * videoDurationSec: source length, for the anchor term.
 * Returns { assignments, summary } where each assignment is
 *   { beatIndex, shotIndex|null, startSec, endSec, score, speed,
 *     reused, method: 'semantic'|'heuristic'|'none' }.
 * Rules: best semantic score wins; reuse in the last REUSE_WINDOW
 * beats costs REUSE_PENALTY; a shot used REUSE_MAX_IN_WINDOW times in
 * that window is excluded; a best score below LOW_SCORE (or no captions
 * at all) falls back to the heuristic shot for that beat; no shots at
 * all keeps the beat's original window (method 'none'). A semantic
 * match additionally requires textScore > 0 — the anchor + duration
 * terms alone can reach 0.45, so without this a well-placed but
 * unrelated shot would count as "semantic" even though its caption
 * shares nothing with the beat.
 */
function assignClips(beats, shots, videoDurationSec, opts = {}) {
  const lowScore = opts.lowScore !== undefined ? opts.lowScore : LOW_SCORE;
  const list = Array.isArray(beats) ? beats : [];
  const shotList = Array.isArray(shots) ? shots : [];
  const anyCaption = shotList.some((s) => String(s.caption || '').trim().length > 0);
  const assignments = [];
  const history = []; // shot indexes, in beat order (last REUSE_WINDOW matter)
  let matchedSemantic = 0;
  let matchedHeuristic = 0;
  let matchedNone = 0;
  let scoreSum = 0;
  let scoreCount = 0;
  let clampedCount = 0;
  let reusedCount = 0;

  for (let bi = 0; bi < list.length; bi++) {
    const beat = list[bi];
    const origStart = Number(beat.startSec) || 0;
    const origEnd = Number(beat.endSec) || origStart;
    if (shotList.length === 0) {
      matchedNone++;
      assignments.push({ beatIndex: bi, shotIndex: null, startSec: origStart, endSec: origEnd, score: 0, speed: 1, reused: false, method: 'none' });
      history.push(null);
      continue;
    }
    const recent = history.slice(-REUSE_WINDOW);
    const useCount = new Map();
    for (const idx of recent) if (idx !== null && idx !== undefined) useCount.set(idx, (useCount.get(idx) || 0) + 1);

    let best = null;
    let bestScored = null;
    if (anyCaption) {
      for (const shot of shotList) {
        const used = useCount.get(shot.index) || 0;
        if (used >= REUSE_MAX_IN_WINDOW) continue; // diversity hard cap
        const scored = scoreShot(beat, shot, videoDurationSec);
        // No shared words = no semantic evidence, whatever the anchor
        // and duration say — such a shot can only be a heuristic pick.
        if (scored.textScore <= 0) continue;
        let s = scored.score;
        if (used > 0) s -= REUSE_PENALTY;
        // NOTE: compare against bestScored.score — `best` is the shot
        // object and carries no .score, so `s > best.score` is always
        // false and the first eligible shot would win every time.
        if (!bestScored || s > bestScored.score) {
          best = shot;
          bestScored = { ...scored, score: s, reused: used > 0 };
        }
      }
    }
    if (best && bestScored.score >= lowScore) {
      matchedSemantic++;
      scoreSum += bestScored.score;
      scoreCount++;
      if (bestScored.speedClamped) clampedCount++;
      if (bestScored.reused) reusedCount++;
      assignments.push({
        beatIndex: bi, shotIndex: best.index, startSec: best.startSec, endSec: best.endSec,
        score: bestScored.score, speed: bestScored.speed, reused: bestScored.reused, method: 'semantic',
      });
      history.push(best.index);
      continue;
    }
    // Heuristic fallback for this beat: nearest shot to the anchor.
    const h = heuristicShot(beat, shotList);
    if (!h) {
      matchedNone++;
      assignments.push({ beatIndex: bi, shotIndex: null, startSec: origStart, endSec: origEnd, score: 0, speed: 1, reused: false, method: 'none' });
      history.push(null);
      continue;
    }
    const hScored = scoreShot(beat, h, videoDurationSec);
    matchedHeuristic++;
    if (hScored.speedClamped) clampedCount++;
    assignments.push({
      beatIndex: bi, shotIndex: h.index, startSec: h.startSec, endSec: h.endSec,
      score: hScored.score, speed: hScored.speed, reused: (useCount.get(h.index) || 0) > 0, method: 'heuristic',
    });
    history.push(h.index);
  }

  const summary = buildMatchSummary({
    shotsDetected: shotList.length,
    shotsCaptioned: shotList.filter((s) => String(s.caption || '').trim().length > 0).length,
    beats: list.length,
    matchedSemantic,
    matchedHeuristic,
    matchedNone,
    avgScore: scoreCount > 0 ? scoreSum / scoreCount : 0,
    clampedCount,
    reusedCount,
  });
  return { assignments, summary };
}

/**
 * match_summary, movie-narrator structure. degraded_reason is set when
 * matching is unreliable so the job record can say so honestly:
 *  - 'too_few_captions': captioned shots < 30% of detected — the
 *    semantic path had almost nothing to match on.
 *  - 'all_heuristic': no beat matched semantically.
 *  - 'low_confidence': mean semantic score below LOW_SCORE.
 * Priority: too_few_captions > all_heuristic > low_confidence > null.
 * Pure — unit-tested.
 */
function buildMatchSummary(stats) {
  const {
    shotsDetected = 0, shotsCaptioned = 0, beats = 0,
    matchedSemantic = 0, matchedHeuristic = 0, matchedNone = 0,
    avgScore = 0, clampedCount = 0, reusedCount = 0, captionCalls = 0,
  } = stats || {};
  let degradedReason = null;
  const coverage = shotsDetected > 0 ? shotsCaptioned / shotsDetected : 0;
  if (shotsDetected > 0 && coverage < MIN_CAPTION_COVERAGE) {
    degradedReason = 'too_few_captions';
  } else if (beats > 0 && matchedSemantic === 0) {
    degradedReason = 'all_heuristic';
  } else if (matchedSemantic > 0 && avgScore < LOW_SCORE) {
    degradedReason = 'low_confidence';
  }
  return {
    shotsDetected,
    shotsCaptioned,
    captionCalls,
    beats,
    matchedSemantic,
    matchedHeuristic,
    matchedNone,
    avgScore: Math.round(avgScore * 1000) / 1000,
    clampedCount,
    reusedCount,
    degraded_reason: degradedReason,
  };
}

/**
 * Build the per-scene beats for assignment from the approved scenes.
 * anchorSec: the midpoint of the scene's LLM-guessed window — the
 * story-position anchor (movie-narrator's approx_ratio, as seconds).
 * text: the overlapping analysis beat's English description when one
 * exists (token overlap needs English; the narration itself is
 * Burmese) — falls back to the narration text.
 * Pure — unit-tested.
 */
function buildBeats(scenes, sceneSummaries) {
  const list = Array.isArray(scenes) ? scenes : [];
  return list.map((scene, i) => {
    const startSec = Number(scene.sourceStartSec) || 0;
    const endSec = Number(scene.sourceEndSec) || startSec;
    let text = String(scene.narrationMy || '').trim();
    if (Array.isArray(sceneSummaries) && sceneSummaries.length > 0) {
      let best = null;
      let bestOverlap = -1;
      for (const b of sceneSummaries) {
        const overlap = Math.max(0, Math.min(endSec, b.endSec) - Math.max(startSec, b.startSec));
        if (overlap > bestOverlap) {
          bestOverlap = overlap;
          best = b;
        }
      }
      if (best && bestOverlap > 0 && String(best.description || '').trim()) {
        text = String(best.description).trim();
      }
    }
    return {
      sceneIndex: i,
      text,
      anchorSec: (startSec + endSec) / 2,
      durationSec: Math.max(0.1, Number(scene.audioDurationSec) || (endSec - startSec)),
      startSec,
      endSec,
    };
  });
}

/**
 * Full orchestrator: detect shots -> caption them with Gemini -> assign
 * each approved scene to a shot -> return scenes with refined windows
 * plus the match summary. Returns null on ANY failure (or when
 * disabled / no scenes) — the caller keeps the LLM-guessed windows.
 * opts (all optional, for tests): enabled, maxShots, threshold,
 * detectFn(videoPath) -> shots, captionFn(shots, videoPath, workDir)
 * -> { shots, stats }, videoDurationSec.
 */
async function matchScenesToShots({ sourceVideo, scenes, sceneSummaries, workDir, jobId, opts = {} }) {
  const enabled = opts.enabled !== undefined ? opts.enabled : SHOT_MATCH_ENABLED;
  if (!enabled) return null;
  if (!Array.isArray(scenes) || scenes.length === 0) return null;
  if (!sourceVideo || !fs.existsSync(sourceVideo)) return null;
  const label = jobId ? `[job ${jobId}] ` : '';
  try {
    const videoDurationSec = opts.videoDurationSec || (await ffprobeDuration(sourceVideo));
    const detectFn = opts.detectFn || detectShots;
    const allShots = await detectFn(sourceVideo, { threshold: opts.threshold });
    if (!Array.isArray(allShots) || allShots.length === 0) {
      console.log(`${label}shot match: no shots detected — keeping LLM windows`);
      return null;
    }
    const maxShots = opts.maxShots || SHOT_MATCH_MAX_SHOTS;
    const shots = sampleShots(allShots, maxShots);
    const captionFn = opts.captionFn || null;
    let captioned;
    if (captionFn) {
      captioned = await captionFn(shots, sourceVideo, workDir);
    } else {
      const gemini = require('./gemini');
      captioned = await gemini.captionShotFrames(shots, sourceVideo, workDir);
    }
    const captionedShots = (captioned && Array.isArray(captioned.shots) ? captioned.shots : shots);
    const captionCalls = (captioned && captioned.stats && captioned.stats.calls) || 0;
    console.log(`${label}shot match: ${allShots.length} shots detected, ${captionedShots.filter((s) => String(s.caption || '').trim()).length}/${captionedShots.length} captioned in ${captionCalls} Gemini call(s)`);
    const beats = buildBeats(scenes, sceneSummaries);
    const { assignments, summary } = assignClips(beats, captionedShots, videoDurationSec);
    summary.captionCalls = captionCalls;
    summary.shotsDetected = allShots.length;
    summary.shotsCaptioned = captionedShots.filter((s) => String(s.caption || '').trim()).length;
    // Recompute the degraded flag with the true detected/captioned
    // counts (assignClips only saw the sampled subset).
    const rebuilt = buildMatchSummary(summary);
    summary.degraded_reason = rebuilt.degraded_reason;
    const newScenes = scenes.map((scene, i) => {
      const a = assignments[i];
      if (!a || a.method === 'none') return scene;
      return {
        ...scene,
        sourceStartSec: a.startSec,
        sourceEndSec: a.endSec,
        startSec: a.startSec,
        endSec: a.endSec,
        shotIndex: a.shotIndex,
        matchScore: a.score,
        matchMethod: a.method,
      };
    });
    console.log(`${label}shot match: ${summary.matchedSemantic} semantic, ${summary.matchedHeuristic} heuristic, ${summary.matchedNone} unchanged, avg score ${summary.avgScore}${summary.degraded_reason ? `, DEGRADED: ${summary.degraded_reason}` : ''}`);
    return { scenes: newScenes, summary, assignments };
  } catch (err) {
    console.log(`${label}shot match failed (${String(err.message).slice(0, 160)}) — keeping LLM windows`);
    return null;
  }
}

module.exports = {
  SHOT_MATCH_ENABLED,
  SHOT_MATCH_MAX_SHOTS,
  SHOT_DETECT_THRESHOLD,
  LOW_SCORE,
  REUSE_PENALTY,
  REUSE_WINDOW,
  REUSE_MAX_IN_WINDOW,
  SPEED_MIN,
  SPEED_MAX,
  clampSpeed,
  ffprobeDuration,
  parseShowinfoTimestamps,
  detectShots,
  sampleShots,
  tokenize,
  textOverlap,
  durationFit,
  anchorScore,
  scoreShot,
  heuristicShot,
  assignClips,
  buildMatchSummary,
  buildBeats,
  matchScenesToShots,
};
