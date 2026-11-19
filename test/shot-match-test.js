'use strict';
/**
 * Shot-match test — NO Gemini calls, NO API key needed.
 *
 * What it proves (movie-narrator step 9 structure, our own code):
 *  1. FFmpeg scene detection finds REAL cut points: a synthetic
 *     3-colour video (red 0-3s / green 3-6s / blue 6-9s) yields
 *     boundaries near 3s and 6s; a static video yields one shot.
 *  2. showinfo parsing is robust: config lines ignored, timestamps
 *     sorted and de-duplicated.
 *  3. Shot sampling caps the count and keeps the first/last shots.
 *  4. Semantic text overlap: shared content words score high,
 *     unrelated text scores ~0, plurals normalise, stopwords ignored.
 *  5. Greedy assignment: the semantically best shot wins over a nearer
 *     but irrelevant shot; the story anchor breaks ties.
 *  6. Reuse penalty + diversity: a shot used in the last 3 beats costs
 *     -0.15, and a shot used 2x in the last 3 beats is excluded.
 *  7. Speed clamp: every assignment's speed is inside [0.85, 1.25],
 *     and beats whose raw ratio fell outside are counted.
 *  8. Low score falls back to the heuristic (nearest shot to the
 *     anchor) for that beat alone; no captions at all means all
 *     heuristic; no shots at all keeps the original window.
 *  9. match_summary degraded_reason: too_few_captions (<30% captioned
 *     or <10), all_heuristic (nothing matched semantically),
 *     low_confidence (mean score < 0.25), null when healthy.
 * 10. Beat building prefers the English beat description over the
 *     Burmese narration (token overlap needs English) and anchors on
 *     the window midpoint.
 * 11. Shot captioning (stubbed model): 80 shots cost <= 2 model calls
 *     (40 per call), captions attach by id, a missing caption is ''
 *     (not a crash), and a throwing model returns status 'failed'
 *     with the original shots — captioning never throws.
 * 12. matchScenesToShots (stubbed detect + caption): scenes get
 *     refined windows + summary; a throwing caption step returns null
 *     (pipeline keeps LLM windows); disabled returns null.
 * 13. Real video: Tears of Steel yields >50 shot boundaries at the
 *     default threshold (skipped if the file is absent).
 *
 * Run: node test/shot-match-test.js
 */
const fs = require('fs');
const path = require('path');
const { execFile } = require('child_process');
const { promisify } = require('util');

const execFileAsync = promisify(execFile);

const {
  parseShowinfoTimestamps,
  detectShots,
  sampleShots,
  tokenize,
  textOverlap,
  scoreShot,
  assignClips,
  buildMatchSummary,
  buildBeats,
  matchScenesToShots,
  SPEED_MIN,
  SPEED_MAX,
  LOW_SCORE,
} = require('../src/shots');
const { captionShotFrames } = require('../src/gemini');

const TEST_DIR = path.join(__dirname, 'output-shotmatch');
const SOURCE_PATH = path.join(TEST_DIR, 'colours.mp4');
const STATIC_PATH = path.join(__dirname, 'test-source.mp4');
const TEARS_PATH = path.join(__dirname, 'tears-of-steel-800p.ogv');

let passed = 0;
let failed = 0;
function check(name, ok, detail) {
  if (ok) {
    passed++;
    console.log(`  PASS: ${name}`);
  } else {
    failed++;
    console.log(`  FAIL: ${name} — ${detail !== undefined ? detail : ''}`);
  }
}

async function run(cmd, args) {
  await execFileAsync(cmd, args);
}

function makeShots(n, captions) {
  // n shots of 10s each, back to back; captions[i] or ''.
  return Array.from({ length: n }, (_, i) => ({
    index: i,
    startSec: i * 10,
    endSec: (i + 1) * 10,
    midSec: i * 10 + 5,
    durationSec: 10,
    caption: captions ? captions[i] || '' : '',
  }));
}

async function main() {
  fs.mkdirSync(TEST_DIR, { recursive: true });

  console.log('--- 1. showinfo parsing ---');
  const sample = [
    '[Parsed_showinfo_1 @ 0x1] config in time_base: 1/15360, frame_rate: 30/1',
    '[Parsed_showinfo_1 @ 0x1] n:   0 pts:    214 pts_time:8.916667 duration:      1 duration_time:0.0416667 fmt:yuv420p',
    '[Parsed_showinfo_1 @ 0x1] n:   1 pts:    323 pts_time:13.458333 duration:      1 duration_time:0.0416667 fmt:yuv420p',
    '[Parsed_showinfo_1 @ 0x1] n:   2 pts:    323 pts_time:13.458333 duration:      1 duration_time:0.0416667 fmt:yuv420p',
    '[Parsed_showinfo_1 @ 0x1] n:   3 pts:    600 pts_time:25      duration:      1 duration_time:0.0416667 fmt:yuv420p',
  ].join('\n');
  const parsed = parseShowinfoTimestamps(sample);
  check('parses pts_time values, sorted, de-duplicated', JSON.stringify(parsed) === JSON.stringify([8.916667, 13.458333, 25]), JSON.stringify(parsed));
  check('empty stderr -> no boundaries', parseShowinfoTimestamps('').length === 0);
  check('config lines without pts_time ignored', parseShowinfoTimestamps('[Parsed_showinfo_1] config in time_base: 1/15360').length === 0);

  console.log('--- 2. real FFmpeg detection on a synthetic 3-colour video ---');
  await run('ffmpeg', ['-y',
    '-f', 'lavfi', '-i', 'color=c=0xFF0000:s=320x180:d=3',
    '-f', 'lavfi', '-i', 'color=c=0x00FF00:s=320x180:d=3',
    '-f', 'lavfi', '-i', 'color=c=0x0000FF:s=320x180:d=3',
    '-filter_complex', '[0:v][1:v][2:v]concat=n=3:v=1:a=0[v]',
    '-map', '[v]', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-r', '30', SOURCE_PATH]);
  const shots = await detectShots(SOURCE_PATH);
  check('3 colour blocks -> 3 shots', shots.length === 3, JSON.stringify(shots.map((s) => [s.startSec, s.endSec])));
  check('first shot starts at 0', shots.length > 0 && shots[0].startSec === 0, JSON.stringify(shots[0]));
  check('boundary near 3s (red->green cut)', shots.length >= 2 && Math.abs(shots[1].startSec - 3) < 0.6, JSON.stringify(shots.map((s) => s.startSec)));
  check('boundary near 6s (green->blue cut)', shots.length >= 3 && Math.abs(shots[2].startSec - 6) < 0.6, JSON.stringify(shots.map((s) => s.startSec)));
  check('last shot ends at the video duration (~9s)', shots.length > 0 && Math.abs(shots[shots.length - 1].endSec - 9) < 0.5, JSON.stringify(shots[shots.length - 1]));
  check('shots are contiguous (end == next start)', shots.every((s, i) => i === 0 || Math.abs(s.startSec - shots[i - 1].endSec) < 0.01), JSON.stringify(shots.map((s) => [s.startSec, s.endSec])));
  check('midSec is the midpoint', shots.every((s) => Math.abs(s.midSec - (s.startSec + s.endSec) / 2) < 0.001));

  console.log('--- 3. static video -> one shot ---');
  const staticShots = await detectShots(STATIC_PATH);
  check('static 10s video -> exactly 1 shot', staticShots.length === 1, JSON.stringify(staticShots));
  check('single shot covers the whole video', staticShots.length === 1 && staticShots[0].startSec === 0 && Math.abs(staticShots[0].endSec - 10) < 0.5, JSON.stringify(staticShots[0]));

  console.log('--- 4. shot sampling ---');
  const many = makeShots(10, null);
  const sampled = sampleShots(many, 4);
  check('sampling caps the count', sampled.length === 4, JSON.stringify(sampled.map((s) => s.index)));
  check('sampling keeps the first shot', sampled[0].index === 0, JSON.stringify(sampled.map((s) => s.index)));
  check('sampling keeps the last shot', sampled[sampled.length - 1].index === 9, JSON.stringify(sampled.map((s) => s.index)));
  check('no sampling when under the cap', sampleShots(many, 20).length === 10);
  check('sampling is evenly spread', JSON.stringify(sampled.map((s) => s.index)) === JSON.stringify([0, 3, 6, 9]), JSON.stringify(sampled.map((s) => s.index)));

  console.log('--- 5. semantic text overlap ---');
  check('shared content words score high', textOverlap('a sniper aims at the airship over rooftops', 'a sniper aiming a rifle on a rooftop') > 0.4, String(textOverlap('a sniper aims at the airship over rooftops', 'a sniper aiming a rifle on a rooftop')));
  check('unrelated text scores ~0', textOverlap('a sniper aims at the airship over rooftops', 'a bowl of noodles on a kitchen table') < 0.1, String(textOverlap('a sniper aims at the airship over rooftops', 'a bowl of noodles on a kitchen table')));
  check('plurals normalise (rooftops == rooftop)', textOverlap('men walk on rooftops', 'a man walks on a rooftop') > 0.4, String(textOverlap('men walk on rooftops', 'a man walks on a rooftop')));
  check('stopwords do not count as a match', textOverlap('the tall man reads a newspaper', 'the short woman reads a magazine') < 0.35, String(textOverlap('the tall man reads a newspaper', 'the short woman reads a magazine')));
  check('empty caption scores 0', textOverlap('a sniper aims', '') === 0);
  check('empty beat text scores 0', textOverlap('', 'a sniper aiming') === 0);
  check('tokenize drops stopwords', !tokenize('the man and the dog').has('the') && tokenize('the man and the dog').has('man'));

  console.log('--- 6. shot scoring + speed clamp ---');
  const beat10 = { text: 'a sniper aims at the airship', anchorSec: 50, durationSec: 10, startSec: 45, endSec: 55 };
  const good = { index: 0, startSec: 45, endSec: 55, midSec: 50, durationSec: 10, caption: 'a sniper aims a rifle on a rooftop' };
  const bad = { index: 1, startSec: 45, endSec: 55, midSec: 50, durationSec: 10, caption: 'a bowl of noodles on a kitchen table' };
  const sGood = scoreShot(beat10, good, 100);
  const sBad = scoreShot(beat10, bad, 100);
  check('matching caption outscores an unrelated one at the same anchor', sGood.score > sBad.score + 0.2, `${sGood.score} vs ${sBad.score}`);
  check('speed is 1.0 when shot and narration match in length', Math.abs(sGood.speed - 1) < 0.01, String(sGood.speed));
  const tiny = { ...good, durationSec: 1, startSec: 45, endSec: 46, midSec: 45.5 };
  const sTiny = scoreShot(beat10, tiny, 100);
  check('speed clamps to the floor (0.85) for a too-short shot', Math.abs(sTiny.speed - SPEED_MIN) < 0.001 && sTiny.speedClamped === true, String(sTiny.speed));
  const huge = { ...good, durationSec: 60, startSec: 45, endSec: 105, midSec: 75 };
  const sHuge = scoreShot(beat10, huge, 100);
  check('speed clamps to the ceiling (1.25) for a too-long shot', Math.abs(sHuge.speed - SPEED_MAX) < 0.001 && sHuge.speedClamped === true, String(sHuge.speed));

  console.log('--- 7. greedy assignment ---');
  const shots7 = makeShots(5, [
    'a man examines a glowing brain on a plate in a laboratory',
    'a sniper aiming a rifle on a rooftop at night',
    'a young man and a girl walking on a bridge',
    'an explosion in a city street with smoke',
    'a robot hand touching a boy face in a laboratory',
  ]);
  const beats7 = [
    { text: 'a man examines a glowing brain on a plate', anchorSec: 5, durationSec: 10, startSec: 0, endSec: 10 },
    { text: 'a sniper aims at the flying airship over rooftops', anchorSec: 15, durationSec: 10, startSec: 10, endSec: 20 },
    { text: 'the young man and the girl walk on the bridge', anchorSec: 25, durationSec: 10, startSec: 20, endSec: 30 },
  ];
  const res7 = assignClips(beats7, shots7, 50);
  check('semantic match wins: brain beat -> brain shot', res7.assignments[0].shotIndex === 0 && res7.assignments[0].method === 'semantic', JSON.stringify(res7.assignments[0]));
  check('semantic match wins: sniper beat -> sniper shot', res7.assignments[1].shotIndex === 1, JSON.stringify(res7.assignments[1]));
  check('semantic match wins: bridge beat -> bridge shot', res7.assignments[2].shotIndex === 2, JSON.stringify(res7.assignments[2]));
  check('all three beats matched semantically', res7.summary.matchedSemantic === 3, JSON.stringify(res7.summary));
  check('healthy summary has no degraded reason', res7.summary.degraded_reason === null, JSON.stringify(res7.summary));
  check('all speeds inside the clamp', res7.assignments.every((a) => a.speed >= SPEED_MIN && a.speed <= SPEED_MAX), JSON.stringify(res7.assignments.map((a) => a.speed)));

  console.log('--- 7b. anchor breaks ties when text is equal ---');
  const shots7b = makeShots(3, ['a man walks in a street', 'a man walks in a street', 'a man walks in a street']);
  const beats7b = [{ text: 'a man walks in a street', anchorSec: 25, durationSec: 10, startSec: 20, endSec: 30 }];
  const res7b = assignClips(beats7b, shots7b, 30);
  check('identical captions: nearest shot to the anchor wins', res7b.assignments[0].shotIndex === 2, JSON.stringify(res7b.assignments[0]));

  console.log('--- 7c. reuse penalty + diversity hard cap ---');
  // Four identical beats, all anchored at 5s (the same story position),
  // three identical dog shots. Traced by hand:
  //  beat 0 -> shot 0 (anchor 1.0, no penalty)
  //  beat 1 -> shot 1 (shot 0 penalised -0.15, shot 1's anchor wins)
  //  beat 2 -> shot 0 again (penalised but still best; used only 1x in
  //    the window, so the hard cap does not apply yet)
  //  beat 3 -> shot 2 (shot 0 is EXCLUDED — used 2x in the last 3 beats;
  //    shot 2 beats the penalised shot 1)
  const shots7c = makeShots(3, ['a dog runs in a park', 'a dog runs in a park', 'a dog runs in a park']);
  const beats7c = [0, 1, 2, 3].map((i) => ({
    text: 'a dog runs in a park', anchorSec: 5, durationSec: 10, startSec: i * 10, endSec: (i + 1) * 10,
  }));
  const res7c = assignClips(beats7c, shots7c, 30);
  check('reuse penalty steers beat 1 to the second dog shot', res7c.assignments[0].shotIndex === 0 && res7c.assignments[1].shotIndex === 1, JSON.stringify(res7c.assignments.map((a) => a.shotIndex)));
  check('beat 2 reuses shot 0 once (penalised, still best, under the cap)', res7c.assignments[2].shotIndex === 0 && res7c.assignments[2].reused === true, JSON.stringify(res7c.assignments.map((a) => [a.shotIndex, a.reused])));
  check('beat 3 cannot reuse shot 0 a 2nd time in the window (diversity cap) — takes shot 2', res7c.assignments[3].shotIndex === 2, JSON.stringify(res7c.assignments.map((a) => a.shotIndex)));
  check('reuse is recorded in the summary', res7c.summary.reusedCount === 1, JSON.stringify(res7c.summary));

  console.log('--- 7d. low score -> heuristic fallback, no captions -> all heuristic, no shots -> unchanged ---');
  const beats7d = [{ text: 'a submarine dives under Arctic ice', anchorSec: 45, durationSec: 10, startSec: 40, endSec: 50 }];
  const res7d = assignClips(beats7d, shots7, 50);
  check('unmatchable beat falls back to heuristic (nearest shot to anchor)', res7d.assignments[0].method === 'heuristic' && res7d.assignments[0].shotIndex === 4, JSON.stringify(res7d.assignments[0]));
  check('heuristic beat keeps the nearest shot window', res7d.assignments[0].startSec === 40 && res7d.assignments[0].endSec === 50, JSON.stringify(res7d.assignments[0]));
  const noCap = makeShots(3, ['', '', '']);
  const res7e = assignClips(beats7, noCap, 30);
  check('no captions at all -> every beat heuristic', res7e.summary.matchedSemantic === 0 && res7e.summary.matchedHeuristic === 3, JSON.stringify(res7e.summary));
  const res7f = assignClips(beats7, [], 50);
  check('no shots at all -> original windows kept, method none', res7f.assignments.every((a) => a.method === 'none') && res7f.assignments[0].startSec === 0 && res7f.assignments[0].endSec === 10, JSON.stringify(res7f.assignments));

  console.log('--- 8. degraded_reason flagging ---');
  const d1 = buildMatchSummary({ shotsDetected: 100, shotsCaptioned: 20, beats: 10, matchedSemantic: 8, matchedHeuristic: 2, matchedNone: 0, avgScore: 0.6, clampedCount: 0, reusedCount: 0 });
  check('captioned <30% -> too_few_captions', d1.degraded_reason === 'too_few_captions', JSON.stringify(d1));
  const d2 = buildMatchSummary({ shotsDetected: 100, shotsCaptioned: 100, beats: 10, matchedSemantic: 0, matchedHeuristic: 10, matchedNone: 0, avgScore: 0, clampedCount: 0, reusedCount: 0 });
  check('nothing semantic -> all_heuristic', d2.degraded_reason === 'all_heuristic', JSON.stringify(d2));
  const d3 = buildMatchSummary({ shotsDetected: 100, shotsCaptioned: 100, beats: 10, matchedSemantic: 5, matchedHeuristic: 5, matchedNone: 0, avgScore: 0.2, clampedCount: 0, reusedCount: 0 });
  check('mean score < 0.25 -> low_confidence', d3.degraded_reason === 'low_confidence', JSON.stringify(d3));
  const d4 = buildMatchSummary({ shotsDetected: 100, shotsCaptioned: 90, beats: 10, matchedSemantic: 9, matchedHeuristic: 1, matchedNone: 0, avgScore: 0.6, clampedCount: 1, reusedCount: 0 });
  check('healthy match -> no degraded reason', d4.degraded_reason === null, JSON.stringify(d4));
  const d5 = buildMatchSummary({ shotsDetected: 0, shotsCaptioned: 0, beats: 0, matchedSemantic: 0, matchedHeuristic: 0, matchedNone: 0, avgScore: 0, clampedCount: 0, reusedCount: 0 });
  check('no data at all -> no degraded reason (nothing to judge)', d5.degraded_reason === null, JSON.stringify(d5));

  console.log('--- 9. beat building prefers the English description ---');
  const scenes9 = [
    { sourceStartSec: 0, sourceEndSec: 12, narrationMy: 'လူငယ်လေးနဲ့ မိန်းကလေး တံတားပေါ်မှာ လမ်းလျှောက်နေတယ်', audioDurationSec: 8 },
    { sourceStartSec: 12, sourceEndSec: 24, narrationMy: 'စက်ရုပ် ပေါက်ကွဲသွားတယ်', audioDurationSec: 9 },
  ];
  const summaries9 = [
    { startSec: 0, endSec: 12, description: 'a young man and a girl walk on a bridge' },
    { startSec: 12, endSec: 24, description: 'an explosion in a city street' },
  ];
  const beats9 = buildBeats(scenes9, summaries9);
  check('beat text is the English description, not the Burmese narration', beats9[0].text === 'a young man and a girl walk on a bridge', beats9[0].text);
  check('anchor is the window midpoint', beats9[0].anchorSec === 6 && beats9[1].anchorSec === 18, JSON.stringify(beats9.map((b) => b.anchorSec)));
  check('duration comes from the audio length', beats9[0].durationSec === 8 && beats9[1].durationSec === 9, JSON.stringify(beats9.map((b) => b.durationSec)));
  const beats9b = buildBeats(scenes9, null);
  check('no summaries -> falls back to the narration text', beats9b[0].text.includes('လူငယ်လေး'), beats9b[0].text);

  console.log('--- 10. shot captioning (stubbed model, no Gemini) ---');
  const stubFrame = async (_video, _atSec, outPath) => {
    fs.mkdirSync(path.dirname(outPath), { recursive: true });
    fs.writeFileSync(outPath, Buffer.from([0xff, 0xd8, 0xff, 0xd9]));
    return outPath;
  };
  let modelCalls = 0;
  const mockCaption = async (group) => {
    modelCalls++;
    // Real Gemini shape per the caption prompt: {id, caption} — NOT
    // {index, caption}. A stub returning `index` masked the 2026-11-19
    // live bug where every real caption was discarded (0/107).
    return group.map((s) => ({ id: s.index, caption: `shot ${s.index} shows a test frame` }));
  };
  const eighty = makeShots(80, null);
  const cap10 = await captionShotFrames(eighty, SOURCE_PATH, path.join(TEST_DIR, 'c10'), {
    modelFn: mockCaption, frameProvider: stubFrame,
  });
  check('80 shots cost <= 2 model calls (40 per call)', modelCalls === 2, `calls=${modelCalls}`);
  check('all 80 shots captioned', cap10.stats.captioned === 80, JSON.stringify(cap10.stats));
  check('captions attach to the right shot by id', cap10.shots[37].caption === 'shot 37 shows a test frame', cap10.shots[37].caption);
  check('captioning status is captioned', cap10.stats.status === 'captioned', JSON.stringify(cap10.stats));
  // Regression (live bug 2026-11-19): a model returning captions with NO
  // usable id at all must report status failed, not "captioned" with 0.
  const emptyMock = async (group) => group.map(() => ({ caption: 'no id here' }));
  const cap10e = await captionShotFrames(makeShots(3, null), SOURCE_PATH, path.join(TEST_DIR, 'c10e'), {
    modelFn: emptyMock, frameProvider: stubFrame,
  });
  check('captions with no id attach to nothing and status is failed', cap10e.stats.captioned === 0 && cap10e.stats.status === 'failed', JSON.stringify(cap10e.stats));
  const partialMock = async (group) => group.slice(0, 1).map((s) => ({ id: s.index, caption: 'only the first' }));
  const cap10b = await captionShotFrames(makeShots(3, null), SOURCE_PATH, path.join(TEST_DIR, 'c10b'), {
    modelFn: partialMock, frameProvider: stubFrame,
  });
  check('a shot with no caption gets empty caption, not a crash', cap10b.shots[1].caption === '' && cap10b.shots[2].caption === '', JSON.stringify(cap10b.shots.map((s) => s.caption)));
  check('partial captioning still counts what it got', cap10b.stats.captioned === 1, JSON.stringify(cap10b.stats));
  const throwingMock = async () => { throw new Error('503 model overloaded'); };
  const cap10c = await captionShotFrames(makeShots(3, null), SOURCE_PATH, path.join(TEST_DIR, 'c10c'), {
    modelFn: throwingMock, frameProvider: stubFrame,
  });
  check('throwing model -> status failed, original shots returned, no throw', cap10c.stats.status === 'failed' && cap10c.shots.length === 3 && cap10c.shots.every((s) => s.caption === ''), JSON.stringify(cap10c.stats));
  const cap10d = await captionShotFrames(makeShots(3, null), SOURCE_PATH, path.join(TEST_DIR, 'c10d'), {
    modelFn: mockCaption, frameProvider: stubFrame, enabled: false,
  });
  check('disabled -> status skipped, zero model calls', cap10d.stats.status === 'skipped' && cap10d.stats.calls === 0, JSON.stringify(cap10d.stats));

  console.log('--- 11. matchScenesToShots orchestration (stubbed) ---');
  const fakeDetect = async () => makeShots(5, null);
  const fakeCaption = async (shotList) => ({
    shots: shotList.map((s, i) => ({
      ...s,
      caption: ['a man examines a glowing brain on a plate', 'a sniper aiming a rifle on a rooftop', 'a young man and a girl walking on a bridge', 'an explosion in a city street', 'a robot hand touching a face'][i] || '',
    })),
    stats: { shots: 5, captioned: 5, calls: 1, status: 'captioned' },
  });
  const scenes11 = [
    { sceneNumber: 1, sourceStartSec: 100, sourceEndSec: 110, narrationMy: 'ဦးနှောက်ကို စစ်နေတယ်', audioDurationSec: 10 },
    { sceneNumber: 2, sourceStartSec: 200, sourceEndSec: 210, narrationMy: 'တံတားပေါ်မှာ လမ်းလျှောက်တယ်', audioDurationSec: 10 },
  ];
  const summaries11 = [
    { startSec: 100, endSec: 110, description: 'a man examines a glowing brain on a plate' },
    { startSec: 200, endSec: 210, description: 'a young man and a girl walk on a bridge' },
  ];
  const res11 = await matchScenesToShots({
    sourceVideo: SOURCE_PATH, scenes: scenes11, sceneSummaries: summaries11,
    workDir: path.join(TEST_DIR, 'm11'), jobId: 'test',
    opts: { detectFn: fakeDetect, captionFn: fakeCaption, videoDurationSec: 300 },
  });
  check('orchestrator returns refined scenes + summary', !!res11 && Array.isArray(res11.scenes) && !!res11.summary, JSON.stringify(res11 && res11.summary));
  check('brain scene re-windowed to the brain shot (0-10s)', res11 && res11.scenes[0].sourceStartSec === 0 && res11.scenes[0].sourceEndSec === 10, JSON.stringify(res11 && res11.scenes[0]));
  check('bridge scene re-windowed to the bridge shot (20-30s)', res11 && res11.scenes[1].sourceStartSec === 20 && res11.scenes[1].sourceEndSec === 30, JSON.stringify(res11 && res11.scenes[1]));
  check('matched scenes carry shotIndex + matchScore + matchMethod', res11 && res11.scenes[0].shotIndex === 0 && typeof res11.scenes[0].matchScore === 'number' && res11.scenes[0].matchMethod === 'semantic', JSON.stringify(res11 && res11.scenes[0]));
  check('summary records the caption call count', res11 && res11.summary.captionCalls === 1, JSON.stringify(res11 && res11.summary));
  check('summary records detected vs captioned shots', res11 && res11.summary.shotsDetected === 5 && res11.summary.shotsCaptioned === 5, JSON.stringify(res11 && res11.summary));
  const res11b = await matchScenesToShots({
    sourceVideo: SOURCE_PATH, scenes: scenes11, sceneSummaries: summaries11,
    workDir: path.join(TEST_DIR, 'm11b'), jobId: 'test',
    opts: { detectFn: fakeDetect, captionFn: async () => { throw new Error('caption boom'); }, videoDurationSec: 300 },
  });
  check('throwing caption step -> null (pipeline keeps LLM windows)', res11b === null);
  const res11c = await matchScenesToShots({
    sourceVideo: SOURCE_PATH, scenes: scenes11, sceneSummaries: summaries11,
    workDir: path.join(TEST_DIR, 'm11c'), jobId: 'test',
    opts: { detectFn: fakeDetect, captionFn: fakeCaption, videoDurationSec: 300, enabled: false },
  });
  check('disabled -> null', res11c === null);
  const res11d = await matchScenesToShots({
    sourceVideo: SOURCE_PATH, scenes: [], sceneSummaries: summaries11,
    workDir: path.join(TEST_DIR, 'm11d'), jobId: 'test',
    opts: { detectFn: fakeDetect, captionFn: fakeCaption, videoDurationSec: 300 },
  });
  check('no scenes -> null', res11d === null);

  console.log('--- 12. real video: Tears of Steel shot boundaries ---');
  if (fs.existsSync(TEARS_PATH)) {
    const tears = await detectShots(TEARS_PATH);
    check('Tears of Steel yields >50 shot boundaries at 0.4', tears.length > 50, `shots=${tears.length}`);
    check('first Tears shot starts at 0', tears.length > 0 && tears[0].startSec === 0, JSON.stringify(tears[0]));
    check('last Tears shot ends near the duration (~734s)', tears.length > 0 && Math.abs(tears[tears.length - 1].endSec - 734) < 2, JSON.stringify(tears[tears.length - 1]));
  } else {
    console.log('  SKIP: tears-of-steel-800p.ogv not present');
  }

  if (failed > 0) throw new Error(`${failed} assertion(s) failed, ${passed} passed.`);
  console.log(`SHOT-MATCH TEST PASSED: ${passed} assertions, 0 failed.`);
}

main().catch((err) => {
  console.error('SHOT-MATCH TEST FAILED:', err.message);
  process.exit(1);
});
