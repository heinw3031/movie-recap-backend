'use strict';
/**
 * Moment-verification test — NO Gemini calls, NO API key needed.
 *
 * What it proves (the analysis model's moments are sometimes mis-grounded
 * — job 6b70028f: @98s "holographic cables pulse" over a sniper-scope
 * robot frame; @432s "Celia releases Thom" over Barley aiming a rifle):
 *  1. A moment the model confirms stays (same timestamp, same claim).
 *  2. A moment the model rejects is DROPPED from its beat — and the
 *     sentence that described it is dropped from the narration. The
 *     script is written AFTER verification, from the verified beats, so
 *     the dropped event is never narrated; buildSegmentsFromMoments only
 *     ever sees verified moments.
 *  3. A moment the model corrects (event visible nearby) MOVES to the
 *     corrected timestamp and keeps its claim.
 *  4. A correction outside its own beat is treated as no correction —
 *     the moment is dropped, not moved across beats.
 *  5. A moment with no verdict is KEPT (unverified) — an incomplete
 *     verification never guts the script.
 *  6. Verification failure (mock throws) falls back to the unverified
 *     moments without failing.
 *  7. VERIFY_MOMENTS off (opts.enabled=false) makes zero model calls and
 *     leaves the beats unchanged.
 *  8. Batching: 40 beats x 2 moments = 80 moments costs <= 2 model calls
 *     (40 moments per call, aligned to the 40-beat script chunks), and a
 *     beat's moments are never split across calls.
 *  9. The real frame extraction looks at the frame AT the moment's
 *     timestamp (proven on real pixels: a synthetic colour video, red
 *     0-10s / green 10-20s / blue 20-30s — the frame at 2s is red, at
 *     12s green, at 22s blue).
 * 10. A beat that loses ALL its moments falls back to scene-level
 *     (buildSegmentsFromMoments returns null) and gets driftRisk flagged.
 *
 * Run: node test/moment-verify-test.js
 */
process.env.MOMENT_CUT = '1';

const fs = require('fs');
const path = require('path');
const { execFile } = require('child_process');
const { promisify } = require('util');

const execFileAsync = promisify(execFile);
const {
  verifyMoments,
  applyVerdicts,
  splitSentences,
  buildSegmentsFromMoments,
  computeDriftRisk,
  ffprobeDuration,
} = require('../src/gemini');

const TEST_DIR = path.join(__dirname, 'output-verify');
const FIXTURE_JPG = path.join(TEST_DIR, 'fixture.jpg');
const SOURCE_PATH = path.join(TEST_DIR, 'colour-source.mp4');
const FAKE_VIDEO = '/nonexistent/video.mp4'; // never read: the stub frame provider ignores it

let passed = 0;
let failed = 0;
function check(name, cond, extra) {
  if (cond) {
    passed++;
    console.log(`  PASS: ${name}`);
  } else {
    failed++;
    console.log(`  FAIL: ${name}${extra ? ' — ' + extra : ''}`);
  }
}

async function run(cmd, args) {
  await execFileAsync(cmd, args);
}

/** Dominant colour of a JPEG file (raw RGB mean). */
async function jpgColour(jpgPath) {
  const { stdout } = await execFileAsync('ffmpeg', [
    '-i', jpgPath, '-vframes', '1', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-',
  ], { encoding: 'buffer', maxBuffer: 64 * 1024 * 1024 });
  const buf = stdout;
  let r = 0;
  let g = 0;
  let b = 0;
  let n = 0;
  for (let i = 0; i + 2 < buf.length; i += 3 * 997) {
    r += buf[i];
    g += buf[i + 1];
    b += buf[i + 2];
    n++;
  }
  r /= n;
  g /= n;
  b /= n;
  return r > g && r > b ? 'red' : g > b ? 'green' : 'blue';
}

// Frame stub: copies a pre-made fixture JPEG to the requested path
// (the mock model decides by claim text, so frame content is irrelevant
// here — the real extraction path is tested separately in section 9).
function stubFrameProvider(fixture) {
  return async (_videoPath, _atSec, outPath) => {
    fs.mkdirSync(path.dirname(outPath), { recursive: true });
    fs.copyFileSync(fixture, outPath);
    return outPath;
  };
}

// Mock model: decide by the claim text.
//  - claim contains '[WRONG]' -> reject, no correction (drop).
//  - claim contains '[MOVE:NN]' -> reject, correctedAtSec NN.
//  - claim contains '[FAR:NN]' -> reject, correctedAtSec NN (outside the beat).
//  - otherwise -> confirm.
function mockByClaim(group) {
  return group.map((m) => {
    const move = m.what.match(/\[MOVE:(\d+)\]/);
    if (move) return { id: m.id, confirmed: false, actual: 'the claimed event, a few seconds later', correctedAtSec: Number(move[1]) };
    const far = m.what.match(/\[FAR:(\d+)\]/);
    if (far) return { id: m.id, confirmed: false, actual: 'something else entirely', correctedAtSec: Number(far[1]) };
    if (m.what.includes('[WRONG]')) return { id: m.id, confirmed: false, actual: 'an empty street', correctedAtSec: null };
    return { id: m.id, confirmed: true };
  });
}

async function main() {
  fs.mkdirSync(TEST_DIR, { recursive: true });
  // One fixture JPEG for the stub frame provider.
  await run('ffmpeg', ['-y', '-f', 'lavfi', '-i', 'color=c=0xFF0000:s=320x180:d=0.1', '-vframes', '1', '-q:v', '3', FIXTURE_JPG]);
  const stub = stubFrameProvider(FIXTURE_JPG);

  console.log('--- 1. confirmed moment stays ---');
  const beats1 = [
    { startSec: 30, endSec: 42, description: 'bridge', moments: [
      { atSec: 31, what: 'couple talks on the bridge' },
      { atSec: 37, what: 'girl turns to leave' },
    ] },
  ];
  const res1 = await verifyMoments(beats1, FAKE_VIDEO, path.join(TEST_DIR, 'v1'), {
    modelFn: mockByClaim, frameProvider: stub,
  });
  check('confirmed moments stay (same atSec and what)', res1.beats[0].moments.length === 2 && res1.beats[0].moments[0].atSec === 31 && res1.beats[0].moments[1].atSec === 37, JSON.stringify(res1.beats[0].moments));
  check('stats: 2 verified, 0 dropped', res1.stats.verified === 2 && res1.stats.dropped === 0, JSON.stringify(res1.stats));
  check('1 model call for 2 moments', res1.stats.calls === 1, `calls=${res1.stats.calls}`);
  check('status verified', res1.stats.status === 'verified');
  check('input beats are not mutated', beats1[0].moments.length === 2 && beats1[0].moments[0].atSec === 31);
  check('verification frames are cleaned up', !fs.existsSync(path.join(TEST_DIR, 'v1')));

  console.log('--- 2. rejected moment is dropped, and its sentence is dropped ---');
  const beats2 = [
    { startSec: 30, endSec: 45, description: 'lab', moments: [
      { atSec: 31, what: 'doctor inspects the brain on the table' },
      { atSec: 36, what: 'the brain pulses with light [WRONG]' },
      { atSec: 41, what: 'robot eye lights up' },
    ] },
  ];
  const res2 = await verifyMoments(beats2, FAKE_VIDEO, path.join(TEST_DIR, 'v2'), {
    modelFn: mockByClaim, frameProvider: stub,
  });
  check('rejected moment is dropped from the beat', res2.beats[0].moments.length === 2 && res2.beats[0].moments.every((m) => !m.what.includes('[WRONG]')), JSON.stringify(res2.beats[0].moments));
  check('stats: 1 dropped, 2 verified', res2.stats.dropped === 1 && res2.stats.verified === 2, JSON.stringify(res2.stats));
  // The script is written AFTER verification, from the verified beats, so
  // it writes sentences only for the moments that survived. The dropped
  // event's sentence ("the brain pulses") is never written — the narration
  // is the two surviving sentences, each over its own verified moment.
  const droppedSentence = 'ဦးနှောက်က လင်းထိန်နေပါတယ်။';
  const scriptSentences = 'ဆရာဝန်ကြီးက စားပွဲပေါ်က ဦးနှောက်ကို စစ်ဆေးနေပါတယ်။ စက်ရုပ်မျက်လုံးက လင်းထိန်နေပါတယ်။';
  const seg2 = buildSegmentsFromMoments(splitSentences(scriptSentences), res2.beats[0], 30, 45);
  check('verified beat builds 2 segments (one per verified moment)', Array.isArray(seg2) && seg2.length === 2, JSON.stringify(seg2));
  check('segment 1 starts at the brain moment (31s)', seg2 && seg2[0].sourceStartSec === 31, JSON.stringify(seg2 && seg2[0]));
  check('segment 2 starts at the robot-eye moment (41s)', seg2 && seg2[1].sourceStartSec === 41, JSON.stringify(seg2 && seg2[1]));
  check('the dropped sentence is not in the narration', seg2 && !seg2.map((g) => g.text).join(' ').includes(droppedSentence), seg2 ? seg2.map((g) => g.text).join(' ') : '');

  console.log('--- 3. corrected moment moves to the right timestamp ---');
  const beats3 = [
    { startSec: 90, endSec: 105, description: 'cables', moments: [
      { atSec: 98, what: 'holographic cables pulse with light [MOVE:103]' },
      { atSec: 101, what: 'a technician watches the cables' },
    ] },
  ];
  const res3 = await verifyMoments(beats3, FAKE_VIDEO, path.join(TEST_DIR, 'v3'), {
    modelFn: mockByClaim, frameProvider: stub,
  });
  const moved = res3.beats[0].moments.find((m) => m.what.includes('holographic'));
  check('corrected moment moves to the corrected timestamp (103s)', moved && moved.atSec === 103, JSON.stringify(res3.beats[0].moments));
  check('corrected moment keeps its claim', moved && moved.what.includes('holographic cables pulse with light'));
  check('moments stay in time order after the move (101s, then 103s)', res3.beats[0].moments[0].atSec === 101 && res3.beats[0].moments[1].atSec === 103, JSON.stringify(res3.beats[0].moments));
  check('stats: 1 corrected, 0 dropped', res3.stats.corrected === 1 && res3.stats.dropped === 0, JSON.stringify(res3.stats));

  console.log('--- 4. correction outside the beat -> dropped, not moved ---');
  const beats4 = [
    { startSec: 30, endSec: 42, description: 'bridge', moments: [
      { atSec: 35, what: 'couple walks on the bridge [FAR:200]' },
      { atSec: 39, what: 'girl looks back' },
    ] },
  ];
  const res4 = await verifyMoments(beats4, FAKE_VIDEO, path.join(TEST_DIR, 'v4'), {
    modelFn: mockByClaim, frameProvider: stub,
  });
  check('correction outside the beat is dropped, not moved', res4.beats[0].moments.length === 1 && res4.beats[0].moments[0].atSec === 39, JSON.stringify(res4.beats[0].moments));
  check('stats: 1 dropped, 0 corrected', res4.stats.dropped === 1 && res4.stats.corrected === 0, JSON.stringify(res4.stats));

  console.log('--- 5. missing verdict -> moment kept (unverified) ---');
  const partialMock = async (group) => [{ id: group[0].id, confirmed: true }]; // only the first moment gets a verdict
  const beats5 = [
    { startSec: 30, endSec: 42, description: 'bridge', moments: [
      { atSec: 31, what: 'couple talks on the bridge' },
      { atSec: 37, what: 'girl turns to leave' },
    ] },
  ];
  const res5 = await verifyMoments(beats5, FAKE_VIDEO, path.join(TEST_DIR, 'v5'), {
    modelFn: partialMock, frameProvider: stub,
  });
  check('moment with no verdict is kept (unverified)', res5.beats[0].moments.length === 2, JSON.stringify(res5.beats[0].moments));
  check('stats: 1 verified, 1 unverified', res5.stats.verified === 1 && res5.stats.unverified === 1, JSON.stringify(res5.stats));

  console.log('--- 6. verification failure falls back to unverified moments ---');
  const throwingMock = async () => { throw new Error('model exploded'); };
  const res6 = await verifyMoments(beats1, FAKE_VIDEO, path.join(TEST_DIR, 'v6'), {
    modelFn: throwingMock, frameProvider: stub,
  });
  check('a throwing mock does not throw verifyMoments', true);
  check('beats are unchanged after a failed verification', JSON.stringify(res6.beats) === JSON.stringify(beats1), JSON.stringify(res6.beats));
  check('status failed, 0 verified', res6.stats.status === 'failed' && res6.stats.verified === 0, JSON.stringify(res6.stats));
  check('unverified count equals total', res6.stats.unverified === 2, JSON.stringify(res6.stats));

  console.log('--- 7. flag off -> zero calls, beats unchanged ---');
  let calls7 = 0;
  const countingOff = async (group) => { calls7++; return group.map((m) => ({ id: m.id, confirmed: true })); };
  const res7 = await verifyMoments(beats1, FAKE_VIDEO, path.join(TEST_DIR, 'v7'), {
    enabled: false, modelFn: countingOff, frameProvider: stub,
  });
  check('flag off makes zero model calls', calls7 === 0 && res7.stats.calls === 0, `calls=${calls7}`);
  check('flag off leaves beats unchanged', JSON.stringify(res7.beats) === JSON.stringify(beats1));
  check('flag off status skipped', res7.stats.status === 'skipped');

  console.log('--- 8. batching: 40 beats x 2 moments <= 2 calls ---');
  const beats8 = Array.from({ length: 40 }, (_, i) => ({
    startSec: i * 12, endSec: i * 12 + 12, description: `beat ${i + 1}`,
    moments: [
      { atSec: i * 12 + 2, what: `moment A of beat ${i + 1}` },
      { atSec: i * 12 + 8, what: `moment B of beat ${i + 1}` },
    ],
  }));
  const counter = { calls: 0, moments: 0, beatsPerCall: [] };
  const batchMock = async (group) => {
    counter.calls++;
    counter.moments += group.length;
    counter.beatsPerCall.push(new Set(group.map((m) => m.beatIndex)));
    return group.map((m) => ({ id: m.id, confirmed: true }));
  };
  const res8 = await verifyMoments(beats8, FAKE_VIDEO, path.join(TEST_DIR, 'v8'), {
    modelFn: batchMock, frameProvider: stub,
  });
  check('40 beats x 2 moments = 80 moments verified', res8.stats.total === 80 && res8.stats.verified === 80, JSON.stringify(res8.stats));
  check('calls <= 2 per 40-beat chunk', counter.calls <= 2, `calls=${counter.calls}`);
  check('all 80 moments were sent to the model', counter.moments === 80, `moments=${counter.moments}`);
  check('no beat is split across two calls', (() => {
    const seen = new Set();
    for (const s of counter.beatsPerCall) {
      for (const b of s) { if (seen.has(b)) return false; seen.add(b); }
    }
    return true;
  })(), counter.beatsPerCall.map((s) => s.size).join('/'));

  console.log('--- 9. real frame extraction looks at the frame AT the moment ---');
  await run('ffmpeg', ['-y',
    '-f', 'lavfi', '-i', 'color=c=0xFF0000:s=640x360:d=10',
    '-f', 'lavfi', '-i', 'color=c=0x00FF00:s=640x360:d=10',
    '-f', 'lavfi', '-i', 'color=c=0x0000FF:s=640x360:d=10',
    '-filter_complex', '[0:v][1:v][2:v]concat=n=3:v=1:a=0[v]',
    '-map', '[v]', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-r', '30', SOURCE_PATH]);
  const srcDur = await ffprobeDuration(SOURCE_PATH);
  check('colour source is 30s (red 0-10, green 10-20, blue 20-30)', Math.abs(srcDur - 30) < 0.5, `got ${srcDur}`);
  const seenColours = [];
  const colourMock = async (group) => {
    for (const m of group) {
      seenColours.push({ id: m.id, atSec: m.atSec, colour: await jpgColour(m.framePath) });
    }
    return group.map((m) => ({ id: m.id, confirmed: true }));
  };
  const beats9 = [
    { startSec: 0, endSec: 10, description: 'red', moments: [{ atSec: 2, what: 'red frame' }] },
    { startSec: 10, endSec: 20, description: 'green', moments: [{ atSec: 12, what: 'green frame' }] },
    { startSec: 20, endSec: 30, description: 'blue', moments: [{ atSec: 22, what: 'blue frame' }] },
  ];
  // NO frameProvider here — the real ffmpeg extraction runs.
  const res9 = await verifyMoments(beats9, SOURCE_PATH, path.join(TEST_DIR, 'v9'), {
    modelFn: colourMock,
  });
  check('frame at 2s is red (the frame AT the moment, not a keyframe)', seenColours.find((c) => c.atSec === 2)?.colour === 'red', JSON.stringify(seenColours));
  check('frame at 12s is green', seenColours.find((c) => c.atSec === 12)?.colour === 'green', JSON.stringify(seenColours));
  check('frame at 22s is blue', seenColours.find((c) => c.atSec === 22)?.colour === 'blue', JSON.stringify(seenColours));
  check('real extraction: all 3 moments verified', res9.stats.verified === 3, JSON.stringify(res9.stats));

  console.log('--- 10. beat loses all moments -> scene-level + driftRisk ---');
  const beats10 = [
    { startSec: 30, endSec: 70, description: 'empty beat', moments: [
      { atSec: 35, what: 'nothing here [WRONG]' },
      { atSec: 55, what: 'also nothing [WRONG]' },
    ] },
  ];
  const res10 = await verifyMoments(beats10, FAKE_VIDEO, path.join(TEST_DIR, 'v10'), {
    modelFn: mockByClaim, frameProvider: stub,
  });
  check('beat that loses all moments keeps an empty moments list', res10.beats[0].moments.length === 0, JSON.stringify(res10.beats[0].moments));
  check('stats: 2 dropped', res10.stats.dropped === 2, JSON.stringify(res10.stats));
  const seg10 = buildSegmentsFromMoments(
    splitSentences('လူငယ်လေးက လမ်းလျှောက်နေပါတယ်။ မိန်းကလေးက နောက်ကနေ လိုက်လာပါတယ်။'),
    res10.beats[0], 30, 70
  );
  check('no moments -> no segments (scene-level fallback)', seg10 === null);
  check('short narration over a 40s range -> driftRisk flagged', computeDriftRisk('x'.repeat(100), 30, 70, false) === true);

  console.log('--- 11. applyVerdicts: correction just past the beat is clamped in ---');
  const av = applyVerdicts(
    [{ startSec: 30, endSec: 42, description: 'b', moments: [{ atSec: 35, what: 'event' }] }],
    [{ id: 0, beatIndex: 0, momentIndex: 0, atSec: 35, what: 'event' }],
    [{ id: 0, confirmed: false, actual: 'the event, just past the beat', correctedAtSec: 45 }]
  );
  check('correction just past the beat end is clamped into the beat (42s)', av.beats[0].moments[0].atSec === 42, JSON.stringify(av.beats[0].moments));
  check('clamp counts as corrected', av.stats.corrected === 1 && av.stats.dropped === 0, JSON.stringify(av.stats));

  console.log('--- 12. description grounding (job da2908ab root cause) ---');
  // Verification used to rebuild ONLY the moments — the beat description
  // was untouched, and the script is written from the description, so
  // dropping 91/100 moments still left the narration wrong. A touched
  // beat's description must now come from verified evidence only.
  const av12 = applyVerdicts(
    [
      { startSec: 50, endSec: 65, description: 'Thom examines a brain on a plate, then walks the workshop floor', moments: [
        { atSec: 52, what: 'brain on a plate' },
        { atSec: 60, what: 'Thom walks the workshop floor [WRONG]' },
      ] },
      { startSec: 65, endSec: 80, description: 'bridge walk', moments: [
        { atSec: 70, what: 'couple walks on the bridge' },
      ] },
    ],
    [
      { id: 0, beatIndex: 0, momentIndex: 0, atSec: 52, what: 'brain on a plate' },
      { id: 1, beatIndex: 0, momentIndex: 1, atSec: 60, what: 'Thom walks the workshop floor [WRONG]' },
      { id: 2, beatIndex: 1, momentIndex: 0, atSec: 70, what: 'couple walks on the bridge' },
    ],
    [
      { id: 0, confirmed: true },
      { id: 1, confirmed: false, actual: 'a sniper aiming a rifle', correctedAtSec: null },
      { id: 2, confirmed: true },
    ]
  );
  check('touched beat: wrong moment dropped', av12.beats[0].moments.length === 1 && av12.beats[0].moments[0].what === 'brain on a plate', JSON.stringify(av12.beats[0].moments));
  check('touched beat: description rebuilt from verified evidence, not the wrong claim', av12.beats[0].description.includes('brain on a plate') && av12.beats[0].description.includes('a sniper aiming a rifle') && !av12.beats[0].description.includes('walks the workshop floor'), av12.beats[0].description);
  check('untouched beat: description kept exactly', av12.beats[1].description === 'bridge walk', av12.beats[1].description);
  check('stats: 1 description rewritten', av12.stats.descriptionsRewritten === 1, JSON.stringify(av12.stats));
  // Beat with no verified evidence at all: original kept, driftRisk flagged.
  const av12b = applyVerdicts(
    [{ startSec: 0, endSec: 12, description: 'unverifiable claim', moments: [{ atSec: 5, what: 'something [WRONG]' }] }],
    [{ id: 0, beatIndex: 0, momentIndex: 0, atSec: 5, what: 'something [WRONG]' }],
    [{ id: 0, confirmed: false, actual: '', correctedAtSec: null }]
  );
  check('no evidence: description kept, driftRisk flagged', av12b.beats[0].description === 'unverifiable claim' && av12b.beats[0].driftRisk === true, JSON.stringify(av12b.beats[0]));
  check('stats: 1 beat ungrounded', av12b.stats.beatsUngrounded === 1, JSON.stringify(av12b.stats));

  if (failed > 0) throw new Error(`${failed} assertion(s) failed, ${passed} passed.`);
  console.log(`MOMENT-VERIFY TEST PASSED: ${passed} assertions, 0 failed.`);
}

main().catch((err) => {
  console.error('MOMENT-VERIFY TEST FAILED:', err.message);
  process.exit(1);
});
