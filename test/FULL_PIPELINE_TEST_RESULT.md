# Full pipeline test result — REAL Gemini test

Date: 2026-11-14 (UTC). Started 10:05 UTC, finished 10:14 UTC.
Tester: subagent run for the Myanmar Movie Recap backend.
Source video: `test/test-source.mp4` — Big Buck Bunny, 10.0 s, 1280x720, 969,201 bytes.
Requested recap: 5 minutes, category Action.

## Verdict: BLOCKED — pipeline did NOT complete

The backend is correct up to the first Gemini call, the API key is VALID,
but **no Gemini API call can succeed from this VM**: every HTTPS POST
hangs with 0 bytes received (VM egress proxy), while GET works fine.
Gemini analysis / script / TTS are all HTTPS POST, so the pipeline
cannot finish here. This is an environment limitation, not a backend
bug and not a key problem. The same backend should work on Cloud Run
or any machine with normal internet egress.

No API key value appears anywhere in this file.

## Stage-by-stage

| Stage | Result | Evidence |
|---|---|---|
| Server start (`npm start`, port 8080, dotenv loads `.env`) | PASS | `GET /health` → `{"ok":true,"service":"myanmar-movie-recap-backend"}`; log: `GEMINI_API_KEY set: true` (boolean only) |
| Upload `POST /api/jobs` (test-source.mp4, lengthMinutes=5, category=Action) | PASS | Job 1: `34286e37-f4e2-4cc2-86c2-c82d4c18661d`; Job 2: `e4dfcc5a-612c-4f01-9e65-119df4ba674d` |
| Frame extraction (`analyseVideo` → `extractFrames`) | FAIL then FIXED, PASS after fix | See bug note below. After fix, `frame_0001.jpg` (177,478 bytes, 1280x720) extracted in job 2 |
| Gemini video analysis (frames → `gemini-2.5-flash`) | FAIL | Job 2 stuck in `analysing` (progress 10) 10:06:37–10:11:36 UTC (~5 min), then `status: failed`, error: `fetch failed` |
| Burmese script generation | NOT REACHED | No analysis output, so no script was produced. Title: null, scenes: 0 |
| Script approval (`POST /api/jobs/:id/script`) | NOT REACHED | Job never became `awaiting-approval` |
| Generate / TTS (`POST /api/jobs/:id/generate`) | NOT REACHED | — |
| FFmpeg render (720p + 480p) | NOT REACHED in this run | Render path was already proven separately in `test/TEST_RESULT.md` (final_720p.mp4 6.40 s, h264 1280x720 + aac) |
| Final video download (`GET /api/jobs/:id/video`) | NOT REACHED | No final MP4 exists for these jobs |

## Bug found and fixed during this test (real bug, in the code)

Job 1 failed in `extractFrames` (`src/gemini.js`) with:
`ffmpeg -y -i source.mp4 -vf fps=1/30 -q:v 3 frames/frame_%04d.jpg` →
`[mjpeg] Non full-range YUV is non-standard, set strict_std_compliance to at most unofficial` …
`ff_frame_thread_encoder_init failed` … `Conversion failed!`

Two causes, both in the one ffmpeg command:
1. ffmpeg 8.1.2's MJPEG encoder rejects non-full-range YUV input.
2. `fps=1/30` emits **zero frames** for any video shorter than 30 s (verified:
   `fps=1/30` on the 10 s test video → 0 files; `fps=1/5` → 2 files).

Fix applied to `src/gemini.js` `extractFrames()` (server restarted after):
`-vf fps=1/${intervalSec}:round=up,format=yuvj420p`
- `round=up` makes the t=0 frame emit even for short videos (verified: 1 frame).
- `format=yuvj420p` gives the MJPEG encoder full-range YUV (verified: frame encodes fine).

Verified after fix: job 2 extracted `frames/frame_0001.jpg` (177,478 bytes) at 10:06 UTC
before the Gemini call hung. No other code was changed.

## Network diagnosis (why Gemini calls fail here)

All run from this VM, 2026-11-14 ~10:09–10:14 UTC:

- `GET https://generativelanguage.googleapis.com/v1beta/models` (curl, invalid key): HTTP 400 in 0.9 s — GET works.
- `GET` with `x-goog-api-key: INVALID` header (Node fetch): HTTP 400 in 0.9 s — GET works.
- `GET /v1beta/models` with the real key from `.env` (Node fetch, key never printed): **HTTP 200 in 0.6 s, models `gemini-2.5-flash`, `gemini-2.5-pro` — the API key is VALID.**
- `POST https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent` (curl, Node fetch, invalid key): hangs, 0 bytes received, killed by 25–30 s timeout.
- `POST https://httpbin.org/post` and `POST https://example.com/` (curl): same hang, 0 bytes received. curl `-v` shows TLS handshake completes and the 10-byte body is sent, then nothing comes back.
- `POST http://httpbin.org/post` (plain HTTP): **HTTP 200 in 0.18 s — works.**
- Direct connection without proxy (`curl --noproxy '*'`): HTTP 000 in 0.04 s — no direct internet; all egress goes via `hatch-egress-proxy:3128` (`HTTP_PROXY`/`HTTPS_PROXY` set, `NODE_USE_ENV_PROXY=1`).

Conclusion: this VM's egress path drops HTTPS POST requests (body sent, zero
bytes returned) for every host, while HTTPS GET and plain-HTTP POST work.
Every Gemini API call (analysis, script, TTS) is an HTTPS POST, so none of
them can complete from this VM. A direct SDK probe (text-only
`generateContent`, `gemini-2.5-flash`) also failed after 300.8 s with
`fetch failed`. The SDK's `models.list` GET also hung (60 s timeout) — the
SDK's transport does not behave like bare fetch here — but that does not
matter: even a working GET path cannot run the pipeline, because the
pipeline needs POST.

## What this proves / does not prove

- PROVEN: server, upload, job store, frame extraction (after fix), API key
  validity, FFmpeg render path (earlier test).
- NOT PROVEN: Gemini frame analysis, Burmese script generation, per-scene
  Gemini TTS, end-to-end render with real narration. None of these ran —
  the first Gemini POST never returned.
- The failure is environmental. Nothing in the job error points at the
  backend code, the key, or the video file.

## Next step (for the parent agent / user)

Run this same backend on Cloud Run (free tier, Dockerfile is included) or
any machine with normal internet, with the same `.env` key:
`npm start` → upload a short video → approve the script → Generate.
A 5-minute recap should take several minutes of real TTS + render, not ~3 s.
Do NOT retry the full pipeline from this VM — HTTPS POST is blocked here,
so it will fail the same way.

## Job records

- Job 1: `34286e37-f4e2-4cc2-86c2-c82d4c18661d`, created 2026-11-14T10:05:35Z,
  failed at frame extraction (ffmpeg bug, now fixed).
- Job 2: `e4dfcc5a-612c-4f01-9e65-119df4ba674d`, created 2026-11-14T10:06:35Z,
  failed 2026-11-14T10:11:36Z, stage `analysing`, progress 10,
  error: `fetch failed`. Title null, scenes 0, `sourceDurationSec` null.
- No `final_720p.mp4` / `final_480p.mp4` was produced for either job.
- Server stopped after this test. No deployment, no publish, no commit.
