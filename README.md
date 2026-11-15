# Myanmar Movie Recap — Real Backend

Real working backend for the Myanmar Movie Recap app. It replaces the three
fake parts found in the AI Studio Android build (checked from source, 2026-11-14):

| Android app today | This backend |
|---|---|
| 14 samples stream Google demo videos (`gtv-videos-bucket/sample/…`) with invented Burmese titles | No demo videos. Real uploaded video is analysed. |
| Script prompt uses only `fileName + category + length` — video never watched | `analyzeVideo()` extracts 1 frame / 30s and sends real frames to Gemini 2.5 Flash |
| `startVoiceAndSyncGeneration()` = `delay(550ms) × 5` (~3 s), no audio, no video | Real Gemini TTS per scene + real FFmpeg render to a playable MP4 |

## How it works

```
Phone → POST /api/jobs (upload video, length 5/10/15/30)
  → analysing (Gemini watches frames) → scripting (Burmese script)
  → awaiting-approval (creator edits / approves scenes)
  → POST /api/jobs/:id/generate → tts (real Gemini TTS per scene)
  → rendering (FFmpeg cuts clips to audio length, 0.9x–1.1x speed only)
  → done → GET /api/jobs/:id/video streams final 720p / 480p MP4
```

Job stages (real progress, never faked):
`queued → analysing → scripting → awaiting-approval → tts → rendering → done | failed`

If Gemini returns no audio, or analysis fails, the job is marked `failed`
with the real error message. Nothing is ever reported as "done ✓" in ~3 seconds.

## Run locally

```bash
cd movie-recap-backend
npm install
cp .env.example .env   # then put your key in .env (see below)
npm start              # listens on http://localhost:8080
npm test               # render test, no API key needed (see below)
```

### Get a Gemini API key (free tier)

1. Open https://aistudio.google.com/ and sign in with Google.
2. Click **Get API key** → create a key in a Google Cloud project.
3. Put it in `.env`: `GEMINI_API_KEY=your-key-here`
4. Never commit `.env`, and never put the key in code. The server reads it
   only from the environment and fails honestly if it is missing.

## API

| Method | Endpoint | What it does |
|---|---|---|
| `GET` | `/health` | `{ ok: true }` |
| `POST` | `/api/jobs` | Multipart upload, field `video`, plus `lengthMinutes` (5/10/15/30) and `category`. Returns `{ jobId, status: "queued" }`. Max upload 500 MB by default (`MAX_UPLOAD_MB`). |
| `GET` | `/api/jobs/:id` | Status, progress %, scenes, title, error, `finalVideoUrl` when done. |
| `GET` | `/api/jobs` | List jobs (newest first). |
| `POST` | `/api/jobs/:id/script` | Approve / edit the script. Body `{ title?, scenes: [{ sceneNumber, sourceStartSec, sourceEndSec, narrationMy }] }`. Only works when status is `awaiting-approval`. |
| `POST` | `/api/jobs/:id/generate` | Start real TTS + render. Only after script approval. Returns 202; poll `GET /api/jobs/:id`. |
| `GET` | `/api/jobs/:id/video` | Stream the final MP4. `?quality=480` for the 480p version (default 720p). 409 while the job is not `done`. |

Job data is stored under `./data/jobs/<jobId>/` (`job.json`, `source.mp4`,
`frames/`, `audio/`, `final_720p.mp4`, `final_480p.mp4`). The uploaded
source video is deleted after a successful render (set `KEEP_SOURCE=1` to
keep it for local debugging) — this saves storage and limits copyright risk.

## Test (no API key)

```bash
npm test
```

This downloads a short Google demo video (`ForBiggerBlazes.mp4`), generates
2 short sine-tone MP3s with ffmpeg as **TEST ONLY** stand-ins for TTS audio
(they are tones, not Burmese speech), renders a final MP4 with
`src/render.js`, and checks it with ffprobe. Results are saved to
`test/TEST_RESULT.md` with the output path, duration and streams.

It proves the FFmpeg render path is real. It does **not** test Gemini
analysis, script generation or TTS — those need `GEMINI_API_KEY`.

To test the full pipeline with a real key, upload a short (2–3 min) video
with a clear filename, pick 5 minutes, approve the script, then generate.
A 5-minute recap should take several minutes (not ~3 seconds — if it
finishes in 3 seconds, something is still fake).

## Deploy for FREE (test version — no billing required)

Target: **free tier only** — Gemini API free tier, Firebase AI Logic free
starter tier (Android side), and Cloud Run free tier. AI Studio lets you
deploy up to 2 apps to Cloud Run free (per Google's AI Studio docs).
No paid service is required for the test version.

### Option A — Deploy this backend as an AI Studio web app / Cloud Run (recommended)

1. In Google AI Studio, open **Build mode** and create a **web app** from
   this backend (or deploy the Docker image below to Cloud Run).
2. Set the secret `GEMINI_API_KEY` in Cloud Run / AI Studio Secrets — never
   in the code or the image.
3. Deploy. You get a `https://<your-app>.run.app` URL — that is the
   `BASE_URL` the Android app calls (see `movie-recap-ai-studio-v2-prompt.md`).
4. Test with `curl https://<your-app>.run.app/health` — it should return
   `{ "ok": true }`.

### Option B — Deploy the Docker image to Cloud Run yourself

```bash
# From this folder. Needs gcloud + a Google Cloud project (no billing needed
# for the free tier, but the project must exist).
gcloud builds submit --tag gcr.io/PROJECT-ID/movie-recap-backend .
gcloud run deploy movie-recap-backend \
  --image gcr.io/PROJECT-ID/movie-recap-backend \
  --platform managed --region asia-southeast1 \
  --allow-unauthenticated \
  --set-secrets GEMINI_API_KEY=gemini-api-key:latest \
  --memory 2Gi --timeout 900
```

Note: rendering a 5–30 min recap can take 5–15 minutes. Cloud Run's
default request timeout is 5 minutes, so the app polls `GET /api/jobs/:id`
— it never waits on one long request. The `--timeout 900` above is for
the render worker; long renders are also why a queue / Cloud Tasks is the
next step if you outgrow the free tier.

### Free quotas / limits — what fits and what does not

Quotas change, so check the current pages before you rely on them
(AI Studio / Google Cloud Console → Quotas). The figures below are the
approximate free-tier shape as of 2026-11-14:

- **Gemini API free tier** — a limited number of requests per day / per
  minute per model (e.g. Gemini 2.5 Flash is in the low hundreds of
  requests/day on the free tier; TTS preview models are more limited).
  A 5-minute test recap uses roughly: 1 analysis call with ~10 frames,
  1 script call, and ~7 TTS calls — that fits comfortably in a day's free
  quota. Making many recaps in one day, or 30-minute recaps (~38 TTS
  calls + more frames), will hit the quota and jobs will fail with a
  quota error. That is expected on the free tier — add a daily limit in
  the app (e.g. 3–5 recaps/day) and show the real error to the creator.
- **Cloud Run free tier** — roughly 2 million requests/month, 180,000
  vCPU-seconds/month, 360,000 GiB-seconds/month, and 1 GB egress/month
  in the free allowance. A few test recaps a day fit. Heavy video
  rendering (long FFmpeg jobs, large uploads) burns vCPU-seconds fast —
  keep test videos short (2–3 min source, 5-min recap) and delete the
  source after render (this backend already does).
- **Storage** — job files live on the container's local disk in this
  version. That is fine for local tests, but Cloud Run containers are
  ephemeral: for a real deployment, move `data/jobs/` to Cloud Storage
  (or Firebase Storage) and return signed URLs instead of streaming
  from disk. That is the first paid-adjacent step if you outgrow free.
- **Firebase AI Logic (Android side)** — free starter tier covers the
  script call the old app made; the new app should call this backend
  instead, so the key stays server-side and is never embedded in the APK.

**Bottom line:** 5-minute test recaps fit in the free tier. Heavy use —
many users, 30-minute recaps all day, or storing lots of finished videos —
will hit Gemini quota, Cloud Run CPU, or storage limits. That is the
signal to add limits, a queue, and (only then) billing.

## Costs if you outgrow free (rough, 2026)

From the spec (`movie-recap-real-backend.md`): a 10-min recap ≈ $1–3
(Gemini Flash + TTS + Cloud Run + storage). A 5-min recap is cheaper.
These are estimates, not a promise — check current pricing in Google Cloud.

## Files

```
package.json  server.js  Dockerfile  .env.example
src/
  store.js    # job store (memory + ./data/jobs/<id>/job.json)
  gemini.js   # analyzeVideo / generateScript / generateSceneAudio (real Gemini)
  render.js   # cutClip / renderRecap (real FFmpeg)
  pipeline.js # runAnalysis / runGenerate, real progress stages
test/
  render-test.js  # npm test — FFmpeg render test, no API key
  TEST_RESULT.md  # written by npm test
```

## What still needs work (not in this version)

- Cloud Storage for finished videos (currently local disk only).
- A queue (e.g. Cloud Tasks) so many jobs do not run at once on Cloud Run.
- User accounts / per-user daily limits (quota protection).
- Copyright: only upload videos you have the right to recap. The backend
  deletes the source after render, but it cannot check your rights.
- The Android app still needs the v2 wiring (see
  `../movie-recap-ai-studio-v2-prompt.md`) — the old app still contains
  the fake `delay(550ms)` path until it is rebuilt with that prompt.
