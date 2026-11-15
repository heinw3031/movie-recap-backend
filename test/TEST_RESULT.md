# Render test result

Date: 2026-11-14T09:28:38.578Z
Test: `npm test` (test/render-test.js) — NO Gemini calls, NO API key used.

## What was tested
- Source: https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/720/Big_Buck_Bunny_720_10s_1MB.mp4 (10.0s)
- Audio: TEST ONLY sine-tone MP3s generated with ffmpeg (440 Hz / 3s, 660 Hz / 4s) as stand-ins for TTS. These are NOT real Burmese TTS.
- Render: src/render.js renderRecap() — 2 scenes, clip cut to match audio duration, concat, 720p + 480p export.

## Result
- Output 720p: test/output/final_720p.mp4
- Output 480p: test/output/final_480p.mp4
- Duration: 6.40s
- Streams (720p): [{"codec_name":"h264","codec_type":"video","width":1280,"height":720},{"codec_name":"aac","codec_type":"audio"}]
- Streams (480p): [{"codec_name":"h264","codec_type":"video","width":854,"height":480},{"codec_name":"aac","codec_type":"audio"}]
- Has video stream: true
- Has audio stream: true
- PASS: YES — real MP4 with video + audio

## What this does NOT prove
- Gemini analysis, Burmese script generation, and Gemini TTS are NOT tested here — they need GEMINI_API_KEY.
- A full 5-minute recap (7 scenes, real TTS) is not tested here — only the FFmpeg render path with 2 scenes.
