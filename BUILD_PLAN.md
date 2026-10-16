# ClipAI — build plan (2026-10-16)

OpusClip-style AI clipping tool as a hosted web app. User greenlit the build on
2026-10-16 ("ok suru koro") with: web (not exe), Groq FREE + OpenAI PAID options,
real face tracking, "same as Opus" quality target.

## Pipeline (all implemented in pipeline.py)

1. **Upload** — video file via web form (2 GB cap). YouTube-link input is a
   possible v2 add-on; v1 is file upload only.
2. **Transcribe** — audio extracted with ffmpeg, chunked to fit Groq's 25 MB
   per-request limit, sent to `whisper-large-v3-turbo` with
   `response_format=verbose_json` + `timestamp_granularities=["word"]`.
   → word-level timestamps (needed for karaoke captions). Free tier covers
   ~120 min audio/day.
3. **Viral-moment curation** — transcript is chunked into `[start-end]` segments
   and sent to the LLM with a JSON-only prompt (hook strength, emotion,
   surprise, relatability, shareability; complete thoughts; no overlaps).
   - ⚡ Groq (FREE): `openai/gpt-oss-120b`
   - 🧠 OpenAI (PAID, smarter): `gpt-4o`
   Returns `[{start, end, score, title, hook}]`, snapped to word boundaries,
   de-overlapped, sorted by score.
4. **Face tracking** — YuNet ONNX model (`models/yunet.onnx`) detects faces at
   1 fps per clip; largest face's center-x is moving-average smoothed, then each
   clip is rendered in 3 s chunks with the 9:16 crop window following the face.
   Portrait source videos skip tracking (just scale).
5. **Captions** — word timings → ASS karaoke file: 3–4 words per line, active
   word highlighted gold, rest white (Hormozi style), burned in with ffmpeg.
6. **Export** — 1080×1920 H.264 + AAC, original audio kept. Job page shows
   progress log, scored clip list, preview player, and download links.

## Honest scope (do not over-promise)

- OpusClip's "virality score" is their proprietary model — our score is the
  LLM's ranking, not theirs.
- No AI B-roll, no auto-scheduler, no team features (v1).
- Face tracking follows the face horizontally (9:16 crop); it does not
  zoom/punch-in like Opus's speaker tracking.

## Costs per video (verified 2026-10-16)

- Groq path: $0 (transcribe free tier + free chat model).
- OpenAI path: transcript still Groq (free); curation call ≈ 8–12k input +
  ~2k output tokens on GPT-4o ($2.50/$10.00 per 1M) ≈ **$0.03–0.05/video**.
  $10 credit ≈ 200+ videos. (GPT-4o API is deprecated Oct 2026 but live until
  Jan 2027; swap model name in `find_moments` if needed.)
- Hosting: Railway like PageManager (nixpacks installs ffmpeg + fontconfig).

## Deploy checklist (Railway)

1. Push this folder to GitHub, deploy as Railway service (nixpacks.toml
   already installs ffmpeg).
2. Set env vars: `GROQ_KEY` (server default), optionally `OPENAI_KEY`.
   Keys live in Railway env — never in chat, never in files.
3. Open the public URL, upload a real talking-head video, run end-to-end:
   - transcript word count sane, no empty words
   - moments are complete thoughts, scores vary
   - face stays inside the 9:16 frame through the clip
   - karaoke highlight syncs with speech
   - all downloads play

## Fixes applied 2026-10-16 (research stage)

- Job status was in-memory only → broke under gunicorn multi-worker. Now
  written to `jobs/<jid>/status.json`; API reads disk first. Procfile set to
  single worker + 8 threads, timeout 1800 s for long jobs.

## Still open / user decisions

- Product name + pricing/subscription model (their call; this build supports
  server-held keys, which is what a subscription model needs).
- OpenAI API key with credit — needed only for the paid option.
- UI polish after first real-video test; demo video for selling.
