#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ClipAI pipeline: transcribe -> viral moments -> face-tracked 9:16 clips + karaoke captions."""
import os, json, re, math, subprocess, tempfile, requests
import cv2

GROQ_BASE = "https://api.groq.com/openai/v1"
OPENAI_BASE = "https://api.openai.com/v1"

def log(job_dir, msg):
    print("[CLIPAI] %s" % msg, flush=True)
    try:
        with open(os.path.join(job_dir, "log.txt"), "a", encoding="utf-8") as f:
            f.write(msg + "\n")
    except Exception:
        pass

def run_cmd(args, timeout=900):
    p = subprocess.run(args, capture_output=True, timeout=timeout)
    return p.returncode, p.stdout.decode("utf-8", "replace"), p.stderr.decode("utf-8", "replace")[-800:]

def media_duration(path):
    rc, out, _ = run_cmd(["ffprobe", "-v", "error", "-show_entries",
                          "format=duration", "-of", "csv=p=0", path], timeout=60)
    try:
        return float(out.strip().split("\n")[0])
    except Exception:
        return 0.0

def video_size(path):
    rc, out, _ = run_cmd(["ffprobe", "-v", "error", "-select_streams", "v:0",
                          "-show_entries", "stream=width,height",
                          "-of", "csv=p=0", path], timeout=60)
    try:
        w, h = out.strip().split("\n")[0].split(",")
        return int(w), int(h)
    except Exception:
        return 1920, 1080

# ---------------- transcription (Groq Whisper, free) ----------------
def transcribe(groq_key, video_path, job_dir):
    log(job_dir, "Extracting audio…")
    audio = os.path.join(job_dir, "audio.mp3")
    rc, _, err = run_cmd(["ffmpeg", "-y", "-i", video_path, "-vn",
                          "-c:a", "libmp3lame", "-b:a", "128k", audio])
    if rc != 0 or not os.path.exists(audio):
        return False, "audio extract failed: %s" % err[:200]
    log(job_dir, "Transcribing with Whisper…")
    # chunk audio into 10-min pieces for the API
    dur = media_duration(audio)
    words = []
    t = 0.0
    while t < dur:
        chunk = os.path.join(job_dir, "achunk.mp3")
        rc, _, _ = run_cmd(["ffmpeg", "-y", "-ss", str(t), "-t", "600",
                            "-i", audio, "-c", "copy", chunk])
        with open(chunk, "rb") as f:
            r = requests.post(
                GROQ_BASE + "/audio/transcriptions",
                headers={"Authorization": "Bearer " + groq_key},
                files={"file": ("audio.mp3", f, "audio/mpeg")},
                data={"model": "whisper-large-v3-turbo",
                      "response_format": "verbose_json",
                      "timestamp_granularities[]": "word"},
                timeout=600)
        try:
            j = r.json()
        except Exception:
            return False, "whisper bad response: %s" % r.text[:200]
        if "words" not in j and "segments" not in j:
            return False, "whisper error: %s" % str(j)[:300]
        for w in j.get("words", []):
            words.append({"w": w["word"].strip(),
                          "s": float(w["start"]) + t,
                          "e": float(w["end"]) + t})
        for s in (j.get("segments") or []):
            if not j.get("words"):
                words.append({"w": s["text"].strip(),
                              "s": float(s["start"]) + t,
                              "e": float(s["end"]) + t})
        try:
            os.remove(chunk)
        except Exception:
            pass
        t += 600
    words = [w for w in words if w["w"]]
    open(os.path.join(job_dir, "words.json"), "w", encoding="utf-8").write(
        json.dumps(words, ensure_ascii=False))
    log(job_dir, "Transcript: %d words" % len(words))
    return True, words

# ---------------- viral moments (Groq free or OpenAI paid) ----------------
MOMENT_PROMPT = """You are a viral short-form video editor. Given a transcript with [start-end sec] segments, pick the {n} MOST VIRAL moments for TikTok/Reels/Shorts.

Rules:
- Each moment {lo}-{hi} seconds long, must be a complete thought (don't cut mid-sentence)
- Rank by hook strength, emotion, surprise, relatability, shareability
- Moments must not overlap
- Respond ONLY with JSON: [{{"start": 12.5, "end": 47.0, "score": 95, "title": "short punchy title", "hook": "first line that grabs attention"}}]

Transcript:
{transcript}
"""

def find_moments(provider, api_key, words, n_clips, lo, hi, job_dir):
    # build compact transcript
    segs = []
    cur, cur_s = [], None
    for w in words:
        if cur_s is None:
            cur_s = w["s"]
        cur.append(w["w"])
        if w["e"] - cur_s >= 15 or len(" ".join(cur)) > 220:
            segs.append("[%.1f-%.1f] %s" % (cur_s, w["e"], " ".join(cur)))
            cur, cur_s = [], None
    if cur:
        segs.append("[%.1f-%.1f] %s" % (cur_s, words[-1]["e"], " ".join(cur)))
    transcript = "\n".join(segs)
    # truncate to ~12k chars to fit context
    if len(transcript) > 12000:
        transcript = transcript[:12000]
    prompt = MOMENT_PROMPT.format(n=n_clips, lo=lo, hi=hi, transcript=transcript)
    log(job_dir, "AI finding viral moments (%s)…" % provider)
    if provider == "openai":
        url = OPENAI_BASE + "/chat/completions"
        model = "gpt-4o"
        key = api_key
    else:
        url = GROQ_BASE + "/chat/completions"
        model = "openai/gpt-oss-120b"
        key = api_key
    r = requests.post(url, headers={"Authorization": "Bearer " + key,
                                    "Content-Type": "application/json"},
                      json={"model": model, "temperature": 0.7,
                            "response_format": {"type": "json_object"},
                            "messages": [
                                {"role": "system", "content": "You are a viral video editor. Output JSON only."},
                                {"role": "user", "content": prompt}]},
                      timeout=300)
    try:
        j = r.json()
        txt = j["choices"][0]["message"]["content"]
    except Exception:
        return False, "LLM error: %s" % r.text[:300]
    # extract JSON array
    m = re.search(r"\[.*\]", txt, re.S)
    if not m:
        return False, "LLM returned no JSON"
    try:
        moments = json.loads(m.group(0))
    except Exception:
        return False, "LLM JSON parse failed"
    # sanitize
    dur = words[-1]["e"] if words else 0
    clean = []
    for mo in moments[:n_clips]:
        try:
            s, e = float(mo["start"]), float(mo["end"])
        except Exception:
            continue
        if e <= s or e - s < lo * 0.7 or s < 0 or e > dur + 1:
            continue
        # snap to word boundaries
        s = snap_time(words, s, "s")
        e = snap_time(words, e, "e")
        if e <= s or e - s < 5:
            continue
        clean.append({"start": round(s, 2), "end": round(e, 2),
                      "score": int(mo.get("score", 80)),
                      "title": str(mo.get("title", "Clip"))[:60],
                      "hook": str(mo.get("hook", ""))[:120]})
    clean.sort(key=lambda m: -m["score"])
    # de-overlap
    final = []
    for mo in clean:
        if all(mo["end"] <= f["start"] or mo["start"] >= f["end"] for f in final):
            final.append(mo)
    log(job_dir, "Found %d moments" % len(final))
    return True, final

def snap_time(words, t, which):
    best, bd = t, 1e9
    for w in words:
        c = w["s"] if which == "s" else w["e"]
        if abs(c - t) < bd:
            bd, best = abs(c - t), c
    return best

# ---------------- face tracking (YuNet) ----------------
_YUNET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "yunet.onnx")
_FONTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")

def face_track_x(video_path, start, end, vw, job_dir):
    """Return smoothed face-center x positions: list of (time, x)."""
    detector = cv2.FaceDetectorYN_create(_YUNET, "", (320, 320))
    cap = cv2.VideoCapture(video_path)
    pts = []
    t = start
    last_x = vw / 2
    while t < end:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            t += 1.0
            continue
        h, w = frame.shape[:2]
        detector.setInputSize((w, h))
        _, faces = detector.detect(frame)
        if faces is not None and len(faces):
            # largest face: [x, y, w, h, ...]
            f = max(faces, key=lambda b: b[2] * b[3])
            last_x = float(f[0] + f[2] / 2) / w * vw
        pts.append((t - start, last_x))
        t += 1.0
    cap.release()
    # smooth with moving average
    sm = []
    for i, (tt, x) in enumerate(pts):
        win = pts[max(0, i - 2):i + 3]
        sm.append((tt, sum(p[1] for p in win) / len(win)))
    return sm

# ---------------- word-highlight captions (ASS) ----------------
def _ass_safe(t):
    """Escape ASS special chars in transcribed words."""
    return t.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")

def write_ass(job_dir, words, start, end, path, max_chars=30):
    """Hormozi-style captions: 3-4 words/line, active word gold, rest white."""
    segs = [w for w in words if w["e"] > start and w["s"] < end]
    lines, cur = [], []
    for i, w in enumerate(segs):
        cur.append(w)
        txt = " ".join(c["w"] for c in cur)
        nxt = segs[i + 1] if i + 1 < len(segs) else None
        pause = (nxt["s"] - w["e"]) > 0.5 if nxt else True
        if len(txt) >= max_chars or pause or len(cur) >= 4:
            lines.append(cur)
            cur = []
    if cur:
        lines.append(cur)
    def ts(t):
        t = max(0, t - start)
        return "%d:%02d:%05.2f" % (int(t // 3600), int(t % 3600 // 60), t % 60)
    GOLD = r"{\c&H00D7FF&}"
    WHITE = r"{\c&H00FFFFFF&}"
    style = ("Style: Cap,DejaVu Sans,72,&H00FFFFFF,&H000000FF,&H90000000,&H00000000,"
             "-1,0,0,0,100,100,0,0,1,4,0.5,2,10,10,140,1")
    out = ["[Script Info]", "ScriptType: v4.00+", "PlayResX: 1080", "PlayResY: 1920",
           "WrapStyle: 2", "", "[V4+ Styles]",
           "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
           style, "", "[Events]",
           "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"]
    for ln in lines:
        if not ln:
            continue
        for wi, w in enumerate(ln):
            wend = ln[wi + 1]["s"] if wi + 1 < len(ln) else w["e"] + 0.25
            parts = []
            for ci, c in enumerate(ln):
                col = GOLD if ci == wi else WHITE
                parts.append("%s%s" % (col, _ass_safe(c["w"])))
            out.append("Dialogue: 0,%s,%s,Cap,,0,0,0,,%s" %
                       (ts(w["s"]), ts(min(wend, ln[-1]["e"] + 0.3)), " ".join(parts)))
    open(path, "w", encoding="utf-8").write("\n".join(out))

# ---------------- render one clip ----------------
def render_clip(video_path, words, moment, idx, job_dir, out_dir):
    s, e = moment["start"], moment["end"]
    dur = e - s
    if dur <= 0:
        log(job_dir, "Clip %d: invalid time range %.1f-%.1f, skipping" % (idx + 1, s, e))
        return False, "invalid time range"
    vw, vh = video_size(video_path)
    if vw <= 0 or vh <= 0:
        log(job_dir, "Clip %d: bad video size %dx%d" % (idx + 1, vw, vh))
        return False, "bad video size"
    # 9:16 crop width; if video already portrait, just scale
    is_portrait = vh >= vw
    cw = int(vh * 9 / 16)
    if cw >= vw or cw <= 0:
        is_portrait = True
        cw = vw
    log(job_dir, "Clip %d: %.0fs-%.0fs (%s) size=%dx%d portrait=%s" %
        (idx + 1, s, e, moment["title"], vw, vh, is_portrait))
    # face track (only needed for landscape crop)
    track = [] if is_portrait else face_track_x(video_path, s, e, vw, job_dir)
    # build crop expression from track (step every 0.5s)
    # use crop with animated x via sendcmd is complex; instead re-encode with
    # per-segment crops: split moment into 3s chunks, crop each, concat
    chunk = 3.0
    parts = []
    t = 0.0
    ci = 0
    while t < dur:
        te = min(t + chunk, dur)
        # face x at midpoint
        mid = t + (te - t) / 2
        fx = vw / 2
        for tt, x in track:
            if tt <= mid:
                fx = x
        cx = int(max(0, min(vw - cw, fx - cw / 2)))
        # crop dims must be positive and inside frame
        cx = max(0, min(cx, vw - cw))
        part = os.path.join(job_dir, "clip%d_p%d.mp4" % (idx, ci))
        ass = os.path.join(job_dir, "clip%d_p%d.ass" % (idx, ci))
        write_ass(job_dir, words, s + t, s + te, ass)
        ass_esc = ass.replace("'", r"'\''").replace(":", r"\:")
        sub = "subtitles='%s':fontsdir='%s'" % (ass_esc, _FONTS)
        if is_portrait:
            vf = "scale=1080:1920,%s" % sub
        else:
            vf = "crop=%d:%d:%d:0,scale=1080:1920,%s" % (cw, vh, cx, sub)
        rc, _, err = run_cmd(["ffmpeg", "-y", "-ss", str(s + t), "-t", str(te - t),
                              "-i", video_path, "-vf", vf,
                              "-c:v", "libx264", "-preset", "fast", "-crf", "19",
                              "-c:a", "aac", "-b:a", "160k", part])
        if (rc != 0 or not os.path.exists(part)) and os.path.exists(ass):
            # fallback: render without subtitles (libass/font issue)
            log(job_dir, "chunk %d subtitles failed, retry plain" % ci)
            vf_plain = "scale=1080:1920" if is_portrait else "crop=%d:%d:%d:0,scale=1080:1920" % (cw, vh, cx)
            rc, _, err = run_cmd(["ffmpeg", "-y", "-ss", str(s + t), "-t", str(te - t),
                                  "-i", video_path, "-vf", vf_plain,
                                  "-c:v", "libx264", "-preset", "fast", "-crf", "19",
                                  "-c:a", "aac", "-b:a", "160k", part])
        if rc != 0 or not os.path.exists(part):
            log(job_dir, "chunk render failed: %s" % err[-1500:])
            return False, "render failed"
        parts.append(part)
        try:
            os.remove(ass)
        except Exception:
            pass
        t = te
        ci += 1
    # concat
    lst = os.path.join(job_dir, "clip%d.txt" % idx)
    open(lst, "w").write("".join("file '%s'\n" % p for p in parts))
    out = os.path.join(out_dir, "clip%d_%s.mp4" % (idx + 1, re.sub(r"\W+", "_", moment["title"])[:30]))
    rc, _, err = run_cmd(["ffmpeg", "-y", "-f", "concat", "-safe", "0",
                          "-i", lst, "-c", "copy", out])
    for p in parts:
        try:
            os.remove(p)
        except Exception:
            pass
    if rc != 0 or not os.path.exists(out):
        return False, "concat failed: %s" % err[:200]
    return True, out

# ---------------- full job ----------------
def run_job(job_dir, video_path, cfg):
    os.makedirs(job_dir, exist_ok=True)
    out_dir = os.path.join(job_dir, "clips")
    os.makedirs(out_dir, exist_ok=True)
    open(os.path.join(job_dir, "log.txt"), "w").write("")
    provider = cfg.get("provider", "groq")
    groq_key = cfg.get("groq_key", "")
    openai_key = cfg.get("openai_key", "")
    llm_key = openai_key if provider == "openai" else groq_key
    if not llm_key:
        return False, "API key missing"
    ok, words = transcribe(groq_key, video_path, job_dir)
    if not ok:
        return False, words
    ok, moments = find_moments(provider, llm_key, words, cfg.get("n_clips", 3),
                               cfg.get("min_len", 20), cfg.get("max_len", 50),
                               job_dir)
    if not ok:
        return False, moments
    if not moments:
        return False, "no moments found"
    open(os.path.join(job_dir, "moments.json"), "w", encoding="utf-8").write(
        json.dumps(moments, ensure_ascii=False, indent=1))
    done = []
    for i, mo in enumerate(moments):
        ok, out = render_clip(video_path, words, mo, i, job_dir, out_dir)
        if ok:
            done.append({"file": os.path.basename(out), "title": mo["title"],
                         "score": mo["score"], "start": mo["start"], "end": mo["end"]})
        else:
            log(job_dir, "clip %d failed: %s" % (i + 1, out))
    open(os.path.join(job_dir, "done.json"), "w", encoding="utf-8").write(
        json.dumps(done, ensure_ascii=False, indent=1))
    log(job_dir, "DONE: %d clips" % len(done))
    return True, done
