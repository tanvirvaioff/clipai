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

# ---------------- transcription (Groq Whisper free / OpenAI Whisper paid) ----------------
def transcribe(provider, api_key, video_path, job_dir):
    log(job_dir, "Extracting audio…")
    audio = os.path.join(job_dir, "audio.mp3")
    rc, _, err = run_cmd(["ffmpeg", "-y", "-i", video_path, "-vn",
                          "-c:a", "libmp3lame", "-b:a", "128k", audio])
    if rc != 0 or not os.path.exists(audio):
        return False, "audio extract failed: %s" % err[:200]
    log(job_dir, "Transcribing with Whisper (%s)…" % provider)
    if provider == "openai":
        base, model = OPENAI_BASE, "whisper-1"
    else:
        base, model = GROQ_BASE, "whisper-large-v3-turbo"
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
                base + "/audio/transcriptions",
                headers={"Authorization": "Bearer " + api_key},
                files={"file": ("audio.mp3", f, "audio/mpeg")},
                data={"model": model,
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
    payload = {"model": model, "temperature": 0.7,
               "messages": [
                   {"role": "system", "content": "You are a viral video editor. Output JSON only."},
                   {"role": "user", "content": prompt}]}
    if provider == "openai":
        payload["response_format"] = {"type": "json_object"}
    r = requests.post(url, headers={"Authorization": "Bearer " + key,
                                    "Content-Type": "application/json"},
                      json=payload, timeout=300)
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

def face_track_x(video_path, start, end, vw, job_dir, step=1.0):
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
            t += step
            continue
        h, w = frame.shape[:2]
        detector.setInputSize((w, h))
        _, faces = detector.detect(frame)
        if faces is not None and len(faces):
            # largest face: [x, y, w, h, ...]
            f = max(faces, key=lambda b: b[2] * b[3])
            last_x = float(f[0] + f[2] / 2) / w * vw
        pts.append((t - start, last_x))
        t += step
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

def write_ass(job_dir, words, start, end, path, style="hormozi", max_chars=30):
    """Caption styles: hormozi (gold karaoke), beast (big yellow caps), minimal (small white)."""
    segs = [w for w in words if w["e"] > start and w["s"] < end]
    lines, cur = [], []
    per_line = 4 if style == "hormozi" else (3 if style == "beast" else 6)
    chars = max_chars if style == "hormozi" else (24 if style == "beast" else 42)
    for i, w in enumerate(segs):
        cur.append(w)
        txt = " ".join(c["w"] for c in cur)
        nxt = segs[i + 1] if i + 1 < len(segs) else None
        pause = (nxt["s"] - w["e"]) > 0.5 if nxt else True
        if len(txt) >= chars or pause or len(cur) >= per_line:
            lines.append(cur)
            cur = []
    if cur:
        lines.append(cur)
    def ts(t):
        t = max(0, t - start)
        return "%d:%02d:%05.2f" % (int(t // 3600), int(t % 3600 // 60), t % 60)
    if style == "beast":
        # big yellow all-caps, thick black outline
        st = ("Style: Cap,DejaVu Sans,88,&H0000FFFF,&H000000FF,&H00000000,&H00000000,"
              "-1,0,0,0,100,100,0,0,1,6,0,2,10,10,120,1")
        body = lambda ln: _ass_safe(" ".join(c["w"] for c in ln).upper())
        hl = False
    elif style == "minimal":
        # small white bottom, no highlight
        st = ("Style: Cap,DejaVu Sans,48,&H00FFFFFF,&H000000FF,&H90000000,&H00000000,"
              "-1,0,0,0,100,100,0,0,1,3,0,2,10,10,60,1")
        body = lambda ln: _ass_safe(" ".join(c["w"] for c in ln))
        hl = False
    else:
        # hormozi: karaoke gold highlight
        st = ("Style: Cap,DejaVu Sans,72,&H00FFFFFF,&H000000FF,&H90000000,&H00000000,"
              "-1,0,0,0,100,100,0,0,1,4,0.5,2,10,10,140,1")
        hl = True
    out = ["[Script Info]", "ScriptType: v4.00+", "PlayResX: 1080", "PlayResY: 1920",
           "WrapStyle: 2", "", "[V4+ Styles]",
           "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
           st, "", "[Events]",
           "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"]
    GOLD = r"{\c&H00D7FF&}"
    WHITE = r"{\c&H00FFFFFF&}"
    for ln in lines:
        if not ln:
            continue
        if hl:
            for wi, w in enumerate(ln):
                wend = ln[wi + 1]["s"] if wi + 1 < len(ln) else w["e"] + 0.25
                parts = []
                for ci, c in enumerate(ln):
                    col = GOLD if ci == wi else WHITE
                    parts.append("%s%s" % (col, _ass_safe(c["w"])))
                out.append("Dialogue: 0,%s,%s,Cap,,0,0,0,,%s" %
                           (ts(w["s"]), ts(min(wend, ln[-1]["e"] + 0.3)), " ".join(parts)))
        else:
            s_t, e_t = ts(ln[0]["s"]), ts(ln[-1]["e"] + 0.25)
            out.append("Dialogue: 0,%s,%s,Cap,,0,0,0,,%s" % (s_t, e_t, body(ln)))
    open(path, "w", encoding="utf-8").write("\n".join(out))

def write_srt(words, start, end, path):
    """SRT subtitles for a clip time range."""
    segs = [w for w in words if w["e"] > start and w["s"] < end]
    lines, cur = [], []
    for i, w in enumerate(segs):
        cur.append(w)
        txt = " ".join(c["w"] for c in cur)
        nxt = segs[i + 1] if i + 1 < len(segs) else None
        pause = (nxt["s"] - w["e"]) > 0.5 if nxt else True
        if len(txt) >= 42 or pause or len(cur) >= 6:
            lines.append(cur)
            cur = []
    if cur:
        lines.append(cur)
    def ts(t):
        t = max(0, t - start)
        ms = int(round((t - int(t)) * 1000))
        return "%02d:%02d:%02d,%03d" % (int(t // 3600), int(t % 3600 // 60), int(t % 60), ms)
    out = []
    for n, ln in enumerate(lines, 1):
        if not ln:
            continue
        out.append(str(n))
        out.append("%s --> %s" % (ts(ln[0]["s"]), ts(ln[-1]["e"])))
        out.append(" ".join(c["w"] for c in ln))
        out.append("")
    open(path, "w", encoding="utf-8").write("\n".join(out))
    return len(lines)

# ---------------- render one clip ----------------
def detect_silences(video_path, start, end, job_dir, thresh_db=-40, min_dur=0.8):
    """Find silences > min_dur in [start,end]. Returns list of (s,e) relative to start."""
    rc, _, err = run_cmd([
        "ffmpeg", "-v", "info", "-ss", str(start), "-t", str(end - start),
        "-i", video_path, "-af",
        "silencedetect=noise=%sdB:d=%s" % (thresh_db, min_dur),
        "-f", "null", "-"], timeout=300)
    sils = []
    for m in re.finditer(r"silence_start: ([\d.]+)", err):
        ss = float(m.group(1))
        me = re.search(r"silence_end: ([\d.]+)", err[m.end():])
        if me:
            sils.append((ss, float(me.group(1))))
    if sils:
        log(job_dir, "silences in clip: %s" %
            ", ".join("%.1f-%.1fs" % p for p in sils))
    return sils

def render_clip(video_path, words, moment, idx, job_dir, out_dir, style="hormozi"):
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
    # silence check: trim leading/trailing silence (timestamps relative to clip start)
    sils = detect_silences(video_path, s, e, job_dir)
    trim_s, trim_e = 0.0, 0.0
    for ss, se in sils:
        if ss < 0.5 and se - ss >= 0.8:
            trim_s = max(trim_s, se)
        if se >= dur - 0.5 and se - ss >= 0.8:
            trim_e = max(trim_e, dur - ss)
    if trim_s > 0:
        s += trim_s
        log(job_dir, "trimmed %.1fs leading silence" % trim_s)
    if trim_e > 0:
        e -= trim_e
        log(job_dir, "trimmed %.1fs trailing silence" % trim_e)
    dur = e - s
    if dur < 5:
        log(job_dir, "Clip %d: too short after silence trim (%.1fs)" % (idx + 1, dur))
        return False, "too short after trim"
    log(job_dir, "Clip %d: %.0fs-%.0fs (%s) size=%dx%d portrait=%s style=%s" %
        (idx + 1, s, e, moment["title"], vw, vh, is_portrait, style))
    base = "clip%d_%s" % (idx + 1, re.sub(r"\W+", "_", moment["title"])[:30])
    out = os.path.join(out_dir, base + ".mp4")
    # single ASS + SRT for the whole clip
    ass = os.path.join(job_dir, "clip%d.ass" % idx)
    write_ass(job_dir, words, s, e, ass, style=style)
    srt_path = os.path.join(out_dir, base + ".srt")
    write_srt(words, s, e, srt_path)
    ass_esc = ass.replace("'", r"'\''").replace(":", r"\:")
    sub = "subtitles='%s':fontsdir='%s'" % (ass_esc, _FONTS)
    if is_portrait:
        vf = "scale=1080:1920,%s" % sub
        rc, _, err = run_cmd(["ffmpeg", "-y", "-ss", str(s), "-t", str(dur),
                              "-i", video_path, "-vf", vf,
                              "-c:v", "libx264", "-preset", "fast", "-crf", "19",
                              "-c:a", "aac", "-b:a", "160k", out])
    else:
        # smooth face-tracked crop in ONE pass via sendcmd (0.5s samples)
        track = face_track_x(video_path, s, e, vw, job_dir, step=0.5)
        cmds = os.path.join(job_dir, "clip%d.cmds" % idx)
        with open(cmds, "w") as f:
            for tt, x in track:
                cx = int(max(0, min(vw - cw, x - cw / 2)))
                f.write("%.1f crop x %d;\n" % (tt, cx))
        init_cx = int(max(0, min(vw - cw, (track[0][1] if track else vw / 2) - cw / 2)))
        vf = ("sendcmd=f='%s',crop=%d:%d:x=%d:y=0,scale=1080:1920,%s" %
              (cmds.replace("'", r"'\''"), cw, vh, init_cx, sub))
        rc, _, err = run_cmd(["ffmpeg", "-y", "-ss", str(s), "-t", str(dur),
                              "-i", video_path, "-vf", vf,
                              "-c:v", "libx264", "-preset", "fast", "-crf", "19",
                              "-c:a", "aac", "-b:a", "160k", out])
        try:
            os.remove(cmds)
        except Exception:
            pass
        if (rc != 0 or not os.path.exists(out)) and os.path.exists(ass):
            log(job_dir, "clip %d subtitles failed, retry plain" % (idx + 1))
            vf_plain = "crop=%d:%d:%d:0,scale=1080:1920" % (cw, vh, init_cx)
            rc, _, err = run_cmd(["ffmpeg", "-y", "-ss", str(s), "-t", str(dur),
                                  "-i", video_path, "-vf", vf_plain,
                                  "-c:v", "libx264", "-preset", "fast", "-crf", "19",
                                  "-c:a", "aac", "-b:a", "160k", out])
    if rc != 0 or not os.path.exists(out):
        log(job_dir, "clip render failed: %s" % err[-1500:])
        return False, "render failed"
    try:
        os.remove(ass)
    except Exception:
        pass
    return True, out

# ---------------- full job ----------------
def run_job(job_dir, video_path, cfg):
    os.makedirs(job_dir, exist_ok=True)
    out_dir = os.path.join(job_dir, "clips")
    os.makedirs(out_dir, exist_ok=True)
    open(os.path.join(job_dir, "log.txt"), "w").write("")
    provider = cfg.get("provider", "openai")
    groq_key = cfg.get("groq_key", "")
    openai_key = cfg.get("openai_key", "")
    llm_key = openai_key if provider == "openai" else groq_key
    stt_key = openai_key if provider == "openai" else groq_key
    if not llm_key:
        return False, "API key missing"
    ok, words = transcribe(provider, stt_key, video_path, job_dir)
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
    style = cfg.get("caption_style", "hormozi")
    for i, mo in enumerate(moments):
        ok, out = render_clip(video_path, words, mo, i, job_dir, out_dir, style=style)
        if ok:
            base = os.path.splitext(os.path.basename(out))[0]
            done.append({"file": os.path.basename(out), "title": mo["title"],
                         "score": mo["score"], "start": mo["start"], "end": mo["end"],
                         "srt": base + ".srt"})
        else:
            log(job_dir, "clip %d failed: %s" % (i + 1, out))
    open(os.path.join(job_dir, "done.json"), "w", encoding="utf-8").write(
        json.dumps(done, ensure_ascii=False, indent=1))
    log(job_dir, "DONE: %d clips" % len(done))
    return True, done
