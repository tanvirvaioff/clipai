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

def video_fps(path):
    rc, out, _ = run_cmd(["ffprobe", "-v", "error", "-select_streams", "v:0",
                          "-show_entries", "stream=r_frame_rate",
                          "-of", "csv=p=0", path], timeout=60)
    try:
        num, den = out.strip().split("\n")[0].split("/")
        return float(num) / float(den) if float(den) else 30.0
    except Exception:
        return 30.0

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
- For each moment give 4 sub-scores (0-25 each, MUST sum exactly to the total score):
  - hook: does the intro grab attention in the first 3 seconds?
  - flow: does it flow logically with a satisfying conclusion?
  - value: does it offer value, resonate emotionally, create personal connection?
  - trend: is it aligned with current trends and audience interests?
- Respond ONLY with JSON: [{{"start": 12.5, "end": 47.0, "score": 95, "subs": {{"hook": 24, "flow": 23, "value": 25, "trend": 23}}, "title": "short punchy title", "hook": "first line that grabs attention"}}]

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
                      "hook": str(mo.get("hook", ""))[:120],
                      "subs": _parse_subs(mo)})
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

def _parse_subs(mo):
    """Parse hook/flow/value/trend sub-scores (0-25 each). Fall back to even split."""
    score = int(mo.get("score", 80))
    subs = mo.get("subs") or {}
    out = {}
    for k in ("hook", "flow", "value", "trend"):
        try:
            out[k] = max(0, min(25, int(subs.get(k, 0))))
        except Exception:
            out[k] = 0
    if sum(out.values()) <= 0:
        q, r = divmod(max(0, score), 4)
        out = {"hook": q + (1 if r > 0 else 0), "flow": q + (1 if r > 1 else 0),
               "value": q + (1 if r > 2 else 0), "trend": q}
    return out

# ---------------- face tracking (YuNet) ----------------
_YUNET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "yunet.onnx")
_FONTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")

def face_track_x(video_path, start, end, vw, job_dir, step=0.5):
    """Return smoothed face-center x positions: list of (time, x).

    Fixes for erratic jumping:
    - hysteresis: only switch to a NEW face region after it is the consistent
      largest face for 2+ seconds (avoids flicker between people)
    - no-face: KEEP last position (never jump to center/wide)
    - largest face preferred (closest to camera = likely speaker)
    - slew-rate limit: crop x interpolates, max ~0.8 frame-widths/sec
    """
    detector = cv2.FaceDetectorYN_create(_YUNET, "", (320, 320))
    cap = cv2.VideoCapture(video_path)
    samples = []  # (t_rel, x or None)
    t = start
    while t < end:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            samples.append((t - start, None))
            t += step
            continue
        h, w = frame.shape[:2]
        detector.setInputSize((w, h))
        _, faces = detector.detect(frame)
        if faces is not None and len(faces):
            # largest face: [x, y, w, h, ...] — closest to camera = speaker
            f = max(faces, key=lambda b: b[2] * b[3])
            x = float(f[0] + f[2] / 2) / w * vw
            samples.append((t - start, x))
        else:
            samples.append((t - start, None))
        t += step
    cap.release()
    if not samples:
        return [(0.0, vw / 2)]

    # Stage 1: hysteresis target selection
    REGION = vw * 0.18          # same-face region radius
    HYST_N = max(2, int(2.0 / step) + 1)  # 2+ full seconds of consistency to switch
    target = samples[0][1] if samples[0][1] is not None else vw / 2
    pending_x, pending_n = None, 0
    targets = []
    for t_rel, x in samples:
        if x is None:
            # NO FACE: keep last position, do NOT drift to center
            pending_x, pending_n = None, 0
            targets.append((t_rel, target))
            continue
        if abs(x - target) <= REGION:
            # same face region: follow smoothly
            target = 0.75 * target + 0.25 * x
            pending_x, pending_n = None, 0
        else:
            # candidate new face: require consistency before switching
            if pending_x is None or abs(x - pending_x) > REGION:
                pending_x, pending_n = x, 1
            else:
                pending_n += 1
                pending_x = 0.75 * pending_x + 0.25 * x
            if pending_n >= HYST_N:
                target = pending_x
                pending_x, pending_n = None, 0
        targets.append((t_rel, target))

    # Stage 2: slew-rate limit (~0.8 widths/sec) + moving average
    MAX_DX = vw * 0.8 * step
    limited = []
    prev = targets[0][1]
    for t_rel, x in targets:
        dx = x - prev
        if abs(dx) > MAX_DX:
            x = prev + math.copysign(MAX_DX, dx)
        limited.append((t_rel, x))
        prev = x
    sm = []
    for i, (t_rel, x) in enumerate(limited):
        win = limited[max(0, i - 2):i + 3]
        sm.append((t_rel, sum(p[1] for p in win) / len(win)))
    return sm

# ---------------- word-highlight captions (ASS) ----------------
def _ass_safe(t):
    """Escape ASS special chars in transcribed words."""
    return t.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")

# words that get OpusClip-style keyword highlight (green)
_KEYWORDS = {
    "money", "cash", "rich", "million", "billion", "dollar", "dollars", "profit",
    "profits", "free", "secret", "secrets", "amazing", "insane", "crazy", "viral",
    "hack", "hacks", "trick", "tricks", "tip", "tips", "never", "always", "stop",
    "warning", "shocking", "shocked", "exposed", "truth", "lies", "lie", "win",
    "winner", "lose", "loser", "fail", "failed", "success", "successful", "love",
    "hate", "best", "worst", "new", "proven", "guaranteed", "guarantee", "easy",
    "fast", "quick", "now", "today", "limited", "exclusive", "ultimate",
    "perfect", "wrong", "right", "why", "how", "wow", "omg", "unbelievable",
    "millionaire", "broke", "debt", "invest", "investing", "crypto", "stock",
    "business", "mistake", "mistakes", "lesson", "lessons", "fear", "scared",
    "dangerous", "killer", "dead", "death", "kill", "fight", "war", "police",
    "arrest", "caught", "busted", "scam", "fake", "real",
}

def _is_keyword(word):
    """OpusClip-style keyword highlight heuristic."""
    raw = word.strip()
    if not raw:
        return False
    t = raw.lower().strip(".,!?;:'\"()[]-")
    if not t:
        return False
    if t in _KEYWORDS:
        return True
    if len(raw) >= 3 and raw.isupper():
        return True  # ALL CAPS emphasis in original speech
    if len(t) > 6 and t.isalpha():
        return True  # long significant words
    return False

def write_ass(job_dir, words, start, end, path, style="hormozi", max_chars=30,
              pos="middle"):
    """Caption styles: hormozi (gold karaoke), beast (big yellow caps), minimal (small white).
    pos: top / middle / bottom caption placement."""
    # caption placement: (ASS alignment, MarginV)
    _POS = {"top": (8, 150), "middle": (5, 30), "bottom": (2, 300)}
    align, marginv = _POS.get(pos, _POS["middle"])
    segs = [w for w in words if w["e"] > start and w["s"] < end]
    lines, cur = [], []
    per_line = 3 if style == "hormozi" else (3 if style == "beast" else 6)
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
        st = ("Style: Cap,Montserrat Black,96,&H0000FFFF,&H000000FF,&H00000000,&H00000000,"
              "-1,0,0,0,100,100,0,0,1,6,0,%d,10,10,%d,1" % (align, marginv))
        body = lambda ln: _ass_safe(" ".join(c["w"] for c in ln).upper())
        hl = False
    elif style == "minimal":
        # small white, no highlight
        st = ("Style: Cap,DejaVu Sans,56,&H00FFFFFF,&H000000FF,&H90000000,&H00000000,"
              "-1,0,0,0,100,100,0,0,1,3,0,%d,10,10,%d,1" % (align, marginv))
        body = lambda ln: _ass_safe(" ".join(c["w"] for c in ln))
        hl = False
    else:
        # hormozi: Montserrat Black, ALL CAPS, white base, yellow tracker,
        # green keyword highlight (OpusClip dual-color), 8px outline
        st = ("Style: Cap,Montserrat Black,96,&H00FFFFFF&,&H003DD9FF&,&H00000000&,&H80000000&,"
              "-1,0,0,0,100,100,0,0,1,8,0,%d,10,10,%d,1" % (align, marginv))
        hl = True
    out = ["[Script Info]", "ScriptType: v4.00+", "PlayResX: 1080", "PlayResY: 1920",
           "WrapStyle: 2", "", "[V4+ Styles]",
           "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
           st, "", "[Events]",
           "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"]
    YELLOW = r"{\c&H003DD9FF&}"   # tracker: currently-spoken word (#FFD93D)
    GREEN = r"{\c&H0014FF39&}"    # keyword highlight (#39FF14)
    WHITE = r"{\c&H00FFFFFF&}"    # base
    for ln in lines:
        if not ln:
            continue
        if hl:
            for wi, w in enumerate(ln):
                wend = ln[wi + 1]["s"] if wi + 1 < len(ln) else w["e"] + 0.25
                parts = []
                for ci, c in enumerate(ln):
                    cw = _ass_safe(c["w"]).upper()
                    if ci == wi:
                        col = YELLOW
                    elif _is_keyword(c["w"]):
                        col = GREEN
                    else:
                        col = WHITE
                    parts.append("%s%s" % (col, cw))
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

def render_clip(video_path, words, moment, idx, job_dir, out_dir, style="hormozi",
                auto_zoom=True, caption_pos="middle"):
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
    write_ass(job_dir, words, s, e, ass, style=style, pos=caption_pos)
    srt_path = os.path.join(out_dir, base + ".srt")
    write_srt(words, s, e, srt_path)
    ass_esc = ass.replace("'", r"'\''").replace(":", r"\:")
    sub = "subtitles='%s':fontsdir='%s'" % (ass_esc, _FONTS)
    # auto-zoom: subtle 1.0 -> 1.08 push-in over the clip (Submagic-style)
    # zoompan needs oversized input, so prescale to 1296x2304 when zoom is on
    zfilter = ""
    prescale = "scale=1080:1920"
    if auto_zoom:
        fps = video_fps(video_path)
        frames = max(1, int(dur * fps))
        prescale = "scale=1296:2304"
        zfilter = (",zoompan=z='1+0.08*on/%d':x='iw/2-(iw/zoom/2)':"
                   "y='ih/2-(ih/zoom/2)':d=1:s=1080x1920" % frames)
    if is_portrait:
        vf = "%s%s,%s" % (prescale, zfilter, sub)
        rc, _, err = run_cmd(["ffmpeg", "-y", "-i", video_path, "-ss", str(s), "-t", str(dur), "-vf", vf,
                              "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
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
        vf = ("sendcmd=f='%s',crop=%d:%d:x=%d:y=0,%s%s,%s" %
              (cmds.replace("'", r"'\''"), cw, vh, init_cx, prescale, zfilter, sub))
        rc, _, err = run_cmd(["ffmpeg", "-y", "-i", video_path, "-ss", str(s), "-t", str(dur), "-vf", vf,
                              "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                              "-c:a", "aac", "-b:a", "160k", out])
        try:
            os.remove(cmds)
        except Exception:
            pass
        if (rc != 0 or not os.path.exists(out)) and os.path.exists(ass):
            log(job_dir, "clip %d subtitles failed, retry plain" % (idx + 1))
            vf_plain = "crop=%d:%d:%d:0,scale=1080:1920" % (cw, vh, init_cx)
            rc, _, err = run_cmd(["ffmpeg", "-y", "-i", video_path, "-ss", str(s), "-t", str(dur),
                                  "-vf", vf_plain,
                                  "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
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
def run_job(job_dir, video_path, cfg, progress_cb=None):
    os.makedirs(job_dir, exist_ok=True)
    out_dir = os.path.join(job_dir, "clips")
    os.makedirs(out_dir, exist_ok=True)
    open(os.path.join(job_dir, "log.txt"), "w").write("")

    def prog(pct, step, detail=""):
        if progress_cb:
            try:
                progress_cb(pct, step, detail)
            except Exception:
                pass
        log(job_dir, "[%d%%] %s %s" % (pct, step, detail))

    provider = cfg.get("provider", "openai")
    groq_key = cfg.get("groq_key", "")
    openai_key = cfg.get("openai_key", "")
    llm_key = openai_key if provider == "openai" else groq_key
    stt_key = openai_key if provider == "openai" else groq_key
    if not llm_key:
        return False, "API key missing"
    n_clips = cfg.get("n_clips", 3)
    caption_pos = cfg.get("caption_pos", "middle")
    auto_zoom = cfg.get("auto_zoom", True)
    prog(5, "Analyzing content…", "Whisper AI is listening to your video…")
    ok, words = transcribe(provider, stt_key, video_path, job_dir)
    if not ok:
        return False, words
    prog(40, "Analysis complete", "%d words found" % len(words))
    prog(42, "Finding viral moments…", "AI is scoring the best clips…")
    ok, moments = find_moments(provider, llm_key, words, n_clips,
                               cfg.get("min_len", 20), cfg.get("max_len", 50),
                               job_dir)
    if not ok:
        return False, moments
    if not moments:
        return False, "no moments found"
    open(os.path.join(job_dir, "moments.json"), "w", encoding="utf-8").write(
        json.dumps(moments, ensure_ascii=False, indent=1))
    prog(55, "Viral moments found", "%d moments selected" % len(moments))
    prog(57, "Adding captions…", "Writing animated captions for %d clips…" % len(moments))
    done = []
    style = cfg.get("caption_style", "hormozi")
    total = len(moments)
    import concurrent.futures as _cf
    _completed = [0]
    def _render_one(arg):
        i, mo = arg
        pct = 58 + int(37 * i / max(total, 1))
        prog(pct, "Reframing video…",
             'Clip %d of %d "%s" — face tracking + captions…'
             % (i + 1, total, mo.get("title", "")[:40]))
        ok, out = render_clip(video_path, words, mo, i, job_dir, out_dir,
                              style=style, auto_zoom=auto_zoom,
                              caption_pos=caption_pos)
        _completed[0] += 1
        if ok:
            base = os.path.splitext(os.path.basename(out))[0]
            prog(58 + int(37 * _completed[0] / max(total, 1)),
                 "Clip %d of %d done" % (_completed[0], total), "✓")
            return {"file": os.path.basename(out), "title": mo["title"],
                    "score": mo["score"], "subs": mo.get("subs", {}),
                    "start": mo["start"], "end": mo["end"],
                    "srt": base + ".srt"}
        else:
            log(job_dir, "clip %d failed: %s" % (i + 1, out))
            return None
    # render 2 clips in parallel (2 CPU cores)
    with _cf.ThreadPoolExecutor(max_workers=2) as _ex:
        for _r in _ex.map(_render_one, list(enumerate(moments))):
            if _r:
                done.append(_r)
    done.sort(key=lambda d: -d["score"])
    open(os.path.join(job_dir, "done.json"), "w", encoding="utf-8").write(
        json.dumps(done, ensure_ascii=False, indent=1))
    prog(98, "Finalizing…", "Preparing your clips…")
    log(job_dir, "DONE: %d clips" % len(done))
    prog(100, "Done", "%d clips ready 🎉" % len(done))
    return True, done
