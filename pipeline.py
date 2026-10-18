#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ClipAI pipeline: transcribe -> viral moments -> face-tracked 9:16 clips + karaoke captions."""
import os, json, re, math, subprocess, tempfile, threading, requests
import cv2
import mediapipe as mp

GROQ_BASE = "https://api.groq.com/openai/v1"
OPENAI_BASE = "https://api.openai.com/v1"
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"

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

def video_duration(path):
    rc, out, _ = run_cmd(["ffprobe", "-v", "error",
                          "-show_entries", "format=duration",
                          "-of", "csv=p=0", path], timeout=60)
    try:
        return float(out.strip().split("\n")[0])
    except Exception:
        return 0.0

# ---------------- transcription (Groq Whisper free / OpenAI Whisper paid) ----------------
def transcribe(provider, api_key, video_path, job_dir, language="en"):
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
                      "language": language,
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
MOMENT_PROMPT = """You are an elite viral short-form video editor who has created 100M+ view clips. Given a transcript with [start-end sec] segments, pick the {n} MOST VIRAL moments for TikTok/Reels/Shorts.

VIRALITY RULES (follow strictly):
1. HOOK IS EVERYTHING: The first 1-3 seconds MUST have a pattern interrupt — a shocking statement, bold claim, unexpected question, or emotional peak. If the opening is boring, REJECT the moment.
2. NO BUILDUP: Start AT the interesting part, not before. Cut all throat-clearing, "so basically", "um", introductions.
3. EMOTIONAL SPIKES: Prioritize moments with strong emotion — anger, surprise, laughter, controversy, "no way" reactions, mic-drop statements.
4. COMPLETE PAYOFF: Each clip must have a satisfying ending — a punchline, revelation, conclusion, or cliffhanger. Never end mid-thought.
5. SHAREABILITY: Ask "would someone send this to a friend?" If no, reject.
6. REWATCHABILITY: The best clips make people watch twice — dense with meaning, fast-paced, quotable lines.
7. CONTROVERSY & HOT TAKES: Bold opinions, unpopular truths, myth-busting, "everyone is wrong about X" — these go viral.
8. Each moment {lo}-{hi} seconds, must be a complete thought (don't cut mid-sentence)
9. Moments must not overlap

SCORING (be harsh — most content is NOT viral):
- 95-100: Would get 1M+ views. Shocking hook, perfect payoff.
- 85-94: Strong viral potential. Great hook, solid payoff.
- 75-84: Good clip, might do well.
- Below 75: DO NOT SELECT.

For each moment give 4 sub-scores (0-25 each, MUST sum exactly to the total score):
  - hook: does the first 3 seconds GRAB attention? (be strict)
  - flow: tight pacing with satisfying payoff?
  - value: emotional impact, quotability, shareability?
  - trend: controversial/bold/relatable angle?
- In the "hook" field, wrap 1-4 EMPHASIS words in *asterisks* (e.g. "this is *insane* and totally *unbelievable*") — the punchiest high-energy words that deserve on-screen highlight.
- Respond ONLY with JSON: [{{"start": 12.5, "end": 47.0, "score": 95, "subs": {{"hook": 24, "flow": 23, "value": 25, "trend": 23}}, "title": "short punchy title (under 8 words, clickbait style)", "hook": "the exact opening line"}}]

Transcript:
{transcript}
"""

def find_moments(provider, api_key, words, n_clips, lo, hi, job_dir,
                 extra_prompt="", genre="Auto"):
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
    _extra = []
    if genre and genre != "Auto":
        _extra.append("Video genre: %s — prefer moments that fit this genre." % genre)
    if extra_prompt and extra_prompt.strip():
        _extra.append("The user specifically wants these moments included: %s. "
                      "Prioritize segments matching this request." % extra_prompt.strip()[:300])
    if _extra:
        prompt = prompt.replace("Transcript:\n", "\n".join(_extra) + "\n\nTranscript:\n")
    log(job_dir, "AI finding viral moments (%s)…" % provider)
    if provider == "openai":
        url = OPENAI_BASE + "/chat/completions"
        model = "gpt-4o"
        key = api_key
    elif provider == "gemini":
        url = GEMINI_BASE + "/models/gemini-3-flash-preview:generateContent?key=" + api_key
        model = "gemini-3-flash-preview"
        key = api_key
    else:
        url = GROQ_BASE + "/chat/completions"
        model = "openai/gpt-oss-120b"
        key = api_key
    if provider == "gemini":
        payload = {"generationConfig": {"temperature": 0.7, "responseMimeType": "application/json"},
                   "contents": [{"parts": [{"text": "You are a viral video editor. Output JSON only.\n\n" + prompt}]}]}
        r = requests.post(url, headers={"Content-Type": "application/json"},
                          json=payload, timeout=300)
        try:
            j = r.json()
            txt = j["candidates"][0]["content"]["parts"][0]["text"]
        except Exception:
            return False, "LLM error: %s" % r.text[:300]
    else:
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
        # LLM-marked emphasis words: *word* in the hook -> on-screen highlight
        _hk_raw = str(mo.get("hook", ""))
        emph = []
        for _mm in re.findall(r"\*([^*]{1,40})\*", _hk_raw):
            for _w in _mm.split():
                _t = _w.lower().strip(".,!?;:'\"()[]-*")
                if _t and _t not in emph:
                    emph.append(_t)
        _hk = re.sub(r"\*([^*]+)\*", r"\1", _hk_raw)[:120]
        clean.append({"start": round(s, 2), "end": round(e, 2),
                      "score": int(mo.get("score", 80)),
                      "title": str(mo.get("title", "Clip"))[:60],
                      "hook": _hk,
                      "emphasis": emph,
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

# ---------------- face tracking (YOLOv8 primary + MediaPipe fallback) ----------------
_FONTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")

_YOLO = None
_YOLO_TRIED = False
_YOLO_LOCK = threading.Lock()

def _get_yolo():
    """Load YOLOv8n once (primary detector). None if ultralytics/torch missing."""
    global _YOLO, _YOLO_TRIED
    if _YOLO_TRIED:
        return _YOLO
    _YOLO_TRIED = True
    try:
        from ultralytics import YOLO
        # yolov8n.pt auto-downloads on first use (~6MB); pre-downloaded in Docker
        _YOLO = YOLO("yolov8n.pt")
    except Exception:
        _YOLO = None
    return _YOLO

def _detect_faces(frame, w, h, yolo, mp_detector):
    """Detect faces, returning [(x0, y0, bw, bh)] in source pixels.

    Primary: YOLOv8 person detection (very stable, no flicker) with the face
    estimated at the top-center of each person box — ideal for talking-head
    / podcast tracking. Fallback: MediaPipe face detection (true face boxes)
    when YOLO finds nobody or is unavailable.
    """
    boxes = []
    if yolo is not None:
        try:
            # downscale for speed, scale boxes back up
            th = 480
            tw = max(320, int(w * th / h))
            small = cv2.resize(frame, (tw, th))
            sx, sy = w / float(tw), h / float(th)
            with _YOLO_LOCK:
                r = yolo.predict(small, verbose=False, classes=[0],
                                 conf=0.35)[0]
            if r.boxes is not None:
                for b in r.boxes:
                    x0, y0, x1, y1 = b.xyxy[0].tolist()
                    x0, x1 = x0 * sx, x1 * sx
                    y0, y1 = y0 * sy, y1 * sy
                    pw, ph = x1 - x0, y1 - y0
                    if pw < 20 or ph < 20:
                        continue
                    fx, fy = (x0 + x1) / 2.0, y0 + 0.12 * ph
                    fw_, fh_ = 0.55 * pw, 0.38 * ph
                    boxes.append((fx - fw_ / 2.0, fy - fh_ / 2.0, fw_, fh_))
        except Exception:
            boxes = []
    if not boxes and mp_detector is not None:
        try:
            res = mp_detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        except Exception:
            res = None
        if res is not None and res.detections:
            for d in res.detections:
                bb = d.location_data.relative_bounding_box
                boxes.append((bb.xmin * w, bb.ymin * h,
                              bb.width * w, bb.height * h))
    return boxes

def _frame_hist_diff(prev_hist, frame):
    """Histogram chi-square difference between frames (0=same, large=different)."""
    hist = cv2.calcHist([frame], [0, 1, 2], None, [16, 16, 16], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    if prev_hist is None:
        return 0.0, hist
    diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CHISQR)
    return float(diff), hist


def face_track_x(video_path, start, end, vw, vh_src, job_dir, step=0.5):
    """Return smoothed face positions: list of (t_rel, n_faces, [(x, y), ...]).

    Tracks up to 4 faces (ranked by size, largest first). Rank 0 gets
    hysteresis + slew-rate limiting; ranks 1-3 get exponential follow.
    Extra faces must be >=25% of the largest face area to count.

    Improvements (OpenShorts-inspired):
    - scene cut detection: histogram comparison resets tracker on cuts
      (no smoothing across cuts — avoids "ghost pan" artifacts)
    - multi-face: up to 4 faces for split2/grid3/grid4 layouts
    - exponential smoothing (alpha=0.2): smooth, no jitter
    - hysteresis: rank-0 only switches faces after 2+ seconds consistency
    - no-face: KEEP last position (never jump to center)
    - detector: YOLOv8 person tracking (primary, ultra-stable) with
      MediaPipe face detection fallback
    """
    MAXF = 4
    MIN_RATIO = 0.25
    # Primary: YOLOv8 (stable person tracking, no flicker). Fallback:
    # MediaPipe full-range face detection (profile/side faces, low light).
    yolo = _get_yolo()
    mp_detector = mp.solutions.face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5)
    log(job_dir, "face detector: %s" %
        ("YOLOv8 + MediaPipe fallback" if yolo is not None else "MediaPipe"))
    cap = cv2.VideoCapture(video_path)
    # samples: (t_rel, n, [(x, y), ...], is_cut)
    samples = []
    prev_hist = None
    t = start
    while t < end:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            samples.append((t - start, 0, [], False))
            t += step
            continue
        h, w = frame.shape[:2]
        # scene cut detection (downscaled for speed)
        small = cv2.resize(frame, (64, 36))
        diff, prev_hist = _frame_hist_diff(prev_hist, small)
        is_cut = diff > 2.5  # tuned threshold for hard cuts
        det_boxes = _detect_faces(frame, w, h, yolo, mp_detector)
        raw = []
        if det_boxes:
            order = sorted(det_boxes, key=lambda b: b[2] * b[3], reverse=True)
            a0 = order[0][2] * order[0][3]
            for (x0, y0, bwf, bhf) in order[:MAXF]:
                if bwf * bhf < MIN_RATIO * a0:
                    break
                # detector boxes can extend past frame edges — clamp center
                cx = min(max(x0 + bwf / 2, 0), w)
                cy = min(max(y0 + bhf / 2, 0), h)
                raw.append((cx / w * vw, cy / h * vh_src))
        samples.append((t - start, len(raw), raw, is_cut))
        t += step
    cap.release()
    mp_detector.close()
    if not samples:
        return [(0.0, 0, [])]

    # Stage 1: hysteresis for rank 0, scene-cut resets, follow for ranks 1-3
    REGION_X = vw * 0.18
    REGION_Y = vh_src * 0.18
    HYST_N = max(2, int(2.0 / step) + 1)

    def _init(rank):
        r0 = samples[0][2]
        if rank < len(r0):
            return [r0[rank][0], r0[rank][1]]
        return [vw / 2, vh_src / 2]

    trk = [_init(r) for r in range(MAXF)]
    pend, pendy = [None, 0], [None, 0]
    targets = []  # (t_rel, n, [[x, y] x MAXF])
    for t_rel, n, raw, is_cut in samples:
        if is_cut:
            # SCENE CUT: snap to new detections, clear pending
            for r in range(MAXF):
                if r < len(raw):
                    trk[r] = [raw[r][0], raw[r][1]]
            pend, pendy = [None, 0], [None, 0]
            targets.append((t_rel, n, [list(p) for p in trk]))
            continue
        if n == 0:
            # NO FACE: keep last position, do NOT drift to center
            pend, pendy = [None, 0], [None, 0]
            targets.append((t_rel, n, [list(p) for p in trk]))
            continue
        x, y = raw[0]
        tx, ty = trk[0]
        # rank-0 X hysteresis
        if abs(x - tx) <= REGION_X:
            trk[0][0] = 0.75 * tx + 0.25 * x
            pend = [None, 0]
        else:
            if pend[0] is None or abs(x - pend[0]) > REGION_X:
                pend = [x, 1]
            else:
                pend[1] += 1
                pend[0] = 0.75 * pend[0] + 0.25 * x
            if pend[1] >= HYST_N:
                trk[0][0] = pend[0]
                pend = [None, 0]
        # rank-0 Y hysteresis
        if abs(y - ty) <= REGION_Y:
            trk[0][1] = 0.75 * ty + 0.25 * y
            pendy = [None, 0]
        else:
            if pendy[0] is None or abs(y - pendy[0]) > REGION_Y:
                pendy = [y, 1]
            else:
                pendy[1] += 1
                pendy[0] = 0.75 * pendy[0] + 0.25 * y
            if pendy[1] >= HYST_N:
                trk[0][1] = pendy[0]
                pendy = [None, 0]
        # ranks 1-3: exponential follow (grid layouts use averaged positions)
        for r in range(1, MAXF):
            if r < len(raw):
                trk[r][0] = 0.8 * trk[r][0] + 0.2 * raw[r][0]
                trk[r][1] = 0.8 * trk[r][1] + 0.2 * raw[r][1]
        targets.append((t_rel, n, [list(p) for p in trk]))

    # Stage 2: slew-rate limit (rank 0) + EXPONENTIAL smoothing (all ranks)
    MAX_DX = vw * 0.5 * step
    MAX_DY = vh_src * 0.5 * step
    ALPHA = 0.2
    sm = []
    prev = [list(p) for p in targets[0][2]]
    for t_rel, n, tpos in targets:
        cur = []
        for r in range(MAXF):
            x, y = tpos[r]
            px, py = prev[r]
            if r == 0:
                dx, dy = x - px, y - py
                if abs(dx) > MAX_DX:
                    x = px + math.copysign(MAX_DX, dx)
                if abs(dy) > MAX_DY:
                    y = py + math.copysign(MAX_DY, dy)
            cur.append((ALPHA * x + (1 - ALPHA) * px,
                        ALPHA * y + (1 - ALPHA) * py))
        sm.append((t_rel, n, cur))
        prev = [list(p) for p in cur]
    return sm


def grid_layout(track, step=0.5):
    """Detect multi-face scenes and pick a layout.

    Returns (mode, positions):
      mode: 'single' | 'split2' | 'grid3' | 'grid4'
      positions: averaged (x, y) per face in size-rank order.
    A face count k dominates when >=60% of samples show >=k faces for >3s.
    split2 additionally requires the two faces to be side-by-side
    (horizontal separation > vertical), otherwise falls back to single.
    """
    if not track:
        return "single", []
    total = len(track)
    at_least = {2: 0, 3: 0, 4: 0}
    pos_sum = [[0.0, 0.0] for _ in range(4)]
    pos_cnt = [0, 0, 0, 0]
    for _t, n, faces in track:
        for k in (2, 3, 4):
            if n >= k:
                at_least[k] += 1
        for i, (fx, fy) in enumerate(faces[:4]):
            pos_sum[i][0] += fx
            pos_sum[i][1] += fy
            pos_cnt[i] += 1
    best = 1
    for k in (4, 3, 2):
        if at_least[k] >= 0.6 * total and at_least[k] * step >= 3.0:
            best = k
            break
    if best < 2:
        return "single", []
    positions = []
    for i in range(best):
        if pos_cnt[i] == 0:
            return "single", []
        positions.append((pos_sum[i][0] / pos_cnt[i],
                          pos_sum[i][1] / pos_cnt[i]))
    if best == 2:
        (ax1, ay1), (ax2, ay2) = positions
        # faces should be side-by-side (horizontal separation > vertical)
        if abs(ax1 - ax2) < abs(ay1 - ay2) * 0.5:
            return "single", []
        return "split2", positions
    return ("grid3" if best == 3 else "grid4"), positions


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
    # extra high-impact / emotional words (OpusClip-style emphasis)
    "epic", "legendary", "genius", "brilliant", "stupid", "dumb", "idiot",
    "powerful", "powerless", "dominate", "dominates", "destroy", "destroyed",
    "crush", "crushed", "exposed", "banned", "illegal", "million", "billionaire",
    "overnight", "instantly", "immediately", "finally", "actually", "literally",
    "honestly", "seriously", "desperate", "obsessed", "toxic", "redflag",
    "cheat", "cheating", "liar", "beg", "begging", "cry", "crying", "scream",
    "quit", "fired",
}

def _is_keyword(word, extra=None):
    """OpusClip-style keyword highlight heuristic.

    extra: set of lowercase LLM-marked emphasis words for this clip."""
    raw = word.strip()
    if not raw:
        return False
    t = raw.lower().strip(".,!?;:'\"()[]-*")
    if not t:
        return False
    if extra and t in extra:
        return True  # LLM-marked emphasis word
    if t in _KEYWORDS:
        return True
    if len(raw) >= 3 and raw.isupper():
        return True  # ALL CAPS emphasis in original speech
    if len(t) > 6 and t.isalpha():
        return True  # long significant words
    return False

def write_ass(job_dir, words, start, end, path, style="hormozi", max_chars=30,
              pos="middle", emphasis=None):
    """Caption styles: hormozi1 (gold karaoke dual-color), hormozi2 (gold karaoke),
    hormozi3 (inverted: yellow base, white tracker), beast (big yellow caps),
    leon (orange highlight on black boxes), minimal (small white).
    'hormozi' is an alias for hormozi1. pos: top / middle / bottom.
    emphasis: list of lowercase LLM-marked words -> green keyword highlight."""
    if style == "hormozi" or style == "auto":
        style = "hormozi1"
    extra_kw = set(w.lower() for w in (emphasis or []) if w)
    # caption placement: (ASS alignment, MarginV)
    _POS = {"top": (8, 150), "middle": (5, 30), "bottom": (2, 300)}
    align, marginv = _POS.get(pos, _POS["middle"])
    segs = [w for w in words if w["e"] > start and w["s"] < end]
    lines, cur = [], []
    _PL = {"hormozi1": 3, "hormozi2": 3, "hormozi3": 3, "beast": 3,
           "leon": 3, "minimal": 6}
    _CH = {"hormozi1": 30, "hormozi2": 30, "hormozi3": 30, "beast": 24,
           "leon": 28, "minimal": 42}
    per_line = _PL.get(style, 3)
    chars = _CH.get(style, 30)
    if style in ("hormozi1", "hormozi2", "hormozi3"):
        chars = max_chars
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
        # big yellow all-caps, thick black outline + soft drop shadow
        st = ("Style: Cap,Montserrat Black,100,&H0000FFFF,&H000000FF,&H00000000,&H00000000,"
              "-1,0,0,0,100,100,2,0,1,6,2,%d,10,10,%d,1" % (align, marginv))
        body = lambda ln: _ass_safe(" ".join(c["w"] for c in ln).upper())
        hl = False
    elif style == "minimal":
        # small white, no highlight
        st = ("Style: Cap,DejaVu Sans,56,&H00FFFFFF,&H000000FF,&H90000000,&H00000000,"
              "-1,0,0,0,100,100,0,0,1,3,0,%d,10,10,%d,1" % (align, marginv))
        body = lambda ln: _ass_safe(" ".join(c["w"] for c in ln))
        hl = False
    elif style == "leon":
        # Submagic LEON: bold orange highlight on opaque black boxes, karaoke
        st = ("Style: Cap,Montserrat Black,88,&H00FFFFFF&,&H00006BFF&,&H00000000&,&H00000000&,"
              "-1,0,0,0,100,100,0,0,1,6,0,%d,10,10,%d,1" % (align, marginv))
        hl = "leon"
    elif style == "hormozi2":
        # hormozi variant: white base, yellow tracker only (no keyword color)
        st = ("Style: Cap,Montserrat Black,100,&H00FFFFFF&,&H003DD9FF&,&H00000000&,&H80000000&,"
              "-1,0,0,0,100,100,2,0,1,8,2,%d,10,10,%d,1" % (align, marginv))
        hl = "plain"
    elif style == "hormozi3":
        # hormozi inverted: yellow base text, white tracker
        st = ("Style: Cap,Montserrat Black,100,&H003DD9FF&,&H00FFFFFF&,&H00000000&,&H80000000&,"
              "-1,0,0,0,100,100,2,0,1,8,2,%d,10,10,%d,1" % (align, marginv))
        hl = "plain"
    else:
        # hormozi: Montserrat Black, ALL CAPS, white base, yellow tracker,
        # green keyword highlight (OpusClip dual-color), 8px outline + shadow
        st = ("Style: Cap,Montserrat Black,100,&H00FFFFFF&,&H003DD9FF&,&H00000000&,&H80000000&,"
              "-1,0,0,0,100,100,2,0,1,8,2,%d,10,10,%d,1" % (align, marginv))
        hl = True
    out = ["[Script Info]", "ScriptType: v4.00+", "PlayResX: 1080", "PlayResY: 1920",
           "WrapStyle: 2", "", "[V4+ Styles]",
           "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
           st, "", "[Events]",
           "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"]
    YELLOW = r"{\c&H003DD9FF&}"   # tracker: currently-spoken word (#FFD93D)
    GREEN = r"{\c&H0014FF39&}"    # keyword highlight (#39FF14)
    WHITE = r"{\c&H00FFFFFF&}"    # base
    ORANGE = r"{\c&H00006BFF&}"   # leon orange (#FF6B00)
    for ln in lines:
        if not ln:
            continue
        if hl:
            for wi, w in enumerate(ln):
                wend = ln[wi + 1]["s"] if wi + 1 < len(ln) else w["e"] + 0.25
                parts = []
                for ci, c in enumerate(ln):
                    cw = _ass_safe(c["w"]).upper()
                    if hl == "leon":
                        # leon: active word orange on black box, rest white
                        col = ORANGE if ci == wi else WHITE
                    elif ci == wi:
                        # hormozi3 tracker is white (inverted), others yellow
                        col = WHITE if style == "hormozi3" else YELLOW
                    elif hl is True and _is_keyword(c["w"], extra_kw):
                        col = GREEN
                    else:
                        # hormozi3 base is yellow, others white
                        col = YELLOW if style == "hormozi3" else WHITE
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
                auto_zoom=True, caption_pos="middle", face_track=True):
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
    # single ASS + SRT for the whole clip ("none" = no burned captions)
    ass = os.path.join(job_dir, "clip%d.ass" % idx)
    no_cap = (style == "none")
    if not no_cap:
        write_ass(job_dir, words, s, e, ass, style=style, pos=caption_pos,
                  emphasis=moment.get("emphasis"))
    srt_path = os.path.join(out_dir, base + ".srt")
    write_srt(words, s, e, srt_path)
    if no_cap:
        sub = ""
    else:
        # copy ASS to a safe temp path (no special chars) for ffmpeg
        import shutil
        safe_ass = os.path.join("/tmp", "cap_%s.ass" % os.path.basename(job_dir))
        try:
            shutil.copyfile(ass, safe_ass)
        except Exception:
            safe_ass = ass
        sub = "subtitles=%s:fontsdir=%s" % (safe_ass, _FONTS)
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
        vf = ",".join(p for p in ["%s%s" % (prescale, zfilter), sub] if p)
        rc, _, err = run_cmd(["ffmpeg", "-y", "-i", video_path, "-ss", str(s), "-t", str(dur), "-vf", vf,
                              "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-tune", "film",
                              "-c:a", "aac", "-b:a", "160k", out])
    else:
        # smooth face-tracked crop in ONE pass via sendcmd (0.5s samples)
        # (face_track=False → static center crop)
        # MULTI-SPEAKER GRID: split2 (2 stacked) / grid3 (1+2) / grid4 (2x2)
        if face_track:
            track = face_track_x(video_path, s, e, vw, vh, job_dir, step=0.5)
        else:
            track = []
        mode, gpos = grid_layout(track) if track else ("single", [])
        if mode != "single":
            log(job_dir, "clip %d: %s mode (%d speakers)" % (idx + 1, mode.upper(), len(gpos)))
            # panel sizes (must tile exactly 1080x1920) and face ordering
            if mode == "split2":
                panels = [(1080, 960), (1080, 960)]          # left→top, right→bottom
                ordered = sorted(gpos, key=lambda p: p[0])
                stack = "[p0][p1]vstack=inputs=2"
            elif mode == "grid3":
                panels = [(1080, 960), (540, 960), (540, 960)]  # largest on top
                ordered = [gpos[0]] + sorted(gpos[1:], key=lambda p: p[0])
                stack = "[p1][p2]hstack=inputs=2[bottom];[p0][bottom]vstack=inputs=2"
            else:  # grid4: 2x2 reading order
                panels = [(540, 960)] * 4
                ordered = sorted(gpos, key=lambda p: (p[1], p[0]))
                stack = ("[p0][p1]hstack=inputs=2[top];"
                         "[p2][p3]hstack=inputs=2[bottom];"
                         "[top][bottom]vstack=inputs=2")
            # aspect-correct crop per panel (no stretch distortion)
            fc_parts = []
            for i, ((fx, _fy), (pw, ph)) in enumerate(zip(ordered, panels)):
                pcw = int(vh * pw / ph)
                pcw = max(1, min(vw, pcw))
                cx = int(max(0, min(vw - pcw, fx - pcw / 2)))
                fc_parts.append("[0:v]crop=%d:%d:%d:0,scale=%d:%d[p%d]"
                                % (pcw, vh, cx, pw, ph, i))
            fc = ";".join(fc_parts) + ";" + stack + ",grid_out"
            # apply prescale/zoom + subtitles after stacking
            post = ",".join(p for p in ["%s%s" % (prescale, zfilter), sub] if p)
            fc += "," + post if post else ",null"
            rc, _, err = run_cmd(["ffmpeg", "-y", "-i", video_path, "-ss", str(s), "-t", str(dur),
                                  "-filter_complex", fc,
                                  "-map", "[grid_out]", "-map", "0:a?",
                                  "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-tune", "film",
                                  "-c:a", "aac", "-b:a", "160k", out])
            if (rc != 0 or not os.path.exists(out)) and os.path.exists(ass):
                log(job_dir, "clip %d %s subtitles FAILED: %s" % (idx + 1, mode, err[-800:]))
                return False, "caption render failed: %s" % err[-300:]
        else:
            cmds = os.path.join(job_dir, "clip%d.cmds" % idx)
            with open(cmds, "w") as f:
                for tt, _n, faces in track:
                    x, y = faces[0] if faces else (vw / 2, vh / 2)
                    cx = max(0.0, min(float(vw - cw), x - cw / 2.0))
                    cy = max(0.0, min(float(vh - vh), y - vh / 2.0))
                    f.write("%.1f crop x %.2f;\n" % (tt, cx))
                    f.write("%.1f crop y %.2f;\n" % (tt, cy))
            _t0 = track[0][2][0] if (track and track[0][2]) else (vw / 2, vh / 2)
            t0x, t0y = _t0[0], _t0[1]
            init_cx = max(0.0, min(float(vw - cw), t0x - cw / 2.0))
            init_cy = max(0.0, min(float(vh - vh), t0y - vh / 2.0))
            vf = ",".join(p for p in [
                "sendcmd=f='%s'" % cmds.replace("'", r"'\''"),
                "crop=%d:%d:x=%.2f:y=%.2f" % (cw, vh, init_cx, init_cy),
                "%s%s" % (prescale, zfilter), sub] if p)
            rc, _, err = run_cmd(["ffmpeg", "-y", "-i", video_path, "-ss", str(s), "-t", str(dur), "-vf", vf,
                                  "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-tune", "film",
                                  "-c:a", "aac", "-b:a", "160k", out])
            try:
                os.remove(cmds)
            except Exception:
                pass
            if (rc != 0 or not os.path.exists(out)) and os.path.exists(ass):
                log(job_dir, "clip %d subtitles FAILED: %s" % (idx + 1, err[-800:]))
                # Don't silently retry without captions — fail loudly so we can fix it
                return False, "caption render failed: %s" % err[-300:]
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
    gemini_key = cfg.get("gemini_key", "")
    if provider == "openai":
        llm_key = openai_key
    elif provider == "gemini":
        llm_key = gemini_key
    else:
        llm_key = groq_key
    stt_key = openai_key if provider == "openai" else groq_key
    if not llm_key:
        return False, "API key missing"
    n_clips = cfg.get("n_clips", 3)
    caption_pos = cfg.get("caption_pos", "middle")
    auto_zoom = cfg.get("auto_zoom", True)
    face_track = cfg.get("face_track", True)
    speech_lang = cfg.get("speech_lang", "en")
    clip_mode = cfg.get("clip_mode", "ai")          # ai | manual ("Don't clip")
    auto_headline = cfg.get("auto_headline", True)
    moment_prompt = cfg.get("moment_prompt", "")
    genre = cfg.get("genre", "Auto")
    src_name = cfg.get("source_name", "video")
    # processing timeframe [t0, t1] in seconds (None = whole video)
    tf = cfg.get("timeframe")
    prog(5, "Analyzing content…", "Whisper AI is listening to your video…")
    ok, words = transcribe(provider, stt_key, video_path, job_dir, language=speech_lang)
    if not ok:
        return False, words
    if tf:
        try:
            t0, t1 = float(tf[0]), float(tf[1])
            words = [w for w in words if w["e"] > t0 and w["s"] < t1]
            log(job_dir, "timeframe %.0f-%.0fs: %d words" % (t0, t1, len(words)))
        except Exception:
            t0, t1 = 0.0, 0.0
    else:
        t0, t1 = 0.0, 0.0
    prog(40, "Analysis complete", "%d words found" % len(words))
    if not words:
        return False, "no speech found in timeframe"
    if clip_mode == "manual":
        # "Don't clip" — one clip covering the timeframe (or whole video)
        vd = video_duration(video_path)
        ms, me = (t0, t1 if t1 > t0 else (vd if vd > 0 else words[-1]["e"]))
        moments = [{"start": round(ms, 2), "end": round(me, 2), "score": 100,
                    "title": src_name[:60],
                    "hook": "",
                    "emphasis": [],
                    "subs": {"hook": 25, "flow": 25, "value": 25, "trend": 25}}]
        log(job_dir, "manual mode: full clip %.0f-%.0fs" % (ms, me))
    else:
        prog(42, "Finding viral moments…", "AI is scoring the best clips…")
        ok, moments = find_moments(provider, llm_key, words, n_clips,
                                   cfg.get("min_len", 20), cfg.get("max_len", 50),
                                   job_dir, extra_prompt=moment_prompt, genre=genre)
        if not ok:
            return False, moments
        if not moments:
            return False, "no moments found"
    if not auto_headline:
        for i, mo in enumerate(moments):
            mo["title"] = "Clip %d" % (i + 1)
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
                              caption_pos=caption_pos, face_track=face_track)
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
