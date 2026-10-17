#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ClipAI — OpusClip-style AI clipping web app."""
import os, json, uuid, threading, time
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, request, render_template, jsonify, send_from_directory, redirect, url_for
from werkzeug.utils import secure_filename
import pipeline

BASE = os.path.dirname(os.path.abspath(__file__))
UPLOAD = os.path.join(BASE, "uploads")
JOBS = os.path.join(BASE, "jobs")
os.makedirs(UPLOAD, exist_ok=True)
os.makedirs(JOBS, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024  # 2GB

JOBS_STATE = {}
# bulk queue: max 2 videos processing at once (2 CPU cores)
POOL = ThreadPoolExecutor(max_workers=2)

def _status_path(jid):
    return os.path.join(JOBS, jid, "status.json")

def _write_status(jid, st):
    # disk is authoritative: survives gunicorn multi-worker routing
    try:
        with open(_status_path(jid), "w", encoding="utf-8") as f:
            json.dump(st, f, ensure_ascii=False)
    except Exception:
        pass

def _read_status(jid):
    p = _status_path(jid)
    if os.path.exists(p):
        try:
            return json.load(open(p, encoding="utf-8"))
        except Exception:
            pass
    return None

def download_youtube(url, jd):
    """Download video from YouTube URL with yt-dlp. Returns path or (None, err)."""
    out_tmpl = os.path.join(jd, "yt_source.%(ext)s")
    rc, so, se = pipeline.run_cmd([
        "yt-dlp", "-f", "bv*[height<=1080]+ba/b[height<=1080]",
        "--no-check-certificate", "--no-playlist",
        "-o", out_tmpl, url], timeout=1800)
    if rc != 0:
        return None, "yt-dlp failed: %s" % se[-300:]
    import glob
    got = glob.glob(os.path.join(jd, "yt_source.*"))
    if not got:
        return None, "yt-dlp produced no file"
    return got[0], None

def _meta_path(jid):
    return os.path.join(JOBS, jid, "meta.json")

def _write_meta(jid, meta):
    try:
        with open(_meta_path(jid), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)
    except Exception:
        pass

def _read_meta(jid):
    p = _meta_path(jid)
    if os.path.exists(p):
        try:
            return json.load(open(p, encoding="utf-8"))
        except Exception:
            pass
    return {}

def _new_job(name, source, youtube_url=None, video_path=None, cfg=None):
    jid = uuid.uuid4().hex[:12]
    jd = os.path.join(JOBS, jid)
    os.makedirs(jd, exist_ok=True)
    _write_meta(jid, {"name": name, "source": source, "youtube_url": youtube_url,
                      "video_path": video_path, "cfg": cfg or {},
                      "created": time.time()})
    _write_status(jid, {"status": "queued", "progress": 0,
                        "step": "Queued", "detail": "Waiting for a worker…"})
    return jid

def job_worker(jid):
    """Bulk-queue worker: downloads (if YouTube) then runs the pipeline."""
    jd = os.path.join(JOBS, jid)
    meta = _read_meta(jid)
    cfg = meta.get("cfg", {})
    video_path = meta.get("video_path")
    _write_status(jid, {"status": "running", "progress": 0,
                        "step": "Starting", "detail": "Preparing…"})
    def _prog(pct, step, detail=""):
        _write_status(jid, {"status": "running", "progress": pct,
                            "step": step, "detail": detail})
    try:
        if meta.get("youtube_url"):
            _prog(2, "Downloading video…", "Fetching from YouTube…")
            pipeline.log(jd, "Downloading from YouTube…")
            video_path, err = download_youtube(meta["youtube_url"], jd)
            if not video_path:
                raise RuntimeError("YouTube download failed: %s" % err)
        if not video_path or not os.path.exists(video_path):
            raise RuntimeError("video file missing")
        ok, res = pipeline.run_job(jd, video_path, cfg, progress_cb=_prog)
        st = {"status": "done" if ok else "error",
              "result": res if ok else None,
              "error": None if ok else str(res)}
        JOBS_STATE[jid] = st
        _write_status(jid, st)
    except Exception as e:
        st = {"status": "error", "error": str(e)[:500]}
        JOBS_STATE[jid] = st
        _write_status(jid, st)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/start", methods=["POST"])
def start():
    # BULK: multiple files + multiple YouTube URLs (one per line)
    files = [f for f in request.files.getlist("video") if f and f.filename]
    yt_urls = []
    for u in request.form.get("youtube_urls", "").replace(",", "\n").splitlines():
        u = u.strip()
        if u and u not in yt_urls:
            yt_urls.append(u)
    single_yt = request.form.get("youtube_url", "").strip()
    if single_yt and single_yt not in yt_urls:
        yt_urls.insert(0, single_yt)
    if not files and not yt_urls:
        return "No videos uploaded or YouTube URLs given", 400
    cfg = {
        "provider": request.form.get("provider", "openai"),
        "groq_key": request.form.get("groq_key", "").strip(),
        "openai_key": request.form.get("openai_key", "").strip(),
        "n_clips": int(request.form.get("n_clips", 3)),
        "min_len": int(request.form.get("min_len", 20)),
        "max_len": int(request.form.get("max_len", 50)),
        "caption_style": request.form.get("caption_style", "hormozi"),
        "caption_pos": request.form.get("caption_pos", "middle"),
        "auto_zoom": request.form.get("auto_zoom", "on") == "on",
    }
    # server default keys fallback
    if not cfg["groq_key"]:
        cfg["groq_key"] = os.environ.get("GROQ_KEY", "")
    if not cfg["openai_key"]:
        cfg["openai_key"] = os.environ.get("OPENAI_KEY", "")
    jids = []
    for f in files:
        fn = secure_filename(f.filename)
        jid = _new_job(fn, "upload", cfg=cfg)
        vp = os.path.join(UPLOAD, jid + "_" + fn)
        f.save(vp)
        meta = _read_meta(jid)
        meta["video_path"] = vp
        _write_meta(jid, meta)
        POOL.submit(job_worker, jid)
        jids.append(jid)
    for url in yt_urls:
        jid = _new_job(url[:60], "youtube", youtube_url=url, cfg=cfg)
        POOL.submit(job_worker, jid)
        jids.append(jid)
    if len(jids) == 1:
        return redirect(url_for("job", jid=jids[0]))
    return redirect(url_for("jobs"))

@app.route("/jobs")
def jobs():
    """Bulk dashboard: all jobs with status."""
    items = []
    if os.path.isdir(JOBS):
        for jid in os.listdir(JOBS):
            jd = os.path.join(JOBS, jid)
            if not os.path.isdir(jd):
                continue
            meta = _read_meta(jid)
            st = _read_status(jid) or JOBS_STATE.get(jid, {"status": "unknown"})
            nclips = 0
            try:
                res = st.get("result") or []
                nclips = len(res)
            except Exception:
                pass
            items.append({
                "jid": jid,
                "name": meta.get("name") or jid,
                "source": meta.get("source") or "?",
                "created": meta.get("created") or 0,
                "status": st.get("status", "unknown"),
                "progress": st.get("progress", 0),
                "step": st.get("step", ""),
                "nclips": nclips,
                "error": st.get("error", ""),
            })
    items.sort(key=lambda x: -x["created"])
    return render_template("jobs.html", jobs=items)

@app.route("/api/jobs")
def api_jobs():
    items = []
    if os.path.isdir(JOBS):
        for jid in os.listdir(JOBS):
            jd = os.path.join(JOBS, jid)
            if not os.path.isdir(jd):
                continue
            meta = _read_meta(jid)
            st = _read_status(jid) or JOBS_STATE.get(jid, {"status": "unknown"})
            try:
                nclips = len(st.get("result") or [])
            except Exception:
                nclips = 0
            items.append({
                "jid": jid,
                "name": meta.get("name") or jid,
                "source": meta.get("source") or "?",
                "created": meta.get("created") or 0,
                "status": st.get("status", "unknown"),
                "progress": st.get("progress", 0),
                "step": st.get("step", ""),
                "nclips": nclips,
                "error": st.get("error", ""),
            })
    items.sort(key=lambda x: -x["created"])
    return jsonify({"jobs": items})

@app.route("/job/<jid>")
def job(jid):
    return render_template("job.html", jid=jid)

@app.route("/api/job/<jid>")
def api_job(jid):
    st = _read_status(jid) or JOBS_STATE.get(jid, {"status": "unknown"})
    jd = os.path.join(JOBS, jid)
    lp = os.path.join(jd, "log.txt")
    log = ""
    if os.path.exists(lp):
        log = open(lp, encoding="utf-8").read()[-3000:]
    out = dict(st)
    out["log"] = log
    return jsonify(out)

@app.route("/dl/<jid>/<fn>")
def dl(jid, fn):
    return send_from_directory(os.path.join(JOBS, jid, "clips"), fn, as_attachment=True)

@app.route("/preview/<jid>/<fn>")
def preview(jid, fn):
    return send_from_directory(os.path.join(JOBS, jid, "clips"), fn)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), threaded=True)
