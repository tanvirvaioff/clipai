#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ClipAI — OpusClip-style AI clipping web app."""
import os, json, uuid, threading, time
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

def job_worker(jid, video_path, cfg):
    jd = os.path.join(JOBS, jid)
    _write_status(jid, {"status": "running", "progress": "starting…"})
    try:
        ok, res = pipeline.run_job(jd, video_path, cfg)
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
    youtube_url = request.form.get("youtube_url", "").strip()
    f = request.files.get("video")
    jid = uuid.uuid4().hex[:12]
    jd = os.path.join(JOBS, jid)
    os.makedirs(jd, exist_ok=True)
    if youtube_url:
        pipeline.log(jd, "Downloading from YouTube…")
        vp, err = download_youtube(youtube_url, jd)
        if not vp:
            return "YouTube download failed: %s" % err, 400
    else:
        if not f or not f.filename:
            return "No video uploaded or YouTube URL given", 400
        fn = secure_filename(f.filename)
        vp = os.path.join(UPLOAD, jid + "_" + fn)
        f.save(vp)
    cfg = {
        "provider": request.form.get("provider", "openai"),
        "groq_key": request.form.get("groq_key", "").strip(),
        "openai_key": request.form.get("openai_key", "").strip(),
        "n_clips": int(request.form.get("n_clips", 3)),
        "min_len": int(request.form.get("min_len", 20)),
        "max_len": int(request.form.get("max_len", 50)),
        "caption_style": request.form.get("caption_style", "hormozi"),
    }
    # server default keys fallback
    if not cfg["groq_key"]:
        cfg["groq_key"] = os.environ.get("GROQ_KEY", "")
    if not cfg["openai_key"]:
        cfg["openai_key"] = os.environ.get("OPENAI_KEY", "")
    th = threading.Thread(target=job_worker, args=(jid, vp, cfg), daemon=True)
    th.start()
    return redirect(url_for("job", jid=jid))

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
