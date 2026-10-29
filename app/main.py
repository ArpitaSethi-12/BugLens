"""BugLens web server.

Run from the project folder:
    python -m uvicorn app.main:app --port 8000
Then open http://localhost:8000
"""
import glob
import json
import os
import re

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
load_dotenv(os.path.join(ROOT, ".env"))

from .detector import Detector  # noqa: E402
from .reviewer import Reviewer  # noqa: E402
from .samples import SAMPLES  # noqa: E402

app = FastAPI(title="BugLens")
detector = Detector()
reviewer = Reviewer()

# Published CodeXGLUE defect detection accuracy (Wang et al., CodeT5, EMNLP 2021, Table 4)
BASELINES = {"codebert": 0.6208, "codet5": 0.6340}

MAX_CHARS = 20000


class DetectRequest(BaseModel):
    code: str
    models: list[str] = ["codebert"]


class ReviewRequest(BaseModel):
    code: str
    language: str = "python"
    detection: dict | None = None


def guess_language(code):
    if re.search(r"#include|->|\bmalloc\(|\bprintf\(|;\s*$", code, re.M) and not re.search(r"^\s*def ", code, re.M):
        return "c"
    if re.search(r"\bfunction\b|=>|\bconst\b|\blet\b", code):
        return "javascript"
    if re.search(r"\bpublic\b|\bprivate\b|\bstatic void\b", code):
        return "java"
    return "python"


@app.get("/api/status")
def status():
    return {"detector": detector.status(), "reviewer": reviewer.status()}


@app.get("/api/samples")
def samples():
    return SAMPLES


@app.post("/api/detect")
def detect(req: DetectRequest):
    code = req.code
    if not code.strip():
        raise HTTPException(400, "Paste some code first.")
    if len(code) > MAX_CHARS:
        raise HTTPException(400, f"Code is too long ({len(code)} characters). Keep it under {MAX_CHARS}.")
    wanted = [m for m in req.models if m in detector.models]
    if not wanted:
        raise HTTPException(503, "No trained model is loaded. Put the trained model folders in models/ and restart.")
    return {"language": guess_language(code), "results": {m: detector.detect(m, code) for m in wanted}}


@app.post("/api/review")
def review(req: ReviewRequest):
    if not req.code.strip():
        raise HTTPException(400, "Paste some code first.")
    try:
        return reviewer.review(req.code, req.language, req.detection)
    except RuntimeError as e:
        raise HTTPException(502, str(e))


@app.get("/api/results")
def results():
    out = {}
    for path in sorted(glob.glob(os.path.join(ROOT, "results", "*.json"))):
        name = os.path.splitext(os.path.basename(path))[0]
        with open(path) as f:
            out[name] = json.load(f)
    return {"models": out, "baselines": BASELINES}


app.mount("/static", StaticFiles(directory=os.path.join(ROOT, "static")), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(ROOT, "static", "index.html"))
