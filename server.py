# server.py
import argparse
import os
import sys
from pathlib import Path
from threading import Lock
from typing import List, Union

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pyngrok import ngrok

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import run_qwen_exaone_pipeline as pipeline

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"
DATA_DIR = BASE_DIR / "data"

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    persona: Union[int, str]
    brand: str
    product: str
    stage_index: int
    style_index: int
    is_event: int = 0
    top_k: int = 3
    qwen_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    exa_model: str = "LGAI-EXAONE/EXAONE-4.0-1.2B"
    disable_cache: bool = False
    n: int = 1


class BatchRequest(BaseModel):
    items: List[GenerateRequest]
    disable_cache: bool = False


_PIPELINE_LOCK = Lock()
_PIPELINE_CONTEXT = {}


def _get_context(qwen_model: str, exa_model: str, disable_cache: bool):
    if disable_cache:
        if hasattr(pipeline, "_set_cache_enabled"):
            pipeline._set_cache_enabled(False)
        if hasattr(pipeline, "load_json") and hasattr(pipeline.load_json, "cache_clear"):
            pipeline.load_json.cache_clear()
        return {"data": None, "q_generator": None, "exa_generator": None}

    if hasattr(pipeline, "_set_cache_enabled"):
        pipeline._set_cache_enabled(True)

    key = (qwen_model, exa_model)
    with _PIPELINE_LOCK:
        cached = _PIPELINE_CONTEXT.get(key)
        if cached:
            return cached
        base = Path(pipeline.__file__).resolve().parent.parent
        data = pipeline._load_data(str(base))
        q_generator = pipeline._get_qwen_generator(qwen_model)
        exa_generator = pipeline._get_exaone_generator(exa_model)
        cached = {"data": data, "q_generator": q_generator, "exa_generator": exa_generator}
        _PIPELINE_CONTEXT[key] = cached
        return cached


def _run_pipeline(req: GenerateRequest):
    args = argparse.Namespace(
        persona=req.persona,
        brand=req.brand,
        product=req.product,
        stage_index=req.stage_index,
        style_index=req.style_index,
        is_event=req.is_event,
        top_k=req.top_k,
        qwen_model=req.qwen_model,
        exa_model=req.exa_model,
        out_path=None,
        batch_json=None,
        disable_cache=req.disable_cache,
    )
    ctx = _get_context(req.qwen_model, req.exa_model, req.disable_cache)
    return pipeline._run_pipeline(
        args,
        data=ctx.get("data"),
        q_generator=ctx.get("q_generator"),
        exa_generator=ctx.get("exa_generator"),
    )


@app.post("/generate")
def generate(req: GenerateRequest):
    try:
        if req.n <= 1:
            result = _run_pipeline(req)
            return {"result": result}
        results = []
        for _ in range(req.n):
            results.append(_run_pipeline(req))
        return {"results": results}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/generate_batch")
def generate_batch(req: BatchRequest):
    try:
        results = []
        for item in req.items:
            if req.disable_cache:
                item.disable_cache = True
            results.append(_run_pipeline(item))
        return {"results": results}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


app.mount("/data", StaticFiles(directory=str(DATA_DIR)), name="data")
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    ngrok_token = os.getenv("NGROK_AUTH_TOKEN")
    if ngrok_token:
        ngrok.set_auth_token(ngrok_token)
    public_url = ngrok.connect(port)
    print(f"ngrok tunnel: {public_url}")
    uvicorn.run(app, host="0.0.0.0", port=port)
