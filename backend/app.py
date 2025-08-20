# server.py
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import os, sys, typing as t

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from react import generate, generate_stream, probability_ai

app = FastAPI()

# ------- Models -------
class CheckReq(BaseModel):
    story: str
    word_count: int
    temperature: float
    is_watermarked: bool

class CheckResp(BaseModel):
    result: str

class DetectReq(BaseModel):
    text: str

class DetectResp(BaseModel):
    prob: float  # 0..1

# ------- Routes -------
@app.post("/check", response_model = CheckResp)
def check(req: CheckReq):
    # order: (prompt, word_count, is_watermarked, temperature)
    out = generate(req.story, req.word_count, req.is_watermarked, req.temperature)
    return CheckResp(result=out)

@app.post("/check_stream")
async def check_stream(req: CheckReq):
    async def streamer() -> t.AsyncGenerator[str, None]:
        async for chunk in generate_stream(
            req.story, req.word_count, req.is_watermarked, req.temperature, delay_s=0.00
        ):
            yield chunk
    return StreamingResponse(
        streamer(),
        media_type="text/plain; charset=utf-8",
        headers={"Cache-Control": "no-cache"},
    )

@app.post("/detect", response_model=DetectResp)
def detect(req: DetectReq):
    p = probability_ai(req.text)
    return DetectResp(prob=p)
