from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import os, sys, typing as t

# Ensure we can import simple.py from project root
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from simple import (  # noqa: E402
    simple_function,
    simple_function_stream,
    simple_probability_ai,
)

app = FastAPI()

# ------- Models -------
class CheckReq(BaseModel):
    story: str
    word_count: int

class CheckResp(BaseModel):
    result: str

class DetectReq(BaseModel):
    text: str

class DetectResp(BaseModel):
    prob: float  # 0..1

# ------- Routes -------
@app.post("/check", response_model=CheckResp)
def check(req: CheckReq):
    return CheckResp(result=simple_function(req.story, req.word_count))

@app.post("/check_stream")
async def check_stream(req: CheckReq):
    async def streamer() -> t.AsyncGenerator[str, None]:
        async for chunk in simple_function_stream(req.story, req.word_count, delay_s=0.12):
            yield chunk
    return StreamingResponse(streamer(), media_type="text/plain; charset=utf-8")

@app.post("/detect", response_model=DetectResp)
def detect(req: DetectReq):
    """Runs the naive probability function and returns a number between 0 and 1."""
    p = simple_probability_ai(req.text)
    return DetectResp(prob=p)
