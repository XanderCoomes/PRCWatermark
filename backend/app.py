# backend/app.py
import os
from dataclasses import dataclass
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Make sure your src/ is importable (install with `pip install -e .`)
from transformers import AutoTokenizer, AutoModelForCausalLM

from water_llm import WaterLLM  # your class

# ---------------- Config dataclasses to satisfy WaterLLM ctor ----------------
@dataclass
class GenerationConfig:
    temperature: float = 0.8
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    no_repeat_ngram_size: int = 0
    token_buffer: int = 32
    skip_special_tokens: bool = True
    add_special_tokens: bool = False

@dataclass
class WaterConfig:
    sparsity_function: callable
    encoding_noise_rate: float = 0.0
    majority_encoding_rate: int = 3
    key_dir: str = "keys"

# Example sparsity function (adjust to your scheme)
def default_sparsity(n: int) -> Optional[int]:
    # return None or an int; keep it simple here
    return None

# ---------------- FastAPI app ----------------
app = FastAPI(title="WaterLLM API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GenReq(BaseModel):
    prompt: str
    num_words: int = 300
    is_watermarked: bool = False

class GenResp(BaseModel):
    text: str

# ---------------- Model load ----------------
MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen3-4B-Instruct-2507")
ATTN_IMPL = os.getenv("ATTN_IMPL")  # for Falcon set to "eager" or "flash_attention_2"

print(f"[backend] Loading model: {MODEL_ID}")
tok_kwargs = dict(trust_remote_code=True)
model_kwargs = dict(
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True,
)
if ATTN_IMPL:
    model_kwargs["attn_implementation"] = ATTN_IMPL

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, **tok_kwargs)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **model_kwargs)

# Falcon nicety (KV cache impl compatibility)
if ATTN_IMPL:
    model.config.attn_implementation = ATTN_IMPL

# Some tokenizers miss pad; align to eos
if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
    tokenizer.pad_token = tokenizer.eos_token

gen_cfg = GenerationConfig()
water_cfg = WaterConfig(
    sparsity_function=default_sparsity,
    encoding_noise_rate=float(os.getenv("ENCODING_NOISE", "0.0")),
    majority_encoding_rate=int(os.getenv("MAJORITY_L", "3")),
    key_dir=os.getenv("KEY_DIR", "keys"),
)

water_llm = WaterLLM(
    name=MODEL_ID,
    model=model,
    tokenizer=tokenizer,
    generation_config=gen_cfg,
    water_config=water_cfg,
)

@app.get("/health")
def health():
    return {"ok": True, "model": MODEL_ID}

@app.post("/api/generate", response_model=GenResp)
def generate(req: GenReq):
    text = water_llm.generate(req.prompt, req.num_words, req.is_watermarked)
    return GenResp(text=text)
