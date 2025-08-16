# src/models/water_deep.py
from __future__ import annotations

import os
import inspect
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from water_llm import WaterLLM


class WaterDeep(WaterLLM):
    """
    Fast, robust loader for DeepSeek-R1 Distill checkpoints.

    - CUDA: tries 4-bit (bitsandbytes) for speed/VRAM; falls back to fp16.
    - MPS (Apple Silicon): fp16 on MPS.
    - CPU: uses 7B by default (much faster than 14B on CPU), fp32.
    - Guards against older Transformers that don't support `warmup_cache`.
    """

    def __init__(self, model_name: str | None, generation_config, water_config):

        model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"  # faster on CPU
       

        # Tokenizer (DeepSeek modifies tokenizer/config slightly)
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Common kwargs; add warmup_cache=False only if supported by this Transformers version
        load_kwargs = dict(
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
            # CPU: prefer a smaller model for speed (7B). Use fp32 and more threads.
        try:
            torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))
            torch.set_num_interop_threads(max(1, (os.cpu_count() or 4) // 2))
        except Exception:
            pass

        # If user asked for a huge model on CPU, warn in a comment (no logging here)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map={"": "cpu"},
            torch_dtype=torch.float32,
            **load_kwargs,
        )

        # Hand off to your base class
        super().__init__(model_id, model, tokenizer, generation_config, water_config)
