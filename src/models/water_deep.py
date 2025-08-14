# src/models/water_deep.py
from __future__ import annotations

import os
import inspect
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from models.water_llm import WaterLLM


class WaterDeep(WaterLLM):
    """
    Fast, robust loader for DeepSeek-R1 Distill checkpoints.

    - CUDA: tries 4-bit (bitsandbytes) for speed/VRAM; falls back to fp16.
    - MPS (Apple Silicon): fp16 on MPS.
    - CPU: uses 7B by default (much faster than 14B on CPU), fp32.
    - Guards against older Transformers that don't support `warmup_cache`.
    """

    def __init__(self, model_name: str | None, generation_config, water_config):
        has_cuda = torch.cuda.is_available()
        has_mps = getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()

        # Choose a sensible default if caller didn't supply a model name
        if model_name is None or model_name.strip() == "":
            if has_cuda or has_mps:
                model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
            else:
                model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"  # faster on CPU
        else:
            model_id = model_name

        # Tokenizer (DeepSeek modifies tokenizer/config slightly)
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Common kwargs; add warmup_cache=False only if supported by this Transformers version
        load_kwargs = dict(
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        if "warmup_cache" in inspect.signature(AutoModelForCausalLM.from_pretrained).parameters:
            load_kwargs["warmup_cache"] = False

        # Backend-specific loading
        if has_cuda:
            # Prefer 4-bit for speed/VRAM; fall back gracefully if bitsandbytes is missing
            try:
                import bitsandbytes as _bnb  # noqa: F401

                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    device_map="auto",
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    **load_kwargs,
                )
            except Exception:
                # Fallback: fp16 on GPU
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    **load_kwargs,
                )

            # Small CUDA perf tweaks (safe no-ops if unsupported)
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

        elif has_mps:
            # Apple Silicon (Metal)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map={"": "mps"},
                torch_dtype=torch.float16,
                **load_kwargs,
            )

        else:
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
