import asyncio
from models.water_qwen import WaterQwen
from configs.water_config import WaterConfig
from configs.generation_config import GenerationConfig
import numpy as np
from water_editor import WaterEditor
from pathlib import Path

def sparsity_function(codeword_len): 
    return int(np.log(codeword_len))

def constant_sparsity_function(codeword_len): 
    return 2

encoding_noise_rate =  0.0
majority_encoding_rate = 1
key_dir = "keys"

water_config = WaterConfig(constant_sparsity_function, encoding_noise_rate, majority_encoding_rate, key_dir)

temperature = 1.0
top_p = 0.9
repetition_penalty = 1.2
no_repeat_ngram_size = 3
token_buffer = 100
skip_special_tokens = True
add_special_tokens = False

gen_config = GenerationConfig(temperature, top_p, repetition_penalty, no_repeat_ngram_size, token_buffer, skip_special_tokens, add_special_tokens)

model_name = "Qwen"

model = WaterQwen(model_name, gen_config, water_config)

is_watermarked = False

def generate(prompt, word_count):
    # Backwards-compatible: return full string
    return model.generate(prompt, word_count, is_watermarked)

async def generate_stream(story, word_count, delay_s: float = 0.00):
    """
    Prefer true token streaming if available on the underlying model;
    otherwise, fall back to splitting the final string.
    """
    # Try common names for a streaming generator exposed by WaterFalcon or its inner LLM
    # e.g., generate_iter / generate_stream / iter_response
    # Also try to reach an inner llm attribute if present (llm / _llm / model).
    candidate = None
    for attr in ("generate_iter", "generate_stream", "iter_response"):
        if hasattr(model, attr):
            candidate = getattr(model, attr)
            break
    if candidate is None:
        inner = getattr(model, "llm", None) or getattr(model, "_llm", None) or getattr(model, "model", None)
        if inner is not None:
            for attr in ("generate_iter", "generate_stream", "iter_response"):
                if hasattr(inner, attr):
                    candidate = getattr(inner, attr)
                    break

    if callable(candidate):
        # True streaming path: consume the model's generator directly
        # candidate should be a sync generator that yields fragments (strings)
        for frag in candidate(story, word_count, is_watermarked):
            yield frag
            if delay_s:
                await asyncio.sleep(delay_s)
        return

    # Fallback: generate fully, then stream word-by-word
    message = generate(story, word_count)
    for word in message.split():
        yield word + " "
        if delay_s:
            await asyncio.sleep(delay_s)

def probability_ai(response: str) -> float:
    return model.detect_water(response)
