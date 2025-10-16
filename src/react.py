# react.py
import asyncio
import inspect
from models.water_qwen import WaterQwen
from configs.water_config import WaterConfig
from configs.generation_config import GenerationConfig
import numpy as np

def constant_sparsity_function(codeword_len):
    return 1

encoding_noise_rate = 0.0
majority_encoding_rate = 3
key_dir = "keys"
water_config = WaterConfig(constant_sparsity_function, encoding_noise_rate, majority_encoding_rate, key_dir)

gen_config = GenerationConfig(
    temperature = 1.0, top_p = 0.9, repetition_penalty = 1.2,
    no_repeat_ngram_size=3, token_buffer = 100,
    skip_special_tokens=True, add_special_tokens=False
)

model = WaterQwen("Qwen", gen_config, water_config)

def generate(prompt, word_count, is_watermarked, temperature):
    model.set_temperature(temperature)
    return model.generate(prompt, word_count, is_watermarked)

async def generate_stream(story, word_count, is_watermarked, temperature, delay_s: float = 0.00):
    """
    Stream text fragments. Tries a real token stream if the model exposes one,
    otherwise falls back to splitting a full generation.
    """
    model.set_temperature(temperature)

    # Probe for a streaming generator on the wrapper or inner model
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
    # --- inside generate_stream(...) after you find `candidate` ---

    if candidate is not None:
        # Call with/without watermark flag (some impls accept 2 args)
        try:
            res = candidate(story, word_count, is_watermarked)
        except TypeError:
            res = candidate(story, word_count)

        # Handle async generator OR async iterable
        if inspect.isasyncgen(res) or hasattr(res, "__aiter__"):
            async for frag in res:
                if frag:  # avoid empty chunks
                    yield frag if isinstance(frag, str) else str(frag)
                    # allow loop to flush even if delay_s==0
                    await asyncio.sleep(delay_s if delay_s else 0)
            return

        # Handle sync generator OR sync iterable
        if inspect.isgenerator(res) or hasattr(res, "__iter__"):
            # Beware: strings are iterable char-by-char
            if isinstance(res, str):
                if res:
                    yield res
                    await asyncio.sleep(delay_s if delay_s else 0)
                return
            for frag in res:
                if frag:  # avoid empty chunks
                    yield frag if isinstance(frag, str) else str(frag)
                    await asyncio.sleep(delay_s if delay_s else 0)
            return

        # If the "streamer" returned a full string once
        if isinstance(res, str):
            if res:
                yield res
                await asyncio.sleep(delay_s if delay_s else 0)
            return

    # Fallback: full generate, then stream by tokens/words
    msg = generate(story, word_count, is_watermarked, temperature)
    for tok in msg.split():
        yield tok + " "
        if delay_s:
            await asyncio.sleep(delay_s)

def probability_ai(response: str) -> float:
    # Return a float (no trailing comma)
    return float(model.detect_water(response))
