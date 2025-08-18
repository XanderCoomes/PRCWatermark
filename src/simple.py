# simple.py
import asyncio
from models.water_falcon import WaterFalcon
from configs.water_config import WaterConfig
from configs.generation_config import GenerationConfig
import numpy as np
from water_editor import WaterEditor
from pathlib import Path


def sparsity_function(codeword_len): 
    return int(np.log(codeword_len))

def constant_sparsity_function(codeword_len): 
    return 1

encoding_noise_rate =  0.0
majority_encoding_rate = 3
key_dir = "keys"

water_config = WaterConfig(constant_sparsity_function, encoding_noise_rate, majority_encoding_rate, key_dir)

temperature = 1.0
top_p = 0.9
repetition_penalty = 1.2
no_repeat_ngram_size = 3
token_buffer = 3
skip_special_tokens = True
add_special_tokens = False

gen_config = GenerationConfig(temperature, top_p, repetition_penalty, no_repeat_ngram_size, token_buffer, skip_special_tokens, add_special_tokens)

model_name = "Falcon"

default_falcon = WaterFalcon(model_name, gen_config, water_config)

is_watermarked = True

def simple_function(prompt, word_count):
    return default_falcon.generate(prompt, word_count, is_watermarked)

async def simple_function_stream(story, word_count, delay_s: float = 0.12):
    """Yields the result word-by-word with a delay (used by /check_stream)."""
    message = simple_function(story, word_count)
    for word in message.split():
        yield word + " "
        await asyncio.sleep(delay_s)

def simple_probability_ai(story: str) -> float:
    """
    Very naive 'probability the text is AI':
    - If more than 10 words -> 0.8
    - Otherwise -> 0.2
    Returns a float in [0, 1].
    """
    s = (story or "").strip()
    if not s:
        return 0.0
    word_count = len(s.split())
    prob = 0.8 if word_count > 10 else 0.2
    return round(prob, 4)
