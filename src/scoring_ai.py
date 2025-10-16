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
majority_encoding_rate = 1
key_dir = "keys"

water_config = WaterConfig(constant_sparsity_function, encoding_noise_rate, majority_encoding_rate, key_dir)

temperature = 0.7
top_p = 0.8
repetition_penalty = 1.2
no_repeat_ngram_size = 3
token_buffer = 0
skip_special_tokens = True
add_special_tokens = False

gen_config = GenerationConfig(temperature, top_p, repetition_penalty, no_repeat_ngram_size, token_buffer, skip_special_tokens, add_special_tokens)

model_name = "falcon"

falcon = WaterFalcon(model_name, gen_config, water_config)

num_words = 7

is_watermarked = False

import json, re

def parse_score_from_json(text):
    s = text

    # 1) Try to decode the first JSON object anywhere in the string
    dec = json.JSONDecoder()
    for i, ch in enumerate(s):
        if ch == '{':
            try:
                obj, end = dec.raw_decode(s[i:])   # decodes until end of that JSON object
                if "score" in obj:
                    score = float(obj["score"])
                    if 0 <= score <= 10:           # optional bounds check
                        return score
                    return float(obj["score"])     # return anyway if you don't want bounds
            except json.JSONDecodeError:
                continue

    # 2) Fallback: direct regex for {"score": <num>}
    m = re.search(r'{"score"\s*:\s*(-?\d+(?:\.\d+)?)\s*}', s)
    if m:
        return float(m.group(1))
    #horrendously undreadable prompt? 
    return 0; 



def score_response(prompt, response): 
    grader_prompt = "You are a strict grader. Return exactly this JSON object and nothing else:\{score : <number from 0 to 10>\} Prompt:" + prompt + "Response:" + response

    grade = falcon.generate(grader_prompt, num_words, is_watermarked)

    return float(parse_score_from_json(grade)); 





