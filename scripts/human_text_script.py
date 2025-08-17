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

temperature = 0.8
top_p = 0.9
repetition_penalty = 1.2
no_repeat_ngram_size = 3
token_buffer = 3
skip_special_tokens = True
add_special_tokens = False

gen_config = GenerationConfig(temperature, top_p, repetition_penalty, no_repeat_ngram_size, token_buffer, skip_special_tokens, add_special_tokens)

model_name = "DefaultFalcon"

default_falcon = WaterFalcon(model_name, gen_config, water_config)


num_words = 20             


repo_root = Path(__file__).resolve().parents[1] 
in_dir = repo_root / "input" 
in_dir.mkdir(parents = True, exist_ok=True)


falcon_editor = WaterEditor(default_falcon, in_dir)

path_to_response = in_dir / "human_text.txt"

falcon_editor.detect_content(path_to_response)



