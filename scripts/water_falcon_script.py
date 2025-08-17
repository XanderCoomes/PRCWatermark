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


num_words = 300  


repo_root = Path(__file__).resolve().parents[1] 
out_dir = repo_root / "output" 
out_dir.mkdir(parents = True, exist_ok=True)

falcon_editor = WaterEditor(default_falcon, out_dir)


prompt = "Write an essay about a historic murder"
is_watermarked = True

response = default_falcon.generate(prompt, num_words, is_watermarked)
falcon_editor.save_response(prompt, num_words, response)

path_to_response = falcon_editor.get_out_path(out_dir, prompt, num_words)
falcon_editor.edit_detect_loop(path_to_response)



