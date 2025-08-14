from models.water_calme import WaterCalme
from configs.water_config import WaterConfig
from configs.generation_config import GenerationConfig
import numpy as np


def sparsity_function(codeword_len): 
    return int(np.log(codeword_len))

def constant_sparsity_function(codeword_len): 
    return 1

encoding_noise_rate =  0.00
majority_encoding_rate = 3
key_dir = "keys"

water_config = WaterConfig(sparsity_function, encoding_noise_rate, majority_encoding_rate, key_dir)

temperature = 2.5
top_p = 0.9
repetition_penalty = 1.2
no_repeat_ngram_size = 3
token_buffer = 5
skip_special_tokens = False
add_special_tokens = False

gen_config = GenerationConfig(temperature, top_p, repetition_penalty, no_repeat_ngram_size, token_buffer, skip_special_tokens, add_special_tokens)


model_name = "DefaultCalme"

default_llama = WaterCalme(model_name, gen_config, water_config)


prompt = "Write something about the political state we are in right now"
num_words = 20
is_watermarked = True

default_llama.generate(prompt, num_words, is_watermarked)
