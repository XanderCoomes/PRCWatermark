from models.water_qwen import WaterQwen
from configs.water_config import WaterConfig
from configs.generation_config import GenerationConfig
import numpy as np
import math
from water_editor import WaterEditor
from pathlib import Path


def sparsity_function(codeword_len): 
    return int(math.log10(codeword_len))

def constant_sparsity_function(codeword_len): 
    return 1

encoding_noise_rate =  0.0
majority_encoding_rate = 1
key_dir = "keys"

water_config = WaterConfig(sparsity_function, encoding_noise_rate, majority_encoding_rate, key_dir)

temperatures = np.arange(1.0, 2.0, 0.1)
encoding_error_rate_sum = np.zeros_like(temperatures)
top_p = 0.8
repetition_penalty = 1.2
no_repeat_ngram_size = 3
token_buffer = 0
skip_special_tokens = True
add_special_tokens = False

trials = 100
for trial in range (trials):
    
    model_name = "QwenEncodingErrorCalculations"
    for i, temp in enumerate(temperatures): 
        print("Trial: ", trial)
        print("Temperature: ", temp)
        gen_config = GenerationConfig(temp, top_p, repetition_penalty, no_repeat_ngram_size, token_buffer, skip_special_tokens, add_special_tokens)

        qwen = WaterQwen(model_name, gen_config, water_config)

        num_words = 100    


        repo_root = Path(__file__).resolve().parents[1] 
        out_dir = repo_root / "output" 
        out_dir.mkdir(parents = True, exist_ok=True)

        editor = WaterEditor(qwen, out_dir)


        prompt = "Write an essay on the use of AI in education"
        is_watermarked = True

        response, encoding_error_rate = qwen.generate_response_with_error_rate(prompt, num_words, is_watermarked)
        encoding_error_rate_sum[i] += encoding_error_rate
        print()
        print("Encoding Error Rate: ", encoding_error_rate)
        nuanced_prompt = "Temperature: " + str(temp) + prompt
        editor.save_response(nuanced_prompt, num_words, response)

        path_to_response = editor.get_out_path(out_dir, prompt, num_words)
    print("Encoding Error Rate Summary: ")
    encoding_error_rate = {x / (trial + 1) for x in encoding_error_rate_sum}
    for i, error in enumerate(encoding_error_rate): 
        print("Temp:", temperatures[i], "Encoding Error Rate", round(error, 2))

print()

print("Final Error Summary: ")
print("Temperatures")
for temp in temperatures: 
    print(temp)

print("Encoding Error Rates")
for error in encoding_error_rate: 
    print(error)
