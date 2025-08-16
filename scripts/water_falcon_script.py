from models.water_falcon import WaterFalcon
from configs.water_config import WaterConfig
from configs.generation_config import GenerationConfig
import numpy as np
import os
from pathlib import Path


def sparsity_function(codeword_len): 
    return int(np.log(codeword_len))

def constant_sparsity_function(codeword_len): 
    return 1

encoding_noise_rate =  0.05
majority_encoding_rate = 1
key_dir = "keys"

water_config = WaterConfig(constant_sparsity_function, encoding_noise_rate, majority_encoding_rate, key_dir)

temperature = 1.0
top_p = 0.9
repetition_penalty = 1.2
no_repeat_ngram_size = 3
token_buffer = 3
skip_special_tokens = False
add_special_tokens = False

gen_config = GenerationConfig(temperature, top_p, repetition_penalty, no_repeat_ngram_size, token_buffer, skip_special_tokens, add_special_tokens)

model_name = "DefaultFalcon"

default_falcon = WaterFalcon(model_name, gen_config, water_config)


prompt = "Write a tale of two sitting ducks"
num_words = 50                  # give it room to write
is_watermarked = True

response = default_falcon.generate(prompt, num_words, is_watermarked)

# --- paths (write outside src/ at repo root) ---
repo_root = Path(__file__).resolve().parents[1]
out_dir = repo_root / "watermarked_output"
out_dir.mkdir(parents=True, exist_ok=True)

res = out_dir / "response.txt"

done_editing = False
round_i = 1
current_text = response

while not done_editing:
    # write the current text (initial draft or your last edit) to disk
    res.write_text(current_text, encoding="utf-8")
    print(f"\n Edit round {round_i} written to: {res}")
    input("Open the file, edit as you like, save, then press Enter to run detection... ")

    # read your edits back in
    edited_text = res.read_text(encoding="utf-8")
    print("Edited Text", edited_text)

    # detect watermark on your edited text
    print("\nDetecting watermark on your edited text...")
    result = default_falcon.detect_water(edited_text)
    if result is not None:
        print(result)

    # continue editing?
    ans = input("Edit again? [y/N]: ").strip().lower()
    if ans in ("y", "yes"):
        # carry your edits forward to the next round
        current_text = edited_text
        round_i += 1
    else:
        done_editing = True