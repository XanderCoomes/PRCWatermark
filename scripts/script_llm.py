from llama import Llama
import galois
import numpy as np
from key_manager import fetch_key, clear_key, clear_all_keys



GF = galois.GF(2)

LLama = Llama()
prompt =  """
<|system|>
You are an academic and mindful assistant.

<|user|>
Write a poem about a hummingbird.

<|assistant|> 
"""
num_tokens = 2 ** 6
is_watermarked = True

if(is_watermarked): 
    text  = "Watermarked Response:"
else: 
    text = "Non-Watermarked Response:"

print(text)
response = LLama.gen_response(prompt, num_tokens, is_watermarked)
print("\n")

print("Detection:")
if(LLama.detect_watermarked_response(response) == True): 
    print("Watermarked response detected.")
else: 
    print("No watermark detected.")

