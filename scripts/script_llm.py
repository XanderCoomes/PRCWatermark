from llama import Llama
import numpy as np

LLama = Llama()
prompt =  """
<|system|>
You are a knowledgeable and articulate academic assistant.

<|user|>
Write a poem about something interesting. 
<|assistant|> 
"""
max_tokens = 50
codeword = np.ones(max_tokens) 
short_poem = LLama.gen_watermarked_response(prompt, max_tokens, codeword)
 

print("Decoding the Poem:")
LLama.detect_watermarked_response(short_poem, decoding_key = None)



