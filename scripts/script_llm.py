from llama import Llama

LLama = Llama()
prompt =  """
<|system|>
You are a knowledgeable and articulate academic assistant.

<|user|>
Write a poem about natural beauty of the frog. 
<|assistant|> 
"""
short_poem = LLama.gen_response(prompt, max_tokens = 50)

print(short_poem)


