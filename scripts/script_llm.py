from llama import Llama

LLama = Llama()
prompt =  """
<|system|>
You are a knowledgeable and articulate academic assistant.

<|user|>
Write a short poem on the impact of artificial intelligence on the future of education. Discuss opportunities, risks, and give examples.

<|assistant|>
"""
short_poem = LLama.gen_response(prompt, max_tokens = 30)
print(short_poem)


