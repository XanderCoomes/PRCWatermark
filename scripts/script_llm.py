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

print(LLama.response_to_token_ids(short_poem))


print(LLama.tokenizer("In", return_tensors="pt")["input_ids"][0].tolist())
print(LLama.tokenizer("AI", return_tensors="pt")["input_ids"][0].tolist())
print(LLama.tokenizer("In AI", return_tensors="pt")["input_ids"][0].tolist())
print(LLama.tokenizer("A", return_tensors="pt")["input_ids"][0].tolist())
print(LLama.tokenizer("I", return_tensors="pt")["input_ids"][0].tolist())

