from llama import Llama

LLama = Llama()
prompt = """Write a 30 word poem on the importance of AI in modern society."""

short_poem = LLama.gen_response(prompt, max_tokens = 100)
print(short_poem)