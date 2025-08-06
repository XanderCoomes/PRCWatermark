from llm import LLM
from transformers import AutoTokenizer, AutoModelForCausalLM

class Llama(LLM):
    def __init__(self):
        model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)
        super().__init__(tokenizer, model)



    