
# Use a pipeline as a high-level helper
from water_llm import WaterLLM
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from transformers import pipeline

class WaterCalme(WaterLLM):
    def __init__(self, model_name, generation_config, water_config):
        messages = [
        {"role": "user", "content": "Who are you?"},
        ]
        pipe = pipeline("text-generation", model="MaziyarPanahi/calme-3.2-instruct-78b")
        pipe(messages)

        tokenizer = AutoTokenizer.from_pretrained("MaziyarPanahi/calme-3.2-instruct-78b")
        model = AutoModelForCausalLM.from_pretrained("MaziyarPanahi/calme-3.2-instruct-78b")

        super().__init__(model_name, model, tokenizer, generation_config, water_config)



