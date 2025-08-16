from water_llm import WaterLLM
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class WaterLlama(WaterLLM):
    def __init__(self, model_name, generation_config, water_config):
        model_id = "meta-llama/Meta-Llama-3-8B"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16
        )
        super().__init__(model_name, model, tokenizer, generation_config, water_config)


    