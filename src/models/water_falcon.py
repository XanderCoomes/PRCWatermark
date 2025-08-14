from models.water_llm import WaterLLM
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch



class WaterFalcon(WaterLLM):
    def __init__(self, model_name, generation_config, water_config):
        model_id = "tiiuae/falcon-7b-instruct"  # Falcon-H1-34B on Hugging Face
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",            # Automatically map to available GPUs
            torch_dtype=torch.bfloat16    # Falcon-H1-34B prefers bfloat16 if GPU supports it
        )
        super().__init__(model_name, model, tokenizer, generation_config, water_config)
