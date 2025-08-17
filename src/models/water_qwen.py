from transformers import AutoModelForCausalLM, AutoTokenizer
from water_llm import WaterLLM
from transformers import AutoTokenizer, AutoModelForCausalLM

class WaterQwen(WaterLLM):
    def __init__(self, model_name, generation_config, water_config):
        model_id = "Qwen/Qwen3-4B-Instruct-2507"
        # load the tokenizer and the model
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto"
        )
        super().__init__(model_name, model, tokenizer, generation_config, water_config)


# We suggest using Temperature = 0.7, TopP = 0.8, TopK = 20, and MinP = 0.

