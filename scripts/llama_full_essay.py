from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Model name (fully open source)
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",             # Automatically uses GPU if available
    torch_dtype=torch.float16      # Use float16 for performance
)
model.eval()

# Prompt (chat-style format for TinyLlama)
prompt = """
<|system|>
You are a knowledgeable and articulate academic assistant.

<|user|>
Write a detailed essay on the impact of artificial intelligence on the future of education. Discuss opportunities, risks, and give examples.

<|assistant|>
"""

# Tokenize input
input_ids = tokenizer(prompt.strip(), return_tensors="pt").to(model.device)

# Generate the essay
output = model.generate(
    **input_ids,
    max_new_tokens= 100,
    do_sample=True,
    temperature=0.85,
    top_p=0.9,
    repetition_penalty=1.15,
    no_repeat_ngram_size=3,
    eos_token_id=tokenizer.eos_token_id
)

# Decode and print
essay = tokenizer.decode(output[0], skip_special_tokens=True)
print(essay)
