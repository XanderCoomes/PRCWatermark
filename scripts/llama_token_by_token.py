from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F

# Load model and tokenizer
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16
)
model.eval()

# Prompt
prompt = """
<|system|>
You are a knowledgeable and articulate academic assistant.

<|user|>
Write a detailed essay on the impact of artificial intelligence on the future of education. Discuss opportunities, risks, and give examples.

<|assistant|>
""".strip()

# Tokenize input
input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
generated_ids = input_ids["input_ids"]

# Sampling config
max_new_tokens = 100
temperature = 0.85
top_p = 0.9
repetition_penalty = 1.15
no_repeat_ngram_size = 3

# Initial decoded text
decoded_so_far = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

for _ in range(max_new_tokens):
    # Get logits
    outputs = model(input_ids=generated_ids)
    logits = outputs.logits[:, -1, :]

    # Apply repetition penalty
    if repetition_penalty != 1.0:
        for token_id in set(generated_ids[0].tolist()):
            logits[0, token_id] /= repetition_penalty

    # No-repeat ngram
    if no_repeat_ngram_size > 0 and generated_ids.size(1) >= no_repeat_ngram_size:
        banned_tokens = []
        context = generated_ids[0].tolist()
        prev_ngram = tuple(context[-(no_repeat_ngram_size - 1):])
        for i in range(len(context) - no_repeat_ngram_size + 1):
            ngram = tuple(context[i:i + no_repeat_ngram_size])
            if tuple(ngram[:-1]) == prev_ngram:
                banned_tokens.append(ngram[-1])
        for token in banned_tokens:
            logits[0, token] = -float("inf")

    # Temperature
    logits = logits / temperature

    # Top-p sampling
    probs = F.softmax(logits, dim=-1) 
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    probs[0, indices_to_remove] = 0
    probs = probs / probs.sum()

    # Sample token
    next_token = torch.multinomial(probs, num_samples=1)
    generated_ids = torch.cat([generated_ids, next_token], dim=-1)

    # Decode full text so far
    new_decoded = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    # Find the new part
    new_piece = new_decoded[len(decoded_so_far):]
    print(new_piece, end='', flush=True)

    # Update tracker
    decoded_so_far = new_decoded

    # Stop if EOS
    if next_token.item() == tokenizer.eos_token_id:
        break

print(decoded_so_far)