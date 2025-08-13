from key_manager import KeyManager
import torch
import numpy as np
import torch.nn.functional as F
from prc.zero_bit_prc import encode, decode
from prc.majority_encoding import majority_encode, majority_decode


class WaterLLM(): 
    def __init__(self, name, model, tokenizer, generation_config, water_config):
        self._model = model
        self._tokenizer = tokenizer

        self.name = name

       

        self._hash = self.simple_hash

        # Generation Config
        self.temperature = generation_config.temperature
        self.top_p = generation_config.top_p
        self.repetition_penalty = generation_config.repetition_penalty
        self.no_repeat_ngram_size = generation_config.no_repeat_ngram_size
        self.token_buffer = generation_config.token_buffer
        self.skip_special_tokens = generation_config.skip_special_tokens
        self.add_special_tokens = generation_config.add_special_tokens

        # Water Config
        self.sparsity_function = water_config.sparsity_function
        self.encoding_noise_rate = water_config.encoding_noise_rate
        self.majority_encoding_rate = water_config.majority_encoding_rate
        self.key_dir = water_config.key_dir

        self._key_manager = KeyManager(self.name, self.key_dir)

    @staticmethod
    def simple_hash(token_id): 
        if(token_id % 2 == 0): 
            return 0
        else: 
            return 1
    def complete_prompt(self, prompt): 
        full_prompt = "<|system|> You are an academic and mindful assistant. <|user|>"  + prompt + "<|assistant|>"
        return full_prompt
    
    def generate(self, prompt, num_words, is_watermarked):
        print("Prompt: ", prompt)
        if(is_watermarked): 
            text = "Watermarked"
        else: 
            text = "Dry"
        print(text, " Text: ")
        num_tokens = int(num_words * 1.25)
        prompt = self.complete_prompt(prompt)
        codeword = self.__gen_codeword(num_tokens)
        response = self.__sample_response(prompt, codeword, num_tokens, is_watermarked) 
        self.detect_water(response, num_tokens)
        return response
    
    def detect_water(self, text, num_tokens): 
        token_ids = self._tokenizer.encode(text, add_special_tokens = self.add_special_tokens)
        noisy_majority_codeword = np.empty(0, dtype = int)
        for tid in token_ids[0 : num_tokens]:
            h = self.simple_hash(tid)
            noisy_majority_codeword = np.append(noisy_majority_codeword, h)
        
        codeword_len = int(num_tokens / self.majority_encoding_rate)
        sparsity = self.sparsity_function(codeword_len)

        generator_matrix, parity_check_matrix,one_time_pad = self._key_manager.fetch_key(codeword_len, sparsity)
        decoding_key = (parity_check_matrix, one_time_pad)
        noisy_codeword = majority_decode(noisy_majority_codeword, codeword_len)
        print(decode(decoding_key, noisy_codeword))

    
    def __gen_codeword(self, num_tokens):
        codeword_len = int(num_tokens / self.majority_encoding_rate)
        sparsity = self.sparsity_function(codeword_len)
        key = self._key_manager.fetch_key(codeword_len, self.sparsity_function(codeword_len))
        if(key is not None): 
            generator_matrix, parity_check_matrix, one_time_pad = key
        else: 
            generator_matrix, parity_check_matrix, one_time_pad = self._key_manager.gen_key(codeword_len, sparsity)
        encoding_key = (generator_matrix, one_time_pad)
        codeword = encode(encoding_key, self.encoding_noise_rate)
        majority_codeword = majority_encode(codeword, self.majority_encoding_rate)
        return majority_codeword

    def __apply_repetition_penalty(self, logits, generated_ids):
        if self. repetition_penalty != 1.0:
            for token_id in set(generated_ids[0].tolist()):
                logits[0, token_id] /= self.repetition_penalty
        return logits

    def __apply_no_repeat_ngram(self, logits, generated_ids):
        if self.no_repeat_ngram_size > 0 and generated_ids.size(1) >= self.no_repeat_ngram_size:
                banned_tokens = []
                context = generated_ids[0].tolist()
                prev_ngram = tuple(context[-(self.no_repeat_ngram_size - 1):])
                for i in range(len(context) - self.no_repeat_ngram_size + 1):
                    ngram = tuple(context[i:i + self.no_repeat_ngram_size])
                    if tuple(ngram[:-1]) == prev_ngram:
                        banned_tokens.append(ngram[-1])
                for token in banned_tokens:
                    logits[0, token] = -float("inf")
        return logits
    
    def __apply_top_p_sampling(self, probs): 
        sorted_probs, sorted_indices = torch.sort(probs, descending = True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_indices_to_remove = cumulative_probs > self.top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        probs[0, indices_to_remove] = 0
        probs = probs / probs.sum()
        return probs

      #Bias probabilities towards so that the sampled token's hash matches bit
    def __bias_probs(self, probs, bit):
        vocab_size = probs.shape[-1]
        device = probs.device

        # Vectorize hash over all indices once
        token_hash_vals = torch.tensor([self._hash(i) for i in range(vocab_size)], device=device)

        # Mask where hash == bit
        bits_to_boost = (token_hash_vals == bit)
        
        # Extract total probabilitiy of sampling a token with hash == bit
        group_probs = probs[0][bits_to_boost]
        unbiased_bit_prob = group_probs.sum().item()

        # Handle edge cases - avoid a divide by zero
        if unbiased_bit_prob == 0.0 or unbiased_bit_prob == 1.0:
            return probs

        # Compute biased sum
        biased_bit_prob = 1.0 if unbiased_bit_prob > 0.5 else 2 * unbiased_bit_prob

        #Boost the probabilities of bits which agree with 'bit' and reduce others
        biased_probs = probs.clone()
        biased_probs[0, bits_to_boost] = (probs[0, bits_to_boost] / unbiased_bit_prob) * biased_bit_prob
        biased_probs[0, ~bits_to_boost] = (probs[0, ~bits_to_boost] / (1 - unbiased_bit_prob)) * (1 - biased_bit_prob)

        # Normalize
        biased_probs = biased_probs / biased_probs.sum()

        return biased_probs

    
    def __sample_response(self, prompt, codeword, num_tokens, is_watermarked): 
        input_ids = self._tokenizer(prompt, return_tensors="pt", add_special_tokens = self.add_special_tokens).to(self._model.device)
        generated_ids = input_ids["input_ids"]

        # Initial decoded text
        decoded_so_far = self._tokenizer.decode(generated_ids[0], skip_special_tokens = self.skip_special_tokens)
        for token_idx in range(num_tokens + self.token_buffer):
            # Get logits
            outputs = self._model(input_ids=generated_ids)
            logits = outputs.logits[:, -1, :]

            # Apply sampling strategies
            logits = self.__apply_repetition_penalty(logits, generated_ids)
            logits = self.__apply_no_repeat_ngram(logits, generated_ids)
            logits = logits / self.temperature
            probs = F.softmax(logits, dim=-1)
            probs = self.__apply_top_p_sampling(probs)


            # Bias probabilities towards desired bit we are watermarking
            desired_bit = 0
            if(is_watermarked and token_idx < len(codeword)):
                desired_bit = codeword[token_idx]
                probs = self.__bias_probs(probs, desired_bit)

            # Sample token
            next_token = torch.multinomial(probs, num_samples=1)


            true_bit = self._hash(next_token.item())
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            
            # Print out newly sampled token
            full_decoded_text = self._tokenizer.decode(generated_ids[0], skip_special_tokens = self.skip_special_tokens)
            new_token = full_decoded_text[len(decoded_so_far):]

            ending_tokens = ['.','?','!']
            
            # Stop if EOS
            if (next_token.item() == self._tokenizer.eos_token_id): 
                if(token_idx > num_tokens):
                    break
            else: 
                if(is_watermarked and token_idx < len(codeword)): 
                    if(true_bit == desired_bit): 
                        print(f"\033[30;42m{new_token}\033[0m", end = '', flush = True)
                    else: 
                        print(f"\033[30;41m{new_token}\033[0m", end = '', flush = True)
                else: 
                    print(f"{new_token}", end = '', flush = True)
                if(new_token in ending_tokens and token_idx > num_tokens): 
                    break
            decoded_so_far = full_decoded_text
        print()

        # Remove prompt from output and return response
        input_len = input_ids["input_ids"].shape[1]
        response = self._tokenizer.decode(generated_ids[0][input_len:], skip_special_tokens = self.skip_special_tokens)
       
        return response