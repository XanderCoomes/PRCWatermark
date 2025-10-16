from prc.zero_bit_prc import key_gen, encode, threshold_decode
import numpy as np
import math


codeword_len = 500
sparsity = int(math.log10(codeword_len))

num_trials = 100

detection_arr = [0] * 51
noise_rates = np.arange(0.0, 0.51, 0.01)

is_watermark = False; 


for _ in range(num_trials):
    generator_matrix, parity_check_matrix, one_time_pad, permutation = key_gen(codeword_len, sparsity)
    encoding_key = generator_matrix, one_time_pad, permutation
    decoding_key = parity_check_matrix, one_time_pad, permutation


    for i, noise in enumerate(noise_rates):
        if(is_watermark):
            codeword = encode(encoding_key, noise)
        else:
            codeword = np.random.binomial(1, 0.5, size=codeword_len).astype(np.uint8)
        codeword_detected = threshold_decode(decoding_key, codeword, sparsity)
        if(codeword_detected): 
            detection_arr[i] += 1

probs = [x / num_trials  for x in detection_arr]

for prob in probs: 
    print(prob)

