from prc.zero_bit_prc import key_gen, encode, decode
import numpy as np

codeword_len = int(75/3)
# sparsity = int(np.log2(codeword_len))
sparsity = 1
generator_matrix, parity_check_matrix, one_time_pad = key_gen(codeword_len, sparsity)
encoding_key = generator_matrix, one_time_pad
decoding_key = parity_check_matrix, one_time_pad

noise_rates = np.arange(0.0, 0.51, 0.01)

for noise in noise_rates:
    print("noise: ", noise)
    codeword = encode(encoding_key, noise)
    decode(decoding_key, codeword)