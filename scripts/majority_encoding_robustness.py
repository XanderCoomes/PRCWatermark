from prc.zero_bit_prc import key_gen, encode, decode
from majority_encoding_robustness import encode, decode
import numpy as np

codeword_len = 25
majority_encoding_rate = 3
sparsity = 1
generator_matrix, parity_check_matrix, one_time_pad = key_gen(codeword_len, sparsity)
encoding_key = generator_matrix, one_time_pad
decoding_key = parity_check_matrix, one_time_pad

noise_rate = 0.00

codeword = encode(encoding_key, noise_rate)
majority_codeword = encode()
decode(decoding_key, codeword)