import numpy as np
import galois
from scipy.stats import binom

GF = galois.GF(2)


def calc_secret_len(codeword_len):
    return int(np.log2(codeword_len)) ** 2

def calc_num_parity_checks(codeword_len): 
    return int(0.99 * codeword_len)

def sample_parity_check_matrix(codeword_len, num_parity_checks, sparsity):
    parity_check_matrix = np.zeros((num_parity_checks, codeword_len), dtype = int)

    for i in range (num_parity_checks):
        row = np.zeros(codeword_len, dtype=int)
        ones_indices = np.random.choice(codeword_len, size = sparsity, replace = False)
        row[ones_indices] = 1
        parity_check_matrix[i] = row 

    return GF(parity_check_matrix)

def sample_generator_matrix(codeword_len, secret_len, parity_check_matrix):
    null_space = parity_check_matrix.null_space()
    null_space = null_space.T 

    # Generate a random generator matrix G, such that PG = 0
    generator_matrix = np.zeros((codeword_len, secret_len), dtype = int)
    for i in range (secret_len): 
        rand_null_vector = null_space @ GF.Random(null_space.shape[1])
        generator_matrix[:, i] = rand_null_vector
    
    return GF(generator_matrix)

def key_gen(codeword_len, sparsity):
    # Generate a random sparse parity check matrix, P
    num_parity_checks = calc_num_parity_checks(codeword_len)
    secret_len = calc_secret_len(codeword_len)

    parity_check_matrix = sample_parity_check_matrix(codeword_len, num_parity_checks, sparsity)
    generator_matrix = sample_generator_matrix(codeword_len, secret_len, parity_check_matrix)
   
    one_time_pad = GF.Random(codeword_len)
    
    return generator_matrix, parity_check_matrix, one_time_pad
    
def encode(encoding_key, noise_rate):
    generator_matrix, one_time_pad = encoding_key
    
    codeword_len = generator_matrix.shape[0]
    secret_len = generator_matrix.shape[1]

    # Generate a random secret
    secret = GF.Random(secret_len)

    # Error is added to ensure pseudorandomness
    error = GF(np.random.binomial(1, noise_rate, codeword_len))

    codeword = (generator_matrix @ secret + one_time_pad + error)
    return codeword

def decode(decoding_key, codeword, sparsity):  
    parity_check_matrix, one_time_pad = decoding_key
    codeword_len = len(codeword)
    num_parity_checks = calc_num_parity_checks(codeword_len)

    codeword = GF(codeword) + one_time_pad
    syndrome = parity_check_matrix @ codeword
    failed_parity_checks = np.sum(syndrome == 1)
    

    n = num_parity_checks
    k = failed_parity_checks
    p_random = 0.5

    prob_given_dry = binom.pmf(k, n, p_random)

    assumed_error_rate = 0.3

    p_water = (1 - (1 - 2 * assumed_error_rate)**sparsity) / 2

    prob_given_water = binom.pmf(k, n, p_water)

    prob_watermarked = prob_given_water / (prob_given_water + prob_given_dry)

    return prob_watermarked





    


    

   
     







