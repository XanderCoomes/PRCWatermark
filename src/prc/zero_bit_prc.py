import numpy as np
import galois
import math
from scipy.stats import binom

GF = galois.GF(2)


def calc_secret_len(codeword_len):
    return int(math.log10(codeword_len)) ** 2

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

    permutation = np.random.permutation(codeword_len)

    
    return generator_matrix, parity_check_matrix, one_time_pad, permutation
    
def encode(encoding_key, noise_rate):
    generator_matrix, one_time_pad, permutation = encoding_key
    
    codeword_len = generator_matrix.shape[0]
    secret_len = generator_matrix.shape[1]

    # Generate a random secret
    secret = GF.Random(secret_len)

    # Error is added to ensure pseudorandomness
    error = GF(np.random.binomial(1, noise_rate, codeword_len))

    codeword = (generator_matrix @ secret + one_time_pad + error)
    permuted_codeword = codeword[permutation]
    return permuted_codeword

def decode(decoding_key, permuted_codeword, sparsity):
    assumed_error_rate = 0.4
    parity_check_matrix, otp, perm = decoding_key
    inv_perm = np.empty_like(perm)
    inv_perm[perm] = np.arange(len(perm))   
    codeword = permuted_codeword[inv_perm]
    num_parity_checks = calc_num_parity_checks(len(codeword))

    # --- Ensure GF(2) arrays and consistent shapes ---


    cw  = GF(np.asarray(codeword, dtype=np.uint8))       # [n_bits]
    otp = GF(np.asarray(otp,     dtype=np.uint8))        # [n_bits]
    if parity_check_matrix.shape[1] != cw.size:
        raise ValueError(f"Parity-check cols ({parity_check_matrix.shape[1]}) != codeword length ({cw.size})")
    if cw.size != otp.size:
        raise ValueError(f"Codeword/OTP length mismatch: {cw.size} vs {otp.size}")

    # --- Syndrome over GF(2) and #failed checks ---
    masked = cw + otp                 # XOR in GF(2)
    synd   = parity_check_matrix @ masked               # GF(2) mat-vec
    num_satisfied_checks = np.sum(synd == 1)

    # print("Parity Checks: ", num_parity_checks)
    # print("Satisfied", num_satisfied_checks)

    # --- Binomial parameters ---
    p_dry = (1.0 - (1.0 - 2 * 0.5) ** sparsity) / 2
    p_water = (1.0 - (1.0 - 2.0 * assumed_error_rate) ** sparsity) / 2.0

    # print("p_dry", p_dry)
    # print("p_water", p_water)

    prob_synd_given_water = binom.pmf(num_satisfied_checks, num_parity_checks, p_water)
    prob_synd_given_dry = binom.pmf(num_satisfied_checks, num_parity_checks, p_dry)

    # print("dry prob: ", prob_synd_given_dry)
    # print("water prob", prob_synd_given_water)

    prob_water = prob_synd_given_water/ (prob_synd_given_water + prob_synd_given_dry)

    return prob_water


def threshold_decode(decoding_key, permuted_codeword, sparsity):
    parity_check_matrix, otp, perm = decoding_key
    inv_perm = np.empty_like(perm)
    inv_perm[perm] = np.arange(len(perm))   
    codeword = permuted_codeword[inv_perm]
    num_parity_checks = calc_num_parity_checks(len(codeword))
    
    cw  = GF(np.asarray(codeword, dtype=np.uint8))      
    otp = GF(np.asarray(otp,      dtype=np.uint8))    
   
    masked = cw + otp             
    synd   = parity_check_matrix @ masked             
    num_sat_checks = np.sum(synd == 1)
    threshold = (0.5 - (num_parity_checks ** (-0.25))) * num_parity_checks; 

    if(num_sat_checks < threshold):
        return True;
    else:
        return False; 



    




    





    


    

   
     







