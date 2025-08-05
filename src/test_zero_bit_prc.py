from zero_bit_prc import ZeroBitPRC
import numpy as np
import galois
import pytest
import os

GF = galois.GF(2)

#Testing Parameters
noise_levels = np.linspace(0, 0.09, 10)
noise_levels = noise_levels.round(2)
n = 2 ** 9


@pytest .mark.parametrize("noise", noise_levels)
def test_encode_with_noise(noise: float, codeword_len: int = n): 
    PRC = ZeroBitPRC(codeword_len)
    generator_matrix, parity_check_matrix, one_time_pad = fetch_keys(codeword_len)
    encoding_key = generator_matrix, one_time_pad
    decoding_key = parity_check_matrix, one_time_pad
    codeword = PRC.Encode(encoding_key, noise)
    watermarked = PRC.Decode(decoding_key, codeword)
    assert (watermarked == True), "Parity checks exceeded threshold"

def empirical_false_positive_rate(codeword_len = n, num_trials = 1000): 
    PRC = ZeroBitPRC(codeword_len, 0)
    generator_matrix, parity_check_matrix, one_time_pad = PRC.KeyGen()
    decoding_key = parity_check_matrix, one_time_pad
    false_positive_count = 0
    for _ in range(num_trials):
        random_codeword = GF.random(codeword_len)
        watermarked = PRC.Decode(decoding_key, random_codeword)
        if not watermarked:
            false_positive_count += 1
    false_positive_rate =  (false_positive_count / num_trials)
    print(false_positive_rate)
    
#Helper Methods

def fetch_keys(codeword_len: int):
    save_dir = "keys" 
    os.makedirs(save_dir, exist_ok = True)
    filename = os.path.join(save_dir, f"keys_n{codeword_len}.npz")

    if os.path.exists(filename):
        print(f"[+] Loading keys for n={codeword_len}")
        data = np.load(filename)
        g = data["generator_matrix"]
        p = data["parity_check_matrix"]
        otp = data["one_time_pad"]
    else:
        print(f"[!] No keys found for n={codeword_len}, generating new ones...")
        PRC = ZeroBitPRC(codeword_len)
        g, p, otp = PRC.KeyGen()
        np.savez(filename,
                 generator_matrix=g,
                 parity_check_matrix=p,
                 one_time_pad=otp)
        print(f"[+] Saved new keys to {filename}")
    
    return GF(g), GF(p), GF(otp)
