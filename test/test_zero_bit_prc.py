from zero_bit_prc import ZeroBitPRC
import numpy as np
import galois
import pytest
import os

GF = galois.GF(2)

#Testing Parameters
noise_levels = np.linspace(0, 0.09, 10)
noise_levels = noise_levels.round(2)

# For undetectability, n ^ t should be greater than 2 ** 100
n = 2 ** 10
t = None

def test_setup(): 
    clear_key(n)
    return


@pytest .mark.parametrize("noise", noise_levels)
def test_encode_with_noise(noise, codeword_len = n, sparsity = t): 
    PRC = ZeroBitPRC(codeword_len)
    generator_matrix, parity_check_matrix, one_time_pad = fetch_key(codeword_len, sparsity)
    encoding_key = generator_matrix, one_time_pad
    decoding_key = parity_check_matrix, one_time_pad
    codeword = PRC.Encode(encoding_key, noise)
    is_detected = PRC.Decode(decoding_key, codeword)
    assert (is_detected == True), "PRC codeword not detected"

def test_fpr(codeword_len = n, num_trials = 1000): 
    PRC = ZeroBitPRC(codeword_len)
    generator_matrix, parity_check_matrix, one_time_pad = fetch_key(codeword_len)
    decoding_key = parity_check_matrix, one_time_pad
    false_positive_count = 0
    for trial in range(num_trials):
        random_bits = GF.Random(codeword_len)
        is_detected = PRC.Decode(decoding_key, random_bits)
        if is_detected:
            false_positive_count += 1
    false_positive_rate =  (false_positive_count / num_trials)
    assert(false_positive_rate < 0.01), "False positive rate exceeded threshold"

#Helper Methods

def fetch_key(codeword_len, sparsity = None):
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
        PRC = ZeroBitPRC(codeword_len, sparsity)
        g, p, otp = PRC.KeyGen()
        np.savez(filename,
                 generator_matrix=g,
                 parity_check_matrix=p,
                 one_time_pad=otp)
        print(f"[+] Saved new keys to {filename}")
    
    return GF(g), GF(p), GF(otp)

def clear_key(codeword_len): 
    save_dir = "keys"
    filename = os.path.join(save_dir, f"keys_n{codeword_len}.npz")
    if os.path.exists(filename):
        os.remove(filename)
        print(f"[+] Cleared keys for n={codeword_len}")
    else:
        print(f"[!] No keys found for n={codeword_len} to clear")


def clear_keys(): 
    save_dir = "keys"
    if os.path.exists(save_dir):
        for filename in os.listdir(save_dir):
            file_path = os.path.join(save_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"[+] Cleared {file_path}")
    else:
        print("[!] No keys directory found to clear")


