import os
from zero_bit_prc import ZeroBitPRC
import numpy as np
import galois

GF = galois.GF(2)

def gen_key(codeword_len, sparsity = None): 
    save_dir = "keys" 
    os.makedirs(save_dir, exist_ok = True)
    filename = os.path.join(save_dir, f"keys_n{codeword_len}.npz")
    PRC = ZeroBitPRC(codeword_len, sparsity)
    g, p, otp = PRC.key_gen()
    np.savez(filename,
                generator_matrix=g,
                parity_check_matrix=p,
                one_time_pad=otp)

def fetch_key(codeword_len):
    save_dir = "keys" 
    os.makedirs(save_dir, exist_ok = True)
    filename = os.path.join(save_dir, f"keys_n{codeword_len}.npz")
    if os.path.exists(filename):
        data = np.load(filename)
        g = data["generator_matrix"]
        p = data["parity_check_matrix"]
        otp = data["one_time_pad"]
    else:
        return None
        
    
    return GF(g), GF(p), GF(otp)

def clear_key(codeword_len): 
    save_dir = "keys"
    filename = os.path.join(save_dir, f"keys_n{codeword_len}.npz")
    if os.path.exists(filename):
        os.remove(filename)
    else:
        print(f"[!] No keys found for n = {codeword_len} to clear")


def clear_all_keys(): 
    save_dir = "keys"
    if os.path.exists(save_dir):
        for filename in os.listdir(save_dir):
            file_path = os.path.join(save_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
    else:
        print("[!] No keys directory found to clear")
