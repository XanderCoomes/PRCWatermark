from ZeroBitPRC import ZeroBitPRC
def main(): 
    codeword_len = 2 ** 10
    noise_rate = 0.05
    PRC = ZeroBitPRC(codeword_len, noise_rate)
    generator_matrix, parity_check_matrix, one_time_pad = PRC.KeyGen2()
    encoding_key = generator_matrix, one_time_pad
    decoding_key = parity_check_matrix, one_time_pad
    codeword = PRC.Encode(encoding_key)
    PRC.print_field_info()
    watermarked = PRC.Decode(decoding_key, codeword)
    print("watermarked:", watermarked)





if __name__ == "__main__":
    main()