from ZeroBitPRC import ZeroBitPRC
def main(): 
    codeword_len = 3000
    noise_rate = 0.01
    PRC = ZeroBitPRC(codeword_len, noise_rate)
    generator, parity = PRC.KeyGen()
    PRC.print_field_info()

if __name__ == "__main__":
    main()