from ZeroBitPRC import ZeroBitPRC
def main(): 
    codeword_len = 1024
    PRC = ZeroBitPRC(codeword_len)
    PRC.print_field_info()

if __name__ == "__main__":
    main()