def edit_content(self, prompt, num_words, response):
        # --- paths (write outside src/ at repo root) ---
        out_dir = self.get_out_dir(self.output_dir, prompt)
        res = out_dir / ("response" + str(num_words) + "words.txt")

        done_editing = False
        round_i = 1
        current_text = response

        while not done_editing:
            # write the current text (initial draft or your last edit) to disk
            res.write_text(current_text, encoding="utf-8")
            print(f"\n Edit round {round_i} written to: {res}")
            input("Open the file, edit as you like, save, then press Enter to run detection... ")

            # read your edits back in
            edited_text = res.read_text(encoding="utf-8")
            print("Edited Text", edited_text)

            # detect watermark on your edited text
            print("Watermark Detection")
            result = self.water_model.detect_water(edited_text)
            if result is not None:
                print(result)

            # continue editing?
            ans = input("Edit again? [y/N]: ").strip().lower()
            if ans in ("y", "yes"):
                # carry your edits forward to the next round
                current_text = edited_text
                round_i += 1
            else:
                done_editing = True