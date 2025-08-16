from pathlib import Path
import re

class WaterEditor(): 
    def __init__(self, water_model, output_dir): 
        self.water_model = water_model
        self.output_dir = output_dir
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
            print("\nDetecting watermark on your edited text...")
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
    @staticmethod
    def _slug(text: str, max_len: int = 80) -> str:
        """Filesystem-safe slug: collapse whitespace, keep [A-Za-z0-9._-], replace others with _."""
        if not isinstance(text, str):
            text = str(text)
        text = text.strip()
        text = re.sub(r"\s+", " ", text)                  # collapse runs of whitespace
        text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)     # replace disallowed chars with _
        text = text[:max_len]                              # enforce length
        return text or "untitled"

    def get_out_dir(self, base_dir, prompt: str) -> Path:
        """Return path: repo_root/watermarked_output/<model>/<prompt>/<filename>."""
        model_slug = self._slug(getattr(self.water_model, "name", "model"))
        prompt_slug = self._slug(prompt)

        out_dir = base_dir / model_slug / prompt_slug
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir