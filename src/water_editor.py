from pathlib import Path
import re
import os

class WaterEditor(): 
    def __init__(self, water_model, output_dir): 
        self.water_model = water_model
        self.output_dir = output_dir
    def save_response(self, prompt, num_words, response): 
        res = self.get_out_path(self.output_dir, prompt, num_words)
        current_text = response
        res.write_text(current_text, encoding="utf-8")
    def detect_content_from_path(self, response_path): 
        edited_text = response_path.read_text(encoding="utf-8")
        print("Text: ", edited_text)
        print("\nDetecting watermark on your edited text...")
        result = self.water_model.detect_water(edited_text)
        return result
    def edit_detect_loop(self, response_path):
        print("Options:")
        print("  [y] Detect watermark on current file")
        print("  [n] Stop program")
        print("=================================\n")

        while True:
            ans = input("Your choice [y/n]: ").strip().lower()
            if ans == "y":
                self.detect_content(response_path)
                print("\n---------------------------------\n")
            elif ans == "n":
                break
            else:
                print("⚠️  Invalid choice. Please enter 'y' or 'n'.")

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

    def get_out_path(self, base_dir, prompt: str, num_words) -> Path:
        """Return path: repo_root/watermarked_output/<model>/<prompt>/<filename>."""
        model_slug = self._slug(getattr(self.water_model, "name", "model"))
        prompt_slug = self._slug(prompt)

        out_dir = base_dir / model_slug / prompt_slug
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir /("response" + str(num_words) + "words.txt")
        return out_path