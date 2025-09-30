import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from ocr import make_client, detect_document_from_clipboard, KEY_PATH
from solver import solve_or_simplify

if __name__ == "__main__":
    client = make_client(KEY_PATH)
    print("After each Win+Shift+S snip, press Enter here to solve. Ctrl+C to quit.")
    try:
        while True:
            input()
            try:
                expr = detect_document_from_clipboard(client)
                result = solve_or_simplify(expr)
                print(f"\nReading: {expr}\nAnswer:  {result}\n")
            except Exception as e:
                print("Error:", e)
    except KeyboardInterrupt:
        pass