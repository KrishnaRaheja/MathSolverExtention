from pathlib import Path
import re
import sys
from google.cloud import vision
from google.oauth2 import service_account

KEY_PATH = r"C:\Users\rahej\Documents\Programming\Keys\woven.json"

BASE_DIR = Path(__file__).resolve().parent         # folder of this file
DEFAULT_IMAGE = BASE_DIR / "test.png"              # .../src/test.png
# allow CLI override: python src/main.py path/to/img.png
IMAGE_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_IMAGE

def make_client(key_path) -> "vision.ImageAnnotatorClient":
    creds = service_account.Credentials.from_service_account_file(key_path)
    return vision.ImageAnnotatorClient(credentials=creds)

# ---------- text normalization functions ----------
_MATH_CHARS = set("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ+-*/^()=.")

def _ascii_ops_and_supers(s: str) -> str:
    """Normalize common math glyphs to ASCII and map superscripts to ^n, but keep spaces/newlines."""
    # Operators to ASCII, the multiplication key symbols are distinct unicode characters to account for OCR readings
    trans = {
        "×": "*", "·": "*", "∙": "*", "•": "*",
        "−": "-", "–": "-", "—": "-",
        "÷": "/", "⁄": "/",
    }
    for k, v in trans.items():
        s = s.replace(k, v)

    # Superscripts -> ^n
    sup_map = {
        "⁰": "^0", "¹": "^1", "²": "^2", "³": "^3", "⁴": "^4",
        "⁵": "^5", "⁶": "^6", "⁷": "^7", "⁸": "^8", "⁹": "^9",
    }
    for k, v in sup_map.items():
        s = s.replace(k, v)

    return s

def _pick_best_math_line(stitched_text: str) -> str:
    """
    Split into lines and pick the one that looks most like 'the math expression':
    maximize (# of math chars), tie-break by length.
    """
    lines = [ln.strip() for ln in stitched_text.splitlines() if ln.strip()]
    if not lines:
        return ""

    def score(ln: str):
        math_count = sum(ch in _MATH_CHARS for ch in ln)
        return (math_count, len(ln))  # prioritize math density, then length

    best = max(lines, key=score)
    return best

def _normalize_line_for_solver(line: str) -> str:
    """
    Turn a single math line into solver-friendly ASCII:
    - remove whitespace
    - insert explicit multiplication where implied
    """
    if not line:
        return ""

    s = line.replace(" ", "").replace("\t", "")

    # Insert explicit multiplication:
    # number before letter or '('  -> 10x, 2( -> 10*x, 2*(...
    s = re.sub(r'(?<=\d)(?=[A-Za-z(])', '*', s)
    # letter or ')' before digit or '(' -> x2, x( -> x*2, x*(...
    s = re.sub(r'(?<=[A-Za-z\)])(?=\d|\()', '*', s)

    return s

# ---------- main OCR function ----------
def detect_document(path: str, client: "vision.ImageAnnotatorClient") -> str:
    """Returns a single-line, solver-ready math expression from the image."""
    content = Path(path).read_bytes()
    image = vision.Image(content=content)

    response = client.document_text_detection(image=image)

    if response.error.message:
        raise RuntimeError(
            f"Vision API error: {response.error.message}\n"
            "See: https://cloud.google.com/apis/design/errors"
        )

    stitched = response.full_text_annotation.text or ""
    # 1) Normalize glyphs but KEEP line breaks
    stitched_norm = _ascii_ops_and_supers(stitched)
    # 2) Choose the best single line
    best_line = _pick_best_math_line(stitched_norm)
    # 3) Make that one line solver-ready
    expr = _normalize_line_for_solver(best_line)

    print(expr)  # e.g., 10*x+14^2*12
    return expr

if __name__ == "__main__":
    client = make_client(KEY_PATH)
    detect_document(IMAGE_PATH, client)
