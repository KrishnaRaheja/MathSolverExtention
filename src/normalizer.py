import re

# -----------------------------------------TEXT NORMALIZATION:

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

    """removes spaces or long spaces (tab space \t)"""
    s = line.replace(" ", "").replace("\t", "")

    """Insert '*' between a digit and a letter or '(', e.g.:
      10x     → 10*x
      3(x+1)  → 3*(x+1)"""
    s = re.sub(r'(?<=\d)(?=[A-Za-z(])', '*', s)
    
    """Insert '*' between a letter or ')' and a digit or '(', e.g.:
      x2      → x*2
      x(2)    → x*(2)
      (x+1)(x-1) → (x+1)*(x-1)"""
    s = re.sub(r'(?<=[A-Za-z\)])(?=\d|\()', '*', s)

    return s