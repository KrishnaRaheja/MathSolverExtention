from pathlib import Path
import re
import sys
from google.cloud import vision
from google.oauth2 import service_account
from sympy import Eq, symbols, simplify, solve
from sympy.parsing.sympy_parser import parse_expr
from PIL import ImageGrab
import io

"""IDEA: could just use snipping tool and change save location to project folder,
and then the program could take the most recent screenshot and solve the problem.

TWO WAYS OF DOING THIS:
1. Snipping tool automatically saves to clipboard, so program could watch for an image in clipboard to solve
2. Change snipping tool save location to project folder, and read from project


-Skill shown: Building off of existing availible tools like snipping tool
"""

# -----------------------------------------Setup:

KEY_PATH = r"C:\Users\rahej\Documents\Programming\Keys\woven.json"

# BASE_DIR = Path(__file__).resolve().parent
# DEFAULT_IMAGE = BASE_DIR / "test2.png"
# # allow CLI override: python src/main.py path/to/img.png
# IMAGE_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_IMAGE

def make_client(key_path) -> "vision.ImageAnnotatorClient":
    creds = service_account.Credentials.from_service_account_file(key_path)
    return vision.ImageAnnotatorClient(credentials=creds)

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

    # removes spaces or long spaces (tab space \t)
    s = line.replace(" ", "").replace("\t", "")

    # puts * symbol 
    s = re.sub(r'(?<=\d)(?=[A-Za-z(])', '*', s)
    
    # letter or ')' before digit or '(' -> x2, x( -> x*2, x*(...
    s = re.sub(r'(?<=[A-Za-z\)])(?=\d|\()', '*', s)

    return s

# -----------------------------------------Document detection:

# def detect_document(path: str, client: "vision.ImageAnnotatorClient") -> str:
#     """Returns a single-line, solver-ready math expression from the image."""
#     content = Path(path).read_bytes()
#     image = vision.Image(content=content)

#     response = client.document_text_detection(image=image)

#     if response.error.message:
#         raise RuntimeError(
#             f"Vision API error: {response.error.message}\n"
#             "See: https://cloud.google.com/apis/design/errors"
#         )

#     stitched = response.full_text_annotation.text or ""
#     # 1) Normalize glyphs but KEEP line breaks
#     stitched_norm = _ascii_ops_and_supers(stitched)
#     # 2) Choose the best single line
#     best_line = _pick_best_math_line(stitched_norm)
#     # 3) Make that one line solver-ready
#     expr = _normalize_line_for_solver(best_line)

#     print(expr)  # prints expression
#     return expr

# -----------------------------------------Solving problem:

_VAR_RE = re.compile(r"[A-Za-z]")

def _find_vars(s: str) -> list[str]:
    """Collect single-letter variables present in s (e.g., x,y,z)."""
    return sorted(set(_VAR_RE.findall(s)))

def _parse_side(side: str, var_names: list[str]):
    """
    Parse one side of an equation/expression using SymPy, with the given variables.
    We convert '^' -> '**' for exponentiation.
    """
    # convert caret to Python power for SymPy
    side = side.replace("^", "**")
    local = {name: symbols(name) for name in var_names}  # x -> Symbol('x'), etc.
    return parse_expr(side, local_dict=local)

def solve_or_simplify(expr_text: str) -> dict:
    """
    Decide what to do:
      - If contains '=', solve Eq(lhs, rhs) for a variable (prefer 'x', else first one).
      - Else, if numeric only, evaluate; else simplify symbolically.
    Returns a small dict so your GUI can display cleanly.
    """
    if not expr_text:
        return {"type": "empty", "message": "No math found."}

    # Collect variables seen anywhere (both sides if present)
    var_names = _find_vars(expr_text)
    has_equals = "=" in expr_text

    if has_equals:
        lhs_text, rhs_text = expr_text.split("=", 1)
        lhs = _parse_side(lhs_text, var_names)
        rhs = _parse_side(rhs_text, var_names)
        eq = Eq(lhs, rhs)

        # Pick a variable to solve for (prefer x)
        syms = list(eq.free_symbols)
        if not syms:
            # no symbols – just check if it's true/false numerically
            is_true = bool(simplify(lhs - rhs) == 0)
            return {"type": "equation", "equation": str(eq), "result": "true" if is_true else "false"}

        prefer = symbols("x")
        target = prefer if prefer in syms else syms[0]

        sols = solve(eq, target, dict=True)  # list[ {x: value}, ... ]
        # turn into strings for UI
        pretty = [f"{target} = {val}" for d in sols for (target, val) in d.items()]
        return {"type": "equation", "equation": str(eq), "solutions": pretty}

    else:
        # Just an expression (no '=')
        expr = _parse_side(expr_text, var_names)
        if not expr.free_symbols:
            # purely numeric
            val = expr.evalf()  # numeric value
            return {"type": "numeric", "original": expr_text, "value": str(val)}
        else:
            # symbolic expression -> simplify
            simp = simplify(expr)
            return {"type": "expression", "original": expr_text, "simplified": str(simp)}

def detect_document_from_clipboard(client) -> str:
    """
    If an image is on the clipboard, OCR it and return the solved expression.
    Use after you take a snip (Win+Shift+S).
    """
    img = ImageGrab.grabclipboard()
    if img is None:
        raise RuntimeError("No image found on clipboard. Take a snip first.")

    # Convert to bytes in-memory for Vision API
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    image = vision.Image(content=buf.getvalue())

    response = client.document_text_detection(image=image)
    if response.error.message:
        raise RuntimeError(
            f"Vision API error: {response.error.message}\n"
            "See: https://cloud.google.com/apis/design/errors"
        )

    stitched = response.full_text_annotation.text or ""
    stitched_norm = _ascii_ops_and_supers(stitched)
    best_line = _pick_best_math_line(stitched_norm)
    expr = _normalize_line_for_solver(best_line)
    return expr

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