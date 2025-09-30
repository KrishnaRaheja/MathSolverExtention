from google.cloud import vision
from google.oauth2 import service_account
from PIL import ImageGrab
import io
from normalizer import _ascii_ops_and_supers, _pick_best_math_line, _normalize_line_for_solver

# -----------------------------------------Setup:

KEY_PATH = r"{Put your key path here}"

def make_client(key_path) -> "vision.ImageAnnotatorClient":
    creds = service_account.Credentials.from_service_account_file(key_path)
    return vision.ImageAnnotatorClient(credentials=creds)

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