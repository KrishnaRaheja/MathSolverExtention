import re
from sympy import Eq, symbols, simplify, solve
from sympy.parsing.sympy_parser import parse_expr

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