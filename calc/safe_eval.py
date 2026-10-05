"""Safe evaluation of arithmetic expressions for the edge-case Calc tool.

Deliberately does NOT use eval()/exec() — those would let an LLM-generated
expression run arbitrary Python (attribute access, imports, etc.). Instead
this walks a parsed AST and only permits a small whitelist of node types:
numeric literals, named variables (resolved from a values dict the LLM
never controls), arithmetic operators, and a few safe built-in functions.

Anything else (attribute access, subscripting, function calls outside the
whitelist, comprehensions, string literals, etc.) is rejected before any
evaluation happens.
"""
from __future__ import annotations

import ast
import operator

from calc.metrics import CalcError

_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}

_ALLOWED_UNARYOPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_ALLOWED_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
}


def safe_eval(expression: str, values: dict[str, float]) -> float:
    """Evaluate a restricted arithmetic expression.

    `expression` is untrusted (LLM-generated) and may only reference names
    present in `values` (which the calling code supplies from real,
    already-known numbers — never from LLM-generated text).
    """
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        raise CalcError(f"Expression is not valid syntax: {e}") from e

    return _eval_node(tree.body, values)


def _eval_node(node: ast.AST, values: dict[str, float]) -> float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise CalcError(f"Disallowed constant type: {type(node.value).__name__}")

    if isinstance(node, ast.Name):
        if node.id not in values:
            raise CalcError(f"Unknown variable in expression: {node.id!r}")
        return values[node.id]

    if isinstance(node, ast.BinOp):
        op_fn = _ALLOWED_BINOPS.get(type(node.op))
        if op_fn is None:
            raise CalcError(f"Disallowed operator: {type(node.op).__name__}")
        left = _eval_node(node.left, values)
        right = _eval_node(node.right, values)
        if isinstance(node.op, ast.Div) and right == 0:
            raise CalcError("Division by zero in expression.")
        return op_fn(left, right)

    if isinstance(node, ast.UnaryOp):
        op_fn = _ALLOWED_UNARYOPS.get(type(node.op))
        if op_fn is None:
            raise CalcError(f"Disallowed unary operator: {type(node.op).__name__}")
        return op_fn(_eval_node(node.operand, values))

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_FUNCTIONS:
            func_repr = getattr(node.func, "id", type(node.func).__name__)
            raise CalcError(f"Disallowed function call: {func_repr!r}")
        if node.keywords:
            raise CalcError("Keyword arguments are not allowed in expressions.")
        args = [_eval_node(a, values) for a in node.args]
        return _ALLOWED_FUNCTIONS[node.func.id](*args)

    raise CalcError(f"Disallowed expression element: {type(node).__name__}")
