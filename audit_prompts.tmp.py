"""Find long string literals that flow into an LLM call — i.e. prompts living in code."""
import ast, sys
from pathlib import Path

LLM_MARKERS = ("tracked_claude_call_async", "messages.create", "anthropic.com/v1/messages",
               "claude_helper", "_call_claude", "call_claude")
PROMPT_HINTS = ("you are", "respond", "return json", "extract", "analyze", "analyse",
                "classify", "json object", "task:", "instructions", "rules:")

def static_len(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return len(node.value), node.value
    if isinstance(node, ast.JoinedStr):
        txt = "".join(v.value for v in node.values
                      if isinstance(v, ast.Constant) and isinstance(v.value, str))
        return len(txt), txt
    return 0, ""

rows = []
for path in sorted(Path("app").rglob("*.py")):
    src = path.read_text(encoding="utf-8")
    if not any(m in src for m in LLM_MARKERS):
        continue
    try: tree = ast.parse(src)
    except SyntaxError: continue
    # docstrings are not prompts
    docstrings = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if n.body and isinstance(n.body[0], ast.Expr):
                docstrings.add(id(n.body[0].value))
    for n in ast.walk(tree):
        if id(n) in docstrings: continue
        ln, txt = static_len(n)
        if ln < 220: continue
        low = txt.lower()
        if sum(h in low for h in PROMPT_HINTS) < 2: continue
        rows.append((str(path).replace("\\","/"), n.lineno, ln, " ".join(txt.split())[:88]))

rows.sort(key=lambda r: -r[2])
print(f"{len(rows)} candidate in-code prompts\n")
for f, line, ln, preview in rows:
    print(f"{ln:6}  {f}:{line}\n        {preview}")
