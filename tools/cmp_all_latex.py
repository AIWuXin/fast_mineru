# -*- coding: utf-8 -*-
"""对比两个输出目录里所有公式(含行内 inline_equation)的 latex。深挖 middle.json。"""
import io, sys, json, glob, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

def collect(out_dir):
    js = glob.glob(os.path.join(out_dir, "**", "*_middle.json"), recursive=True)
    if not js:
        print(f"  !! 无 middle.json: {out_dir}"); return []
    data = json.load(open(js[0], encoding="utf-8"))
    forms = []
    def walk(o):
        if isinstance(o, dict):
            t = o.get("type", "")
            if t in ("inline_equation", "interline_equation") and "content" in o:
                forms.append(o["content"])
            # span 里也可能有 latex
            if o.get("type") in ("interline_equation", "inline_equation") and "latex" in o:
                forms.append(o["latex"])
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(data)
    return forms

a, b = sys.argv[1], sys.argv[2]
A, B = collect(a), collect(b)
print(f"[{a}] 公式数={len(A)}")
print(f"[{b}] 公式数={len(B)}")
n = min(len(A), len(B))
same = sum(1 for i in range(n) if A[i] == B[i])
print(f"完全一致: {same}/{n}")
diffs = [i for i in range(n) if A[i] != B[i]]
for i in diffs[:20]:
    print("=" * 70)
    print(f"#{i}\n  torch: {A[i][:220]}\n  trt  : {B[i][:220]}")
if len(A) != len(B):
    print(f"!! 数量不同 {len(A)} vs {len(B)}")
