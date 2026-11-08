# eval_structured_v2.py  (v2.2)
# 修正：
#   1) Content_Correct 依赖 Schema_Valid（表格类不再「行数不对却算内容对」）
#   2) Markdown 只解析第一个表格块，遇空行/非表格行停止；并记录 extra_text（是否夹带解释）
#   3) 新增 Nested_OK（function 类：参数藏在 params/arguments/args 层也算合规）
#   4) 新增 Lenient_Correct（忽略大小写/空白/全半角的值比对，用于区分「翻译」与「错误」）
import json, os, re
import pandas as pd
import jsonschema

RAW_DIR    = r"C:\Users\cjy13\quant_eval\05_raw\structured_v2"
DATA_FILE  = r"C:\Users\cjy13\quant_eval\03_data\structured\struct_80_v2.jsonl"
OUT_SUMMARY = r"C:\Users\cjy13\quant_eval\06_results\table_struct_summary_v2.csv"
OUT_DETAIL  = r"C:\Users\cjy13\quant_eval\06_results\table_struct_detail_v2.csv"
os.makedirs(os.path.dirname(OUT_SUMMARY), exist_ok=True)

MODEL_MAP = {
    "ll32": "Llama-3.2-1B", "llama3.1-8b-instruct": "Llama-3.1-8B",
    "phi3": "Phi-3-mini", "qwen0.5b": "Qwen2.5-0.5B",
    "qwen1.5b": "Qwen2.5-1.5B", "qwen3b": "Qwen2.5-3B",
    "qwen2.5-7b-instruct": "Qwen2.5-7B",
}
def norm_model(m): return MODEL_MAP.get(m.strip().lower(), m.strip())
def norm_quant(q):
    q = q.strip().upper()
    return {"F16": "F16", "FP16": "F16", "BASE": "F16"}.get(q, q)

# ---------- JSON 候选提取 ----------
def extract_json_candidates(text):
    t = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", t, re.DOTALL)
    if m: t = m.group(1).strip()
    cands = []
    for oc, cc in (("{", "}"), ("[", "]")):
        idx = 0
        while True:
            s = t.find(oc, idx)
            if s == -1: break
            depth, in_str, esc, end = 0, False, False, -1
            for i in range(s, len(t)):
                ch = t[i]
                if in_str:
                    if esc: esc = False
                    elif ch == "\\": esc = True
                    elif ch == '"': in_str = False
                    continue
                if ch == '"': in_str = True
                elif ch == oc: depth += 1
                elif ch == cc:
                    depth -= 1
                    if depth == 0: end = i; break
            if end == -1: break
            cands.append(t[s:end+1]); idx = end + 1
    return cands if cands else [t]

def _as_obj(c):
    try:
        o = json.loads(c)
    except Exception:
        return None
    if isinstance(o, list) and o:
        o = o[0]
    return o

def choose_json(cands, required):
    """优先顶层 required 齐全；其次嵌套层（params/arguments/args/parameters）齐全；最后第一个可解析"""
    nested_ok = None
    fallback = None
    NEST_KEYS = ("params", "arguments", "args", "parameters", "input")
    for c in cands:
        o = _as_obj(c)
        if not isinstance(o, dict):
            if o is not None and fallback is None: fallback = o
            continue
        if all(k in o for k in required):
            return o, False
        for nk in NEST_KEYS:
            inner = o.get(nk)
            if isinstance(inner, dict) and all(k in inner for k in required):
                if nested_ok is None: nested_ok = (inner, True)
        if fallback is None: fallback = o
    if nested_ok: return nested_ok
    return fallback, False

# ---------- 表格解析（只取第一个块）----------
def parse_md_blocks(text):
    """按空行/非表格行切分，返回所有表格块（每块为行列表，分隔线已剔除）"""
    blocks, cur = [], []
    for line in text.splitlines():
        s = line.strip()
        if "|" in s:
            cur.append(s)
        else:
            if cur:
                blocks.append(cur); cur = []
    if cur:
        blocks.append(cur)
    out = []
    for b in blocks:
        rows = []
        for s in b:
            cells = [c.strip() for c in s.strip("|").split("|")]
            if cells and all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
                continue
            rows.append(cells)
        if len(rows) >= 2:
            out.append(rows)
    return out

def has_extra_text_md(text):
    """表格之外是否夹带解释性文字（已剔除表格行、分隔线、代码块标记）"""
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("```"):
            continue
        if "|" in s:
            continue
        if re.fullmatch(r"[:\-\|\s]+", s):   # 表格分隔线
            continue
        return 1
    return 0

def has_extra_text_json(text, cands):
    """JSON 片段之外是否夹带解释性文字"""
    rest = text
    for c in cands:
        rest = rest.replace(c, "")
    rest = re.sub(r"```(?:json)?", "", rest)
    return 1 if rest.strip() else 0

def norm_str(x):
    s = str(x).strip().lower()
    s = s.replace("　", " ")
    for a, b in (("，", ","), ("：", ":"), ("（", "("), ("）", ")")):
        s = s.replace(a, b)
    return re.sub(r"\s+", "", s)

def eq_strict(a, b):
    if isinstance(b, int):
        try: return int(str(a).strip()) == b
        except Exception: return False
    return str(a).strip() == str(b).strip()

def eq_lenient(a, b):
    if isinstance(b, int):
        try: return int(str(a).strip()) == b
        except Exception: return False
    return norm_str(a) == norm_str(b)

def content_ok_json(obj, expected, cmp):
    return isinstance(obj, dict) and all(k in obj and cmp(obj[k], v) for k, v in expected.items())

def content_ok_table(rows, headers, exp_rows, cmp):
    if len(rows) != 1 + len(exp_rows): return False
    if [c.strip() for c in rows[0]] != headers: return False
    for got, ex in zip(rows[1:], exp_rows):
        if len(got) != len(ex) or any(not cmp(g, e) for g, e in zip(got, ex)): return False
    return True

# ---------- 主流程 ----------
questions = {json.loads(l)["id"]: json.loads(l) for l in open(DATA_FILE, encoding="utf-8") if l.strip()}
details = []

for fname in sorted(os.listdir(RAW_DIR)):
    if not fname.endswith(".jsonl"): continue
    recs = [json.loads(l) for l in open(os.path.join(RAW_DIR, fname), encoding="utf-8") if l.strip()]
    if not recs: continue
    model = norm_model(recs[0].get("model", fname))
    quant = norm_quant(recs[0].get("quant", "UNKNOWN"))

    for r in recs:
        tid = r.get("id")
        if tid == "__ALL__" or tid not in questions: continue
        q = questions[tid]
        raw = (r.get("raw_output") or "").strip()
        status = r.get("status", "ok")

        p=f=sch=nn=c=l=ct=ex=0
        if status == "ok" and raw:
            if q["type"] == "markdown_table":
                blocks = parse_md_blocks(raw)
                rows = blocks[0] if blocks else []
                p = 1 if len(rows) >= 2 else 0
                f = 1 if (p and [c.strip() for c in rows[0]] == q["expected_headers"]) else 0
                sch = 1 if (f and len(rows) - 1 == len(q["expected_rows"])) else 0
                c = 1 if (sch and content_ok_table(rows, q["expected_headers"], q["expected_rows"], eq_strict)) else 0
                l = 1 if (sch and content_ok_table(rows, q["expected_headers"], q["expected_rows"], eq_lenient)) else 0
                ct = 1 if any(content_ok_table(b, q["expected_headers"], q["expected_rows"], eq_lenient) for b in blocks) else 0
                ex = 1 if len(blocks) > 1 else 0   # 表格类：以“是否输出多个表格块”判定夹带
            else:
                req = q["schema"].get("required", [])
                obj, nested = choose_json(extract_json_candidates(raw), req)
                if obj is not None:
                    p = 1
                    f = 1 if (isinstance(obj, dict) and all(k in obj for k in req)) else 0
                    nn = 1 if nested else 0
                    try:
                        jsonschema.validate(instance=obj, schema=q["schema"]); sch = 1
                    except Exception:
                        sch = 0
                    c = 1 if (sch and content_ok_json(obj, q.get("expected", {}), eq_strict)) else 0
                    l = 1 if (sch and content_ok_json(obj, q.get("expected", {}), eq_lenient)) else 0
                    ex = 1 if len(extract_json_candidates(raw)) > 1 else 0   # JSON 类：以“是否输出多个候选片段”判定冗余
                    for cand2 in extract_json_candidates(raw):
                        o2 = _as_obj(cand2)
                        if not isinstance(o2, dict):
                            continue
                        try:
                            jsonschema.validate(instance=o2, schema=q["schema"])
                        except Exception:
                            continue
                        if content_ok_json(o2, q.get("expected", {}), eq_lenient):
                            ct = 1
                            break

        details.append({"Model": model, "Quant": quant, "Task_ID": tid, "Type": q["type"],
                        "Status": status, "Parseable": p, "Fields_Complete": f, "Nested_OK": nn,
                        "Schema_Valid": sch, "Content_Correct": c, "Lenient_Correct": l,
                        "Contains_Correct": ct, "Extra_Text": ex})

df = pd.DataFrame(details)
df.to_csv(OUT_DETAIL, index=False, encoding="utf-8-sig")

g = df.groupby(["Model", "Quant"]).agg(
    N_Tasks=("Task_ID", "count"),
    N_Empty=("Status", lambda s: int((s != "ok").sum())),
    Parseable=("Parseable", "mean"), Fields=("Fields_Complete", "mean"),
    Schema=("Schema_Valid", "mean"), Content=("Content_Correct", "mean"),
    Lenient=("Lenient_Correct", "mean"),
    Contains=("Contains_Correct", "mean"),
    Extra=("Extra_Text", "mean"),
).reset_index()
for k, v in (("Parseable(%)", "Parseable"), ("Fields(%)", "Fields"), ("Schema(%)", "Schema"),
             ("Content(%)", "Content"), ("Lenient(%)", "Lenient"),
             ("Contains(%)", "Contains"), ("Extra(%)", "Extra")):
    g[k] = (g[v] * 100).round(1)
g["Note"] = g["N_Empty"].apply(lambda n: "run_incomplete" if n > 0 else "")
g = g.drop(columns=["Parseable", "Fields", "Schema", "Content", "Lenient", "Contains", "Extra"])
g = g[["Model", "Quant", "N_Tasks", "N_Empty", "Parseable(%)", "Fields(%)", "Schema(%)",
       "Content(%)", "Contains(%)", "Extra(%)", "Note"]]
g.to_csv(OUT_SUMMARY, index=False, encoding="utf-8-sig")
print(g.to_string(index=False))
print("\n汇总:", OUT_SUMMARY, "\n明细:", OUT_DETAIL)
