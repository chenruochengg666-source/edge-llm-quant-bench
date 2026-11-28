# -*- coding: utf-8 -*-
"""
ARC 统一协议重跑（ARC-Easy / ARC-Challenge）+ FP16 缺失题补跑与一致性抽检
=======================================================================
修正 run_arc_full.py 中 FP16 与量化档评测协议不一致的问题：
  原脚本：FP16 追加“只输出一个大写字母”约束、max_tokens=64；
          量化档无约束、max_tokens=8；答案提取用全文 \\b[A-E]\\b；数字标签题（1~4）未转换；
          无预热，服务刚启动时的请求全部失败（各文件开头连续空输出）。
  本脚本：所有档位、两个题库使用同一协议——
          · 所有档位追加同一约束句
          · max_tokens=64，temperature=0，top_p=1，seed=42
          · 答案提取统一用 extract_letter()（与 run_arc_challenge.py 相同）
          · 数字标签 1~4 转换为 A~D
          · 启动后先预热，空输出自动重试 3 次

FP16（ARC-Easy）不整体重跑：原 FP16 协议本来就与统一协议一致，缺的只是开头因模型未加载完而失败的题。
  · 把原 results/raw/arc/<fam>-f16 中有输出的记录复制到新文件（标记 source=original，答案按新规则重新提取）
  · 只补跑缺失题（source=rerun）
  · 另从原有题中固定抽 100 题重跑，与原输出逐题比对（写入 <模型>.audit.jsonl，不计入正式结果）

输出（不覆盖原始数据）：
  ARC-Easy      → results/raw/arc_unified/<模型>.gguf.jsonl          （FP16 抽检：<模型>.audit.jsonl）
  ARC-Challenge → results/raw/arc_challenge_unified/<模型>.gguf.jsonl
  运行记录      → 各目录下 _run_log.csv；FP16 抽检汇总 → arc_unified/_fp16_audit_summary.csv

推荐用法（在仓库根目录下，先设置环境变量 LLAMA_CPP_BIN 与 EDGE_LLM_MODEL_DIR）：
  python scripts/02_evaluation/arc_unified/run_arc_easy_unified.py --check       # 只检查文件、打印计划与预计耗时，不跑模型
  python -u scripts/02_evaluation/arc_unified/run_arc_easy_unified.py --plan A   # 白天：ARC-Easy 优先 6 档 + FP16 补跑/抽检 + 0.5B Challenge（约 4 小时）
  python -u scripts/02_evaluation/arc_unified/run_arc_easy_unified.py --plan B   # 晚上：0.5B 自由作答对照 + 其余 Qwen 档 + Phi-3 Challenge + 两个 Q2_K（约 9 小时）
  python scripts/02_evaluation/arc_unified/run_arc_easy_unified.py --task freeform  # 只跑 0.5B 自由作答对照（输出到 results/raw/arc_freeform）
也可单独指定：
  python scripts/02_evaluation/arc_unified/run_arc_easy_unified.py --task challenge --models qwen0.5b-f16 qwen0.5b-Q4_K_M
  python scripts/02_evaluation/arc_unified/run_arc_easy_unified.py --task fill --models phi3-f16
中断后重新执行同一命令即可续跑；已有输出的题会跳过，空输出的题会重试。
运行期间会阻止电脑自动睡眠（屏幕可以关闭），结束后自动恢复。
"""
import argparse, csv, json, os, random, re, subprocess, sys, time
import requests

# ================= 避免 Clash / 系统代理干扰（同 run_arc_challenge.py） =================
for _k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
    os.environ.pop(_k, None)
os.environ["NO_PROXY"] = "127.0.0.1,localhost,0.0.0.0"
os.environ["no_proxy"] = "127.0.0.1,localhost,0.0.0.0"
NO_PROXY_ARG = {"http": None, "https": None}

# ---------- 路径配置（相对仓库根目录；模型与 llama.cpp 位置用环境变量指定） ----------
from pathlib import Path as _Path
REPO = _Path(__file__).resolve().parents[3]
DATA_DIR = os.environ.get("EDGE_LLM_DATA_DIR", str(REPO / "data"))   # 题库：data/arc/*.jsonl（仓库不含，见 data/README.md）
RAW_DIR = str(REPO / "results" / "raw")              # 逐题原始输出
SUMMARY_DIR = str(REPO / "results" / "summary")      # 汇总表
LLAMA_BIN = os.environ.get("LLAMA_CPP_BIN", str(REPO.parent / "llama.cpp" / "build" / "bin" / "Release"))
MODEL_DIRS = [p for p in os.environ.get("EDGE_LLM_MODEL_DIR", str(REPO / "models")).split(os.pathsep) if p]
# 示例（PowerShell）：$env:LLAMA_CPP_BIN="D:\llama.cpp\build\bin\Release"; $env:EDGE_LLM_MODEL_DIR="D:\gguf"
# 注意：llama-bench 无法读取含中文字符的模型路径，模型目录请使用纯英文路径。
# -----------------------------------------------------------------------------------

# ================= 配置区 =================
LLAMA_SERVER = os.path.join(LLAMA_BIN, "llama-server.exe")
TASKS = {
    "easy": {"data": os.path.join(DATA_DIR, "arc", "ARC-Easy-test-full.jsonl"),
             "out": os.path.join(RAW_DIR, "arc_unified"), "n": 2376},
    "challenge": {"data": os.path.join(DATA_DIR, "arc", "ARC-Challenge-test-full.jsonl"),
                  "out": os.path.join(RAW_DIR, "arc_challenge_unified"), "n": 1172},
    # 自由作答对照：不加“只输出一个字母”约束（即原 run_arc_full.py 量化档的提示词），max_tokens 同为 64
    "freeform": {"data": os.path.join(DATA_DIR, "arc", "ARC-Easy-test-full.jsonl"),
                 "out": os.path.join(RAW_DIR, "arc_freeform"), "n": 2376},
}
OLD_EASY_DIR = os.path.join(RAW_DIR, "arc")      # 原 ARC-Easy 输出（只读）
MAX_TOKENS = 64
THREADS = "8"
RETRIES = 3
AUDIT_N = 100
AUDIT_SEED = 20261127
PROTOCOL = "unified_v1: constraint_all + max_tokens64 + T0 + extract_letter + digit_label_fix"
PROTOCOL_FREE = "freeform_v1: no_constraint + max_tokens64 + T0 + extract_letter + digit_label_fix"
CONSTRAINT = "只输出一个大写字母 (A/B/C/D)，不要解释，不要输出任何其他内容：\n"

EASY_PRIORITY = ["qwen0.5b-Q4_K_M", "qwen0.5b-Q2_K", "ll32-Q4_K_M", "ll32-Q2_K", "phi3-Q4_K_M", "phi3-Q2_K"]
EASY_QWEN = ["qwen1.5b-Q4_K_M", "qwen1.5b-Q3_K_M", "qwen1.5b-Q5_K_M", "qwen1.5b-Q6_K", "qwen1.5b-Q8_0",
             "qwen3b-Q4_K_M", "qwen3b-Q3_K_M"]
EASY_Q2 = ["qwen1.5b-Q2_K", "qwen3b-Q2_K"]
EASY_FP16 = ["qwen0.5b-f16", "ll32-f16", "phi3-f16", "qwen1.5b-f16", "qwen3b-f16"]
CH_05B = ["qwen0.5b-f16", "qwen0.5b-Q4_K_M", "qwen0.5b-Q2_K"]
CH_PHI3 = ["phi3-f16", "phi3-Q4_K_M", "phi3-Q2_K"]
FREE_05B = ["qwen0.5b-f16", "qwen0.5b-Q4_K_M"]   # 答题格式 × 量化精度 2×2 对照
FILL_7B = ["qwen2.5-7b-instruct-q4_k_m"]          # 7B 子集开头 32 题因无预热而空输出，按原 7B 协议补跑
SUB7B_DIR = os.path.join(RAW_DIR, "arc_7b_subset")
SUB7B_OUT = os.path.join(RAW_DIR, "arc_7b_subset_fill")
AUDIT_7B_N = 30
PLANS = {
    "A": [("easy", EASY_PRIORITY), ("fill", EASY_FP16), ("challenge", CH_05B)],
    "B": [("freeform", FREE_05B), ("fill7b", FILL_7B), ("easy", EASY_QWEN), ("challenge", CH_PHI3), ("easy", EASY_Q2)],
}
SMOKE = ("easy", "qwen0.5b-Q4_K_M", 20)   # 计划 A 开头的冒烟测试；这 20 题计入正式结果

# 预计每题秒数（依据 results/raw/arc 与 arc_challenge 日志的文件时间估算），仅用于打印计划
EST_SPQ = {"qwen0.5b-f16": 0.22, "qwen0.5b-Q4_K_M": 0.25, "qwen0.5b-Q2_K": 0.30,
           "ll32-f16": 0.71, "ll32-Q4_K_M": 0.35, "ll32-Q2_K": 0.76,
           "phi3-f16": 2.46, "phi3-Q4_K_M": 1.20, "phi3-Q2_K": 1.50,
           "qwen1.5b-f16": 0.78, "qwen1.5b-Q8_0": 0.74, "qwen1.5b-Q6_K": 0.48, "qwen1.5b-Q5_K_M": 0.46,
           "qwen1.5b-Q4_K_M": 0.44, "qwen1.5b-Q3_K_M": 0.41, "qwen1.5b-Q2_K": 1.67,
           "qwen3b-f16": 1.30, "qwen3b-Q4_K_M": 0.85, "qwen3b-Q3_K_M": 0.80, "qwen3b-Q2_K": 2.50}
EST_FREE = {"qwen0.5b-f16": 1.5, "qwen0.5b-Q4_K_M": 1.0}   # 自由作答输出更长，按 64 token 上限估算
EST_7B = 6.0                                                # 7B Q4_K_M 每题约 6 秒（预填充 + 16 token）


def spq(task, name):
    if task == "fill7b":
        return EST_7B
    return EST_FREE.get(name, 1.5) if task == "freeform" else EST_SPQ.get(name, 1.0)
# ==========================================


def find_model(name):
    for d in MODEL_DIRS:
        p = os.path.join(d, name + ".gguf")
        if os.path.exists(p):
            return p
    return None


def is_fp16(name):
    n = name.lower()
    return "f16" in n or "base" in n


def keep_awake(on):
    """运行期间阻止 Windows 自动睡眠（屏幕可关）。"""
    try:
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0))
    except Exception:
        pass


def kill_all_servers():
    os.system('taskkill /F /IM llama-server.exe > nul 2>&1')
    time.sleep(5)


def start_server(model_path):
    kill_all_servers()
    port = random.randint(8100, 9000)
    ctx = "2048" if is_fp16(os.path.basename(model_path)) else "4096"   # 提示词 < 600 token，ctx 不影响结果
    cmd = [LLAMA_SERVER, "-m", model_path, "-c", ctx, "-t", THREADS, "--port", str(port)]
    print(f"  [启动服务] {os.path.basename(model_path)}  端口 {port}  ctx={ctx}")
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            creationflags=subprocess.CREATE_NO_WINDOW)
    for _ in range(90):
        try:
            requests.get(f"http://127.0.0.1:{port}/health", timeout=2, proxies=NO_PROXY_ARG)
            return proc, port
        except Exception:
            time.sleep(1)
    raise RuntimeError(f"服务启动超时: {model_path}")


def warmup(port, max_wait=300):
    t0 = time.time()
    while time.time() - t0 < max_wait:
        try:
            r = requests.post(f"http://127.0.0.1:{port}/v1/completions",
                              json={"prompt": "Hello", "max_tokens": 1, "temperature": 0},
                              timeout=30, proxies=NO_PROXY_ARG)
            if r.status_code == 200:
                print(f"  [预热] 就绪，耗时 {time.time() - t0:.1f} 秒")
                return True
        except Exception:
            pass
        time.sleep(5)
    return False


def extract_letter(ans: str) -> str:
    """与 run_arc_challenge.py 完全相同的答案提取逻辑（宁可返回 ?，不误判）。"""
    if not ans:
        return "?"
    a = ans.strip().upper()
    if a in ("A", "B", "C", "D", "E"):
        return a
    m = re.match(r"^[\s\.\:\,\u3001\uff0c\uff1a\-\*\uff08\(]*([A-E])(?![A-Z])", a)
    if m:
        return m.group(1)
    patterns = [
        r"(?:ANSWER\s*(?:IS|:))\s*\(?([A-E])\)?",
        r"(?:答案\s*(?:是|为)?\s*[：:])\s*\(?([A-E])\)?",
        r"(?:答案\s*(?:是|为)\s*)\(?([A-E])\)?(?![\u4e00-\u9fff])",
        r"^\(([A-E])\)",
        r"^([A-E])[\.\:、，,]",
        r"(?:选项|OPTION|CHOICE)\s*([A-E])\s*(?:正确|IS|CORRECT)",
    ]
    for pat in patterns:
        m = re.search(pat, a)
        if m:
            return m.group(1)
    if len(a) <= 3:
        m = re.search(r"\b([A-E])\b", a)
        if m:
            return m.group(1)
    return "?"


def build_prompt(q, model_name, constrained=True):
    q_text = q["question"] + "\n"
    labels = [chr(65 + i) for i in range(len(q["choices"]["text"]))]
    for j, opt in enumerate(q["choices"]["text"]):
        q_text += f"{labels[j]}. {opt}\n"
    if constrained:
        q_text += CONSTRAINT                              # ← 所有档位统一追加（自由作答对照除外）
    n = model_name.lower()
    if "phi3" in n:
        return f"<|user|>\n{q_text}答案: <|end|>\n<|assistant|>\n", labels
    if "ll32" in n or "llama" in n:
        return ("<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                f"{q_text}答案: <|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"), labels
    if "qwen" in n:
        return f"<|im_start|>user\n{q_text}答案：<|im_end|>\n<|im_start|>assistant\n", labels
    return q_text + "答案：", labels


def norm_gold(q, labels):
    g = str(q["answerKey"]).strip().upper()
    if g.isdigit() and 0 < int(g) <= len(labels):         # 数字标签 1~4 → A~D
        g = labels[int(g) - 1]
    return g


def ask(port, prompt):
    for attempt in range(1, RETRIES + 1):
        try:
            r = requests.post(f"http://127.0.0.1:{port}/v1/completions",
                              json={"prompt": prompt, "max_tokens": MAX_TOKENS,
                                    "temperature": 0, "top_p": 1, "seed": 42},
                              timeout=90, proxies=NO_PROXY_ARG)
            j = r.json()
            if j.get("choices"):
                txt = j["choices"][0]["text"].strip()
                if txt:
                    return txt, attempt
        except Exception as e:
            print(f"      [重试 {attempt}/{RETRIES}] {e}")
        time.sleep(3)
    return "", RETRIES


def load_questions(task):
    with open(TASKS[task]["data"], encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_jsonl(path):
    out = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        out.append(json.loads(line))
                    except Exception:
                        pass
    return out


def log_run(task, row):
    path = os.path.join(TASKS[task]["out"], "_run_log.csv")
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["task", "model", "start", "end", "minutes", "n_done", "n_correct", "n_qmark", "n_empty", "protocol"])
        w.writerow([task] + row)


# ---------------- FP16 补跑与抽检 ----------------
def old_fp16_valid(name):
    """原 results/raw/arc 中该 FP16 有输出的记录 {id: raw}（同题多条取最后一条有输出的）。"""
    recs = {}
    for it in read_jsonl(os.path.join(OLD_EASY_DIR, f"{name}.gguf.jsonl")):
        raw = str(it.get("raw", "") or "").strip()
        if raw:
            recs[it["id"]] = raw
    return recs


def seed_fp16(name, questions):
    """首次运行时，把原 FP16 有输出的记录复制进新文件（按新规则重新提取答案、修正标签）。"""
    out_file = os.path.join(TASKS["easy"]["out"], f"{name}.gguf.jsonl")
    if os.path.exists(out_file):
        return
    os.makedirs(TASKS["easy"]["out"], exist_ok=True)
    old = old_fp16_valid(name)
    n = 0
    with open(out_file, "w", encoding="utf-8") as f:
        for q in questions:
            if q["id"] not in old:
                continue
            raw = old[q["id"]]
            labels = [chr(65 + i) for i in range(len(q["choices"]["text"]))]
            pred, gold = extract_letter(raw), norm_gold(q, labels)
            rec = {"id": q["id"], "pred": pred, "gold": gold, "ok": pred == gold, "raw": raw,
                   "attempts": 0, "task": "easy", "source": "original", "protocol": PROTOCOL}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n"); n += 1
    print(f"  [补跑准备] {name}：从原始文件复制 {n} 条有效记录，缺失 {len(questions) - n} 题待补跑")


def audit_ids(name, questions):
    """从原 FP16 有输出的题中固定抽 AUDIT_N 题（按题库顺序）。"""
    old = old_fp16_valid(name)
    pool = [q["id"] for q in questions if q["id"] in old]
    rng = random.Random(AUDIT_SEED)
    chosen = set(rng.sample(pool, min(AUDIT_N, len(pool))))
    return [q for q in questions if q["id"] in chosen], old


def run_model(task, name, questions, limit=0, audit=None):
    """返回本次运行的 (n_done, n_ok, n_qmark, n_empty, 前几条原始输出)。audit=(题目列表, 原始输出字典)。"""
    path = find_model(name)
    if path is None:
        print(f"❌ 找不到模型文件：{name}.gguf（已搜索 {MODEL_DIRS}）")
        return None
    out_task = "easy" if task == "fill" else task
    os.makedirs(TASKS[out_task]["out"], exist_ok=True)
    out_file = os.path.join(TASKS[out_task]["out"], f"{name}.gguf.jsonl")
    done = {it["id"] for it in read_jsonl(out_file) if str(it.get("raw", "")).strip()}
    todo = [q for q in questions if q["id"] not in done]
    if limit:
        todo = todo[:max(0, limit - len(done))]
    audit_file = os.path.join(TASKS["easy"]["out"], f"{name}.audit.jsonl")
    audit_todo = []
    if audit:
        a_done = {it["id"] for it in read_jsonl(audit_file) if str(it.get("raw_new", "")).strip()}
        audit_todo = [q for q in audit[0] if q["id"] not in a_done]
    print(f"\n=== [{task}] {name}：已完成 {len(done)} 题，本次待跑 {len(todo)} 题"
          + (f"，一致性抽检待跑 {len(audit_todo)} 题" if audit else "") + " ===")
    if not todo and not audit_todo:
        return (0, 0, 0, 0, [])

    t0 = time.time(); start = time.strftime("%Y-%m-%d %H:%M:%S")
    proc, port = start_server(path)
    if not warmup(port):
        print(f"⚠️ {name} 预热超时，跳过")
        proc.terminate(); kill_all_servers(); return None

    n_ok = n_q = n_e = 0; samples = []
    source = {"fill": "rerun", "freeform": "freeform"}.get(task, "unified")
    proto = PROTOCOL_FREE if task == "freeform" else PROTOCOL
    for k, q in enumerate(todo, 1):
        prompt, labels = build_prompt(q, name, constrained=(task != "freeform"))
        ans, attempts = ask(port, prompt)
        pred = extract_letter(ans)
        gold = norm_gold(q, labels)
        ok = (pred == gold)
        n_ok += ok; n_q += (pred == "?"); n_e += (not ans)
        if len(samples) < 6:
            samples.append(ans[:60].replace("\n", "\\n"))
        rec = {"id": q["id"], "pred": pred, "gold": gold, "ok": ok, "raw": ans,
               "attempts": attempts, "task": out_task, "source": source, "protocol": proto}
        with open(out_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if k % 100 == 0 or k == len(todo):
            el = time.time() - t0
            print(f"    {k}/{len(todo)} | 本次准确率 {n_ok / k:.4f} | 未提取 {n_q} | 空输出 {n_e} | "
                  f"已用 {el / 60:.1f} 分，预计剩余 {el / k * (len(todo) - k) / 60:.1f} 分")

    if audit_todo:
        print(f"    [一致性抽检] 重跑 {len(audit_todo)} 道原已有输出的题 ...")
        for q in audit_todo:
            prompt, labels = build_prompt(q, name)
            ans, attempts = ask(port, prompt)
            raw_orig = audit[1][q["id"]]; gold = norm_gold(q, labels)
            p_new, p_old = extract_letter(ans), extract_letter(raw_orig)
            rec = {"id": q["id"], "gold": gold, "raw_orig": raw_orig, "raw_new": ans,
                   "pred_orig": p_old, "pred_new": p_new, "same_raw": raw_orig == ans,
                   "same_pred": p_old == p_new, "ok_orig": p_old == gold, "ok_new": p_new == gold}
            with open(audit_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        a = read_jsonl(audit_file)
        print(f"    [一致性抽检] {len(a)} 题：答案一致 {sum(x['same_pred'] for x in a)}，原始输出完全一致 {sum(x['same_raw'] for x in a)}")

    proc.terminate(); kill_all_servers()
    mins = (time.time() - t0) / 60
    log_run(out_task, [name if task != "fill" else name + "(fill)", start, time.strftime("%Y-%m-%d %H:%M:%S"),
                       f"{mins:.1f}", len(todo), n_ok, n_q, n_e, proto])
    print(f"✅ [{task}] {name} 完成：补/跑 {len(todo)} 题，答对 {n_ok}，未提取 {n_q}，空输出 {n_e}，用时 {mins:.1f} 分")
    return (len(todo), n_ok, n_q, n_e, samples)


def fill_fp16(name, questions):
    seed_fp16(name, questions)
    aq, old = audit_ids(name, questions)
    return run_model("fill", name, questions, audit=(aq, old))


def fill_7b(name):
    """按原 run_arc_7b_subset 协议（约束句前后带换行、句号结尾，max_tokens=16，ctx=4096）补跑开头空输出的题。
    原文件只读；结果写入 results/raw/arc_7b_subset_fill/：原有输出复制（source=original），缺失题补跑（source=rerun），
    另抽 AUDIT_7B_N 道原有输出的题重跑比对（<模型>.audit.jsonl）。"""
    src = read_jsonl(os.path.join(SUB7B_DIR, f"{name}.gguf.jsonl"))
    if not src:
        print(f"❌ 找不到原 7B 子集输出：{name}"); return
    qmap = {q["id"]: q for q in load_questions("easy")}
    os.makedirs(SUB7B_OUT, exist_ok=True)
    out_file = os.path.join(SUB7B_OUT, f"{name}.gguf.jsonl"); audit_file = os.path.join(SUB7B_OUT, f"{name}.audit.jsonl")
    if not os.path.exists(out_file):
        with open(out_file, "w", encoding="utf-8") as f:
            for x in src:
                if str(x.get("raw", "")).strip():
                    f.write(json.dumps(dict(x, source="original"), ensure_ascii=False) + "\n")
    done = {x["id"] for x in read_jsonl(out_file) if str(x.get("raw", "")).strip()}
    todo = [x["id"] for x in src if x["id"] not in done]
    valid = [x for x in src if str(x.get("raw", "")).strip()]
    a_done = {x["id"] for x in read_jsonl(audit_file)}
    a_todo = [x for x in random.Random(AUDIT_SEED).sample(valid, min(AUDIT_7B_N, len(valid))) if x["id"] not in a_done]
    print(f"\n=== [fill7b] {name}：缺失 {len(todo)} 题待补跑，抽检 {len(a_todo)} 题 ===")
    if not todo and not a_todo:
        return
    path = find_model(name)
    if path is None:
        print(f"❌ 找不到模型文件：{name}.gguf"); return
    t0 = time.time(); start = time.strftime("%Y-%m-%d %H:%M:%S")
    kill_all_servers()
    port = random.randint(8100, 9000)
    proc = subprocess.Popen([LLAMA_SERVER, "-m", path, "-c", "4096", "-t", THREADS, "--port", str(port)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW)
    for _ in range(120):
        try:
            requests.get(f"http://127.0.0.1:{port}/health", timeout=2, proxies=NO_PROXY_ARG); break
        except Exception:
            time.sleep(1)
    if not warmup(port, max_wait=600):
        print("⚠️ 7B 预热超时，跳过"); proc.terminate(); kill_all_servers(); return

    def ask7b(q):
        labels = [chr(65 + i) for i in range(len(q["choices"]["text"]))]
        q_text = q["question"] + "\n" + "".join(f"{labels[j]}. {o}\n" for j, o in enumerate(q["choices"]["text"]))
        q_text += "\n只输出一个大写字母 (A/B/C/D)，不要解释，不要输出任何其他内容。\n"
        prompt = f"<|im_start|>user\n{q_text}答案：<|im_end|>\n<|im_start|>assistant\n"
        for attempt in range(1, RETRIES + 1):
            try:
                r = requests.post(f"http://127.0.0.1:{port}/v1/completions",
                                  json={"prompt": prompt, "max_tokens": 16, "temperature": 0, "top_p": 1, "seed": 42},
                                  timeout=120, proxies=NO_PROXY_ARG)
                j = r.json()
                if j.get("choices"):
                    t = j["choices"][0]["text"].strip()
                    if t:
                        return t, labels
            except Exception as e:
                print(f"      [重试 {attempt}/{RETRIES}] {e}")
            time.sleep(3)
        return "", labels

    n_ok = 0
    for k, i in enumerate(todo, 1):
        q = qmap[i]; ans, labels = ask7b(q); pred = extract_letter(ans); gold = norm_gold(q, labels)
        n_ok += (pred == gold)
        with open(out_file, "a", encoding="utf-8") as f:
            f.write(json.dumps({"id": i, "pred": pred, "gold": gold, "ok": pred == gold, "raw": ans,
                                "source": "rerun", "protocol": "orig7b_v1: constraint7b + max_tokens16 + T0"}, ensure_ascii=False) + "\n")
        print(f"    补跑 {k}/{len(todo)}", end="\r")
    for x in a_todo:
        q = qmap[x["id"]]; ans, labels = ask7b(q); gold = norm_gold(q, labels)
        rec = {"id": x["id"], "gold": gold, "raw_orig": x["raw"], "raw_new": ans,
               "pred_orig": extract_letter(x["raw"]), "pred_new": extract_letter(ans)}
        rec.update(same_raw=rec["raw_orig"] == ans, same_pred=rec["pred_orig"] == rec["pred_new"])
        with open(audit_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    proc.terminate(); kill_all_servers()
    a = read_jsonl(audit_file)
    print(f"\n✅ [fill7b] {name}：补跑 {len(todo)} 题（答对 {n_ok}），抽检 {len(a)} 题答案一致 {sum(x['same_pred'] for x in a)}，"
          f"用时 {(time.time() - t0) / 60:.1f} 分")
    log_run("easy", [name + "(fill7b)", start, time.strftime("%Y-%m-%d %H:%M:%S"), f"{(time.time() - t0) / 60:.1f}",
                     len(todo), n_ok, 0, 0, "orig7b_v1"])


def audit_summary():
    rows = []
    for name in EASY_FP16:
        a = read_jsonl(os.path.join(TASKS["easy"]["out"], f"{name}.audit.jsonl"))
        if not a:
            continue
        n = len(a)
        rows.append([name, n, sum(x["same_pred"] for x in a), sum(x["same_raw"] for x in a),
                     sum(x["ok_orig"] for x in a), sum(x["ok_new"] for x in a)])
    if not rows:
        return
    print("\n===== FP16 一致性抽检汇总 =====")
    print("模型 | 抽检题数 | 答案一致 | 原始输出完全一致 | 原答对 | 新答对")
    for r in rows:
        print(" | ".join(str(x) for x in r))
    tot = sum(r[1] for r in rows); same = sum(r[2] for r in rows)
    print(f"合计 {tot} 题，答案一致率 {same / tot * 100:.1f}%")
    if same / tot < 0.98:
        print("⚠️ 一致率低于 98%，说明运行环境与原实验有差异，建议把 FP16 整体重跑（--task easy --include-fp16 前先清空对应文件）。")
    with open(os.path.join(TASKS["easy"]["out"], "_fp16_audit_summary.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["model", "n_audit", "same_pred", "same_raw", "ok_orig", "ok_new"]); w.writerows(rows)


# ---------------- 计划与检查 ----------------
def step_count(task, name):
    if task == "fill7b":
        r = read_jsonl(os.path.join(SUB7B_DIR, f"{name}.gguf.jsonl"))
        return sum(1 for x in r if not str(x.get("raw", "")).strip()) + AUDIT_7B_N
    if task == "fill":
        return TASKS["easy"]["n"] - len(old_fp16_valid(name)) + AUDIT_N
    return TASKS[task]["n"]


def estimate(steps):
    sec = 0
    for task, names in steps:
        for n in names:
            sec += spq(task, n) * step_count(task, n) + 60
    return sec / 3600


def check(steps_by_plan):
    ok = True
    for label, p in (("llama-server", LLAMA_SERVER), ("ARC-Easy 题库", TASKS["easy"]["data"]),
                     ("ARC-Challenge 题库", TASKS["challenge"]["data"])):
        e = os.path.exists(p); ok &= e
        print(f"  {'✓' if e else '✗'} {label}: {p}")
    for plan, steps in steps_by_plan.items():
        print(f"\n计划 {plan}（预计约 {estimate(steps):.1f} 小时）")
        for task, names in steps:
            for n in names:
                p = find_model(n); ok &= bool(p)
                extra = ""
                if task == "fill7b":
                    extra = f"  缺失 {step_count(task, n) - AUDIT_7B_N} 题 + 抽检 {AUDIT_7B_N} 题"
                if task == "fill":
                    op = os.path.join(OLD_EASY_DIR, f"{n}.gguf.jsonl"); e = os.path.exists(op); ok &= e
                    extra = f"  缺失 {TASKS['easy']['n'] - len(old_fp16_valid(n))} 题 + 抽检 {AUDIT_N} 题" if e else "  ✗ 找不到原始 FP16 输出"
                mins = spq(task, n) * step_count(task, n) / 60
                print(f"  {'✓' if p else '✗'} [{task:9s}] {n:18s} 约 {mins:5.0f} 分{extra}  {p or '找不到模型文件'}")
    print("\n检查通过，可以开始。" if ok else "\n有文件缺失，请先处理再运行。")
    return ok


def run_plan(plan):
    steps = PLANS[plan]
    if not check({plan: steps}):
        sys.exit(1)
    keep_awake(True)
    try:
        if plan == "A":
            task, name, n = SMOKE
            print(f"\n----- 冒烟测试：[{task}] {name} 前 {n} 题 -----")
            r = run_model(task, name, load_questions(task), limit=n)
            if r is None:
                print("冒烟测试失败：模型无法启动。"); sys.exit(1)
            done, n_ok, n_q, n_e, samples = r
            if done == 0:
                print("  冒烟测试此前已通过，跳过。")
            else:
                print("  原始输出示例：", samples)
                if n_e > 2 or n_q > 2:
                    print(f"\n⚠️ 冒烟测试异常：{done} 题中未提取 {n_q}、空输出 {n_e}。已停止，请把这段输出发给助手。")
                    sys.exit(1)
                print("  冒烟测试正常，继续正式运行。")
        for task, names in steps:
            qs = load_questions("easy" if task in ("fill", "fill7b") else task)
            for name in names:
                if task == "fill":
                    fill_fp16(name, qs)
                elif task == "fill7b":
                    fill_7b(name)
                else:
                    run_model(task, name, qs)
        audit_summary()
    finally:
        keep_awake(False)
        kill_all_servers()
    print(f"\n计划 {plan} 全部完成。可以告诉助手“计划 {plan} 跑完了”。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", choices=["A", "B"], help="按预设计划运行")
    ap.add_argument("--check", action="store_true", help="只检查文件并打印计划与预计耗时")
    ap.add_argument("--task", choices=["easy", "challenge", "fill", "freeform"], default="easy")
    ap.add_argument("--models", nargs="*", help="只跑指定档位，如 qwen0.5b-Q4_K_M")
    ap.add_argument("--include-fp16", action="store_true", help="ARC-Easy 连 FP16 一起整体重跑")
    ap.add_argument("--limit", type=int, default=0, help="每档最多跑多少题（冒烟测试用）")
    args = ap.parse_args()

    if args.check:
        check(PLANS); return
    if args.plan:
        run_plan(args.plan); return

    if args.task == "fill":
        names = args.models or EASY_FP16
    elif args.task == "freeform":
        names = args.models or FREE_05B
    elif args.task == "easy":
        names = args.models or (EASY_PRIORITY + EASY_QWEN + EASY_Q2 + (EASY_FP16 if args.include_fp16 else []))
    else:
        names = args.models or (CH_05B + CH_PHI3)
    qs = load_questions("easy" if args.task == "fill" else args.task)
    print(f"题库 {args.task} {len(qs)} 题 | 档位 {len(names)} 个 | 协议：{PROTOCOL}")
    keep_awake(True)
    try:
        for name in names:
            if args.task == "fill":
                fill_fp16(name, qs)
            else:
                run_model(args.task, name, qs, args.limit)
        if args.task == "fill":
            audit_summary()
    finally:
        keep_awake(False)
    print("\n全部完成。下一步：python scripts/03_analysis/analyze_arc_unified.py")


if __name__ == "__main__":
    main()
