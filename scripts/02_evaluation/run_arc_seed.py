# run_arc_seed.py (v5)
# v3 修正（模板/参数口径）；v4 修正（max_tokens、全角字母、记录 prompt）
# v5 修正（提取器增强、全档硬约束、max_tokens 16）
# v3 关键修正：与主实验 run_arc_full.py 口径完全对齐
#   1) 注入对应模型的聊天模板（Qwen/Phi3/Llama），v2 直接裸 prompt 导致准确率暴跌且不可比
#   2) max_tokens: FP16=64、量化档=8（与主实验一致）；-c: FP16=2048、量化档=4096
#   3) 提取支持 A–E，并优先精确匹配
#   4) 保留：断点续跑、原子写盘、坏行容错
import json, os, re, time, subprocess, requests, random
import warnings
warnings.filterwarnings('ignore')

LLAMA_SERVER = r"C:\Users\cjy13\Desktop\llama.cpp-master\llama.cpp-master\build\bin\Release\llama-server.exe"
MODEL_DIR    = r"C:\Users\cjy13\quant_eval\02_gguf\quantized"
DATA_FILE    = r"C:\Users\cjy13\quant_eval\03_data\arc\ARC-Easy-subset-500.jsonl"
OUT_DIR      = r"C:\Users\cjy13\quant_eval\05_raw\arc_seed"

TARGET_MODELS = [
    #"qwen1.5b-base.gguf",
    #"qwen1.5b-Q4_K_M.gguf",
    #"qwen1.5b-Q2_K.gguf",
    #"qwen3b-base.gguf",
    #"qwen3b-Q4_K_M.gguf",
    "qwen3b-Q2_K.gguf",
]
SEEDS = [42, 43, 44, 45, 46]


def is_fp16(name):
    n = name.lower()
    return ("f16" in n) or ("base" in n)


def kill_servers():
    os.system("taskkill /F /IM llama-server.exe > nul 2>&1")
    time.sleep(3)


def wait_ready(port, timeout=120):
    for _ in range(timeout):
        try:
            r = requests.get(f"http://127.0.0.1:{port}/health", timeout=2,
                             proxies={"http": None, "https": None})
            if r.status_code == 200:
                time.sleep(2)
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def start_server(model_path, port, retries=2):
    ctx = "2048" if is_fp16(os.path.basename(model_path)) else "4096"
    for attempt in range(1, retries + 1):
        kill_servers()
        cmd = [LLAMA_SERVER, "-m", model_path, "-c", ctx, "-t", "8", "--port", str(port)]
        proc = subprocess.Popen(cmd, creationflags=subprocess.CREATE_NO_WINDOW,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if wait_ready(port):
            print(f"     [就绪] 第 {attempt} 次启动成功 (ctx={ctx})")
            return proc
        print(f"     [失败] 第 {attempt} 次启动未就绪，清理重试")
        try:
            proc.terminate()
        except Exception:
            pass
    return None


def build_prompt(q, model_file):
    """与主实验 run_arc_full.py 完全一致的 prompt 构建"""
    q_text = q["question"] + "\n"
    labels = [chr(65 + i) for i in range(len(q["choices"]["text"]))]
    for j, opt in enumerate(q["choices"]["text"]):
        q_text += f"{labels[j]}. {opt}\n"

    # v5：所有档位统一加硬约束（原仅 FP16 加）。理由：量化档模型倾向长篇解释，
    # 32 token 内说不出字母，导致大量 empty，与主实验"直接答字母"的行为不一致。
    q_text += "只输出一个大写字母 (A/B/C/D)，不要解释，不要输出任何其他内容：\n"

    mn = model_file.lower()
    if "phi3" in mn:
        return f"<|user|>\n{q_text}答案: <|end|>\n<|assistant|>\n"
    if "ll32" in mn or "llama" in mn:
        return ("<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                f"{q_text}答案: <|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")
    if "qwen" in mn:
        return f"<|im_start|>user\n{q_text}答案：<|im_end|>\n<|im_start|>assistant\n"
    return q_text + "答案："


def ask(port, prompt, seed, max_tokens, retries=1):
    err = "UNKNOWN"
    for k in range(retries + 1):
        try:
            r = requests.post(f"http://127.0.0.1:{port}/v1/completions",
                              json={"prompt": prompt, "max_tokens": max_tokens,
                                    "temperature": 0, "top_p": 1, "seed": seed},
                              timeout=90, proxies={"http": None, "https": None})
            if r.status_code == 200:
                j = r.json()
                txt = ((j.get("choices") or [{}])[0].get("text") or "").strip()
                return txt, "ok"
            err = f"HTTP_{r.status_code}"
        except Exception as e:
            err = f"EXC_{type(e).__name__}"
        if k < retries:
            time.sleep(2)
    return "", err


_FW = str.maketrans({**{chr(0xFF21 + i): chr(65 + i) for i in range(26)},
                     **{chr(0xFF41 + i): chr(65 + i) for i in range(26)}})

def extract_letter(txt):
    """v5：全角归一 → 精确 → 答案后字母(允许"是/为") → 选项标记 "C." → 开头 → 兜底"""
    if not txt:
        return ""
    t = txt.strip().translate(_FW).upper()
    if t in ("A", "B", "C", "D", "E"):
        return t
    head = t[:160]
    m = re.search(r'(?:答案|ANSWER|OPTION)\s*[是为]?\s*[：:]?\s*\(?\[?\s*([A-E])\b', head)
    if m:
        return m.group(1)
    m = re.search(r'(?<![A-Z])([A-E])\s*[.、\)]', head)
    if m:
        return m.group(1)
    m = re.match(r'^[\s\(\[\"\'\*]*([A-E])\b', head)
    if m:
        return m.group(1)
    m = re.search(r'\b([A-E])\b', head[:120])
    return m.group(1) if m else ""


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.exists(DATA_FILE):
        print("[ERR] 题库不存在，请先运行 prepare_arc_subset.py")
        return

    questions = [json.loads(l) for l in open(DATA_FILE, encoding="utf-8") if l.strip()]
    total = len(questions) * len(SEEDS)
    print(f"题库 {len(questions)} 题 | 模型 {len(TARGET_MODELS)} 个 | Seed {SEEDS}\n")

    for mf in TARGET_MODELS:
        out_file = os.path.join(OUT_DIR, f"{mf}_seed.jsonl")
        recs = {}
        if os.path.exists(out_file):
            with open(out_file, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        r = json.loads(line)
                    except Exception:
                        continue
                    if "seed" in r and "id" in r:
                        recs[f"{r['seed']}_{r['id']}"] = r

        todo = [(sd, q) for sd in SEEDS for q in questions
                if recs.get(f"{sd}_{q['id']}", {}).get("status") != "ok"]

        print(f"=== {mf} | 已完成 {len([1 for r in recs.values() if r.get('status')=='ok'])}/{total}，待跑 {len(todo)}")
        if not todo:
            print("     全部完成，跳过\n")
            continue

        port = random.randint(8100, 9000)
        proc = start_server(os.path.join(MODEL_DIR, mf), port)
        if proc is None:
            print("     [ERR] 启动失败，进度保留，可续跑\n")
            continue

        mt = 64 if is_fp16(mf) else 16   # v5: 32→16，配合硬约束，够说 "C. 内容" 但不给长篇解释留空间
        n_ok = n_err = 0
        for k, (sd, q) in enumerate(todo, 1):
            prompt = build_prompt(q, mf)
            txt, st = ask(port, prompt, sd, mt)
            pred = extract_letter(txt)
            if st == "ok" and not pred:
                st = "empty"
            ok = 1 if (st == "ok" and pred == q["answerKey"]) else 0

            recs[f"{sd}_{q['id']}"] = {
                "model": mf, "seed": sd, "id": q["id"], "type": "arc",
                "status": st, "raw_output": txt, "clean_pred": pred,
                "answer": q["answerKey"], "is_correct": ok,
                "max_tokens": mt, "prompt_tail": prompt[-60:],
            }
            n_ok += 1 if st == "ok" else 0
            n_err += 1 if st != "ok" else 0

            if k % 20 == 0 or k == len(todo):
                tmp = out_file + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    for rv in recs.values():
                        f.write(json.dumps(rv, ensure_ascii=False) + "\n")
                os.replace(tmp, out_file)
                print(f"     [{k}/{len(todo)}] ok={n_ok} err={n_err}", end="\r")

        print(f"\n     [DONE] ok={n_ok} err={n_err}\n")
        try:
            proc.terminate()
        except Exception:
            pass
        kill_servers()


if __name__ == "__main__":
    main()
