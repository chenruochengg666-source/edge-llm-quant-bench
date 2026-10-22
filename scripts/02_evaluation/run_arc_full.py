# C:\Users\cjy13\quant_eval\04_scripts\run_arc_full.py
import json, os, re, time, subprocess, requests, random

# ================= 配置区 =================
LLAMA_SERVER = r"C:\Users\cjy13\Desktop\llama.cpp-master\llama.cpp-master\build\bin\Release\llama-server.exe"
MODEL_DIR = r"C:\Users\cjy13\quant_eval\02_gguf\quantized"
DATA_FILE = r"C:\Users\cjy13\quant_eval\03_data\arc\ARC-Easy-test-full.jsonl"
OUT_DIR = r"C:\Users\cjy13\quant_eval\05_raw\arc"
# ==========================================

os.makedirs(OUT_DIR, exist_ok=True)
models = [f for f in os.listdir(MODEL_DIR) if f.endswith(".gguf")]
with open(DATA_FILE, 'r', encoding='utf-8') as f:
    questions = [json.loads(line) for line in f]


def kill_all_servers():
    os.system('taskkill /F /IM llama-server.exe > nul 2>&1')
    time.sleep(5)


def start_server(model_path):
    kill_all_servers()
    port = random.randint(8100, 9000)
    ctx_size = "2048" if "f16" in model_path else "4096"
    cmd = [LLAMA_SERVER, "-m", model_path, "-c", ctx_size, "-t", "8", "--port", str(port)]
    print(f"  [启动服务] 使用端口 {port}...")
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            creationflags=subprocess.CREATE_NO_WINDOW)
    for _ in range(60):
        try:
            requests.get(f"http://127.0.0.1:{port}/health", timeout=2)
            return proc, port
        except:
            time.sleep(1)
    raise RuntimeError(f"服务启动超时: {model_path}")


for model_file in models:
    model_path = os.path.join(MODEL_DIR, model_file)
    out_file = os.path.join(OUT_DIR, f"{model_file}.jsonl")

    done_ids = set()
    results = []
    if os.path.exists(out_file):
        with open(out_file, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                done_ids.add(item["id"])
                results.append(item)
        print(f"  [续跑] {model_file} 已完成 {len(done_ids)} 题。")

    if len(done_ids) == len(questions):
        print(f"✅ {model_file} 已全部完成，跳过。")
        continue

    try:
        proc, port = start_server(model_path)
    except Exception as e:
        print(f"❌ {model_file} 服务启动失败: {e}")
        continue

    correct_count = sum(1 for r in results if r["ok"])
    for idx, q in enumerate(questions, 1):
        if q["id"] in done_ids:
            continue

        # ============ 1. 动态构建选项文本 ============
        q_text = q["question"] + "\n"
        labels = [chr(65 + i) for i in range(len(q["choices"]["text"]))]
        for j, opt in enumerate(q["choices"]["text"]):
            q_text += f"{labels[j]}. {opt}\n"

        # 全精度模型容易过度思考，强制要求只输出单个字母
        if "f16" in model_file:
            q_text += "只输出一个大写字母 (A/B/C/D)，不要解释，不要输出任何其他内容：\n"

        # ============ 2. 注入聊天模板（关键修复）===========
        model_name = model_file.lower()
        if "phi3" in model_name:
            prompt = f"<|user|>\n{q_text}答案: <|end|>\n<|assistant|>\n"
        elif "ll32" in model_name or "llama" in model_name:
            prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{q_text}答案: <|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        elif "qwen" in model_name:
            prompt = f"<|im_start|>user\n{q_text}答案：<|im_end|>\n<|im_start|>assistant\n"
        else:
            prompt = q_text + "答案："

        # ============ 3. 发起请求并提取答案（关键修复）===========
        ans = ""
        try:
            # 针对 f16 模型放宽 token 限制，防止答案被截断
            max_tokens = 64 if "f16" in model_file else 8
            r = requests.post(
                f"http://127.0.0.1:{port}/v1/completions",
                json={"prompt": prompt, "max_tokens": max_tokens, "temperature": 0, "top_p": 1, "seed": 42},
                timeout=60
            )
            resp_json = r.json()
            if "choices" in resp_json and len(resp_json["choices"]) > 0:
                ans = resp_json["choices"][0]["text"].strip()
        except Exception as e:
            ans = ""  # 崩溃时置空，不写入 ERROR 字符

        ans_upper = ans.strip().upper()
        pred = "?"
        if ans_upper in ["A", "B", "C", "D", "E"]:
            pred = ans_upper
        else:
            match = re.search(r'\b[A-E]\b', ans_upper)
            if match:
                pred = match.group(0)
            else:
                match = re.match(r'^([A-E])[\s\.,\:，。：]?', ans_upper)
                if match:
                    pred = match.group(1)

        gold = q["answerKey"]
        if isinstance(gold, int): gold = labels[gold]
        gold = gold.strip().upper()
        ok = (pred == gold)

        if ok: correct_count += 1
        record = {"id": q["id"], "pred": pred, "gold": gold, "ok": ok, "raw": ans}
        results.append(record)
        done_ids.add(q["id"])

        with open(out_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        if idx % 100 == 0:
            print(f"    进度: {idx}/{len(questions)} | 当前准确率: {correct_count / len(done_ids):.4f}")

    print(f"✅ {model_file} 完成，最终准确率: {correct_count}/{len(questions)} = {correct_count / len(questions):.4f}")
    proc.terminate()
    kill_all_servers()
    time.sleep(3)

print("所有模型 ARC 评测完成！")