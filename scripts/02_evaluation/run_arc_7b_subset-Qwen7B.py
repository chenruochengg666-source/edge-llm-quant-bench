# C:\Users\cjy13\quant_eval\04_scripts\run_arc_7b_subset - Qwen7B.py
import json, os, re, time, subprocess, requests, random
import sys

# 强制禁用代理
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""
os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""
os.environ["NO_PROXY"] = "127.0.0.1,localhost"

# ================= 配置区 =================
LLAMA_SERVER = r"C:\Users\cjy13\Desktop\llama.cpp-master\llama.cpp-master\build\bin\Release\llama-server.exe"
MODEL_DIR = r"C:\Users\cjy13\quant_eval\02_gguf\quantized"
DATA_FILE = r"C:\Users\cjy13\quant_eval\03_data\arc\ARC-Easy-test-full.jsonl"
OUT_DIR = r"C:\Users\cjy13\quant_eval\05_raw\arc_7b_subset"
# ==========================================

os.makedirs(OUT_DIR, exist_ok=True)
models = ["qwen2.5-7b-instruct-q4_k_m.gguf", "qwen2.5-7b-instruct-q2_k.gguf"]

random.seed(42)
with open(DATA_FILE, 'r', encoding='utf-8') as f:
    all_questions = [json.loads(line) for line in f]
questions = random.sample(all_questions, 500)
print(f"✅ 成功加载并随机抽取了 {len(questions)} 道 ARC 题目。\n")


def kill_all_servers():
    os.system('taskkill /F /IM llama-server.exe > nul 2>&1')
    time.sleep(5)


def start_server(model_path):
    kill_all_servers()
    port = random.randint(8100, 9000)
    cmd = [LLAMA_SERVER, "-m", model_path, "-c", "4096", "-t", "8", "--port", str(port)]
    print(f"  [启动服务] 使用端口 {port}，正在加载模型...")
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            creationflags=subprocess.CREATE_NO_WINDOW)
    for _ in range(90):
        try:
            requests.get(f"http://127.0.0.1:{port}/health", timeout=2, proxies={"http": None, "https": None})
            print("  [服务就绪] 预热 3 秒...")
            time.sleep(3)
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
        print(f"  ⚠️ 检测到断点文件，已完成 {len(done_ids)} 题。")

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

        # ============ 1. 构建选项文本 ============
        q_text = q["question"] + "\n"
        labels = [chr(65 + i) for i in range(len(q["choices"]["text"]))]
        for j, opt in enumerate(q["choices"]["text"]):
            q_text += f"{labels[j]}. {opt}\n"

        # 🚨【关键修复 1】强制模型闭嘴，只给字母！
        q_text += "\n只输出一个大写字母 (A/B/C/D)，不要解释，不要输出任何其他内容。\n"

        # ============ 2. 注入 Qwen 聊天模板 ============
        prompt = f"<|im_start|>user\n{q_text}答案：<|im_end|>\n<|im_start|>assistant\n"

        # ============ 3. 发起请求（带自动重试） ============
        ans = ""
        max_retries = 3
        request_success = False

        for attempt in range(max_retries):
            try:
                # 🚨【关键修复 2】不管什么模型，统一给 16 个 token，足够输出字母加标点
                r = requests.post(
                    f"http://127.0.0.1:{port}/v1/completions",
                    json={"prompt": prompt, "max_tokens": 16, "temperature": 0, "top_p": 1, "seed": 42},
                    timeout=60,
                    proxies={"http": None, "https": None}
                )
                resp_json = r.json()
                if "choices" in resp_json and len(resp_json["choices"]) > 0:
                    ans = resp_json["choices"][0]["text"].strip()
                request_success = True
                break
            except Exception as e:
                print(f"    [重试 {attempt + 1}/{max_retries}] 请求失败: {e}")
                time.sleep(2)

        if not request_success:
            print(f"\n❌ 致命错误：在题目 {q['id']} 连续 3 次请求失败。脚本停止。")
            proc.terminate()
            sys.exit(1)

        # ============ 4. 解析答案 ============
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

        if idx % 50 == 0:
            print(f"    进度: {idx}/{len(questions)} | 当前准确率: {correct_count / len(done_ids):.4f}")

    print(f"✅ {model_file} 完成，最终准确率: {correct_count}/{len(questions)} = {correct_count / len(questions):.4f}\n")
    proc.terminate()
    kill_all_servers()
    time.sleep(3)

print("🎉 所有模型 ARC 评测完成！")