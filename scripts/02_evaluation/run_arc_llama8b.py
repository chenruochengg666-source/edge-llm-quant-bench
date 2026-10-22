# C:\Users\cjy13\quant_eval\04_scripts\run_arc_llama8b.py
import json, os, re, time, subprocess, requests, random
import sys

os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""
os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""
os.environ["NO_PROXY"] = "127.0.0.1,localhost"

# ================= 配置区 =================
LLAMA_SERVER = r"C:\Users\cjy13\Desktop\llama.cpp-master\llama.cpp-master\build\bin\Release\llama-server.exe"
MODEL_DIR = r"C:\Users\cjy13\quant_eval\02_gguf\quantized"
DATA_FILE = r"C:\Users\cjy13\quant_eval\03_data\arc\ARC-Easy-test-full.jsonl"
OUT_DIR = r"C:\Users\cjy13\quant_eval\05_raw\arc_llama8b_subset"
# ==========================================

os.makedirs(OUT_DIR, exist_ok=True)
models = ["llama3.1-8b-instruct-q4_k_m.gguf", "llama3.1-8b-instruct-q2_k.gguf"]

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
    # 修复 1：正确带上 chat-template 参数
    cmd = [LLAMA_SERVER, "-m", model_path, "-c", "4096", "-t", "8", "--port", str(port), "--chat-template", "llama3"]
    print(f"  [启动服务] 使用端口 {port}，正在加载模型...")

    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            creationflags=subprocess.CREATE_NO_WINDOW)

    # 等待健康检查
    for _ in range(90):
        try:
            requests.get(f"http://127.0.0.1:{port}/health", timeout=2, proxies={"http": None, "https": None})
            break
        except:
            time.sleep(1)
    else:
        raise RuntimeError(f"服务启动超时: {model_path}")

    # 修复 2：真实推理预热（防假死）
    print("  [服务就绪] 正在执行真实推理预热...")
    test_prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nWhat is 1+1? Answer with a single letter A.\nA. 2\nB. 3<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    for attempt in range(3):
        try:
            r = requests.post(
                f"http://127.0.0.1:{port}/v1/completions",
                json={"prompt": test_prompt, "max_tokens": 16, "temperature": 0},
                timeout=30,
                proxies={"http": None, "https": None}
            )
            ans = r.json()['choices'][0]['text'].strip()
            print(f"  [预热成功] 模型回复: {ans}")
            time.sleep(2)
            return proc, port
        except Exception as e:
            print(f"    [预热失败，重试 {attempt + 1}/3] {e}")
            time.sleep(3)

    proc.terminate()
    raise RuntimeError("模型推理预热彻底失败，请检查模型文件。")


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

        q_text = q["question"] + "\n"
        labels = [chr(65 + i) for i in range(len(q["choices"]["text"]))]
        for j, opt in enumerate(q["choices"]["text"]):
            q_text += f"{labels[j]}. {opt}\n"

        q_text += "\n只输出一个大写字母 (A/B/C/D)，不要解释，不要输出任何其他内容。\n"
        prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{q_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

        ans = ""
        request_success = False

        for attempt in range(3):
            try:
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
                time.sleep(2)

        if not request_success:
            print(f"\n❌ 致命错误：在题目 {q['id']} 连续 3 次请求失败。脚本停止。")
            proc.terminate()
            sys.exit(1)

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

print("🎉 Llama-3.1-8B ARC 评测完成！")