# C:\Users\cjy13\quant_eval\04_scripts\run_reasoning_full.py
import json, os, re, time, subprocess, requests, random

# ================= 配置区 =================
LLAMA_SERVER = r"C:\Users\cjy13\Desktop\llama.cpp-master\llama.cpp-master\build\bin\Release\llama-server.exe"
MODEL_DIR = r"C:\Users\cjy13\quant_eval\02_gguf\quantized"
DATA_FILE = r"C:\Users\cjy13\quant_eval\03_data\reasoning\reasoning100.jsonl"
OUT_DIR = r"C:\Users\cjy13\quant_eval\05_raw\reasoning"
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
    cmd = [LLAMA_SERVER, "-m", model_path, "-c", "4096", "-t", "8", "--port", str(port)]
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
    if os.path.exists(out_file):
        with open(out_file, 'r', encoding='utf-8') as f:
            for line in f:
                done_ids.add(json.loads(line)["id"])
        print(f"  [续跑] {model_file} 已完成 {len(done_ids)} 题。")

    if len(done_ids) >= len(questions):
        print(f"✅ {model_file} 已全部完成，跳过。")
        continue

    try:
        proc, port = start_server(model_path)
    except Exception as e:
        print(f"❌ {model_file} 服务启动失败: {e}")
        continue

    for q in questions:
        if q["id"] in done_ids:
            continue

        # 构建推理题目 Prompt
        q_text = f"{q['question']}\n请逐步推理，最后一行必须以'答案：'开头给出最终结果。"

        # 注入聊天模板（根据模型类型自动适配）
        model_name = model_file.lower()
        if "phi3" in model_name:
            prompt = f"<|user|>\n{q_text}<|end|>\n<|assistant|>\n"
        elif "ll32" in model_name or "llama" in model_name:
            # Llama 强制用英文，防止彻底崩坏
            prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{q_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        elif "qwen" in model_name:
            prompt = f"<|im_start|>user\n{q_text}<|im_end|>\n<|im_start|>assistant\n"
        else:
            prompt = q_text

        ans = ""
        try:
            r = requests.post(
                f"http://127.0.0.1:{port}/v1/completions",
                json={"prompt": prompt, "max_tokens": 512, "temperature": 0, "top_p": 1, "seed": 42},
                timeout=120
            )
            resp_json = r.json()
            if "choices" in resp_json and len(resp_json["choices"]) > 0:
                ans = resp_json["choices"][0]["text"].strip()
        except Exception as e:
            ans = ""

        # 提取最后一行“答案：”后的内容
        pred = ""
        for line in ans.splitlines()[::-1]:
            if "答案：" in line:
                pred = line.split("答案：")[-1].strip()
                break
        if not pred and ans:
            pred = ans.strip().split()[-1]

        record = {"id": q["id"], "level": q["level"], "question": q["question"],
                  "gold": q["answer"], "pred": pred, "raw": ans}

        with open(out_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        done_ids.add(q["id"])
        print(f"  {model_file} {q['id']} 完成", end="\r")

    print(f"\n✅ {model_file} 推理完成！")
    proc.terminate()
    kill_all_servers()
    time.sleep(3)

print("\n所有模型推理评测完成！")