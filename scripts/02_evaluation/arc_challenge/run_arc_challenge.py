# -*- coding: utf-8 -*-
"""
ARC-Challenge 全档位评测（与 run_arc_full.py 逻辑一致，仅换题库）。

用途：回应审稿意见第 3 条——检验 ARC-Easy 作为主判据的代表性。
     ARC-Challenge 过滤了检索/词共现可解的题目，保留需要推理的题，
     若在 Challenge 上"沉默区"变为"同步退化区"，说明 ARC-Easy 低估了量化损害。

与 run_arc_full.py 的唯一差别：
    DATA_FILE → ARC-Challenge-test-full.jsonl
    OUT_DIR   → 05_raw/arc_challenge（不覆盖原数据）

prompt 构造、采样参数、答案提取逻辑均保持一致，以确保两套题库可比。

用法：
    python run_arc_challenge.py
"""
import json, os, re, time, subprocess, requests, random

# ================= 避免 Clash / 系统代理干扰 =================
# Clash 的 TUN 模式或系统代理会劫持 127.0.0.1 的请求，导致本地 llama-server
# 请求被转发到代理而超时/失败。这里在进程内清除所有代理环境变量，
# 并显式声明 localhost 直连。
for _k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
           "http_proxy", "https_proxy", "all_proxy"):
    os.environ.pop(_k, None)
os.environ["NO_PROXY"] = "127.0.0.1,localhost,0.0.0.0"
os.environ["no_proxy"] = "127.0.0.1,localhost,0.0.0.0"
# 所有 requests 调用统一使用此参数，双重保险
NO_PROXY_ARG = {"http": None, "https": None}
# ============================================================

# ================= 配置区 =================
LLAMA_SERVER = r"C:\Users\cjy13\Desktop\llama.cpp-master\llama.cpp-master\build\bin\Release\llama-server.exe"
MODEL_DIR = r"C:\Users\cjy13\quant_eval\02_gguf\quantized"
DATA_FILE = r"C:\Users\cjy13\quant_eval\03_data\arc\ARC-Challenge-test-full.jsonl"
OUT_DIR = r"C:\Users\cjy13\quant_eval\05_raw\arc_challenge"
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
    ctx_size = "2048" if ("f16" in model_path or "base" in model_path) else "4096"   # FP16 体积大，用 2048 省内存；量化用 4096
    cmd = [LLAMA_SERVER, "-m", model_path, "-c", ctx_size, "-t", "8", "--port", str(port)]
    print(f"  [启动服务] 使用端口 {port}...")
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            creationflags=subprocess.CREATE_NO_WINDOW)
    for _ in range(60):
        try:
            requests.get(f"http://127.0.0.1:{port}/health", timeout=2, proxies=NO_PROXY_ARG)
            return proc, port
        except Exception:
            time.sleep(1)
    raise RuntimeError(f"服务启动超时: {model_path}")


def warmup(port, max_wait=300):
    """模型预热：反复探测直到模型真正就绪。

    关键：llama-server 的 /health 返回 200 只代表进程已起，
    模型权重仍在加载时 /v1/completions 会返回 503。
    因此必须循环探测，直到真正的 200 响应。
    """
    print("  [预热] 开始探测模型就绪状态 ...")
    t0 = time.time()
    attempt = 0
    while time.time() - t0 < max_wait:
        attempt += 1
        try:
            r = requests.post(
                f"http://127.0.0.1:{port}/v1/completions",
                json={"prompt": "Hello", "max_tokens": 1, "temperature": 0},
                timeout=30,
                proxies=NO_PROXY_ARG,
            )
            if r.status_code == 200:
                print(f"  [预热] 就绪，耗时 {time.time() - t0:.1f} 秒（第 {attempt} 次探测）")
                return True
            if r.status_code == 503:
                print(f"  [预热] 模型加载中（503），第 {attempt} 次探测，等待 5 秒 ...")
                time.sleep(5)
                continue
            print(f"  [预热] 异常返回码 {r.status_code}，第 {attempt} 次探测，等待 5 秒 ...")
            time.sleep(5)
        except Exception as e:
            print(f"  [预热] 探测异常（第 {attempt} 次）：{e}，等待 5 秒 ...")
            time.sleep(5)
    print(f"  [预热] 等待 {max_wait} 秒仍未就绪")
    return False


def extract_letter(ans: str) -> str:
    """从模型输出中提取选项字母。

    设计原则：宁可返回 "?"（记为错误），也不要误判。
    原实现使用 re.search(r'\b[A-E]\b', ...) 全文搜索，
    会把英文冠词 "a"/"an" 误判为选项 A，导致预测分布严重偏向 A。

    三级优先级：
      1) 整个输出就是一个字母
      2) 开头（允许前导空白/标点）就是字母
      3) 明确的答案模式（"答案是X" / "answer is X" / "(X)"）
    """
    if not ans:
        return "?"

    a = ans.strip().upper()

    # 1) 整个输出就是单个字母
    if a in ("A", "B", "C", "D", "E"):
        return a

    # 2) 开头就是字母，且后面不是紧接其他字母（排除 "ADD"、"BE" 这类词）
    m = re.match(r"^[\s\.\:\,\u3001\uff0c\uff1a\-\*\uff08\(]*([A-E])(?![A-Z])", a)
    if m:
        return m.group(1)

    # 3) 明确的答案表述
    patterns = [
        r"(?:ANSWER\s*(?:IS|:))\s*\(?([A-E])\)?",
        r"(?:答案\s*(?:是|为)?\s*[：:])\s*\(?([A-E])\)?",          # "答案是：X"
        r"(?:答案\s*(?:是|为)\s*)\(?([A-E])\)?(?![\u4e00-\u9fff])",  # "答案是 X"（无冒号）
        r"^\(([A-E])\)",
        r"^([A-E])[\.\:、，,]",
        r"(?:选项|OPTION|CHOICE)\s*([A-E])\s*(?:正确|IS|CORRECT)",
    ]
    for pat in patterns:
        m = re.search(pat, a)
        if m:
            return m.group(1)

    # 4) 兜底：仅在极短输出（<=3 字符）中找独立字母，避免长文本误判
    if len(a) <= 3:
        m = re.search(r"\b([A-E])\b", a)
        if m:
            return m.group(1)

    return "?"


for model_file in models:
    model_path = os.path.join(MODEL_DIR, model_file)
    out_file = os.path.join(OUT_DIR, f"{model_file}.jsonl")

    done_ids = set()
    results = []
    if os.path.exists(out_file):
        with open(out_file, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                results.append(item)
                # 修复：仅当 pred 非空时才视为已完成，避免失败记录被永久固化
                if str(item.get("pred", "")).strip():
                    done_ids.add(item["id"])
        print(f"  [续跑] {model_file} 已完成 {len(done_ids)} 题。")

    if len(done_ids) == len(questions):
        print(f"✅ {model_file} 已全部完成，跳过。")
        continue

    try:
        proc, port = start_server(model_path)
    except Exception as e:
        print(f"❌ {model_file} 服务启动失败: {e}")
        continue

    # 模型预热：必须真正就绪才继续，否则整批结果作废
    if not warmup(port):
        print(f"⚠️ {model_file} 预热超时未就绪，跳过该模型")
        proc.terminate()
        kill_all_servers()
        time.sleep(3)
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

        # 所有模型统一强制格式：只输出单个字母。
        # 原脚本仅对 f16/base 加约束，量化模型会输出长篇解释并被截断，
        # 导致答案提取失败或被正则误匹配到冠词 "a"。
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
            max_tokens = 64   # 统一 64：量化模型也可能先输出换行/引导语，8 个 token 不够提取字母
            r = requests.post(
                f"http://127.0.0.1:{port}/v1/completions",
                json={"prompt": prompt, "max_tokens": max_tokens, "temperature": 0, "top_p": 1, "seed": 42},
                timeout=60,
                proxies=NO_PROXY_ARG  # 防代理干扰
            )
            resp_json = r.json()
            if "choices" in resp_json and len(resp_json["choices"]) > 0:
                ans = resp_json["choices"][0]["text"].strip()
        except Exception as e:
            ans = ""  # 崩溃时置空，不写入 ERROR 字符

        pred = extract_letter(ans)

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
            qmark = sum(1 for r in results if str(r.get("pred", "")) == "?")
            print(f"    进度: {idx}/{len(questions)} | 当前准确率: {correct_count / len(done_ids):.4f}"
                  f" | 未提取(pred=?): {qmark} ({qmark / len(done_ids) * 100:.1f}%)")

    print(f"✅ {model_file} 完成，最终准确率: {correct_count}/{len(questions)} = {correct_count / len(questions):.4f}")
    proc.terminate()
    kill_all_servers()
    time.sleep(3)

print("所有模型 ARC-Challenge 评测完成！输出目录：05_raw/arc_challenge")