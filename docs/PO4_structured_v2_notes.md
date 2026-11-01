# PO-4 结构化生成任务（v2 修订说明）

## 一、v1 存在的问题（诊断结论）

| # | 问题 | 证据 |
|---|---|---|
| 1 | **6 个档位「0%」实为运行失败**，不是能力差 | `raw_output` 全空（80/80）；`f16` 与 `q4_k_m` 的 jsonl 字节数完全相同（18266） |
| 2 | 由此产生**荒谬趋势**：F16 0% → Q2_K 100%（7B） | summary 中 `qwen2.5-7b-instruct/q2_k=100%`，而 f16/q4_k_m=0% |
| 3 | **Markdown 表格指标失效** | 旧判定仅 `"|" in raw and 表头词 in raw` → parse=fields=schema 恒等（76.0% 三项相同） |
| 4 | **JSON 提取题有歧义** | 偶数题文本 "今天去{i+10}度，北京天气晴朗，气温{i+5}摄氏度" → 两个温度、语句不通 |
| 5 | 旧提取器 `rfind('}')` 可能跨对象截取 | 多个 JSON 时取到错误片段 |
| 6 | 模型/档位命名不统一 | `ll32` vs `llama3.1-8b-instruct`、`F16` vs `f16` |

## 二、v2 的三处改动

### 1. 题库（`build_structured_set_v2.py`）
- 每题新增 **`expected` 标准答案**（JSON 类：字段值；表格类：`expected_rows`）；
- 修正偶数题文本歧义（"北京今天晴朗，气温 25 摄氏度"）；
- 人名/职业/城市/天气 **轮换池**（15 组），避免"全是张伟/北京"的记忆型捷径；
- schema 全部加 `additionalProperties: false`，键名明确要求英文；
- 输出：`03_data/structured/struct_80_v2.jsonl`（80 题：30 提取 + 25 表格 + 25 函数）。

### 2. 运行（`run_structured_v2.py`）
- 启动失败 → 写一条 `status=server_failed` 记录，**不再伪造 80 行空数据**；
- 启动重试 2 次，每题请求重试 1 次；
- jsonl 内写入 `model / quant / status(ok|empty|error)`，评分不再依赖文件名猜测；
- 输出：`05_raw/structured_v2/<model>.gguf.jsonl`。

### 3. 评分（`eval_structured_v2.py`）
- **四级指标**：
  - `Parseable`：能被解析为 JSON / 表格结构
  - `Fields_Complete`：required 字段齐全（表格：表头正确）
  - `Schema_Valid`：jsonschema 通过（表格：行数=3）
  - **`Content_Correct`：值与 expected 完全一致（新增，区分度最高）**
- **区分运行失败**：`N_Empty` 与 `Note=run_incomplete`，不再把空输出算成 0 能力；
- JSON 提取改为**候选枚举 + 优先选 required 齐全者**（可正确处理"废话+假JSON+真JSON"）；
- 表格做**真实结构解析**（去分隔线、校验列数与数值）；
- 模型名/档位统一（`MODEL_MAP`：ll32→Llama-3.2-1B 等）。

## 三、复现步骤

```powershell
# 1) 生成题库（覆盖旧题库，不影响旧文件）
python 04_scripts\build_structured_set_v2.py

# 2) 跑全部档位（自动重试；失败会显式标记）
python 04_scripts\run_structured_v2.py

# 3) 评分
python 04_scripts\eval_structured_v2.py
```

输出：
- `06_results/table_struct_summary_v2.csv`（含 `N_Empty`、`Note`）
- `06_results/table_struct_detail_v2.csv`（逐题四级指标 + Status）

## 四、结果判读规则

1. **`Note=run_incomplete` 的档位不得参与横向比较**，需重跑；
2. 论文中报告 **`Content(%)`** 为主指标，`Parseable(%)` 作辅助；
3. 若某档位 `Parseable` 高但 `Content` 低，说明"格式能出、内容不对"，须在正文说明；
4. v1 数据（`table_struct_summary.csv` / `detail.csv`）**全部作废**，保留仅作追溯。
