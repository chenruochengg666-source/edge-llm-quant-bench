README：基于纯 CPU 边缘设备的大模型量化评估
1. 项目简介
本项目在纯 CPU 边缘设备（Intel Core Ultra 5 225H / 16 GB DDR5 / 无独显）上，基于 llama.cpp K-quant 方案，对 Qwen2.5 系列（0.5B / 1.5B / 3B）以及跨架构模型 Phi-3-mini-4k-instruct、Llama-3.2-1B-Instruct 进行从 FP16 到 Q2_K 的系统量化评估。评估采用双轨框架：

能力质量轨：PPL、KLD、ARC-Easy（2376 全量）、自建多步逻辑推理（120 题，4 难度层）、Correctness Agreement。

性能工程轨：模型体积、峰值 RSS、预填充/解码速度、并发吞吐、KV Cache 消融、上下文伸缩、电源模式对照。

项目核心目标是检验 PPL/KLD 这类保真度指标能否代表下游能力，并给出 Q4_K_M / Q2_K 等档位在 CPU 边缘设备上的适用边界。

2. 硬件与软件环境
项	配置
设备	ASUS Vivobook 14
CPU	Intel Core Ultra 5 225H（14 线程，异构）
指令集	AVX2、AVX-VNNI（不支持 AVX-512）
内存	16 GB DDR5
显卡	无独显（纯 CPU 推理）
操作系统	Windows 11
推理框架	llama.cpp（commit 4da633776，build 11223，AVX2 编译版）
Python	3.10
主要依赖	pandas, numpy, scikit-learn, matplotlib, seaborn, scipy
3. 目录结构
text
C:\Users\cjy13\quant_eval\
├── 02_gguf\
│   └── quantized\                  # 各量化档位 GGUF 文件
├── 03_data\
│   ├── arc\                        # ARC-Easy 测试集
│   └── reasoning\                  # 多步推理题库
├── 04_scripts\                     # 全部实验脚本
├── 05_raw\
│   ├── arc\                        # ARC 原始推理结果
│   ├── reasoning\                  # 多步推理原始输出
│   ├── kld\                        # PPL/KLD 日志与 .kld 文件
│   ├── perf\                       # llama-bench 结果
│   ├── rss\                        # 峰值 RSS
│   ├── ctx\                        # 长上下文结果
│   ├── concurrency\                # 并发吞吐
│   └── manual_scoring\             # 双人评分 CSV
└── 06_results\
    ├── figures\                    # 论文图表（PDF）
    ├── table2_engineering.csv
    ├── table_kld_summary.csv       # PPL/KLD 汇总
    ├── table4_downstream_arc.csv
    ├── table5_reasoning_final_summary.csv
    ├── table7_ca_kappa_final.csv
    ├── table_stat_tests.csv        # 统计检验
    └── ...
4. 模型与量化档位
模型	档位
Qwen2.5-0.5B-Instruct	FP16, Q4_K_M, Q2_K
Qwen2.5-1.5B-Instruct	FP16, Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q3_K_M, Q2_K
Qwen2.5-3B-Instruct	FP16, Q4_K_M, Q3_K_M, Q2_K
Phi-3-mini-4k-instruct	FP16, Q4_K_M, Q2_K
Llama-3.2-1B-Instruct	FP16, Q4_K_M, Q2_K
量化命令示例：

bash
llama-quantize.exe qwen1.5b-f16.gguf qwen1.5b-Q4_K_M.gguf Q4_K_M
5. 实验流程
5.1 量化模型生成
由 HuggingFace 原始权重经 convert_hf_to_gguf.py 转 FP16 GGUF，再用 llama-quantize 生成各档位。未使用 imatrix。

5.2 PPL / KLD 计算
bash
llama-perplexity.exe -m <model>.gguf -f wiki.test.raw -c 512 -t 8 --chunks 20 \
  --kl-divergence-base <f16-base>.kld --kl-divergence
使用 chunks = 20，报告均值、SD、95% CI。

F16 为基准，量化档位与 F16 比较。

结果汇总于 table_kld_summary.csv 与 table3_fidelity.csv。

5.3 ARC-Easy 评测
数据集：allenai/ai2_arc，ARC-Easy test split，2376 题全量。

脚本：run_arc_full.py。

通过本地 llama-server 的 OpenAI 兼容接口逐题提问。

输出：table4_downstream_arc.csv，含 Accuracy 与 95% CI。

5.4 多步推理评测
题库：reasoning100.jsonl，120 题，分 2step / 3step / 4step / 5step+ 四层。

脚本：run_reasoning_full.py。

评分：1.0（完全正确）/ 0.5（排序正确格式错）/ 0.0（错误）。

双人独立评分（rater1.csv / rater2.csv），Cohen’s Kappa = 1.0。

汇总：table5_reasoning_final_summary.csv。

5.5 工程性能测试
指标	脚本
pp512 / tg128	extract_performance.py（llama-bench）
峰值 RSS	measure_rss.ps1、measure_ctx.ps1
并发吞吐	concurrency_test.py
KV Cache 消融	extract_kv_power.py
电源模式	extract_kv_power.py
长上下文	measure_ctx.ps1
结果汇总于 table2_engineering.csv、tableA2_power.csv、tableA3_kv_cache.csv。

5.6 统计检验
脚本：stat_tests.py

方法：

ARC：两比例 z 检验 + FDR（Benjamini-Hochberg）校正。

KLD：Welch t 检验 + FDR 校正。

输出：table_stat_tests.csv。

6. 人工评分与 AI 评分
双人评分文件：rater1_scored.csv、rater2_scored.csv

合并脚本：merge_manual_final.py

AI 评分脚本：ai_grader_final.py

最终分：人工平均分，Kappa = 1.0

明细：table5_reasoning_detail_human_final.csv

未评分档位：Qwen1.5B 的 Q5_K_M、Q6_K、Q8_0，Qwen3B 的 Q3_K_M 未纳入人工多步推理评分。

7. 图表生成
主脚本：plot_all_final_v3.py

图	内容
fig01	量化档位 vs pp512/tg128
fig02	KLD vs ARC-Easy（双轴）
fig03	1.5B vs 3B PPL/KLD
fig04	Qwen1.5B 关键指标热力图
fig05	跨架构 ARC
fig06	长上下文峰值 RSS
fig07	电源模式
fig08	KV Cache 消融
fig09	并发吞吐
fig10	体积 vs 解码速度
fig11	Correctness Agreement 热力图
fig12	多步推理得分矩阵
fig13	多步推理总分条形图
fig14	KLD / PPL / Same top p 三联图
运行：

powershell
python "C:\Users\cjy13\quant_eval\04_scripts\plot_all_final_v3.py"
8. 主要结论
Q2_K 在 1.5B/3B 上灾难性崩塌：KLD 显著升高，ARC 大幅下降，多步推理几乎全 0。

Q4_K_M 是较优平衡点：在 Qwen1.5B 上与 FP16 无统计显著差异（ARC q = 0.81）。

跨架构差异明显：Phi-3-mini 对量化较鲁棒，Llama-3.2-1B 较脆弱（Q4_K_M 就显著退化）。

保真度指标局限：KLD 与下游能力存在脱节，跨词表 KLD 不可比。

多步推理地板效应：1.5B 在 4 步及以上触及能力天花板，无法区分近基线档位。

9. 复现步骤
安装依赖：

powershell
pip install pandas numpy scikit-learn matplotlib seaborn scipy
生成量化模型（按第 5.1 节）。

运行 PPL/KLD：

powershell
llama-perplexity.exe -m <model>.gguf -f wiki.test.raw -c 512 -t 8 --chunks 20 --kl-divergence-base <base>.kld --kl-divergence
ARC 评测：

powershell
python "C:\Users\cjy13\quant_eval\04_scripts\run_arc_full.py"
多步推理评测：

powershell
python "C:\Users\cjy13\quant_eval\04_scripts\run_reasoning_full.py"
评分与合并：

powershell
python "C:\Users\cjy13\quant_eval\04_scripts\merge_manual_final.py"
统计检验：

powershell
python "C:\Users\cjy13\quant_eval\04_scripts\stat_tests.py"
生成图表：

powershell
python "C:\Users\cjy13\quant_eval\04_scripts\plot_all_final_v3.py"
10. 注意事项与局限性
跨词表 KLD 不可比：Phi-3-mini（32064）、Llama-3.2-1B（128256）与 Qwen2.5（151936）词表不同，KLD 数值只能同模型内比较。

TVD/JSD 未做，论文中以 KLD / PPL / Same top p 三联图替代。

多步推理未评分档位已在论文中注明，不参与推理对比。

长上下文仅测到 8K，16K 及以上因内存不足 OOM。

Qwen3B Q2_K 的 PPL 已崩溃，KLD 为 9.06，属灾难性退化。

11. 引用
Qwen2.5 Technical Report, arXiv:2412.15115

Phi-3 Technical Report, arXiv:2404.14219

Llama 3 Herd of Models, arXiv:2407.21783

llama.cpp, https://github.com/ggerganov/llama.cpp

ARC, arXiv:1803.05457

最后更新：2026-10-07
维护者：cjy13
状态：实验已完成，论文写作中。