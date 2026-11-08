# C:\Users\cjy13\quant_eval\04_scripts\fix_fig08.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

res_dir = r"C:\Users\cjy13\quant_eval\06_results"
fig_dir = os.path.join(res_dir, "figures")
os.makedirs(fig_dir, exist_ok=True)

df = pd.read_csv(os.path.join(res_dir, "tableA3_kv_cache.csv"), encoding="utf-8-sig")
print("KV Cache 表列名:", df.columns.tolist())
print(df.to_string(index=False))

# 期望列：KV_Cache_Type, pp4096 (t/s), tg128 (t/s)
kv_types = df["KV_Cache_Type"].tolist()
pp = df["pp4096 (t/s)"].tolist()
tg = df["tg128 (t/s)"].tolist()

x = np.arange(len(kv_types))
width = 0.35

fig, ax1 = plt.subplots(figsize=(7, 4))
ax1.bar(x - width/2, pp, width, label="pp4096", color="tab:blue")
ax1.set_ylabel("pp4096 (t/s)", color="tab:blue")
ax1.set_xticks(x)
ax1.set_xticklabels(kv_types)
ax1.tick_params(axis="y", labelcolor="tab:blue")

ax2 = ax1.twinx()
ax2.bar(x + width/2, tg, width, label="tg128", color="tab:red")
ax2.set_ylabel("tg128 (t/s)", color="tab:red")
ax2.tick_params(axis="y", labelcolor="tab:red")

plt.title("KV Cache Ablation (Context 4096)")
fig.tight_layout()
plt.savefig(os.path.join(fig_dir, "fig08_kv_cache.pdf"))
plt.close()
print("✅ fig08 已重新生成")