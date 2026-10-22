# C:\Users\cjy13\quant_eval\04_scripts\check_kld_size.py
import os

kld_dir = r"C:\Users\cjy13\quant_eval\05_raw\kld"
print("=== .kld 文件大小 ===")
for f in sorted(os.listdir(kld_dir)):
    if f.endswith(".kld"):
        size = os.path.getsize(os.path.join(kld_dir, f))
        print(f"{f}: {size/1024:.1f} KB ({size/1024/1024:.2f} MB)")

print("\n=== .log 文件大小 ===")
for f in sorted(os.listdir(kld_dir)):
    if f.endswith(".log"):
        size = os.path.getsize(os.path.join(kld_dir, f))
        print(f"{f}: {size/1024:.1f} KB")