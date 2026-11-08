# -*- coding: utf-8 -*-
"""
clean_reasoning_jsonl.py
作用：清理 05_raw/reasoning 下 pred 为空的失败记录，使 run_reasoning_full.py 在续跑时能自动重跑这些题目。
说明：脚本会先为每个被修改的文件生成 .bak_日期 备份，原文件不丢。
用法：python clean_reasoning_jsonl.py --dir ../../05_raw/reasoning
"""
import argparse, json, os, shutil, datetime

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="reasoning jsonl 所在目录")
    ap.add_argument("--apply", action="store_true", help="实际写回；不加则该参数时仅统计")
    ap.add_argument("--match", default="", help="只处理文件名包含该子串的 jsonl，例如 phi3")
    a = ap.parse_args()
    stamp = datetime.date.today().strftime("%Y%m%d")
    total_drop = 0
    for fn in sorted(os.listdir(a.dir)):
        if not fn.endswith(".jsonl"):
            continue
        if a.match and a.match.lower() not in fn.lower():
            continue
        path = os.path.join(a.dir, fn)
        keep, drop = [], 0
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if str(rec.get("pred", "")).strip():
                    keep.append(rec)
                else:
                    drop += 1
        if drop == 0:
            continue
        total_drop += drop
        print("%-46s 保留 %3d / 丢弃空记录 %3d" % (fn, len(keep), drop))
        if a.apply:
            shutil.copy2(path, path + ".bak_" + stamp)
            with open(path, "w", encoding="utf-8") as f:
                for rec in keep:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print("\n合计待重跑题目：%d" % total_drop)
    print("（加 --apply 才会真正写回，并自动生成 .bak_%s 备份）" % stamp)

if __name__ == "__main__":
    main()
