# build_structured_set_v2.py
# 结构化生成题库 v2：每题附带 expected（标准答案），支持内容级校验
# 输出：03_data/structured/struct_80_v2.jsonl
import json, os, random

OUT_DIR = r"C:\Users\cjy13\quant_eval\03_data\structured"
OUT_PATH = os.path.join(OUT_DIR, "struct_80_v2.jsonl")
os.makedirs(OUT_DIR, exist_ok=True)

NAMES = ["张伟", "李娜", "王强", "刘洋", "陈静", "赵磊", "孙悦", "周涛",
         "吴敏", "郑凯", "冯雪", "何俊", "许雯", "曹阳", "邓琳"]
OCCS  = ["软件工程师", "中学教师", "外科医生", "平面设计师", "会计",
         "产品经理", "数据分析师", "律师", "护士", "机械工程师",
         "市场专员", "翻译", "建筑师", "记者", "药剂师"]
CITIES = ["北京", "上海", "广州", "深圳", "杭州", "成都", "武汉", "西安",
          "南京", "重庆", "苏州", "天津", "长沙", "青岛", "郑州"]
CONDS  = ["晴朗", "多云", "小雨", "阴天", "雷阵雨", "小雪", "雾", "大风",
          "晴转多云", "阵雨", "霾", "多云转晴", "中雨", "冻雨", "晴"]

data = []

# ============ 1. JSON 字段提取（30 题：15 人物 + 15 天气）============
for i in range(1, 16):
    name, occ = NAMES[i-1], OCCS[i-1]
    age = 20 + i
    data.append({
        "id": f"extract_{2*i-1:02d}",
        "type": "json_extraction",
        "prompt": (
            "请从以下文本中提取 name（姓名）、age（年龄）、occupation（职业），"
            "严格以 JSON 对象输出，键名使用英文，不要包含任何其他文字或 Markdown 标记。\n"
            f"文本：我叫{name}，今年{age}岁，是一名{occ}。"
        ),
        "schema": {
            "type": "object",
            "properties": {"name": {"type": "string"},
                           "age": {"type": "integer"},
                           "occupation": {"type": "string"}},
            "required": ["name", "age", "occupation"],
            "additionalProperties": False
        },
        "expected": {"name": name, "age": age, "occupation": occ}
    })

for i in range(1, 16):
    city, cond = CITIES[i-1], CONDS[i-1]
    temp = 18 + i
    data.append({
        "id": f"extract_{2*i:02d}",
        "type": "json_extraction",
        "prompt": (
            "请从以下文本中提取 city（城市）、temperature（气温，整数）、condition（天气状况），"
            "严格以 JSON 对象输出，键名使用英文，不要包含任何其他文字或 Markdown 标记。\n"
            f"文本：{city}今天{cond}，气温{temp}摄氏度。"
        ),
        "schema": {
            "type": "object",
            "properties": {"city": {"type": "string"},
                           "temperature": {"type": "integer"},
                           "condition": {"type": "string"}},
            "required": ["city", "temperature", "condition"],
            "additionalProperties": False
        },
        "expected": {"city": city, "temperature": temp, "condition": cond}
    })

# ============ 2. Markdown 表格填充（25 题，3 行 3 列）============
SUBJECTS = ["语文", "数学", "英语"]
for i in range(1, 26):
    rows = [["张三", "语文", 70 + i],
            ["李四", "数学", 80 + i],
            ["王五", "英语", 60 + i]]
    data.append({
        "id": f"table_{i:02d}",
        "type": "markdown_table",
        "prompt": (
            "请将以下数据转换为 Markdown 表格，表头必须为「姓名|科目|分数」，共 3 行数据，"
            "只输出表格本身，不要有多余解释。\n"
            f"数据：张三的语文考了{70+i}分，李四的数学考了{80+i}分，王五的英语考了{60+i}分。"
        ),
        "expected_headers": ["姓名", "科目", "分数"],
        "expected_rows": rows
    })

# ============ 3. 函数调用 JSON（25 题）============
for i in range(1, 26):
    if i % 2 == 1:
        city = CITIES[(i-1) % 15]
        data.append({
            "id": f"func_{i:02d}",
            "type": "function_call",
            "prompt": (
                "请根据用户需求调用正确的函数，并输出该函数的 JSON 参数对象（键名使用参数名），"
                "不要输出其他内容。\n可用函数：get_weather(city: str)\n"
                f"用户：帮我查一下{city}今天的天气。"
            ),
            "schema": {"type": "object",
                       "properties": {"city": {"type": "string"}},
                       "required": ["city"],
                       "additionalProperties": False},
            "expected": {"city": city}
        })
    else:
        n = i + 3
        data.append({
            "id": f"func_{i:02d}",
            "type": "function_call",
            "prompt": (
                "请根据用户需求调用正确的函数，并输出该函数的 JSON 参数对象（键名使用参数名），"
                "不要输出其他内容。\n可用函数：search_web(query: str, num_results: int)\n"
                f"用户：帮我搜一下最新的AI论文，要{n}篇。"
            ),
            "schema": {"type": "object",
                       "properties": {"query": {"type": "string"},
                                      "num_results": {"type": "integer"}},
                       "required": ["query", "num_results"],
                       "additionalProperties": False},
            "expected": {"query": "最新的AI论文", "num_results": n}
        })

random.seed(42)
random.shuffle(data)

with open(OUT_PATH, "w", encoding="utf-8") as f:
    for item in data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

by_type = {}
for d in data:
    by_type[d["type"]] = by_type.get(d["type"], 0) + 1
print(f"题库生成完毕：{len(data)} 题 {by_type}")
print("输出:", OUT_PATH)
