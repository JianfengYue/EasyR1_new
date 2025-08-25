from datasets import load_dataset

# 读取原始数据集
ds = load_dataset("Elliott/Openr1-Math-46k-8192", split="train")

# 把 reward_model 里的 ground_truth 抽出来当新列 "answer"
def extract_label(example):
    example["answer"] = example["reward_model"].get("ground_truth", None)
    return example

ds = ds.map(extract_label)

# 重命名需要的字段
ds = ds.rename_columns({
    "prompt": "problem",
    "target": "sft_label"
})

# 删除其他没用的字段
ds = ds.remove_columns(['data_source', 'ability', 'reward_model', 'extra_info'])

# 再次 map，提取 user/assistant 的 content
def simplify(example):
    # 从 problem 中提取 role=user
    if isinstance(example["problem"], list):
        user_msgs = [m["content"] for m in example["problem"] if m.get("role") == "user"]
        example["problem"] = user_msgs[0] if user_msgs else ""

    # 从 sft_label 中提取 role=assistant，并在末尾加 </think>
    if isinstance(example["sft_label"], list):
        assistant_msgs = [m["content"] for m in example["sft_label"] if m.get("role") == "assistant"]
        if assistant_msgs:
            content = assistant_msgs[0].rstrip()
            if not content.endswith("</think>"):
                content += "\n</think>"
            example["sft_label"] = content
        else:
            example["sft_label"] = ""

    return example

ds = ds.map(simplify)

print(ds)
print(ds[0])


ds.save_to_disk("/home/hpc/b273dd/b273dd11/data")

# # 下次直接加载
# from datasets import load_from_disk
# ds = load_from_disk("data/Openr1_edited")