from collections import defaultdict
import math

# 原始数据集
data = []
with open("C://Users//北弋//Desktop//Quatrain-main//data//bugreport_patch.txt", "r", encoding="utf-8") as file:
    for line in file:
        # 假设每行数据以"$$"分隔不同的字段
        entry = line.strip().split("$$")
        data.append(entry)

# 使用字典来按bugid存储数据集列表
bug_data_dict = defaultdict(list)

# 将数据按bugid存储到字典中
for entry in data:
    bug_id = entry[0]
    bug_data_dict[bug_id].append(entry)

# 读取组信息文件
groups_file = "C:\\Users\\北弋\\Desktop\\Quatrain-main\\groups.txt"  # 替换为实际文件路径
groups = []
with open(groups_file, "r", encoding="utf-8") as f:
    current_group = []
    for line in f:
        line = line.strip()
        if line.startswith("Group"):
            if current_group:
                groups.append(current_group)
            current_group = []
        elif line:
            current_group.append(line)
    if current_group:
        groups.append(current_group)

# 十次循环，每次选取不同的组作为测试集
for i in range(10):
    train_set = []
    test_set = []

    for j, group in enumerate(groups):
        if j == i:
            test_set.extend(group)
        else:
            train_set.extend(group)

    # 保存训练集和测试集至文件
    with open(f"{i+1}_train.txt", "w", encoding="utf-8") as train_file:
        for bug_id in train_set:
            for entry in bug_data_dict[bug_id]:
                train_file.write("$$".join(entry) + "\n")

    with open(f"{i+1}_test.txt", "w", encoding="utf-8") as test_file:
        for bug_id in test_set:
            for entry in bug_data_dict[bug_id]:
                test_file.write("$$".join(entry) + "\n")

    print(f"Iteration {i+1}: Train and test sets saved successfully.")
