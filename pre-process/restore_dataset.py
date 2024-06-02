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

# 将字典按键（bugid）排序并将其分成10组
sorted_bug_ids = sorted(bug_data_dict.keys())
bug_ids_groups = [sorted_bug_ids[i::10] for i in range(10)]

# 划分数据集为8:1:1
num_groups = len(bug_ids_groups)
train_size = math.ceil(num_groups * 0.8)
val_size = math.ceil(num_groups * 0.1)

# 从分组的bugid中恢复原始数据集并写入新的txt文件
for i, bug_ids_group in enumerate(bug_ids_groups):
    if i < train_size:
        output_file = f"train_data_{i}.txt"
    elif i < train_size + val_size:
        output_file = f"val_data_{i - train_size}.txt"
    else:
        output_file = f"test_data_{i - train_size - val_size}.txt"

    with open(output_file, "w", encoding="utf-8") as f:
        for bug_id in bug_ids_group:
            for entry in bug_data_dict[bug_id]:
                f.write("$$".join(entry) + "\n")

    print(f"第 {i+1} 组数据已成功写入文件: {output_file}")
