import json
import numpy as np
import random
import os


def process_file(file_path):
    """
    处理单个文件的函数
    """
    # 读取开发者提交消息的 JSON 文件
    with open('C:\\Users\\北弋\\Desktop\\Quatrain-main\\data\\CommitMessage\\Developer_commit_message.json', 'r',encoding="utf-8") as f:
        Developer_commit_message_dict = json.load(f)

    train_ids = set()  # 用集合来保存不重复的 bug-id

    # 读取 txt 文件并解析为数据样本列表
    with open(file_path, "r", encoding="utf-8") as file:
        dataset_txt_content = file.readlines()

    dataset_samples = []  # 用来存储数据样本

    # 解析每一行数据，并将其添加到 dataset_samples 中
    for line in dataset_txt_content:
        parts = line.strip().split("$$")  # 按照 "$$" 分割每行数据
        bug_id = parts[0].strip()
        dataset_samples.append(parts)
        train_ids.add(bug_id)  # 将 bug-id 添加到集合中

    # 提取出训练集中包含的 bug-id，并构建随机 bug 报告列表
    random_bug_report_list = [line for line in dataset_txt_content if line.strip().split("$$")[0] in train_ids]
    train_features = []

    # 遍历每个训练样本
    for sample in dataset_samples:
        bug_id = sample[0].strip()
        bug_report_summary = sample[1].strip()
        bug_report_description = sample[2].strip()
        patch_id = sample[3].strip()

        patch_description = sample[4].strip()
        label = sample[5].strip()

        project, id = patch_id.split('_')[0].split('-')[1], patch_id.split('_')[0].split('-')[2]
        project_id = f"{project}-{id}"

        if project_id == 'closure-63':
            developer_commit_message = Developer_commit_message_dict['closure-62']
        elif project_id == 'closure-93':
            developer_commit_message = Developer_commit_message_dict['closure-92']
        else:
            developer_commit_message = Developer_commit_message_dict[project_id]

        label = int(label)

        if '_Developer_' in patch_id:
            features = f"{bug_id}$${bug_report_summary}$${bug_report_description}$${patch_id}$${developer_commit_message}$${label}"
            train_features.append(features)

            random.seed(1)
            random_negative_example = random.choice(random_bug_report_list)
            random_negative_example_parts = random_negative_example.split("$$")[:3]
            features_random = "$$".join(random_negative_example_parts) + f"$${patch_id}$${developer_commit_message}$$0"
            train_features.append(features_random)

    with open(file_path, 'w', encoding="utf-8") as f:
        for line in dataset_txt_content:
            if '_Developer_' not in line:
                f.write(line)

    with open(file_path, 'a', encoding="utf-8") as file:
        for feature in train_features:
            file.write(feature + '\n')


# 指定目录路径
directory = 'C:\\Users\\北弋\\Desktop\\Quatrain-main\\dataprocess'

# 遍历目录中的文件
for filename in os.listdir(directory):
    if filename.endswith("_train.txt"):
        # 构建完整的文件路径
        filepath = os.path.join(directory, filename)

        # 调用处理文件的函数
        process_file(filepath)
