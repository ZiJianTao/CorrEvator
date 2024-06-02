from gmn.util import wordext
import os
import json
# 读取BATS结果的JSON文件
with open('C:\\Users\\Karrol\\Desktop\\Quatrain-main\\data\\BATS_RESULT_0.8.json', 'r') as f:
    BATS_RESULTS_json = json.load(f)

def get_tokens(text):
    # 确保 text 是字符串
    text = str(text)
    return wordext.get_words_from_text(text)

def reshape_dataset(data):
    """
    重新塑造数据集的函数
    """
    reshaped_data = []
    for entry in data:
        # 按照分隔符 "$$" 切分每个条目
        parts = entry.split("$$")

        # Perform special check on parts[3]
        if parts[3].lower() in BATS_RESULTS_json.keys():
            # 构造重新塑造后的字符串
            reshaped_entry = f'{{"A_title": "{parts[1] + parts[2]}", "A_clean_title": {get_tokens(parts[1] + parts[2])}, "B_title": "{parts[4]}", "B_clean_title": {get_tokens(parts[4])}}},'

            # 添加到重新塑造的数据集中
            reshaped_data.append(reshaped_entry)
        #print("len(reshaped_data):",len(reshaped_data))
    return reshaped_data

# 指定目录路径
directory = 'C:\\Users\\karrol\\Desktop\\Quatrain-main\\TEST DATA\\RAW_DATA'

# 遍历目录中的文件
for filename in os.listdir(directory):
    if filename.endswith(".txt"):
        input_file = os.path.join(directory, filename)  # 构建完整的文件路径
        output_file = f"C:\\Users\\karrol\\Desktop\\Quatrain-main\\TEST DATA\\preprocessed_BATS0.8_{filename}"  # 构建输出文件路径，加上前缀 "preprocessed_"

        # 从本地txt文件读取数据
        with open(input_file, 'r', encoding='utf-8') as file:
            raw_dataset = file.read().splitlines()

        # 调用函数并将结果保存为文本文件
        reshaped_dataset = reshape_dataset(raw_dataset)

        # 将结果保存到文本文件
        with open(output_file, 'w', encoding='utf-8') as file:
            for entry in reshaped_dataset:
                file.write(entry + '\n')

        print(f"处理完成，数据已保存到 {output_file} 文件。")
