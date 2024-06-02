import random

def random_sample(input_file, output_file, percent):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = f.readlines()

    random.seed(1)
    sample_size = round(len(data) * percent / 100)
    sampled_data = random.sample(data, sample_size)

    with open(output_file, 'w', encoding='utf-8') as f:  # 修改这一行
        f.writelines(sampled_data)


# 文件路径
input_file = 'C:\\Users\\Karrol\\Desktop\\train-dataset\\first_msr_pairs_pull_info_X.txt'
output_files = {
    '20%': 'DUP_20.txt',
    '40%': 'DUP_40.txt',
    '60%': 'DUP_60.txt',
    '80%': 'DUP_80.txt',
    '100%': 'DUP_100.txt'
}

# 随机选择并保存不同比例的数据
for percent, output_file in output_files.items():
    random_sample(input_file, output_file, int(percent[:-1]))

print("数据集已按指定比例保存。")
