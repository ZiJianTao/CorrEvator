import os


def count_lines_and_create_new_file(directory):
    # 获取目录下所有txt文件
    txt_files = [f for f in os.listdir(directory) if f.endswith('.txt')]

    for txt_file in txt_files:
        with open(os.path.join(directory, txt_file), 'r', encoding='utf-8') as file:

            lines = file.readlines()
            line_count = len(lines)

        # 构造新的txt文件名
        new_file_name = os.path.splitext(txt_file)[0] + 'y.txt'

        # 写入新的txt文件
        with open(os.path.join(directory, new_file_name), 'w',encoding='utf-8') as new_file:
            new_file.write(str([1] * line_count))

        print(f"Processed {txt_file}, created {new_file_name} with {line_count} lines.")


# 调用函数并指定目录
directory_path = "C:\\Users\\Karrol\\Desktop\\Quatrain-main\\TEST DATA\\RESHAPED\\PATCH-SIM"
count_lines_and_create_new_file(directory_path)
