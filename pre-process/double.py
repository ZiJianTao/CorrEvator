import os

# 指定目录
directory = "C:\\Users\\Karrol\\Desktop\\Quatrain-main\\TEST DATA\\RESHAPED"

# 遍历目录下的所有txt文件
for filename in os.listdir(directory):
    if filename.endswith(".txt"):
        file_path = os.path.join(directory, filename)

        # 读取文件内容
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()

        # 替换单引号为双引号
        content = content.replace("'", '"')

        # 将处理后的内容写回文件
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(content)

        print(f"已处理文件 {filename}。")
