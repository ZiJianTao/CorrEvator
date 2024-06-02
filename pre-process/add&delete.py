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

        # 如果文件不为空，则处理文件内容
        if content:
            # 在内容开头加上 "["
            content = "[" + content
            # 去掉最后一个逗号，并在末尾加上 "]"
            #if content.endswith(",]"):
            content = content[:-2] + "]"
            #else:
                #content += "]"

            # 将处理后的内容写回文件
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(content)

        print(f"已处理文件 {filename}。")
