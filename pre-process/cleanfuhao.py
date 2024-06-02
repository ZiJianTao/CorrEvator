import os


def remove_punctuation(text):
    punctuation = "\'\,\[\]\{\}\""
    for char in punctuation:
        text = text.replace(char, "")
    return text


# 指定要处理的目录
directory = "C:\\Users\\Karrol\\Desktop\\Quatrain-main\\TEST DATA\\RESHAPED"

# 遍历目录下的所有txt文件并删除标点符号
for filename in os.listdir(directory):
    if filename.endswith(".txt"):
        file_path = os.path.join(directory, filename)
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()

        # 删除标点符号
        content_cleaned = remove_punctuation(content)

        # 将处理后的内容写回文件
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(content_cleaned)

        print(f"已删除文件 {filename} 中的标点符号。")
