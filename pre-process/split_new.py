import os

# 指定目录路径
directory = 'C:\\Users\\北弋\\Desktop\\Quatrain-main\\dataprocess'  # 目录路径根据实际情况修改

# 遍历目录中的文件
for filename in os.listdir(directory):
    if filename.endswith(".txt"):
        input_file = os.path.join(directory, filename)  # 构建完整的文件路径
        output_file_0 = f"C:\\Users\\北弋\\Desktop\\Quatrain-main\\dataprocess\\{filename[:-4]}_0.txt"  # 修改输出文件路径，根据实际情况修改
        output_file_1 = f"C:\\Users\\北弋\\Desktop\\Quatrain-main\\dataprocess\\{filename[:-4]}_1.txt"  # 修改输出文件路径，根据实际情况修改

        with open(input_file, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        # 分别创建两个txt文件用于存储标号为0和1的行
        file_0 = open(output_file_0, 'w', encoding='utf-8')
        file_1 = open(output_file_1, 'w', encoding='utf-8')

        # 根据标号将每一行写入相应的文件
        for line in lines:
            # 假设标号是每行数据的最后一个字符
            label = int(line.strip()[-1])

            if label == 0:
                file_0.write(line)
            elif label == 1:
                file_1.write(line)

        # 关闭文件
        file_0.close()
        file_1.close()

        print(f"分离完成，数据已写入{output_file_0}和{output_file_1}文件。")
