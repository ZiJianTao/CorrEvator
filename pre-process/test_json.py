import json
import os

def validate_and_correct_json(file_path):
    if os.path.getsize(file_path) == 0:
        print(f"File {file_path} is empty. Skipping...")
        return

    while True:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                json.loads(content)
                print(f"The content of {file_path} is valid JSON.")
                break  # 如果验证成功，退出循环
        except json.JSONDecodeError as e:
            print(f"Error in {file_path}: {e}")

            if e.pos is not None:  # 仅在有错误位置信息时执行修正
                # 获取错误位置的字符
                error_index = e.pos
                corrected_content = content[:error_index] + content[error_index + 1:]

                # 重新保存到文件
                with open(file_path, 'w', encoding='utf-8') as corrected_file:
                    corrected_file.write(corrected_content)
                    print(f"The content has been corrected and saved to {file_path}")
        except (FileNotFoundError, OSError) as e:
            print(f"Error opening file {file_path}: {e}")
            break


# 设置要处理的目录
directory = "C:\\Users\\Karrol\\Desktop\\Quatrain-main\\TEST DATA\\RESHAPED\\PATCH-SIM"

# 遍历目录下的所有txt文件
for filename in os.listdir(directory):
    if filename.endswith(".txt"):
        file_path = os.path.join(directory, filename)
        validate_and_correct_json(file_path)
