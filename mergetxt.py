import os

output_file = ".\\training\dataset\\prompt.json"  # 合并后的文件名
input_dir = ".\\training\dataset\prompt\\"  # 输入文件所在的目录
prefix = ""  # 文件名前缀
suffix = ".txt"  # 文件名后缀
num_files = 0  # 输入文件的数量

# 读取所有输入文件的内容
file_texts = []
for file_name in sorted(os.listdir(input_dir)):
    if file_name.startswith(prefix) and file_name.endswith(suffix):
        with open(os.path.join(input_dir, file_name), "r") as f:
            file_texts.append('{"source": "source/' + file_name[:-4] + '.png", "target": "target/' + file_name[:-4] + '.png", "prompt": "' + f.readline().strip() + '"}')
        num_files += 1

# 将所有输入文件的内容写入输出文件中
with open(output_file, "w") as f:
    for text in file_texts:
        f.write(text + "\n")

print(f"Merged {num_files} files into {output_file}")