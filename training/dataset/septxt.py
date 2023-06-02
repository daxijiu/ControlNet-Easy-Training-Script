input_file = "prompt.txt"  # 输入文件名
output_prefix = ""  # 输出文件名前缀
output_suffix = ".txt"  # 输出文件名后缀

with open(input_file, "r") as f:
    for i, line in enumerate(f):
        output_file = output_prefix + str(i) + output_suffix
        with open(output_file, "w") as out_f:
            out_f.write(line)

print(f"Saved {i+1} lines to files")