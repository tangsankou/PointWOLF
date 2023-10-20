import os
import csv

def write_to_csv(checkpoints_dir, output_file):

    # 创建 csv 文件并写入表头
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Folder Name", "Test Acc", "Test Avg Acc"])

    # 遍历 checkpoints 文件夹下的所有文件夹
    for folder_name in os.listdir(checkpoints_dir):
        # 判断文件夹是否以 "_eval" 结尾
        if folder_name.endswith("eval"):
            # 构造 run.log 文件的路径
            log_file = os.path.join(checkpoints_dir, folder_name, "run.log")
            # 如果 run.log 文件存在，则读取其中的 "test acc:" 和 "test avg acc:" 后面的数字
            if os.path.exists(log_file):
                with open(log_file, "r") as f:
                    content = f.read()
                    test_acc = float(str(content.split("test acc:")[1].split(",")[0].strip()))
                    test_avg_acc = float(str(content.split("test avg acc:")[1].split("\n")[0].strip()))
                # 将读取到的数据写入 csv 文件
                with open(output_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([folder_name, test_acc, test_avg_acc])

if __name__ == "__main__":
    # 示例用法
    # 定义 checkpoints 文件夹的路径
    checkpoints_dir = "/home/user_tp/workspace/code/defense/PointWOLF/checkpoints"

    # 定义输出 csv 文件的路径和文件名
    output_file = "/home/user_tp/workspace/code/defense/PointWOLF/saliency_method1.csv"
    write_to_csv(checkpoints_dir, output_file)