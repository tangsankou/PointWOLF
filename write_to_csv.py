import os
import csv
import pandas as pd

NOISE = ['uniform',
        'gaussian',
       'background',
       'impulse',
       'upsampling',
       'shear',
       'rotation',
       'cutout',
       'density',
       'density_inc',
       'distortion',
       'distortion_rbf',
       'distortion_rbf_inv',
       'occlusion',
       'lidar',
       'original'
]

FOLD = ['P_pointnet',
        'P_pointnet_0.5',
       'P_pointnet_saliency',
       'PA_pointnet',
       'PA_pointnet_0.5',
       'PA_pointnet_gsaliency'
]

def write_to_csv(checkpoints_dir, output_file):
    # noise = [x for x in NOISE]
    # # 创建 csv 文件并写入表头
    # with open(output_file, "w", newline="") as f:
    #     writer = csv.writer(f)
    #     writer.writerow([" ", "Test Acc", "Test Avg Acc"])

    # 遍历 checkpoints 文件夹下的所有文件夹
    test_acc = []
    class_acc = []
    noise = []
    for i in range(0, len(FOLD)):
        acc = []
        cacc = []
        log_file = os.path.join(checkpoints_dir, FOLD[i], "testc.log")
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                line = f.read().strip()
                linestr = line.split("\n")         # 以换行符分隔
                for j in range(2, len(linestr)):
                    # print(linestr[j])
                    if i == 0:
                        noise.append(str(linestr[j].split(":")[0].strip()))
                    acc.append(float(str(linestr[j].split("Test Instance Accuracy: ")[1].split(",")[0].strip())))
                    cacc.append(float(str(linestr[j].split("Class Accuracy: ")[1].split(",")[0].strip())))   
            test_acc.append(acc)
            class_acc.append(cacc)
                # test_avg_acc = float(str(content.split("test avg acc:")[1].split("\n")[0].strip()))
                # 将读取到的数据写入 csv 文件
    dataframe = pd.DataFrame({' ':noise, f'{FOLD[0]}'+'_tacc':test_acc[0], f'{FOLD[1]}'+'_tacc':test_acc[1], f'{FOLD[2]}'+'_tacc':test_acc[2], f'{FOLD[3]}'+'_tacc':test_acc[3], f'{FOLD[4]}'+'_tacc':test_acc[4], f'{FOLD[5]}'+'_tacc':test_acc[5], 
                              f'{FOLD[0]}'+'_cacc':class_acc[0], f'{FOLD[1]}'+'_cacc':class_acc[1], f'{FOLD[2]}'+'_cacc':class_acc[2], f'{FOLD[3]}'+'_cacc':class_acc[3], f'{FOLD[4]}'+'_cacc':class_acc[4], f'{FOLD[5]}'+'_cacc':class_acc[5]})
    dataframe.to_csv(output_file, sep=',')
    # with open(output_file, "a", newline="") as f:
    #     writer = csv.writer(f)
    #     writer.writerow([folder_name, test_acc])
      # writer.writerow([folder_name, test_acc, test_avg_acc])

if __name__ == "__main__":
    # 示例用法
    # 定义 checkpoints 文件夹的路径
    checkpoints_dir = "/home/user_tp/workspace/code/defense/PointWOLF/checkpoints"

    # 定义输出 csv 文件的路径和文件名
    output_file = "/home/user_tp/workspace/code/defense/PointWOLF/test_modelnet40c.csv"
    write_to_csv(checkpoints_dir, output_file)