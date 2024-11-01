import os


folder_paths ={ "C:\\workspace\\datasets\\coco-pp\\labels\\train",
               "C:\\workspace\\datasets\\coco-pp\\labels\\val"
}

def change_class(folder_path,class_no):
    # 遍历文件夹中的所有txt文件
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):
            file_path = os.path.join(folder_path, file_name)

            # 读取文件内容
            with open(file_path, 'r') as file:
                lines = file.readlines()

            # 修改 class == 0
            updated_lines = []
            for line in lines:
                components = line.strip().split()

                components[0] = class_no  # 修改class编号
                updated_lines.append(' '.join(components) + '\n')

            # 保存修改后的文件
            with open(file_path, 'w') as file:
                file.writelines(updated_lines)

for folder_path in folder_paths:
    change_class(folder_path,'80')


print("修改完成！")
