

# 移动 "C:\\workspace\\datasets\\coco-pp\\images\\val" 中文件 到  "C:\\workspace\\datasets\\coco-pp\\images\\train",
 #移动  "C:\\workspace\\datasets\\coco-pp\\labels\\val" 中文件 到  "C:\\workspace\\datasets\\coco-pp\\labels\\train",

 #随机抽取  "C:\\workspace\\datasets\\coco-pp\\images\\train" 20%数据 , 移动到 "C:\\workspace\\datasets\\coco-pp\\images\\val"
 # 根据"C:\\workspace\\datasets\\coco-pp\\images\\val" 里面文件名称（去除后缀），去"C:\\workspace\\datasets\\coco-pp\\labels\\train"里面找对应的txt , 并移动到  "C:\\workspace\\datasets\\coco-pp\\labels\\val"


import os
import random
import shutil

# 定义数据集路径
images_train_dir = "C:\\workspace\\datasets\\coco-pp\\images\\train"
images_val_dir = "C:\\workspace\\datasets\\coco-pp\\images\\val"
labels_train_dir = "C:\\workspace\\datasets\\coco-pp\\labels\\train"
labels_val_dir = "C:\\workspace\\datasets\\coco-pp\\labels\\val"


# 将 val 文件夹中的数据移动到 train 文件夹
def move_val_to_train(images_val_dir, images_train_dir, labels_val_dir, labels_train_dir):
    # 移动图片文件
    for file_name in os.listdir(images_val_dir):
        src_path = os.path.join(images_val_dir, file_name)
        dst_path = os.path.join(images_train_dir, file_name)
        shutil.move(src_path, dst_path)

    # 移动标签文件
    for file_name in os.listdir(labels_val_dir):
        src_path = os.path.join(labels_val_dir, file_name)
        dst_path = os.path.join(labels_train_dir, file_name)
        shutil.move(src_path, dst_path)


# 随机抽取 20% 的图片文件并移动到新的 val 文件夹
def split_train_val(images_train_dir, images_val_dir, labels_train_dir, labels_val_dir, val_ratio=0.2):
    all_images = [f for f in os.listdir(images_train_dir) if f.endswith('.jpg')]
    val_size = int(len(all_images) * val_ratio)
    val_images = random.sample(all_images, val_size)

    for image_file in val_images:
        # 移动图片文件到 val 文件夹
        src_image_path = os.path.join(images_train_dir, image_file)
        dst_image_path = os.path.join(images_val_dir, image_file)
        shutil.move(src_image_path, dst_image_path)

        # 根据图片文件名找到对应的标签文件并移动
        label_file = image_file.replace('.jpg', '.txt')
        src_label_path = os.path.join(labels_train_dir, label_file)
        dst_label_path = os.path.join(labels_val_dir, label_file)
        if os.path.exists(src_label_path):  # 检查标签文件是否存在
            shutil.move(src_label_path, dst_label_path)


# 执行步骤
move_val_to_train(images_val_dir, images_train_dir, labels_val_dir, labels_train_dir)
split_train_val(images_train_dir, images_val_dir, labels_train_dir, labels_val_dir, val_ratio=0.2)

print("数据重新分配完成！")
