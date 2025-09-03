# -*- coding:utf-8 -*
import os
import random
import os
import shutil
import argparse

def data_split(full_list, ratio):

    n_total = len(full_list)
    offset = int(n_total * ratio)
    if n_total == 0 or offset < 1:
        return [], full_list
    random.shuffle(full_list)
    sublist_1 = full_list[:offset]
    sublist_2 = full_list[offset:]
    return sublist_1, sublist_2


# train_p="C:/Users/HL/Downloads/wendang_labels/train"
# val_p="C:/Users/HL/Downloads/wendang_labels/val"
# imgs_p="images"
# labels_p="labels"

def spilit_dataset(train_p,val_p,imgs_p,labels_p,images_dir,labels_dir,proportion_ ):
    #创建训练集
    if not os.path.exists(train_p):#指定要创建的目录
        os.mkdir(train_p)
    tp1=os.path.join(train_p,imgs_p)
    tp2=os.path.join(train_p,labels_p)
    print(tp1,tp2)
    if not os.path.exists(tp1):#指定要创建的目录
        os.mkdir(tp1)
    if not os.path.exists(tp2):  # 指定要创建的目录
        os.mkdir(tp2)

    #创建测试集文件夹
    if not os.path.exists(val_p):#指定要创建的目录
        os.mkdir(val_p)
    vp1=os.path.join(val_p,imgs_p)
    vp2=os.path.join(val_p,labels_p)
    print(vp1,vp2)
    if not os.path.exists(vp1):#指定要创建的目录
        os.mkdir(vp1)
    if not os.path.exists(vp2):  # 指定要创建的目录
        os.mkdir(vp2)

    #划分数据集，设置数据集数量占比

    # total_file = os.listdir(images_dir)
    total_file =  [f for f in os.listdir(images_dir) if f.lower().endswith('.json')]

    num = len(total_file)  # 统计所有的标注文件
    list_=[]
    for i in range(0,num):
        list_.append(i)

    list1,list2=data_split(list_,proportion_)

    for i in range(0,num):
        file=total_file[i]
        print(i,' - ',total_file[i])
        name=file.split('.json')[0]
        if i in list1:
            jpg_1 = os.path.join(images_dir, file)
            jpg_2 = os.path.join(train_p, imgs_p, file)
            txt_1 = os.path.join(labels_dir, name + '.txt')
            txt_2 = os.path.join(train_p, labels_p, name + '.txt')
            if os.path.exists(txt_1) and os.path.exists(jpg_1):
                shutil.copyfile(jpg_1, jpg_2)
                shutil.copyfile(txt_1, txt_2)
            elif os.path.exists(txt_1):
                print(txt_1)
            else:
                print(jpg_1)

        elif i in list2:
            jpg_1 = os.path.join(images_dir, file)
            jpg_2 = os.path.join(val_p, imgs_p, file)
            txt_1 = os.path.join(labels_dir, name + '.txt')
            txt_2 = os.path.join(val_p, labels_p, name + '.txt')
            shutil.copyfile(jpg_1, jpg_2)
            shutil.copyfile(txt_1, txt_2)

    print("数据集划分完成： 总数量：",num," 训练集数量：",len(list1)," 验证集数量：",len(list2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='split dataset params')
    parser.add_argument('--train_p', type=str, default='C:/Users/HL/Downloads/wendang_labels/train', help='train path')
    parser.add_argument('--val_p', type=str, default='C:/Users/HL/Downloads/wendang_labels/val', help='val path')
    parser.add_argument('--imgs_p', type=str, default='images', help='images path')
    parser.add_argument('--labels_p', type=str, default='labels', help='labels path')
    parser.add_argument('--images_dir', type=str, default='C:/Users/HL/Downloads/wendang_labels/images20250826/', help='images dir')
    parser.add_argument('--labels_dir', type=str, default='C:/Users/HL/Downloads/wendang_labels/label_test/', help='labels dir')
    parser.add_argument('--proportion', type=float, default=0.9, help='proportion')
    args = parser.parse_args()

    train_p = args.train_p
    val_p = args.val_p
    imgs_p = args.imgs_p
    labels_p = args.labels_p
    images_dir = args.images_dir
    labels_dir = args.labels_dir
    # images_dir = "C:/Users/HL/Downloads/wendang_labels/images_test/"
    # labels_dir = "C:/Users/HL/Downloads/wendang_labels/label_test/"
    proportion_ = args.proportion

    spilit_dataset(train_p, val_p, imgs_p, labels_p, images_dir, labels_dir, proportion_)

