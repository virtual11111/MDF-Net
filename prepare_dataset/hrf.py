#=========================================================
#   HRF 数据集路径列表生成（只保留三者都存在的样本，mask名为原图名去后缀加_mask.tif）
#=========================================================
import os
from os.path import join, exists, normpath, splitext

def get_path_list(root_path, img_dir, mask_dir, fov_dir):
    img_list = sorted(os.listdir(join(root_path, img_dir)))
    # 只保留图片文件
    img_list = [f for f in img_list if f.lower().endswith(('jpg', 'jpeg', 'png', 'tif', 'tiff'))]
    triplets = []
    for img_name in img_list:
        base, _ = splitext(img_name)
        mask_name = f"{base}_mask.tif"
        fov_name = f"{base}.tif"
        img_path = normpath(join(root_path, img_dir, img_name)).replace('\\', '/')
        mask_path = normpath(join(root_path, mask_dir, mask_name)).replace('\\', '/')
        fov_path = normpath(join(root_path, fov_dir, fov_name)).replace('\\', '/')
        if exists(img_path) and exists(mask_path) and exists(fov_path):
            triplets.append((img_path, mask_path, fov_path))
        else:
            print(f"[skip] 缺失: {img_path if not exists(img_path) else ''} {mask_path if not exists(mask_path) else ''} {fov_path if not exists(fov_path) else ''}")
    return triplets

def write_path_list(triplet_list, save_path, file_name):
    with open(join(save_path, file_name), 'w') as f:
        for img, mask, fov in triplet_list:
            f.write(f"{img} {mask} {fov}\n")

if __name__ == "__main__":
    data_root_path = 'E:/VesselSeg-Pytorch-master/data/HRF'
    img_dir = 'images'
    mask_dir = 'mask'
    fov_dir = 'manual1'
    save_path = 'E:/VesselSeg-Pytorch-master/prepare_dataset/data_path_list/HRF'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    triplet_list = get_path_list(data_root_path, img_dir, mask_dir, fov_dir)
    print('有效图片数:', len(triplet_list))
    test_range = (15, 30)  # 第15到第30个为测试集
    test_list = triplet_list[test_range[0]:test_range[1]]
    train_list = triplet_list[:test_range[0]] + triplet_list[test_range[1]:]
    print('训练集:', len(train_list), '测试集:', len(test_list))
    write_path_list(train_list, save_path, 'train.txt')
    write_path_list(test_list, save_path, 'test.txt')
    print('Finish!') 