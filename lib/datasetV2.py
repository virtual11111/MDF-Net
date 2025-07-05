"""
This dataset is used reduce memory usage during training
"""
import cv2
from torch.utils.data import Dataset
import torch
import numpy as np
import random
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms.functional import normalize

from .extract_patches import load_data, my_PreProc, is_patch_inside_FOV
from .dataset import RandomCrop, RandomFlip_LR, RandomFlip_UD, RandomRotate, Compose

class TrainDatasetV2(Dataset):
    def __init__(self, imgs, masks, fovs, patches_idx, mode, args):
        self.imgs = imgs
        self.masks = masks
        self.fovs = fovs
        self.patch_h, self.patch_w = args.train_patch_height, args.train_patch_width
        self.patches_idx = patches_idx
        self.inside_FOV = args.inside_FOV
        self.transforms = None
        if mode == "train":
            self.transforms = Compose([
                RandomCrop((48, 48)),
                RandomFlip_LR(prob=0.5),
                RandomFlip_UD(prob=0.5),
                RandomRotate()  # 使用修改后的 RandomRotate 类
            ])

    def __len__(self):
        return len(self.patches_idx)

    def __getitem__(self, idx):
        n, x_center, y_center = self.patches_idx[idx]

        data = self.imgs[n, :, y_center - int(self.patch_h / 2):y_center + int(self.patch_h / 2),
                         x_center - int(self.patch_w / 2):x_center + int(self.patch_w / 2)]
        mask = self.masks[n, :, y_center - int(self.patch_h / 2):y_center + int(self.patch_h / 2),
                         x_center - int(self.patch_w / 2):x_center + int(self.patch_w / 2)]

        data = torch.from_numpy(data).float()
        mask = torch.from_numpy(mask).long()

        if self.transforms:
            data, mask = self.transforms(data, mask)
        return data, mask.squeeze(0)



'''class TrainDatasetV2(Dataset):
    def __init__(self, imgs,masks,fovs,patches_idx,mode,args):
        self.imgs = imgs

        self.masks = masks
        self.fovs = fovs
        self.patch_h, self.patch_w = args.train_patch_height, args.train_patch_width
        self.patches_idx = patches_idx
        self.inside_FOV = args.inside_FOV
        self.transforms = None
        if mode == "train":
            self.transforms = Compose([
                # RandomResize([56,72],[56,72]),
                RandomCrop((48, 48)),
                RandomFlip_LR(prob=0.5),
                RandomFlip_UD(prob=0.5),
                RandomRotate()
            ])

    def __len__(self):
        return len(self.patches_idx)

    def __getitem__(self, idx):
        n, x_center, y_center = self.patches_idx[idx]

        data = self.imgs[n,:,y_center-int(self.patch_h/2):y_center+int(self.patch_h/2),x_center-int(self.patch_w/2):x_center+int(self.patch_w/2)]
        mask = self.masks[n,:,y_center-int(self.patch_h/2):y_center+int(self.patch_h/2),x_center-int(self.patch_w/2):x_center+int(self.patch_w/2)]

        data = torch.from_numpy(data).float()
        mask = torch.from_numpy(mask).long()

        if self.transforms:
            data, mask = self.transforms(data, mask)
        return data, mask.squeeze(0)'''


#----------------------Related Methon--------------------------------------
def data_preprocess(data_path_list):
    train_imgs_original, train_masks, train_FOVs = load_data(data_path_list)
    # save_img(group_images(train_imgs_original[0:20,:,:,:],5),'imgs_train.png')#.show()  #check original train imgs 

    train_imgs = my_PreProc(train_imgs_original)
    #添加resize
    train_masks = train_masks//255
    train_FOVs = train_FOVs//255
    return train_imgs, train_masks, train_FOVs



def create_patch_idx(img_fovs, args):
    assert len(img_fovs.shape)==4
    N,C,img_h,img_w = img_fovs.shape
    res = np.empty((args.N_patches,3),dtype=int)
    print("")

    seed=2021
    count = 0
    while count < args.N_patches:
        random.seed(seed) # fuxian
        seed+=1
        n = random.randint(0,N-1)
        x_center = random.randint(0+int(args.train_patch_width/2),img_w-int(args.train_patch_width/2))
        y_center = random.randint(0+int(args.train_patch_height/2),img_h-int(args.train_patch_height/2))

        #check whether the patch is contained in the FOV
        if args.inside_FOV=='center' or args.inside_FOV == 'all':
            if not is_patch_inside_FOV(x_center,y_center,img_fovs[n,0],args.train_patch_height,args.train_patch_width,mode=args.inside_FOV):
                continue
        res[count] = np.asarray([n,x_center,y_center])
        count+=1

    return res

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # 假设 imgs 和 masks 是预处理后的训练图像和掩膜数据
    imgs = np.random.rand(5, 3, 128, 128)  # 5 张随机生成的图像
    masks = np.random.randint(0, 2, (5, 1, 128, 128))  # 5 张随机生成的掩膜
    fovs = np.ones((5, 1, 128, 128))  # 随机生成的FOV

    patches_idx = [(i, 64, 64) for i in range(5)]  # 生成随机索引
    class Args:
        train_patch_height = 48
        train_patch_width = 48
        inside_FOV = 'center'
    args = Args()

    dataset = TrainDatasetV2(imgs, masks, fovs, patches_idx, mode="train", args=args)

    # 查看增强后的数据
    for i in range(5):  # 查看前5个样本
        img, mask = dataset[i]
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(img.permute(1, 2, 0).numpy())  # 转换为HWC格式
        plt.title("Augmented Image")
        plt.subplot(1, 2, 2)
        plt.imshow(mask.numpy(), cmap="gray")
        plt.title("Augmented Mask")
        plt.show()
