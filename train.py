import torch.backends.cudnn as cudnn
import torch.optim as optim
import sys, time
from os.path import join
import torch
from torch.utils.data import DataLoader
import random
from lib.datasetV2 import TrainDatasetV2, create_patch_idx, data_preprocess
from lib.losses.loss import *
from lib.common import *
from config import parse_args
from lib.logger import Logger, Print_Logger
import models
from models.MDF import MDF
from test import Test
from function import get_dataloader, train, val, get_dataloaderV2
from torch.utils.data.dataset import Subset
from lib.extract_patches import get_data_train
from lib.losses.loss import *
from lib.visualize import group_images, save_img
from collections import OrderedDict
from lib.metrics import Evaluate
from tqdm import tqdm
torch.backends.cudnn.benchmark = True

from sklearn.model_selection import KFold


def main():
    setpu_seed(2021)
    args = parse_args()
    save_path = join(args.outf, args.save)
    save_args(args, save_path)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    cudnn.benchmark = True

    kf = KFold(n_splits=5, shuffle=True, random_state=0)

    log = Logger(save_path)
    sys.stdout = Print_Logger(os.path.join(save_path, 'train_log.txt'))
    print('The computing device used is: ', 'GPU' if device.type == 'cuda' else 'CPU')

    net = MDF(img_ch=args.in_channels, output_ch=args.classes).to(device)
    print("Total number of parameters: " + str(count_parameters(net)))

    log.save_graph(net, torch.randn((1, 1, 48, 48)).to(device).to(
        device=device))  # Save the model structure to the tensorboard file
    # torch.nn.init.kaiming_normal(net, mode='fan_out')      # Modify default initialization method
    # net.apply(weight_init)
    criterion = CrossEntropyLoss2d()  # Initialize loss function
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    # The training speed of this task is fast, so pre training is not recommended
    if args.pre_trained is not None:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load(args.outf + '%s/latest_model.pth' % args.pre_trained)
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.start_epoch = checkpoint['epoch'] + 1

    # criterion = LossMulti(jaccard_weight=0,class_weights=np.array([0.5,0.5]))
    # create a list of learning rate with epochs
    # lr_schedule = make_lr_schedule(np.array([50, args.N_epochs]),np.array([0.001, 0.0001]))
    # lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.5)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.N_epochs, eta_min=0)

    imgs_train, masks_train, fovs_train = data_preprocess(data_path_list=args.train_data_path_list)

    patches_idx = create_patch_idx(fovs_train, args)


    # Save some samples of feeding to the neural network
    if args.sample_visualization:
        visual_set = TrainDatasetV2(imgs_train, masks_train, fovs_train, patches_idx, mode="val", args=args)
        visual_loader = DataLoader(visual_set, batch_size=1, shuffle=True, num_workers=0)
        N_sample = 50
        visual_imgs = np.empty((N_sample, 1, args.train_patch_height, args.train_patch_width))
        visual_masks = np.empty((N_sample, 1, args.train_patch_height, args.train_patch_width))

        for i, (img, mask) in tqdm(enumerate(visual_loader)):
            visual_imgs[i] = np.squeeze(img.numpy(), axis=0)
            visual_masks[i, 0] = np.squeeze(mask.numpy(), axis=0)
            if i >= N_sample - 1:
                break

    for fold, (train_index, val_index) in enumerate(kf.split(patches_idx)):
        print(f'Fold {fold + 1} / 5')
        train_set = TrainDatasetV2(imgs_train, masks_train, fovs_train, patches_idx[train_index], mode="train", args=args)
        val_set = TrainDatasetV2(imgs_train, masks_train, fovs_train, patches_idx[val_index], mode="val", args=args)
        train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
        val_loader = DataLoader(val_set, batch_size=args.batch_size,
                             shuffle=False, num_workers=0)


        if args.val_on_test:
            print('\033[0;32m===============Validation on Testset!!!===============\033[0m')
            val_tool = Test(args)

        best = {'epoch': 0, 'acc': 0.5}  # Initialize the best epoch and performance(AUC of ROC)
        trigger = 0  # Early stop Counter
        for epoch in range(args.start_epoch, args.N_epochs + 1):
            print('\nEPOCH: %d/%d --(learn_rate:%.6f) | Time: %s' % \
                  (epoch, args.N_epochs, optimizer.state_dict()['param_groups'][0]['lr'], time.asctime()))

            # train stage
            train_log = train(train_loader, net, criterion, optimizer, device)
            # val stage
            if not args.val_on_test:
                val_log = val(val_loader, net, criterion, device)
            else:
                val_tool.inference(net)
                val_log = val_tool.val()

            log.update(epoch, train_log, val_log, fold)  # Add log information
            lr_scheduler.step()

            # Save checkpoint of latest and best model.
            state = {'net': net.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
            torch.save(state, join(save_path, f'latest_model_fold_{fold}.pth'))
            trigger += 1
            if val_log['val_acc'] > best['acc']:
                print('\033[0;33mSaving best model for this fold!\033[0m')
                torch.save(state, join(save_path, f'best_model_fold_{fold}.pth'))
                best['epoch'] = epoch
                best['acc'] = val_log['val_acc']
                trigger = 0
            print(f'Best performance at Epoch (Fold {fold}): {best["epoch"]} | acc {best["acc"]}')
            # early stopping
            if not args.early_stop is None:
                if trigger >= args.early_stop:
                    print("=> early stopping for this fold")
                    break
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()