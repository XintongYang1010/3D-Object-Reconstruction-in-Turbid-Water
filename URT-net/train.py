import numpy as np
import torch
import argparse
import cv2
import torch.utils.data as data
import torchvision
import random
import torch.nn.functional as F
import torch.nn as nn
# from tensorboardX import SummaryWriter
import torch.optim as optim
import os
from model.model import model_fn_decorator
from model.nets import my_model
from dataset.load_data import *
from tqdm import tqdm
from utils.loss_util import *
from utils.common import *
import lpips
from config.config import args,save_config
from  utils.log import save_log
def train_epoch(args, TrainImgLoader, model, model_fn, optimizer, epoch, iters, lr_scheduler):
    """
    Training Loop for each epoch
    """
    tbar = tqdm(TrainImgLoader)
    total_loss = 0
    lr = optimizer.state_dict()['param_groups'][0]['lr']
    for batch_idx, data in enumerate(tbar):

        loss = model_fn(args, data, model, iters)
        # backward and update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        iters += 1
        total_loss += loss.item()
        avg_train_loss = total_loss / (batch_idx+1)
        desc = 'Training  : Epoch %d,GPU: %s, %s lr %.7f, Avg. Loss = %.5f' % (epoch,args.GPU_ID,args.EXP_NAME,lr, avg_train_loss)
        tbar.set_description(desc)
        tbar.update()
    lr = optimizer.state_dict()['param_groups'][0]['lr']
    # the learning rate is adjusted after each epoch
    lr_scheduler.step()

    return lr, avg_train_loss, iters

def init():
    # Make dirs
    args.LOGS_DIR = os.path.join(args.SAVE_PREFIX, args.EXP_NAME, 'logs')
    args.NETS_DIR = os.path.join(args.SAVE_PREFIX, args.EXP_NAME, 'net_checkpoints')
    args.VISUALS_DIR = os.path.join(args.SAVE_PREFIX, args.EXP_NAME, 'train_visual')
    mkdir(args.LOGS_DIR)
    mkdir(args.NETS_DIR)
    mkdir(args.VISUALS_DIR)

    os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % args.GPU_ID

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # random seed
    random.seed(args.SEED)
    np.random.seed(args.SEED)
    torch.manual_seed(args.SEED)
    torch.cuda.manual_seed_all(args.SEED)
    if args.SEED == 0:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    # summary writer
    # logger = SummaryWriter(args.LOGS_DIR)
    
    # return logger, device

    return device


def main():
    device = init()
    # create model
    model = my_model(en_feature_num=args.EN_FEATURE_NUM,
                     en_inter_num=args.EN_INTER_NUM,
                     de_feature_num=args.DE_FEATURE_NUM,
                     de_inter_num=args.DE_INTER_NUM,
                     sam_number=args.SAM_NUMBER,
                     CHANNEL=args.CHANNEL
                     ).to(device)

    model._initialize_weights()

    # create optimizer
    optimizer = optim.Adam([{'params': model.parameters(), 'initial_lr': args.BASE_LR}], betas=(0.9, 0.999))
    learning_rate = args.BASE_LR
    iters = 0
    # create loss function
    loss_fn = multi_VGGPerceptualLoss(lam=args.LAM, lam_p=args.LAM_P).to(device)
    # create learning rate scheduler
    lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=args.T_0, T_mult=args.T_MULT, eta_min=args.ETA_MIN,
                                               last_epoch=args.LOAD_EPOCH - 1)
    # create training function
    model_fn = model_fn_decorator(loss_fn=loss_fn, device=device)
    # create dataset
    train_path = args.TRAIN_DATASET
    TrainImgLoader = create_dataset(args, data_path=train_path, mode='train')

    # start training
    print("****start traininig!!!****")

    best_loss =9999999
    save_path = os.path.join(args.NETS_DIR,args.CHECKPOINT_NAME)
    log_path =args.LOGS_DIR +f'/log_{args.EXP_NAME}.txt'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for epoch in range(args.LOAD_EPOCH + 1, args.EPOCHS + 1):
        learning_rate, avg_train_loss, iters = train_epoch(args, TrainImgLoader, model, model_fn, optimizer, epoch,
                                                           iters, lr_scheduler)

        savefilename = save_path+ f'/train_{epoch}.pth'
        torch.save({
            'learning_rate': learning_rate,
            'iters': iters,
            'optimizer': optimizer.state_dict(),
            'state_dict': model.state_dict()
        }, savefilename)

        # Save the latest model
        if best_loss > avg_train_loss:
            savefilename = save_path+ 'minimum_kpt.pth'
            torch.save({
                'learning_rate': learning_rate,
                'iters': iters,
                'optimizer': optimizer.state_dict(),
                'state_dict': model.state_dict()
            }, savefilename)
            best_loss = avg_train_loss
        info = f'Epoch:{epoch}    loss:{avg_train_loss}'

        save_log(log_path,info)

    save_config(args,args.NETS_DIR)

if __name__ == '__main__':
    main()
