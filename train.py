import argparse
import os
import random
import shutil
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed

from datasets.dataset import shopeeDataset
from torchvision.models import resnet18, resnet34
from efficientnet_pytorch import EfficientNet
import geffnet
from utils.utils import get_state_dict
from tqdm import tqdm
import timm
from datasets.transforms_factory import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--network', default='efficientdet-d0', type=str,
                                    help='efficientdet-[d0, d1, ..]')
parser.add_argument('--save_folder', default='./weights/', type=str,
                                    help='Directory for saving checkpoint models')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                            help='number of data loading workers (default: 4)')
parser.add_argument('--height', default=224, type=int, metavar='N',
                            help='height of image (default: 224)')
parser.add_argument('--width', default=224, type=int, metavar='N',
                            help='width of image (default: 224)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                            help='number of total epochs to run')
parser.add_argument('--num_classes', default=42, type=int,
                            help='Number of class used in model')
parser.add_argument('-b', '--batch_size', default=256, type=int,
                            metavar='N', help='mini-batch size (default: 256), this is the total batch size of all GPUs on the current node when using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                            metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                            help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                            metavar='W', help='weight decay (default: 1e-4)',
                                                dest='weight_decay')
parser.add_argument('--resume', default=None, type=str,
                                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                            help='manual epoch number (useful on restarts)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--grad_accumulation_steps', default=1, type=int,
                    help='GPU id to use.')


def train(train_loader, model, criterion, optimizer,  epoch, args):
    # switch to train mode

    losses = []
    model.train()
    for idx, (images, target) in enumerate(train_loader):
        images = images.cuda()
        target = target.cuda()

        optimizer.zero_grad()
        # compute output
        output = model(images)
        loss = criterion(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        losses.append(loss.item())
        if idx % 200 ==0:
            print('Epoch: [{0}][{1}/{2}]\t'
                'loss {3}\t'
                'mean_loss: {4} \t'
                'Learning rate: {5}'.format(epoch, idx, len(train_loader),
                                                                    np.mean(losses),
                                                                    loss.item(),
                                                                    optimizer.param_groups[0]['lr']))
    return np.mean(losses)


def validation(valid_loader, model, criterion, args):
    model.eval()
    total = 0
    correct = 0
    losses = []
    with torch.no_grad():
        for idx, (images, target) in enumerate(valid_loader):
            images = images.cuda()
            target = target.cuda()

            output = model(images)

            loss = criterion(output, target)
            output = output.cpu()
            target = target.cpu()

            losses.append(loss.item())
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        accuracy = correct*100.0/total

    return np.mean(losses), accuracy

def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)


    train_df = pd.read_csv('./datas/train.csv')
    test_df = pd.read_csv('./datas/test.csv')
    train_df, valid_df = train_test_split(train_df, test_size=0.1, random_state=42, stratify=train_df['category'])
    args.train_df_index = train_df.index
    args.valid_df_index = valid_df.index


    train_dataset = shopeeDataset(df=train_df, height=args.height, width=args.width,  phase='train')
    valid_dataset = shopeeDataset(df=valid_df, height=args.height, width=args.width, phase='valid')
    # test_dataset = shopeeDataset(df=test_df, phase='test')

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    # test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    checkpoint = []
    if(args.resume is not None):
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
        params = checkpoint['parser']
        args.start_epoch = checkpoint['epoch'] + 1
        del params

    # model = resnet18(num_classes=args.num_classes)
    # model = EfficientNet.from_pretrained("efficientnet-b0", advprop=True, num_classes=42)
    model = None
    if args.network == 'efficientnet_b1':
        print('Load EfficientNet-b1')
        model = EfficientNet.from_pretrained('efficientnet-b1', advprop=False, num_classes=args.num_classes)
    if args.network == 'geffnet_efficientnet_b3':
        print('Load efficinetnet_b3 of geffnet')
        model = geffnet.efficientnet_b3(pretrained=True, drop_rate=0.25, drop_connect_rate=0.2)
    elif args.network == 'geffnet_efficientnet_l2_ns':
        model = geffnet.tf_efficientnet_l2_ns_475(pretrained=True, drop_rate=0.25, drop_connect_rate=0.2)
    elif args.network ==  'timm_efficientnet_b3a':
        print('Load model timm_efficientnet_b3a')
        model = timm.create_model('efficientnet_b3a', pretrained=True, num_classes=args.num_classes)
    elif args.network == 'tf_efficientnet_b4_ns':
        print('Load model tf_efficientnet_b4_ns')
        model = timm.create_model('tf_efficientnet_b4_ns', pretrained=True, num_classes=args.num_classes)
    elif args.network == 'tf_efficientnet_b7_ns':
        print('Load model tf efficientnet_b7_ns')
        model = timm.create_model('tf_efficientnet_b7_ns', pretrained=True, num_classes=args.num_classes)
    elif args.network == 'mixnet_xl':
        print('Load mixnet_xl')
        model = timm.create_model('mixnet_xl', pretrained=True, num_classes=args.num_classes)

    if(args.resume is not None):
        model.load_state_dict(checkpoint['state_dict'])
    del checkpoint
    # for idx, child in enumerate(model.children()):
    #     if idx < 4:
    #         for param in child.parameters():
    #             param.requires_grad = False
    #     else:
    #         print(idx, child)


    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(
                (args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu], find_unused_parameters=True)
            print('Run with DistributedDataParallel with divice_ids....')
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
            print('Run with DistributedDataParallel without device_ids....')
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = model.cuda()
        print('Run with DataParallel ....')
        model = torch.nn.DataParallel(model).cuda()
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.AdamW(model.parameters(), args.lr,
                                weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, verbose=True)
    cudnn.benchmark = True

    print('Done main workers')

    for epoch in range(args.start_epoch, args.epochs):
        train_loss = train(train_loader, model, criterion, optimizer, epoch, args)
        print('Start validation ...')
        val_loss, acc= validation(valid_loader, model, criterion, args)
        print('Epoch: {}, train_loss: {}, val_loss: {}, val_acc: {} %'.format(epoch, train_loss, val_loss, acc))
        scheduler.step(acc)
        state = {
                'epoch': epoch,
                'parser': args,
                'state_dict': get_state_dict(model)
            }
        torch.save(
            state,
            os.path.join(
                args.save_folder,
                args.network,
                "{}_{}.pth".format(args.network, epoch)))




def main():
    args = parser.parse_args()
    if(not os.path.exists(os.path.join(args.save_folder, args.network))):
        os.makedirs(os.path.join(args.save_folder, args.network))

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


if __name__=='__main__':
    main()



