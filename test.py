import argparse
import numpy as np
import pandas as pd
import os
import torch
from datasets import shopeeDataset
from efficientnet_pytorch import EfficientNet
import geffnet
from tqdm import tqdm
import timm
from datasets.transforms_factory import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--network', default='sub', type=str)
parser.add_argument('--resume', default=None, type=str, help='Checkpoint state_dict file to resume training from')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                            help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch_size', default=256, type=int,
                            metavar='N', help='mini-batch size (default: 256), this is the total batch size of all GPUs on the current node when using Data Parallel or Distributed Data Parallel')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--width', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--height', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--name', default=None, type=str,
                    help='GPU id to use.')
def test(test_loader, model):
    preds = []
    probas = []
    model.eval()
    with torch.no_grad():
        for idx, images in enumerate(tqdm(test_loader)):
            images = images.cuda()
            output = model(images)
            output = output.cpu()
            probas.append(output.numpy())
            _, predicted = torch.max(output.data, 1)
            preds.append(predicted)

    return np.hstack(preds), np.vstack(probas)


if __name__=='__main__':
    args = parser.parse_args()
    test_df = pd.read_csv('./datas/test.csv')

    # test_dataset = shopeeDataset(df=train_df, phase='train')
    test_dataset = shopeeDataset(df=test_df, phase='test', width=args.width, height=args.height)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)


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
            args.num_classes = params.num_classes
            del params

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

    if(args.resume is not None):
        print('Load state_dict')
        model.load_state_dict(checkpoint['state_dict'])
    model = model.cuda()
    preds, probas = test(test_loader, model)
    print('probas: ', probas.shape)
    # proba_df = pd.DataFrame(probas, columns=range(42))
    submit = pd.read_csv('./datas/test.csv')
    # submit[proba_df.columns] = proba_df
    submit['category'] = submit['category'].astype(str)
    zeropad = lambda x: '0' + str(x) if len(str(x))==1 else str(x)
    preds = [zeropad(p) for p in preds]
    submit['category'] = preds
    submit['category'] = submit['category'].apply(lambda x: '0'+str(x) if len(str(x))==1 else str(x))
    submit.to_csv('./submit/{}_{}.csv'.format(args.network, args.name), index=False)



