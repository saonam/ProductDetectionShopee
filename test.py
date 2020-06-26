import argparse
import numpy as np
import pandas as pd
import os
import torch
from efficientnet_pytorch import EfficientNet



parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--resume', default=None, type=str,
                                    help='Checkpoint state_dict file to resume training from')
def test(test_loader, model):
    preds = []
    model.eval()
    with torch.no_grad():
        for idx, images in enumerate(test_loader):
            images = images.cuda()
            output = model(images)
            output = output.cpu()

            _, predicted = torch.max(output.data, 1)
            preds.append(predicted)

    return np.hstack(preds)


if __name__=='__main__':
    args = parser.parse_args()

    test_dataset = shopeeDataset(df=test_df, phase='test')

    test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)


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
        args = params
        del params

    model = EfficientNet.from_pretrained("efficientnet-b0", advprop=True, num_classes=42)
    if(args.resume is not None):
        model.load_state_dict(checkpoint['state_dict'])
    model = model.cuda()
    preds = test(test_loader, model)
    submit = pd.read_csv('./datas/test.csv')
    submit['category'] = preds
    submit.to_csv('./weights/sub.csv', index=False)



