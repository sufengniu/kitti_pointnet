import time
import os
import argparse

import numpy as np

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as LS
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as data
from torchvision import transforms

parser = argparse.ArgumentParser()
parser.add_argument(
    '--batch-size', '-N', type=int, default=1, help='batch size')
parser.add_argument(
    '--evaluation', '-f', required=True, type=str, help='folder of evaluation images')
parser.add_argument(
    '--max-epochs', '-e', type=int, default=200, help='max epochs')
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
parser.add_argument('--cuda', '-g', action='store_true', help='enables cuda')
parser.add_argument(
    '--iterations', type=int, default=16, help='unroll iterations')
parser.add_argument('--checkpoint', type=int, help='unroll iterations')
args = parser.parse_args()


from global_level_representation.my_dataset import CompressDataset

evaluation_set = CompressDataset(filename=args.evaluation)

evaluation_loader = data.DataLoader(
    dataset=evaluation_set, batch_size=args.batch_size, shuffle=True, num_workers=1)

print('total images: {}; total batches: {}'.format(
    len(evaluation_set), len(evaluation_loader)))

## load networks on GPU
import global_level_representation.network

encoder = network.EncoderCell().cuda()
binarizer = network.Binarizer().cuda()
decoder = network.DecoderCell().cuda()

print("encoder: {}".format(encoder))
print("binarizer: {}".format(binarizer))
print("decoder: {}".format(decoder))


losses = None

for batch, data in enumerate(evaluation_loader):
    data = data.float()
    batch_t0 = time.time()

    ## init lstm state
    encoder_h_1 = (Variable(torch.zeros(data.size(0), 256, 8, 8).cuda()),
                   Variable(torch.zeros(data.size(0), 256, 8, 8).cuda()))
    encoder_h_2 = (Variable(torch.zeros(data.size(0), 512, 4, 4).cuda()),
                   Variable(torch.zeros(data.size(0), 512, 4, 4).cuda()))
    encoder_h_3 = (Variable(torch.zeros(data.size(0), 512, 2, 2).cuda()),
                   Variable(torch.zeros(data.size(0), 512, 2, 2).cuda()))

    decoder_h_1 = (Variable(torch.zeros(data.size(0), 512, 2, 2).cuda()),
                   Variable(torch.zeros(data.size(0), 512, 2, 2).cuda()))
    decoder_h_2 = (Variable(torch.zeros(data.size(0), 512, 4, 4).cuda()),
                   Variable(torch.zeros(data.size(0), 512, 4, 4).cuda()))
    decoder_h_3 = (Variable(torch.zeros(data.size(0), 256, 8, 8).cuda()),
                   Variable(torch.zeros(data.size(0), 256, 8, 8).cuda()))
    decoder_h_4 = (Variable(torch.zeros(data.size(0), 128, 16, 16).cuda()),
                   Variable(torch.zeros(data.size(0), 128, 16, 16).cuda()))

    patches = Variable(data.cuda())


    batch_losses = []
    res = res0 = patches - 0.5

    for _ in range(args.iterations):
        encoded, encoder_h_1, encoder_h_2, encoder_h_3 = encoder(
            res, encoder_h_1, encoder_h_2, encoder_h_3)

        codes = binarizer(encoded)

        output, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4 = decoder(
            codes, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4)

        res = res - output
        loss = (res / res0).abs().mean().cpu().detach().numpy()
        batch_losses.append(np.reshape(loss, (1,1)))

    batch_losses = np.concatenate(batch_losses)

    if losses is None:
        losses = batch_losses
    else:
        losses += batch_losses

losses = losses / len(evaluation_set)
print("losses: {}".format(losses))