"""

Pytorch implementation of Pointer Network.

http://arxiv.org/pdf/1506.03134v1.pdf.

"""

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader

import numpy as np
import argparse
from tqdm import tqdm

from PointerNet import PointerNet
from Data_Generator import TSPDataset

parser = argparse.ArgumentParser(description="Pytorch implementation of Pointer-Net")

# Data
parser.add_argument('--train_size', default=1000000, type=int, help='Training data size')
parser.add_argument('--val_size', default=10000, type=int, help='Validation data size')
parser.add_argument('--test_size', default=10000, type=int, help='Test data size')
parser.add_argument('--batch_size', default=256, type=int, help='Batch size')
# Train
parser.add_argument('--nof_epoch', default=50000, type=int, help='Number of epochs')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
# GPU
parser.add_argument('--gpu', default=True, action='store_true', help='Enable gpu')
# TSP
parser.add_argument('--nof_points', type=int, default=5, help='Number of points in TSP')
# Network
parser.add_argument('--embedding_size', type=int, default=128, help='Embedding size')
parser.add_argument('--hiddens', type=int, default=512, help='Number of hidden units')
parser.add_argument('--nof_lstms', type=int, default=2, help='Number of LSTM layers')
parser.add_argument('--dropout', type=float, default=0., help='Dropout value')
parser.add_argument('--bidir', default=True, action='store_true', help='Bidirectional')

params = parser.parse_args()

if params.gpu and torch.cuda.is_available():
    USE_CUDA = True
    print('Using GPU, %i devices.' % torch.cuda.device_count())
else:
    USE_CUDA = False

model = PointerNet(params.embedding_size,
                   params.hiddens,
                   params.nof_lstms,
                   params.dropout,
                   params.bidir)

dataset = TSPDataset(params.train_size,
                     params.nof_points)

dataloader = DataLoader(dataset,
                        batch_size=params.batch_size,
                        shuffle=True,
                        num_workers=4)

if USE_CUDA:
    model.cuda()
    net = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

CCE = torch.nn.CrossEntropyLoss()
model_optim = optim.Adam(filter(lambda p: p.requires_grad,
                                model.parameters()),
                         lr=params.lr)

losses = []

for epoch in range(params.nof_epoch):
    batch_loss = []
    iterator = tqdm(dataloader, unit='Batch')

    for i_batch, sample_batched in enumerate(iterator):
        iterator.set_description('Batch %i/%i' % (epoch+1, params.nof_epoch))

        train_batch = Variable(sample_batched['Points'])
        target_batch = Variable(sample_batched['Solution'])

        if USE_CUDA:
            train_batch = train_batch.cuda()
            target_batch = target_batch.cuda()

        o, p = model(train_batch)
        o = o.contiguous().view(-1, o.size()[-1])

        target_batch = target_batch.view(-1)

        loss = CCE(o, target_batch)

        losses.append(loss.data[0])
        batch_loss.append(loss.data[0])

        model_optim.zero_grad()
        loss.backward()
        model_optim.step()

        iterator.set_postfix(loss='{}'.format(loss.data[0]))

    iterator.set_postfix(loss=np.average(batch_loss))
