import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

import numpy as np

from dataset.dataset_spliter import dset_gen
from networks import IntModel

import os
from os.path import join as opj
from os.path import basename as opb

from utils import create_dir_with_st

from visualize_heatmaps import batch_save

from evaluate import evaluate

from tqdm import tqdm

from logger import make_logger

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--frames', type=int, default=8, help='number of frames in one sequence employed for training')
parser.add_argument('--batch_size', '--bs', type=int, default=32, help='batch size for training, each mini-batch contains a frame-long sequence')
parser.add_argument('--val_batch', type=int, default=1, help='batch size for validation, each mini-batch contains a frame-long sequence')
parser.add_argument('--n_workers', type=int, default=8, help='number of workers for DataLoader')

parser.add_argument('--root', type=str, default='./data/npy', help='pre extracted npy from raw images by hourglass 2 stack')
parser.add_argument('--dataset', type=str, default='patchset1', help='can be either patchset1 or patchset2')

parser.add_argument('--max_epoch', type=int, default=200, help='maximum training epoch')
parser.add_argument('--permutation_epoch', type=int, default=10, help='re-split train and validation')
parser.add_argument('--permutation_seed', type=int, default=None, help='random seed for splitting training and validating')

parser.add_argument('--resume', action='store_true', default=False, help='if resume from latest checkpoint')
parser.add_argument('--gpu_id', type=int, default=0, help='indicating which gpu to utilize')

parser.add_argument('--log_root', type=str, default='./run', help='path to save checkpoints and tensorboard')
parser.add_argument('--save_every', type=int, default=1, help='save and test model every x epoch(s)')
parser.add_argument('--log_every', type=int, default=1, help='save tensorboard every x iteration(s)')

args = parser.parse_args()

cuda = torch.cuda.is_available()
device = torch.device('cuda:{}'.format(args.gpu_id)) if cuda else torch.device('cpu')


"""
def validation_permutation(grp_range):
    assert isinstance(grp_range, (list, tuple))
    return np.random.RandomState(seed=args.permutation_seed).permutation(list(range(grp_range[0], grp_range[1])))

def permutate_UT_dataset(patchset):
    if patchset == 'patchset1':
        grps = [1, 11]
    elif patchset == 'patchset2':
        grps = [11, 20]
    else:
        raise NotImplementedError('Wrong patchset type')

    grp_order = validation_permutation(grps)
    return grp_order

def get_data_set(val_idx):
    # actor_groups = permutate_UT_dataset(args.dataset)

    train_set, validation_set = split_train_val(root=args.root, dataset=args.dataset,
                                                val_grp=val_idx, frames=args.frames)
    return train_set, validation_set
"""


def train(**kwargs):

    logger = kwargs['logger']
    log_path = kwargs['log_path']

    start_epoch = 1
    counting_iter = 0
    best_acc = 0

    trans = transforms.Compose([transforms.Resize((256, 256)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5],[0.5])])

    dset_constructor = dset_gen(opj(args.root, args.dataset), args.frames, trans)
    train_set, validation_set = dset_constructor.loop()
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.n_workers, drop_last=False)
    validation_loader = DataLoader(validation_set, batch_size=args.val_batch, shuffle=False, pin_memory=True, num_workers=args.n_workers, drop_last=False)

    criterion = nn.CrossEntropyLoss().to(device)
    net = IntModel().to(device)
    optimizer = torch.optim.Adam(params=net.parameters(), lr=1e-4,
                                  weight_decay=1e-5)

    epoch_accs = {}
    for epoch in range(start_epoch, args.max_epoch):

        net.train()
        average_loss = 0

        with tqdm(total=len(train_loader), leave=False, desc='Running Epoch {}/{}'.format(epoch, args.max_epoch)) as pbar:
            for cnt, (x1, x2, _, _, y) in enumerate(train_loader):
                # INPUT has a shape (bs, depth, channel, width, height)
                pbar.update(1)
                # batch_save(x1, './visual_inputs')
                # import ipdb; ipdb.set_trace()

                counting_iter += args.batch_size

                x1 = x1.to(device)
                x2 = x2.to(device)
                y = y.to(device)

                prob = net(x1, x2)

                optimizer.zero_grad()

                loss = criterion(prob, y.squeeze(1))
                loss.backward()

                optimizer.step()

                average_loss += loss.item()

                # print('=====> Epoch [{}/{}] Iter [{}/{}] Loss {:.4f}, Validation Group is {}'
                    #   .format(epoch, args.max_epoch, cnt+1, len(train_loader), loss, dset_constructor.current_val))

                # if cnt % args.log_every == 0:
                    # writer.add_scalar('TrainingLoss', loss.item(), counting_iter)
        logger.info('Training Loss {:4f} for Epoch [{}/{}]'.format(average_loss/len(train_loader), epoch, args.max_epoch))
        
        # 10-fold leave-one-out validation
        # if RESET triggered, reset group counter and average over all groups
        # else update scores
        with torch.no_grad():
            net.eval()
            acc = evaluate(net, validation_loader, device)
        epoch_accs.update({dset_constructor.current_val: acc})

        if dset_constructor.check_reset_status():
            dset_constructor.reset_counter()
            try:
                assert len(epoch_accs) == len(dset_constructor.grps)
            except:
                import ipdb; ipdb.set_trace()
            avg_acc = 0
            for _, v in epoch_accs.items():
                avg_acc += v
            avg_acc /= len(epoch_accs)

            # writer.add_scalar('Accuracy', avg_acc, epoch)
            logger.info('Accuracy {:4f} for Epoch [{}/{}]'.format(avg_acc, epoch, args.max_epoch))
            # print('[ * ] =====> Evaluation Done. Accuracy {:.4f}'.format(avg_acc))

            if avg_acc > best_acc:
                best_acc = avg_acc
                best_dict = dict({
                    'epoch': epoch,
                    'best': best_acc,
                    'state_dict': net.state_dict(),
                })
                torch.save(best_dict, opj(log_path, 'checkpoints', 'best.pth'))
            else:
                saving_dict = dict({
                    'epoch': epoch,
                    'avg_acc': avg_acc,
                    'state_dict': net.state_dict(),
                })
                torch.save(saving_dict, opj(log_path, 'checkpoints', 'epoch_{:03d}.pth'.format(epoch)))

        train_set, validation_set = dset_constructor.loop()
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False,
                                  pin_memory=True, num_workers=args.n_workers, drop_last=False)
        validation_loader = DataLoader(validation_set, batch_size=args.val_batch, shuffle=False,
                                  pin_memory=True, num_workers=args.n_workers, drop_last=False)

if __name__ == '__main__':

    log_path = create_dir_with_st(args.log_root)

    checkpoint_path = opj(log_path, 'checkpoints')
    os.makedirs(checkpoint_path, exist_ok=True)

    logger_path = opj(log_path, 'log')
    logger = make_logger('Action recognition', log_path, 'log')

    train(logger=logger, log_path=log_path)
