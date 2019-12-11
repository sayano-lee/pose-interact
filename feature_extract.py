import torch
from torchvision import transforms

from models import hg
from utils import *

import os
from os.path import join as opj
from os.path import basename as opb
from glob import glob
from PIL import Image

import numpy as np

import argparse

from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument('--model_path', type=str, default='./pretrained/model_best.pth.tar')
parser.add_argument('--raw_data', type=str, default='./data/raw')
parser.add_argument('--npy_data', type=str, default='npy_normalize')
parser.add_argument('--persons', type=list, default=['A', 'B'])
parser.add_argument('--input_height', type=int, default=256)
parser.add_argument('--input_width', type=int, default=256)

opt = parser.parse_args()

transforms1=transforms.Compose([transforms.Resize((opt.input_height, opt.input_width)),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=[0.5], std=[0.5])])

class walker():
    def __init__(self, model):
        super(walker, self).__init__()

        patches = glob(opj(opt.raw_data, '*'))
        fns_1 = self.patch_proc(sorted(patches)[0])
        fns_2 = self.patch_proc(sorted(patches)[1])

        self._make_npy_dirs(fns_1)
        self._make_npy_dirs(fns_2)

        # mean1, std1 = self.calculate_std_mean(fns_1)
        # mean2, std2 = self.calculate_std_mean(fns_2)


        patchset1_mean, patchset1_std = [tuple(np.load('./data/patchset1_meanstd.npy')[i]) for i in range(2)]
        patchset2_mean, patchset2_std = [tuple(np.load('./data/patchset2_meanstd.npy')[j]) for j in range(2)]



        self.model = model

        self.extract(transform=transforms.Compose([transforms.Resize((opt.input_height, opt.input_width)),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize(mean=[0.5], std=[0.5])]),
                     fns=fns_1)
        self.extract(transform=transforms.Compose([transforms.Resize((opt.input_height, opt.input_width)),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize(mean=[0.5], std=[0.5])]),
                     fns=fns_2)

    def _make_npy_dirs(self, fns):
        normed_fns = set('/'.join(fn.split('/')[:-1]).replace('raw', opt.npy_data) for fn in fns)

        for normed_fn in normed_fns:
            os.makedirs(normed_fn, exist_ok=True)

    def person_proc(self, path, person):
        person_path = opj(path, person)
        fns = glob(opj(person_path, '*.jpg'))
        return fns

    def group_proc(self, path):
        person_fns = []
        for person in opt.persons:
            person_fn = self.person_proc(path, person)
            person_fns.extend(person_fn)
        return person_fns

    def patch_proc(self, patch):
        group_fns = []
        subfolders = glob(opj(patch, '*'))
        for sub in subfolders:
            group_fn = self.group_proc(sub)
            group_fns.extend(group_fn)
        return group_fns

    def _im2tensor(self, im):
        return torch.FloatTensor(np.array(im)) / 255.0

    def calculate_std_mean(self, fns):
        print('=====> Calculating mean and std...')
        mean = 0
        std = 0
        for cnt, fn in enumerate(fns):
            im = self._im2tensor(Image.open(fn))
            mean += torch.mean(im, (0, 1))
            std += torch.std(im, (0, 1))
        return mean/len(fns), std/len(fns)

    def extract(self, transform, fns):
        # pbar = tqdm(total=len(fns))
        for cnt, fn in enumerate(fns):
            print('=====> [{}/{}] Extracting from {}'.format(cnt+1, len(fns), fn))

            im = Image.open(fn)

            input = transform(im).unsqueeze(0).cuda()
            feat = self.model(input)
            np.save(fn.replace('raw', opt.npy_data).replace('.jpg', '.npy'), torch.cat(feat).cpu().numpy())


if __name__ == '__main__':
    model = hg(num_stacks=2, num_blocks=1, num_classes=16)
    ckpt = torch.load(opt.model_path)
    model.load_state_dict(convert_state_dict(ckpt['state_dict']))
    model.eval().cuda()

    with torch.no_grad():
        walker(model)
