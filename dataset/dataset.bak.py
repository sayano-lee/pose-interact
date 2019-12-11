import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import numpy as np

from os.path import join as opj
from os.path import basename as opb

from glob import glob

from scipy import ndimage

import re

from PIL import Image

class CustomDset(Dataset):
    def __init__(self, folders, frames, transform, seed=None):
        super(CustomDset, self).__init__()
        self.persons = ['A', 'B']
        self.frames = frames
        self.folders = folders
        self.transform = transform
        self.walker = []

        self.suffix = '.npz'

        np.random.seed(seed)

        self.gen_walker()

    def gen_walker(self):

        self.walker = []

        for folder in self.folders:
            self.walker.extend(self._make_sequence(folder))

        np.random.shuffle(self.walker)

    def _make_sequence(self, folder):

        fns_A = glob(opj(folder, self.persons[0], '*'+self.suffix))

        groups = self._devide_by_frames(fns_A)
        cls = self._make_classes(folder)

        for grp in groups:
            grp.append(cls)

        return groups

    def _devide_by_frames(self, fns):

        sampler = self._gen_sampler(len(fns))
        sequence = self._gen_sequence(sampler)
        npy_fns = self._pack_dict(sequence, fns)

        return npy_fns

    def _pack_dict(self, sequence, fns):
        prefix = '/'.join(fns[0].split('/')[:-1])
        fn = []
        for seq in sequence:
            fn.append([opj(prefix, 'im_'+'{:03d}{}'.format(idx, self.suffix)) for idx in seq])
        return fn

    def _gen_sequence(self, sampler):
        loop = []
        for sa in sampler:
            loop.append(list(np.random.permutation(range(sa[0], sa[1]+1))))
        return list(zip(*loop))

    def _gen_sampler(self, num_files):
        inter = []
        block_size = num_files // self.frames
        for cnt in range(self.frames-1):
            inter.append([cnt*block_size+1, (cnt+1)*block_size])
        inter.append([(cnt+1)*block_size+1, num_files])
        return inter

    def _make_classes(self, folder):
        return opb(folder).split('_')[-1]

    def __getitem__(self, index):
        sample = self.walker[index]

        cls = int(sample[-1])
        fns = sample[:-1]

        ims_A, ims_B = self._read_ims(fns)
        import ipdb; ipdb.set_trace()
        feats_A, feats_B = self._read_npy(fns)
        import ipdb; ipdb.set_trace()

        return torch.from_numpy(feats_A), torch.from_numpy(feats_B), torch.LongTensor([cls])
    
    def _read_ims(self, fns):
        fns_A = self.__reg_names(fns)
        ims_A = self._read_ims_person(fns_A)

        fns_B = self._read_names_B(fns_A)
        import ipdb; ipdb.set_trace()
        return ims_A
    
    def _read_names_B(self, fns):
        fns_B = [fn.replace('A', 'B') for fn in fns]
        return fns_B

    def _read_ims_person(self, fns):
        # fns = self.__reg_names(fns)
        try:
            ims = [self.transform(Image.open(fn)) for fn in fns]
        except FileNotFoundError:
            pass
        return torch.stack(ims)
    
    def __reg_names(self, fns):
        pattern1 = r'(\d+)_(\d+)+_(\d+)'
        pattern2 = r'im_(\d+)'
        new_fns = []
        for fn in fns:
            groups1 = re.search(pattern1, fn)
            groups2 = re.search(pattern2, fn)
            # fn = fn.replace() 
            # fn = fn.replace(groups1[0], '{}_{}_{}'.format(int(groups1[1]), int(groups1[2]), int(groups1[3])))
            # fn = fn.replace(groups2[0], 'im_{}'.format(int(groups2[1])))
            for k, v in {groups1[0]: '{}_{}_{}'.format(int(groups1[1]), int(groups1[2]), int(groups1[3])),
                         groups2[0]: 'im_{}'.format(int(groups2[1])),
                         'npy': 'raw',
                         self.suffix: '.jpg'}.items():
                fn = fn.replace(k, v)
            new_fns.append(fn)
        return new_fns

    def _read_npy(self, fns):
        feats_A = []
        feats_B = []
        for fn_A in fns:
            fn_B = fn_A.replace('A', 'B')
            feat_A = np.load(fn_A)['arr_0']
            try:
                feat_B = np.load(fn_B)['arr_0']
                feat_B = self._get_max_joint_stacks(feat_B)
            except:
                # Person B does not exist, zero mask B
                feat_B = np.zeros_like(feat_A)

            feat_A = self._get_max_joint_stacks(feat_A)

            feats_A.append(np.concatenate(feat_A, axis=0))
            feats_B.append(np.concatenate(feat_B, axis=0))

        return np.stack(feats_A, axis=0), np.stack(feats_B, axis=0)
    
    def _get_max_joint_stacks(self, arr):
        max_arr_stack = []
        for ar in arr:
            max_arr_stack.append(self._get_max_joint_stack(ar)[np.newaxis,:])
        return np.concatenate(max_arr_stack, axis=0)
    
    def _get_max_joint_stack(self, arr):
        max_arr_channel = []
        for ar in arr:
            max_arr_channel.append(self._get_max_joint_channel(ar)[np.newaxis,:])
        
        return np.concatenate(max_arr_channel, axis=0)

    def _get_max_joint_channel(self, arr):
        joint_map = torch.zeros(arr.flatten().shape)
        joint_map[np.argmax(arr)] = 1.0
        joint_map = joint_map.reshape(arr.shape)

        joint_map = ndimage.filters.gaussian_filter(joint_map, sigma=2)
        return joint_map/joint_map.max()

    def __len__(self):
        return len(self.walker)

    def shuffle(self):
        np.random.shuffle(self.walker)
    
    # def _shuffle_train_val(self):
    #     self.actor_groups = np.random.permutation(self.actor_groups)
    #     self.tolerance = 0


def split_train_val(root, dataset, val_grp, frames):
    folders = glob(opj(root, dataset, '*'))

    """
    for folder in folders:
        grp_ix = int(folder.split('/')[-1].split('_')[1])
        if grp_ix == val_grp:
            vals.append(folder)
        else:
            trains.append(folder)

    assert len(trains) + len(vals) == len(folders)
    # import ipdb; ipdb.set_trace()
    """
    return CustomDset(folders)
    # return CustomDset(trains, frames), CustomDset(vals, frames)
    


if __name__ == '__main__':
    # datasets = ['patchset1', 'patchset2']
    # dset = CustomDset(datasets[0])
    # dloader = DataLoader(dset, batch_size=4, shuffle=False)

    dset = split_train_val('./data/npy', 'patchset1', 3, 8)
    dloader = DataLoader(dset, batch_size=128)
    
    # train_loader = DataLoader(train_set)
    # val_loader = DataLoader(validation_set)

    for cnt, data in enumerate(dloader):
        pass
        # import ipdb
        # ipdb.set_trace()

    import ipdb; ipdb.set_trace()
