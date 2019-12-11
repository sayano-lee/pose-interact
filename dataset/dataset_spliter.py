import numpy as np

try:
    from .dataset import CustomDset
except:
    from dataset import CustomDset
from os.path import basename as opb
from os.path import join as opj

from glob import glob

class dset_gen():
    def __init__(self, folder, frames, transform, seed=None):
        super(dset_gen, self).__init__()
    
        self.folders = sorted(glob(opj(folder, '*')))
        self.frames = frames
        self.transform = transform

        # for 10-folder leave-one-out validation
        self.switch_counter = 0
        self.grps = set()
        self._collect_grps()

        self.folder_dict = {}
        self._proc_folders_by_grps()

        self.shuffle()
    
    def __call__(self):
        raise NotImplementedError

    def _collect_grps(self):
        for folder in self.folders:
            self.grps.add(opb(folder).split('_')[1])
        self.grps = sorted(self.grps)
    
    def _proc_folders_by_grps(self):
        for f in self.folders:
            grp_num = opb(f).split('_')[1]
            if grp_num in list(self.folder_dict.keys()):
                self.folder_dict[grp_num].append(f)
            else:
                self.folder_dict.update({grp_num:[f]})

    def re_split(self):
        train_folder = []
        for k, v in self.folder_dict.items():
            if k != self.grps[self.switch_counter]:
                train_folder.extend(v)
        val_folder = self.folder_dict[self.grps[self.switch_counter]]
        self._update_counter()

        return train_folder, val_folder
    
    def construct_dset(self, train_folder, val_folder):
        return CustomDset(train_folder, self.frames, self.transform),\
            CustomDset(val_folder, self.frames, self.transform)
    
    def loop(self):
        tf, vf = self.re_split()
        ts, vs = self.construct_dset(tf, vf)
        return ts, vs
    
    def check_reset_status(self):
        return False if self.switch_counter < len(self.grps) else True
    
    # NOTE: must not be called before self.re_split
    @property
    def current_val(self):
        return self.grps[self.switch_counter-1]

    def _update_counter(self):
        self.switch_counter += 1
    
    def reset_counter(self):
        self.switch_counter = 0
    
    def shuffle(self):
        np.random.shuffle(self.grps)


if __name__ == "__main__":
    import time
    from torch.utils.data import DataLoader
    from torchvision import transforms
    trans = transforms.Compose([transforms.Resize((128, 256)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5],[0.5])])
    dset = dset_gen('../data/npy/patchset1', 10, trans, 2501)
    while(True):
        train_set, val_set = dset.loop()
        train_loader = DataLoader(train_set, batch_size=128)
        val_loader = DataLoader(val_set, batch_size=128)
        # print('Switch counter {}, Current validation Group {}'.format(dset.switch_counter,
        # dset.current_val))
        # time.sleep(1)
        for cnt, data in enumerate(train_loader):
            pass
        if dset.check_reset_status():
            dset.reset_counter()
            dset.shuffle()