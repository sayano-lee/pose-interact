import numpy as np
from glob import glob

import os
from os.path import join as opj
from os.path import basename as opb

from tqdm import tqdm
import re

def proc_folder(folder_name):
    persons = glob(opj(folder_name, '*'))
    for person in persons:
        print('Processing {} ...'.format(person))
        match = re.search(r'(\d+)_(\d+)_(\d+)', person)
        try:
            compressed_person = person.replace(match.group(0), '{:02d}_{:02d}_{:02d}'.format(int(match.group(1)),
                                                                                         int(match.group(2)),
                                                                                         int(match.group(3))))\
                                  .replace('npy', 'npy_compressed')
        except:
            import ipdb; ipdb.set_trace()
        os.makedirs(compressed_person, exist_ok=True)
        npys = sorted(glob(opj(person, '*.npy')))
        with tqdm(total=len(npys), leave=False) as pbar:
            for cnt, npy in enumerate(npys):
                pbar.update(1)

                base_no = int(opb(npy).strip('.npy').split('_')[-1])
                # try:
                    # assert base_no == cnt + 1
                # except AssertionError:
                    # import ipdb; ipdb.set_trace()
                data = np.load(npy)
                # np.save(opj(compressed_person, 'normal_person.npy'), data)
                np.savez_compressed(opj(compressed_person, 'im_{:03d}.npz'.format(base_no)), data)

                # import ipdb; ipdb.set_trace()

if __name__ == "__main__":
    patchset1 = glob(opj('./data/npy/patchset1', '*'))
    patchset2 = glob(opj('./data/npy/patchset2', '*'))
    for set1_folder in patchset1:
        proc_folder(set1_folder)
    for set2_folder in patchset2:
        proc_folder(set2_folder)
    import ipdb; ipdb.set_trace()