from matplotlib import pyplot as plt

import torch
from torchvision.utils import save_image

import os
from os.path import join as opj

import numpy as np

from scipy import ndimage

def normalize(arr):
    arr = arr - arr.min()
    arr = arr / arr.max()
    return arr

def max_joint(arr):
    assert isinstance(arr, (torch.Tensor, np.ndarray))
    try:
        arr = arr.cpu().numpy()
    except:
        pass

    joint_map = torch.zeros(arr.flatten().shape)
    joint_map[np.argmax(arr)] = 1.0
    joint_map = joint_map.reshape(arr.shape)

    joint_map = ndimage.filters.gaussian_filter(joint_map, sigma=2)

    return joint_map/joint_map.max()

def batch_save(arr, folder):
    # sanity check
    assert isinstance(arr, (np.ndarray, torch.Tensor))
    try:
        arr = arr.numpy()
    except:
        pass
    assert arr.ndim == 5
    
    # arr = np.transpose(arr, ())
    bs_stack1 = arr[0][0][:16]
    bs_stack2 = arr[0][0][16:]
    os.makedirs(folder, exist_ok=True)

    stack1_assemble = []
    for cnt, mat in enumerate(bs_stack1):
        joint_heatmap = max_joint(mat)
        stack1_assemble.append(joint_heatmap)
        plt.imshow(joint_heatmap)
        plt.savefig(opj(folder,'stack_1_joint_{}.jpg'.format(cnt)))
        plt.cla();plt.close()

    stack2_assemble = []
    for cnt2, mat2 in enumerate(bs_stack2):
        joint_heatmap = max_joint(mat2)
        stack2_assemble.append(joint_heatmap)
        plt.imshow(joint_heatmap)
        plt.savefig(opj(folder,'stack_2_joint_{}.jpg'.format(cnt2)))
        plt.cla();plt.close()

    joint_heatmap_stack1 = sum(np.stack(stack1_assemble, axis=0))
    plt.imshow(joint_heatmap_stack1)
    plt.savefig(opj(folder,'stack_1.jpg'))
    plt.cla();plt.close()

    joint_heatmap_stack2 = sum(np.stack(stack2_assemble, axis=0))
    plt.imshow(joint_heatmap_stack2)
    plt.savefig(opj(folder,'stack_2.jpg'))
    plt.cla();plt.close()

    import ipdb
    ipdb.set_trace()
