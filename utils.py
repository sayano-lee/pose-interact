'''
    Helper Functions
    Created and collected by sayano.lee@gmail.com
    Only for academic use, please contact me if you have any issues
'''
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import functools
from torch.optim import lr_scheduler

import os

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from collections import OrderedDict


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', gpu_ids=[], is_parallel=False, init_gain=0.02):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        if is_parallel:
            net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
        else:
            net = net.to(gpu_ids[0])
    init_weights(net, init_type, init_gain=init_gain)
    return net


def visualize(x, imgformats):
    """Visualize a torch.tensor or a numpy.ndarray
    Parameters:
        x (torch.tensor or numpy.ndarray)   -- input tensor or ndarray
        imgformats (str)                    -- indicating x format
                                            -- 'nchw' 'nhwc' 'chw' 'hwc'

    Return None
    """
    assert isinstance(x, np.ndarray) or torch.is_tensor(x)
    assert len(x) == len(imgformats)

    def gallery(array, ncols=3):
        nindex, height, width, intensity = array.shape
        nrows = nindex//ncols
        assert nindex == nrows*ncols
        # want result.shape = (height*nrows, width*ncols, intensity)
        result = (array.reshape(nrows, ncols, height, width, intensity)
                  .swapaxes(1,2)
                  .reshape(height*nrows, width*ncols, intensity))
        return result

    def make_array(arr):
        """Convert into RGB images
        Parameters:
            arr (torch.tensor or numpy.ndarray)     -- input tensor or ndarray

        Return:
             image array in ndarray format
        """
        if torch.is_tensor(arr):
            arr = arr.cpu().numpy()
        if 0.0 <= arr.all() <= 255.0 :
            im = arr.astype('uint8')
        elif 0.0 <= arr.all() <= 1.0:
            im = (arr*255).astype('uint8')
        elif -1.0 <= arr.all() <= 1.0:
            im = arr*255+127.5
        assert (0.0 <= arr.all() <= 255.0) or\
               (0.0 <= arr.all() <= 1.0) or\
               (-1.0 <= arr.all() <= 1.0)
        return np.array([np.asarray(Image.open('face.png').convert('RGB'))]*12)


def visualize_arr(arr):
    """Visualize tensors in standard scale"""
    from torchvision import transforms
    im = transforms.ToPILImage()(arr)
    im.show()


def restore_image(arr, size=None):
    """Restore (-1, 1) back into (0, 1)
       if grayscale triple channel dims

       Parameters:
            arr: in NCHW format
    """
    assert torch.is_tensor(arr)
    im = (arr + 1) / 2.0
    if im.shape[1] == 1:
        im = torch.cat([im]*3, dim=1)
    if size is not None:
        return F.interpolate(im, size)
    else:
        return im


def decode_keypoints(landmark, size=None):
    """Restore 18 channels keypoints map to grey scale image"""
    landmarks = map(sum, torch.unbind(landmark))
    restored = []
    for lm in landmarks:
        # lm[lm != 0] = 255
        # lm = lm/255.0
        restored.append(torch.cat([torch.unsqueeze(lm, 0)]*3, dim=0))
    restored = torch.stack(restored)
    restored[restored>1] = 1
    if size is not None:
        return F.interpolate(restored, size)
    else:
        return restored


def decode_body_part(maps, size=None):
    maps = maps.squeeze(0)
    map = sum(maps)
    map[map>1] = 1
    map = torch.cat([torch.unsqueeze(map, 0)]*3, dim=0).unsqueeze(0)
    if size is not None:
        return F.interpolate(map, size)
    else:
        return map


def get_time_stamp_str():
    """Return time stamp in string format"""
    import time
    import datetime
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d_%H_%M_%S')
    return st


def create_dir_with_st(path):
    """Create folder with name of timestamp, return folder path
       Parameters:
           path (str)   -- root path of folder to be created
    """
    from os.path import join as opj
    from os import makedirs
    path = opj(path, get_time_stamp_str())
    makedirs(path, exist_ok=True)
    return path


def read_as_bw(fn):
    """Convert an image into black-white mask
       Parameters:
           fn (str)     -- path to image
    """
    im = Image.open(fn)
    # pixel value greater than thresh equals 255 otherwise 0

    def mask(thresh):
        return lambda x: 255 if x > thresh else 0

    r = im.convert('L').point(mask(0), mode='1')
    return r


def calculate_n_model(model):
    """Calculate # of parameters in model"""
    return sum([m.numel() for m in model.parameters()])


def concatenate_images(images):
    """Concatenate an image list and return result"""
    assert isinstance(images, (list, tuple))
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
      new_im.paste(im, (x_offset,0))
      x_offset += im.size[0]

    return new_im


def load_model(path, latest=True):
    """Load latest model from checkpoint or from specified path in pth suffix"""
    if latest:
        print('===> [*] No model specified, loading lastest...')
        latest = sorted(os.listdir(path))[-1]
        ckpt = os.path.join(path, latest)
    else:
        print('===> [*] Model specified, loading...')
        ckpt = path

    print('===> [*] Loading from {}...'.format(ckpt))
    checkpoint = torch.load(ckpt)

    return checkpoint


def convert_state_dict(state_dict):
    """Converts a state dict saved from a dataParallel module to normal
       module state_dict inplace
       :param state_dict is the loaded DataParallel model_state
    """
    if not next(iter(state_dict)).startswith("module."):
        return state_dict  # abort if dict is not a DataParallel model_state
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict


def get_latest(path):
    # newest already made before the run
    return sorted(os.listdir(path))


def get_latest_model(path):
    _path = os.path.join(path, get_latest(path)[-2], 'checkpoints')
    return os.path.join(_path, get_latest(_path)[-1])
