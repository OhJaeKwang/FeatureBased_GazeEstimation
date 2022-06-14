# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Tianheng Cheng(tianhengcheng@gmail.com)
# ------------------------------------------------------------------------------

from .aflw import AFLW
from .cofw import COFW
from .face300w import Face300W
from .wflw import WFLW
from .unityeyes import unityeyes
from .unityeyes_gray import unityeyes_gray

__all__ = ['AFLW', 'COFW', 'Face300W', 'WFLW', 'unityeyes', 'get_dataset' ,'unityeyes_gray']


def get_dataset(config):

    if config.DATASET.DATASET == 'AFLW':
        return AFLW
    elif config.DATASET.DATASET == 'COFW':
        return COFW
    elif config.DATASET.DATASET == '300W':
        return Face300W
    elif config.DATASET.DATASET == 'WFLW':
        return WFLW
    elif config.DATASET.DATASET == 'unityeyes':
        return unityeyes
    elif config.DATASET.DATASET == 'unityeyes_gray':
        return unityeyes_gray
    else:
        raise NotImplemented()
