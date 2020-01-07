import math
import torch
import torch_geometric.transforms as T


class RandomTranslate(object):
    """
    Translation
    """

    def __init__(self, translate):
        self.translate = translate

    def __call__(self, data):

        t = 2*(torch.rand(3)-0.5)*self.translate

        data.pos = data.pos + t
        return data

    def __repr__(self):
        return "Random Translate of translation {}".format(self.translate)


def get_data_augmentation_transform(name, value):
    r"""
    return a callable object that perform data augmentation transform

    """

    if(name == 'jitter'):
        return T.RandomTranslate(value)
    elif(name == 'translate'):
        return RandomTranslate(value)
    elif(name == 'rotate_x'):
        return T.RandomRotate(value, axis=0)
    elif(name == 'rotate_y'):
        return T.RandomRotate(value, axis=1)
    elif(name == 'rotate_z'):
        return T.RandomRotate(value, axis=2)
    elif(name == 'scale'):
        return T.RandomScale(value)
    elif(name == 'flip_x'):
        return T.RandomFlip(axis=0, p=value)
    elif(name == 'flip_y'):
        return T.RandomFlip(axis=1, p=value)
    elif(name == 'flip_z'):
        return T.RandomFlip(axis=2, p=value)
    else:
        raise NotImplementedError
