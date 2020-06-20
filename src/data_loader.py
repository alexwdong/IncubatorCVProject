import numpy as np
import os
import cv2
import re
import random
from PIL import Image
from torchvision.transforms import Pad


def pad_image_w_zeros(image):
    shape = image.shape
    dim1 = shape[0]
    dim2 = shape[1]
    if dim1>= dim2:
        new_image = np.zeros((dim1,dim1))
        size_diff = dim1-dim2
        new_image[:,int(size_diff/2):int(size_diff/2)+dim2] = image
    if dim2> dim1:
        new_image = np.zeros((dim2,dim2))
        size_diff = dim2-dim1
        new_image[int(size_diff/2):int(size_diff/2)+dim1,:] = image
    return new_image


def load_dog_data(data_path : str,
                  image_shape : (int,int),
                  sample_rate : float,
                  simple : bool):
    image_list = []
    label_list = []
    label_dict = {}
    label = 0
    list_subfolders_with_paths = [f.path for f in os.scandir(data_path) if f.is_dir()]

    for folder_path in list_subfolders_with_paths:
        match_obj = re.match('.*\-(.*)$',folder_path)
        
        if match_obj:
            label+=1
            label_dict[label] = match_obj[1]

        else:
            raise RuntimeError('Error in Regex, all folder names should be accounted for, but it seems like the regex missed the following folder:' + folder_path)
        if simple and label==3:
                break
        for f in os.scandir(folder_path):
            if np.random.uniform()>sample_rate:
                continue
            if f.path.endswith('.jpg'):

                image = cv2.imread(f.path, 0)  #Read image, 0 is the flag for grayscale
                image=pad_image_w_zeros(image)
                image = cv2.resize(image,(image_shape))
                image_list.append(image)
                label_list.append(label)
    return((image_list,label_list,label_dict))


class SquarePadding(object):
    """Resize the input PIL Image to the given size.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self):
        pass

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be scaled.

        Returns:
            PIL Image: Rescaled image.
        """
        width,height = img.size
        if width>= height:
            new_image = np.zeros((width,width))
            size_diff =width-height
            return Pad(padding=(0,int(size_diff/2))).__call__(img)
        if height> width:
            new_image = np.zeros((height,height))
            size_diff = height-width
            return Pad(padding=(int(size_diff/2),0)).__call__(img)
