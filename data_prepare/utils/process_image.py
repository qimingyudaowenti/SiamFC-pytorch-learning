import numpy as np
import collections

from cv2 import resize
from .misc_funcs import get_center

Rectangle = collections.namedtuple('Rectangle', ['x', 'y', 'width', 'height'])

def get_crops(im, bbox, size_z, size_x, context_amount):
    """Obtain image sub-window, padding with avg channel if area goes outside of border

    Adapted from https://github.com/bertinetto/siamese-fc/blob/master/ILSVRC15-curation/save_crops.m#L46

    Args:
        im: Image ndarray
        bbox: Named tuple (x, y, width, height) x, y corresponds to the crops center
        size_z: Target + context size
        size_x: The resultant crop size
        context_amount: The amount of context

    Returns:
        image crop: Image ndarray
    """
    cy, cx, h, w = bbox.y, bbox.x, bbox.height, bbox.width
    wc_z = w + context_amount * (w + h)
    hc_z = h + context_amount * (w + h)
    s_z = np.sqrt(wc_z * hc_z)    # size after padding
    scale_z = size_z / s_z    # the scale

    d_search = (size_x - size_z) / 2
    pad = d_search / scale_z
    s_x = s_z + 2 * pad
    scale_x = size_x / s_x

    image_crop_x, _, _, _, _ = get_subwindow_avg(im, [cy, cx],
                                                 [size_x, size_x],
                                                 [np.round(s_x), np.round(s_x)])

    return image_crop_x, scale_x


def get_subwindow_avg(im, pos, model_sz, original_sz):
    # avg_chans = np.mean(im, axis=(0, 1)) # This version is 3x slower
    avg_chans = [np.mean(im[:, :, 0]), np.mean(im[:, :, 1]), np.mean(im[:, :, 2])]
    if not original_sz:
        original_sz = model_sz
    sz = original_sz
    im_sz = im.shape
    # make sure the size is not too small
    assert im_sz[0] > 2 and im_sz[1] > 2
    c = [get_center(s) for s in sz]

    # check out-of-bounds coordinates, and set them to avg_chans
    context_xmin = np.int(np.round(pos[1] - c[1]))
    context_xmax = np.int(context_xmin + sz[1] - 1)
    context_ymin = np.int(np.round(pos[0] - c[0]))
    context_ymax = np.int(context_ymin + sz[0] - 1)
    left_pad = np.int(np.maximum(0, -context_xmin))
    top_pad = np.int(np.maximum(0, -context_ymin))
    right_pad = np.int(np.maximum(0, context_xmax - im_sz[1] + 1))
    bottom_pad = np.int(np.maximum(0, context_ymax - im_sz[0] + 1))

    context_xmin = context_xmin + left_pad
    context_xmax = context_xmax + left_pad
    context_ymin = context_ymin + top_pad
    context_ymax = context_ymax + top_pad
    if top_pad > 0 or bottom_pad > 0 or left_pad > 0 or right_pad > 0:
        R = np.pad(im[:, :, 0], ((top_pad, bottom_pad), (left_pad, right_pad)),
                             'constant', constant_values=(avg_chans[0]))
        G = np.pad(im[:, :, 1], ((top_pad, bottom_pad), (left_pad, right_pad)),
                             'constant', constant_values=(avg_chans[1]))
        B = np.pad(im[:, :, 2], ((top_pad, bottom_pad), (left_pad, right_pad)),
                             'constant', constant_values=(avg_chans[2]))

        im = np.stack((R, G, B), axis=2)

    im_patch_original = im[context_ymin:context_ymax + 1,
                                            context_xmin:context_xmax + 1, :]
    if not (model_sz[0] == original_sz[0] and model_sz[1] == original_sz[1]):
        im_patch = resize(im_patch_original, tuple(model_sz))
    else:
        im_patch = im_patch_original
    return im_patch, left_pad, top_pad, right_pad, bottom_pad


def convert_bbox_format(bbox, to):
    x, y, target_width, target_height = bbox.x, bbox.y, bbox.width, bbox.height
    if to == 'top-left-based':
        x -= get_center(target_width)
        y -= get_center(target_height)
    elif to == 'center-based':
        y += get_center(target_height)
        x += get_center(target_width)
    else:
        raise ValueError("Bbox format: {} was not recognized".format(to))
    return Rectangle(x, y, target_width, target_height)
