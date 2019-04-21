import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
    plt.show()


if __name__ == '__main__':
    # weights_path = '../saved/test_model_weights/epoch_0_step_0_.pth'
    weights_path = '../saved/test_model_weights/epoch_49_step_117528_.pth'

    state_dict = torch.load(weights_path, map_location="cpu")

    layers = list(state_dict.keys())
    for i, v in enumerate(layers):
        print(i, v)

    layers_index = [0, 7, 14, 21, 28]  # conv1 - conv5

    layer_id = 1
    index_in_state = layers_index[layer_id]

    conv_kernels = state_dict[layers[index_in_state]]
    show_kernels = conv_kernels[:, 0:3, ...]

    print(show_kernels.shape)

    show(make_grid(show_kernels, nrow=12, padding=1, normalize=True, scale_each=True))

