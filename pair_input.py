import os
import pickle
import cv2
import numpy as np

from torch.utils.data import Dataset
from torchvision.transforms import Compose, CenterCrop, RandomCrop, ToTensor
from PIL import Image


class RandomStretch(object):

    def __init__(self, max_stretch=0.05, interpolation='bilinear'):
        assert interpolation in ['bilinear', 'bicubic']
        self.max_stretch = max_stretch
        self.interpolation = interpolation

    def __call__(self, img):
        scale = 1.0 + np.random.uniform(
            -self.max_stretch, self.max_stretch)
        size = np.round(np.array(img.size, float) * scale).astype(int)
        if self.interpolation == 'bilinear':
            method = Image.BILINEAR
        elif self.interpolation == 'bicubic':
            method = Image.BICUBIC
        return img.resize(tuple(size), method)


class InputPair(Dataset):
    def __init__(self, data_path):
        super(InputPair, self).__init__()
        self.config = self.setup_config()
        self.data_path = data_path

        imdb = self._load_pickle()

        self.videos_num = imdb['n_videos']
        self.videos = imdb['videos']
        self.indices = np.random.permutation(self.videos_num)

        # https://github.com/bilylee/SiamFC-TensorFlow/issues/69
        safe_size = self.config['instance_sz'] - \
                    int(self.config['instance_sz'] * self.config['max_stretch_scale'])
        perturbation_size = safe_size - 8  # size for center perturbation

        self.transform_z = Compose([
            RandomStretch(max_stretch=self.config['max_stretch_scale']),
            # after RandomStretch, the size may smaller than 255, make sure CenterCrop area in it.
            CenterCrop(safe_size),
            RandomCrop(perturbation_size),
            CenterCrop(self.config['exemplar_sz']),
            ToTensor()])

        # input x do not have to be 255*255*3
        self.transform_x = Compose([
            RandomStretch(max_stretch=self.config['max_stretch_scale']),
            CenterCrop(safe_size),
            RandomCrop(perturbation_size),
            ToTensor()])

    def __len__(self):
        return sum([len(i) for i in self.videos])

    def __getitem__(self, index):
        index = self.indices[index % self.videos_num]
        img_files = self.videos[index]
        rand_z, rand_x = self._sample_pair(len(img_files))

        exemplar_image = Image.open(img_files[rand_z])
        instance_image = Image.open(img_files[rand_x])

        exemplar_image = 255.0 * self.transform_z(exemplar_image)  # torch.Size: [3, 127, 127]
        instance_image = 255.0 * self.transform_x(instance_image)  # torch.Size: smaller than [3, 255, 255]

        return exemplar_image, instance_image

    def _load_pickle(self):
        imdb_path = os.path.join(self.data_path, self.config['input_imdb'])
        with open(imdb_path, 'rb') as f:
            imdb = pickle.load(f)
        return imdb

    def _sample_pair(self, video_len):
        assert video_len > 0

        if video_len == 1:
            return 0, 0
        elif video_len == 2:
            return 0, 1
        else:
            max_dist = min(video_len - 1, self.config['max_dist'])
            rand_dist = np.random.choice(max_dist) + 1
            rand_z = np.random.choice(video_len - rand_dist)
            rand_x = rand_z + rand_dist

        return rand_z, rand_x

    @staticmethod
    def setup_config():
        config = {
            'input_imdb': 'train_imdb.pickle',

            'max_dist': 100,
            'exemplar_sz': 127,
            'instance_sz': 255,
            'context': 0.5,
            'max_stretch_scale': 0.05,
        }

        return config


if __name__ == '__main__':
    input_data = InputPair('data')
    print(len(input_data))

    for i in range(3):
        for img in input_data[i]:
            img = img.numpy().astype(np.uint8)
            img = np.transpose(img, (1, 2, 0))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            center = (int(img.shape[1] / 2), int(img.shape[0] / 2))  # (x, y)
            cv2.circle(img, center, 2, (0, 0, 255), -1)

            cv2.imshow("OpenCV", img)
            cv2.waitKey()
