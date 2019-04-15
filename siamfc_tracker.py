import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import time
from PIL import Image


class SiamFC(nn.Module):
    def __init__(self):
        super(SiamFC, self).__init__()

        # add batch norm, if not, loss will be nan
        self.feature_extract = nn.Sequential(
            # conv1
            nn.Conv2d(3, 96, 11, stride=2),
            nn.BatchNorm2d(96, eps=1e-6, momentum=0.05),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),

            # conv2
            nn.Conv2d(96, 256, 5, stride=1, groups=2),
            nn.BatchNorm2d(256, eps=1e-6, momentum=0.05),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),

            # conv3
            nn.Conv2d(256, 384, 3, stride=1),
            nn.BatchNorm2d(384, eps=1e-6, momentum=0.05),
            nn.ReLU(inplace=True),

            # conv4
            nn.Conv2d(384, 384, 3, stride=1, groups=2),
            nn.BatchNorm2d(384, eps=1e-6, momentum=0.05),
            nn.ReLU(inplace=True),

            # conv5
            nn.Conv2d(384, 256, 3, stride=1, groups=2))

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out',
                                        nonlinearity='relu')
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, exemplar_image, candidate_image):
        """
        :param exemplar_image: first frame which size is n*3*127*127,
        :param candidate_image: behind frames, each size is n*3*255*255,
        :return responses_maps: each size is 1*17*17
        """

        exemplar_z = self.feature_extract(exemplar_image)  # n*128*6*6
        candidate_x = self.feature_extract(candidate_image)  # n*128*22*22


        n, c, h, w = candidate_x.size()
        candidate_x = candidate_x.view(1, n*c, h, w)

        responses_maps = F.conv2d(candidate_x, exemplar_z, groups=n)
        responses_maps.view(n, 1, responses_maps.size(-2), responses_maps.size(-1))

        # https://github.com/bilylee/SiamFC-TensorFlow/issues/17
        responses_maps = 0.001*responses_maps

        return responses_maps


class TrackerSiamFC(object):
    def __init__(self, load_path=None, mode=None):

        self.config = self.setup_config()

        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
            print('GPU found!')
        else:
            self.device = torch.device('cpu')
            print('GPU not found!')

        model = SiamFC()

        self.__mode = mode

        if self.__mode == 'train':

            # use multi GPUs
            GPU_num = torch.cuda.device_count()
            if GPU_num > 1:
                print("Use {} GPUs!".format(torch.cuda.device_count()))
                self.model = nn.DataParallel(model)
            elif GPU_num == 1:
                print("Use one GPU!")
                self.model = model
            else:
                print("Use CPU!")
                self.model = model
            self.model.to(self.device)

            self.optimizer = torch.optim.SGD(self.model.parameters(),
                                             lr=self.config['initial_lr'],
                                             momentum=self.config['momentum'],
                                             weight_decay=self.config['weight_decay'])

            self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer,
                                                                       gamma=self.config['lr_decay'])
            self.batch_size = None
            self.labels = None
            self.weights = None

            self.model.train()

        elif self.__mode == 'test':

            # only need one GPU (GPU0) if exist
            self.model = model
            self.model.to(self.device)

            self.prvs_bbox = None  # should be [c_y, c_x, h, w] ndarray
            self.kernel = None

            # upsample size
            self.upscale_size = self.config['response_size'] * self.config['response_up_scale']

            # make window to penalize the response map after upsample
            hanning_window = np.outer(np.hanning(self.upscale_size),
                                   np.hanning(self.upscale_size))
            self.hanning_window = hanning_window / hanning_window.sum()

            # make scales to handle scale variations
            powers_index = np.linspace(-(self.config['scale_num'] // 2),
                                       self.config['scale_num'] // 2,
                                       self.config['scale_num'])
            self.scales = self.config['scale_step'] ** powers_index

            self.x_pad_size = None  # record pad size in search image x for locating the center of predict object

            # load weights
            if load_path is not None:
                # self.model.load_state_dict(torch.load(load_path,
                #                                       map_location=lambda storage, loc: storage))
                state_dict = torch.load(load_path, map_location="cuda:0")

                self.model.load_state_dict(state_dict)

                print('Loading weights from', load_path, 'success!')
            else:
                raise Exception('No weights loaded in' + load_path)

            self.model.eval()

        else:
            raise Exception('Select TrackerSiamfFC mode! (train / test)')

    @staticmethod
    def setup_config():
        config = {
            # ----------train-----------
            # used in training
            'initial_lr': 0.01,
            'lr_decay': 0.8685113737513527,  # https://github.com/bilylee/SiamFC-TensorFlow/issues/50
            'weight_decay': 5e-4,
            'momentum': 0.9,

            # used in making gt labels
            'total_stride': 8,  # sum the stride in each layer
            'r_pos': 16,

            # used in making input
            'exemplar_size': 127,
            'search_size': 255,
            'context': 0.5,

            # ----------eval-----------
            # used in analysing response map
            'response_size': 17,
            'response_up_scale': 16,
            'scale_step': 1.0375,
            'scale_num': 3,
            'adjust_scale': 0.001,  # https://github.com/bilylee/SiamFC-TensorFlow/issues/17
            'scale_penalty': 0.9745,
            'window_influence': 0.176,  # TODO: without explanation
            'scale_damping': 0.59,  # TODO: without explanation
        }

        return config

    def _create_labels(self, labels_size):
        n, c, h, w = labels_size

        if self.labels is not None and self.labels.size == labels_size:  # make labels once unless batch size change
            return
        else:
            self.batch_size = n

            x = np.arange(w) - w // 2
            y = np.arange(h) - w // 2
            x, y = np.meshgrid(x, y)

            dist_grid = np.abs(x) + np.abs(y)

            labels = np.where(dist_grid <= self.config['r_pos'] / self.config['total_stride'],
                              1.0,
                              0.0)

            pos_num = np.sum(labels == 1)
            neg_num = np.sum(labels == 0)

            weights = np.zeros_like(labels)

            # normalization
            weights[labels == 1] = 0.5 / pos_num
            weights[labels == 0] = 0.5 / neg_num
            weights *= pos_num + neg_num

            # batch label and weight
            labels = np.tile(labels, (n, c, 1, 1))
            weights = np.tile(weights, (n, c, 1, 1))

            # convert to tensors
            self.labels = torch.from_numpy(labels).to(self.device).float()
            self.weights = torch.from_numpy(weights).to(self.device).float()

    def _crop_resize(self, image, bbox, target='z'):
        """
        :param image: image before cropped.
        :param bbox: should be [c_y, c_x, h, w]
        :param target: if 'z', return exemplar image with size (127*127*3)
                       if 'x', return candidate search image with size (255*255*3)
                       if 'x_scaled', return candidate search image with scaled size
        :param scales: ndarray, scales
        :return patch: based on param target.
        """
        def make_patch(image, center, crop_size, out_size, pad_color):
            corners = np.concatenate((np.round(center - (crop_size - 1) / 2),
                                      np.round(center - (crop_size - 1) / 2) + crop_size)).astype(int)

            # image shape: (h, w, c)
            pads = np.concatenate((-corners[:2], corners[2:] - image.shape[:2]))

            npad = max(0, int(pads.max()))
            if npad > 0:
                image = cv2.copyMakeBorder(image,
                                           npad, npad, npad, npad,
                                           cv2.BORDER_CONSTANT, value=pad_color)

            corners = (corners + npad).astype(int)
            patch = image[corners[0]:corners[2], corners[1]:corners[3]]

            # resize to out_size
            patch = cv2.resize(patch, (out_size, out_size))

            return patch

        # according to the formula in paper.
        center = bbox[:2]  # (y, x)
        h, w = bbox[2:]
        wc_z = w + self.config['context'] * (w + h)
        hc_z = h + self.config['context'] * (w + h)
        size_p_z = np.sqrt(wc_z * hc_z)  # size after padding (square region)

        avg_color = np.mean(image, axis=(0, 1))

        if target == 'z':
            crop_size = round(size_p_z)
            out_size = self.config['exemplar_size']

            return make_patch(image, center, crop_size, out_size, avg_color)

        elif target == 'x':
            # size after padding (square region)
            size_p_x = size_p_z * self.config['search_size'] / self.config['exemplar_size']
            crop_size = round(size_p_x)
            out_size = self.config['search_size']

            return make_patch(image, center, crop_size, out_size, avg_color)

        elif target == 'x_scaled':
            # size after padding (square region)
            size_p_x = size_p_z * self.config['search_size'] / self.config['exemplar_size']
            self.x_pad_size = size_p_x
            out_size = self.config['search_size']
            patches = []
            for scale in self.scales:
                crop_size = size_p_x * scale
                patches.append(make_patch(image, center, crop_size, out_size, avg_color))

            return np.array(patches)
        else:
            raise Exception('Confirm the target to crop and resize!')

    def train_step(self, batch, update_lr=False):
        if self.__mode != 'train':
            raise Exception('Tracker is not in train mode! Confirm TrackerSiamfFC config!')

        z = batch[0].to(self.device)
        x = batch[1].to(self.device)

        if update_lr:
            self.lr_scheduler.step()

        responses = self.model(z, x)  # remember the calling of the forward method in nn.Module.

        self._create_labels(responses.size())

        loss = F.binary_cross_entropy_with_logits(responses, self.labels, weight=self.weights)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def init_first_frame(self, first_frame, box):
        # the input box should be [top_left_x, top_left_y, w, h]

        if self.__mode != 'test':
            raise Exception('Tracker is not in test mode! Confirm TrackerSiamfFC config!')

        image = np.asarray(first_frame)

        # [top_left_x, top_left_y, w, h] to [c_y, c_x, h, w]
        box = np.array([
            box[1] - 1 + (box[3] - 1) / 2,
            box[0] - 1 + (box[2] - 1) / 2,
            box[3], box[2]], dtype=np.float32)

        # box should be [c_y, c_x, h, w]
        self.prvs_bbox = box

        exemplar_image = self._crop_resize(image, box, target='z')

        # permute: [H, W, C] -> [C, H, W]
        # unsqueeze: [C, H, W] -> [1, C, H, W]
        exemplar_image = torch.from_numpy(exemplar_image).to(
            self.device, torch.float).permute([2, 0, 1]).unsqueeze(0)

        with torch.no_grad():
            self.kernel = self.model.feature_extract(exemplar_image)

    def predict_box_one_img(self, image):
        # 1. crop based on prvs_box
        # 2. get response maps
        # 3. select scale
        # 4. upsample
        if self.kernel is None:
            raise Exception('Make sure the func \'init_first_frame\' is called!')

        image = np.asarray(image)
        search_imgs = self._crop_resize(image, self.prvs_bbox, target='x_scaled')
        search_imgs = torch.from_numpy(search_imgs).to(
            self.device, torch.float).permute([0, 3, 1, 2])

        with torch.no_grad():
            features = self.model.feature_extract(search_imgs)

            # can not use forward method
            responses = self.config['adjust_scale'] * F.conv2d(features, self.kernel)

        # before squeeze torch.Size([3, 1, 17, 17])
        # can't convert CUDA tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
        responses = responses.squeeze(1).cpu().numpy()
        # after squeeze (3, 17, 17)

        # upsample and penalize
        responses = np.stack([cv2.resize(
            t, (self.upscale_size, self.upscale_size),
            interpolation=cv2.INTER_CUBIC) for t in responses], axis=0)  # (3, 272, 272)

        # penalties value: (0.9745, 0.9745, 0.9745)
        penalties = self.config['scale_penalty'] * np.ones((self.config['scale_num']))
        current_scale_idx = int((self.config['scale_num']-1) / 2)
        penalties[current_scale_idx] = 1.0  # penalties value: (0.9745, 1.0, 0.9745)

        each_scale_max = np.max(responses, axis=(1, 2))
        response_penalized = each_scale_max * penalties
        best_scale_id = np.argmax(response_penalized)
        response = responses[best_scale_id]

        response -= response.min()
        response /= response.sum() + 1e-16

        # put hanning window
        window_influence = self.config['window_influence']
        response = (1 - window_influence) * response + window_influence * self.hanning_window

        # peak point [y, x]
        r_max, c_max = np.unravel_index(response.argmax(), response.shape)

        # locate target center
        # after conv, the position can only finded by the distance to origin center
        # the distance between peak point and center point, [y, x]
        # distance in up_response (272, 272)
        dist_in_response = np.array([r_max, c_max]) - self.upscale_size // 2
        # distance in search (255, 255)
        dist_in_search = dist_in_response / self.config['response_up_scale'] * self.config['total_stride']
        # distance in scaled crop
        distance_in_scaled_crop = dist_in_search * \
                                  (self.x_pad_size * self.scales[best_scale_id]) / self.config['search_size']

        self.prvs_bbox[:2] += distance_in_scaled_crop

        # update target size
        update_scale = (1 - self.config['scale_damping']) * 1.0 + \
                       self.config['scale_damping'] * self.scales[best_scale_id]

        self.prvs_bbox[2:] *= update_scale

        # [c_y, c_x, h, w] to [top_left_x, top_left_y, w, h]
        box = np.array([
            self.prvs_bbox[1] + 1 - (self.prvs_bbox[3] - 1) / 2,
            self.prvs_bbox[0] + 1 - (self.prvs_bbox[2] - 1) / 2,
            self.prvs_bbox[3], self.prvs_bbox[2]])

        return box  # predict box in present frame

    def track_in_imgs(self, img_files, box):
        # box should be [c_y, c_x, h, w]

        frame_num = len(img_files)
        boxes = np.zeros((frame_num, 4))
        boxes[0] = box
        times = np.zeros(frame_num)

        for f, img_file in enumerate(img_files):
            image = Image.open(img_file)
            if not image.mode == 'RGB':
                image = image.convert('RGB')

            start_time = time.time()
            if f == 0:
                self.init_first_frame(image, box)
            else:
                boxes[f, :] = self.predict_box_one_img(image)
            times[f] = time.time() - start_time

        return boxes, times








