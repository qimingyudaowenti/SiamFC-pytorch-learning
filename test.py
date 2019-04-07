import os
import cv2
import glob

from siamfc_tracker import TrackerSiamFC


def newest_weights_path():
    weights_path = 'saved/test_model_weights'
    pths = os.listdir(weights_path)
    pths_flag = [(int(a.split('_')[1]), int(a.split('_')[3])) for a in pths if len(a.split('_')) == 5]

    if len(pths_flag) == 0:
        print('no weights loaded in', weights_path)
        return None
    else:
        pths_flag.sort()
        newest_epoch, newest_step = pths_flag[-1]
        pth_name = 'epoch_{}_step_{}_.pth'.format(newest_epoch, newest_step)

        print('load newest weights:', pth_name)

        return os.path.join(weights_path, pth_name)


def choose_video(sequence_num):

    if sequence_num == 1:
        data_path = '/home/ubuntu/Downloads/dataset/VOT/vot2016/marching'
        data_type = 'VOT'
        bbox = [735,217,816,217,816,388,735,388]
    elif sequence_num == 2:
        data_path = '/home/ubuntu/Downloads/dataset/VOT/vot2016/road'
        data_type = 'VOT'
        bbox = [442.7,336.8,484.39,337.1,484.14,370.51,442.46,370.21]
    elif sequence_num == 3:
        data_path = '/home/ubuntu/Downloads/dataset/VOT/vot2016/octopus'  # bad
        data_type = 'VOT'
        bbox = [628.85,297.33,757.88,264.86,778.38,346.35,649.35,378.82]
    elif sequence_num == 4:
        data_path = '/home/ubuntu/Downloads/dataset/VOT/vot2016/bmx'
        data_type = 'VOT'
        bbox = [811.41,265.46,881.66,288.63,828.78,448.97,758.52,425.79]
    elif sequence_num == 5:
        data_path = '/home/ubuntu/Downloads/dataset/VOT/vot2016/bolt1'
        data_type = 'VOT'
        bbox = [348.13,162.4,368.15,170.96,350.05,213.28,330.04,204.72]
    elif sequence_num == 6:
        data_type = 'VOT'
        data_path = '/home/ubuntu/Downloads/dataset/VOT/vot2016/glove'
        bbox = [214.38, 155.66, 245.78, 152.2, 253.62, 223.4, 222.22, 226.86]
    elif sequence_num == 7:
        data_type = 'VOT'
        data_path = '/home/ubuntu/Downloads/dataset/VOT/vot2017/tiger'
        bbox = [28.788, 57.572, 97.714, 57.116, 98.27, 141.12, 29.344, 141.58]
    elif 8 == sequence_num:
        data_type = 'VOT'
        data_path = '/home/ubuntu/Downloads/dataset/VOT/vot2016/fish1'
        bbox = [182.56, 23.644, 230.88, 18.812, 234.44, 54.356, 186.12, 59.188]
    elif 9 == sequence_num:
        data_type = 'VOT'
        data_path = '/home/ubuntu/Downloads/dataset/VOT/vot2016/singer3'
        bbox = [444, 180.7, 483.45, 175.17, 489.65, 219.38, 450.2, 224.92]
    elif 10 == sequence_num:
        data_type = 'VOT'
        data_path = '/home/ubuntu/Downloads/dataset/VOT/vot2016/car1'
        bbox = [246.13, 162.64, 362.69, 168.59, 357.02, 279.72, 240.46, 273.77]
    elif 11 == sequence_num:  # lose object
        data_type = 'VOT'
        data_path = '/home/ubuntu/Downloads/dataset/VOT/vot2017/ants1'
        bbox = [137.21, 458.36, 156.83, 460.78, 148.35, 529.41, 128.72, 526.99]
    elif 12 == sequence_num:
        data_type = 'VOT'
        data_path = '/home/ubuntu/Downloads/dataset/VOT/vot2017/ball1'
        bbox = [496, 419, 536, 419, 536, 461, 496, 461]
    elif 13 == sequence_num:
        data_type = 'VOT'
        data_path = '/home/ubuntu/Downloads/dataset/VOT/vot2016/gymnastics4'
        bbox = [422.52, 214.84, 472.26, 206.17, 490.56, 311.21, 440.83, 319.88]
    elif 14 == sequence_num:
        data_type = 'VOT'
        data_path = '/home/ubuntu/Downloads/dataset/VOT/vot2017/frisbee'
        bbox = [419.77, 460.75, 459.49, 490.17, 416.37, 548.38, 376.65, 518.96]


    if data_type == 'VOT': # VOT data. format: x0, y0, x1, y1, x2, y2, x3, y3
        # change to [c_y, c_x, h, w]

        init_bbox = [0] * 4
        init_bbox[0] = (bbox[3] + bbox[7]) / 2
        init_bbox[1] = (bbox[0] + bbox[4]) / 2
        init_bbox[2] = abs(bbox[7] - bbox[3])
        init_bbox[3] = abs(bbox[4] - bbox[0])

        picture_type = 'jpg'

        # files = os.listdir(data_path)
        # frames = [i for i in files if i.split('.')[-1] == picture_type]
        # frames.sort(key=lambda n: int(n.split('.')[0]))
        # frames_path = [os.path.join(data_path, p) for p in frames]

        frames_path = sorted(glob.glob(os.path.join(data_path, '*.jpg')))

    return frames_path, init_bbox


def read_frame_RGB(frame_path):
    BGR_img = cv2.imread(frame_path)
    RGB_img = cv2.cvtColor(BGR_img, cv2.COLOR_BGR2RGB)

    return RGB_img


def center_box_2_corner_box(box):
    # [c_y, c_x, h, w] to [x1, y1, x2, y2]

    c_y, c_x, h, w = box
    new_box = [c_x - w / 2,
               c_y - h / 2,
               c_x + w / 2,
               c_y + h / 2]

    return new_box


def show_tracking(frame, center_box):
    corner_box = center_box_2_corner_box(center_box)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    cv2.rectangle(frame,
            (int(corner_box[0]), int(corner_box[1])),
            (int(corner_box[2]), int(corner_box[3])),
            [0,0,255], 2)

    cv2.imshow('tracking', frame)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        raise Exception('Stop')


# the image format should be 'RGB'
if __name__ == '__main__':
    weights_path = newest_weights_path()
    tracker = TrackerSiamFC(load_path=weights_path, mode='test')

    frames_path, init_box = choose_video(1)

    for i, frame_path in enumerate(frames_path):

        if i == 0:
            first_frame = read_frame_RGB(frame_path)
            tracker.init_first_frame(first_frame, init_box)
        else:
            frame = read_frame_RGB(frame_path)
            predicted_box = tracker.predict_box_one_img(frame)  # center based box
            show_tracking(frame, predicted_box)

    # you can also use 'track_in_imgs' method in class TrackerSiamFC.



