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
        bbox = [439.75,351.48,475.95,334.84,490.17,365.79,453.97,382.43]
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
        data_type = 'TPS2C'
        data_path = '/home/ubuntu/Downloads/dataset/ThreePastShop2cor/cut'
        bbox = [188, 134, 246, 284]  # like MOT data. format: x1, y1, x2, y2
    elif 15 == sequence_num:  # large box problem
        data_type = 'TPS2C'
        data_path = '/home/ubuntu/Downloads/dataset/ThreePastShop2cor/cut'
        bbox = [245, 127, 298, 272]  # like MOT data. format: x1, y1, x2, y2

    if data_type == 'VOT': # VOT data. format: x0, y0, x1, y1, x2, y2, x3, y3
        left_top_x = min([bbox[i] for i in range(0, 8, 2)])
        left_top_y = min([bbox[i] for i in range(1, 8, 2)])
        right_bottom_x = max([bbox[i] for i in range(0, 8, 2)])
        right_bottom_y = max([bbox[i] for i in range(1, 8, 2)])

        # change to [top-left_x, top-left_y, w, h]
        init_bbox = [0.0] * 4
        init_bbox[0] = left_top_x
        init_bbox[1] = left_top_y
        init_bbox[2] = right_bottom_x - left_top_x
        init_bbox[3] = right_bottom_y - left_top_y
    elif data_type == 'TPS2C':
        init_bbox = [bbox[0],
                     bbox[1],
                     bbox[2] - bbox[0],
                     bbox[3] - bbox[1]]

    frames_path = sorted(glob.glob(os.path.join(data_path, '*.jpg')))

    return frames_path, init_bbox


def read_frame_rgb(frame_path):
    rgb_img = cv2.imread(frame_path)
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

    return rgb_img


def show_tracking(frame, center_box, is_first=False):
    # [left_top_x, left_top_y, w, h] to [x1, y1, x2, y2]
    corner_box = [center_box[0],
                  center_box[1],
                  center_box[0] + center_box[2],
                  center_box[1] + center_box[3]]

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    cv2.rectangle(frame,
                  (int(corner_box[0]), int(corner_box[1])),
                  (int(corner_box[2]), int(corner_box[3])),
                  [0, 0, 255], 2)

    cv2.imshow('tracking', frame)

    if is_first:
        cv2.waitKey(0)
    else:
        k = cv2.waitKey(10) & 0xff
        if k == 27:
            raise Exception('Stop')


# the image format should be 'RGB'
if __name__ == '__main__':
    # weights_path = newest_weights_path()
    weights_path = 'saved/test_model_weights/epoch_0_step_0_.pth'
    tracker = TrackerSiamFC(load_path=weights_path, mode='test')

    # init_box format: [left_top_x, left_top_y, w, h]
    frames_path, init_box = choose_video(15)

    for i, frame_path in enumerate(frames_path):

        if i == 0:
            first_frame = read_frame_rgb(frame_path)
            tracker.init_first_frame(first_frame, init_box)
            show_tracking(first_frame, init_box, is_first=True)
        else:
            frame = read_frame_rgb(frame_path)
            predicted_box = tracker.predict_box_one_img(frame)  # [left_top_x, left_top_y, w, h]

            show_tracking(frame, predicted_box)

    # you can also use 'track_in_imgs' method in class TrackerSiamFC.



