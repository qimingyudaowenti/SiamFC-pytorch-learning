import vot
import sys
# import cv2
from PIL import Image  # PIL or opencv result in different results.
import os.path

# make sure the paths you need are append
basedir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir)))
from siamfc_tracker import TrackerSiamFC


def read_frame_rgb(frame_path):
    rgb_img = cv2.imread(frame_path)
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

    return rgb_img


handle = vot.VOT("rectangle")

# selection: collections.namedtuple('Rectangle', ['x', 'y', 'width', 'height'])
# Be careful. ['x', 'y'] is the top-left corner! not center!
selection = handle.region()
image_path = handle.frame()

if not image_path:
    sys.exit(0)

weights_path = basedir + '/../saved/test_model_weights/epoch_0_step_0_.pth'
tracker = TrackerSiamFC(load_path=weights_path, mode='test')

# first_frame = read_frame_rgb(image_path)
first_frame = Image.open(image_path)

# [left_top_x, left_top_y, w, h]
init_bbox = [selection.x,
             selection.y,
             selection.width,
             selection.height]

tracker.init_first_frame(first_frame, init_bbox)

while True:
    image_path = handle.frame()

    if not image_path:
        break

    # rgb image
    # image = read_frame_rgb(image_path)
    image = Image.open(image_path)

    predicted_bbox = tracker.predict_box_one_img(image)  # predicted_bbox: (top_left_x, top_left_y, w, h)

    # to (top_left_x, top_left_y, w, h)
    selection = selection._replace(x=predicted_bbox[0],
                                   y=predicted_bbox[1],
                                   width=predicted_bbox[2],
                                   height=predicted_bbox[3])

    handle.report(selection)

