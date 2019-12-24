from __future__ import print_function
import argparse
import os
import pickle
import shutil
import sys
import time
import copy
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from data.SIXray import SIXray_CLASSES as labelmap, SIXrayDetection, BaseTransform, \
    SIXrayAnnotationTransform, TEST_SET_PATH, TEST_SET_PATH_coreless
from ssd import build_ssd
import os.path as osp

if sys.version_info[0] == 2:
    pass
else:
    pass


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "a1")

TRAIN_SET_PATH = osp.join(osp.abspath('.'), 'data_sets', '6000')
TEST_SET_PATH = osp.join(osp.abspath('.'), 'data_sets', 'test','core_3000')
TEST_SET_PATH_coreless = osp.join(osp.abspath('.'), 'data_sets', 'test','coreless_3000')

SAVE_FOLDER_ONE = "predicted_file_level1/"
SAVE_FOLDER_TWO = "predicted_file_level2/"
SAVE_FOLDER_THREE = "predicted_file_level3/"


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Evaluation')
parser.add_argument('--trained_model',
                    default="weights/XRay_1212.pth", type=str,
                    help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default= SAVE_FOLDER_ONE, type=str, help='File path to save results')
parser.add_argument('--confidence_threshold', default=0.2, type=float,
                    help='Detection confidence threshold')
parser.add_argument('--top_k', default=5, type=int,
                    help='Further restrict the number of predictions to parse')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use cuda to train model')
parser.add_argument('--SIXray_root_core', default=TEST_SET_PATH,
                    help='Dataset root directory path')
parser.add_argument('--cleanup', default=True, type=str2bool,
                    help='Cleanup and remove results files following eval')
parser.add_argument('--imagesetfile', default="./img_ids.txt", type=str, help='imageset file path to open')

args = parser.parse_args()
annopath = os.path.join(args.SIXray_root_core, 'Annotation', 'core_battery%s.txt')
imgpath = os.path.join(args.SIXray_root_core, 'Image', 'core_battery%s.jpg')

option = 1
if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't using \
                 CUDA.  Run with --cuda for optimal eval speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


devkit_path = args.save_folder
dataset_mean = (104, 117, 123)
set_type = 'test'


class Timer(object):
    """A simple timer."""

    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff


def get_output_dir(name, phase):
    """Return the directory where experimental artifacts are placed.
    If the directory does not exist, it is created.
    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    filedir = os.path.join(name, phase)
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    return filedir


def get_voc_results_file_template(image_set, cls):
    # VOCdevkit/VOC2007/results/det_test_aeroplane.txt
    filename = 'det_' + image_set + '_%s.txt' % (cls)
    filedir = os.path.join(devkit_path)
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    path = os.path.join(filedir, filename)
    return path


def write_voc_results_file(all_boxes, dataset):
    for cls_ind, cls in enumerate(labelmap):
        filename = get_voc_results_file_template(set_type, cls)
        with open(filename, 'wt') as f:
            for im_ind, index in enumerate(dataset.ids):
                dets = all_boxes[cls_ind + 1][im_ind]
                if dets == []:
                    continue
                # the VOCdevkit expects a1-based indices
                for k in range(dets.shape[0]):
                    f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                            format(index, dets[k, -1],
                                   dets[k, 0] + 1, dets[k, 1] + 1,
                                   dets[k, 2] + 1, dets[k, 3] + 1))

def test_net(save_folder, net, cuda, dataset, transform, top_k,
             im_size=300, thresh=0.05):
    # //
    # //
    num_images = len(dataset)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(len(labelmap) + 1)]

    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}
    output_dir = get_output_dir('ssd300_120000', set_type)
    det_file = os.path.join(output_dir, 'detections.pkl')

    for i in range(num_images):
        im, gt, h, w = dataset.pull_item(i)
        # 这里im的颜色偏暗，因为BaseTransform减去了一个mean
        # im_saver = cv2.resize(im[(a2,a1,0),:,:].permute((a1,a2,0)).numpy(), (w,h))
        im_det = dataset.pull_image(i)

        # print(im_det)
        x = Variable(im.unsqueeze(0))
        if args.cuda:
            x = x.cuda()
        _t['im_detect'].tic()
        detections = net(x).data
        detect_time = _t['im_detect'].toc(average=False)

        # skip j = 0, because it's the background class
        # //
        # //
        # print(detections)
        for j in range(1, detections.size(1)):
            dets = detections[0, j, :]
            mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
            dets = torch.masked_select(dets, mask).view(-1, 5)
            if dets.size(0) == 0:
                continue
            boxes = dets[:, 1:]
            boxes[:, 0] *= w
            boxes[:, 2] *= w
            boxes[:, 1] *= h
            boxes[:, 3] *= h
            # print(boxes)
            scores = dets[:, 0].cpu().numpy()
            cls_dets = np.hstack((boxes.cpu().numpy(),
                                  scores[:, np.newaxis])).astype(np.float32,
                                                                 copy=False)
            all_boxes[j][i] = cls_dets

            # print(all_boxes)
            for item in cls_dets:
                # print(item)
                # print(item[5])
                if item[4] > thresh:
                    # print(item)
                    chinese = labelmap[j - 1] + str(round(item[4], 2))
                    # print(chinese+'det\n\n')
                    if chinese[0] == '带':
                        chinese = 'P_Battery_Core' + chinese[6:]
                    else:
                        chinese = 'P_Battery_No_Core' + chinese[7:]
                    cv2.rectangle(im_det, (item[0], item[1]), (item[2], item[3]), (0, 0, 255), 2)
                    cv2.putText(im_det, chinese, (int(item[0]), int(item[1]) - 5), 0,
                                0.6, (0, 0, 255), 2)
        real = 0
        if gt[0][4] == 3:
            real = 0
        else:
            real = 1

        for item in gt:
            if real == 0:
                print('this pic dont have the obj:', dataset.ids[i])
                break
            chinese = labelmap[int(item[4])]
            # print(chinese+'gt\n\n')
            if chinese[0] == '带':
                chinese = 'P_Battery_Core'
            else:
                chinese = 'P_Battery_No_Core'
            cv2.rectangle(im_det, (int(item[0] * w), int(item[1] * h)), (int(item[2] * w), int(item[3] * h)),
                          (0, 255, 255), 2)
            cv2.putText(im_det, chinese, (int(item[0] * w), int(item[1] * h) - 5), 0, 0.6, (0, 255, 255), 2)

        print('im_detect: {:d}/{:d} {:.3f}s'.format(i + 1, num_images, detect_time))


    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    # print('Evaluating detections')
    evaluate_detections(all_boxes, output_dir, dataset)

def evaluate_detections(box_list, output_dir, dataset):
    write_voc_results_file(box_list, dataset)


def test(img_path,anno_path):
     # load net
    num_classes = len(labelmap) + 1  # +a1 for background
    net = build_ssd('test', 300, num_classes)  # initialize SSD
    # map_location = torch.device('cpu')
    net.load_state_dict(torch.load(args.trained_model))
    #net.load_state_dict(torch.load(args.trained_model, map_location='cpu'))
    net.eval()
    # print('Finished loading model!')
    # load data
    dataset = SIXrayDetection(img_path,anno_path,args.imagesetfile,
                              BaseTransform(300, dataset_mean),
                              SIXrayAnnotationTransform())
    test_net(args.save_folder, net, args.cuda, dataset,
             BaseTransform(net.size, dataset_mean), args.top_k, 300,
             thresh=args.confidence_threshold)

    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True



if __name__ == '__main__':
   imgpath = ""
   annopath = ""



