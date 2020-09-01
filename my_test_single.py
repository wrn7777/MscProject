import os
import glob
import json
import pandas as pd
import csv
import torch
from sklearn.metrics import *
from torch.autograd import Variable
from torch.nn import functional as F

from opts import parse_opts_online
from model import generate_model
from mean import get_mean, get_std
from spatial_transforms import *
from temporal_transforms import *
from target_transforms import ClassLabel
from dataset import get_online_data
from utils import AverageMeter, LevenshteinDistance, Queue
from types import SimpleNamespace


import pdb
import time
import numpy as np
import datetime
from natsort import natsorted


def load_models(opt):
    if opt.root_path != '':
        opt.result_path = os.path.join(opt.root_path, opt.result_path)
        if opt.resume_path:
            opt.resume_path = os.path.join(opt.root_path, opt.resume_path)
        if opt.pretrain_path:
            opt.pretrain_path = os.path.join(opt.root_path, opt.pretrain_path)

    opt.scales = [opt.initial_scale]
    for i in range(1, opt.n_scales):
        opt.scales.append(opt.scales[-1] * opt.scale_step)
    opt.arch = '{}'.format(opt.model)
    opt.mean = get_mean(opt.norm_value)
    opt.std = get_std(opt.norm_value)

    torch.manual_seed(opt.manual_seed)
    classifier, parameters = generate_model(opt)

    if opt.resume_path:
        # print('loading checkpoint {}'.format(opt.resume_path))
        checkpoint = torch.load(
            opt.resume_path, map_location=torch.device('cpu'))
#        assert opt.arch == checkpoint['arch']

        classifier.load_state_dict(checkpoint['state_dict'])

    return classifier


def classify_video(opt, video_path):

    classifier = load_models(opt)

    if opt.no_mean_norm and not opt.std_norm:
        norm_method = Normalize([0, 0, 0], [1, 1, 1])
    elif not opt.std_norm:
        norm_method = Normalize(opt.mean, [1, 1, 1])
    else:
        norm_method = Normalize(opt.mean, opt.std)

    spatial_transform = Compose([
        Scale(150),
        CenterCrop(112),
        ToTensor(opt.norm_value), norm_method
    ])


    # vedio open
    idx2label = ["Zoom_in_with_fingers", "Click_with_index_finger", "Sweep_diagonal", "Sweep_circle",
                 "Sweep_cross", "Make_a_phone_call", "Wave_finger", "Knock", "Dual_hands_heart", "Move_fingers_left"]
    opt.sample_duration = opt.sample_duration_clf

    cap = cv2.VideoCapture(video_path)
    # cap = cv2.VideoCapture(0)
    num_frame = 0
    clip = []
    classifier.eval()
    spatial_transform.randomize_parameters()
    temporal_transform = TemporalCenterCrop(
        opt.sample_duration, opt.downsample)
    cur_label = ""
    step = 2

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    t1 = time.time()
    # print('toral:',total_frames)
    while cap.isOpened():
        num_frame += 1
        if num_frame == total_frames - 1:
            break
        ret, frame = cap.read()
        cur_frame = cv2.resize(frame, (320, 240))
        cur_frame = Image.fromarray(cv2.cvtColor(cur_frame, cv2.COLOR_BGR2RGB))
        cur_frame = cur_frame.convert('RGB')
        if num_frame % step == 0:
            clip.append(cur_frame)

    indexes = temporal_transform([i for i in range(len(clip))])
    # print(indexes)
    new_clip = []
    for i in indexes:
        new_clip.append(clip[i])
    new_clip = [spatial_transform(img) for img in new_clip]
    im_dim = new_clip[0].size()[-2:]
    try:
        test_data = torch.cat(new_clip, 0).view(
            (opt.sample_duration, -1) + im_dim).permute(1, 0, 2, 3)
    except Exception as e:
        pdb.set_trace()
        raise e
    inputs = torch.cat([test_data], 0).view(
        1, 3, opt.sample_duration, 112, 112)
    # print(inputs.size())

    with torch.no_grad():
        inputs = Variable(inputs)
        if opt.modality_clf == 'RGB':
            inputs_clf = inputs[:, :, :, :, :]
        inputs_clf = torch.Tensor(inputs_clf.numpy()[:, :, ::2, :, :])
        outputs_clf = classifier(inputs_clf)
        outputs_clf = F.softmax(outputs_clf, dim=1)
        outputs_clf = outputs_clf.cpu().numpy()[0].reshape(-1, )

        best2, best1 = tuple(outputs_clf.argsort()[-2:][::1])
        cur_label = idx2label[best1]

    elapsedTime = time.time() - t1
    return cur_label, elapsedTime,total_frames


def getFiles(dir, suffix):  # 查找根目录，文件后缀
    res = []
    for root, directory, files in os.walk(dir):  # =>当前根,根下目录,目录下的文件
        for filename in files:
            name, suf = os.path.splitext(filename)  # =>文件名,文件后缀
            if suf == suffix:
                res.append(filename)  # =>吧一串字符串组合成路径
    return res


with open("./res/ui_opt2.json", 'r') as opt_file:
    opt = json.load(opt_file)
opt = SimpleNamespace(**opt)



labels = ['Zoom_in_with_fingers',
          'Click_with_index_finger', 'Make_a_phone_call', 'Wave_finger', 'Knock', 'Move_fingers_left', 'Sweep_diagonal', 'Sweep_circle', 'Sweep_cross', 'Dual_hands_heart']

def get_dataFrame(dir):
    files = getFiles(dir, ".mov")
    files = natsorted(files)

    files_list = []
    result_list = []
    time_durations = []
    frames_videos = []
    for file in files:
        if len(file) > 6:
            path = os.path.join(dir, file)
            result, time_duration, total_frames = classify_video(opt, path)
            files_list.append(file)
            result_list.append(result)
            time_durations.append(time_duration)
            frames_videos.append(total_frames)

    df = pd.DataFrame({'file': files_list, 'total_frames': frames_videos, 'time': time_durations, 'result': result_list, 'ground_truth':labels*3})
    return df

dir = "/Users/ryan/Desktop/TestGesture/subject1"
df1 = get_dataFrame(dir)

dir = "/Users/ryan/Desktop/TestGesture/subject2"
df2 = get_dataFrame(dir)

dir = "/Users/ryan/Desktop/TestGesture/subject3"
df3 = get_dataFrame(dir)


df1.to_csv("/Users/ryan/Desktop/TestGesture/single/subject1.csv", index=False)
df2.to_csv("/Users/ryan/Desktop/TestGesture/single/subject2.csv", index=False)
df3.to_csv("/Users/ryan/Desktop/TestGesture/single/subject3.csv", index=False)

