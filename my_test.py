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

step = 1

def weighting_func(x):
    return (1 / (1 + np.exp(-0.2 * (x - 18))))

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
        print('loading checkpoint {}'.format(opt.resume_path))
        checkpoint = torch.load(opt.resume_path, map_location=torch.device('cpu'))
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
        Scale(112),
        CenterCrop(112),
        ToTensor(opt.norm_value), norm_method
    ])

    target_transform = ClassLabel()

    # vedio open
    idx2label = ["Zoom_in_with_fingers", "Click_with_index_finger", "Sweep_diagonal", "Sweep_circle", "Sweep_cross", "Make_a_phone_call", "Wave_finger", "Knock", "Dual_hands_heart", "Move_fingers_left"]
    opt.sample_duration = opt.sample_duration_clf


    fps = ""
    # 

    cap = cv2.VideoCapture(video_path)
    # cap = cv2.VideoCapture(0)
    num_frame = 0
    clip = []
    active_index = 0
    passive_count = 0
    active = False
    prev_active = False
    finished_prediction = None
    pre_predict = False
    classifier.eval()
    cum_sum = np.zeros(opt.n_classes_clf, )
    clf_selected_queue = np.zeros(opt.n_classes_clf, )
    myqueue_clf = Queue(opt.clf_queue_size, n_classes=opt.n_classes_clf)
    results = []
    prev_best1 = opt.n_classes_clf
    spatial_transform.randomize_parameters()

    pre_len_result = 0
    cur_label = ""
    fps_r = []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    t1 = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if num_frame == 0:
            active = True
            cur_frame = cv2.resize(frame,(320,240))
            cur_frame = Image.fromarray(cv2.cvtColor(cur_frame,cv2.COLOR_BGR2RGB))
            cur_frame = cur_frame.convert('RGB')
            for i in range(opt.sample_duration):
                clip.append(cur_frame)
            clip = [spatial_transform(img) for img in clip]
            
        elif num_frame == total_frames:
            break
        elif num_frame == total_frames - 3:
            active = False

        if num_frame % step == 0:
            clip.pop(0)
            _frame = cv2.resize(frame,(320,240))
            _frame = Image.fromarray(cv2.cvtColor(_frame,cv2.COLOR_BGR2RGB))
            _frame = _frame.convert('RGB')
            _frame = spatial_transform(_frame)
            clip.append(_frame)
            im_dim = clip[0].size()[-2:]

            try:
                test_data = torch.cat(clip, 0).view((opt.sample_duration, -1) + im_dim).permute(1, 0, 2, 3)
            except Exception as e:
                pdb.set_trace()
                raise e
            inputs = torch.cat([test_data],0).view(1,3,opt.sample_duration,112,112)
            # print(inputs.size())
        
            with torch.no_grad():
                inputs = Variable(inputs)
                if opt.modality_clf == 'RGB':
                    inputs_clf = inputs[:, :, :, :, :]
                inputs_clf = torch.Tensor(inputs_clf.numpy()[:, :, ::2, :, :])
                outputs_clf = classifier(inputs_clf)
                outputs_clf = F.softmax(outputs_clf, dim=1)
                outputs_clf = outputs_clf.cpu().numpy()[0].reshape(-1, )

        
                myqueue_clf.enqueue(outputs_clf.tolist())

                if opt.clf_strategy == 'raw':
                    clf_selected_queue = outputs_clf
                elif opt.clf_strategy == 'median':
                    clf_selected_queue = myqueue_clf.median
                elif opt.clf_strategy == 'ma':
                    clf_selected_queue = myqueue_clf.ma
                elif opt.clf_strategy == 'ewma':
                    clf_selected_queue = myqueue_clf.ewma

                # print(clf_selected_queue)

            # one of the following line need to be commented !!!!
            if active:
                active_index += 1
                # cum_sum = ((cum_sum * (active_index - 1)) + (weighting_func(active_index) * clf_selected_queue)) / active_index  # Weighted Aproach
                cum_sum = ((cum_sum * (active_index-1)) + (1.0 * clf_selected_queue))/active_index #Not Weighting Aproach
                best2, best1 = tuple(cum_sum.argsort()[-2:][::1])
                if float(cum_sum[best1] - cum_sum[best2]) > opt.clf_threshold_pre:
                    finished_prediction = True
                    pre_predict = True
            else:
                active_index = 0
            if active == False and prev_active == True:
                finished_prediction = True
            elif active == True and prev_active == False:
                finished_prediction = False

            if finished_prediction == True:
                # print("fnishsed_prediction")
                #print(finished_prediction,pre_predict)
                best2, best1 = tuple(cum_sum.argsort()[-2:][::1])
                if cum_sum[best1] > opt.clf_threshold_final:
                    results.append(((i * opt.stride_len) + opt.sample_duration_clf, best1))
                    finished_prediction = False
                    prev_best1 = best1

                cum_sum = np.zeros(opt.n_classes_clf, )
            
            if active == False and prev_active == True:
                pre_predict = False

            prev_active = active
            

            if len(results) != 0:
                predicted = np.array(results)[:, 1]
                prev_best1 = -1
            else:
                predicted = []

            if len(results) > pre_len_result:
                cur_label = idx2label[predicted[pre_len_result]]
                pre_len_result = len(results)
        num_frame += 1
    elapsedTime = time.time() - t1
    return cur_label, elapsedTime, total_frames


def getFiles(dir, suffix):  # 查找根目录，文件后缀
    res = []
    for root, directory, files in os.walk(dir):  # =>当前根,根下目录,目录下的文件
        for filename in files:
            name, suf = os.path.splitext(filename)  # =>文件名,文件后缀
            if suf == suffix:
                res.append(filename)  # =>吧一串字符串组合成路径
    return res


with open("./res/ui_opt.json", 'r') as opt_file:
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

dir = "./TestGesture/subject1"
df1 = get_dataFrame(dir)

dir = "./TestGesture/subject2"
df2 = get_dataFrame(dir)

dir = "./TestGesture/subject3"
df3 = get_dataFrame(dir)


df1.to_csv("./TestGesture/continuous/subject1_"+str(opt.clf_threshold_final) + "_"+str(step)+".csv", index=False)
df2.to_csv("./TestGesture/continuous/subject2_"+str(opt.clf_threshold_final) + "_"+str(step)+ ".csv", index=False)
df3.to_csv("./TestGesture/continuous/subject3_"+str(opt.clf_threshold_final) + "_"+str(step)+ ".csv", index=False)