import os
import glob
import json
import pandas as pd
import csv
import torch
from torch.autograd import Variable
from torch.nn import functional as F

from opts import parse_opts_online
from model import generate_model
from mean import get_mean, get_std
from spatial_transforms import *
from temporal_transforms import *
from target_transforms import ClassLabel
from dataset import get_online_data
from utils import  AverageMeter, LevenshteinDistance, Queue

import pdb
import time
import numpy as np
import datetime


def weighting_func(x):
    return (1 / (1 + np.exp(-0.2 * (x - 9))))


opt = parse_opts_online()


opt = parse_opts_online()


def load_models(opt):
    opt.resume_path = opt.resume_path_clf
    opt.pretrain_path = opt.pretrain_path_clf
    opt.sample_duration = opt.sample_duration_clf
    opt.model = opt.model_clf
    opt.model_depth = opt.model_depth_clf
    opt.width_mult = opt.width_mult_clf
    opt.modality = opt.modality_clf
    opt.resnet_shortcut = opt.resnet_shortcut_clf
    opt.n_classes = opt.n_classes_clf
    opt.n_finetune_classes = opt.n_finetune_classes_clf
    if opt.root_path != '':
        opt.video_path = os.path.join(opt.root_path, opt.video_path)
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

    print(opt)
    with open(os.path.join(opt.result_path, 'opts_clf.json'), 'w') as opt_file:
        json.dump(vars(opt), opt_file)

    torch.manual_seed(opt.manual_seed)
    classifier, parameters = generate_model(opt)

    if opt.resume_path:
        print('loading checkpoint {}'.format(opt.resume_path))
        checkpoint = torch.load(opt.resume_path, map_location=torch.device('cpu'))
#        assert opt.arch == checkpoint['arch']

        classifier.load_state_dict(checkpoint['state_dict'])

    print('Model 2 \n', classifier)
    pytorch_total_params = sum(p.numel() for p in classifier.parameters() if
                               p.requires_grad)
    print("Total number of trainable parameters: ", pytorch_total_params)

    return classifier


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
# cap = cv2.VideoCapture(opt.video)
cap = cv2.VideoCapture(0)
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
while cap.isOpened():
    key = cv2.waitKey(1)&0xFF
    if key == ord(' '):
        active = not active
    elif key == ord('q'):
        break
    t1 = time.time()
    ret, frame = cap.read()
    if num_frame == 0:
        cur_frame = cv2.resize(frame,(320,240))
        cur_frame = Image.fromarray(cv2.cvtColor(cur_frame,cv2.COLOR_BGR2RGB))
        cur_frame = cur_frame.convert('RGB')
        for i in range(opt.sample_duration):
            clip.append(cur_frame)
        clip = [spatial_transform(img) for img in clip]
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
    num_frame += 1


    ground_truth_array = np.zeros(opt.n_classes_clf + 1, )
    with torch.no_grad():
        inputs = Variable(inputs)
        if opt.modality_clf == 'RGB':
            inputs_clf = inputs[:, :, :, :, :]
        inputs_clf = torch.Tensor(inputs_clf.numpy()[:, :, ::2, :, :])
        # print(inputs_clf.size())
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
        # #pdb.set_trace()
        # #### State of the detector is checked here as detector act as a switch for the classifier
        # # print(np.argmax(outputs_clf))
        # if  np.argmax(outputs_clf) != opt.n_classes_clf-1:
        #     # Push the probabilities to queue
        #     myqueue_clf.enqueue(outputs_clf.tolist())
        #     passive_count = 0

        #     if opt.clf_strategy == 'raw':
        #         clf_selected_queue = outputs_clf
        #     elif opt.clf_strategy == 'median':
        #         clf_selected_queue = myqueue_clf.median
        #     elif opt.clf_strategy == 'ma':
        #         clf_selected_queue = myqueue_clf.ma
        #     elif opt.clf_strategy == 'ewma':
        #         clf_selected_queue = myqueue_clf.ewma

        # else:
        #     outputs_clf = np.zeros(opt.n_classes_clf, )
        #     # Push the probabilities to queue
        #     myqueue_clf.enqueue(outputs_clf.tolist())
        #     passive_count += 1
    
    # if passive_count >= opt.det_counter:
    #     active = False
    # else:
    #     active = True

    # one of the following line need to be commented !!!!
    if active:
        active_index += 1
        # cum_sum = ((cum_sum * (active_index - 1)) + (
        #             weighting_func(active_index) * clf_selected_queue)) / active_index  # Weighted Aproach
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
    elapsedTime = time.time() - t1
    fps = "(Playback) {:.1f} FPS".format(1/elapsedTime)

    if len(results) != 0:
        predicted = np.array(results)[:, 1]
        prev_best1 = -1
    else:
        predicted = []

    if len(results) > pre_len_result:
        cur_label = idx2label[predicted[pre_len_result]]
        pre_len_result = len(results)

    print('predicted classes: \t', [idx2label[i] for i in predicted])
    
    if active:
        cv2.putText(frame, "Active", (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (38, 0, 255), 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, "Inactive", (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (38, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, fps, (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (38, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, cur_label, (0, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (38, 0, 255), 1, cv2.LINE_AA)
    cv2.imshow("Result", frame)

    
cv2.destroyAllWindows()