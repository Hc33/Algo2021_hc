###
# This file will:
# 1. Generate and save Alexnet features in a given folder
# 2. preprocess Alexnet features using PCA and save them in another folder
###
import glob
from alexnet import *
import numpy as np
import urllib
import torch
import cv2
import argparse
import time
import difflib
import random
from tqdm import tqdm
from torchvision import transforms as trn
import os
from PIL import Image
from sklearn.preprocessing import StandardScaler
from torch.autograd import Variable as V
from sklearn.decomposition import PCA, IncrementalPCA

seed = 42
# Torch RNG
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# Python RNG
np.random.seed(seed)
random.seed(seed)



def get_video_from_mp4(file, sampling_rate):
    """This function takes a mp4 video file as input and returns
    an array of frames in numpy format.

    Parameters
    ----------
    file : str
        path to mp4 video file
    sampling_rate : int
        how many frames to skip when doing frame sampling.

    Returns
    -------
    video: np.array

    num_frames: int
        number of frames extracted

    """
    cap = cv2.VideoCapture(file)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    buf = np.empty((int(frameCount / sampling_rate), frameHeight,
                   frameWidth, 3), np.dtype('uint8'))
    fc = 0 # 
    ret = True
    while fc < frameCount and ret:
        fc += 1
        if fc % sampling_rate == 0:
            (ret, buf[int((fc - 1) / sampling_rate)]) = cap.read()

    cap.release()

    return np.expand_dims(buf, axis=0),int(frameCount / sampling_rate)


def load_alexnet(model_checkpoints):
    """This function initializes an Alexnet and load
    its weights from a pretrained model
    ----------
    model_checkpoints : str
        model checkpoints location.

    Returns
    -------
    model
        pytorch model of alexnet

    """


    model = alexnet()
    model_file = model_checkpoints
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    model_dict =["conv1.0.weight", "conv1.0.bias", "conv2.0.weight", "conv2.0.bias", "conv3.0.weight", "conv3.0.bias", "conv4.0.weight", "conv4.0.bias", "conv5.0.weight", "conv5.0.bias", "fc6.1.weight", "fc6.1.bias", "fc7.1.weight", "fc7.1.bias", "fc8.1.weight", "fc8.1.bias"]
    state_dict={}
    i=0
    for k,v in checkpoint.items():
        state_dict[model_dict[i]] =  v
        i+=1

    model.load_state_dict(state_dict)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    return model


def get_activations_and_save(model, video_list, activations_dir, sampling_rate = 4):
    """This function generates Alexnet features and save them in a specified directory.

    Parameters
    ----------
    model :
        pytorch model : alexnet.
    video_list : list
        the list contains path to all videos.
    activations_dir : str
        save path for extracted features.
    sampling_rate : int
        how many frames to skip when feeding into the network.

    """
    # 一些预处理
    centre_crop = trn.Compose([
            trn.ToPILImage(),
            trn.Resize((224,224)),
            trn.ToTensor(),
            trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    dir_file = os.listdir(activations_dir)

    for video_file in tqdm(video_list):
        video_file_name = os.path.split(video_file)[-1].split(".")[0] # basename without suffix
        res = difflib.get_close_matches(video_file_name, dir_file, 3, cutoff=0.6)  # 模糊匹配, 判断是否已经提取过特征
        if len(res) != 0:
            break
        vid, num_frames = get_video_from_mp4(video_file, sampling_rate) # vid [1, 22, 268, 268, 3]
        activations = [] # list
        for frame in range(num_frames): # 每一帧
            img =  vid[0,frame,:,:,:]
            input_img = V(centre_crop(img).unsqueeze(0)) # 预处理
            if torch.cuda.is_available():
                input_img=input_img.cuda()
            x = model.forward(input_img) # 送进模型，x是每一层的feature
            for i, feat in enumerate(x): # 
                if frame==0:
                    activations.append(feat.data.cpu().numpy().ravel()) # flatten
                else:
                    activations[i] = activations[i] + feat.data.cpu().numpy().ravel() # 每一帧的累加
        for layer in range(len(activations)):
            save_path = os.path.join(activations_dir, video_file_name+"_"+"layer" + "_" + str(layer+1) + ".npy")
            np.save(save_path,activations[layer]/float(num_frames))


def do_PCA_and_save(activations_dir, save_dir):
    """This function preprocesses Neural Network features using PCA and save the results
    in  a specified directory
.
    Parameters
    ----------
    activations_dir : str
        save path for extracted features.
    save_dir : str
        save path for extracted PCA features.

    """
    layers = ['layer_1','layer_2','layer_3','layer_4','layer_5','layer_6','layer_7','layer_8']
    n_components = 100
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for layer in tqdm(layers):
        activations_file_list = glob.glob(activations_dir +'/*'+layer+'.npy')
        activations_file_list.sort()
        feature_dim = np.load(activations_file_list[0])
        x = np.zeros((len(activations_file_list),feature_dim.shape[0]))
        for i, activation_file in enumerate(activations_file_list):
            temp = np.load(activation_file)
            x[i,:] = temp
        x_train = x[:1000,:]  # 前1k例作为训练集
        x_test = x[1000:,:]  # 1k之后得到作为测试集

        start_time = time.time()
        x_test = StandardScaler().fit_transform(x_test)
        x_train = StandardScaler().fit_transform(x_train)
        ipca = PCA(n_components=n_components) #, batch_size=20)
        ipca.fit(x_train)

        x_train = ipca.transform(x_train)
        x_test = ipca.transform(x_test)
        train_save_path = os.path.join(save_dir,"train_"+layer)
        test_save_path = os.path.join(save_dir,"test_"+layer)
        np.save(train_save_path,x_train)
        np.save(test_save_path,x_test)


def main():
    parser = argparse.ArgumentParser(description='Feature Extraction from Alexnet and preprocessing using PCA')
    parser.add_argument('-vdir','--video_data_dir', help='video data directory',default = '/openbayes/input/input0/AlgonautsVideos268_All_30fpsmax/', type=str)
    parser.add_argument('-sdir','--save_dir', help='saves processed features',default = './alexnet', type=str)
    args = vars(parser.parse_args())

    save_dir=args['save_dir']  # alexnet中每一层特征的 存储路径
    if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    video_dir = args['video_data_dir']  # 视频数据存放路径
    video_list = glob.glob(video_dir + '/*.mp4')
    video_list.sort()
    print('Total Number of Videos: ', len(video_list))

    # load Alexnet
    # Download pretrained Alexnet from:
    # https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth
    # and save in the current directory
    checkpoint_path = "./model_save/alexnet.pth"  # 预训练模型下载
    if not os.path.exists(checkpoint_path):
        url = "https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth"
        urllib.request.urlretrieve(url, checkpoint_path)
    model = load_alexnet(checkpoint_path)

    # get and save activations
    activations_dir = os.path.join(save_dir)
    if not os.path.exists(activations_dir):
        os.makedirs(activations_dir)
    print("-------------Saving activations ----------------------------")
    get_activations_and_save(model, video_list, activations_dir) # 采样率4

    # preprocessing using PCA and save
    pca_dir = os.path.join(save_dir, 'pca_100')
    print("-------------performing  PCA----------------------------")
    do_PCA_and_save(activations_dir, pca_dir)


if __name__ == "__main__":
    main()

