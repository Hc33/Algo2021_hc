###
# This file will:
# 1. Generate and save Alexnet features in a given folder
# 2. preprocess Alexnet features using PCA and save them in another folder
###
import glob

from torchvision.transforms.transforms import ToTensor
from vgg import *
import numpy as np
import urllib
import torch
import cv2
import argparse
import time
import random
import difflib
from tqdm import tqdm
from einops import rearrange
from torchvision import transforms as trn
import os
from PIL import Image
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
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



def get_video_from_mp4(file, sampling_frames):
    """This function takes a mp4 video file as input and returns
    an array of frames in numpy format.

    Parameters
    ----------
    file : str
        path to mp4 video file
    sampling_frames : int
        how many frames to get when doing frame sampling.

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
    buf = []
    interval = int(float(frameCount)/float(sampling_frames))
    fc = 0 # buf下标指针
    for i in range(frameCount):
        ret = cap.grab()  # 1.捕获
        if ret and i%interval == 0:
            (ret, frame) = cap.retrieve() # 2.解码
            buf.append(frame[None])
            fc += 1 #下标后移
        if fc >= sampling_frames:
            break
    cap.release()
    assert len(buf) == sampling_frames, 'length of buf {} is not the same as sampling frames {}.'.format(len(buf), sampling_frames)
    return np.concatenate(buf, axis=0), len(buf)


def load_model(model_checkpoints):
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
    from torchvision.models.video import r2plus1d_18
    model = r2plus1d_18()
    model_file = model_checkpoints
    checkpoint = torch.load(model_file)
    model.load_state_dict(checkpoint)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    return model


def video_preprocess(video_arr, mean, std):
    """
    视频的预处理
    video_arr : 
    0. numpy.ndarray -> torch.tensor 
    1. dimension convert
    2. resize to 112  TODO
    3. [0,255] -> [0.0, 1.0] with dtype convert
    4. normalize with mean,std
    """
    # 0.to tensor
    video_tensor = torch.from_numpy(video_arr).to(torch.uint8)
    # 1. T,H,W,C -> C,T,H,W
    video_tensor = rearrange(video_tensor, 't h w c -> c t h w')

    # 2.resize
    from torchvision.transforms.functional import resize
    video_tensor = resize(video_tensor, [112, 112])

    # 3. [0,255] -> [0.0, 1.0]
    video_tensor = video_tensor.to(torch.float32)/255.0  
    
    # 4.normalize
    dtype = video_tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=video_tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=video_tensor.device)
    if mean.ndim == 1:
        mean = mean.view(-1,1,1,1)
    if std.ndim == 1:
        std = std.view(-1,1,1,1)
    video_tensor.sub_(mean).div_(std)

    return video_tensor[None]


def get_activations_and_save(model, video_list, activations_dir, sampling_frames=20):
    """This function generates Alexnet features and save them in a specified directory.

    Parameters
    ----------
    model :
        pytorch model : vgg.
    video_list : list
        the list contains path to all videos.
    activations_dir : str
        save path for extracted features.
    sampling_frames : int
        how many frames to skip when feeding into the network.

    """
    def get_activations(module, fea_in, fea_out): # hooker
        activations.append(fea_out)  # 获取特征
        
    for idx, child in enumerate(model.children()):  # 注册hooker
        if idx>=1 and idx<=4:  # 指定层
            child.register_forward_hook(hook=get_activations)

    dir_file = os.listdir(activations_dir)

    for video_file in tqdm(video_list):
        video_file_name = os.path.split(video_file)[-1].split(".")[0] # basename without suffix
        # res = difflib.get_close_matches(video_file_name, dir_file, 3, cutoff=0.6)  # 模糊匹配, 通过文件名判断是否已经提取过特征
        # if len(res) != 0:  # 若有文件，则跳过
        #     break
        vid, num_frames = get_video_from_mp4(video_file, sampling_frames) #
        assert num_frames == sampling_frames, "{} the number of frames is not {}!, is {}".format(video_file_name,sampling_frames,num_frames)
        input_vid = video_preprocess(vid, mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]) # 预处理

        activations = []
        if torch.cuda.is_available():
            input_vid = input_vid.cuda()
        x = model.forward(input_vid)
        assert len(activations) == 4, f'ERROR!, len is {len(activations)}'
        for layer in range(len(activations)):
            save_path = os.path.join(activations_dir, video_file_name+"_"+"layer" + "_" + str(layer+1) + ".npy")
            np.save(save_path, activations[layer].data.cpu().numpy().ravel())


def do_PCA_and_save(activations_dir, save_dir, n_components):
    """This function preprocesses Neural Network features using PCA and save the results
    in  a specified directory

    Parameters
    ----------
    activations_dir : str
        save path for extracted features.
    save_dir : str
        save path for extracted PCA features.

    """
    layers = ['layer_1','layer_2','layer_3','layer_4']  # 'layer_5','layer_6','layer_7','layer_8'
    n_components = n_components
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
        x_test = x[1000:,:]  # 1k之后(102例)得到作为测试集

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
    parser.add_argument('-vdir','--video_data_dir', help='video data directory',default = '/home/hc/Algonauts2021/data/AlgonautsVideos268_All_30fpsmax/', type=str)
    parser.add_argument('-sdir','--save_dir', help='saves processed features',default = './features_from_models/video', type=str)
    args = vars(parser.parse_args())

    save_dir=args['save_dir']  # net中层特征的 存储路径
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
    checkpoint_path = "./model_save/r2plus1d_18.pth"  # 预训练模型下载
    # if not os.path.exists(checkpoint_path):
    #     url = "https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth"
    #     urllib.request.urlretrieve(url, checkpoint_path)
    model = load_model(checkpoint_path)

    # get and save activations
    activations_dir = os.path.join(save_dir)
    if not os.path.exists(activations_dir):
        os.makedirs(activations_dir)
    print("-------------Saving activations ----------------------------")
    get_activations_and_save(model, video_list, activations_dir, sampling_frames=10) # 采集帧数20

    # preprocessing using PCA and save
    n_components = 100
    pca_dir = os.path.join(save_dir, f'pca_{n_components}')
    print("-------------performing  PCA----------------------------")
    do_PCA_and_save(activations_dir, pca_dir, n_components)


if __name__ == "__main__":
    main()

