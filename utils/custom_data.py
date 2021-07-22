"""
数据集
"""
from sys import breakpointhook
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from decord import VideoReader
from decord import cpu
from .helper import load_dict
import torch
import os
import numpy as np
from torchvision import transforms as trn



resize_normalize = trn.Compose([
            trn.Resize((224,224)),
            trn.ToTensor(),
            trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])  # imageNet 


class CustomDataset(Dataset):
    def __init__(self, video_list, fmri_nda, mode="train"):
        """
        mode -> train or val
        """
        self.video_list = video_list
        self.fmri_nda = fmri_nda
        self.mode = mode
        if fmri_nda is not None:
            assert len(video_list) == fmri_nda.shape[0], "Error!"

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, index):
        # 1.selecting video and sampling frames, preprocessing.
        video_f = self.video_list[index]
        vr = VideoReader(video_f)
        total_frames = len(vr)
        seg_ind = total_frames//2 # 取序列中间那张图像
        image = Image.fromarray(vr[seg_ind].asnumpy())
        image = resize_normalize(image)
        # 2.get fmri
        if self.mode == "train" or self.mode == "val":
            label = self.fmri_nda[index]
            return image, label
        return image


if __name__ == "__main__":
    pass
    # from glob import glob
    # video_dir = r"/home/hc/Algonauts2021/data/AlgonautsVideos268_All_30fpsmax"
    # video_list = glob(video_dir + '/*.mp4')
    # video_list.sort()
    # video_list = video_list[:1000]
    
    # sub = "sub01"
    # roi = "LOC"

    # ds = CustomDataset(video_list, sub, roi, "test")
    # a = ds[1]
    # print(a.shape)
    # # print(b.shape)
    # dl = DataLoader(ds, batch_size=4, shuffle=True, num_workers=4)
    # for b in dl:
    #     print(b.shape)
    #     print(b[1].shape)
    #     break
