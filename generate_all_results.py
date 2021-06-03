from subprocess import Popen
import numpy as np
import os
import glob
import random
import argparse
import itertools
import nibabel as nib
from nilearn import plotting
from tqdm import tqdm
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import torch
import time
import pickle
import pandas as pd
from tqdm import tqdm
from utils.helper import save_dict, load_dict
from perform_encoding import encoding



def main():
    """
    perform_encoding.py for all ROI with all subjects.
    """
    parser = argparse.ArgumentParser(description='Generates predictions for all subs all ROIs for a given track')
    
    parser.add_argument('-t','--track', help='mini_track for all ROIs, full_track for whole brain (WB)', default = 'mini_track', type=str)
    parser.add_argument('-ad','--activation_dir',help='directory containing DNN activations',default = './alexnet/', type=str)
    parser.add_argument('-fd','--fmri_dir',help='directory containing fMRI activity', default = '/openbayes/input/input0/participants_data_v2021', type=str)
    # modified args
    parser.add_argument('-model','--model',help='model name under which predicted fMRI activity will be saved', default = 'alexnet_devkit', type=str)
    parser.add_argument('-m', '--mode', help='test or val, val returns mean correlation by using 10% of training data for validation', default='val', type=str)
    parser.add_argument('-v','--visualize',help='visualize whole brain results in MNI space or not, only available if --track full_track', default = 'False', type=str)
    parser.add_argument('-l','--layer',help='layer from which activations will be used to train and predict fMRI activity', default = 'layer_5', type=str)
    parser.add_argument('-b', '--batch_size',help=' number of voxel to fit at one time in case of memory constraints', default = 1000, type=int)
    parser.add_argument('-pca', '--pca_components', help='n_components of PCA', default=100, type=int)
    args = vars(parser.parse_args())

    track = args['track']
    ad = args["activation_dir"]
    model = args['model']
    layer = args['layer']
    mode = args['mode']
    fmri_dir = args['fmri_dir']
    vr = args['visualize']
    bs = args['batch_size']
    n_pca = args['pca_components']
    
    if track == 'full_track':
        ROIs = ['WB']
    else:
        ROIs = ['LOC','FFA','STS','EBA','PPA','V1','V2','V3','V4']

    num_subs = 10
    subs=[]  
    for s in range(num_subs):
        subs.append('sub'+str(s+1).zfill(2))
    
    if mode== 'val':
        corr_ = {}
        mse_ = {}
        for roi in ROIs:  # roi-ROI
            current_roi_corr = []
            current_roi_mse = []
            for sub in subs: #
            #   cmd_string = 'python perform_encoding.py' + ' --roi ' + roi + ' -ad ' + activation_dir + ' --sub ' + sub + ' -fd ' + fmri_dir + ' -m ' + mode + ' -v ' + visualize_results + ' -l ' + layer + ' -model ' + model + ' -pca '+ n_pca
                print("----------------------------------------------------------------------------")
                print("----------------------------------------------------------------------------")
                print ("Starting ROI: ", roi, "sub: ",sub)

                score, mse = encoding(activation_dir=ad, model=model,layer=layer, sub=sub, 
                                ROI=roi, mode=mode, fmri_dir=fmri_dir, visualize_results=vr,
                                batch_size=bs, n_pca=n_pca)
                current_roi_corr.append(score)
                current_roi_mse.append(mse)
            #   os.system(cmd_string)
                print ("Completed ROI: ", roi, "sub: ",sub)
                print("----------------------------------------------------------------------------")
                print("----------------------------------------------------------------------------")
            corr_[roi] = current_roi_corr
            mse_[roi] = current_roi_mse
        # take a record with score and mse
        df_corr = pd.DataFrame(corr_, index=subs)
        df_mse = pd.DataFrame(mse_, index=subs)
        
        # save metrics to log_path
        log_path = f"./log/{model}/{layer}/{track}"
        writer = pd.ExcelWriter(os.path.join(log_path, f'Metrics_bs{bs}_pca{n_pca}.xlsx'))
        df_corr.to_excel(writer, sheet_name='Correlation Score')
        df_mse.to_excel(writer, sheet_name='MSE')
        writer.save()
        writer.close()
    
    else:
        for roi in ROIs:
            for sub in subs:
                # cmd_string = 'python perform_encoding.py' + ' --roi ' + roi + ' -ad ' + ad + ' --sub ' + sub + ' -fd ' + fmri_dir + ' -m ' + mode + ' -v ' + vr + ' -l ' + layer + ' -model ' + model + ' -pca '+ str(n_pca)
                print("----------------------------------------------------------------------------")
                print("----------------------------------------------------------------------------")
                print ("Starting ROI: ", roi, "sub: ",sub)
                encoding(activation_dir=ad, model=model,layer=layer, sub=sub, 
                                ROI=roi, mode=mode, fmri_dir=fmri_dir, visualize_results=vr,
                                batch_size=bs, n_pca=n_pca)
                print ("Completed ROI: ", roi, "sub: ",sub)
                print("----------------------------------------------------------------------------")
                print("----------------------------------------------------------------------------")

if __name__ == "__main__":
    main()

