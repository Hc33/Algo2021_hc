import nibabel as nib
import pickle


def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

def load_dict(filename_):
    with open(filename_, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        ret_di = u.load()
        #print(p)
        #ret_di = pickle.load(f)
    return ret_di

def saveasnii(brain_mask, nii_save_path, nii_data):
    img = nib.load(brain_mask) # need affine, header
    print(img.shape)
    nii_img = nib.Nifti1Image(nii_data, img.affine, img.header)
    nib.save(nii_img, nii_save_path)


if __name__ == "__main__":
    a = r"/home/hc/Algonauts2021/Algo2021_hc/activations/resnet/swsl_resnext101_32x8d/pca_25/test_layer_3.npy"
    import numpy as np
    nda = np.load(a)
    import pdb
    pdb.set_trace()
