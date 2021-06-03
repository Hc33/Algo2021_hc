import os
from urllib.request import urlretrieve
import argparse


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
}

def cbk(a, b, c):    
    ''' call back func       
        a: 已经下载的数据块        
        b: 数据块的大小        
        c: 远程文件的大小    
    '''    
    per = 100.0 * a * b / c    
    if per > 100:        
        per = 100    
    print('%.1f%% of %.2fM' % (per,c/(1024*1024)))

def main():
    """
    Download pretrained models.
    """
    parser = argparse.ArgumentParser(description='Download pretrained models and save in target dir.')
    parser.add_argument('-m', '--model', help='Model which you want', default='alexnet', type=str)
    parser.add_argument('-sd','--save_dir', help='Dir to save pretrained models.', default = './', type=str)
    args = vars(parser.parse_args())
    
    model = args['model']
    save_dir = args['save_dir']

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    checkpoint_path = os.path.join(save_dir, model+".pth")  # 预训练模型下载
    if not os.path.exists(checkpoint_path):
        url = model_urls[model]
        urlretrieve(url, checkpoint_path, cbk)


if __name__ == "__main__":
    main()
    
