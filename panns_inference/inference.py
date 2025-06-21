import os
import numpy as np
import argparse
import librosa
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import urllib.request

from .pytorch_utils import move_data_to_device
from .models import Cnn14, Cnn14_DecisionLevelMax
from .config import labels, classes_num


def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)
        
        
def get_filename(path):
    path = os.path.realpath(path)
    na_ext = path.split('/')[-1]
    na = os.path.splitext(na_ext)[0]
    return na


class AudioTagging(object):
    def __init__(self, model=None, checkpoint_path=None, device='cuda', weights_only=False):
        """Audio tagging inference wrapper.
        """
        if checkpoint_path is not None and not os.path.exists(checkpoint_path):
            raise FileNotFoundError("Checkpoint doesn't exist at given path or path not found")

        if not checkpoint_path:
            checkpoint_path='{}/panns_data/Cnn14_DecisionLevelMax.pth'.format(str(Path.home()))
            
            if not os.path.exists(checkpoint_path) or os.path.getsize(checkpoint_path) < 3e8:
                dpath = os.path.dirname(checkpoint_path)
                create_folder(dpath)
                print("Given path empty, downloading the checkpoint...")
                print(f"Downloading at {dpath}")
                zenodo_path = 'https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1'
                
                def download_progress_hook(block_num, block_size, total_size):
                    downloaded = block_num * block_size
                    percent = int(downloaded * 100 / total_size) if total_size > 0 else 0
                    print(f"\rDownloading: {percent}% ({downloaded // (1024 * 1024)} MB / {total_size // (1024 * 1024)} MB)", end='')

                try:
                    urllib.request.urlretrieve(zenodo_path, checkpoint_path, reporthook=download_progress_hook)
                    print("Download completed.")
                except Exception as e:
                    print(f"Download failed: {e}")
                
                else:
                    print('Given path empty, using checkpoint at Checkpoint path: {}'.format(checkpoint_path))


        else: print('Loading checkpoint, Checkpoint path: {}'.format(checkpoint_path))
        
       
       #checking device
        if device == 'cuda' and torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        
        self.labels = labels
        self.classes_num = classes_num

        # Model
        if model is None:
            self.model = Cnn14(sample_rate=32000, window_size=1024, 
                hop_size=320, mel_bins=64, fmin=50, fmax=14000, 
                classes_num=self.classes_num)
        else:
            self.model = model

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=weights_only)
        self.model.load_state_dict(checkpoint['model'])

        # Parallel
        if 'cuda' in str(self.device):
            self.model.to(self.device)
            print('GPU number: {}'.format(torch.cuda.device_count()))
            self.model = torch.nn.DataParallel(self.model)
        else:
            print('Using CPU.')

    def inference(self, audio):
        audio = move_data_to_device(audio, self.device)

        with torch.no_grad():
            self.model.eval()
            output_dict = self.model(audio, None)

        clipwise_output = output_dict['clipwise_output'].data.cpu().numpy()
        embedding = output_dict['embedding'].data.cpu().numpy()

        return clipwise_output, embedding


class SoundEventDetection(object):
    def __init__(self, model=None, checkpoint_path=None, device='cuda', interpolate_mode='nearest'):
        """Sound event detection inference wrapper.

        Args:
            model: None | nn.Module
            checkpoint_path: str
            device: str, 'cpu' | 'cuda'
            interpolate_mode, 'nearest' |'linear'
        """
        if checkpoint_path is not None and not os.path.exists(checkpoint_path):
            raise FileNotFoundError("Checkpoint doesn't exist at given path or path not found.")

        if not checkpoint_path:
            checkpoint_path='{}/panns_data/Cnn14_DecisionLevelMax.pth'.format(str(Path.home()))
           
            if not os.path.exists(checkpoint_path) or os.path.getsize(checkpoint_path) < 3e8:
                dpath = os.path.dirname(checkpoint_path)
                create_folder(dpath)
                print("Given path empty, downloading the checkpoint...")
                print(f"Downloading at {dpath}")
                zenodo_path = 'https://zenodo.org/record/3987831/files/Cnn14_DecisionLevelMax_mAP%3D0.385.pth?download=1'
                
                def download_progress_hook(block_num, block_size, total_size):
                    downloaded = block_num * block_size
                    percent = int(downloaded * 100 / total_size) if total_size > 0 else 0
                    print(f"\rDownloading: {percent}% ({downloaded // (1024 * 1024)} MB / {total_size // (1024 * 1024)} MB)", end='')

                try:
                    urllib.request.urlretrieve(zenodo_path, checkpoint_path, reporthook=download_progress_hook)
                    print("Download completed.")
                except Exception as e:
                    print(f"Download failed: {e}")
                
                else:
                    print('Given path empty, using checkpoint at Checkpoint path: {}'.format(checkpoint_path))

        else: print('Loading checkpoint, Checkpoint path: {}'.format(checkpoint_path))

        # check device availability
        if device == 'cuda' and torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        
        self.labels = labels
        self.classes_num = classes_num

        # Model
        if model is None:
            self.model = Cnn14_DecisionLevelMax(sample_rate=32000, window_size=1024, 
                hop_size=320, mel_bins=64, fmin=50, fmax=14000, 
                classes_num=self.classes_num, interpolate_mode=interpolate_mode)
        else:
            self.model = model
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])

        # Parallel
        if 'cuda' in str(self.device):
            self.model.to(self.device)
            print('GPU number: {}'.format(torch.cuda.device_count()))
            self.model = torch.nn.DataParallel(self.model)
        else:
            print('Using CPU.')

    def inference(self, audio):
        audio = move_data_to_device(audio, self.device)

        with torch.no_grad():
            self.model.eval()
            output_dict = self.model(
                input=audio, 
                mixup_lambda=None
            )
        # print(output_dict)
        framewise_output = output_dict['framewise_output'].data.cpu().numpy()

        return framewise_output
  
