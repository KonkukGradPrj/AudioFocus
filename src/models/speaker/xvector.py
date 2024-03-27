import torch
import torch.nn as nn
import torch.nn.functional as F

import torchaudio
import glob
import os

from collections import OrderedDict
# from src.models.speaker.base import BaseSpeaker
from abc import *
class BaseSpeaker(nn.Module, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()
        self.model = None

    @abstractmethod
    def extract_feature(self, wav):
        pass

# https://github.com/manojpamk/pytorch_xvectors?tab=readme-ov-file#pretrained-model

class XVector(BaseSpeaker):
    name = 'xvector'
    def __init__(self, p_dropout=0, numSpkrs=7323):
        super(XVector, self).__init__()
        self.tdnn1 = nn.Conv1d(in_channels=30, out_channels=512, kernel_size=5, dilation=1)
        self.bn_tdnn1 = nn.BatchNorm1d(512, momentum=0.1, affine=False)
        self.dropout_tdnn1 = nn.Dropout(p=p_dropout)

        self.tdnn2 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=5, dilation=2)
        self.bn_tdnn2 = nn.BatchNorm1d(512, momentum=0.1, affine=False)
        self.dropout_tdnn2 = nn.Dropout(p=p_dropout)

        self.tdnn3 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=7, dilation=3)
        self.bn_tdnn3 = nn.BatchNorm1d(512, momentum=0.1, affine=False)
        self.dropout_tdnn3 = nn.Dropout(p=p_dropout)

        self.tdnn4 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, dilation=1)
        self.bn_tdnn4 = nn.BatchNorm1d(512, momentum=0.1, affine=False)
        self.dropout_tdnn4 = nn.Dropout(p=p_dropout)

        self.tdnn5 = nn.Conv1d(in_channels=512, out_channels=1500, kernel_size=1, dilation=1)
        self.bn_tdnn5 = nn.BatchNorm1d(1500, momentum=0.1, affine=False)
        self.dropout_tdnn5 = nn.Dropout(p=p_dropout)

        self.fc1 = nn.Linear(3000,512)
        self.bn_fc1 = nn.BatchNorm1d(512, momentum=0.1, affine=False)
        self.dropout_fc1 = nn.Dropout(p=p_dropout)

        # self.fc2 = nn.Linear(512,512)
        # self.bn_fc2 = nn.BatchNorm1d(512, momentum=0.1, affine=False)
        # self.dropout_fc2 = nn.Dropout(p=p_dropout)
        # self.fc3 = nn.Linear(512,numSpkrs)
        self.load_pretrained()

    def extract_feature(self, wav):
        # MFCC
        x = self.transform(wav)

        # extract
        x = self.dropout_tdnn1(self.bn_tdnn1(F.relu(self.tdnn1(x))))
        x = self.dropout_tdnn2(self.bn_tdnn2(F.relu(self.tdnn2(x))))
        x = self.dropout_tdnn3(self.bn_tdnn3(F.relu(self.tdnn3(x))))
        x = self.dropout_tdnn4(self.bn_tdnn4(F.relu(self.tdnn4(x))))
        x = self.dropout_tdnn5(self.bn_tdnn5(F.relu(self.tdnn5(x))))

        stats = torch.cat((x.mean(dim=2), x.std(dim=2)), dim=1)
        x = self.dropout_fc1(self.bn_fc1(F.relu(self.fc1(stats))))

        return x

    def load_pretrained(self, path='/home/hyeons/workspace/AudioFocus/data/weights/xvec_preTrained.tar'):
        modelFile = max(glob.glob(path), key=os.path.getctime)
        checkpoint = torch.load(modelFile, map_location=torch.device('cuda'))['model_state_dict']
        self.load_state_dict(checkpoint)
        self.transform = torchaudio.transforms.MFCC()



if __name__ == '__main__':
    model = XVector()
    model.load_pretrained()
    model = model.cuda()
    model.eval()
    print(model)