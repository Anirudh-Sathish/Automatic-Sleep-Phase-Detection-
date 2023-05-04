import torch.nn as nn
import torch.nn.functional as F

from sleepyco import SleePyCoBackbone

from classifiers import get_classifier


last_chn_dict = {
    'SleePyCo': 256
}


class MainModel(nn.Module):
    
    def __init__(self):

        super(MainModel, self).__init__()

        self.bb_cfg = {
        "name": "SleePyCo",
        "init_weights": False,
        "dropout": False}

        self.training_mode = "pretrain"

        self.feature = SleePyCoBackbone()
        proj_dim = 128

        if self.training_mode == 'pretrain':
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(last_chn_dict["SleePyCo"], proj_dim),
                nn.ReLU(inplace=True),
                nn.Linear(proj_dim, proj_dim)
                )
    def get_max_len(self, features):
        len_list = []
        for feature in features:
            len_list.append(feature.shape[1])
        
        return max(len_list)

    def forward(self, x):
        outputs = []
        features = self.feature(x)
        
        for feature in features:
                
            if self.training_mode == 'pretrain':
                outputs.append(F.normalize(self.head(feature)))
            elif self.training_mode in ['scratch', 'fullfinetune', 'freezefinetune']:
                feature = feature.transpose(1, 2)
                output = self.classifier(feature)
                outputs.append(output)    # (B, L, H)
            else:
                raise NotImplementedError
            
        return outputs