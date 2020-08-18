import torch
import torch.utils.data as data
import random
import numpy as np

class ContextData(data.Dataset):
    def __init__(self, data, item_num, user_sessions, neg_num, is_training = True):
        super().__init__()
        self.features = data
        self.item_num = item_num
        self.all_items = list(range(item_num))
        self.user_sessions = user_sessions
        self.is_training = is_training
        self.num_ng = neg_num

    
    def __len__(self):
        return self.num_ng * len(self.features) if \
                self.is_training else len(self.features)
    
    def ng_sample(self):
        self.features_fill = self.features*self.num_ng
        
    def __getitem__(self, idx):
        user = self.features_fill[idx][0]
        inputs = torch.from_numpy(np.array(self.features_fill[idx][1]))
        pos_item = self.features_fill[idx][2]
        i_idx = random.randint(0, self.item_num-1)
        while self.all_items[i_idx] in self.user_sessions[user]:
            i_idx = random.randint(0, self.item_num-1)
        neg_item = self.all_items[i_idx] if self.is_training else features[idx][2]
        
        return user, inputs, pos_item, neg_item