import torch
import torch.nn as nn

class BPR(nn.Module):
    def __init__(self, args): 
        super().__init__()
        """
        user_num: number of users;
        item_num: number of items;
        factor_num: number of predictive factors.
        """
        self.user_num = args.num_users
        self.item_num = args.num_items
        self.factor_num = args.latent_dim
        
        self.user_embedding = nn.Embedding(self.user_num, self.factor_num)
        self.item_embedding = nn.Embedding(self.item_num, self.factor_num)
        
        nn.init.normal_(self.user_embedding.weight, 0, 0.01)
        nn.init.normal_(self.item_embedding.weight, 0, 0.01)
    
    def forward(self, user, pos_item, neg_item):
        
        user_embed = self.user_embedding(user)
        pos_item_embed = self.item_embedding(pos_item)
        neg_item_embed = self.item_embedding(neg_item)
        
        pos_pref_score = torch.mul(user_embed, pos_item_embed).sum(dim=-1)
        neg_pref_score = torch.mul(user_embed, neg_item_embed).sum(dim=-1)
        
        return pos_pref_score, neg_pref_score
    
    def get_item_embed(self, item):
        
        item_embed = self.item_embedding(item)
        
        return item_embed
    
    def get_user_embed(self, user):
        
        user_embed = self.user_embedding(user)
        
        return user_embed
    
    def get_user_weight(self):
        return self.user_embedding.weight
    
    def get_item_weight(self):
        return self.item_embedding.weight