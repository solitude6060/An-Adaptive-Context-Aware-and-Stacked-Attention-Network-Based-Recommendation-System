import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
from torch.autograd import Variable

class multi_head_project(nn.Module):
    def __init__(self, head_num, latent_dim, isBN = False):
        super().__init__()
        self.head_num = head_num
        self.factor_num = latent_dim
        for i in range(self.head_num):
            name = 'head_{}'.format(i)
            setattr(self, name, nn.Linear(self.factor_num, self.factor_num, bias=False))
  
        self.user_fc = nn.Linear(self.factor_num, self.factor_num, bias=False)
        self.isBN = isBN
        if self.isBN : 
            self.BN = nn.BatchNorm1d(self.factor_num)
        
    def forward(self, user_embed, item_embed):
        proj_emb = []
        weights = []
        
        for i in range(self.head_num):
            name = 'head_{}'.format(i)
            x = getattr(self, name)(user_embed)
            proj_emb.append(x.view(-1, 1, self.factor_num))
            weights.append(torch.mul(x, item_embed).sum(-1).view(x.shape[0], 1, 1))
            
        proj_emb = torch.cat(proj_emb, dim=1)
        weights = F.softmax(torch.cat(weights, dim=1), dim=1)
        weights_sum = torch.mul(weights, proj_emb).sum(dim=1)
        
        non_linear_emb = self.user_fc(weights_sum)
        non_linear_emb = torch.tanh(non_linear_emb)
        if self.isBN:
            non_linear_emb = self.BN(non_linear_emb)
        weights_proj_emb = non_linear_emb # + user_embed
        
        return weights_proj_emb # non_linear_emb, user_embed

# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------

class AttentiveRec(nn.Module):
    def __init__(self, args, is_pretrained_item_weight, bpr_item_weight):
        super().__init__()
        self.user_num = args.num_users
        self.item_num = args.num_items
        self.factor_num = args.latent_dim
        
        self.user_embedding = nn.Embedding(self.user_num, self.factor_num)
        nn.init.normal_(self.user_embedding.weight, 0, 0.01)
        # self.user_embedding.weight = nn.Parameter(torch.from_numpy(bpr_user_weight))
        
        self.item_embedding = nn.Embedding(self.item_num, self.factor_num)
        if is_pretrained_item_weight:
            # item with pretrain weight
            print("Using item pretrain")
            self.item_embedding.weight = nn.Parameter(torch.from_numpy(bpr_item_weight))
        else:
            nn.init.normal_(self.item_embedding.weight, 0, 0.01)
        

# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------

class stacked_AttRec(AttentiveRec):
    def __init__(self, args, stack_num, head_num, time_info_mode = 0, window_size = 5, is_pretrained_item_weight = True, bpr_item_weight=None,  isItemGrad = True, isUserBN = False): 
        super().__init__(args, is_pretrained_item_weight, bpr_item_weight)
        
        # update item embedding or not
        self.item_embedding.weight.requires_grad = isItemGrad
        self.stack_num = stack_num
        self.head_num = head_num
        self.isConv = False
        self.isPosi = False
        self.time_info_mode = time_info_mode

        if self.time_info_mode == 0:
            self.isConv = False
            self.isPosi = False
        elif self.time_info_mode == 1:
            self.isConv = False
            self.isPosi = True
        elif self.time_info_mode == 2:
            self.isConv = True
            self.isPosi = False
        elif self.time_info_mode == 3 or self.time_info_mode == 4 or self.time_info_mode == 5:
            self.isConv = True
            self.isPosi = True
        
        for i in range(self.stack_num):
            name = 'user_attention{}'.format(i)
            setattr(self, name, multi_head_project(head_num = self.head_num, latent_dim = self.factor_num, isBN = isUserBN))
        
        if self.isConv:
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), stride=1, padding=(0, 1))
            self.conv2 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(3, 3), stride=1, padding=(0, 1))
        
        if self.isPosi:
            self.position_enc = torch.tensor([[pos / torch.pow(10000, torch.tensor(2 * (j // 2) /self.factor_num)) \
                                               for j in range(self.factor_num)] for pos in range(window_size)])
            self.position_enc[1:, 0::2] = torch.sin(self.position_enc[1:, 0::2]) # dim 2i
            self.position_enc[1:, 1::2] = torch.cos(self.position_enc[1:, 1::2]) # dim 2i+1
            self.position_enc = self.position_enc.type(torch.FloatTensor).cuda()
            self.position_enc.requires_grad = False
        
        if self.time_info_mode == 5:
            self.time_fc = nn.Linear(self.factor_num*2, self.factor_num, bias=True)
        
        self.model_description = "{}x{}_stacked_attention_isItemGrad_{}_isConv_{}_isPosi_{}_isUserBN_{}_timeInfo_{}_window_size_{}".format(self.head_num, self.stack_num, isItemGrad, self.isConv, self.isPosi, isUserBN, time_info_mode, window_size)
        
    def forward(self, user, items, pos_item, neg_item):
        
        user_embed = self.user_embedding(user)
        items_embed= self.item_embedding(items)

        pos_item_embed = self.item_embedding(pos_item)
        neg_item_embed = self.item_embedding(neg_item)
        
        if self.time_info_mode != 5:
            # Add item position encoding
            if self.time_info_mode == 1:
                items_embed = items_embed + self.position_enc

            # Add conv item time feature
            if self.time_info_mode == 2:
                kernel_items_embed = items_embed.view(items_embed.shape[0], 1, items_embed.shape[1], items_embed.shape[-1])
                conv_items1 = self.conv1(kernel_items_embed)
                conv_items2 = self.conv2(conv_items1)
                conv_items = conv_items2.view(conv_items2.shape[0], 1, conv_items2.shape[-1])
                items_embed = items_embed + conv_items
            
            # Add both pos and cnn at same time
            if self.time_info_mode == 3:
                kernel_items_embed = items_embed.view(items_embed.shape[0], 1, items_embed.shape[1], items_embed.shape[-1])
                conv_items1 = self.conv1(kernel_items_embed)
                conv_items2 = self.conv2(conv_items1)
                conv_items = conv_items2.view(conv_items2.shape[0], 1, conv_items2.shape[-1])
                items_embed = items_embed + conv_items
                items_embed = items_embed + self.position_enc

            # Add pos first then cnn
            if self.time_info_mode == 4:
                items_embed = items_embed + self.position_enc
                kernel_items_embed = items_embed.view(items_embed.shape[0], 1, items_embed.shape[1], items_embed.shape[-1])
                conv_items1 = self.conv1(kernel_items_embed)
                conv_items2 = self.conv2(conv_items1)
                conv_items = conv_items2.view(conv_items2.shape[0], 1, conv_items2.shape[-1])
                items_embed = items_embed + conv_items
            
            last_item = items_embed[:,-1,:]
            
            # item attention
            weight = torch.mul(items_embed, user_embed.view(user_embed.shape[0],1,user_embed.shape[1])).sum(-1).clamp(min=1e-12)
            item_weight = F.softmax(input=weight, dim=0)
            weighted_item = torch.mul(items_embed.transpose(1,2), item_weight.view(item_weight.shape[0],1,item_weight.shape[1]))
            context_item = weighted_item.transpose(1,2).sum(1)
            
            # user attention
            if self.stack_num == 0:
                # user_rep = user_embed + context_item
                user_rep = context_item
            for i in range(self.stack_num):
                name = 'user_attention{}'.format(i)
                if i == 0:
                    user_rep = getattr(self, name)(user_embed, context_item)
                    user_rep = user_rep + user_embed
                else:
                    prev_rep = user_rep.copy()
                    user_rep = getattr(self, name)(user_rep, context_item)
                    user_rep = user_rep + prev_rep
            
            # user_rep = user_rep + user_embed
                    
            pos_pref_score = torch.mul(user_rep, pos_item_embed).sum(dim=-1)
            neg_pref_score = torch.mul(user_rep, neg_item_embed).sum(dim=-1)
        
        # concat pos result and cnn result then pass fully connection layer
        else:
            posi_items_embed = items_embed + self.position_enc
            posi_last_item = posi_items_embed[:,-1,:]
            
            # posi item attention
            posi_weight = torch.mul(posi_items_embed, user_embed.view(user_embed.shape[0],1,user_embed.shape[1])).sum(-1).clamp(min=1e-12)
            posi_item_weight = F.softmax(input=posi_weight, dim=0)
            posi_weighted_item = torch.mul(posi_items_embed.transpose(1,2), posi_item_weight.view(posi_item_weight.shape[0],1,posi_item_weight.shape[1]))
            posi_context_item = posi_weighted_item.transpose(1,2).sum(1)
            
            # posi_user attention
            if self.stack_num == 0:
                # user_rep = user_embed + context_item
                posi_user_rep = posi_context_item
            for i in range(self.stack_num):
                name = 'user_attention{}'.format(i)
                if i == 0:
                    posi_user_rep = getattr(self, name)(user_embed, posi_context_item)
                    posi_user_rep = posi_user_rep + user_embed
                else:
                    posi_prev_rep = posi_user_rep
                    posi_user_rep = getattr(self, name)(posi_user_rep, posi_context_item)
                    posi_user_rep = posi_user_rep + posi_prev_rep
            # get posi_user_rep in posi final
            # - - - - - - - - - - - - - - - - 
            # cnn part
            kernel_items_embed = items_embed.view(items_embed.shape[0], 1, items_embed.shape[1], items_embed.shape[-1])
            conv_items1 = self.conv1(kernel_items_embed)
            conv_items2 = self.conv2(conv_items1)
            conv_items = conv_items2.view(conv_items2.shape[0], 1, conv_items2.shape[-1])
            cnn_items_embed  = items_embed + conv_items
            
            cnn_last_item = cnn_items_embed[:,-1,:]
            
            # cnn item attention
            cnn_weight = torch.mul(cnn_items_embed, user_embed.view(user_embed.shape[0],1,user_embed.shape[1])).sum(-1).clamp(min=1e-12)
            cnn_item_weight = F.softmax(input=cnn_weight, dim=0)
            cnn_weighted_item = torch.mul(cnn_items_embed.transpose(1,2), cnn_item_weight.view(cnn_item_weight.shape[0],1,cnn_item_weight.shape[1]))
            cnn_context_item = cnn_weighted_item.transpose(1,2).sum(1)
            
            # cnn user attention
            if self.stack_num == 0:
                # user_rep = user_embed + context_item
                cnn_user_rep = cnn_context_item
            for i in range(self.stack_num):
                name = 'user_attention{}'.format(i)
                if i == 0:
                    cnn_user_rep = getattr(self, name)(user_embed, cnn_context_item)
                    cnn_user_rep = cnn_user_rep + user_embed
                else:
                    cnn_prev_rep = cnn_user_rep
                    cnn_user_rep = getattr(self, name)(cnn_user_rep, cnn_context_item)
                    cnn_user_rep = cnn_user_rep + cnn_prev_rep
            
            # get cnn_user_rep in cnn final
            time_user_rep = torch.cat((posi_user_rep, cnn_user_rep), dim=1)
            user_rep = self.time_fc(time_user_rep)
            user_rep = F.relu(user_rep)
            
            pos_pref_score = torch.mul(user_rep, pos_item_embed).sum(dim=-1)
            neg_pref_score = torch.mul(user_rep, neg_item_embed).sum(dim=-1)

        return pos_pref_score, neg_pref_score 
        
    def get_user_weights(self):
        return self.user_embedding.weight
    
    def get_item_weights(self):
        return self.item_embedding.weight

