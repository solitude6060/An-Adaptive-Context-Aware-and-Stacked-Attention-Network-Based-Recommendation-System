import math
import numpy as np
import torch
import datetime

# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------

class ARGS():
    def __init__(self,  num_users, num_items, latent_dim, learning_rate=0.001):
        self.num_users = num_users
        self.num_items = num_items
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate

# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------

class Evaluate():
    def __init__(self, testing_data, neg_testing, isShow=False):
        self.user_num = len(testing_data)
        self.testing_data = testing_data
        self.neg_testing = neg_testing
        self.isShow = isShow
    
    def set_eva_function(self, predice_function):
        self.predice_function = predice_function

    def run(self):
        topk = [1, 3, 5, 10]
        self.results = {'num_hit': {i: 0.0 for i in topk},
                    'MAP_score': {i: 0.0 for i in topk},
                    'NDCG_score': {i: 0.0 for i in topk}}

        for uid in list(self.testing_data.keys()):
            user_test_inputs = self.testing_data[uid][0]
            negs = self.neg_testing[uid][:-1]
            test_set = negs+[self.testing_data[uid][1]]
        
            tensor_user = torch.from_numpy(np.array([uid]*100)).cuda()
            tensor_input = torch.from_numpy(np.array(user_test_inputs))
            tensor_input = tensor_input.view(1, tensor_input.shape[0]).cuda()
            
            tensor_test = torch.from_numpy(np.array(test_set)).cuda()
        

            scores, _ = self.predice_function(tensor_user, tensor_input, tensor_test, tensor_test)
            scores_np = scores.cpu().detach().numpy()
            rank = scores_np[-1] <= scores_np
            
            
            if self.isShow:
                print(uid)
                print(rank)
                print(scores_np)
                
            true_rank = rank.sum()
            for k in topk:
                if true_rank <= k:
                    self.results['num_hit'][k] += 1
                    self.results['MAP_score'][k] += float(1.0/true_rank)
                    self.results['NDCG_score'][k] += float(math.log(2.0, 2.0) / math.log(true_rank-1.0+2.0, 2.0))
        
        for metric in self.results.keys():
            for topk in self.results[metric].keys():
                self.results[metric][topk] = float(self.results[metric][topk]/self.user_num)
        
        return self.results
    
    def print_result(self):
        result = ""
        result += "\nEvaluate : "
        print("\nEvaluate : ")
        
        result += 'HIT Rate:\t@1\t@3\t@5\t@10' + "\n\t\t"
        print('HIT Rate:\t@1\t@3\t@5\t@10', end="\n\t\t")
        for i in [1, 3, 5, 10]:
            result += str(self.results['num_hit'][i])+'\t'
            print("%.4f" %(self.results['num_hit'][i]), end="\t")
        
        result += '\nMAP_score:\t@1\t@3\t@5\t@10'+"\n\t\t"
        print('\nMAP_score:\t@1\t@3\t@5\t@10', end="\n\t\t")
        for i in [1, 3, 5, 10]:
            result += str(self.results['MAP_score'][i])+'\t'
            print("%.4f" %(self.results['MAP_score'][i]), end="\t")
        
        result += '\nNDCG_score:\t@1\t@3\t@5\t@10'+"\n\t\t"
        print('\nNDCG_score:\t@1\t@3\t@5\t@10', end="\n\t\t")
        for i in [1, 3, 5, 10]:
            result += str(self.results['NDCG_score'][i])+'\t'
            print("%.4f" %(self.results['NDCG_score'][i]), end="\t")
        
        result += "\n"
        print()

        return result

# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------

def make_training_data(user_session, window_size=5):
    one_session_len = window_size+1
    gru_user_data = {}
    train_gru_data = []
    test_gru_data = []

    for u, items in user_session.items():
        user_gru_user_list = []
        gru_data = []
        session_len = len(items)
        for i in range(session_len-one_session_len+1) :
            temp = []
            for k in range(one_session_len-1):
                temp.append(items[i+k])
            gru_data.append((temp, items[i+one_session_len-1]))
            user_gru_user_list.append((u, temp, items[i+one_session_len-1]))
        train_gru_data += user_gru_user_list[:-1]
        test_gru_data += [user_gru_user_list[-1]]
        gru_user_data[u] = gru_data

    gru_user_dict = {}
    for i in test_gru_data:
        # print(i)
        gru_user_dict[i[0]] = (i[1], i[2])
    
    return train_gru_data, gru_user_dict

# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------