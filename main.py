import sys
import getopt
import torch
import torch.optim as optim
import torch.utils.data as torch_data
from torchsummaryX import summary

from tqdm.auto import tqdm

from dataset import *
from utils import *
from dataloader import *
from bpr import *
from model import *


def usage():
    print(' -d (int) embedding dimensions\n'
          ' -l (float) learning rate\n'
          ' -e (int) training epochs\n'
          ' -D (str) dataset\n'
          ' -b (int) batch size\n'
          ' -w (int>2) window size\n'
          ' -n (int) numbers of negative sample\n'
          ' -i (0 or 1) using pretrained item weight\n'
          ' -s (int) numbers of stacked\n'
          ' -h (int) numbers of heads\n'
          ' -t (0, 1, 2, 3, 4, 5) none, posi, cnn, both, first posi then cnn, concat then fc'
            )


if __name__ == '__main__':
    # default
    emb_size = 32
    learning_rate = 0.001
    epochs = 50
    dataset = "ml-100k"
    batch_size = 256
    is_pretrained_item_weight = True
    num_stack = 5
    num_head = 5
    time_information_mode = 0
    neg_num = 10
    window_size = 5

    try:
        options, args = getopt.getopt(sys.argv[1:], "d:l:e:D:b:w:n:i:s:h:t:", ["dim=", "learning_rate=", "epochs=", "dataset=", "batch_size=", "window_size=", "neg_num=", "is_pretrained_item_weight=", "num_stack=", "num_head=", "time_information_mode="])

        for opt, value in options:
            if opt in ('-d', '--dim'):
                emb_size = int(value)
            
            elif opt in ('-l', '--learning_rate'):
                learning_rate = float(value)

            elif opt in ('-e', '--epochs'):
                epochs = int(value)

            elif opt in ('-D', '--dataset'):
                dataset = str(value)

            elif opt in ('-b', '--batch_size'):
                batch_size = int(value)
            
            elif opt in ('-w', '--window_size'):
                window_size = int(value)
            
            elif opt in ('-n', '--neg_num'):
                neg_num = int(value)

            elif opt in ('-i', '--is_pretrained_item_weight'):
                is_pretrained_item_weight = True if int(value) == 1 else False

            elif opt in ('-s', '--num_stack'):
                num_stack = int(value)

            elif opt in ('-h', '--num_head'):
                num_head = int(value)

            elif opt in ('-t', '--time_information_mode'):
                print("value")
                time_information_mode = int(value)
                
    
    except getopt.GetoptError:
        usage()

    info = 'Model Learning Information:\n'\
        +'Dimension: {}\n'\
        'Learning rate: {}\n' \
        'Epochs: {}\t, BatchSize: {}\t, Dataset: {}\n' \
        'Window size : {}\t, Numbers of Negative sample: {}\n' \
        'Using pretrained item weight: {}\t, Numbers of stack: {}\t, Numbers of head: {}\n' \
        'Time information mode: {}\t'.format(emb_size, learning_rate, epochs, batch_size, dataset, window_size, neg_num,is_pretrained_item_weight, num_stack, num_head, time_information_mode)

    print(info)

    data = Dataset(dataset)
    data.load_data()
    num_user = data.num_user
    num_item = data.num_item

    
    print("Data loading...\n\n")
    model_args = ARGS(num_users=num_user, num_items=num_item, latent_dim=emb_size, learning_rate=learning_rate)
    train_context_data, test_context_data = make_training_data(data.user_session, window_size=window_size)

    if is_pretrained_item_weight:
        print("Pretrain weight loading...\n\n")

        if dataset == 'ml-1m':
            bpr_weight_path = "./model/bpr/"+dataset+"/2020.03.19.03.08.48.dataset_ml-1m.epoch_7.NDCG_num_dim_32_BPR_state_dict"
        elif dataset == 'ml-100k':
            bpr_weight_path = "./model/bpr/"+dataset+"/2020.03.19.16.03.41.dataset_ml-100k.epoch_9.NDCG_num_dim_32_BPR_state_dict"
        elif dataset == 'pinterest':
            bpr_weight_path = "./model/bpr/"+dataset+"/2020.04.26.22.03.00.dataset_pinterest.epoch_9.hitRate_num_dim_32_BPR_state_dict"
        elif dataset == 'beauty':
            bpr_weight_path = "./model/bpr/"+dataset+"/2020.08.16.18.57.26.dataset_Amazon_Beauty.epoch_6.NDCG_num_dim_32_BPR_state_dict"
        
        bpr = torch.load(bpr_weight_path)
        bpr_user_weight = bpr.user_embedding.weight.detach().numpy()
        bpr_item_weight = bpr.item_embedding.weight.detach().numpy()
    
    isRun = True

    if window_size!=5 and time_information_mode != 0 and time_information_mode != 1:
        print("!! Fatal Error !!")
        print("Current CNN size only for windows size 5!")  
        print("!! Fatal Error !!") 
        isRun = False
    if isRun:
        print("Data processing...\n\n")
        train_data = ContextData(data=train_context_data, item_num=num_item, user_sessions=data.user_session, neg_num=neg_num)
        train_data.ng_sample()
        train_loader = torch_data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=16)

        print("Model Init...\n\n")
        model = stacked_AttRec(model_args, stack_num=num_stack, head_num=num_head, time_info_mode = time_information_mode, window_size = window_size, is_pretrained_item_weight = is_pretrained_item_weight, bpr_item_weight=bpr_item_weight,  isItemGrad = True, isUserBN = False).cuda()
        optimizer = optim.Adam(model.parameters(), lr=model_args.learning_rate)
        eva = Evaluate(test_context_data, data.testing_data)

        print("Model summary...\n\n")
        u = torch.randint(0, 10, torch.Size([1])).type(torch.LongTensor).cuda()
        i = torch.randint(0, 10, size=(1, window_size)).cuda()
        summary(model, u, i, u, u)
        print("Dataset :", dataset)
        print("Description :", model.model_description)
        # Model Training
        print("Model training...\n\n")
        
        best_ndcg = 0
        best_hit_rate = 0

        # train
        date = datetime.datetime.now().strftime('%Y.%m.%d.%H.%M.%S')

        txt_path = '{}/{}/{}.dataset_{}.num_dim_{}_{}.txt'.format('result', dataset, date, dataset, emb_size, model.model_description)
        eva_output = str(date)+"\n"
        with open(txt_path, 'w+') as f:
            f.write(eva_output)
        f.close()

        for ep in range(epochs):
            print("epoch :"+str(ep+1)+"/"+str(epochs))
            eva_output = "epoch :"+str(ep+1)+"/"+str(epochs)+"\n"
            model.train() 
            start_time = time.time()
            loss_list = []
            kl_list = []

            for user, inputs, pos_item, neg_item in tqdm(train_loader):
                
                user = user.cuda()
                inputs = inputs.cuda()
                pos_item = pos_item.cuda()
                neg_item = neg_item.cuda()

                optimizer.zero_grad()

                pos_pref_score, neg_pref_score = model(user, inputs, pos_item, neg_item)
                
                loss = (1.0 - (pos_pref_score - neg_pref_score).sigmoid()).mean()
                loss_list.append(loss)
                loss.backward()
                optimizer.step()
            
            print("loss : ", sum(loss_list)/len(loss_list))
            eva_output += "loss : "+str(sum(loss_list)/len(loss_list))+"\n"
            
            elapsed_time = time.time() - start_time
            print("The time elapse of epoch {:03d}".format(ep+1) + " is: " + time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
            eva_output += "The time elapse of epoch {:03d}".format(ep+1) + " is: " + time.strftime("%H: %M: %S", time.gmtime(elapsed_time))+"\n"
            
            model.eval()
            eva.set_eva_function(model)
            results = eva.run()
            output_result = eva.print_result()
            eva_output += output_result+"\n"
            
            with open(txt_path, 'a') as f:
                f.write(eva_output)
            f.close()
            
            if results['num_hit'][10] > best_hit_rate:
                ep_str = str(ep+1)
                model_path = '{}/{}/{}/{}.dataset_{}.epoch_{}.{}_num_dim_{}_{}_state_dict'.format('model', 'models', dataset, date, dataset, ep_str, "hitRate", emb_size, model.model_description)
                best_hit_rate = results['num_hit'][10]
                if results['NDCG_score'][10] > best_ndcg:
                    best_ndcg = results['NDCG_score'][10]
                torch.save(model, model_path)
                best_hr_user_weight = model.get_user_weights()
                best_hr_item_weight = model.get_item_weights()
                print("Hit rate save model epoch "+ep_str)
                continue
            if results['NDCG_score'][10] > best_ndcg and results['num_hit'][10] >= best_hit_rate:
                ep_str = str(ep+1)
                model_path = '{}/{}/{}/{}.dataset_{}.epoch_{}.{}_num_dim_{}_{}_state_dict'.format('model', 'models', dataset, date, dataset, ep_str, "NDCG", emb_size, model.model_description)
                best_ndcg = results['NDCG_score'][10]
                print("NDCG score save model epoch "+ep_str)
                torch.save(model, model_path)
                best_ndcg_user_weight = model.get_user_weights()
                best_ndcg_item_weight = model.get_item_weights()
            





