# Adaptive Context Aware Recommendation System via Stacked Attention Network
This is our offical implement for the master thesis :
**Adaptive Context Aware Recommendation System via Stacked Attention Network**
in NCKU CSIE

**Author : Chung-Yao, Ma, This code was finished in 2020/8/18** 

**Connect to me : chungyao.ma@gmail.com**

This deep-learning-based recommendation system using two kinds of attention modules to capture users' changing preferences and learn how to adapt to the dynamic changing pattern

## Quick to Start
---
```
    Parameters' definitions:
        -d (int) embedding dimensions
        -l (float) learning rate
        -e (int) training epochs
        -D (str) dataset
        -b (int) batch size
        -w (int>2) window size
        -n (int) numbers of negative sample
        -i (0 or 1) using pretrained item weight
        -s (int) numbers of stacked
        -h (int) numbers of heads
        -t (0, 1, 2, 3, 4, 5) time information mode : 0-none, 1-positional encoding, 2-CNN, 3-add both, 4-first posi then cnn, 5-concat two kinds of output then passing a fc
```

Run : 
```shell
python python.py --d 32 -l 0.001 -e 50 -D ml-100k -b 256 -w 5 -n 10 -i 1 -s 5 -h 5 -t 0
```

## Environment:
---
Python 3.6

Torch >= 1.4.0

numpy == 1.17.3

pandas == 0.25.3

torchsummaryX == 1.3.0

tqdm == 4.31.1

Or you can just pip install requirement.txt


PS. For your reference, our server environment is NVIDIA® Tesla V100 GPU


## Dataset:
---
We provide four processed datasets: MovieLens 1 Million (ml-1m), MovieLens 100k (ml-100k), Amazon Beauty (beauty) and Pinterest with pretrained bpr weight in Data

Pretrained bpr weight for each model are save in model/bpr/dataset

user_session.pkl : 

- Using to make training and testing data
- Saving users' all interacted items with python dictionary structure into a binary pickle file
- { uid_1 : [ iid_0, iid_1, iid_2,.... ] }

testing_data.pkl : 

- Using to evaluation
- Saving users' leave one out testing items and 99 negative items with python dictionary structure into a binary pickle file
- { uid_1 : [ neg_1, neg_2, ..., neg_99, test_item ] }


