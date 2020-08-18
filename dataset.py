import pickle as pkl
import pandas as pd
import time

class Dataset():
    def __init__(self, dataset):
        self.dataset = str(dataset)
        dataset_path = './dataset/'+self.dataset
        raw_data_path = dataset_path + '/raw_data/'
        preprocessed_data_path = dataset_path + '/processed_data/'

        if self.dataset == 'ml-1m':
            print("MovieLens-1m")
            self.movies_path = raw_data_path+'movies.dat'
        elif self.dataset == 'ml-100k':
            print("MovieLens-100k")
            self.movies_path = raw_data_path+'movies.csv'
        elif self.dataset == 'beauty':
            print("Amazon-Beauty")
        elif self.dataset == 'pinterest':
            print("Pinterest")
        
        self.user_session_path = preprocessed_data_path+'user_session.pkl'
        self.test_data_path = preprocessed_data_path+'testing_data.pkl'

    def load_data(self):
        
        with open(self.user_session_path, 'rb') as user_session_file:
            self.user_session = pkl.load(user_session_file)

        with open(self.test_data_path, 'rb') as test_data_file:
            self.testing_data = pkl.load(test_data_file)


        if self.dataset == 'ml-1m' :
            self.num_user = 6040+1 # start from 1
            self.num_item = 3883 # len(pd.read_csv(self.movies_path, sep='::')['movieId'].tolist())
            
        elif self.dataset == 'ml-100k':
            self.num_user = 943+1 # start from 1
            self.num_item = 1682 # len(pd.read_csv(self.movies_path)['movieId'].tolist())
        
        elif self.dataset == 'beauty':
            self.num_user = 4412+1 # start from 1
            self.num_item = 8670

        elif self.dataset == 'pinterest':
            self.num_user = 24935+1 # start from 1
            self.num_item = 38722


