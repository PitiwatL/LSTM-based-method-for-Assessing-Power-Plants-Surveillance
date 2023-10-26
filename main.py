import torch
import numpy as np 
import os, shutil
import json

import deepcut
from torchvision import transforms

import torch
import torch.nn as nn
import torchvision.models as models
import tqdm
from tqdm import tqdm

from pythainlp.tag import pos_tag
from gensim.models import KeyedVectors, Word2Vec

from utils.Lstm import LSTMCells
from utils.functions import remove_spaces, passag2words, pred_passage, decision

# from torchvision.transforms import ToTensor
# import torch.nn.functional as F
#################################################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define W2V models
wrd2vec = KeyedVectors.load_word2vec_format('LTW2V_v0.1.bin', 
                                            binary=True, unicode_errors='ignore')
##################################################################################

class Model(nn.Module):
    def __init__(self, hidden_unit):
        super(Model, self).__init__()
        self.hidden_dim = hidden_unit
        self.cf1 = nn.Linear(128, 128)
        self.relu = nn.ReLU()
        self.cf2 = nn.Linear(128, 3)
        
        # initialize the attention blocks defined above
        self.LSTM1 = LSTMCells(400, hidden_unit, return_sequence = False)
        # self.LSTM2 = LSTMCells(128, 128, return_sequence = False)
        
        # self.double()
        
    def forward(self, x):
        LSTM1 = self.LSTM1(x)
        cf1 = self.cf1(LSTM1)
        relu = self.relu(cf1)
        cf2 = self.cf2(relu)
        
        return cf2

if __name__ == '__main__':
    Lstm = Model(128)
    Lstm = Lstm.to(device)
    Lstm.load_state_dict(torch.load('Lstm_Best.pt'))

    count11, count12, count13 = decision("text1", Lstm, wrd2vec)
    count21, count22, count23 = decision("text2", Lstm, wrd2vec)
    count31, count32, count33 = decision("text3", Lstm, wrd2vec)
    count41, count42, count43 = decision("text4", Lstm, wrd2vec)
    
    prd11 = round(((count11 + count12)/(count11 + count12 + count13)), 4)
    prd12 = 1 - prd11
    
    prd21 = round(((count21 + count22)/(count21 + count22 + count23)), 4)
    prd22 = 1 - prd21
    
    prd31 = round(((count31 + count32)/(count31 + count32 + count33)), 4)
    prd32 = 1 - prd31
    
    count_positive = count11 + count21 + count31 + count41 + count12 + count22 + count32 + count42
    count_negative = count13 + count23 + count33 + count43
    
    prdA1 = count_positive/(count_positive + count_negative)
    prdA2 = 1 - prdA1
    
    JSON_FILE = {   "Topic1": [{
                        "Positive Opinion " : str(prd11 * 100) + ' %', 
                        "Negative Opinion " : str(prd12 * 100) + ' %'}] ,
                    
                    "Topic2": [{
                        "Positive Opinion " : str(prd21 * 100) + ' %', 
                        "Negative Opinion " : str(prd22 * 100) + ' %'}],
                    
                    "Topic3": [{
                        "Positive Opinion " : str(prd31 * 100) + ' %', 
                        "Negative Opinion " : str(prd32 * 100) + ' %'}],
                    
                    "AI Opinion on this Document": [{
                        "Positive Opinion " : str(prdA1 * 100) + ' %', 
                        "Negative Opinion " : str(prdA2 * 100) + ' %'}]
                    
                }
    
    with open('OUTPUT.json', 'w') as outfile:
        json.dump(JSON_FILE, outfile)
                