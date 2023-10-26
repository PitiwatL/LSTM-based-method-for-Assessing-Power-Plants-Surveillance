import deepcut
from pythainlp.tag import pos_tag
import numpy as np
import torch
from gensim.models import KeyedVectors, Word2Vec
import json

def remove_spaces(string):
    """Removes all spaces from a string."""
    return ''.join(c for c in string if not c.isspace())

def passag2words(passage):
    Passage = []
    if len(passage) < 120 :
        Passage = [passage]
    
    elif len(passage) >= 120: 
        num_char = len(passage)
        for idx in range(num_char//120) :
            Passage.append(passage[idx*120:((idx+1)*120)])
    
    Clean_Passage = []
    for sent in Passage:
        Clean_Passage.append(remove_spaces(sent.split('\n')[0]))
    
    Clean_Passagee = []
    for sent in Clean_Passage:
        if sent != '' :
            Clean_Passagee.append(remove_spaces(sent))
    
    TargetPos = ['NUM', 'CCONJ', 'DET', 'AUX', 'ADV', 'SCONJ', 'PUNCT', 'ADP']
    
    Words = []
    for idx, sent in enumerate(Clean_Passagee):
        WORD = []
        for chunks in sent.split(' '):
            try:
                for word in pos_tag(deepcut.tokenize(chunks), corpus='orchid_ud') :
                    if word[1] not in TargetPos :
                        WORD.append(word[0])
            except: 
                pass
        Words.append(WORD)
        WORD = []
    
    return Words

def pred_passage(ListWords, Lstm, wrd2vec):
    Prediction = []
    for sent in ListWords:
        Vprd = []
        for idx in range(60) :
            try:
                Vprd.append(wrd2vec[sent[idx]])
            except:
                Vprd.append(np.zeros(400))
        
        with torch.no_grad():
            Vprd = torch.from_numpy(np.array(Vprd)).unsqueeze(0).cuda().float()
            prd = Lstm(Vprd)
            
        prd = torch.argmax(prd, 1)
        Prediction.append(prd.cpu().numpy()[0])
    
    return Prediction

def decision(key, Lstm, wrd2vec):
    f = open('INPUT.json', encoding="utf8")
    passage = json.load(f)
    
    passage = passage[key]
    
    if remove_spaces(passage) == "" :
        count0, count1, count2 = 1, 1, 0
        
        return count0, count1, count2
    
    if remove_spaces(passage) != "" :
        # print(len(passage))
        ListSent = passag2words(passage)
        
        # # 0 ทั่วไป 1 เชิงบวก 2 เชิวลบ
        Prd = pred_passage(ListSent, Lstm, wrd2vec)
        
        count0, count1, count2 = 0, 0, 0
        for prd in Prd :
            if prd == 0:
                count0 += 1
            if prd == 1:
                count1 += 1
            if prd == 2:
                count2 += 1
        
        # prd1 = round((1 - count2/(count0 + count1 + count2)), 4)
        # prd2 = round(count2/(count0 + count1 + count2), 4)
        
    return count0, count1, count2
