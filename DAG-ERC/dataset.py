from builtins import enumerate
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle, pandas as pd
import json
import numpy as np
import random
from pandas import DataFrame


class IEMOCAPDataset(Dataset):

    def __init__(self, dataset_name = 'IEMOCAP', split = 'train', speaker_vocab=None, label_vocab=None, args = None, tokenizer = None):
        self.speaker_vocab = speaker_vocab
        self.label_vocab = label_vocab
        #for i in range(6):
            #print(self.label_vocab['itos'][i])
        self.args = args
        self.data = self.read(dataset_name, split, tokenizer)
        print(len(self.data))

        self.len = len(self.data)

    def read(self, dataset_name, split, tokenizer):
        with open('../data/%s/%s_data_roberta.json.feature'%(dataset_name, split), encoding='utf-8') as f:
            raw_data = json.load(f)

        with open('../data/%s/%s_data_parsing.json'%(dataset_name, split), encoding='utf-8') as f:
            parsing = json.load(f)

        # process dialogue
        dialogs = []
        #sp=[]
        #de=[]

        for d, p in zip(raw_data, parsing):
            utterances = []
            labels = []
            speakers = []
            features = []
            relations = p['relations']
            dependency = [(-1, None, []) for _ in range(len(relations)+1)]

            for relation in relations:
                x = relation['x']
                y = relation['y']
                r = relation['type']
                p = relation['link probs']
                dependency[y] = (x, r, p)      

            for i,u in enumerate(d):
                utterances.append(u['text'])
                labels.append(self.label_vocab['stoi'][u['label']] if 'label' in u.keys() else -1)
                speakers.append(self.speaker_vocab['stoi'][u['speaker']])
                features.append(u['cls'])

            #sp.append(speakers)
            #de.append(dependency)
            dialogs.append({
                'utterances': utterances, 
                'labels': labels,
                'speakers': speakers,
                'features': features,
                'dependency': dependency
            })

        '''
        n = 0
        m = 0
        for k,speaker in enumerate(sp):
            for i,s in enumerate(speaker):
                cnt = 0
                a = [0 for _ in range(len(speaker))]
                for j in range(i - 1, -1, -1):            
                    a[j] = 1 
                    if speaker[j] == s:
                        cnt += 1
                        if cnt==self.args.windowp:
                            break
                if i:
                    link_probs = de[k][i][2]
                    for j,p in enumerate(link_probs):
                        if p>self.args.threshold:
                            a[j]=1
                
                n+=sum(a)
                m+=1
        print(n/m)
        '''

        random.shuffle(dialogs)
        return dialogs        

    def __getitem__(self, index):
        '''
        :param index:
        :return:
            feature,
            label
            speaker
            length
            text
            dependency
        '''
        return torch.FloatTensor(self.data[index]['features']), \
               torch.LongTensor(self.data[index]['labels']),\
               self.data[index]['speakers'], \
               len(self.data[index]['labels']), \
               self.data[index]['utterances'], \
               self.data[index]['dependency']

    def __len__(self):
        return self.len

    def get_adj(self, speakers, max_dialog_len):
        '''
        get adj matrix
        :param speakers:  (B, N)
        :param max_dialog_len:
        :return:
            adj: (B, N, N) adj[:,i,:] means the direct predecessors of node i
        '''
        adj = []
        for speaker in speakers:
            a = torch.zeros(max_dialog_len, max_dialog_len)
            for i,s in enumerate(speaker):
                get_local_pred = False
                get_global_pred = False
                for j in range(i - 1, -1, -1):
                    if speaker[j] == s and not get_local_pred:
                        get_local_pred = True
                        a[i,j] = 1
                    elif speaker[j] != s and not get_global_pred:
                        get_global_pred = True
                        a[i,j] = 1
                    if get_global_pred and get_local_pred:
                        break
            adj.append(a)
        return torch.stack(adj)

    def get_adj_v1(self, speakers, max_dialog_len):
        '''
        get adj matrix
        :param speakers:  (B, N)
        :param max_dialog_len:
        :return:
            adj: (B, N, N) adj[:,i,:] means the direct predecessors of node i
        '''
        adj = []
        for speaker in speakers:
            a = torch.zeros(max_dialog_len, max_dialog_len)
            for i,s in enumerate(speaker):
                cnt = 0
                for j in range(i - 1, -1, -1):             
                    a[i,j] = 1
                    if speaker[j] == s:
                        a[i,j] = 1
                        cnt += 1
                        if cnt==self.args.windowp:
                            break
            adj.append(a)
        return torch.stack(adj)
    
    def get_adj_v2(self, speakers, dependency, max_dialog_len):
        '''
        get adj matrix
        :param speakers:  (B, N)
        :param max_dialog_len:
        :return:
            adj: (B, N, N) adj[:,i,:] means the direct predecessors of node i
        '''
        adj = []
        for k,speaker in enumerate(speakers):
            a = torch.zeros(max_dialog_len, max_dialog_len)
            for i in range(1, len(speaker)):
                w = dependency[k][i][0]
                a[i, w] = 1
                #a[i, j] = 1
                #if dependency[k][i][1]=='Continuation':
                    #a[i,w] = 1
                    #continue
                if w:
                    w = dependency[k][w][0]
                    a[i, w] = 1
                #for j in range(i - 1, -1, -1):   
                    #a[i,j] = 1
                    #if j == w:          
                        #break
            adj.append(a)
        return torch.stack(adj) # (B, N, N)

    def get_adj_v3(self, speakers, dependency, max_dialog_len):
        '''
        get adj matrix
        :param speakers:  (B, N)
        :param max_dialog_len:
        :return:
            adj: (B, N, N) adj[:,i,:] means the direct predecessors of node i
        '''
        adj = []
        for k,speaker in enumerate(speakers):
            a = torch.zeros(max_dialog_len, max_dialog_len)
            for i in range(1, len(speaker)):
                for j in range(i - 1, -1, -1):    
                    a[i,j] = dependency[k][i][-1][j]
            adj.append(a)
        return torch.stack(adj) # (B, N, N)

    def get_adj_v4(self, speakers, dependency, max_dialog_len):
        '''
        get adj matrix
        :param speakers:  (B, N)
        :param max_dialog_len:
        :return:
            adj: (B, N, N) adj[:,i,:] means the direct predecessors of node i
        '''
        adj = []
        for k,speaker in enumerate(speakers):
            a = torch.zeros(max_dialog_len, max_dialog_len)
            for i,s in enumerate(speaker):
                cnt = 0
                for j in range(i - 1, -1, -1):             
                    a[i,j] = 1
                    if speaker[j] == s:
                        a[i,j] = 1
                        cnt += 1
                        if cnt==self.args.windowp:
                            break
                if i:
                    link_probs = dependency[k][i][2]
                    sorted = np.argsort(link_probs)
                    for j in range(1,min(i+1,4)):
                        a[i,sorted[-j]] = 1
            adj.append(a)
        return torch.stack(adj) # (B, N, N)

    def get_adj_v5(self, speakers, dependency, max_dialog_len):
        '''
        get adj matrix
        :param speakers:  (B, N)
        :param max_dialog_len:
        :return:
            adj: (B, N, N) adj[:,i,:] means the direct predecessors of node i
        '''
        adj = []
        for k,speaker in enumerate(speakers):
            a = torch.zeros(max_dialog_len, max_dialog_len)
            for i,s in enumerate(speaker):
                cnt = 0
                for j in range(i - 1, -1, -1):            
                    a[i,j] = 1
                    '''
                    cnt+=1
                    if cnt==self.args.windowp:
                        break
                    '''
                    if speaker[j] == s:
                        a[i,j] = 1
                        cnt += 1
                        if cnt==self.args.windowp:
                            break
                    
                if i:
                    link_probs = dependency[k][i][2]
                    for j,p in enumerate(link_probs):
                        if p>self.args.threshold:
                           a[i,j] = 1 
            adj.append(a)
        return torch.stack(adj) # (B, N, N)

    def get_s_mask(self, speakers, max_dialog_len):
        '''
        :param speakers:
        :param max_dialog_len:
        :return:
         s_mask: (B, N, N) s_mask[:,i,:] means the speaker informations for predecessors of node i, where 1 denotes the same speaker, 0 denotes the different speaker
         s_mask_onehot (B, N, N, 2) onehot emcoding of s_mask
        '''
        s_mask = []
        s_mask_onehot = []
        for speaker in speakers:
            s = torch.zeros(max_dialog_len, max_dialog_len, dtype = torch.long)
            s_onehot = torch.zeros(max_dialog_len, max_dialog_len, 2)
            for i in range(len(speaker)):
                for j in range(len(speaker)):
                    if speaker[i] == speaker[j]:
                        s[i,j] = 1
                        s_onehot[i,j,1] = 1
                    else:
                        s_onehot[i,j,0] = 1

            s_mask.append(s)
            s_mask_onehot.append(s_onehot)
        return torch.stack(s_mask), torch.stack(s_mask_onehot)

    def get_s_mask_v2(self, speakers, dependency, max_dialog_len):
        '''
        :param speakers:
        :param max_dialog_len:
        :return:
         s_mask: (B, N, N) s_mask[:,i,:] means the speaker informations for predecessors of node i, where 1 denotes the same speaker, 0 denotes the different speaker
         s_mask_onehot (B, N, N, 2) onehot emcoding of s_mask
        '''
        map_relation0 = {'Comment':0,'Clarification_question':1,'Continuation':2,'Acknowledgement':3,'Question-answer_pair':4,'Elaboration':4,'Contrast':4,'Explanation':4,'Q-Elab':4,'Correction':4, \
            'Result':4,'Conditional':4,'Background':4,'Narration':4,'Alternation':4,'Parallel':4}
        map_relation1 = {'Comment':5,'Clarification_question':6,'Continuation':7,'Acknowledgement':8,'Question-answer_pair':9,'Elaboration':9,'Contrast':9,'Explanation':9,'Q-Elab':9,'Correction':9, \
            'Result':9,'Conditional':9,'Background':9,'Narration':9,'Alternation':9,'Parallel':9}
        s_mask = []
        s_mask_onehot = []
        for k, speaker in enumerate(speakers):
            s = torch.zeros(max_dialog_len, max_dialog_len, dtype = torch.long)
            s_onehot = torch.zeros(max_dialog_len, max_dialog_len, 10)
            for i in range(1, len(speaker)):
                t = r = dependency[k][i][1]
                q = p = dependency[k][i][0]
                if p:
                    r = dependency[k][p][1]
                    p = dependency[k][p][0]    
                for j in range(i-1, p-1, -1):
                    if speaker[j]==speaker[i]:
                        if j>=q:
                            s[i, j] = map_relation0[t]
                            s_onehot[i, j, map_relation0[t]] = 1
                        else:
                            s[i, j] = map_relation0[r]
                            s_onehot[i, j, map_relation0[r]] = 1
                    else:
                        if j>=q:
                            s[i, j] = map_relation1[t]
                            s_onehot[i, j, map_relation1[t]] = 1
                        else:
                            s[i, j] = map_relation1[r]
                            s_onehot[i, j, map_relation1[r]] = 1                 
            s_mask.append(s)
            s_mask_onehot.append(s_onehot)
        return torch.stack(s_mask), torch.stack(s_mask_onehot)

    def get_s_mask_v3(self, speakers, dependency, max_dialog_len):
        '''
        :param speakers:
        :param max_dialog_len:
        :return:
         s_mask: (B, N, N) s_mask[:,i,:] means the speaker informations for predecessors of node i, where 1 denotes the same speaker, 0 denotes the different speaker
         s_mask_onehot (B, N, N, 2) onehot emcoding of s_mask
        '''
        s_mask = []
        s_mask_onehot = []
        for k, speaker in enumerate(speakers):
            s = torch.zeros(3, max_dialog_len, max_dialog_len, dtype = torch.long)
            s_onehot = torch.zeros(max_dialog_len, max_dialog_len, 3)
            for i in range(len(speaker)):
                for j in range(len(speaker)):
                    if speaker[i] == speaker[j]:
                        s[1,i,j] = 1
                        s_onehot[i,j,1] = 1
                    else:
                        s[0,i,j] = 1
                        s_onehot[i,j,0] = 1
                if i:
                    link_probs = dependency[k][i][2]
                    sorted = np.argsort(link_probs)
                    for j in range(1,min(i+1,2)):
                        s[2,i,sorted[-j]] = 1
                        s_onehot[i,sorted[-j],2] = 1
            s_mask.append(s)
            s_mask_onehot.append(s_onehot)
        return torch.stack(s_mask), torch.stack(s_mask_onehot)

    def get_s_mask_v4(self, speakers, dependency, max_dialog_len):
        '''
        :param speakers:
        :param max_dialog_len:
        :return:
         s_mask: (B, N, N) s_mask[:,i,:] means the speaker informations for predecessors of node i, where 1 denotes the same speaker, 0 denotes the different speaker
         s_mask_onehot (B, N, N, 2) onehot emcoding of s_mask
        '''
        s_mask = []
        s_mask_onehot = []
        for k, speaker in enumerate(speakers):
            s = torch.zeros(3, max_dialog_len, max_dialog_len, dtype = torch.long)
            s_onehot = torch.zeros(max_dialog_len, max_dialog_len, 3)
            for i in range(len(speaker)):
                for j in range(len(speaker)):
                    if speaker[i] == speaker[j]:
                        s[1,i,j] = 1
                        s_onehot[i,j,1] = 1
                    else:
                        s[0,i,j] = 1
                        s_onehot[i,j,0] = 1
                
                if i:
                    '''
                    link_probs = dependency[k][i][2]
                    sorted = np.argsort(link_probs)
                    for j in range(1,min(i+1,4)):
                        s[2,i,sorted[-j]] = 1
                        s_onehot[i,sorted[-j],2] = 1
                    '''
                    link_probs = dependency[k][i][2]
                    for j,p in enumerate(link_probs):
                        if p>self.args.threshold:
                            s[2,i,j] = 1
                            s_onehot[i,j,2] = 1
                    
            s_mask.append(s)
            s_mask_onehot.append(s_onehot)
        return torch.stack(s_mask), torch.stack(s_mask_onehot)

    def get_r_mask(self, speakers, dependency, max_dialog_len):
        '''
        :param speakers:
        :param max_dialog_len:
        :return:
         s_mask: (B, N, N) s_mask[:,i,:] means the speaker informations for predecessors of node i, where 1 denotes the same speaker, 0 denotes the different speaker
         s_mask_onehot (B, N, N, 2) onehot emcoding of s_mask
        '''
        map_relation = {'Comment':0,'Clarification_question':1,'Continuation':2,'Acknowledgement':0,'Question-answer_pair':3,'Elaboration':0,'Contrast':3,'Explanation':0,'Q-Elab':0,'Correction':3, \
            'Result':3,'Conditional':3,'Background':3,'Narration':3,'Alternation':3,'Parallel':3}
        s_mask = []
        s_mask_onehot = []
        for k, speaker in enumerate(speakers):
            s = torch.zeros(max_dialog_len, max_dialog_len, dtype = torch.long)
            s_onehot = torch.zeros(max_dialog_len, max_dialog_len, 5)
            for i in range(1, len(speaker)):
                t = r = dependency[k][i][1]
                q = p = dependency[k][i][0]
                if p:
                    r = dependency[k][p][1]
                    p = dependency[k][p][0]    
                for j in range(i-1, p-1, -1):
                        if j==q:
                            s[i, j] = map_relation[t]
                            s_onehot[i, j, map_relation[t]] = 1
                        elif j==p:
                            s[i, j] = map_relation[r]
                            s_onehot[i, j, map_relation[r]] = 1
                        else:
                            s[i, j] = 4
                            s_onehot[i, j, 4] = 1
             
            s_mask.append(s)
            s_mask_onehot.append(s_onehot)
        return torch.stack(s_mask), torch.stack(s_mask_onehot)

    def collate_fn(self, data):
        '''

        :param data:
            features, labels, speakers, length, utterances, dependency
        :return:
            features: (B, N, D) padded
            labels: (B, N) padded
            adj: (B, N, N) adj[:,i,:] means the direct predecessors of node i
            s_mask: (B, N, N) s_mask[:,i,:] means the speaker informations for predecessors of node i, where 1 denotes the same speaker, 0 denotes the different speaker
            lengths: (B, )
            utterances:  not a tensor
        '''
        max_dialog_len = max([d[3] for d in data])
        feaures = pad_sequence([d[0] for d in data], batch_first = True) # (B, N, D)
        labels = pad_sequence([d[1] for d in data], batch_first = True, padding_value = -1) # (B, N )
        adj = self.get_adj_v5([d[2] for d in data], [d[5] for d in data], max_dialog_len)
        #adj = self.get_adj_v2([d[2] for d in data], [d[5] for d in data], max_dialog_len)
        s_mask, s_mask_onehot = self.get_s_mask_v4([d[2] for d in data], [d[5] for d in data], max_dialog_len)
        #s_mask, s_mask_onehot = self.get_s_mask([d[2] for d in data], max_dialog_len)
        lengths = torch.LongTensor([d[3] for d in data])
        speakers = pad_sequence([torch.LongTensor(d[2]) for d in data], batch_first = True, padding_value = -1)
        utterances = [d[4] for d in data]

        return feaures, labels, adj, s_mask, s_mask_onehot, lengths, speakers, utterances
