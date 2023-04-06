import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle, pandas as pd
import json
import numpy as np
import argparse
import random
from  transformers import BertTokenizer, TransfoXLTokenizer, XLNetTokenizer
import torch.distributed as dist
import math

def load_vocab(dataset_name):
    speaker_vocab = pickle.load(open('../data/%s/speaker_vocab.pkl' % (dataset_name), 'rb'))
    label_vocab = pickle.load(open('../data/%s/label_vocab.pkl' % (dataset_name), 'rb'))
    person_vec_dir = '../data/%s/person_vect.pkl' % (dataset_name)
    # if os.path.exists(person_vec_dir):
    #     print('Load person vec from ' + person_vec_dir)
    #     person_vec = pickle.load(open(person_vec_dir, 'rb'))
    # else:
    #     print('Creating personality vectors')
    #     person_vec = np.random.randn(len(speaker_vocab['itos']), 100)
    #     print('Saving personality vectors to' + person_vec_dir)
    #     with open(person_vec_dir,'wb') as f:
    #         pickle.dump(person_vec, f, -1)
    person_vec = None

    return speaker_vocab, label_vocab, person_vec


class IEMOCAPDataset_transfo_xl(Dataset):

    def __init__(self, data, speaker_vocab=None, label_vocab=None, args = None, tokenizer = None):
        self.speaker_vocab = speaker_vocab
        self.label_vocab = label_vocab
        self.args = args
        self.content_ids, self.labels, self.content_mask, self.ids = self.process(data, tokenizer)

        self.len = len(self.content_ids)

    def process(self, data, tokenizer):
        '''
        :param data:
            list of dialogues
        :param tokenizer:
        :return:
        '''
        max_dialog_len = max([len(d) for d in data])
        content_ids = [] # list, content_ids for each segment
        labels = [] # list, labels for each segment
        content_mask = [] # list, masks of tokens for each segment
        indecies = []

        for i in range(max_dialog_len):
            ids = []
            lbs = []
            mask = []
            for d in data:
                if i < len(d):
                    token_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(d[i]['text'], add_space_before_punct_symbol=True))
                    m = [1 for i in range(len(token_ids))]
                    ids.append(token_ids)
                    mask.append(m)
                    lbs.append(self.label_vocab['stoi'][d[i]['label']] if 'label' in d[i].keys() else -1)
                else:
                    ids.append([0])
                    mask.append([0])
                    lbs.append(-1)
            content_ids.append(ids)
            labels.append(lbs)
            content_mask.append(mask)
            indecies.append(i)

        for i in range(max_dialog_len):
            max_sent_len = min(max([len(x) for x in content_ids[i]]), self.args.max_sent_len)
            if max_sent_len == 0:
                max_sent_len = 1
            for j in range(len(content_ids[i])):
                if len(content_ids[i][j]) > max_sent_len:
                    content_ids[i][j] = content_ids[i][j][:max_sent_len]
                    content_mask[i][j] = content_mask[i][j][:max_sent_len]
                else:
                    content_ids[i][j] += [0] * (max_sent_len - len(content_ids[i][j]))
                    content_mask[i][j] += [0] * (max_sent_len - len(content_mask[i][j]))
                # print(len(content_mask[i][j]))
            # print('\n')

        # exit()

        return content_ids, labels, content_mask, indecies

    def __getitem__(self, index):
        '''

        :param index:
        :return:
            text_ids:
            token_types:
            label
        '''
        return torch.LongTensor(self.content_ids[index]), \
               torch.LongTensor(self.labels[index]), \
               torch.FloatTensor(self.content_mask[index]), \
               self.ids[index]

    def __len__(self):
        return self.len

    # def collate_fn(self, data):
    #     '''
    #
    #     :param data:
    #         content_ids
    #         token_types
    #         labels
    #     :return:
    #
    #     '''
    #     content_ids = pad_sequence([d[0] for d in data], batch_first = True) # (B, T, )
    #     token_types = pad_sequence([d[1] for d in data], batch_first = True) # (B, T, )
    #     labels = torch.LongTensor([d[2] for d in data])
    #
    #     return content_ids, token_types, labels


class IEMOCAPDataset_xlnet(Dataset):

    def __init__(self, data, speaker_vocab=None, label_vocab=None, args = None, tokenizer = None):
        self.speaker_vocab = speaker_vocab
        self.label_vocab = label_vocab
        self.args = args
        self.examples, self.num_total_examples = self.process(data, tokenizer)

        self.len = len(self.examples)

    def process(self, data, tokenizer):
        '''
        :param data:
            list of dialogues
        :param tokenizer:
        :return:
        '''
        examples = []
        num_total_examples = 0

        for d in data:
            ids = []
            lbs = []
            mask = []
            speaker = []
            for i in range(len(d)):
                token_ids = tokenizer.convert_tokens_to_ids(['<cls>'] + tokenizer.tokenize(d[i]['text']))
                m = [1 for i in range(len(token_ids))]
                ids.append(token_ids)
                mask.append(m)
                speaker.append(self.speaker_vocab['stoi'][d[i]['speaker']] + 1)
                lbs.append(self.label_vocab['stoi'][d[i]['label']] if 'label' in d[i].keys() else -1)
            
            examples.append({
                'content_ids': ids,
                'labels': lbs,
                'content_mask': mask,
                'speaker_ids': speaker
            })

        new_examples = []
        num_replicas = dist.get_world_size()

        for j in range(math.ceil(len(examples)/(num_replicas*self.args.batch_size))):
            data = examples[j*num_replicas*self.args.batch_size:(j+1)*num_replicas*self.args.batch_size]
            
            max_dialog_len = max([len(d['content_ids']) for d in data])
            num_total_examples += max_dialog_len * len(data)

            for i in range(len(data)):
                data[i]['content_ids'] = data[i]['content_ids'] + [[3]] * (max_dialog_len - len(data[i]['content_ids']))
                data[i]['content_mask'] = data[i]['content_mask'] + [[1]] * (max_dialog_len - len(data[i]['content_mask']))
                data[i]['speaker_ids'] = data[i]['speaker_ids'] + [0] * (max_dialog_len - len(data[i]['speaker_ids']))
                data[i]['labels'] = data[i]['labels'] + [-1] * (max_dialog_len - len(data[i]['labels']))
                data[i]['content_lengths'] = [0 for _ in range(max_dialog_len)]
                
            for i in range(max_dialog_len):
                max_sent_len = min(max([len(d['content_ids'][i]) for d in data]), self.args.max_sent_len)
                if max_sent_len == 0:
                    max_sent_len = 1
                for k in range(len(data)):
                    if len(data[k]['content_ids'][i]) > max_sent_len:
                        data[k]['content_lengths'][i] = max_sent_len
                        data[k]['content_ids'][i] = data[k]['content_ids'][i][:max_sent_len]
                        data[k]['content_mask'][i] = data[k]['content_mask'][i][:max_sent_len]                    
                    else:
                        data[k]['content_lengths'][i] = len(data[k]['content_ids'][i])
                        data[k]['content_ids'][i] = data[k]['content_ids'][i] +  [5] * (max_sent_len - len(data[k]['content_ids'][i]))
                        data[k]['content_mask'][i] = data[k]['content_mask'][i] +  [0] * (max_sent_len - len(data[k]['content_mask'][i]))                                         

            new_examples += data  

        return new_examples, num_total_examples


    def __getitem__(self, index):
        '''

        :param index:
        :return:
            text_ids:
            token_types:
            label
        '''
        return self.examples[index]['content_ids'], \
               self.examples[index]['labels'], \
               self.examples[index]['content_mask'], \
               self.examples[index]['speaker_ids'], \
               self.examples[index]['content_lengths']


    def __len__(self):
        return self.len


    def collate_fn(self, data):

        content_ids = [] 
        labels = [] 
        content_mask = [] 
        content_lengths = [] 
        speaker_ids = []
        indecies = []
        
        for i in range(len(data[0][0])):
            ids = []
            lbs = []
            mask = []
            speaker = []
            lens = []
            for d in data:             
                ids.append(d[0][i])
                mask.append(d[2][i])
                speaker.append(d[3][i])
                lbs.append(d[1][i])
                lens.append(d[4][i])

            content_ids.append(ids)
            labels.append(lbs)
            content_mask.append(mask)
            speaker_ids.append(speaker)
            indecies.append(i)
            content_lengths.append(lens)
        #print(content_ids[1])
        return [torch.LongTensor(ids) for ids in content_ids], \
                [torch.LongTensor(lb) for lb in labels], \
                [torch.FloatTensor(m) for m in content_mask], \
                [torch.LongTensor(l) for l in content_lengths], \
                [torch.FloatTensor(s) for s in speaker_ids], \
                indecies

        '''
        max_dialog_len = max([len(d[0]) for d in data])
        content_ids = [] 
        labels = [] 
        content_mask = [] 
        content_lengths = [] 
        speaker_ids = []
        indecies = []
        
        for i in range(max_dialog_len):
            ids = []
            lbs = []
            mask = []
            speaker = []
            for d in data:
                if i < len(d[0]):
                    ids.append(d[0][i])
                    mask.append(d[2][i])
                    speaker.append(d[3][i])
                    lbs.append(d[1][i])
                else:
                    ids.append([3])
                    mask.append([1])
                    lbs.append(-1)
                    speaker.append(0)
            content_ids.append(ids)
            labels.append(lbs)
            content_mask.append(mask)
            speaker_ids.append(speaker)
            indecies.append(i)

        for i in range(max_dialog_len):
            lens = []
            max_sent_len = min(max([len(x) for x in content_ids[i]]), self.args.max_sent_len)
            if max_sent_len == 0:
                max_sent_len = 1
            for j in range(len(content_ids[i])):
                if len(content_ids[i][j]) > max_sent_len:
                    lens.append(max_sent_len)
                    content_ids[i][j] = content_ids[i][j][:max_sent_len]
                    content_mask[i][j] = content_mask[i][j][:max_sent_len]
                else:
                    lens.append(len(content_ids[i][j]))
                    content_ids[i][j] += [5] * (max_sent_len - len(content_ids[i][j]))
                    content_mask[i][j] += [0] * (max_sent_len - len(content_mask[i][j]))
            content_lengths.append(lens)

        return [torch.LongTensor(ids) for ids in content_ids], \
                [torch.LongTensor(lb) for lb in labels], \
                [torch.FloatTensor(m) for m in content_mask], \
                [torch.LongTensor(l) for l in content_lengths], \
                [torch.FloatTensor(s) for s in speaker_ids], \
                indecies
        '''

