import numpy as np, argparse, time, pickle, random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report, \
    precision_recall_fscore_support
from utils import person_embed
from tqdm import tqdm
from  transformers import BertTokenizer, TransfoXLTokenizer, XLNetTokenizer
import re
import json
import torch.nn.functional as F
from ghm_loss import *
import matplotlib
import matplotlib.pyplot as plt   
import numpy as np
import pandas as pd
import matplotlib.ticker as ticker
import torch.distributed as dist



def train_or_eval_model_for_transfo_xl(model, loss_function, dataloader, epoch, cuda, args, n_classes, data_size, optimizer=None,scheduler=None, train=False):
    '''

    :param model:
    :param loss_function:
    :param dataloader: list of datasets,
    :param args:
    :param optimizer:
    :param train:
    :return:
    '''
    losses = []
    num_datas = []
    preds, labels = [], []
    scores, vids = [], []
    GHMflag = False

    assert not train or optimizer != None
    if train:
        model.train()
        # dataloader = tqdm(dataloader)
    else:
        model.eval()


    finalDialog=[]
    # dialognumber=0
    for dataset in dataloader:
        
        mems = None
        speaker_mask = None
        window_mask = None
        # if train:
        #     dataset = tqdm(dataset)
        # cnt = 0
        batch_diolags=[]
        utteranceNumber=0

        batch_preds, batch_labels = [], []

        for i in range(len(dataset[0])):

            if train:
                optimizer.zero_grad()


            content_ids = dataset[0][i]
            label = dataset[1][i]
            content_mask = dataset[2][i]
            content_lengths = dataset[3][i]
            speaker_ids = dataset[4][i]

            if cuda:
                content_ids = content_ids.cuda()
                content_mask = content_mask.cuda()
                speaker_ids = speaker_ids.cuda()
                content_lengths = content_lengths.cuda()
                label = label.cuda()

            if args.basemodel == 'transfo_xl':
                logits, mems = model(content_ids, mems, content_mask)
            elif args.basemodel in ['xlnet_dialog', 'xlnet']:
                logits, mems, speaker_mask, window_mask = model(content_ids = content_ids, mems = mems, content_mask = content_mask,
                                                   content_lengths = content_lengths, speaker_ids = speaker_ids,
                                                   speaker_mask = speaker_mask, window_mask = window_mask)


            labelForLoss=[]
            # if -1 in label:
            #     # print('-1')
            #     idx =label!=-1
            #     label = label[idx]
            #     logits = logits[idx]
                
            
            loss = loss_function(logits, label)
            # loss = loss_function(logits, label,torch.ones_like(logits[0])) #2 5

            # print(speaker_mask.detach().cpu().numpy())
            # print(content_lengths)
            # print(speaker_ids)
            # print('------------------------------------------------------')

            losses.append(loss.item())
            num_datas.append((label!=-1).sum())

            pred = torch.argmax(logits, 1) # (B, )

            #label = label.cpu().numpy().tolist() # (B, )
            #groundTruthLb = label            
            #pred.cpu().numpy().tolist() # (B, )
            batch_labels.append(label) # (seq_len, B)
            batch_preds.append(pred) # (seq_len, B)
            

            #for l,p in zip(label, pred):
                #if l != -1:
                    #preds.append(p)
                    #labels.append(l) 
            #preds += pred
            #labels += label
            # print(content_lengths)
            # print(mems[0].size())



            if train:
                loss_val = loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                if args.tensorboard:
                    for param in model.named_parameters():
                        writer.add_histogram(param[0], param[1].grad, epoch)
                optimizer.step()

                # print('lr:{}'.format(scheduler._last_lr))

            # torch.cuda.empty_cache()

        batch_labels = torch.stack(batch_labels, dim=1) # (B, seq_len)
        batch_preds = torch.stack(batch_preds, dim=1) # (B, seq_len)

        reduce_batch_labels = [batch_labels.clone() for _ in range(dist.get_world_size())]
        reduce_batch_preds = [batch_preds.clone() for _ in range(dist.get_world_size())]
        dist.all_gather(reduce_batch_labels, batch_labels) # (world_size, B, seq_len)
        dist.all_gather(reduce_batch_preds, batch_preds) # (world_size, B, seq_len)

        for b in range(batch_labels.size(0)):
            for j in range(dist.get_world_size()):
                preds += reduce_batch_preds[j][b].cpu().tolist()
                labels += reduce_batch_labels[j][b].cpu().tolist()

        if train:
            scheduler.step()
    
    accumulate_loss = 0
    num_data = 0
    for l, n in zip(losses, num_datas):
        accumulate_loss += l * n
        num_data += n
    reduce_accumulate_loss = accumulate_loss.clone()
    reduce_num_data = num_data.clone()
    dist.all_reduce(reduce_accumulate_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(reduce_num_data, op=dist.ReduceOp.SUM)
    avg_loss = round((reduce_accumulate_loss / reduce_num_data).item(), 4)
    #avg_loss = round(np.sum(losses) / len(losses), 4)

    #reduce_preds = distributed_gather(preds, data_size)
    #reduce_labels = distributed_gather(labels, data_size)
    
    #valid_reduce_preds = []
    #valid_reduce_labels = []

    # truncate the dummy elements added by mylDistributedSampler_v2
    preds = preds[:data_size]
    labels = labels[:data_size]
    # delete the preds with label -1
    valid_preds, valid_labels = [], []
    for l,p in zip(labels, preds):
        if l != -1:
            valid_preds.append(p)
            valid_labels.append(l) 

    if valid_preds != []:
        valid_preds = np.array(valid_preds)
        valid_labels = np.array(valid_labels)
    else:
        return float('nan'), float('nan'), [], [], float('nan'), [], [], [], [], []

    # print(preds.tolist())
    # print(labels.tolist())
    #avg_loss = round(np.sum(losses) / len(losses), 4)
    avg_accuracy = round(accuracy_score(valid_labels, valid_preds) * 100, 2)
    if args.dataset_name in ['IEMOCAP', 'MELD', 'EmoryNLP']:
        avg_fscore = round(f1_score(valid_labels, valid_preds, average='weighted') * 100, 2)
    else:
        avg_fscore = round(f1_score(valid_labels, valid_preds, labels = list(range(1,7)), average='micro') * 100, 2)
    # print(classification_report(labels,preds))


    return avg_loss, avg_accuracy, labels, preds, avg_fscore, GHMflag


def eval_model_for_transfo_xl(model, loss_function, dataloader, epoch, cuda, args, n_classes, data_size):
    '''

    :param model:
    :param loss_function:
    :param dataloader: list of datasets,
    :param args:
    :param optimizer:
    :param train:
    :return:
    '''
    losses = []
    num_datas = []
    preds, labels = [], []
    scores, vids = [], []
    GHMflag = False

    model.eval()

    with torch.no_grad():
        for dataset in dataloader:     
            mems = None
            speaker_mask = None
            window_mask = None

            batch_preds, batch_labels = [], []

            for i in range(len(dataset[0])):

                content_ids = dataset[0][i]
                label = dataset[1][i]
                content_mask = dataset[2][i]
                content_lengths = dataset[3][i]
                speaker_ids = dataset[4][i]

                if cuda:
                    content_ids = content_ids.cuda()
                    content_mask = content_mask.cuda()
                    speaker_ids = speaker_ids.cuda()
                    content_lengths = content_lengths.cuda()
                    label = label.cuda()

                if args.basemodel == 'transfo_xl':
                    logits, mems = model(content_ids, mems, content_mask)
                elif args.basemodel in ['xlnet_dialog', 'xlnet']:
                    logits, mems, speaker_mask, window_mask = model(content_ids = content_ids, mems = mems, content_mask = content_mask,
                                                    content_lengths = content_lengths, speaker_ids = speaker_ids,
                                                    speaker_mask = speaker_mask, window_mask = window_mask)
                    
                
                loss = loss_function(logits, label)

                losses.append(loss.item())
                num_datas.append((label!=-1).sum())

                pred = torch.argmax(logits, 1) # (B, )

                batch_labels.append(label) # (seq_len, B)
                batch_preds.append(pred) # (seq_len, B)


            batch_labels = torch.stack(batch_labels, dim=1) # (B, seq_len)
            batch_preds = torch.stack(batch_preds, dim=1) # (B, seq_len)

            reduce_batch_labels = [batch_labels.clone() for _ in range(dist.get_world_size())]
            reduce_batch_preds = [batch_preds.clone() for _ in range(dist.get_world_size())]
            dist.all_gather(reduce_batch_labels, batch_labels) # (world_size, B, seq_len)
            dist.all_gather(reduce_batch_preds, batch_preds) # (world_size, B, seq_len)

            for b in range(batch_labels.size(0)):
                for j in range(dist.get_world_size()):
                    preds += reduce_batch_preds[j][b].cpu().tolist()
                    labels += reduce_batch_labels[j][b].cpu().tolist()
    
    accumulate_loss = 0
    num_data = 0
    for l, n in zip(losses, num_datas):
        accumulate_loss += l * n
        num_data += n
    reduce_accumulate_loss = accumulate_loss.clone()
    reduce_num_data = num_data.clone()
    dist.all_reduce(reduce_accumulate_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(reduce_num_data, op=dist.ReduceOp.SUM)
    avg_loss = round((reduce_accumulate_loss / reduce_num_data).item(), 4)

    # truncate the dummy elements added by mylDistributedSampler_v2
    preds = preds[:data_size]
    labels = labels[:data_size]
    # delete the preds with label -1
    valid_preds, valid_labels = [], []
    for l,p in zip(labels, preds):
        if l != -1:
            valid_preds.append(p)
            valid_labels.append(l) 

    if valid_preds != []:
        valid_preds = np.array(valid_preds)
        valid_labels = np.array(valid_labels)
    else:
        return float('nan'), float('nan'), [], [], float('nan'), [], [], [], [], []

    avg_accuracy = round(accuracy_score(valid_labels, valid_preds) * 100, 2)
    if args.dataset_name in ['IEMOCAP', 'MELD','EmoryNLP']:
        avg_fscore = round(f1_score(valid_labels, valid_preds, average='weighted') * 100, 2)
    else:
        avg_fscore = round(f1_score(valid_labels, valid_preds, labels = list(range(1,7)), average='micro') * 100, 2)
    # print(classification_report(labels,preds))


    return avg_loss, avg_accuracy, labels, preds, avg_fscore, GHMflag