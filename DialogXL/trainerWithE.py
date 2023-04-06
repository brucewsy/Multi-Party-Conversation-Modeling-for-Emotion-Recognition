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
import torch.distributed as dist


def my_cross_entropy(input, target, reduction="mean"):
    exp = torch.exp(input)
    tmp1 = exp.gather(1, target.unsqueeze(-1)).squeeze()
    tmp2 = exp.sum(1)
    softmax = tmp1 / tmp2
    log = -torch.log(softmax)
    if reduction == "mean": return log.mean()
    else: return log.sum()


def train_or_eval_model_for_transfo_xl(model, loss_function, dataloader, epoch, cuda, args, n_classes, data_size, optimizer=None, scheduler=None, train=False):
    '''

    :param model:
    :param loss_function:
    :param dataloader: list of datasets,
    :param args:
    :param optimizer:
    :param train:
    :return:
    '''
    ALL_logits, ALL_labels = [], []
    scores, vids = [], []
    CELOSS = nn.CrossEntropyLoss(ignore_index=-1)


    assert not train or optimizer != None
    if train:
        model.train()
    else:
        model.eval()


    ALL_grads = []

    with torch.no_grad():
        for c, dataset in enumerate(dataloader):

            mems = None
            speaker_mask = None
            window_mask = None

            batch_logits, batch_labels = [], []

            #for data in dataset:
            for i in range(len(dataset[0])):
                if train:
                    optimizer.zero_grad()


                # text_ids, text_feature, speaker_ids, labels, umask = [d.cuda() for d in data] if cuda else data
                #content_ids, label, content_mask, content_lengths, speaker_ids,_ = data
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
    
                #if -1 in label:
                    # print('-1')
                    #idx =label!=-1
                    #label = label[idx]
                    #logits = logits[idx]


                batch_labels.append(label) # (seq_len, B)
                prob = torch.softmax(logits, dim=-1) # (B, n_classes)
                batch_logits.append(prob[[k for k in range(label.size(0))], label].detach()) # (seq_len, B)

                    # idx = 
                # loss = F.cross_entropy(logits, label)
                # loss = my_cross_entropy(logits, label)

                # loss = loss_function(logits, label) 
                # su = []
                # for i in range(len(label)):
                #     su.append(F.cross_entropy(logits[i], label[i]))


                # loss = loss_function(logits, label,torch.ones_like(logits[0]))
                #t = F.one_hot(label,n_classes) # (B, n_classes)
                #g = torch.abs(torch.softmax(logits,dim=-1).detach() - t) # (B, n_classes)
                #g= g[t==1].cpu().numpy().tolist() # (B, )
                #ALL_grads.extend(g)

            batch_labels = torch.stack(batch_labels, dim=1) # (B, seq_len)
            batch_logits = torch.stack(batch_logits, dim=1) # (B, seq_len)

            reduce_batch_labels = [batch_labels.clone() for _ in range(dist.get_world_size())]
            reduce_batch_logits = [batch_logits.clone() for _ in range(dist.get_world_size())]
            dist.all_gather(reduce_batch_labels, batch_labels) # (world_size, B, seq_len)
            dist.all_gather(reduce_batch_logits, batch_logits) # (world_size, B, seq_len)
            
            for b in range(batch_labels.size(0)):
                for j in range(dist.get_world_size()):
                    ALL_logits += reduce_batch_logits[j][b].cpu().tolist()
                    ALL_labels += reduce_batch_labels[j][b].cpu().tolist()
    

    valid_logits, valid_labels = [], []
    for l,p in zip(ALL_labels, ALL_logits):
        if l != -1:
            valid_logits.append(p)
            valid_labels.append(l) 
    
    #t = F.one_hot(ALL_labels, n_classes) # (num_data, n_classes)
    #g = torch.abs(torch.softmax(ALL_logits, dim=-1).detach() - t) # (num_data, n_classes)
    #g = g[t==1].cpu().numpy().tolist() # (num_data, )
    #ALL_grads = g

    ALL_grads = abs(np.array(valid_logits) - np.array(valid_labels)) # (num_data, )
    
    n = 0  # n valid bins
    weights = np.zeros(10)
    ALL_grads = torch.tensor(ALL_grads)
    ALL_grads = ALL_grads.cuda()
    num_labels = len(ALL_grads)

    bins = 10
    edges = torch.arange(bins + 1).float().cuda() / bins
    edges[-1] += 1e-6
    for i in range(bins):
        inds = (ALL_grads >= edges[i]) & (ALL_grads < edges[i + 1]) 
        num_in_bin = inds.sum().item()
        if num_in_bin > 0:
            weights[n] = num_labels / num_in_bin 
            n += 1
    if n > 0:
        weights = weights / n

    if n == bins:
        GHMflag = True
    else:
        GHMflag = False

    num_datas = []

    losses, preds, labels = [], [], []
    scores, vids = [], []


    assert not train or optimizer != None
    if train:
        model.train()
    else:
        model.eval()

   
    for dataset in dataloader:

        mems = None
        speaker_mask = None
        window_mask = None

        batch_preds, batch_labels = [], []

        #for data in dataset:
        for i in range(len(dataset[0])):
            if train:
                optimizer.zero_grad()


            # text_ids, text_feature, speaker_ids, labels, umask = [d.cuda() for d in data] if cuda else data
            #content_ids, label, content_mask, content_lengths, speaker_ids,_ = data
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
            #GHMflag = False
            labelForLoss=[]
            #if -1 in label:
                # print('-1')
            idx =label!=-1
            valid_label = label[idx]
            valid_logits = logits[idx]
                # idx = 
            #if valid_label.shape == torch.Size([0]):
                #break    
            if valid_label.shape != torch.Size([0]) and GHMflag:
                loss = loss_function(valid_logits, valid_label, label_weight = weights) # GHM loss
                #GHMflag=True
            else:
                loss = CELOSS(logits, label)   

            # loss = loss_function(logits, label, label_weight = weights) # GHM loss
            # loss = loss_function(logits, label)
            # loss = loss_function(logits, label,torch.ones_like(logits[0])) #2 5
        


            '''
            torch.abs(torch.softmax(logits,dim=-1).detach() - F.one_hot(label,6))
            target = label
            
            target = F.one_hot(target,pred.size(-1))

            # target, label_weight = target.float(), label_weight.float()
            edges = self.edges
            mmt = self.momentum
            weights = torch.zeros_like(pred)
            g = torch.abs(torch.softmax(pred,dim=-1).detach() - target)
            '''
            
            losses.append(loss.item())
            num_datas.append((label!=-1).sum())

            pred = torch.argmax(logits, 1) # (B, )

            batch_labels.append(label) # (seq_len, B)
            batch_preds.append(pred) # (seq_len, B)

            #label = label.cpu().numpy().tolist()
            #groundTruthLb = label
            #pred = torch.argmax(logits, 1).cpu().numpy().tolist()
            #for l,p in zip(label, pred):
                #if l != -1:
                    #preds.append(p)
                    #labels.append(l)
            #losses.append(loss.item())

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
                # scheduler.step()
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

    # truncate the dummy elements added by mylDistributedSampler_v2
    preds = preds[:data_size]
    labels = labels[:data_size]
    # delete the preds with label -1
    valid_preds, valid_labels = [], []
    for l,p in zip(labels, preds):
        if l != -1:
            valid_preds.append(p)
            valid_labels.append(l) 

    if preds != []:
        preds = np.array(preds)
        labels = np.array(labels)
    else:
        return float('nan'), float('nan'), [], [], float('nan'), [], [], [], [], []


    avg_loss = round(np.sum(losses) / len(losses), 4)
    avg_accuracy = round(accuracy_score(valid_labels, valid_preds) * 100, 2)
    if args.dataset_name in ['IEMOCAP', 'MELD', 'EmoryNLP']:
        avg_fscore = round(f1_score(valid_labels, valid_preds, average='weighted') * 100, 2)
    elif args.dataset_name in ['DailyDialog']:
        avg_fscore =round(f1_score(valid_labels, valid_preds, labels=[1,2,3,4,5,6] ,average='micro') * 100, 2)
    #else:
        #avg_fscore = round(f1_score(labels, preds, average='micro') * 100, 2)

    return avg_loss, avg_accuracy, labels, preds, avg_fscore, GHMflag



def eval_model_for_transfo_xl(model, loss_function, dataloader, epoch, cuda, args, n_classes, data_size, optimizer=None, scheduler=None):
    '''

    :param model:
    :param loss_function:
    :param dataloader: list of datasets,
    :param args:
    :param optimizer:
    :param train:
    :return:
    '''
    ALL_logits, ALL_labels = [], []
    scores, vids = [], []
    CELOSS = nn.CrossEntropyLoss(ignore_index=-1)

    model.eval()

    ALL_grads = []

    with torch.no_grad():
        for c, dataset in enumerate(dataloader):

            mems = None
            speaker_mask = None
            window_mask = None

            batch_logits, batch_labels = [], []

            #for data in dataset:
            for i in range(len(dataset[0])):

                # text_ids, text_feature, speaker_ids, labels, umask = [d.cuda() for d in data] if cuda else data
                #content_ids, label, content_mask, content_lengths, speaker_ids,_ = data
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
    
                #if -1 in label:
                    # print('-1')
                    #idx =label!=-1
                    #label = label[idx]
                    #logits = logits[idx]


                batch_labels.append(label) # (seq_len, B)
                prob = torch.softmax(logits, dim=-1) # (B, n_classes)
                batch_logits.append(prob[[k for k in range(label.size(0))], label].detach()) # (seq_len, B)

                    # idx = 
                # loss = F.cross_entropy(logits, label)
                # loss = my_cross_entropy(logits, label)

                # loss = loss_function(logits, label) 
                # su = []
                # for i in range(len(label)):
                #     su.append(F.cross_entropy(logits[i], label[i]))


                # loss = loss_function(logits, label,torch.ones_like(logits[0])) #2 5
                #t = F.one_hot(label,n_classes) # (B, n_classes)
                #g = torch.abs(torch.softmax(logits,dim=-1).detach() - t) # (B, n_classes)
                #g= g[t==1].cpu().numpy().tolist() # (B, )
                #ALL_grads.extend(g)

            batch_labels = torch.stack(batch_labels, dim=1) # (B, seq_len)
            batch_logits = torch.stack(batch_logits, dim=1) # (B, seq_len)

            reduce_batch_labels = [batch_labels.clone() for _ in range(dist.get_world_size())]
            reduce_batch_logits = [batch_logits.clone() for _ in range(dist.get_world_size())]
            dist.all_gather(reduce_batch_labels, batch_labels) # (world_size, B, seq_len)
            dist.all_gather(reduce_batch_logits, batch_logits) # (world_size, B, seq_len)
            
            for b in range(batch_labels.size(0)):
                for j in range(dist.get_world_size()):
                    ALL_logits += reduce_batch_logits[j][b].cpu().tolist()
                    ALL_labels += reduce_batch_labels[j][b].cpu().tolist()
    

    valid_logits, valid_labels = [], []
    for l,p in zip(ALL_labels, ALL_logits):
        if l != -1:
            valid_logits.append(p)
            valid_labels.append(l) 
    
    #t = F.one_hot(ALL_labels, n_classes) # (num_data, n_classes)
    #g = torch.abs(torch.softmax(ALL_logits, dim=-1).detach() - t) # (num_data, n_classes)
    #g = g[t==1].cpu().numpy().tolist() # (num_data, )
    #ALL_grads = g

    ALL_grads = abs(np.array(valid_logits) - np.array(valid_labels)) # (num_data, )
    
    n = 0  # n valid bins
    weights = np.zeros(10)
    ALL_grads = torch.tensor(ALL_grads)
    ALL_grads = ALL_grads.cuda()
    num_labels = len(ALL_grads)
    bins = 10
    edges = torch.arange(bins + 1).float().cuda() / bins
    edges[-1] += 1e-6
    for i in range(bins):
        inds = (ALL_grads >= edges[i]) & (ALL_grads < edges[i + 1]) 
        num_in_bin = inds.sum().item()
        if num_in_bin > 0:
            weights[n] = num_labels / num_in_bin 
            n += 1
    if n > 0:
        weights = weights / n

    if n == bins:
        GHMflag = True
    else:
        GHMflag = False


    num_datas = []

    losses, preds, labels = [], [], []
    scores, vids = [], []



    with torch.no_grad():
        for dataset in dataloader:

            mems = None
            speaker_mask = None
            window_mask = None

            batch_preds, batch_labels = [], []

            #for data in dataset:
            for i in range(len(dataset[0])):


                # text_ids, text_feature, speaker_ids, labels, umask = [d.cuda() for d in data] if cuda else data
                #content_ids, label, content_mask, content_lengths, speaker_ids,_ = data
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
                #GHMflag = False
                labelForLoss=[]
                #if -1 in label:
                    # print('-1')
                idx =label!=-1
                valid_label = label[idx]
                valid_logits = logits[idx]
                    # idx = 
                #if valid_label.shape == torch.Size([0]):
                    #break    
                if valid_label.shape != torch.Size([0]) and GHMflag:
                    loss = loss_function(valid_logits, valid_label, label_weight = weights) # GHM loss
                    #GHMflag=True
                else:
                    loss = CELOSS(logits, label)   

                # loss = loss_function(logits, label, label_weight = weights) # GHM loss
                # loss = loss_function(logits, label)
                # loss = loss_function(logits, label,torch.ones_like(logits[0])) #2 5
            


                '''
                torch.abs(torch.softmax(logits,dim=-1).detach() - F.one_hot(label,6))
                target = label
                
                target = F.one_hot(target,pred.size(-1))

                # target, label_weight = target.float(), label_weight.float()
                edges = self.edges
                mmt = self.momentum
                weights = torch.zeros_like(pred)
                g = torch.abs(torch.softmax(pred,dim=-1).detach() - target)
                '''
                
                losses.append(loss.item())
                num_datas.append((label!=-1).sum())

                pred = torch.argmax(logits, 1) # (B, )

                batch_labels.append(label) # (seq_len, B)
                batch_preds.append(pred) # (seq_len, B)

                #label = label.cpu().numpy().tolist()
                #groundTruthLb = label
                #pred = torch.argmax(logits, 1).cpu().numpy().tolist()
                #for l,p in zip(label, pred):
                    #if l != -1:
                        #preds.append(p)
                        #labels.append(l)
                #losses.append(loss.item())

                # print(content_lengths)
                # print(mems[0].size())


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

    if preds != []:
        preds = np.array(preds)
        labels = np.array(labels)
    else:
        return float('nan'), float('nan'), [], [], float('nan'), [], [], [], [], []


    avg_loss = round(np.sum(losses) / len(losses), 4)
    avg_accuracy = round(accuracy_score(valid_labels, valid_preds) * 100, 2)
    if args.dataset_name in ['IEMOCAP', 'MELD', 'EmoryNLP']:
        avg_fscore = round(f1_score(valid_labels, valid_preds, average='weighted') * 100, 2)
    elif args.dataset_name in ['DailyDialog']:
        avg_fscore =round(f1_score(valid_labels, valid_preds, labels=[1,2,3,4,5,6] ,average='micro') * 100, 2)
    #else:
        #avg_fscore = round(f1_score(labels, preds, average='micro') * 100, 2)

    return avg_loss, avg_accuracy, labels, preds, avg_fscore, GHMflag