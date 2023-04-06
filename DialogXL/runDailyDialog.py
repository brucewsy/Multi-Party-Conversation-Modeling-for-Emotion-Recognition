import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
import numpy as np, argparse, time, pickle, random
np.set_printoptions(threshold = np.inf)
import torch
import torch.nn as nn
import torch.optim as optim
from model import ERC_transfo_xl, ERC_xlnet
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report, \
    precision_recall_fscore_support
from trainerWithE import  train_or_eval_model_for_transfo_xl, eval_model_for_transfo_xl
from dataset import IEMOCAPDataset_transfo_xl
from dataloader import get_IEMOCAP_loaders_transfo_xl, get_IEMOCAP_loaders_xlnet
from transformers import AdamW
from transformers.optimization import get_linear_schedule_with_warmup
import logging
from focal_loss import *
from ghm_loss import *
# We use seed = 100 for reproduction of the results reported in the paper.
# seed = 100


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':

    path = './saved/IEMOCAP/'

    parser = argparse.ArgumentParser()
    parser.add_argument('--home_dir', type=str, default='')
    parser.add_argument('--bert_model_dir', type=str, default='')
    parser.add_argument('--bert_tokenizer_dir', type=str, default='')

    parser.add_argument('--basemodel', type=str, default='xlnet_dialog', choices=['xlnet', 'transfo_xl', 'bert', 'xlnet_dialog'],
                        help = 'base model')

    parser.add_argument('--bert_dim', type = int, default=768)
    parser.add_argument('--hidden_dim', type = int, default=300)
    parser.add_argument('--mlp_layers', type=int, default=2, help='Number of output mlp layers.')

    parser.add_argument('--max_sent_len', type=int, default=300, # according to the pre-trained transformer-xl
                        help='max content length for each text, if set to 0, then the max length has no constrain')
    parser.add_argument('--mem_len', type=int, default=450, help='max memory length')
    parser.add_argument('--windowp', type=int, default=10, help='local attention window size')
    parser.add_argument('--attn_type', type = str, choices=['uni','bi'], default='bi')

    parser.add_argument('--no_cuda', action='store_true', default=False, help='does not use GPU')

    parser.add_argument('--dataset_name', default='DailyDialog',  type= str, help='dataset name, IEMOCAP or MELD or DailyDialog')

    parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Gradient clipping.')

    parser.add_argument('--lr', type=float, default=1e-6, metavar='LR', help='learning rate')

    parser.add_argument('--dropout', type=float, default=0, metavar='dropout', help='dropout rate')

    parser.add_argument('--batch_size', type=int, default=8, metavar='BS', help='batch size')

    parser.add_argument('--epochs', type=int, default=30, metavar='E', help='number of epochs')

    parser.add_argument('--tensorboard', action='store_true', default=False, help='Enables tensorboard log')
    parser.add_argument('--lognum', type=int,  default=1000, help='Log number')
    parser.add_argument('--seed', type=int,  default=100, help='random seed')
    parser.add_argument('--num_heads', nargs='+', type=int, default=[1, 2, 5, 4],
                        help='number of heads:[n_local,n_global,n_speaker,n_listener], default=[3,3,3,3]')

    parser.add_argument('--local_rank', type=int, default=-1)

    args = parser.parse_args()
    if args.local_rank == -1 or args.local_rank == 0:
        print(args)
    seed_everything(args.seed)
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    if args.cuda and args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
    if args.cuda:
        if args.local_rank == -1 or args.local_rank == 0:
            print('Running on GPU')
    else:
        print('Running on CPU')
    #if args.cuda:
        #print('Running on GPU')
    #else:
        #print('Running on CPU')

    if args.tensorboard:
        from tensorboardX import SummaryWriter

        writer = SummaryWriter()


    cuda = args.cuda
    n_epochs = args.epochs
    batch_size = args.batch_size
    # args.num_heads = [2,2,4,4]
    if args.basemodel == 'transfo_xl':
        train_loader, valid_loader, test_loader,speaker_vocab, label_vocab, _ = get_IEMOCAP_loaders_transfo_xl(dataset_name=args.dataset_name, batch_size=batch_size, num_workers=0, args = args)
    elif args.basemodel in ['xlnet', 'xlnet_dialog']:
        train_loader, valid_loader, test_loader, train_size, dev_size, test_size, speaker_vocab, label_vocab, _ = get_IEMOCAP_loaders_xlnet(
            dataset_name=args.dataset_name, batch_size=batch_size, num_workers=0, args=args)
    # renew_cnt = [0 for i in range(len(speaker_vocab['itos']))]
    n_classes = len(label_vocab['itos'])

    if args.local_rank == -1 or args.local_rank == 0:
        print('building model..')

    if args.basemodel == 'transfo_xl':
        model = ERC_transfo_xl(args, n_classes)
    elif args.basemodel in ['xlnet','xlnet_dialog']:
        model = ERC_xlnet(args, n_classes, use_cls = True)

    if cuda:
        if args.local_rank == -1:
            print('Multi-GPU...........')
            model.cuda()
            model = nn.DataParallel(model,device_ids = range(torch.cuda.device_count()))
        else:
            # use distributed parallel
            if args.local_rank == 0:
                print('use DDP...')
            model = model.to(args.local_rank)
            model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    
    #if torch.cuda.device_count() > 1:
        #print('Mutli-GPU...........')
        #model = nn.DataParallel(model,device_ids = range(torch.cuda.device_count()))  

    if cuda:
        model.cuda()



    #loss_function = nn.CrossEntropyLoss(ignore_index=-1)
    loss_function = GHMC()
    optimizer = AdamW(model.parameters() , lr=args.lr) 
    # total_steps = len(train_loader) * n_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=int(len(train_loader)*2), # Default value
                                                    num_training_steps=len(train_loader)*args.epochs)

    if args.local_rank == -1 or args.local_rank == 0:
        logger = get_logger('outs/'+args.dataset_name+'/'+args.dataset_name+str(args.lognum)+'.log')
        logger.info('start training on GPU {}!'.format(os.environ["CUDA_VISIBLE_DEVICES"]))
        logger.info(args)
        logger.info('seed {}!'.format(args.seed))


    best_fscore, best_loss, best_label, best_pred, best_mask = None, None, None, None, None
    all_fscore, all_acc, all_loss = [], [], []

    best_testacc = 0.
    best_testfscore=0.
    best_vadacc = 0.
    best_vadfscore=0. 

 
    for e in range(n_epochs):
        start_time = time.time()

        train_loss, train_acc, _, _, train_fscore,GHMflag = train_or_eval_model_for_transfo_xl(model, loss_function,
                                                                                             train_loader, e, cuda,
                                                                                             args,n_classes, train_size, optimizer,scheduler, True)
        valid_loss, valid_acc, _, _, valid_fscore,GHMflag= eval_model_for_transfo_xl(model, loss_function,
                                                                                             valid_loader, e, cuda, args,n_classes, dev_size)
        test_loss, test_acc, test_label, test_pred, test_fscore,GHMflag= eval_model_for_transfo_xl(model,
                                                                                                           loss_function,
                                                                                                           test_loader,
                                                                                                           e, cuda, args,n_classes, test_size)
        all_fscore.append([valid_fscore, test_fscore])
            # torch.save({'model_state_dict': model.state_dict()}, path + name + args.base_model + '_' + str(e) + '.pkl')
            # print(renew_cnt)
            # print(person_vec[10])
        if GHMflag:
            if args.local_rank == -1 or args.local_rank == 0:
                logger.info('GHMflag is {} in this epoch.'.format(GHMflag))
        if args.local_rank == -1 or args.local_rank == 0:
            logger.info('training LR {} '.format(scheduler._last_lr))



        if args.tensorboard:
            writer.add_scalar('test: accuracy/loss', test_acc / test_loss, e)
            writer.add_scalar('train: accuracy/loss', train_acc / train_loss, e)

        if args.local_rank == -1 or args.local_rank == 0:
            logger.info('epoch: {}, train_loss: {}, train_acc: {}, train_fscore: {}, valid_loss: {}, valid_acc: {}, valid_fscore: {}, test_loss: {}, test_acc: {}, test_fscore: {}, time: {} sec'. \
            format(e + 1, train_loss, train_acc, train_fscore, valid_loss, valid_acc, valid_fscore, test_loss, test_acc,
                   test_fscore, round(time.time() - start_time, 2)))


        #print(
            #'epoch: {}, train_loss: {}, train_acc: {}, train_fscore: {}, valid_loss: {}, valid_acc: {}, valid_fscore: {}, test_loss: {}, test_acc: {}, test_fscore: {}, time: {} sec'. \
            #format(e + 1, train_loss, train_acc, train_fscore, valid_loss, valid_acc, valid_fscore, test_loss, test_acc,
                   #test_fscore, round(time.time() - start_time, 2)))

    if args.tensorboard:
        writer.close()

    if args.local_rank == -1 or args.local_rank == 0:
        print('Test performance..')
        all_fscore = sorted(all_fscore, key=lambda x: x[0], reverse=True)
        print('Best F-Score based on validation:', all_fscore[0][1])
        print('Best F-Score based on test:', max([f[1] for f in all_fscore]))
        logger.info('Best F-Score based on validation {},Best F-Score based on test:{}!'.format(all_fscore[0][1],max([f[1] for f in all_fscore])))
        logger.info('all_fscore {}!'.format(all_fscore))

