from dataset import *
import pickle
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Sampler, DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import os
import math
import argparse
import numpy as np
from  transformers import BertTokenizer, TransfoXLTokenizer, XLNetTokenizer

def get_train_valid_sampler(trainset):
    size = len(trainset)
    idx = list(range(size))
    return SubsetRandomSampler(idx)


class myDistributedSampler(Sampler):
    r"""Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such a case, each
    process can pass a :class`~torch.utils.data.DistributedSampler` instance as a
    :class:`~torch.utils.data.DataLoader` sampler, and load a subset of the
    original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`rank` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.

    .. warning::
        In distributed mode, calling the :meth`set_epoch(epoch) <set_epoch>` method at
        the beginning of each epoch **before** creating the :class:`DataLoader` iterator
        is necessary to make shuffling work properly across multiple epochs. Otherwise,
        the same ordering will be always used.

    Example::

        >>> sampler = DistributedSampler(dataset) if is_distributed else None
        >>> loader = DataLoader(dataset, shuffle=(sampler is None),
        ...                     sampler=sampler)
        >>> for epoch in range(start_epoch, n_epochs):
        ...     if is_distributed:
        ...         sampler.set_epoch(epoch)
        ...     train(loader)
    """

    def __init__(self, dataset, batch_size, num_replicas=None, rank=None, shuffle=True, seed=0):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas)) # 每个进程的数据量
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))


        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        new_indices = []
        for i in range(math.floor(self.num_samples/self.batch_size)):
            new_indices += indices[(i*self.num_replicas+self.rank)*self.batch_size:(i*self.num_replicas+self.rank+1)*self.batch_size]
        rest = indices[-self.num_replicas*(self.num_samples%self.batch_size):]
        new_indices += rest[self.rank*(self.num_samples%self.batch_size):(self.rank+1)*(self.num_samples%self.batch_size)]

        assert len(new_indices) == self.num_samples

        return iter(new_indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Arguments:
            epoch (int): Epoch number.
        """
        self.epoch = epoch


class myDistributedSampler_v2(Sampler):
    r"""Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such a case, each
    process can pass a :class`~torch.utils.data.DistributedSampler` instance as a
    :class:`~torch.utils.data.DataLoader` sampler, and load a subset of the
    original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`rank` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.

    .. warning::
        In distributed mode, calling the :meth`set_epoch(epoch) <set_epoch>` method at
        the beginning of each epoch **before** creating the :class:`DataLoader` iterator
        is necessary to make shuffling work properly across multiple epochs. Otherwise,
        the same ordering will be always used.

    Example::

        >>> sampler = DistributedSampler(dataset) if is_distributed else None
        >>> loader = DataLoader(dataset, shuffle=(sampler is None),
        ...                     sampler=sampler)
        >>> for epoch in range(start_epoch, n_epochs):
        ...     if is_distributed:
        ...         sampler.set_epoch(epoch)
        ...     train(loader)
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas)) 
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))


        # add extra samples to make it evenly divisible
        if self.total_size > len(indices):
            indices += indices[-(self.total_size - len(indices)):]

        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]

        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Arguments:
            epoch (int): Epoch number.
        """
        self.epoch = epoch


class SequentialDistributedSampler(Sampler):
    """
    Distributed Sampler that subsamples indices sequentially, making it easier to collate all results at the end.
    Even though we only use this sampler for eval and predict (no training), which means that the model params won't
    have to be synced (i.e. will not hang for synchronization even if varied number of forward passes), we still add
    extra samples to the sampler to make it evenly divisible (like in `DistributedSampler`) to make it easy to `gather`
    or `reduce` resulting tensors at the end of the loop.
    """

    def __init__(self, dataset, num_replicas=None, rank=None, batch_size=None):

        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        num_samples = len(self.dataset)
        # Add extra samples to make num_samples a multiple of batch_size if passed
        if batch_size is not None:
            self.num_samples = int(math.ceil(num_samples / (batch_size * num_replicas))) * batch_size # num_samples each process is in charge of 
        else:
            self.num_samples = int(math.ceil(num_samples / num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.batch_size = batch_size

    def __iter__(self):
        indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert (
            len(indices) == self.total_size
        ), f"Indices length {len(indices)} and total size {self.total_size} mismatched"

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert (
            len(indices) == self.num_samples
        ), f"Indices length {len(indices)} and sample number {self.num_samples} mismatched"

        return iter(indices)

    def __len__(self):
        return self.num_samples


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

def read_datas(dataset_name, batch_size):
    # training set
    with open('../data/%s/train_data.json.feature' % (dataset_name), encoding='utf-8') as f:
        train_raw = json.load(f)
    new_train_raw = sorted(train_raw,key = lambda x:len(x))

    with open('../data/%s/dev_data.json.feature' % (dataset_name), encoding='utf-8') as f:
        dev_raw = json.load(f)
    new_dev_raw = sorted(dev_raw,key = lambda x:len(x))

    with open('../data/%s/test_data.json.feature' % (dataset_name), encoding='utf-8') as f:
        test_raw = json.load(f)
    new_test_raw = sorted(test_raw,key = lambda x:len(x))

    return new_train_raw, new_dev_raw, new_test_raw


def get_IEMOCAP_loaders_transfo_xl(dataset_name = 'IEMOCAP', batch_size=32, num_workers=0, pin_memory=False, args = None):
    tokenizer = TransfoXLTokenizer.from_pretrained(args.home_dir + args.bert_tokenizer_dir)
    print('building vocab.. ')
    speaker_vocab, label_vocab, person_vec = load_vocab(dataset_name)
    train_data, dev_data, test_data = read_datas(dataset_name, batch_size)
    print('building datasets..')
    trainset = IEMOCAPDataset_transfo_xl(train_data, speaker_vocab, label_vocab, args, tokenizer)
    devset = IEMOCAPDataset_transfo_xl(dev_data, speaker_vocab, label_vocab, args, tokenizer)
    testset = IEMOCAPDataset_transfo_xl(test_data, speaker_vocab, label_vocab, args, tokenizer)

    #train_sampler = SequentialSampler(trainset)
    #train_sampler = myDistributedSampler(trainset, args.batch_size, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=False)
    train_sampler = DistributedSampler(trainset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=False)
    #valid_sampler = SequentialSampler(devset)
    valid_sampler = DistributedSampler(devset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=False)
    #test_sampler = SequentialSampler(testset)
    test_sampler = DistributedSampler(testset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=False)

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    valid_loader = DataLoader(devset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=devset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
                    
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             sampler=test_sampler,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader, speaker_vocab, label_vocab, person_vec

def get_IEMOCAP_loaders_xlnet(dataset_name = 'IEMOCAP', batch_size=32, num_workers=0, pin_memory=False, args = None):
    tokenizer = XLNetTokenizer.from_pretrained(args.home_dir + args.bert_tokenizer_dir)
    if args.local_rank == -1 or args.local_rank == 0:
        print('building vocab.. ')
    speaker_vocab, label_vocab, person_vec = load_vocab(dataset_name)
    train_data, dev_data, test_data = read_datas(dataset_name, batch_size)
    if args.local_rank == -1 or args.local_rank == 0:
        print('building datasets..')
    trainset = IEMOCAPDataset_xlnet(train_data, speaker_vocab, label_vocab, args, tokenizer)
    devset = IEMOCAPDataset_xlnet(dev_data, speaker_vocab, label_vocab, args, tokenizer)
    testset = IEMOCAPDataset_xlnet(test_data, speaker_vocab, label_vocab, args, tokenizer)

    #train_sampler = SequentialSampler(trainset)
    #train_sampler = myDistributedSampler(trainset, args.batch_size, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=False)
    train_sampler = myDistributedSampler_v2(trainset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=False)
    #valid_sampler = SequentialSampler(devset)
    valid_sampler = myDistributedSampler_v2(devset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=False)
    #test_sampler = SequentialSampler(testset)
    test_sampler = myDistributedSampler_v2(testset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=False)

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    valid_loader = DataLoader(devset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=devset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
                    
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             sampler=test_sampler,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return  train_loader, valid_loader, test_loader, trainset.num_total_examples, devset.num_total_examples, testset.num_total_examples, speaker_vocab, label_vocab, person_vec

