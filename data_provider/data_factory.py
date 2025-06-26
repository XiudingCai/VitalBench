import torch
import random
import numpy as np
from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_M4, PSMSegLoader, \
    MSLSegLoader, SMAPSegLoader, SMDSegLoader, SWATSegLoader, UEAloader, Dataset_Solar, Dataset_PEMS, Dataset_DeepSleep
from data_provider.uea import collate_fn
from torch.utils.data import DataLoader

# my import
from data_provider.data_loader import Dataset_Folder

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'Solar': Dataset_Solar,
    'PEMS': Dataset_PEMS,
    'm4': Dataset_M4,
    'PSM': PSMSegLoader,
    'MSL': MSLSegLoader,
    'SMAP': SMAPSegLoader,
    'SMD': SMDSegLoader,
    'SWAT': SWATSegLoader,
    'UEA': UEAloader,
    # defined
    'folder': Dataset_Folder,
    'deepsleep': Dataset_DeepSleep,
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    if args.embed in ['fixed', 'learned']:
        timeenc = 0
    elif args.embed == 'timeF':
        timeenc = 1
    else:
        timeenc = 2

    if flag in ['test', 'SUBMIT']:
        shuffle_flag = False
        drop_last = True
        if args.task_name == 'anomaly_detection' or args.task_name == 'classification':
            batch_size = args.batch_size
        else:
            batch_size = 1  # bsz=1 for evaluation
        # # set by ez.
        # drop_last = False
        # batch_size = args.batch_size
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq

    if args.task_name == 'anomaly_detection':
        drop_last = False
        data_set = Data(
            root_path=args.root_path,
            win_size=args.seq_len,
            flag=flag,
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
    elif args.task_name == 'classification':
        print(flag)

        if args.data == 'deepsleep':
            seed = args.seed
            # seed = 42
            print(f"setting seed {seed} for dataset...")
            from utils.tools import set_seed
            set_seed(seed)

        drop_last = False
        data_set = Data(
            root_path=args.root_path,
            flag=flag,
        )

        if args.imbalanced_sampling:
            from torchsampler import ImbalancedDatasetSampler
            data_sampler = ImbalancedDatasetSampler(data_set)
            shuffle_flag = False
        else:
            data_sampler = torch.utils.data.RandomSampler(data_set)

        if args.info_batch and flag == 'TRAIN':
            print('Use info batch.')
            from utils.infobatch import InfoBatch
            data_set = InfoBatch(data_set, args.train_epochs, args.info_batch_ratio, args.info_batch_delta)
            data_loader = DataLoader(
                data_set,
                batch_size=batch_size,
                shuffle=False,
                sampler=data_set.sampler,
                num_workers=0,
                drop_last=drop_last)
        else:
            data_loader = DataLoader(
                data_set,
                batch_size=batch_size,
                shuffle=shuffle_flag,
                num_workers=args.num_workers,
                drop_last=drop_last,
                collate_fn=lambda x: collate_fn(x, max_len=args.seq_len),
                sampler=data_sampler if flag == 'train' else None
            )
        return data_set, data_loader
    else:
        if args.data == 'm4':
            drop_last = False
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns,
            args=args
        )
        print(flag, len(data_set))
        if args.info_batch and flag == 'train':
            print('Use info batch.')
            from utils.infobatch import InfoBatch
            data_set = InfoBatch(data_set, args.train_epochs, args.info_batch_ratio, args.info_batch_delta)
            data_loader = DataLoader(
                data_set,
                batch_size=batch_size,
                shuffle=False,
                sampler=data_set.sampler,
                num_workers=0,
                drop_last=drop_last)
        else:
            data_loader = DataLoader(
                data_set,
                batch_size=batch_size,
                shuffle=shuffle_flag,
                num_workers=args.num_workers,
                drop_last=drop_last)
        return data_set, data_loader


class MaskedDataLoaderAdapter:
    r"""https://github.com/thuml/Transfer-Learning-Library/blob/master/tllib/utils/data.py#L40"""

    def __init__(self, data_loader: DataLoader, n_covariate=None, n_target=None, fixed_mode=False, normal_mode=False):
        self.data_loader = data_loader
        self.cov_keep_range = [0, n_covariate]
        self.tgt_keep_range = [1, n_target]

        self.n_covariate = n_covariate
        self.n_target = n_target
        self.n_tgt_keep = n_target

        self.normal_mode = normal_mode
        self.fixed_mode = fixed_mode
        self.indices = self.get_masked_indices()

    def __iter__(self):
        if self.normal_mode:
            for batch in self.data_loader:
                batch = - self.n_tgt_keep, *batch
                yield batch
        else:
            for batch in self.data_loader:
                if not self.fixed_mode:
                    self.indices = self.get_masked_indices()
                # if self.fixed_mode:
                #     print(batch[0].shape, self.indices)

                tmp = [-self.n_tgt_keep]
                for i, x in enumerate(batch):
                    # print(x.shape)
                    # print(x.shape, x[:, :, self.indices].shape)
                    if i < 2:
                        x = x[:, :, self.indices]
                    tmp.append(x)
                batch = tmp
                yield batch

    def get_masked_indices(self, debug_mode=False):
        n_cov_keep = np.random.randint(*self.cov_keep_range)
        indices_cov = np.random.permutation(self.n_covariate)[:n_cov_keep]
        indices_cov.sort()

        n_tgt_keep = np.random.randint(*self.tgt_keep_range)
        self.n_tgt_keep = n_tgt_keep

        indices_tgt = np.random.permutation(self.n_target)[:n_tgt_keep] + self.n_covariate
        indices_tgt.sort()

        indices = np.concatenate((indices_cov, indices_tgt))
        # print(indices)
        return indices.tolist()

    #
    # def get_masked_indices(self):
    #     n_cov_keep = np.random.randint(*self.cov_keep_range)
    #     indices_cov = torch.randperm(self.n_covariate)[:n_cov_keep]
    #     indices_cov, _ = torch.sort(indices_cov, descending=False)
    #
    #     n_tgt_keep = np.random.randint(*self.tgt_keep_range)
    #     self.n_tgt_keep = n_tgt_keep
    #     # print(n_cov_keep)
    #     # print(n_tgt_keep)
    #     indices_tgt = torch.randperm(self.n_target)[:n_tgt_keep] + self.n_covariate
    #     # print(indices_tgt)
    #     indices_tgt, _ = torch.sort(indices_tgt, descending=False)
    #
    #     indices = indices_cov.tolist() + indices_tgt.tolist()
    #     return indices

    @staticmethod
    def get_masked_data(data, mask_ratio):
        if mask_ratio == 0:
            return data  # No need to mask if mask_ratio is 0

        length = data.shape[-1]
        num_to_keep = int(length * (1 - mask_ratio))
        indices = torch.randperm(length)[:num_to_keep]
        indices = torch.sort(indices, descending=False)

        return data[:, :, indices]

    def __next__(self):
        try:
            data = next(self.iter)
        except StopIteration:
            self.iter = iter(self.data_loader)
            data = next(self.iter)
        return data

    def __len__(self):
        return len(self.data_loader)

def data_provider_list(args, flag):
    Data = data_dict[args.data]
    if args.embed in ['fixed', 'learned']:
        timeenc = 0
    elif args.embed == 'timeF':
        timeenc = 1
    else:
        timeenc = 2

    if flag == 'test':
        shuffle_flag = False
        # drop_last = True
        # if args.task_name == 'anomaly_detection' or args.task_name == 'classification':
        #     batch_size = args.batch_size
        # else:
        #     batch_size = 1  # bsz=1 for evaluation
        # # set by ez.
        drop_last = False
        batch_size = args.batch_size
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq

    if args.task_name == 'anomaly_detection':
        drop_last = False
        data_set = Data(
            root_path=args.root_path,
            win_size=args.seq_len,
            flag=flag,
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
    elif args.task_name == 'classification':
        drop_last = False
        data_set = Data(
            root_path=args.root_path,
            flag=flag,
        )

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
        )
        return data_set, data_loader
    else:
        if args.data == 'm4':
            drop_last = False
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns,
            args=args
        )
        print(flag, len(data_set))
        if flag == 'train':
            if args.info_batch and flag == 'train':
                print('Use info batch.')
                from utils.infobatch import InfoBatch
                data_set = InfoBatch(data_set, args.train_epochs, args.info_batch_ratio, args.info_batch_delta)
                data_loader = DataLoader(
                    data_set,
                    batch_size=batch_size,
                    shuffle=False,
                    sampler=data_set.sampler,
                    num_workers=0,
                    drop_last=drop_last)
            else:
                data_loader = DataLoader(
                    data_set,
                    batch_size=batch_size,
                    shuffle=shuffle_flag,
                    num_workers=args.num_workers,
                    drop_last=drop_last)
                data_loader = MaskedDataLoaderAdapter(data_loader,
                                                      n_covariate=(args.enc_in - args.num_target),
                                                      n_target=args.num_target,
                                                      fixed_mode=False,
                                                      normal_mode=not args.masked_training)
        else:
            from utils.tools import set_seed
            set_seed(args.seed)
            # get the patient list
            data_set_list = data_set.dataset_map[flag]
            data_loader_list = []
            for data_set in data_set_list:
                data_loader = DataLoader(
                    data_set,
                    batch_size=min(batch_size, len(data_set)),  # avoid too small data length
                    shuffle=False,
                    num_workers=0,
                    drop_last=drop_last)
                data_loader = MaskedDataLoaderAdapter(data_loader,
                                                      n_covariate=(args.enc_in - args.num_target),
                                                      n_target=args.num_target,
                                                      fixed_mode=True,
                                                      normal_mode=not args.masked_training)
                data_loader_list.append(data_loader)
            data_set = data_set_list
            data_loader = data_loader_list

        return data_set, data_loader
