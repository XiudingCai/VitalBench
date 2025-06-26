import argparse
import os
import torch
from utils.print_args import print_args
import random
import numpy as np
from utils.tools import set_seed


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='TimesNet')

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--seed', type=int, default=3047, help='random seed')
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS, MP]; '
                             'M:multivariate predict multivariate, '
                             'S:univariate predict univariate, '
                             'MS:multivariate predict univariate, '
                             'MP:multivariate predict part of variates ')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--num_target', type=int, default=-1,
                        help='number of target feature in MP task, e.g. -10 for the last 10 features.'
                             'You NEED to bring the target features backward.')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str,
                        default='./checkpoints/',
                        help='location of model checkpoints')
    parser.add_argument('--visualization', type=str,
                        default='./test_results',
                        help='location of model checkpoints')
    parser.add_argument('--results', type=str,
                        default='./results',
                        help='location of model checkpoints')
    parser.add_argument('--max_num_cases', type=int, required=False, default=-1,
                        help='the maximum number of cases for training, validation and test')
    parser.add_argument('--norm_pkl', type=str, default=None, help='root path of the data file')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--max_pred_len', type=int, default=-1, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
    parser.add_argument('--test_all_lengths', action='store_true', help='inverse output data', default=False)

    # imputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%)')

    # model define
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')

    # optimization
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--info_batch', action='store_true', help='use automatic mixed precision training',
                        default=False)
    parser.add_argument('--info_batch_ratio', type=float, default=0.5, help='optimizer learning rate')
    parser.add_argument('--info_batch_delta', type=float, default=0.875, help='optimizer learning rate')

    parser.add_argument('--optimizer', type=str, default='adam', help='optimizer')
    parser.add_argument('--no_lradj', action='store_true', )
    parser.add_argument('--reg_weight', default=1e-5, type=float, help='regularization weight')
    parser.add_argument('--momentum', default=0.99, type=float, help='momentum')
    # update lr scheduler by iter
    parser.add_argument('--lradj_by_iter', action='store_true', )
    parser.add_argument('--warmup_steps', default=0.1, type=float, help='warmup')
    parser.add_argument('--iters_per_epoch', default=None, type=str, help='warmup')
    # peft
    parser.add_argument('--peft_type', type=str, default='full_ft', )
    parser.add_argument('--gpt_layers', type=int, default=6)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    # TiFFomer
    parser.add_argument('--patch_len', type=int, default=None, help='prediction sequence length')
    parser.add_argument('--stride', type=int, default=None, help='prediction sequence length')

    parser.add_argument('--mamba_mode', type=int, default=0, help='prediction sequence length')
    parser.add_argument('--shuffle_mode', type=int, default=0, help='prediction sequence length')
    parser.add_argument('--arrange_mode', type=int, default=0, help='prediction sequence length')
    parser.add_argument('--use_casual_conv', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--use_RoPE', action='store_true', help='use multiple gpus', default=False)

    parser.add_argument('--agg_mode', type=bool, default=False, help='use gpu')
    parser.add_argument('--dropout_mode', type=bool, default=False, help='use gpu')
    parser.add_argument('--test_per_patient', action='store_true', help='test on per patient or not.', default=False)
    parser.add_argument('--masked_training', action='store_true', help='test on per patient or not.', default=False)

    # MambaTS
    parser.add_argument('--RevIN_mode', type=int, default=1,
                        help='0 for not using; 1 for Non-stationary Transformers; 2 for RevIN')
    parser.add_argument('--domain_to_generalize', type=int, default=None,
                        help='1 for VitalDB, 2 for MOVER-SIS')
    parser.add_argument('--VISUAL_ONLY', action='store_true', help='inverse output data', default=False)

    # VitalBench
    parser.add_argument('--data_split_json', type=str, default=None, )
    parser.add_argument('--imputer_type', type=str, default=None, choices=['mean', 'linear', 'MI', 'kNN', 'ffill'])
    parser.add_argument('--training_ratio', default=1, type=float,
                        help='For example, if training_ratio = 0.8, 80% of the training data will be used for training')

    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    # args.use_gpu = False
    if args.iters_per_epoch is not None:
        args.iters_per_epoch = eval(args.iters_per_epoch)

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    if args.agg_mode and args.shuffle_mode == 8:
        args.seed = random.randint(0, 100000)

    set_seed(args.seed)

    print('Args in experiment:')
    print_args(args)

    if args.task_name == 'masked_vital_forecast':
        from exp.exp_masked_vital_forecasting import Exp_Masked_Vital_Forecast
        Exp = Exp_Masked_Vital_Forecast
        args.task_name = 'long_term_forecast'

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            exp = Exp(args)  # set experiments
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des, ii)

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            if args.shuffle_mode in [5]:
                print(f"resetting ids shuffle...")
                exp.model.reset_ids_shuffle()

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            set_seed(args.seed)
            exp.test(setting)

            if args.domain_to_generalize is not None:
                print('>>>>>>>testing on domain generalization : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                set_seed(args.seed)
                exp.test_generalization(setting, test=1)

            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.des, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        set_seed(args.seed)
        exp.test(setting, test=1)

        if args.domain_to_generalize is not None:
            print('>>>>>>>testing on domain generalization : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            set_seed(args.seed)
            exp.test_generalization(setting, test=1)

        if args.test_all_lengths:
            exp.test_for_all_lengths(setting, test=1)

        if args.online_learning_test_DG:
            print('>>>>>>> online learning test mode (DG) : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            set_seed(args.seed)
            exp.test_online_learning_DG(setting, test=1)

        torch.cuda.empty_cache()
