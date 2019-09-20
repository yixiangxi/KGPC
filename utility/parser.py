#!/usr/local/bin/bash
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run KGPolicy2.")
    # ------------------------- experimental settings specific for data set --------------------------------------------
    parser.add_argument('--weights_path', nargs='?', default='',
                        help='Input weight path.')
    parser.add_argument('--data_path', nargs='?', default='../Data/',
                        help='Input data path.')
    parser.add_argument('--proj_path', nargs='?', default='',
                        help='Input project path.')
    parser.add_argument('--dataset', nargs='?', default='yelp2018',
                        help='Choose a dataset.')
    parser.add_argument('--pretrain', type=int, default=0,
                        help='0: No pretrain, 1: Pretrain with updating FISM variables, 2:Pretrain with fixed FISM variables.')
    parser.add_argument('--emb_size', type=int, default=64,
                        help='Embedding size.')
    parser.add_argument('--regs', nargs='?', default='1e-5',
                        help='Regularization for user and item embeddings.')
    parser.add_argument('--model_type', nargs='?', default='advnet',
                        help='Specify a loss type (pure_mf or gat_mf).')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='gpu id')
    parser.add_argument('--layer_size', nargs='?', default='[64]',
                        help='Output sizes of every layer')
    parser.add_argument('--k_neg', type=int, default=8,
                        help='number of negative items in list')

    # ------------------------- experimental settings specific for recommender --------------------------------------------
    parser.add_argument('--recommender', type=str, default="MF",
                        help="type for recommender")
    parser.add_argument('--reward_type', nargs='?', default='pure',
                        help='reward function type: pure, prod')
    parser.add_argument('--slr', type=float, default=0.0001,
                        help='Learning rate for sampler.')
    parser.add_argument('--rlr', type=float, default=0.0001,
                        help='Learning rate recommender.')

    # ------------------------- experimental settings specific for sampler --------------------------------------------
    parser.add_argument("--sampler", type=str, default="KGPolicy",
                        help="type for sampler")
    parser.add_argument('--policy_type', nargs='?', default='uj',
                        help='policy function type: uj, uij')
    parser.add_argument('--edge_threshold', type=int, default=8,
                        help='edge threshold to filter knowledge graph')
    parser.add_argument('--in_channel', type=str, default='[64, 32]', 
                        help='input channels for gcn')    
    parser.add_argument('--out_channel', type=str, default='[32, 64]', 
                        help='output channels for gcn')
    parser.add_argument('--num_sample', type=int, default=4,
                        help='number fo samples from gcn')
    parser.add_argument('--pretrained_s', type=bool, default=False,
                        help="load pretrained sampler data or not")
    parser.add_argument('--k_step', type=int, default=2,
                        help="k step from current positive items")

    # ------------------------- experimental settings specific for training --------------------------------------------
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='batch size for training.')
    parser.add_argument('--test_batch_size', type=int, default=1024,
                        help='batch size for test')
    parser.add_argument('--num_threads', type=int, default=4,
                        help='number of threads.')
    parser.add_argument('--epoch', type=int, default=400,
                        help='Number of epoch.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Interval of evaluation.')
    parser.add_argument('--show_step', type=int, default=1,
                        help='test step.')
    parser.add_argument('--adj_epoch', type=int, default=1,
                        help='build adj matrix per _ epoch')
    parser.add_argument('--resume', type=bool, default=False,
                        help="use pretrained model or not")
    parser.add_argument('--freeze_s', type=bool, default=False,
                        help="freeze parameters of recommender or not")
    parser.add_argument('--model_path', type=str, default='model/best_yelp.ckpt',
                        help="path for pretrain model")
    parser.add_argument('--normalize', type=bool, default=False,
                        help="get normalize after embedding")
    parser.add_argument("--out_dir", type=str, default='./weights/',
                        help='output directory for model')
    parser.add_argument("--s_step", type=int, default=1,
                        help="k step for sampler")
    parser.add_argument("--r_step", type=int, default=1,
                        help="k step for recommender")
    parser.add_argument("--filter_edges", type=int, default=8192,
                        help="threshold for filtering node")

    # ------------------------- experimental settings specific for testing ---------------------------------------------
    parser.add_argument('--Ks', nargs='?', default='[20, 40, 60, 80, 100]',
                        help='evaluate K list')
    parser.add_argument('--test_flag', nargs='?', default='part',
                        help='Specify a loss type (org, norm, or mean).')

    return parser.parse_args()
