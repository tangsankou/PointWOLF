"""
@Origin : main.py by Yue Wang
@Contact: yuewangx@mit.edu
@Time: 2018/10/13 10:39 PM

modified by {Sanghyeok Lee, Sihyeon Kim}
@Contact: {cat0626, sh_bs15}@korea.ac.kr
@File: main.py
@Time: 2021.09.30
"""


from __future__ import print_function
import os
import argparse
import torch

from util import IOStream
from train import train_vanilla, train_AugTune, test, test_c

def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/'+args.exp_name):
        os.makedirs('checkpoints/'+args.exp_name)
    if not os.path.exists('checkpoints/'+args.exp_name+'/'+'models'):
        os.makedirs('checkpoints/'+args.exp_name+'/'+'models')
    os.system('cp main.py checkpoints'+'/'+args.exp_name+'/'+'main.py.backup')
    os.system('cp model.py checkpoints' + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('cp util.py checkpoints' + '/' + args.exp_name + '/' + 'util.py.backup')
    os.system('cp data.py checkpoints' + '/' + args.exp_name + '/' + 'data.py.backup')
    os.system('cp PointWOLF.py checkpoints' + '/' + args.exp_name + '/' + 'PointWOLF.py.backup')
    os.system('cp train.py checkpoints' + '/' + args.exp_name + '/' + 'train.py.backup')

###add for test modelnet40c
MAP = ['uniform',
        'gaussian',
       'background',
       'impulse',
    #    'scale',
       'upsampling',
       'shear',
       'rotation',
       'cutout',
       'density',
       'density_inc',
       'distortion',
       'distortion_rbf',
       'distortion_rbf_inv',
       'occlusion',
       'lidar',
       'original'
]



if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='exp_PointWOLF', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                        choices=['pointnet', 'dgcnn'],
                        help='Model to use, [pointnet, dgcnn]')
    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
                        choices=['modelnet40'])
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=True,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # parser.add_argument('--eval', type=bool,  default=False,
    #                     help='evaluate the model')
    ###changed
    parser.add_argument('--eval', type=str,  default='train', metavar='N',
                        choices=['train','test','testc'],
                        help='train or test or testc')
    ###end
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')

    # PointWOLF settings
    parser.add_argument('--PointWOLF', action='store_true', help='Use PointWOLF')
    
    parser.add_argument('--w_num_anchor', type=int, default=4, help='Num of anchor point' ) 
    parser.add_argument('--w_sample_type', type=str, default='fps', help='Sampling method for anchor point, option : (fps, random)') 
    parser.add_argument('--w_sigma', type=float, default=0.5, help='Kernel bandwidth')  

    parser.add_argument('--w_R_range', type=float, default=10, help='Maximum rotation range of local transformation')
    parser.add_argument('--w_S_range', type=float, default=3, help='Maximum scailing range of local transformation')
    parser.add_argument('--w_T_range', type=float, default=0.25, help='Maximum translation range of local transformation')
    
    # AugTune settings
    parser.add_argument('--AugTune', action='store_true', help='Use AugTune')
    parser.add_argument('--l', type=float, default=0.1, help='Difficulty parameter lambda')

    ###add for test or testc
    parser.add_argument('--log', type=str, default='train', metavar='N', help='Name of log_dir')
    
    args = parser.parse_args()

    _init_()

    ###the origin
    # io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    ###change for test or testc
    io = IOStream('checkpoints/' + args.exp_name + '/' +args.log + '.log')
    ###end

    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    ###the origin code
    """ if not args.eval:
        if args.AugTune:
            train_AugTune(args, io)
        else:
            train_vanilla(args, io)
    else:
        test(args, io) """
    ###end
    ###change for test modelnet40c
    if args.eval=='train':
        if args.AugTune:
            train_AugTune(args, io)
        else:
            train_vanilla(args, io)
    elif args.eval=='test':
        model_path = 'checkpoints/' + args.exp_name + '/models/model.t7'
        test(args, io, model_path)
    elif args.eval=='testc':
    # data_path = "/home/user_tp/workspace/data/ModelNet40-C/data_uniform_1.npy"
        label_path = "/home/user_tp/workspace/data/ModelNet40-C/label.npy"
        model_path = 'checkpoints/' + args.exp_name + '/models/model.t7'
        ###
        for cor in MAP:
            if cor in ['original']:
                data_path = "/home/user_tp/workspace/data/ModelNet40-C/data_" + cor + ".npy"
                with torch.no_grad():
                    instance_acc, class_acc = test_c(args, io, model_path, data_path, label_path)
                # instance_acc, class_acc, weights = test(classifier.eval(), testDataLoader, vote_num=args.num_votes, num_class=num_class)
                    outstr = 'data_%s: ,Test Instance Accuracy: %f, Class Accuracy: %f' % (cor, instance_acc, class_acc)
                    io.cprint(outstr)
            #     continue
            else:
                for sev in [1,2,3,4,5]:
                    data_path = "/home/user_tp/workspace/data/ModelNet40-C/data_" + cor + "_" + str(sev) + ".npy"
            
                    with torch.no_grad():
                        instance_acc, class_acc = test_c(args, io, model_path, data_path, label_path)
                    # instance_acc, class_acc, weights = test(classifier.eval(), testDataLoader, vote_num=args.num_votes, num_class=num_class)
                        outstr = 'data_%s_%s: Test Instance Accuracy: %f, Class Accuracy: %f' % (cor, str(sev), instance_acc, class_acc)
                        io.cprint(outstr)
                        # log_string('data_%s_%s: Test Instance Accuracy: %f, Class Accuracy: %f' % (cor, str(sev), instance_acc, class_acc))
            ###
