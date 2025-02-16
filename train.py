"""
@Origin : main.py by Yue Wang
@Contact: yuewangx@mit.edu
@Time: 2018/10/13 10:39 PM

modified by {Sanghyeok Lee, Sihyeon Kim}
@Contact: {cat0626, sh_bs15}@korea.ac.kr
@File: train.py
@Time: 2021.09.29
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from torch.utils.data import DataLoader
import sklearn.metrics as metrics
import torch.nn.functional as F

from data import ModelNet40, ModelNetDataLoaderC
from model import PointNet, DGCNN
from util import cal_loss
from tqdm import tqdm

def train_vanilla(args, io):
    train_loader = DataLoader(ModelNet40(args, partition='train'), num_workers=8,
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(ModelNet40(args, partition='test'), num_workers=8,
                             batch_size=args.test_batch_size, shuffle=True, drop_last=False)
    
    device = torch.device("cuda" if args.cuda else "cpu")
    #Try to load models
    if args.model == 'pointnet':
        model = PointNet(args).to(device)
    elif args.model == 'dgcnn':
        model = DGCNN(args).to(device)
    else:
        raise Exception("Not implemented")
    print(str(model))

    model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)
    criterion = cal_loss
    best_test_acc = 0
    
    for epoch in range(args.epochs):
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        train_pred = []
        train_true = []
        for data, label in train_loader:
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            opt.zero_grad()
            logits = model(data)
            loss = criterion(logits, label)
            loss.backward()
            opt.step()
            preds = logits.max(dim=1)[1]
            count += batch_size
            train_loss += loss.item() * batch_size
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())
            
        scheduler.step()
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (epoch,
                                                                                 train_loss*1.0/count,
                                                                                 metrics.accuracy_score(
                                                                                     train_true, train_pred),
                                                                                 metrics.balanced_accuracy_score(
                                                                                     train_true, train_pred))
        io.cprint(outstr)

        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_pred = []
        test_true = []
        for data, label in test_loader:
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            logits = model(data)
            loss = criterion(logits, label)
            preds = logits.max(dim=1)[1]
            count += batch_size
            test_loss += loss.item() * batch_size
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
        outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f' % (epoch,
                                                                              test_loss*1.0/count,
                                                                              test_acc,
                                                                              avg_per_class_acc)
        io.cprint(outstr)
        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), 'checkpoints/%s/models/model.t7' % args.exp_name)
            
            
def train_AugTune(args, io):
    train_loader = DataLoader(ModelNet40(args, partition='train'), num_workers=8,
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(ModelNet40(args, partition='test'), num_workers=8,
                             batch_size=args.test_batch_size, shuffle=True, drop_last=False)
    
    device = torch.device("cuda" if args.cuda else "cpu")
    #Try to load models
    if args.model == 'pointnet':
        model = PointNet(args).to(device)
    elif args.model == 'dgcnn':
        model = DGCNN(args).to(device)
    else:
        raise Exception("Not implemented")
    print(str(model))

    model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)
    criterion = cal_loss
    best_test_acc = 0
    
    for epoch in range(args.epochs):
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        train_pred = []
        train_true = []
        for origin, data, label in train_loader:
            origin, data, label = origin.to(device), data.to(device), label.to(device).squeeze()
            origin = origin.permute(0, 2, 1)
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            
            #Forward original & augmented sample to get confidence score
            with torch.no_grad():
                pred_origin = model(origin)
                pred_data = model(data)
                c_origin = (pred_origin.exp() * F.one_hot(label, pred_origin.shape[-1])).sum(1) #(B)
                c_data = (pred_data.exp() * F.one_hot(label, pred_data.shape[-1])).sum(1) #(B)

            #Calculate Target Confidence Score
            c_target = torch.max((1-args.l) * c_origin, c_data) #(B)
            alpha = ((c_target-c_data)/(c_origin-c_data + 1e-4)).unsqueeze(1) 
            alpha = torch.clamp(alpha, min=0, max=1).reshape(-1,1,1)

            #Tune the Sample with alpha
            data = alpha * origin + (1-alpha) * data
            #Re-normalize Tuned Sample
            data = normalize_point_cloud_batch(data)
            #CDA
            data = translate_pointcloud_batch(data)
                
            opt.zero_grad()
            logits = model(data)
            loss = criterion(logits, label)

            loss.backward()
            opt.step()
            preds = logits.max(dim=1)[1]
            count += batch_size
            train_loss += loss.item() * batch_size
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())
            
        scheduler.step()
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (epoch,
                                                                                 train_loss*1.0/count,
                                                                                 metrics.accuracy_score(
                                                                                     train_true, train_pred),
                                                                                 metrics.balanced_accuracy_score(
                                                                                     train_true, train_pred))
        io.cprint(outstr)

        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_pred = []
        test_true = []
        for data, label in test_loader:
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            logits = model(data)
            loss = criterion(logits, label)
            preds = logits.max(dim=1)[1]
            count += batch_size
            test_loss += loss.item() * batch_size
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
        outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f' % (epoch,
                                                                              test_loss*1.0/count,
                                                                              test_acc,
                                                                              avg_per_class_acc)
        io.cprint(outstr)
        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), 'checkpoints/%s/models/model.t7' % args.exp_name)

            
def test(args, io, model_path):
    test_loader = DataLoader(ModelNet40(args, partition='test'),
                             batch_size=args.test_batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models

    # model = DGCNN(args).to(device)
    if args.model == 'pointnet':
        model = PointNet(args).to(device)
    elif args.model == 'dgcnn':
        model = DGCNN(args).to(device)

    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(model_path))
    model = model.eval()
    test_acc = 0.0
    count = 0.0
    test_true = []
    test_pred = []
    for data, label in test_loader:
        data, label = data.to(device), label.to(device).squeeze()
        data = data.permute(0, 2, 1)
        batch_size = data.size()[0]
        logits = model(data)
        preds = logits.max(dim=1)[1]
        test_true.append(label.cpu().numpy())
        test_pred.append(preds.detach().cpu().numpy())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_acc = metrics.accuracy_score(test_true, test_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    outstr = 'Test :: test acc: %.6f, test avg acc: %.6f'%(test_acc, avg_per_class_acc)
    io.cprint(outstr)


###add for test modelnet40c
def test_c(args, io, model_path, data_path,label_path):
    #model, loader, num_class=40, 
    vote_num=1
    num_class = 40

    test_loader = DataLoader(ModelNetDataLoaderC(data_root=data_path, label_root=label_path, partition='test', num_points=args.num_points),
                                batch_size=args.test_batch_size, shuffle=True, drop_last=False)
    mean_correct = []
    class_acc = np.zeros((num_class, 3))
    print("len(loader):",len(test_loader))

    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models
    if args.model == 'pointnet':
        model = PointNet(args, num_class).to(device)
    elif args.model == 'dgcnn':
        model = DGCNN(args, num_class).to(device)
    else:
        raise Exception("Not implemented")
    print(str(model))
    
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(model_path))
    model = model.eval()

    for j, (data, label) in tqdm(enumerate(test_loader), total=len(test_loader)):

        """ points = provider.random_point_dropout(points)
        points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
        points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3]) """
        # points = torch.Tensor(points)
        data, label = data.to(device), label.to(device).squeeze()

        data = data.transpose(2, 1)
        vote_pool = torch.zeros(label.size()[0], num_class).cuda()

        # print("target::::::",target.shape)#24

        for _ in range(vote_num):
            logits = model(data)
            
            # pred, _ = model(points)
            # pred, _, weights = classifier(points)
            vote_pool += logits
        
        # pred = logits.max(dim=1)[1]
        pred = vote_pool / vote_num
        pred_choice = pred.data.max(1)[1]
        for cat in np.unique(label.cpu()):
            classacc = pred_choice[label == cat].eq(label[label == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item() / float(data[label == cat].size()[0])
            class_acc[cat, 1] += 1
        correct = pred_choice.eq(label.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(data.size()[0]))

    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    class_acc = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct)

    return instance_acc, class_acc
###end

def normalize_point_cloud_batch(pointcloud):
    """
    input : 
        pointcloud([B,3,N])
        
    output :
        pointcloud([B,3,N]) : Normalized Pointclouds
    """
    pointcloud = pointcloud - pointcloud.mean(dim=-1, keepdim=True) #(B,3,N)
    scale = 1/torch.sqrt((pointcloud**2).sum(1)).max(axis=1)[0]*0.999999 # (B)
    pointcloud = scale.view(-1, 1, 1) * pointcloud
    return pointcloud


def translate_pointcloud_batch(pointcloud):
    """
    input : 
        pointcloud([B,3,N])
        
    output :
        translated_pointcloud([B,3,N]) : Pointclouds after CDA
    """
    B, _, _ = pointcloud.shape
    
    xyz1 = torch.FloatTensor(B,3,1).uniform_(2./3., 3./2.).to(pointcloud.device)
    xyz2 = torch.FloatTensor(B,3,1).uniform_(-0.2, 0.2).to(pointcloud.device)
       
    translated_pointcloud = xyz1 * pointcloud + xyz2
    return translated_pointcloud