from losses import hardloss
import PhotoTour
from models import SDGMNet
import torch
from torch import optim
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from Utils import transform_train, transform_test, FileLogger
import EvalMetrics
import os
import math
from torch.utils.tensorboard import SummaryWriter
from argparse import ArgumentParser
import shutil
import warnings
import random
test_BS = 4096
check_step = 100
setlist = ('liberty', 'yosemite', 'notredame')
abblist = ('li', 'yo', 'no')


def CreateTestLoader(args, name='yosemite'):
    TestData = PhotoTour.PhotoTour(
        root=args.datadir, name=name, download=args.download, train=False, transform=transform_test)
    Testloader = torch.utils.data.DataLoader(TestData, batch_size=test_BS,
                                             shuffle=False, num_workers=args.nw, pin_memory=True)
    return Testloader


def CreateTrainLoader(args, name='liberty'):
    if args.augment:
        transform = transform_train
        fliprot = True
    else:
        transform = transform_test
        fliprot = False
    TrainData = PhotoTour.TrainBatchPhotoTour(root=args.datadir, name=name, download=args.download, fliprot=fliprot,
                                              Mode='pair', transform=transform, batch_size=int(args.bs), num_triplets=args.ntri)
    Trainloader = torch.utils.data.DataLoader(TrainData, batch_size=int(args.bs),
                                              shuffle=False, num_workers=args.nw, pin_memory=True, drop_last=True)
    return Trainloader, TrainData


def opt_adjuster(args, optimizer, epoch):
    """Updates the learning rate given the learning rate decay.
    The routine has been implemented according to the original Lua SGD optimizer
    """
    for group in optimizer.param_groups:
        if epoch % int(args.epoches*0.1) == 0 and epoch > 1:
            group['lr'] = group['lr']/2
        elif epoch == 0:
            group['lr'] = args.lr
    return


def savemodel(modeldir, abb, model, optimizer, new_FPR=1, old_FPR=0, epoch=0):
    optimizer.zero_grad()
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'FPR': new_FPR,
        'opt': optimizer.state_dict()
    }, modeldir+'.pt')
    if new_FPR < old_FPR:
        if os.path.exists(modeldir+'_best.pt'):
            checkpoint = torch.load(modeldir+'_best.pt')
            old_FPR = checkpoint['FPR']
        if new_FPR < old_FPR:
            print('save the best')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'FPR': new_FPR,
                'opt': optimizer.state_dict()
            }, modeldir+'_best.pt')


def test(model, testloader, GPU=True):
    model.eval()
    with torch.no_grad():
        if GPU:
            simi = torch.zeros(0).cuda()
        else:
            simi = torch.zeros(0)
        lbl = torch.zeros(0)
        for i, (data1, data2, m) in enumerate(testloader):
            if GPU:
                data1 = data1.cuda(non_blocking=True)
                data2 = data2.cuda(non_blocking=True)
            t1 = model(data1)
            t2 = model(data2)
            t3 = torch.sum(t1*t2, dim=1).detach().view(-1)
            simi = torch.cat((simi, t3), dim=0)
            lbl = torch.cat((lbl, m.view(-1)), dim=0)
        lbl = lbl.numpy()
        simi = simi.cpu().numpy()
        FPR = EvalMetrics.ErrorRateAt95Recall(labels=lbl, scores=simi+10)
    return FPR


def train(args):
    for j, abb in enumerate(abblist):
        print('---------------------------------------------------------------------')
        t1 = list(setlist)
        t2 = list(abblist)
        train_set = setlist[j]
        if args.train_set != '' and train_set != args.train_set:
            continue
        t1.pop(j)
        t2.pop(j)
        test_set1 = t1[0]
        test_set2 = t1[1]
        abb1 = t2[0]
        abb2 = t2[1]
        TestLoader1 = CreateTestLoader(args, test_set1)
        TestLoader2 = CreateTestLoader(args, test_set2)
        TrainLoader, TrainData = CreateTrainLoader(args, train_set)

        modeldir = args.savedir+args.name+'/models/Model_tr_' + abb
        model = SDGMNet(drop_rate=args.dr)
        if args.cuda:
            model = model.cuda()
        opt = optim.SGD(model.parameters(), lr=args.lr,
                        momentum=0.9, weight_decay=args.wd)
        if args.adam:
            opt = optim.Adam(model.parameters(), lr=args.lr)
        # if os.path.exists(modeldir+'.pt'):
        #     continue
        # savemodel(modeldir, abb, model, opt, 1.0, 1.0, -1)
        FPR_best = 1.0
        start_epoch = 0
        if args.resume:
            try:
                checkpoint = torch.load(modeldir+'.pt')
                model.load_state_dict(
                    checkpoint['model_state_dict'])
                print('resume the model successfully: epoch {}, FPR {}'.format(
                    start_epoch, FPR_best))
                try:
                    opt.load_state_dict(checkpoint['opt'])
                    opt.zero_grad()
                except:
                    warnings.warn(
                        "fail to resume the optimizer, so the optimizer would be initialized")
                FPR_best = checkpoint['FPR']
                start_epoch = checkpoint['epoch']+1
            except:
                warnings.warn(
                    "warning: fail to resume the old model and begin to train a raw model")
                args.resume = False
        if start_epoch >= args.epoches:
            print('model is alreay trained')
            print('---------------------------------------------------------------------')
            continue
        if os.path.exists('./runs/'+args.name+'_tr_'+abb) and not args.resume:
            shutil.rmtree('./runs/'+args.name+'_tr_'+abb)
        writer = SummaryWriter('./runs/'+args.name +
                               '_tr_'+abb, purge_step=start_epoch)
        logger = FileLogger(args.savedir+args.name + '/loggers/logger_')
        learningMetric = hardloss(margin=1.0)
        counter = start_epoch*args.ntri*1.0
        print('train the model on the subset {}'.format(train_set))
        print('test the model on the subsets {} and {}'.format(test_set1, test_set2))
        print('train starts from {} to {}'.format(start_epoch, args.epoches))
        print('best FPR is {}'.format(FPR_best))
        for epoch in range(start_epoch, args.epoches):
            # train
            model.train()
            TrainData.generate_newdata()
            pbar = tqdm(enumerate(TrainLoader))
            running_loss = 0
            running_pos_mean = 0
            running_neg_std = 0
            if not args.adam:
                opt_adjuster(args, opt, epoch)
            for i, (anchor, pos) in pbar:
                timestep = counter*1.0/(args.ntri * float(args.epoches))
                opt.zero_grad()
                if args.cuda:
                    anchor = anchor.cuda(non_blocking=True)
                    pos = pos.cuda(non_blocking=True)
                anchor_des = model(anchor)
                pos_des = model(pos)
                loss, pos_mean, neg_std = learningMetric(
                    anchor_des, pos_des, timestep)
                loss.backward()
                opt.step()
                counter += args.bs
                running_loss += loss.detach()
                running_pos_mean += pos_mean.detach()
                running_neg_std += neg_std.detach()
                if i % int(check_step) == 0:
                    pbar.set_description(
                        'Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                            epoch, i * len(anchor), len(TrainLoader.dataset),
                            100. * i / len(TrainLoader),
                            loss.item()))
            print(running_loss.item()/(i+1))
            logger.log_string(abb, '\n')
            logger.log_stats(abb, '  epoch', epoch)
            logger.log_stats(abb, '  batchsize', args.bs)
            logger.log_stats(abb, '  loss', running_loss.item()/(i+1))
            logger.log_stats(abb, '  pos_mean', running_pos_mean.item()/(i+1))
            logger.log_stats(abb, '  neg_std', running_neg_std.item()/(i+1))
            writer.add_scalar('loss', running_loss.item()/(i+1), epoch)
            writer.add_scalar('pos_mean', running_pos_mean.item()/(i+1), epoch)
            writer.add_scalar('neg_std', running_neg_std.item()/(i+1), epoch)
            # test
            FPR1_95 = test(model, TestLoader1, args.cuda)
            print(FPR1_95)
            FPR2_95 = test(model, TestLoader2, args.cuda)
            print(FPR2_95)
            FPR_new = (FPR1_95 + FPR2_95)/2
            print(FPR_new)
            logger.log_stats(abb, '   ' + abb1 + 'FPR@95', FPR1_95)
            logger.log_stats(abb, '   ' + abb2 + 'FPR@95', FPR2_95)
            writer.add_scalar('FPR', FPR_new, epoch)
            # save
            savemodel(modeldir, abb, model, opt, FPR_new, FPR_best, epoch)
            if FPR_new < FPR_best:
                FPR_best = FPR_new
        writer.close()
        print('train fininshed')
        print('---------------------------------------------------------------------')


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--epoches', type=int, default=200,
                        help="the number of train epoches. Default: 200")
    parser.add_argument('--bs', type=int,
                        default=1024, help="train batchsize. Default: 1024")
    parser.add_argument('--disable_augment', action='store_true',
                        help="do data augment. Default: True")
    parser.add_argument('--nw', type=int, default=12,
                        help=" the number of parallel threads. Default: 12")
    parser.add_argument('--lr', type=float, default=1,
                        help="initial learning rate")
    parser.add_argument('--datadir', default="./Datasets/UBC",
                        help="directory of dataset")
    parser.add_argument('--enable_resume', action='store_true',
                        help="load the trained model. Default: True")
    parser.add_argument('--name', type=str,
                        default='', help="model name")
    parser.add_argument('--savedir', type=str,
                        default="./Checkpoints/", help="where to save train data. Default: ./data/")
    parser.add_argument('--disable_cuda', action='store_true',
                        help="running on CPU or GPU. Default: True")
    parser.add_argument('--gpus', type=str, default="auto",
                        help="GPU id or 'auto'. Default: auto")
    parser.add_argument('--ntri', type=int, default=1024000,
                        help="number of triplets generated in one epoch. Default: 1024000")
    parser.add_argument('--enable_download', action='store_true',
                        help="whether download dataset. Default: False")
    parser.add_argument('--enable_adam',action='store_true',
                        help="whether to use adam optimizer. Default: False")
    parser.add_argument('--dr', type=float, default=0.3,
                        help="drop rate before last convolution layer. Default: 0.3")
    parser.add_argument('--suffix', type=str, default='',
                        help="the suffix name. Default: ''")
    parser.add_argument('--seed', type=int, default=1,
                        help="set random seed of torch, np and random. Default: 1")
    parser.add_argument('--wd', type=float, default=0.0001,
                        help="weight decay. Default: 0.0001")
    parser.add_argument('--train_set', type=str, default='',
                        help="training subset of UBC. Default: ")
    args = parser.parse_args()

    args.cuda = not args.disable_cuda
    args.download = args.enable_download
    args.adam = args.enable_adam
    args.resume = args.enable_resume
    args.augment = not args.disable_augment
    print(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    #torch.backends.cudnn.deterministic = True
    #torch.cuda.manual_seed_all(0)
    if args.name == '':
        args.name = 'HardFRN'.format(args.suffix)

    gpus = args.gpus
    if args.gpus == 'auto':
        gpus = ''
        from pynvml import *
        nvmlInit()
        deviceCount = nvmlDeviceGetCount()
        for i in range(deviceCount):
            handle = nvmlDeviceGetHandleByIndex(i)
            info = nvmlDeviceGetMemoryInfo(handle)
            usage = info.used*1.0/info.total
            if usage < 0.01:
                print('{}'.format(i))
                gpus = '{}'.format(i)
                if args.cuda:
                    print('GPU {} will be used'.format(i))
                nvmlShutdown()
                break
        nvmlShutdown()
        assert gpus != '' "No free gpu to use. You should set 'cuda' to false"
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus

    if not os.path.exists(args.savedir+args.name):
        os.makedirs(args.savedir+args.name)
        os.makedirs(args.savedir+args.name+'/loggers')
        os.makedirs(args.savedir+args.name+'/models')

    train(args)
