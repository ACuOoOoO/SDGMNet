import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from scipy import optimize
from torch.distributions import normal


class SDGMloss(nn.Module):
    def __init__(self, bs=1024, GPU=True, alpha=0.9, prob_margin=0.6, resume=False, logger=None):
        super(SDGMloss, self).__init__()
        self.bs = bs
        self.dim = 128
        self.negthres = 0.99
        self.eps = 1e-5
        self.alpha = alpha
        self.prob_margin = torch.FloatTensor([prob_margin])
        self.resume = resume
        if GPU:
            self.prob_margin = self.prob_margin.cuda()
            self.eye = torch.eye(bs).cuda()
            self.offdiagmask = (1-self.eye).bool()
        else:
            self.eye = torch.eye(bs)
            self.offdiagmask = (1-self.eye)
        self.neg_power = 10000
        self.pos_power = 10000
        self.pos_ang_mean = -1
        self.neg_ang_mean = -1
        self.diff_ang_mean = -1
        self.diff_ang_std = -1
        self.pos_ang_std = -1
        self.neg_ang_std = -1
        self.lam = 0.999
        self.logger = logger
        self.pos_ang = None
        self.neg_ang = None
        if self.logger is not None:
            self.batch_pos_power = 0
            self.batch_neg_power = 0
            self.batch_diff_mean = 0
            self.batch_diff_std = 0
            self.batch_pos_mean = 0
            self.batch_neg_mean = 0
            self.batch_pos_std = 0
            self.batch_neg_std = 0
        print('bias is {}'.format(self.alpha))
        print('prob margin is {}'.format(self.prob_margin.item()))
        print('resume is {}'.format(self.resume))

    def forward(self, anchor, positive, timestep=-1):
        simi_matrix = torch.mm(anchor, torch.t(positive))
        simi_matrix_a = torch.mm(anchor, torch.t(anchor))
        simi_matrix_p = torch.mm(positive, torch.t(positive))
        # trick for little improvement
        if timestep < 0.125:
            self.negthres = 1-self.eps
        else:
            self.negthres = np.cos(0.6)
        # get the neg
        simi_without_max_on_diag = simi_matrix-10*self.eye
        simi_without_max_on_diag_a = simi_matrix_a-10*self.eye
        simi_without_max_on_diag_p = simi_matrix_p-10*self.eye
        max_simi_matrix0 = torch.max(
            simi_without_max_on_diag, torch.t(simi_without_max_on_diag))
        max_simi_matrix1 = torch.max(
            max_simi_matrix0, simi_without_max_on_diag_a)
        max_simi_matrix2 = torch.max(
            max_simi_matrix1, simi_without_max_on_diag_p)
        mask0 = 10.0*max_simi_matrix2.ge(self.negthres)
        max_simi_matrix2 = max_simi_matrix2 - mask0
        neg = torch.max(max_simi_matrix2, dim=0)[0]
        # get the pos
        pos = torch.diag(simi_matrix)
        pos = torch.clamp(pos, max=1-self.eps)
        # weighting
        # ---------------------------------------------
        # coupled weighting
        neg_ang = torch.acos(neg).detach()
        pos_ang = torch.acos(pos).detach()
        self.neg_ang = neg_ang
        self.pos_ang = pos_ang

        # wc
        diff_ang = (pos_ang - neg_ang)
        diff_ang_mean = torch.mean(diff_ang)
        diff_ang_std = torch.std(diff_ang)
        if self.diff_ang_std < 0 or self.resume:
            self.diff_ang_std = diff_ang_std
            if self.resume:
                self.diff_ang_mean = diff_ang_mean
        else:
            self.diff_ang_mean = self.lam * \
                self.diff_ang_mean + (1-self.lam)*diff_ang_mean
            self.diff_ang_std = self.lam * \
                self.diff_ang_std + (1-self.lam)*diff_ang_std
        gaussian = normal.Normal(self.diff_ang_mean, self.diff_ang_std)
        wc = 1
        margin = gaussian.icdf(self.prob_margin)
        wc = (diff_ang.ge(
            margin).float())*wc
        if timestep < 0.1:
            wc = 1.0

        # ws
        pos_ang_mean = torch.mean(pos_ang)
        pos_ang_std = torch.std(pos_ang)
        neg_ang_mean = torch.mean(neg_ang)
        neg_ang_std = torch.std(neg_ang)
        if self.pos_ang_mean < 0 or self.resume:
            self.pos_ang_mean = pos_ang_mean
            self.neg_ang_mean = neg_ang_mean
            self.pos_ang_std = pos_ang_std
            self.neg_ang_std = neg_ang_std
        else:
            self.pos_ang_mean = self.lam * \
                self.pos_ang_mean + (1-self.lam)*pos_ang_mean
            self.neg_ang_mean = self.lam * \
                self.neg_ang_mean + (1-self.lam)*neg_ang_mean
            self.pos_ang_std = self.lam * \
                self.pos_ang_std + (1-self.lam)*pos_ang_std
            self.neg_ang_std = self.lam * \
                self.neg_ang_std + (1-self.lam)*neg_ang_std
        ws_p = torch.exp(-(pos_ang-self.pos_ang_mean).pow(2) /
                         2/(np.pi/6+self.pos_ang_std).pow(2))
        ws_n = torch.exp(-(neg_ang-self.neg_ang_mean).pow(2) /
                         2/(np.pi/6+self.neg_ang_std).pow(2))
        if timestep < 0.1:
            ws_p = ws_p*0+1
            ws_n = ws_n*0+1
        wp = ws_p*wc
        wn = ws_n*wc

        # power adjustment
        pos_power = torch.sum(wp)
        neg_power = torch.sum(wn)
        if self.resume:
            self.resume = False
            self.pos_power = pos_power
            self.neg_power = neg_power
        else:
            self.pos_power = self.lam*self.pos_power + (1-self.lam)*pos_power
            self.neg_power = self.lam*self.neg_power + (1-self.lam)*neg_power
        wp = (wp*100/self.pos_power).detach()
        wn = (wn*100/self.neg_power).detach()

        # pseudo loss
        pos_ang_t = torch.acos(pos)
        neg_ang_t = torch.acos(neg)
        loss0 = self.alpha*wp*pos_ang_t - wn*neg_ang_t
        loss = torch.sum(loss0)/100
        # ---------------------------------------------

        # caculate some useful indices
        pos_mean = pos_ang_mean
        neg_t = torch.masked_select(simi_matrix, self.offdiagmask).detach()
        neg_t = torch.clamp(neg_t, min=-1+self.eps, max=1-self.eps)
        neg_t = torch.acos(neg_t)
        neg_std = torch.std(neg_t)

        if self.logger is not None:
            self.batch_diff_mean += diff_ang_mean
            self.batch_diff_std += diff_ang_std
            self.batch_pos_mean += pos_ang_mean
            self.batch_neg_mean += neg_ang_mean
            self.batch_pos_power += pos_power
            self.batch_neg_power += neg_power
            self.batch_pos_std += pos_ang_std
            self.batch_neg_std += neg_ang_std
        return loss, pos_mean, neg_std

    def save_data(self, epoch, iters, suffix):
        self.logger.log_stats(suffix, '  epoch', epoch)
        self.logger.log_stats(suffix, '  diff_mean',
                              self.batch_diff_mean.item()/iters)
        self.logger.log_stats(suffix, '  diff_std',
                              self.batch_diff_std.item()/iters)
        self.logger.log_stats(suffix, '  pos_mean',
                              self.batch_pos_mean.item()/iters)
        self.logger.log_stats(suffix, '  neg_mean',
                              self.batch_neg_mean.item()/iters)
        self.logger.log_stats(suffix, '  pos_power',
                              self.batch_pos_power.item()/iters)
        self.logger.log_stats(suffix, '  neg_power',
                              self.batch_neg_power.item()/iters)
        self.logger.log_stats(suffix, '  pos_std',
                              self.batch_pos_std.item()/iters)
        self.logger.log_stats(suffix, '  neg_std',
                              self.batch_neg_std.item()/iters)
        self.logger.log_string(suffix, '\n')
        self.batch_diff_mean = 0
        self.batch_diff_std = 0
        self.batch_pos_mean = 0
        self.batch_neg_mean = 0
        self.batch_pos_power = 0
        self.batch_neg_power = 0
        self.batch_pos_std = 0
        self.batch_neg_std = 0


class hardloss(nn.Module):
    '''
    HardNet loss from 'Working hard to know your neighbor’s margins:
    Local descriptor learning loss'(https://github.com/DagnyT/hardnet).
    '''

    def __init__(self, bs=1024, margin=1, GPU=True):
        super(hardloss, self).__init__()
        self.bs = bs
        self.eps = 1e-5
        self.negthres = np.cos(0.6)
        self.margin = margin
        if GPU:
            self.eye = torch.eye(bs).cuda()
            self.offdiagmask = (1-self.eye).bool()
        else:
            self.eye = torch.eye(bs)
            self.offdiagmask = (1-self.eye)

    def forward(self, anchor, positive, timestep=-1):
        simi_matrix = torch.mm(anchor, torch.t(positive))
        simi_matrix_a = torch.mm(anchor, torch.t(anchor))
        simi_matrix_p = torch.mm(positive, torch.t(positive))

        simi_without_max_on_diag = simi_matrix-10*self.eye
        simi_without_max_on_diag_a = simi_matrix_a-10*self.eye
        simi_without_max_on_diag_p = simi_matrix_p-10*self.eye
        max_simi_matrix = torch.max(
            simi_without_max_on_diag, torch.t(simi_without_max_on_diag))
        max_simi_matrix = torch.max(
            max_simi_matrix, simi_without_max_on_diag_a)
        max_simi_matrix = torch.max(
            max_simi_matrix, simi_without_max_on_diag_p)
        mask = max_simi_matrix.ge(self.negthres)*10.0
        max_simi_matrix = max_simi_matrix - mask

        neg = torch.max(max_simi_matrix, dim=1)[0]

        pos = torch.diag(simi_matrix)
        pos = torch.clamp(pos, max=1.0)

        neg_dist = torch.sqrt(2.0-2*neg+self.eps)
        pos_dist = torch.sqrt(2.0-2*pos+self.eps)
        mask = (self.margin+pos_dist-neg_dist).ge(0)
        loss0 = torch.masked_select(pos_dist-neg_dist, mask)

        loss = torch.mean(loss0)
        # caculate some useful indices
        pos_t = torch.acos(pos-self.eps).detach()
        pos_mean = torch.mean(pos_t)
        neg_t = torch.masked_select(simi_matrix, self.offdiagmask).detach()
        neg_t = torch.acos(neg_t-self.eps)
        neg_std = torch.std(neg_t)
        return loss, pos_mean, neg_std


class HyLoss(nn.Module):
    '''
    HyNet main loss from 'HyNet: Learning Local Descriptor with Hybrid 
    Similarity Measure and Triplet Loss' (https://github.com/yuruntian/
    HyNet#hynet-learning-local-descriptor-with-hybrid-similarity-measur
    e-and-triplet-loss)
    '''

    def __init__(self, bs=1024, GPU=True, alpha=2, margin=1.2):
        super(HyLoss, self).__init__()
        self.bs = bs
        self.alpha = alpha
        self.margin = margin
        self.z = self.calculateMaxGrad().item()
        print(self.z)
        self.dim = 128
        self.negthres = 0.9
        if GPU:
            self.eye = torch.eye(bs).cuda()
            self.offdiagmask = (1-self.eye).bool()
        else:
            self.eye = torch.eye(bs)
            self.offdiagmask = (1-self.eye).bool()

    def HybridGrad(self, theta):
        return self.alpha*np.sin(theta)+np.sin(theta)/np.sqrt(2*(1-np.cos(theta)+1e-8))

    def calculateMaxGrad(self):
        minimum = optimize.fmin(lambda x: -self.HybridGrad(x), 1)
        print(minimum[0])
        return self.HybridGrad(minimum)

    def forward(self, anchor, positive, timestep):
        simi_matrix = torch.mm(anchor, torch.t(positive))
        simi_matrix_a = torch.mm(anchor, torch.t(anchor))
        simi_matrix_p = torch.mm(positive, torch.t(positive))

        simi_without_max_on_diag = simi_matrix-10*self.eye
        simi_without_max_on_diag_a = simi_matrix_a-10*self.eye
        simi_without_max_on_diag_p = simi_matrix_p-10*self.eye

        max_simi_matrix = torch.max(
            simi_without_max_on_diag, torch.t(simi_without_max_on_diag))
        max_simi_matrix = torch.max(
            max_simi_matrix, simi_without_max_on_diag_a)
        max_simi_matrix = torch.max(
            max_simi_matrix, simi_without_max_on_diag_p)
        mask = max_simi_matrix.ge(self.negthres)*10.0
        max_simi_matrix = max_simi_matrix - mask

        neg_simi = torch.max(max_simi_matrix, dim=1)[0]
        pos_simi = torch.diag(simi_matrix)
        neg_dist = torch.sqrt(2-2*neg_simi+1e-8)
        pos_dist = torch.sqrt(2-2*pos_simi+1e-8)
        neg_hy = (self.alpha*(1-neg_simi)+neg_dist)
        pos_hy = (self.alpha*(1-pos_simi)+pos_dist)

        loss0 = torch.clamp(self.margin+pos_hy-neg_hy, min=0.0)
        loss = 1/self.z*torch.mean(loss0)

        # caculate some useful indices
        pos_t = torch.acos(pos_simi-self.eps).detach()
        pos_mean = torch.mean(pos_t)
        neg_t = torch.masked_select(simi_matrix, self.offdiagmask).detach()
        neg_t = torch.acos(neg_t-self.eps)
        neg_std = torch.std(neg_t)

        return loss, pos_mean, neg_std


class CorrelationPenaltyRegulation(nn.Module):
    '''
    CPR from 'L2-net: Deep learning of discriminative patch descriptor in 
    euclidean space' (https://openaccess.thecvf.com/content_cvpr_2017/htm
    l/Tian_L2-Net_Deep_Learning_CVPR_2017_paper.html)
    '''

    def __init__(self, dim, GPU=True):
        super(CorrelationPenaltyRegulation, self).__init__()
        if GPU:
            self.offdiag = 1-1.0*torch.eye(dim).cuda()
        else:
            self.offdiag = 1-1.0*torch.eye(dim)

    def forward(self, input, BN_out=False):
        if BN_out:
            cor_mat = torch.bmm(torch.t(input).unsqueeze(
                0), input.unsqueeze(0)).squeeze(0)/input.size(1)
        else:
            mean0 = torch.mean(input, dim=0)
            zeroed = input - mean0.expand_as(input)
            normed = F.normalize(zeroed, dim=0)
            cor_mat = torch.mm(torch.t(normed), normed)/(normed.size(0)-1)
        no_diag = cor_mat * self.offdiag
        d_sq = no_diag * no_diag
        return torch.mean(d_sq)


class GlobalOrthogonalRegularization(nn.Module):
    '''
    GOR from 'Learning spread-out local feature descriptors'(https
    ://openaccess.thecvf.com/content_iccv_2017/html/Zhang_Learni
    ng_Spread-Out_Local_ICCV_2017_paper.html)
    '''

    def __init__(self):
        super(GlobalOrthogonalRegularization, self).__init__()

    def forward(self, anchor_des, neg_des):
        simi = torch.sum(torch.mul(anchor_des, neg_des), 1)
        dim = anchor_des.size(1)
        gor = torch.pow(torch.mean(simi), 2) + \
            torch.clamp(torch.mean(torch.pow(simi, 2))-1.0/dim, min=0.0)
        return gor


class SecondOrderSimiliarityRegulation(nn.Module):
    '''
    SOSR from 'SOSNet:Second Order Similarity Regularization for 
    Local Descriptor Learning' (https://openaccess.thecvf.com/co
    ntent_CVPR_2019/html/Tian_SOSNet_Second_Order_Similarity_Regu
    larization_for_Local_Descriptor_Learning_CVPR_2019_paper.html)
    '''

    def __init__(self, knn=8, bs=1024, GPU=True, sparse=False):
        super(SecondOrderSimiliarityRegulation, self).__init__()
        self.knn = knn
        print('knn={}'.format(knn))
        self.bs = bs
        self.sparse = sparse
        if GPU:
            self.mask = torch.zeros([bs, bs]).cuda()
            self.indx = torch.from_numpy(
                np.array(range(bs)).reshape(1, -1)).cuda()
        else:
            self.mask = torch.zeros([bs, bs])
            self.indx = torch.from_numpy(np.array(range(bs)).reshape(1, -1))

    def forward(self, AA_DisMat, PP_DisMat):
        AAPP = (AA_DisMat - PP_DisMat + 1e-8).pow(2)
        if self.knn != 0:
            indx_AA = torch.topk(AA_DisMat, self.knn, dim=0)[1]
            indx_PP = torch.topk(PP_DisMat, self.knn, dim=0)[1]
            self.mask = self.mask*0+1e-8
            self.mask = self.mask.scatter_(0, indx_AA, 1)
            self.mask = self.mask.scatter_(0, indx_PP, 1)
            if self.sparse:
                knnind = torch.where(self.mask > 0.5)
                knnind_reshape = torch.cat(
                    (knnind[0], knnind[1]), 0).reshape(2, -1)
                ones = torch.ones(len(knnind[0])).to(knnind_reshape.device)
                m = torch.sparse_coo_tensor(
                    knnind_reshape, ones, (self.bs, self.bs)).coalesce().to(AA_DisMat.device)
                temp0 = AAPP.sparse_mask(m)
                temp1 = torch.sparse.sum(temp0, dim=0).to_dense()
            else:
                temp1 = torch.sum(AAPP*self.mask, dim=0)
        else:
            temp1 = torch.sum(AAPP, dim=0)
        SOS = torch.sqrt(temp1+1e-8)
        SOSR = torch.mean(SOS)
        return SOSR


class MagnitudeRegularization(nn.Module):
    '''
    HyNet regularization from 'HyNet: Learning Local Descriptor with Hybrid 
    Similarity Measure and Triplet Loss' (https://github.com/yuruntian/
    HyNet#hynet-learning-local-descriptor-with-hybrid-similarity-measur
    e-and-triplet-loss)
    '''

    def __init__(self):
        super(MagnitudeRegularization, self).__init__()

    def forward(self, raw_anchor, raw_pos):
        anchor_mag = torch.sqrt(torch.sum(raw_anchor.pow(2), dim=1))
        pos_mag = torch.sqrt(torch.sum(raw_pos.pow(2), dim=1))
        re = torch.mean((anchor_mag-pos_mag).pow(2))
        return re
