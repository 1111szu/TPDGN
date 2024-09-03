from utils import dpw3_3d as datasets
from model import AttModel
from utils.opt import Options
from utils import util
from utils import log
import os

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from progress.bar import Bar
import time
import h5py
import torch.optim as optim


def main(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.cuda
    device = util.get_device()
    lr_now = opt.lr_now
    start_epoch = 1
    # opt.is_eval = True
    print('>>> create models')
    in_features = opt.in_features  # 54
    d_model = opt.d_model
    kernel_size = opt.kernel_size
    net_pred = AttModel.AttModel(in_features=in_features, kernel_size=kernel_size, d_model=d_model, num_stage=opt.num_stage, dct_n=opt.dct_n, lamda=opt.lamda, alpha=opt.alpha, n_layer=opt.n_layer, p_dropout=opt.drop_out)
    net_pred.to(device)

    optimizer = optim.Adam(filter(lambda x: x.requires_grad, net_pred.parameters()), lr=opt.lr_now)
    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in net_pred.parameters()) / 1000000.0))

    if opt.is_load or opt.is_eval:
        model_path_len = './{}/ckpt_best.pth.tar'.format(opt.ckpt)
        print(">>> loading ckpt len from '{}'".format(model_path_len))
        ckpt = torch.load(model_path_len)
        start_epoch = ckpt['epoch'] + 1
        err_best = ckpt['err']
        lr_now = ckpt['lr']
        net_pred.load_state_dict(ckpt['state_dict'])
        # net.load_state_dict(ckpt)
        # optimizer.load_state_dict(ckpt['optimizer'])
        # lr_now = util.lr_decay_mine(optimizer, lr_now, 0.2)
        print(">>> ckpt len loaded (epoch: {} | err: {})".format(ckpt['epoch'], ckpt['err']))

    print('>>> loading datasets')

    if not opt.is_eval:
        dataset = datasets.Datasets(opt, split=0)
        print('>>> Training dataset length: {:d}'.format(dataset.__len__()))
        data_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0, pin_memory=False)
        # data_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0, pin_memory=False)
        valid_dataset = datasets.Datasets(opt, split=2)
        print('>>> Validation dataset length: {:d}'.format(valid_dataset.__len__()))
        # valid_loader = DataLoader(valid_dataset, batch_size=opt.test_batch_size, shuffle=True, num_workers=0,
        #                           pin_memory=False)
        valid_loader = DataLoader(valid_dataset, batch_size=opt.test_batch_size, shuffle=False, num_workers=0,
                                  pin_memory=False)
    test_dataset = datasets.Datasets(opt, split=2)
    print('>>> Testing dataset length: {:d}'.format(test_dataset.__len__()))
    test_loader = DataLoader(test_dataset, batch_size=opt.test_batch_size, shuffle=False, num_workers=0,
                             pin_memory=False)

    dim_used = dataset.dim_used

    # evaluation
    if opt.is_eval:
        ret_test = run_model(net_pred, is_train=3, data_loader=test_loader, opt=opt, epo=0, dim_used=dim_used)
        ret_log = np.array([])
        head = np.array([])
        for k in ret_test.keys():
            ret_log = np.append(ret_log, [ret_test[k]])
            head = np.append(head, [k])
        log.save_csv_log(opt, head, ret_log, is_create=True, file_name='test_walking')
        # print('testing error: {:.3f}'.format(ret_test['m_p3d_h36']))
    # training
    if not opt.is_eval:
        err_best = 1000
        for epo in range(start_epoch, opt.epoch + 1):
            is_best = False
            # if epo % opt.lr_decay == 0:
            lr_now = util.lr_decay_mine(optimizer, lr_now, 0.1 ** (1 / opt.epoch))
            print('>>> training epoch: {:d}'.format(epo))
            ret_train = run_model(net_pred, optimizer, is_train=0, data_loader=data_loader, epo=epo, opt=opt, dim_used=dim_used)
            print('train error: {:.3f}'.format(ret_train['m_p3d_h36']))
            ret_valid = run_model(net_pred, is_train=1, data_loader=valid_loader, opt=opt, epo=epo, dim_used=dim_used)
            print('validation error: {:.3f}'.format(ret_valid['m_p3d_h36']))
            ret_test = run_model(net_pred, is_train=2, data_loader=test_loader, opt=opt, epo=epo, dim_used=dim_used)
            print('testing error: {:.3f}'.format(ret_test['#1']))
            ret_log = np.array([epo, lr_now])
            head = np.array(['epoch', 'lr'])
            for k in ret_train.keys():
                ret_log = np.append(ret_log, [ret_train[k]])
                head = np.append(head, [k])
            for k in ret_valid.keys():
                ret_log = np.append(ret_log, [ret_valid[k]])
                head = np.append(head, ['valid_' + k])
            for k in ret_test.keys():
                ret_log = np.append(ret_log, [ret_test[k]])
                head = np.append(head, ['test_' + k])
            log.save_csv_log(opt, head, ret_log, is_create=(epo == 1))
            if ret_valid['m_p3d_h36'] < err_best:
                err_best = ret_valid['m_p3d_h36']
                is_best = True
            log.save_ckpt({'epoch': epo,
                           'lr': lr_now,
                           'err': ret_valid['m_p3d_h36'],
                           'state_dict': net_pred.state_dict(),
                           'optimizer': optimizer.state_dict()},
                          is_best=is_best, opt=opt)

def smooth(src, sample_len, kernel_size):
    """
    data:[bs, 60, 96]
    """
    src_data = src[:, -sample_len:, :].clone()
    smooth_data = src_data.clone()
    for i in range(kernel_size, sample_len):
        smooth_data[:, i] = torch.mean(src_data[:, kernel_size:i+1], dim=1)
    return smooth_data


def run_model(net_pred, optimizer=None, is_train=0, data_loader=None, epo=1, opt=None, dim_used=None):
    if is_train == 0:
        net_pred.train()
        repeat = 2
    else:
        net_pred.eval()
        repeat = 1
    device = util.get_device()
    l_p3d = 0
    # l_beta = 0
    # j17to14 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10]
    if is_train <= 1:
        m_p3d_h36 = 0
    else:
        titles = np.array(range(opt.output_n)) + 1
        m_p3d_h36 = np.zeros([opt.output_n])
    n = 0
    itera = 1
    in_n = opt.input_n
    out_n = opt.output_n

    seq_in = opt.kernel_size
    st = time.time()
    for i, (p3d_h36) in enumerate(data_loader):
        tmp = p3d_h36.clone()
        for rep in range(repeat):
            p3d_h36 = tmp.clone()
            batch_size, seq_n, all_dim = p3d_h36.shape
            if batch_size == 1:
                continue
            n += batch_size
            bt = time.time()
            
            p3d_h36 = p3d_h36.float().to(device)
            
            smooths = []
            for j in range(opt.num_stage):
                # smooths.append(p3d_h36.clone()[:, :, dim_used])
                if len(smooths) == 0:
                    smooths.append(p3d_h36.clone()[:, :, dim_used])
                else:
                    smooths.append(smooth(smooths[-1], sample_len=opt.kernel_size + opt.output_n,kernel_size=opt.kernel_size))
            p3d_sups = []
            # for j in order:
            #    p3d_sups.append(smooths[j].clone()[:, -out_n - seq_in:].reshape([-1, seq_in + out_n, len(dim_used) // 3, 3]))
            for j in range(opt.num_stage):
                # loss [smn, smn-1, smn-2, ..., sm1, raw]
                p3d_sups.append(smooths[-(j+1)].clone()[:, -out_n - seq_in:].reshape([-1, seq_in + out_n, len(dim_used) // 3, 3]))
                # loss [raw, sm1, sm2, ...]
                # p3d_sups.append(smooths[j].clone()[:, -out_n - seq_in:].reshape([-1, seq_in + out_n, len(dim_used) // 3, 3]))

            p3d_src = p3d_h36.clone()[:, :, dim_used]

            p3d_out_alls = net_pred(p3d_src, output_n=out_n, input_n=in_n, itera=itera)

            p3d_out = p3d_h36.clone()[:, in_n:in_n + out_n]
            p3d_out[:, :, dim_used] = p3d_out_alls[-1][:, seq_in:, 0]
            p3d_out = p3d_out.reshape([-1, out_n, all_dim//3, 3])

            p3d_h36 = p3d_h36.reshape([-1, in_n + out_n, all_dim//3, 3])

            for j in range(opt.num_stage):
                p3d_out_alls[j] = p3d_out_alls[j][:, :, 0].reshape([batch_size, seq_in + out_n, len(dim_used)//3, 3])

            # p3d_out_all = p3d_out_all[:, :, 0].reshape([batch_size, seq_in + out_n, len(joint_used), 3])

            # 2d joint loss:
            grad_norm = 0
            if is_train == 0:
                # loss_p3d = torch.mean(torch.sum(torch.abs(p3d_out_all - p3d_sup), dim=4))
                loss_p3d = []
                for j in range(opt.num_stage):
                    loss_p3d.append(torch.mean(torch.norm(p3d_out_alls[j] - p3d_sups[j], dim=3)))

                loss_all = sum(loss_p3d) / float(opt.num_stage)
                optimizer.zero_grad()
                loss_all.backward()
                nn.utils.clip_grad_norm_(list(net_pred.parameters()), max_norm=opt.max_norm)
                optimizer.step()
                # update log values
                l_p3d += loss_p3d[-1].cpu().data.numpy() * batch_size

            if is_train <= 1:  # if is validation or train simply output the overall mean error
                mpjpe_p3d_h36 = torch.mean(torch.norm(p3d_h36[:, in_n:in_n + out_n] - p3d_out, dim=3))
                m_p3d_h36 += mpjpe_p3d_h36.cpu().data.numpy() * batch_size
            else:
                mpjpe_p3d_h36 = torch.sum(torch.mean(torch.norm(p3d_h36[:, in_n:] - p3d_out, dim=3), dim=2), dim=0)
                m_p3d_h36 += mpjpe_p3d_h36.cpu().data.numpy()

            if i % 1000 == 0 and rep == 0:
                print('{}/{}|bt {:.3f}s|tt{:.0f}s'.format(i + 1, len(data_loader), time.time() - bt, time.time() - st))

    ret = {}
    if is_train == 0:
        ret["l_p3d"] = l_p3d / n

    if is_train <= 1:
        ret["m_p3d_h36"] = m_p3d_h36 / n

    else:
        m_p3d_h36 = m_p3d_h36 / n
        for j in range(out_n):
            ret["#{:d}".format(titles[j])] = m_p3d_h36[j]

    return ret


if __name__ == '__main__':
    option = Options().parse()
    main(option)
