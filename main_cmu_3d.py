from utils import cmu3d as datasets
from model import AttModel
from utils.opt import Options
from utils import util
from utils import log
from utils.device import *

import os

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np
import time
import h5py
import torch.optim as optim


def main(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.cuda
    device = get_device()
    lr_now = opt.lr_now
    start_epoch = 1
    # opt.is_eval = True
    print('>>> create models')
    in_features = opt.in_features  # 66
    d_model = opt.d_model
    kernel_size = opt.kernel_size
    net_pred = AttModel.AttModel(in_features=in_features, kernel_size=kernel_size, d_model=d_model,
                                 num_stage=opt.num_stage, dct_n=opt.dct_n, lamda=opt.lamda, alpha=opt.alpha,
                                 n_layer=opt.n_layer, p_dropout=opt.drop_out)
    net_pred.to(device)
    optimizer = optim.Adam(
        filter(lambda x: x.requires_grad, net_pred.parameters()), lr=opt.lr_now)
    print(">>> total params: {:.2f}M".format(sum(p.numel()
          for p in net_pred.parameters()) / 1000000.0))

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
        print(">>> ckpt len loaded (epoch: {} | err: {})".format(
            ckpt['epoch'], ckpt['err']))

    print('>>> loading datasets')

    if not opt.is_eval:
        # dataset = datasets.DatasetsSmooth(opt, split=0)
        # actions = ["walking", "eating", "smoking", "discussion", "directions",
        #            "greeting", "phoning", "posing", "purchases", "sitting",
        #            "sittingdown", "takingphoto", "waiting", "walkingdog",
        #            "walkingtogether"]
        dataset = datasets.CMU_Motion3D(opt, split=0)
        print('>>> Training dataset length: {:d}'.format(dataset.__len__()))
        data_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0, pin_memory=True)
        valid_dataset = datasets.CMU_Motion3D(opt, split=2)
        print('>>> Validation dataset length: {:d}'.format(valid_dataset.__len__()))
        valid_loader = DataLoader(valid_dataset, batch_size=opt.test_batch_size, shuffle=True, num_workers=0,
                                  pin_memory=True)
    test_loader = {}
    acts = ["basketball", "basketball_signal", "directing_traffic", "jumping", "running", "soccer", "walking",
            "washwindow"]
    for act in acts:
        test_dataset = datasets.CMU_Motion3D(opt=opt, split=2, actions=act)
        dim_used = dataset.dim_used
        test_loader[act] = DataLoader(test_dataset, batch_size=opt.test_batch_size, shuffle=False, num_workers=0,
                                      pin_memory=True)

    # test_dataset = CMU_Motion3D.CMU_Motion3D(opt, split=2)
    # print('>>> Testing dataset length: {:d}'.format(test_dataset.__len__()))
    # test_loader = DataLoader(test_dataset, batch_size=opt.test_batch_size, shuffle=False, num_workers=0,
    #                          pin_memory=True)

    dim_used = dataset.dim_used

    # evaluation
    if opt.is_eval:
        ret_test = run_model(net_pred, is_train=3,
                             data_loader=test_loader, opt=opt, dim_used=dim_used)
        ret_log = np.array([])
        head = np.array([])
        for k in ret_test.keys():
            ret_log = np.append(ret_log, [ret_test[k]])
            head = np.append(head, [k])
        log.save_csv_log(opt, head, ret_log, is_create=True,
                         file_name='test_walking')
        # print('testing error: {:.3f}'.format(ret_test['m_p3d_h36']))
    # training
    if not opt.is_eval:
        err_best = 1000
        for epo in range(start_epoch, opt.epoch + 1):
            is_best = False
            # if epo % opt.lr_decay == 0:
            lr_now = util.lr_decay_mine(
                optimizer, lr_now, 0.1 ** (1 / opt.epoch))
            print('>>> training epoch: {:d}'.format(epo))
            ret_train = run_model(
                net_pred, optimizer, is_train=0, data_loader=data_loader, epo=epo, opt=opt, dim_used=dim_used)
            print('train error: {:.3f}'.format(ret_train['m_p3d_h36']))
            ret_valid = run_model(net_pred, is_train=1,
                                  data_loader=valid_loader, opt=opt, epo=epo, dim_used=dim_used)
            print('validation error: {:.3f}'.format(ret_valid['m_p3d_h36']))
            
            errs = np.zeros([len(acts) + 1, opt.output_n])
            for i, act in enumerate(acts):
                ret_test = run_model(net_pred, is_train=3,
                                 data_loader=test_loader[act], opt=opt, epo=epo, dim_used=dim_used)
                ret_log = np.array([])
                for k in ret_test.keys():
                    ret_log = np.append(ret_log, [ret_test[k]])
                errs[i] = ret_log
            errs[-1] = np.mean(errs[:-1], axis=0)
            test_error = errs[-1][0]
            print('testing error: {:.3f}'.format(test_error))
            ret_log = np.array([epo, lr_now])
            head = np.array(['epoch', 'lr'])
            for k in ret_train.keys():
                ret_log = np.append(ret_log, [ret_train[k]])
                head = np.append(head, [k])
            for k in ret_valid.keys():
                ret_log = np.append(ret_log, [ret_valid[k]])
                head = np.append(head, ['valid_' + k])
            for k in range(len(ret_test)):
                ret_log = np.append(ret_log, [errs[-1][k]])
                head = np.append(head, ['test_' + '#' + str(k)])
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


def smooth(src, sample_len, kernel_size, ratio=0):
    """
    data:[bs, 60, 96]
    """
    src_data = src[:, -sample_len:, :].clone()
    smooth_data = src_data.clone()
    for i in range(kernel_size, sample_len):
        smooth_data[:, i] = torch.mean(src_data[:, kernel_size-1:i+1], dim=1)
    return smooth_data * (1 - ratio) + src_data * ratio



def run_model(net_pred, optimizer=None, is_train=0, data_loader=None, epo=1, opt=None, dim_used=None):
    if is_train == 0:
        net_pred.train()
    else:
        net_pred.eval()

    l_p3d = 0
    if is_train <= 1:
        m_p3d_h36 = 0
    else:
        titles = np.array(range(opt.output_n)) + 1
        m_p3d_h36 = np.zeros([opt.output_n])
    device = get_device()
    n = 0
    in_n = opt.input_n
    out_n = opt.output_n
    seq_in = opt.kernel_size
    # joints at same loc
    joint_to_ignore = np.array([16, 20, 29, 24, 27, 33, 36])
    index_to_ignore = np.concatenate((joint_to_ignore * 3, joint_to_ignore * 3 + 1, joint_to_ignore * 3 + 2))
    joint_equal = np.array([15, 15, 15, 23, 23, 32, 32])
    index_to_equal = np.concatenate((joint_equal * 3, joint_equal * 3 + 1, joint_equal * 3 + 2))

    itera = 1
    # idx = np.expand_dims(np.arange(seq_in + out_n), axis=1) + (
    #     out_n - seq_in + np.expand_dims(np.arange(itera), axis=0))
    st = time.time()
    repeat = 1
    if is_train == 0:
        repeat = 3 # repeat one batch 3 times each epoch
    for i, (p3d_h36) in enumerate(data_loader):
        tmp = p3d_h36.clone()
        for _ in range(repeat):
            p3d_h36 = tmp.clone()
            p3d_h36.to(device)
            # p3d_h36 = torch.tensor(p3d_h36)
            batch_size, seq_n, all_dim = p3d_h36.shape
            # when only one sample in this batch
            if batch_size == 1 and is_train == 0:
                continue
            n += batch_size
            bt = time.time()
            p3d_h36 = p3d_h36.float().to(device)
            smooths = []
            # [[raw], [smooth1], [smooth2], [smooth3]...]
            for j in range(opt.num_stage):
                # all gt to guide
                smooths.append(p3d_h36.clone()[:, :, dim_used])

            p3d_sups = []
            for j in range(opt.num_stage):
                # [smoothn, smoothn-1, ..., smooth1, raw]
                p3d_sups.append(smooths[-(j+1)].clone()[:, -out_n - seq_in:].reshape([-1, seq_in + out_n, len(dim_used) // 3, 3]))

            p3d_src = p3d_h36.clone()[:, :, dim_used]
            # [[pred1], [pred2],..., [pred_final]]
            p3d_out_alls = net_pred(p3d_src, input_n=in_n,
                                    output_n=out_n, itera=itera)

            p3d_out = p3d_h36.clone()[:, in_n:in_n + out_n]
            p3d_out[:, :, dim_used] = p3d_out_alls[-1][:, seq_in:, 0]
            p3d_out[:, :, index_to_ignore] = p3d_out[:, :, index_to_equal]
            p3d_out = p3d_out.reshape([-1, out_n, all_dim // 3, 3])

            p3d_h36 = p3d_h36.reshape([-1, in_n + out_n, all_dim // 3, 3])

            for j in range(opt.num_stage):
                p3d_out_alls[j] = p3d_out_alls[j].reshape(
                    [batch_size, seq_in + out_n, itera, len(dim_used) // 3, 3])

            # 2d joint loss:
            grad_norm = 0
            if is_train == 0:
                loss_p3d = []
                for j in range(opt.num_stage):
                    loss_p3d.append(torch.mean(torch.norm(
                        p3d_out_alls[j][:, :, 0] - p3d_sups[j], dim=3)))
                # loss_all = (loss_p3d_3 + loss_p3d_2 + loss_p3d_1)/3.0
                loss_all = sum(loss_p3d) / float(opt.num_stage)
                # db.set_trace()
                optimizer.zero_grad()
                loss_all.backward()
                nn.utils.clip_grad_norm_(
                    list(net_pred.parameters()), max_norm=opt.max_norm)
                optimizer.step()
                # update log values
                l_p3d += loss_p3d[-1].cpu().data.numpy() * batch_size

            if is_train <= 1:  # if is validation or train simply output the overall mean error
                mpjpe_p3d_h36 = torch.mean(torch.norm(
                    p3d_h36[:, in_n:in_n + out_n] - p3d_out, dim=3))
                m_p3d_h36 += mpjpe_p3d_h36.cpu().data.numpy() * batch_size
            else:
                mpjpe_p3d_h36 = torch.sum(torch.mean(torch.norm(
                    p3d_h36[:, in_n:] - p3d_out, dim=3), dim=2), dim=0)
                m_p3d_h36 += mpjpe_p3d_h36.cpu().data.numpy()
            if i % 500 == 0 and is_train <= 1 and _ == 0:
                print('{}/{}|bt {:.3f}s|tt{:.0f}s|gn{}'.format(i + 1, len(data_loader), time.time() - bt,
                                                            time.time() - st, grad_norm))
    ret = {}
    if is_train == 0:
        ret["l_p3d"] = l_p3d / n / repeat

    if is_train <= 1:
        ret["m_p3d_h36"] = m_p3d_h36 / n / repeat
    else:
        m_p3d_h36 = m_p3d_h36 / n
        for j in range(out_n):
            ret["#{:d}".format(titles[j])] = m_p3d_h36[j]
    return ret


if __name__ == '__main__':
    option = Options().parse()
    main(option)
