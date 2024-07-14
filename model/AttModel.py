from torch.nn import Module
from torch import nn
import torch
from model import GCN
import utils.util as util
import numpy as np
from utils.device import *


class AttModel(Module):

    def __init__(self, in_features=66, kernel_size=10, d_model=512, n_layer=6, num_stage=3, dct_n=10, lamda=0.5, alpha=0.1, p_dropout=0.5):
        super(AttModel, self).__init__()

        self.kernel_size = kernel_size
        self.d_model = d_model
        self.dct_n = dct_n
        self.num_stage = num_stage
        self.n_layer = n_layer
        assert kernel_size == 10

        self.convQ = nn.Sequential(nn.Conv1d(in_channels=in_features, out_channels=d_model, kernel_size=6,
                                             bias=False),
                                   nn.ReLU(),
                                   nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=5,
                                             bias=False),
                                   nn.ReLU())
        self.convK = nn.Sequential(nn.Conv1d(in_channels=in_features, out_channels=d_model, kernel_size=6,
                                             bias=False),
                                   nn.ReLU(),
                                   nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=5,
                                             bias=False),
                                   nn.ReLU())

        self.gcns = []
        for i in range(num_stage):
            self.gcns.append(GCN.GCN(input_feature=(dct_n) * 2, hidden_feature=d_model, p_dropout=p_dropout,
                            nlayers=n_layer, node_n=in_features, alpha=alpha, lamda=lamda))
        self.gcns = nn.ModuleList(self.gcns)
        

    def forward(self, src, output_n=10, input_n=50, itera=1):
        """
        :param src: [batch_size,seq_len,feat_dim]
        :param output_n:
        :param input_n:
        :param frame_n:
        :param dct_n:
        :param itera:
        :return:
        """
        device = get_device()
        dct_n = self.dct_n
        src = src[:, :input_n]  # [bs,in_n,dim]
        src_tmp = src.clone()
        bs = src.shape[0]
        src_key_tmp = src_tmp.transpose(
            1, 2)[:, :, :(input_n - output_n)].clone()
        src_query_tmp = src_tmp.transpose(
            1, 2)[:, :, -self.kernel_size:].clone()

        dct_m, idct_m = util.get_dct_matrix(self.kernel_size + output_n)
        dct_m = torch.from_numpy(dct_m).float().to(device)
        idct_m = torch.from_numpy(idct_m).float().to(device)

        vn = input_n - self.kernel_size - output_n + 1
        vl = self.kernel_size + output_n
        idx = np.expand_dims(np.arange(vl), axis=0) + \
            np.expand_dims(np.arange(vn), axis=1)
        src_value_tmp = src_tmp[:, idx].clone().reshape(
            [bs * vn, vl, -1])
        src_value_tmp = torch.matmul(dct_m[:dct_n].unsqueeze(dim=0), src_value_tmp).reshape(
            [bs, vn, dct_n, -1]).transpose(2, 3).reshape(
            [bs, vn, -1])  # [32,40,66*11]

        idx = list(range(-self.kernel_size, 0, 1)) + [-1] * output_n
        outputs = []
        for i in range(self.num_stage):
            outputs.append([])

        key_tmp = self.convK(src_key_tmp / 1000.0)
        for i in range(itera):
            query_tmp = self.convQ(src_query_tmp / 1000.0)
            score_tmp = torch.matmul(
                query_tmp.transpose(1, 2), key_tmp) + 1e-15
            att_tmp = score_tmp / (torch.sum(score_tmp, dim=2, keepdim=True))
            dct_att_tmp = torch.matmul(att_tmp, src_value_tmp)[:, 0].reshape(
                [bs, -1, dct_n])
            input_gcn = src_tmp[:, idx]
            dct_in_tmp = torch.matmul(dct_m[:dct_n].unsqueeze(
                dim=0), input_gcn).transpose(1, 2)

            for j in range(self.num_stage):
                dct_in_tmp = torch.cat([dct_in_tmp, dct_att_tmp], dim=-1)
                if j == 0:
                    dct_out, his = self.gcns[j](dct_in_tmp)
                else:
                    dct_out, his = self.gcns[j](dct_in_tmp, his)
                    

                dct_out = dct_out[:, :, :dct_n]
                out_gcn = torch.matmul(idct_m[:, :dct_n].unsqueeze(dim=0),
                                        dct_out.transpose(1, 2))
                outputs[j].append(out_gcn.unsqueeze(2))
                dct_in_tmp = dct_out.clone()


            if itera > 1:
                # update key-value query
                out_tmp = out_gcn.clone()[:, 0 - output_n:]
                src_tmp = torch.cat([src_tmp, out_tmp], dim=1)

                vn = 1 - 2 * self.kernel_size - output_n
                vl = self.kernel_size + output_n
                idx_dct = np.expand_dims(np.arange(vl), axis=0) + \
                    np.expand_dims(
                        np.arange(vn, -self.kernel_size - output_n + 1), axis=1)

                src_key_tmp = src_tmp[:, idx_dct[0, :-1]].transpose(1, 2)
                key_new = self.convK(src_key_tmp / 1000.0)
                key_tmp = torch.cat([key_tmp, key_new], dim=2)

                src_dct_tmp = src_tmp[:, idx_dct].clone().reshape(
                    [bs * self.kernel_size, vl, -1])
                src_dct_tmp = torch.matmul(dct_m[:dct_n].unsqueeze(dim=0), src_dct_tmp).reshape(
                    [bs, self.kernel_size, dct_n, -1]).transpose(2, 3).reshape(
                    [bs, self.kernel_size, -1])
                src_value_tmp = torch.cat([src_value_tmp, src_dct_tmp], dim=1)

                src_query_tmp = src_tmp[:, -self.kernel_size:].transpose(1, 2)

        for i in range(self.num_stage):
            outputs[i] = torch.cat(outputs[i], dim=2)
        return outputs
