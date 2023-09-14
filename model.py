import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm2d, Conv1d, Conv2d, ModuleList, Parameter
import sys
import numpy as np

def nconv(x, A):
    """Multiply x by adjacency matrix along source node axis"""
    return torch.einsum('ncvl,vw->ncwl', (x, A)).contiguous()

class GraphConvNet(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super().__init__()

        origin_cin = c_in
        self.order = order
        c_in = (order * support_len + 1) * c_in
        self.final_conv = Conv2d(c_in, c_out, (1, 1), padding=(0, 0), stride=(1, 1), bias=True)
        self.dropout = dropout

    def forward(self, x, support: list):

        out = [x]
        
        for a in [support[0]]:
            x1 = nconv(x, a)
            out.append(x1)

            for k in range(2, self.order + 1):
                x2 = nconv(x1, a)
                out.append(x2)
                x1 = x2

        for a in [support[1]]:
            x1 = nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = nconv(x1, a)
                out.append(x2)
                x1 = x2

        for a in [support[2]]:
            x1 = nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = nconv(x1, a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1)
        h = self.final_conv(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h

class OutRegionConv(nn.Module):
    def __init__(self, c_in, c_out, dropout):
        super().__init__()

        origin_cin = c_in
        support_len = 1
        order = 2 
        # c_in = (order * support_len + 1) * c_in
        c_out = (order * support_len) * c_in
        # c_in = (order * support_len-1) * c_in
        # self.final_conv = Conv2d(c_in, c_out, (1, 1), padding=(0, 0), stride=(1, 1), bias=True)
        self.final_conv = Conv2d(c_out, c_in, (1, 1), padding=(0, 0), stride=(1, 1), bias=True)
        self.dropout = dropout
        self.order = order

    def forward(self, x, shared_adp, outregion_adp):
        
        out = []

        x1 = nconv(x, outregion_adp)
        out.append(x1)
        for k in range(1):
            x2 = torch.einsum('bmnt,na->bmat', x1,shared_adp)
            out.append(x2)

        h = torch.cat(out, dim=1)
        h = self.final_conv(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h

class OutRegion_Graph_Generator(nn.Module):
    def __init__(self, device, node_nums):
        super().__init__()
        self.device = device

        mid_dim = node_nums
        num_nodes = node_nums
        self.nodevec1 = Parameter(torch.randn(1, mid_dim).to(device), requires_grad=True)
        self.nodevec2 = Parameter(torch.randn(mid_dim, num_nodes).to(device), requires_grad=True)

    def forward(self, dayEmbedIndex, weekEmbedIndex):
        # return F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
        return F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=0)

class My_Graph_Generator0(nn.Module):
    def __init__(self, device, node_nums):
        super().__init__()
        self.device = device

        mid = 10
        self.nodevec1 = Parameter(torch.randn(node_nums, mid).to(device), requires_grad=True)
        self.nodevec2 = Parameter(torch.randn(mid, node_nums).to(device), requires_grad=True)

    def forward(self, dayEmbedIndex, weekEmbedIndex):
        # return F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
        return F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=0)


class InOutBlock(nn.Module):
    def __init__(self, device, node_nums):
        super().__init__()
        self.device = device
        
        #现在的想法是in和in out和out
        #然后in和out再交叉

        dilation_channels = 56
        residual_channels = 56
        mid_channels = 56
        kernel_size = 2
        #首先定义图卷积1和图卷积2 用于inin outout
        #其中这两个部分共用同一组邻接矩阵 
        dropout = 0.3
        supports_len = 3
        D = 1
        self.inin_graph_convs = GraphConvNet(dilation_channels, residual_channels, dropout, support_len=supports_len)
        self.outout_graph_convs = GraphConvNet(dilation_channels, residual_channels, dropout, support_len=supports_len)
        
        #然后定义图卷积3和图卷积4 用于inout 和 outin
        #这两组也使用同一组邻接矩阵 只是转置区别
        self.inout_graph_convs = GraphConvNet(dilation_channels, residual_channels, dropout, support_len=supports_len)
        self.outin_graph_convs = GraphConvNet(dilation_channels, residual_channels, dropout, support_len=supports_len)

        self.filter_convs_in = Conv2d(mid_channels, mid_channels, (1, kernel_size), stride=(1,2), dilation=D)
        self.gate_convs_in = Conv2d(mid_channels, mid_channels, (1, kernel_size), stride=(1,2), dilation=D)
        self.filter_convs_out = Conv2d(mid_channels, mid_channels, (1, kernel_size), stride=(1,2), dilation=D)
        self.gate_convs_out = Conv2d(mid_channels, mid_channels, (1, kernel_size), stride=(1,2), dilation=D)
             
        self.merge_gate_convs_in1 = Conv2d(mid_channels, mid_channels, (1, 1), dilation=D, bias = False)
        self.merge_gate_convs_in2 = Conv2d(mid_channels, mid_channels, (1, 1), dilation=D, bias = True)
        self.merge_gate_convs_out1 = Conv2d(mid_channels, mid_channels, (1, 1), dilation=D, bias = False)
        self.merge_gate_convs_out2 = Conv2d(mid_channels, mid_channels, (1, 1), dilation=D, bias = True)
        
        self.spilt_outregion_inflow = Conv2d(mid_channels, mid_channels, (1, 1), dilation=D, bias = True)
        self.spilt_outregion_outflow = Conv2d(mid_channels, mid_channels, (1, 1), dilation=D, bias = True)

        
        self.outregion_inflow_conv1 = OutRegionConv(dilation_channels, residual_channels, dropout)
        self.outregion_outflow_conv1 = OutRegionConv(dilation_channels, residual_channels, dropout)

        self.outregion_inflow_conv2 = OutRegionConv(dilation_channels, residual_channels, dropout)
        self.outregion_outflow_conv2 = OutRegionConv(dilation_channels, residual_channels, dropout)
    
        self.w1_in = Parameter(torch.randn(1,mid_channels, node_nums,1).to(device), requires_grad=True)
        self.w2_in = Parameter(torch.randn(1,mid_channels, node_nums,1).to(device), requires_grad=True)
        self.w3_in = Parameter(torch.randn(1,mid_channels, node_nums,1).to(device), requires_grad=True)
        self.w4_in = Parameter(torch.randn(1,mid_channels, node_nums,1).to(device), requires_grad=True)
        
        self.w1_out = Parameter(torch.randn(1,mid_channels, node_nums,1).to(device), requires_grad=True)
        self.w2_out = Parameter(torch.randn(1,mid_channels, node_nums,1).to(device), requires_grad=True)
        self.w3_out = Parameter(torch.randn(1,mid_channels, node_nums,1).to(device), requires_grad=True)
        self.w4_out = Parameter(torch.randn(1,mid_channels, node_nums,1).to(device), requires_grad=True)
        


    def forward(self,in_data, out_data,out_region_flow, adps, adp_xx, adp_yy, adp_xy, adp_yx, outregion_inflowadp, outregion_outflowadp):
    # def forward(self,in_data, out_data, adps, xx_adp, xy_adp):

        adp_inin = adps + [adp_xx]
        adp_outout = adps + [adp_yy]

        inin_data = self.inin_graph_convs(in_data, adp_inin)
        outout_data = self.outout_graph_convs(out_data, adp_outout)
        
        inin_data = inin_data + in_data
        outout_data = outout_data + out_data

        inout_adp = adps + [adp_xy]
        outint_adp = adps + [adp_yx]

        inout_data = self.inout_graph_convs(in_data, inout_adp)
        outin_data = self.outin_graph_convs(out_data, outint_adp)

        inout_data = inout_data + in_data
        outin_data = outin_data + out_data
        
        outregion_inflow = torch.relu(self.spilt_outregion_inflow(out_region_flow))
        outregion_outflow = torch.relu(self.spilt_outregion_outflow(out_region_flow))

        outregion_in2inflow = self.outregion_inflow_conv1(outregion_inflow, adp_xx, outregion_inflowadp)
        outregion_in2outflow = self.outregion_inflow_conv2(outregion_inflow, adp_xy, outregion_inflowadp)
        outregion_out2outflow = self.outregion_outflow_conv1(outregion_outflow, adp_yy, outregion_outflowadp)
        outregion_out2inflow = self.outregion_outflow_conv2(outregion_outflow, adp_yx, outregion_outflowadp)


        w1_in = self.w1_in.repeat(inin_data.shape[0], 1, 1, inin_data.shape[3])
        w2_in = self.w2_in.repeat(inin_data.shape[0], 1, 1, inin_data.shape[3])
        w3_in = self.w3_in.repeat(inin_data.shape[0], 1, 1, inin_data.shape[3])
        w4_in = self.w4_in.repeat(inin_data.shape[0], 1, 1, inin_data.shape[3])

        w1_out = self.w1_out.repeat(inin_data.shape[0], 1, 1, inin_data.shape[3])
        w2_out = self.w2_out.repeat(inin_data.shape[0], 1, 1, inin_data.shape[3])
        w3_out = self.w3_out.repeat(inin_data.shape[0], 1, 1, inin_data.shape[3])
        w4_out = self.w4_out.repeat(inin_data.shape[0], 1, 1, inin_data.shape[3])

        new_in_data = torch.mul(w1_in,inin_data) + torch.mul(w2_in,outin_data) + torch.mul(w3_in,outregion_in2inflow) + torch.mul(w4_in,outregion_out2inflow)
        new_out_data = torch.mul(w1_out,outout_data) + torch.mul(w2_out,inout_data) + torch.mul(w3_out,outregion_out2outflow) + torch.mul(w4_out,outregion_in2outflow)

        filter_in = torch.tanh(self.filter_convs_in(new_in_data))
        gate_in = torch.sigmoid(self.gate_convs_in(new_in_data))
        new_in_data = filter_in * gate_in

        filter_out = torch.tanh(self.filter_convs_out(new_out_data))
        gate_out = torch.sigmoid(self.gate_convs_out(new_out_data))
        new_out_data = filter_out * gate_out

        sum_in_data = torch.sum(new_in_data, axis=2, keepdims=True)
        sum_out_data = torch.sum(new_out_data, axis=2, keepdims=True)

        new_out_region_flow = sum_out_data - sum_in_data

        return new_in_data,new_out_data, new_out_region_flow


class STHGCN(nn.Module):
    def __init__(self, device, dropout=0.3, supports=None,
                 out_dim=12, day_time_embed_channel=8, week_time_embed_channel=8, residual_channels=40,
                 skip_channels=256, end_channels=512, kernel_size=2,
                 node_nums = 200, week_len = 7, day_len = 24, inoutblock_num = 2):
        super().__init__()
        self.dropout = dropout
        
        day_time_embed_channel = 8
        week_time_embed_channel = 8
        residual_channels = 40
        start_conv_channel = residual_channels

        residual_channels = day_time_embed_channel + week_time_embed_channel + residual_channels
        dilation_channels = residual_channels

        self.day_time_embed_conv_in = nn.Conv2d(in_channels=day_len,  # hard code to avoid errors
                                        out_channels=8,
                                        kernel_size=(1, 1))
        self.day_time_embed_conv_out = nn.Conv2d(in_channels=day_len,  # hard code to avoid errors
                                        out_channels=8,
                                        kernel_size=(1, 1))
        self.day_time_embed_conv_outregion = nn.Conv2d(in_channels=day_len,  # hard code to avoid errors
                                        out_channels=8,
                                        kernel_size=(1, 1))

        self.week_time_embed_conv_in = nn.Conv2d(in_channels=week_len,  # hard code to avoid errors
                                        out_channels=8,
                                        kernel_size=(1, 1))
        self.week_time_embed_conv_out = nn.Conv2d(in_channels=week_len,  # hard code to avoid errors
                                        out_channels=8,
                                        kernel_size=(1, 1))
        self.week_time_embed_conv_outregion = nn.Conv2d(in_channels=week_len,  # hard code to avoid errors
                                        out_channels=8,
                                        kernel_size=(1, 1))

        self.start_conv_in = nn.Conv2d(in_channels=1,  # hard code to avoid errors
                                    out_channels=start_conv_channel,
                                    kernel_size=(1, 1))
        self.start_conv_out = nn.Conv2d(in_channels=1,  # hard code to avoid errors
                                    out_channels=start_conv_channel,
                                    kernel_size=(1, 1))
        self.out_region_conv = nn.Conv2d(in_channels=1,  # hard code to avoid errors
                                    out_channels=start_conv_channel,
                                    kernel_size=(1, 1))

        self.fixed_supports = supports or []

        self.MyInOutBlocks = ModuleList()
        self.inoutblock_num = inoutblock_num
        for i in range(inoutblock_num):
            self.MyInOutBlocks.append(InOutBlock(device, node_nums))

        skip_channels_tmp = (int)(skip_channels/2)

        self.skip_convs_in = ModuleList([Conv2d(dilation_channels, 180, (1, 1)) for _ in list(range(inoutblock_num))])
        self.skip_convs_out = ModuleList([Conv2d(dilation_channels, 140, (1, 1)) for _ in list(range(inoutblock_num))])

        self.skip_convs_in1 = ModuleList([Conv2d(dilation_channels, 180, (1, 1)) for _ in list(range(inoutblock_num))])
        self.skip_convs_in2 = ModuleList([Conv2d(dilation_channels, 140, (1, 1)) for _ in list(range(inoutblock_num))])
        self.skip_convs_out1 = ModuleList([Conv2d(dilation_channels, 180, (1, 1)) for _ in list(range(inoutblock_num))])
        self.skip_convs_out2 = ModuleList([Conv2d(dilation_channels, 140, (1, 1)) for _ in list(range(inoutblock_num))])


        self.end_conv_in_1 = Conv2d(skip_channels, end_channels, (1, 1), bias=True)
        self.end_conv_in_2 = Conv2d(end_channels, out_dim, (1, 1), bias=True)


        self.end_conv_out_1 = Conv2d(skip_channels, end_channels, (1, 1), bias=True)
        self.end_conv_out_2 = Conv2d(end_channels, out_dim, (1, 1), bias=True)

        self.my_graph_generator_in2in = My_Graph_Generator0(device, node_nums)
        self.my_graph_generator_out2out = My_Graph_Generator0(device, node_nums)
        self.my_graph_generator_in2out = My_Graph_Generator0(device, node_nums)
        self.my_graph_generator_out2in = My_Graph_Generator0(device, node_nums)

        self.outregion_graph_generator_out = OutRegion_Graph_Generator(device, node_nums)
        self.outregion_graph_generator_in = OutRegion_Graph_Generator(device, node_nums)

        self.device = device

        self.bn_in = ModuleList([BatchNorm2d(residual_channels) for _ in range(inoutblock_num)])
        self.bn_out = ModuleList([BatchNorm2d(residual_channels) for _ in range(inoutblock_num)])
        self.bn_outregion = ModuleList([BatchNorm2d(residual_channels) for _ in range(inoutblock_num)])
        

    @classmethod
    def from_args(cls, args, device, supports, **kwargs):
        if args.map_type == "grid":
            node_nums = args.row_num * args.col_num
        else:
            node_nums = args.node_nums
        defaults = dict(dropout=args.dropout, supports=supports,
                        out_dim=args.seq_length,
                        skip_channels=args.nhid * 8, end_channels=args.nhid * 16,
                        node_nums = node_nums, week_len=args.week_len, day_len=args.day_len, inoutblock_num=args.inoutblock_num)
        defaults.update(**kwargs)
        model = cls(device, **defaults)
        return model

    def forward(self, x):

        day_embed_x_origin = x[2]
        week_embed_x_origin = x[1]
        day_embed_x = x[2]  
        week_embed_x = x[1]

        x = x[0]

        x_in_origin, x_out_origin = x[:,:,:,0,None], x[::,:,:,1,None]
        x_in_sum = torch.sum(x_in_origin, axis=2, keepdims=True)
        x_out_sum = torch.sum(x_out_origin, axis=2, keepdims=True)

        inoutFlowBias_x =  x_out_sum.squeeze(-1) - x_in_sum.squeeze(-1)

        x = x.permute(0,3,2,1)

        x_in = x[:,0,None]
        x_out = x[:,1,None]
        x_in = self.start_conv_in(x_in)
        x_out = self.start_conv_out(x_out)

        inoutFlowBias_x = inoutFlowBias_x.unsqueeze(-1)
        inoutFlowBias_x = inoutFlowBias_x.permute(0,2,1,3)
        inoutFlowBias_x = self.out_region_conv(inoutFlowBias_x)
        inoutFlowBias_x = inoutFlowBias_x.permute(0,1,3,2)

        x_outregion = inoutFlowBias_x

        day_embed_x = day_embed_x.unsqueeze(dim = -1)
        day_embed_x = day_embed_x.permute(0,2,3,1)
        day_embed_x = day_embed_x.repeat(1,1,x.shape[2],1)

        day_embed_x_in = self.day_time_embed_conv_in(day_embed_x)
        day_embed_x_out = self.day_time_embed_conv_out(day_embed_x)
        day_embed_x_outregion = self.day_time_embed_conv_outregion(day_embed_x)

        week_embed_x = week_embed_x.unsqueeze(dim = -1)
        week_embed_x = week_embed_x.permute(0,2,3,1)
        week_embed_x = week_embed_x.repeat(1,1,x.shape[2],1)

        week_embed_x_in = self.week_time_embed_conv_in(week_embed_x)
        week_embed_x_out = self.week_time_embed_conv_out(week_embed_x)
        week_embed_x_outregion = self.week_time_embed_conv_outregion(week_embed_x)


        x_in = torch.cat([x_in, day_embed_x_in, week_embed_x_in], dim=1)
        x_out = torch.cat([x_out, day_embed_x_out, week_embed_x_out], dim=1)
        x_outregion = torch.cat([x_outregion, day_embed_x_outregion[:,:,0,None,:], week_embed_x_outregion[:,:,0,None,:]], dim=1)

        adp_xx = self.my_graph_generator_in2in(day_embed_x_origin, week_embed_x_origin)
        adp_yy = self.my_graph_generator_out2out(day_embed_x_origin, week_embed_x_origin)
        adp_xy = self.my_graph_generator_in2out(day_embed_x_origin, week_embed_x_origin)
        adp_yx = self.my_graph_generator_out2in(day_embed_x_origin, week_embed_x_origin)

        outregion_inflowadp = self.outregion_graph_generator_in(day_embed_x_origin, week_embed_x_origin)
        outregion_outflowadp = self.outregion_graph_generator_out(day_embed_x_origin, week_embed_x_origin)

        in_data = x_in
        out_data = x_out
        for i in range(self.inoutblock_num):
            old_in_data = in_data
            old_out_data = out_data
            old_x_outregion = x_outregion
            
            in_data, out_data, x_outregion = self.MyInOutBlocks[i](in_data ,out_data, x_outregion, self.fixed_supports, adp_xx, adp_yy, adp_xy, adp_yx, outregion_inflowadp, outregion_outflowadp)

            s_in1 = self.skip_convs_in1[i](in_data)  # what are we skipping??
            s_in2 = self.skip_convs_in2[i](out_data)  # what are we skipping??
            
            s_out1 = self.skip_convs_out2[i](in_data)  # what are we skipping??
            s_out2 = self.skip_convs_out1[i](out_data)  # what are we skipping??
            
            s_in_sum = torch.cat([s_in1,s_in2], axis= 1)
            s_out_sum = torch.cat([s_out1,s_out2], axis= 1)
            
            try:  # if i > 0 this works
                skip_in = skip_in[:, :, :,  -s_in_sum.size(3):]  # TODO(SS): Mean/Max Pool?
            except:
                skip_in = 0
            skip_in = s_in_sum + skip_in

            try:  # if i > 0 this works
                skip_out = skip_out[:, :, :,  -s_out_sum.size(3):]  # TODO(SS): Mean/Max Pool?
            except:
                skip_out = 0
            skip_out = s_out_sum + skip_out

            in_data = in_data + old_in_data[:, :, :, -in_data.size(3):]  # TODO(SS): Mean/Max Pool?
            out_data = out_data + old_out_data[:, :, :, -out_data.size(3):]  # TODO(SS): Mean/Max Pool?
            x_outregion = x_outregion + old_x_outregion[:, :, :, -out_data.size(3):]  # TODO(SS): Mean/Max Pool?
            
            in_data = self.bn_in[i](in_data)
            out_data = self.bn_out[i](out_data)
            if x_outregion.shape[0] != 1:
                x_outregion = self.bn_outregion[i](x_outregion)

        x_in = F.relu(skip_in)  # ignore last X?
        x_in = F.relu(self.end_conv_in_1(x_in))
        x_in = self.end_conv_in_2(x_in)  # downsample to (bs, seq_length, 207, nfeatures)

        x_out = F.relu(skip_out)  # ignore last X?
        x_out = F.relu(self.end_conv_out_1(x_out))
        x_out = self.end_conv_out_2(x_out)  # downsample to (bs, seq_length, 207, nfeatures)
        return x_in,x_out





