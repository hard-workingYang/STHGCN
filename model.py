import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm2d, Conv1d, Conv2d, ModuleList, Parameter

def nconv(x, A):
    """Multiply x by adjacency matrix along source node axis"""
    return torch.einsum('ncvl,vw->ncwl', (x, A)).contiguous()

class OutRegionConv(nn.Module):
    def __init__(self, c_in, c_out, dropout):
        super().__init__()
        self.final_conv = Conv2d(c_out, c_in, (1, 1), padding=(0, 0), stride=(1, 1), bias=True)
        self.dropout = dropout

    def forward(self, x, adp):
        out = nconv(x, adp)
        h = self.final_conv(out)
        h = F.dropout(h, self.dropout, training=self.training)
        return h

class GraphConvNet(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super().__init__()

        origin_cin = c_in
        c_in = (order * support_len + 1) * c_in
        self.final_conv = Conv2d(c_in, c_out, (1, 1), padding=(0, 0), stride=(1, 1), bias=True)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support: list):
        out = [x]
        for a in support:
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


class Regions_Graph_Generator(nn.Module):
    def __init__(self, device, node_num = 200, mid_dim = 10):
        super().__init__()
        self.device = device
        self.nodevec1 = Parameter(torch.randn(node_num, mid_dim).to(device), requires_grad=True)
        self.nodevec2 = Parameter(torch.randn(mid_dim, node_num).to(device), requires_grad=True)

    def forward(self):
        return F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
        # return F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=0)


class Outside_Graph_Generator(nn.Module):
    def __init__(self, device, node_num = 200, mid_dim = 200):
        super().__init__()
        self.device = device
        self.nodevec1 = Parameter(torch.randn(1, mid_dim).to(device), requires_grad=True)
        self.nodevec2 = Parameter(torch.randn(mid_dim, node_num).to(device), requires_grad=True)

    def forward(self):
        return F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
        # return F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=0)

#Outside Flow Feedback Module (OFM)
class OFM(nn.Module):
    def __init__(self, dilation_channels = 56, residual_channels = 56 ,dropout = 0.30):
        super().__init__()

        self.outregion_inflow_conv = OutRegionConv(dilation_channels, residual_channels, dropout)
        self.outregion_outflow_conv = OutRegionConv(dilation_channels, residual_channels, dropout)

        self.spilt_outregion_inflow = Conv2d(residual_channels, residual_channels, (1, 1), bias = True)
        self.spilt_outregion_outflow = Conv2d(residual_channels, residual_channels, (1, 1), bias = True)
        
    def forward(self, inflow, outflow, outregion_inflowadp, outregion_outflowadp):

        inflow_sum = torch.sum(inflow, axis=2, keepdims=True)
        outflow_sum = torch.sum(outflow, axis=2, keepdims=True)

        # new_out_region_flow = inflow_sum - outflow_sum
        out_region_flow = outflow_sum - inflow_sum

        outregion_inflow = torch.relu(self.spilt_outregion_inflow(out_region_flow))
        outregion_outflow = torch.relu(self.spilt_outregion_outflow(out_region_flow))

        outside2regions_inflow = self.outregion_inflow_conv(outregion_inflow, outregion_inflowadp)
        outside2regions_outflow = self.outregion_outflow_conv(outregion_outflow, outregion_outflowadp)

        return outside2regions_inflow, outside2regions_outflow

# InFlow/Outflow Feature Extraction Module (FEM)
class FEM(nn.Module):
    def __init__(self, dilation_channels = 56, residual_channels = 56, dropout = 0.30, supports_len = 3):
        super().__init__()

        self.inin_graph_conv = GraphConvNet(dilation_channels, residual_channels, dropout, support_len=supports_len)
        self.outout_graph_conv = GraphConvNet(dilation_channels, residual_channels, dropout, support_len=supports_len)
        
        self.inout_graph_conv = GraphConvNet(dilation_channels, residual_channels, dropout, support_len=supports_len)
        self.outin_graph_conv = GraphConvNet(dilation_channels, residual_channels, dropout, support_len=supports_len)

    def forward(self, inflow, outflow, adps, ii_adp, io_adp, oo_adp):
        inin_adp = adps + [ii_adp]
        outout_adp = adps + [oo_adp]
        inin_data = self.inin_graph_conv(inflow, inin_adp)
        outout_data = self.outout_graph_conv(outflow, outout_adp)
        
        inin_data = inin_data + inflow
        outout_data = outout_data + outflow

        io_adp_T = io_adp.permute(1,0)
        inout_adp = adps + [io_adp]
        outint_adp = adps + [io_adp_T]

        inout_data = self.inout_graph_conv(inflow, inout_adp)
        outin_data = self.outin_graph_conv(outflow, outint_adp)

        inout_data = inout_data + inflow
        outin_data = outin_data + outflow

        return inin_data, inout_data, outout_data, outin_data
        

class FusionIntegrator(nn.Module):
    def __init__(self, device, mid_channels = 56, node_num = 200):
        super().__init__()
        self.w1_in = Parameter(torch.randn(1,mid_channels, node_num,1).to(device), requires_grad=True)
        self.w2_in = Parameter(torch.randn(1,mid_channels, node_num,1).to(device), requires_grad=True)
        self.w3_in = Parameter(torch.randn(1,mid_channels, node_num,1).to(device), requires_grad=True)
        self.w4_in = Parameter(torch.randn(1,mid_channels, node_num,1).to(device), requires_grad=True)
        
        self.w1_out = Parameter(torch.randn(1,mid_channels, node_num,1).to(device), requires_grad=True)
        self.w2_out = Parameter(torch.randn(1,mid_channels, node_num,1).to(device), requires_grad=True)
        self.w3_out = Parameter(torch.randn(1,mid_channels, node_num,1).to(device), requires_grad=True)
        self.w4_out = Parameter(torch.randn(1,mid_channels, 200,1).to(device), requires_grad=True)
        
    def forward(self, inin_data, inout_data, outout_data, outin_data, outregion_inin, outregion_inout, outregion_outout, outregion_outin):

        w1_in = self.w1_in.repeat(inin_data.shape[0], 1, 1, inin_data.shape[3])
        w2_in = self.w2_in.repeat(inin_data.shape[0], 1, 1, inin_data.shape[3])
        w3_in = self.w3_in.repeat(inin_data.shape[0], 1, 1, inin_data.shape[3])
        w4_in = self.w4_in.repeat(inin_data.shape[0], 1, 1, inin_data.shape[3])

        w1_out = self.w1_out.repeat(inin_data.shape[0], 1, 1, inin_data.shape[3])
        w2_out = self.w2_out.repeat(inin_data.shape[0], 1, 1, inin_data.shape[3])
        w3_out = self.w3_out.repeat(inin_data.shape[0], 1, 1, inin_data.shape[3])
        w4_out = self.w4_out.repeat(inin_data.shape[0], 1, 1, inin_data.shape[3])

        new_in_data = torch.mul(w1_in,inin_data) + torch.mul(w2_in,inout_data) + torch.mul(w3_in,outregion_inin) + torch.mul(w4_in,outregion_outin)
        new_out_data = torch.mul(w1_out,outout_data) + torch.mul(w2_out,outin_data) + torch.mul(w3_out,outregion_outout) + torch.mul(w4_out,outregion_inout)

        return new_in_data, new_out_data

# Traffic Flow Separation and Interaction Graph Convolution Module (TSIM)
class TSIM(nn.Module):
    def __init__(self, device, mid_channels = 56, kernel_size = 2, dropout = 0.30, node_num = 200):
        super().__init__()

        self.OFM_1 = OFM(dilation_channels = mid_channels, residual_channels = mid_channels, dropout = dropout)
        self.FEM_1 = FEM(dilation_channels = mid_channels, residual_channels = mid_channels, dropout = dropout, supports_len = 3)
        self.FEM_2 = FEM(dilation_channels = mid_channels, residual_channels = mid_channels, dropout = dropout, supports_len = 3)

        self.Integrator = FusionIntegrator(device = device,mid_channels = mid_channels, node_num = node_num)

        D = 1
        self.filter_convs_in = Conv2d(mid_channels, mid_channels, (1, kernel_size), stride=(1,2), dilation=D)
        self.gate_convs_in = Conv2d(mid_channels, mid_channels, (1, kernel_size), stride=(1,2), dilation=D)
        self.filter_convs_out = Conv2d(mid_channels, mid_channels, (1, kernel_size), stride=(1,2), dilation=D)
        self.gate_convs_out = Conv2d(mid_channels, mid_channels, (1, kernel_size), stride=(1,2), dilation=D)

    def forward(self, inflow, outflow, fixed_supports, inin_adp, outout_adp, intout_adp, outregion_inflowadp, outregion_outflowadp):

        outside2regions_inflow, outside2regions_outflow = self.OFM_1(inflow, outflow, outregion_inflowadp, outregion_outflowadp)

        inner_in2in, inner_in2out, inner_out2out, inner_out2in = self.FEM_1(inflow, outflow, fixed_supports, inin_adp, outout_adp, intout_adp)

        outside_in2in, outside_in2out, outside_out2out,  outside_out2in = self.FEM_2(outside2regions_inflow, outside2regions_outflow, fixed_supports, inin_adp, outout_adp, intout_adp)

        inflow_fusion, outflow_fusion = self.Integrator(inner_in2in, inner_in2out, inner_out2out, inner_out2in, outside_in2in, outside_in2out, outside_out2out,  outside_out2in)

        filter_in = torch.tanh(self.filter_convs_in(inflow_fusion))
        gate_in = torch.sigmoid(self.gate_convs_in(inflow_fusion))
        new_in_data = filter_in * gate_in

        filter_out = torch.tanh(self.filter_convs_out(outflow_fusion))
        gate_out = torch.sigmoid(self.gate_convs_out(outflow_fusion))
        new_out_data = filter_out * gate_out

        return new_in_data, new_out_data

class TimePeriodFusionBlock(nn.Module):
    def __init__(self, slices_in_day = 48, days_in_week = 7,  embed_len_in_day = 8, embed_len_in_week = 8):
        super().__init__()

        self.day_time_embed_conv_in = nn.Conv2d(in_channels=slices_in_day,  # hard code to avoid errors
                                        out_channels=embed_len_in_day,
                                        kernel_size=(1, 1))
        self.day_time_embed_conv_out = nn.Conv2d(in_channels=slices_in_day,  # hard code to avoid errors
                                        out_channels=embed_len_in_day,
                                        kernel_size=(1, 1))

        self.week_time_embed_conv_in = nn.Conv2d(in_channels=days_in_week,  # hard code to avoid errors
                                        out_channels=embed_len_in_week,
                                        kernel_size=(1, 1))
        self.week_time_embed_conv_out = nn.Conv2d(in_channels=days_in_week,  # hard code to avoid errors
                                        out_channels=embed_len_in_week,
                                        kernel_size=(1, 1))

    def forward(self, inflow, outflow, embed_in_day, embed_in_week):
        
        embed_in_day = embed_in_day.unsqueeze(dim = -1)
        embed_in_day = embed_in_day.permute(0,2,3,1)
        embed_in_day = embed_in_day.repeat(1,1,inflow.shape[2],1)

        embed_in_week = embed_in_week.unsqueeze(dim = -1)
        embed_in_week = embed_in_week.permute(0,2,3,1)
        embed_in_week = embed_in_week.repeat(1,1,inflow.shape[2],1)

        day_embed_inflow = self.day_time_embed_conv_in(embed_in_day)
        day_embed_outflow = self.day_time_embed_conv_out(embed_in_day)

        week_embed_inflow = self.week_time_embed_conv_in(embed_in_week)
        week_embed_outflow = self.week_time_embed_conv_out(embed_in_week)

        inflow = torch.cat([inflow, day_embed_inflow, week_embed_inflow], dim=1)
        outflow = torch.cat([outflow, day_embed_outflow, week_embed_outflow], dim=1)

        return inflow, outflow

class STHGCN(nn.Module):
    def __init__(self, device, node_num = 200, start_conv_channel = 40, tsim_num = 3, skip_main_channel = 180, skip_minor_channel = 140,
                 end_channels = 512, out_dim = 1, supports = None, dropout = 0.3, slices_in_day = 48, days_in_week = 7, 
                 day_embed_channel = 8, week_embed_channel = 8):
        super().__init__()

        self.start_conv_in = nn.Conv2d(in_channels=1,  
                                    out_channels=start_conv_channel,
                                    kernel_size=(1, 1))
        self.start_conv_out = nn.Conv2d(in_channels=1, 
                                    out_channels=start_conv_channel,
                                    kernel_size=(1, 1))

        residual_channels = start_conv_channel + day_embed_channel + week_embed_channel
        dilation_channels = residual_channels

        self.fixed_supports = supports or []

        self.inin_regions_dynamic_graph_generator = Regions_Graph_Generator(device, node_num)
        self.inout_regions_dynamic_graph_generator = Regions_Graph_Generator(device, node_num)
        self.outout_regions_dynamic_graph_generator = Regions_Graph_Generator(device, node_num)

        self.inflow_outsdie_dynamic_graph_generator = Outside_Graph_Generator(device, node_num)
        self.outflow_outsdie_dynamic_graph_generator = Outside_Graph_Generator(device, node_num)

        self.timefusion = TimePeriodFusionBlock(slices_in_day = slices_in_day, days_in_week = days_in_week, embed_len_in_day = day_embed_channel, embed_len_in_week = week_embed_channel)

        self.tsim_num = tsim_num
        self.TSIMs = ModuleList([TSIM(device = device, node_num = node_num, dropout=dropout, mid_channels = residual_channels) for _ in range(tsim_num)])

        self.skip_convs_in2in = ModuleList([Conv2d(dilation_channels, skip_main_channel, (1, 1)) for _ in range(tsim_num)])
        self.skip_convs_out2out = ModuleList([Conv2d(dilation_channels, skip_main_channel, (1, 1)) for _ in range(tsim_num)])
        self.skip_convs_out2in = ModuleList([Conv2d(dilation_channels, skip_minor_channel, (1, 1)) for _ in range(tsim_num)])
        self.skip_convs_in2out = ModuleList([Conv2d(dilation_channels, skip_minor_channel, (1, 1)) for _ in range(tsim_num)])

        self.bn_in = ModuleList([BatchNorm2d(residual_channels) for _ in range(tsim_num)])
        self.bn_out = ModuleList([BatchNorm2d(residual_channels) for _ in range(tsim_num)])

        skip_channels = skip_main_channel + skip_minor_channel
        self.end_conv_in_1 = Conv2d(skip_channels, end_channels, (1, 1), bias=True)
        self.end_conv_in_2 = Conv2d(end_channels, out_dim, (1, 1), bias=True)

        self.end_conv_out_1 = Conv2d(skip_channels, end_channels, (1, 1), bias=True)
        self.end_conv_out_2 = Conv2d(end_channels, out_dim, (1, 1), bias=True)


    def forward(self, x):
        
        x, embed_in_week, embed_in_day = x[0], x[1], x[2]

        x = x.permute(0,3,2,1)
        inflow, outflow = x[:,1,None], x[:,0,None]

        inflow = self.start_conv_in(inflow)
        outflow = self.start_conv_out(outflow)
        inflow, outflow = self.timefusion(inflow, outflow, embed_in_day, embed_in_week)

        inin_adp = self.inin_regions_dynamic_graph_generator()
        outout_adp = self.inout_regions_dynamic_graph_generator()
        intout_adp = self.outout_regions_dynamic_graph_generator()

        outregion_inflowadp = self.inflow_outsdie_dynamic_graph_generator()
        outregion_outflowadp = self.outflow_outsdie_dynamic_graph_generator()

        for i in range(self.tsim_num):
            old_inflow = inflow
            old_outflow = outflow

            inflow, outflow = self.TSIMs[i](inflow, outflow, self.fixed_supports, inin_adp, outout_adp, intout_adp, outregion_inflowadp, outregion_outflowadp)

            s_in2in = self.skip_convs_in2in[i](inflow)
            s_out2in = self.skip_convs_out2in[i](outflow)
            s_in2out = self.skip_convs_in2out[i](inflow)
            s_out2out = self.skip_convs_out2out[i](outflow)

            s_in = torch.cat([s_in2in, s_out2in], axis= 1)
            s_out = torch.cat([s_in2out, s_out2out], axis= 1)

            try: 
                skip_in = skip_in[:, :, :,  -s_in.size(3):] 
                skip_out = skip_out[:, :, :,  -s_out.size(3):]
            except:
                skip_in = 0
                skip_out = 0
            skip_in = s_in + skip_in
            skip_out = s_out + skip_out

            inflow = inflow + old_inflow[:, :, :, -inflow.size(3):]  # TODO(SS): Mean/Max Pool?
            outflow = outflow + old_outflow[:, :, :, -outflow.size(3):]  # TODO(SS): Mean/Max Pool?

            inflow = self.bn_in[i](inflow)
            outflow = self.bn_out[i](outflow)

        inflow = F.relu(skip_in) 
        inflow = F.relu(self.end_conv_in_1(inflow))
        inflow = self.end_conv_in_2(inflow) 

        outflow = F.relu(skip_out) 
        outflow = F.relu(self.end_conv_out_1(outflow))
        outflow = self.end_conv_out_2(outflow)
        return inflow, outflow
    
