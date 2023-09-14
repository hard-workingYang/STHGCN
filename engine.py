import torch.optim as optim
from model import *
import util
import torch

class Trainer():
    def __init__(self, model: STHGCN, scaler, lrate, wdecay, clip=3, lr_decay_rate=.97, device="cpu"):
        self.model = model

        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.scaler = scaler
        self.clip = clip

        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=[5,30,80,120,250,370,500], gamma = lr_decay_rate)

        self.loss =  torch.nn.L1Loss()

    @classmethod
    def from_args(cls, model, scaler, args, device):
        return cls(model, scaler, args.learning_rate, args.weight_decay, clip=args.clip,
                   lr_decay_rate=args.lr_decay_rate, device=device)

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()

        inflow_output,outflow_output = self.model(input) 
        inflow_predict = self.scaler.inverse_transform(inflow_output)
        outflow_predict = self.scaler.inverse_transform(outflow_output)

        inflow_mae, inflow_mape, inflow_rmse = util.my_metric(inflow_predict, real_val[...,0,None])
        outflow_mae, outflow_mape, outflow_rmse = util.my_metric(outflow_predict, real_val[...,1,None])
        
        loss = inflow_mae + outflow_mae
        loss.backward()
        
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        return loss.cpu().detach(), inflow_mae.item(),inflow_mape.item(),inflow_rmse.item(),outflow_mae.item(),outflow_mape.item(),outflow_rmse.item()

    def eval(self, input, real_val):
        self.model.eval()

        inflow_output,outflow_output = self.model(input)  
        inflow_predict = self.scaler.inverse_transform(inflow_output)
        outflow_predict = self.scaler.inverse_transform(outflow_output)
        
        inflow_mae, inflow_mape, inflow_rmse = util.my_metric(inflow_predict, real_val[:,:,:,0,None])
        outflow_mae, outflow_mape, outflow_rmse = util.my_metric(outflow_predict, real_val[:,:,:,1,None])

        loss = inflow_mae + outflow_mae
        return loss.cpu().detach(), inflow_mae.item(),inflow_mape.item(),inflow_rmse.item(),outflow_mae.item(),outflow_mape.item(),outflow_rmse.item()
