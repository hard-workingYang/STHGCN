import torch
import numpy as np
import pandas as pd
import time
import util
import pickle
import shutil
import os
import sys 
import torch.nn as nn
from engine import Trainer
from model import STHGCN
import data

path = "logs/temp"
folder = os.path.exists(path)
if not folder:      
    os.makedirs(path)    

def train(args, engine, dataloader, device):
    his_loss =[]
    lowest_mae_yet = 10000  
    epochs_since_best_mae = 0
    for i in range(1,args.epochs+1):
        train_loss = []
        train_inflow_mape = []
        train_inflow_rmse = []
        train_outflow_mape = []
        train_outflow_rmse = []
        t1 = time.time()
        dataloader['train_loader'].shuffle()
        pass_count = 0
        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = [torch.Tensor(x0).to(device) for x0 in x]
            trainy = torch.Tensor(y).to(device)
            if(trainy[:,:,:,:].max() == 0):
                pass_count = pass_count + 1
                continue

            metrics = engine.train(trainx, trainy)
            tmp_loss, tmp_inflow_mae, tmp_inflow_mape, tmp_inflow_rmse, tmp_outflow_mae, tmp_outflow_mape, tmp_outflow_rmse = metrics
            
            train_loss.append(tmp_loss)
            train_inflow_mape.append(tmp_inflow_mape)
            train_inflow_rmse.append(tmp_inflow_rmse)
            train_outflow_mape.append(tmp_outflow_mape)
            train_outflow_rmse.append(tmp_outflow_rmse)

        engine.scheduler.step()
        
        t2 = time.time()
        
        #validation
        valid_loss = []
        valid_inflow_mape = []
        valid_inflow_rmse = []
        valid_outflow_mape = []
        valid_outflow_rmse = []
        for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = [torch.Tensor(x0).to(device) for x0 in x]
            testy = torch.Tensor(y).to(device)
            metrics = engine.eval(testx, testy)
            tmp_loss, tmp_inflow_mae, tmp_inflow_mape, tmp_inflow_rmse, tmp_outflow_mae, tmp_outflow_mape, tmp_outflow_rmse = metrics
            
            valid_loss.append(tmp_loss)
            valid_inflow_mape.append(tmp_inflow_mape)
            valid_inflow_rmse.append(tmp_inflow_rmse)
            valid_outflow_mape.append(tmp_outflow_mape)
            valid_outflow_rmse.append(tmp_outflow_rmse)
        mtrain_loss = np.mean(train_loss)
        mtrain_inflow_mape = np.mean(train_inflow_mape)
        mtrain_inflow_rmse = np.mean(train_inflow_rmse)
        mtrain_outflow_mape = np.mean(train_outflow_mape)
        mtrain_outflow_rmse = np.mean(train_outflow_rmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_inflow_mape = np.mean(valid_inflow_mape)
        mvalid_inflow_rmse = np.mean(valid_inflow_rmse)
        mvalid_outflow_mape = np.mean(valid_outflow_mape)
        mvalid_outflow_mse = np.mean(valid_outflow_rmse)
        his_loss.append(mvalid_loss)

        log0 = 'Epoch: {:03d}, Training Time: {:.4f}/epoch'
        log1 = 'Train Loss: {:.4f}, Train Inflow MAPE: {:.4f}, Train Inflow RMSE: {:.4f}, Train Outflow MAPE: {:.4f}, Train Outflow RMSE: {:.4f}'
        log2 = 'Valid Loss: {:.4f}, Valid Inflow MAPE: {:.4f}, Valid Inflow RMSE: {:.4f}, Valid Outflow MAPE: {:.4f}, Valid Outflow RMSE: {:.4f}'
        print(log0.format(i, (t2 - t1)),flush=True)
        print(log1.format(mtrain_loss, mtrain_inflow_mape, mtrain_inflow_rmse, mtrain_outflow_mape, mtrain_outflow_rmse), flush=True)
        print(log2.format(mvalid_loss, mvalid_inflow_mape, mvalid_inflow_rmse, mvalid_outflow_mape, mvalid_outflow_mse), flush=True)

        torch.save(engine.model.state_dict(), "logs/temp/" + args.save+"_epoch_"+str(i)+"_"+str(round(mvalid_loss,2))+".pth")
    
        if mvalid_loss < lowest_mae_yet:
            lowest_mae_yet = mvalid_loss
            epochs_since_best_mae = 0
        else:
            epochs_since_best_mae += 1
        if epochs_since_best_mae >= args.es_patience: 
            break
    return his_loss

def test(args, engine, dataloader, device):
    test_loss = []
    test_inflow_mape = []
    test_inflow_rmse = []
    test_outflow_mape = []
    test_outflow_rmse = []
    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = [torch.Tensor(x0).to(device) for x0 in x]
        testy = torch.Tensor(y).to(device)
        metrics = engine.eval(testx, testy)

        tmp_loss, tmp_inflow_mae, tmp_inflow_mape, tmp_inflow_rmse, tmp_outflow_mae, tmp_outflow_mape, tmp_outflow_rmse = metrics
        test_loss.append(tmp_loss)
        test_inflow_rmse.append(tmp_inflow_rmse)
        test_inflow_mape.append(tmp_inflow_mape)
        test_outflow_mape.append(tmp_outflow_mape)
        test_outflow_rmse.append(tmp_outflow_rmse)

    mtest_loss = np.mean(test_loss)
    mtest_inflow_mape = np.mean(test_inflow_mape)
    mtest_inflow_rmse = np.mean(test_inflow_rmse)
    mtest_outflow_mape = np.mean(test_outflow_mape)
    mtest_outflow_rmse = np.mean(test_outflow_rmse)
    log = 'Test Loss: {:.4f}, Test Inflow MAPE: {:.4f}, Test Inflow RMSE: {:.4f}, Test Outflow MAPE: {:.4f}, Test Outflow RMSE: {:.4f}'
    print(log.format(mtest_loss, mtest_inflow_mape, mtest_inflow_rmse, mtest_outflow_mape, mtest_outflow_rmse),flush=True)

def main(args, **model_kwargs):
    device = torch.device(args.device)

    datas = data.dataset_partition(args.dataset_path, 8, 2, 2, args.timeslice_num, args.week_len, args.day_len, args.batch_size, args.batch_size, args.batch_size, fill_zeroes=args.fill_zeroes)

    scaler = datas['Scaler']
    supports = util.get_adj_matrix(args, device)
    model = STHGCN.from_args(args, device, supports, **model_kwargs)

    model.to(device)
    engine = Trainer.from_args(model, scaler, args, device)
    
    dataloader = datas

    his_loss = train(args, engine, dataloader, device)

    bestid = np.argmin(his_loss)
    engine.model.load_state_dict(torch.load("logs/temp/" + args.save+"_epoch_"+str(bestid+1)+"_"+str(round(his_loss[bestid],2))+".pth"))

    test(args, engine, dataloader, device)
    
    print("The valid loss on best model is", str(round(his_loss[bestid],4)))
    torch.save(engine.model.state_dict(), "logs/" + args.save+"_exp"+"_best_"+str(round(his_loss[bestid],2))+".pth")

    shutil.rmtree("logs/temp/")

if __name__ == "__main__":
    parser = util.get_shared_arg_parser()
    parser.add_argument('--epochs', type=int, default=1000, help='')

    parser.add_argument('--dataset_path', type=str, default="", help='')
    
    parser.add_argument('--row_num', type=int, default=10, help='')
    parser.add_argument('--col_num', type=int, default=20, help='')
    parser.add_argument('--node_nums', type=int, default=200, help='')
    parser.add_argument('--week_len', type=int, default=7, help='')
    parser.add_argument('--day_len', type=int, default=24, help='')
    parser.add_argument('--timeslice_num', type=int, default=4, help='')
    parser.add_argument('--inoutblock_num', type=int, default=2, help='')
    parser.add_argument('--map_type', type=str, default="grid", help='')
    
    parser.add_argument('--clip', type=int, default=3, help='Gradient Clipping')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')
    parser.add_argument('--learning_rate', type=float, default=0.003, help='learning rate')

    parser.add_argument('--lr_decay_rate', type=float, default=0.3, help='learning rate')

    parser.add_argument('--save', type=str, default='experiment', help='save path')
    parser.add_argument('--n_iters', default=None, help='quit after this many iterations')
    parser.add_argument('--es_patience', type=int, default=20, help='quit if no improvement after this many iterations')

    args = parser.parse_args()
    t1 = time.time()
    main(args)
    t2 = time.time()
    mins = (t2 - t1) / 60
    print(f"Total time spent: {mins:.2f} seconds")
