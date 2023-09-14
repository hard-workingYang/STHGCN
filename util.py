import argparse
import pickle
import numpy as np
import os
import util
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.sparse import linalg
import sys 
import data

def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()

def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat= sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()

def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian

def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32).todense()

ADJ_CHOICES = ['scalap', 'normlap', 'symnadj', 'transition', 'identity']
def get_Adj(adj_mx, adjtype):
    if adjtype == "scalap":
        adj = [(adj_mx)]
    elif adjtype == "normlap":
        adj = [calculate_normalized_laplacian(adj_mx).astype(np.float32).todense()]
    elif adjtype == "symnadj":
        adj = [sym_adj(adj_mx)]
    elif adjtype == "transition":
        adj = [asym_adj(adj_mx)]
    elif adjtype == "doubletransition":
        adj = [asym_adj(adj_mx), asym_adj(np.transpose(adj_mx))]
    elif adjtype == "identity":
        adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]
    else:
        error = 0
        assert error, "adj type not defined"
    return adj

def get_adj_matrix(args, device):
    adj_mx = data.adj_matrix(args.row_num, args.col_num)
    adj_mx = get_Adj(adj_mx, args.adjtype)
    supports = [torch.tensor(i).to(device) for i in adj_mx]
    return supports


def get_shared_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:1', help='')
    parser.add_argument('--adjdata', type=str, default='data/sensor_graph/adj_mx.pkl',
                        help='adj data path')
    parser.add_argument('--adjtype', type=str, default='doubletransition', help='adj type', choices=ADJ_CHOICES)
    parser.add_argument('--seq_length', type=int, default=1, help='')

    parser.add_argument('--nhid', type=int, default=40, help='Number of channels for internal conv')

    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--dropout', type=float, default=0.30, help='dropout rate')
    parser.add_argument('--fill_zeroes', action='store_true')
    return parser



def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)/labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)
def masked_metric(pred, real):
    mae = masked_mae(pred,real,0.0).item()
    mape = masked_mape(pred,real,0.0).item()
    rmse = masked_rmse(pred,real,0.0).item()
    return mae,mape,rmse



def masked_mse_with_threadshold(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    
    mask_value = 10
    # mask_pred = torch.where(preds > mask_value, True, False)
    mask_label = torch.where(labels > mask_value, True, False)
    # preds = preds[mask_label]
    # labels = labels[mask_label]

    loss = (preds-labels)**2
    loss = loss * mask

    loss = loss[mask_label]

    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_rmse_with_threadshold(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse_with_threadshold(preds=preds, labels=labels, null_val=null_val))


def masked_mae_with_threadshold(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    
    mask_value = 10
    # mask_pred = torch.where(preds > mask_value, True, False)
    mask_label = torch.where(labels > mask_value, True, False)
    # preds = preds[mask_label]
    # labels = labels[mask_label]
    
    loss = torch.abs(preds-labels)
    loss = loss * mask
    
    loss = loss[mask_label]
    
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mape_with_threadshold(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    
    mask_value = 10
    # mask_pred = torch.where(preds > mask_value, True, False)
    mask_label = torch.where(labels > mask_value, True, False)
    # preds = preds[mask_label]
    # labels = labels[mask_label]

    loss = torch.abs(preds-labels)/labels
    loss = loss * mask

    loss = loss[mask_label]

    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)
def metric_with_threadshold(pred, real):

    mae = masked_mae_with_threadshold(pred,real,0.0).item()
    mape = masked_mape_with_threadshold(pred,real,0.0).item()
    rmse = masked_rmse_with_threadshold(pred,real,0.0).item()
    return mae,mape,rmse


def MAE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean(torch.abs(true-pred))

def MSE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean((pred - true) ** 2)

def RMSE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.sqrt(torch.mean((pred - true) ** 2))



def MAPE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean(torch.abs(torch.div((true - pred), true)))

def MAPE_torch_unmasked(pred, true, mask_value=None):
 
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)

    # print('pred.shape',pred.shape)
    # print('true.shape',true.shape)
    # idx = torch.nonzero(true)
    # print('idx',idx)
    # print('idx.shape',idx.shape)
    # print('true[idx]',true[idx])
    # print('true[idx].shape',true[idx].shape)

    # return np.mean(np.abs((y_true[idx] - y_pred[idx]) / y_true[idx])) * 100
    true = torch.flatten(true)
    pred = torch.flatten(pred)
    idx = torch.nonzero(true)

    # print('torch.mean(torch.abs(torch.div((true - pred), true)))',torch.mean(torch.abs(torch.div((true - pred), true))))
    # print('torch.mean(torch.abs(torch.div((true - pred), true)))',torch.mean(torch.abs(torch.div((true - pred), true))).shape)
    # return torch.mean(torch.abs(torch.div((true - pred), true)))


    return torch.mean(torch.abs(torch.div((true[idx] - pred[idx]), true[idx])))
    return torch.mean(torch.abs((true[idx] - pred[idx]) / true[idx]))



def my_metric(pred, real):
    # mae = MAE_torch(pred,real, 10).item()
    # mape = MAPE_torch(pred,real, 10).item()
    # rmse = RMSE_torch(pred,real, 10).item()

    # threshold = 9.824
    threshold = 10
    # threshold = 2
    mae = MAE_torch(pred,real, threshold)
    mape = MAPE_torch(pred,real, threshold)
    rmse = RMSE_torch(pred,real, threshold)

    # mae = MAE_torch(pred,real, 10)
    # mape = MAPE_torch(pred,real, 10)
    # rmse = RMSE_torch(pred,real, 10)


    # mae = MAE_torch(pred,real, None)
    # mape = MAPE_torch_unmasked(pred,real, None)
    # rmse = RMSE_torch(pred,real, None)

    return mae,mape,rmse

