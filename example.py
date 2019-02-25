'''
import numpy as np
import pandas as pd
import empyrical as emp
from tqdm import tqdm as tq
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
import os, time, pdb, random
import _pickle as cPickle
import torch.multiprocessing as _mp
mp = _mp.get_context('spawn')
os.environ["CUDA_VISIBLE_DEVICES"]="0"
#with open('weights.pkl', 'rb') as f: weights = cPickle.load(f)
with open('y_train.pkl', 'rb') as f: y_train = cPickle.load(f)
with open('y_test.pkl', 'rb') as f: y_test = cPickle.load(f)
y_train = torch.from_numpy(y_train)
y_test = torch.from_numpy(y_test)
with open('X_train.pkl', 'rb') as f: X_train = cPickle.load(f)
with open('X_test.pkl', 'rb') as f: X_test = cPickle.load(f)
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
train_loader = torch.utils.data.DataLoader(train_dataset, 
    batch_size = 1, shuffle = False, pin_memory = True)
test_loader = torch.utils.data.DataLoader(test_dataset, 
    batch_size = 1, shuffle = False, pin_memory = True)
No_Channels = 12
No_Proccess = 16
cost = 1e-4

def calculate_reward(model, loader, index, skip = None):
    epoch_weights = []
    epoch_levergs =[]
    #pb = tq(loader, position = index)
    dd = None
    last_action = torch.ones(No_Channels).cuda(dd)
    last_action[:-1] /= float(No_Channels-1)
    total_reward = 0.0
    for i, (features, rewards) in enumerate(loader):
        if skip is not None and skip[i]: continue
        features = features.view(-1).cuda(dd, non_blocking=True)
        rewards = rewards.float().cuda(dd, non_blocking=True)
        state = torch.cat([features, last_action])
        action = model(state)
        weights = torch.tanh(action[:-1])
        certain = 0.5 + torch.sigmoid(action[-1]) / 2.0
        weights = weights / weights.abs().sum()
        reward = (weights - last_action[:-1]).abs().sum() * cost
        reward -= (weights * rewards).sum() #- rewards.mean()
        # try risk-sensitive rl e.g. exponential utility
        total_reward = total_reward + (reward / certain)
        last_action[:-1] = weights
        last_action[-1] = certain
        epoch_weights.append(weights.detach().cpu().numpy())
        epoch_levergs.append(certain.detach().cpu().numpy())
        torch.cuda.empty_cache()
    #pb.set_postfix({'R': '{:.6f}'.format(total_reward)})
    skipped = 0 if skip is None else sum(skip)
    total_reward = total_reward / (len(loader) - skipped)
    if skip is None: 
        print('TEST %f' % -total_reward.item())
        '''
        lv = np.concatenate(epoch_levergs, axis=0)
        ew = np.divide(np.concatenate(epoch_weights, axis=0), lv)
        comm = np.sum(np.abs(ew[1:] - ew[:-1]), axis=1)
        ret = np.sum(np.multiply(ew, y_test.numpy()), axis=1)[1:]
        ret = pd.DataFrame(ret - comm * cost)
        exp = np.exp(ret.resample('1D').sum())-1.0
        ggg = 'Drawdown:', emp.max_drawdown(exp).values[0], 'Sharpe:', emp.sharpe_ratio(exp)[0], \
        'Sortino:', emp.sortino_ratio(exp).values[0], 'Stability:', emp.stability_of_timeseries(exp), \
        'Tail:', emp.tail_ratio(exp), 'ValAtRisk:', emp.value_at_risk(exp)
        print(' '.join(str(x) for x in ggg))
        plt.figure()
        np.exp(ret).cumprod().plot(figsize=(48, 12), title=ttt)
        plt.savefig('annual_return')
        plt.close()
        '''
    return total_reward

def train(model, optimizer, index):
    skip = [(i // (len(train_loader)//No_Proccess)) != index for i in range(len(train_loader))]
    train_reward = calculate_reward(model, train_loader, index, skip)
    #print('train %f' % -train_reward.item())
    optimizer.zero_grad()
    train_reward.backward()
    optimizer.step()
    torch.cuda.empty_cache()

if __name__ == '__main__':
    model = nn.Linear(512 + No_Channels, No_Channels, bias = False).cuda().share_memory()
    '''
    model.weight.data.fill_(0)
    weights = torch.from_numpy(weights)[:11, :]
    model.weight.data[:11,:512] = weights.data
    '''
    optimizer = optim.Adam(params = model.parameters(), lr = 1e-4)
    torch.backends.cudnn.benchmark = True
    for epoch in range(100):
        model.train(True)
        processes = []
        for i in range(No_Proccess):
            p = mp.Process(target=train, args=(model, optimizer, i))
            p.start()
            processes.append(p)
        for p in processes: p.join()
        model.eval()
        calculate_reward(model, test_loader, No_Proccess+1)
