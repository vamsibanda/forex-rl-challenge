import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
from tqdm import tqdm as tq
import os, time, pdb
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
No_Proccess = 21
comm = 1e-4

def calculate_reward(model, loader, index, skip = None):
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
        reward = (weights - last_action[:-1]).abs().sum() * comm
        reward -= (weights * rewards).sum() #- rewards.mean()
        # try risk-sensitive rl e.g. exponential utility
        total_reward = total_reward + (reward / certain)
        last_action[:-1] = weights
        last_action[-1] = certain
        torch.cuda.empty_cache()
    #pb.set_postfix({'R': '{:.6f}'.format(total_reward)})
    skipped = 0 if skip is None else sum(skip)
    total_reward = total_reward / (len(loader) - skipped)
    if skip is None: print('TEST %f' % -total_reward.item())
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
    model.weight.data.fill_(0)
    '''
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
