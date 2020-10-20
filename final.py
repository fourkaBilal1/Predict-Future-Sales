import numpy as np
import pandas as pd
from itertools import product
import torch
from numpy import array

import gc
import torch.nn as nn
from torch.utils.data import Dataset
from statistics import mean
# from IPython import get_ipython
# def __reset__(): get_ipython().magic('reset -sf')


# set scripts directory to file location
# abspath = os.path.abspath(__file__)
# dname = os.path.dirname(abspath)
# os.chdir(dname)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

sales_train = pd.read_csv("data/sales_train.csv")
blockNumVar = sales_train['date_block_num'].unique()
sales_train = sales_train[sales_train.shop_id==59]
items = pd.read_csv('data/items.csv')
shops = pd.read_csv('data/shops.csv')
item_categories = pd.read_csv('data/item_categories.csv')
sample_submission = pd.read_csv('data/sample_submission.csv')


grid = []
for block_num in blockNumVar:
    cur_shops = sales_train[sales_train['date_block_num']==block_num]['shop_id'].unique()
    cur_items = sales_train['item_id'].unique()
    grid.append(np.array(list(product(*[cur_shops, cur_items, [block_num]])),dtype='int32'))
        
        
index_cols = ['shop_id', 'item_id', 'date_block_num']
grid = pd.DataFrame(np.vstack(grid), columns = index_cols,dtype=np.int32)

# Aggregations
groups = sales_train.groupby(['shop_id', 'item_id', 'date_block_num'])
 
trainset = groups.agg({'item_cnt_day':'sum', 'item_price':'mean'}).reset_index()
trainset = trainset.rename(columns = {'item_cnt_day' : 'item_cnt_month'})


trainset = pd.merge(grid,trainset,how='left',on=index_cols)
trainset.item_cnt_month = trainset.item_cnt_month.fillna(0)

# Get category id
trainset = pd.merge(trainset, items[['item_id', 'item_category_id']], on = 'item_id')
#trainset.to_csv('trainset_with_grid.csv')

trainset.head()
        
        

baseline_features = ['shop_id', 'item_id', 'item_category_id', 'date_block_num', 'item_cnt_month']
train = trainset[baseline_features]
# Remove pandas index column
train = train.set_index('shop_id')
train.item_cnt_month = train.item_cnt_month.astype(int)
train['item_cnt_month'] = train.item_cnt_month.fillna(0)



#for item in train['item_id'].unique():
    
# trainT = train[train.item_id==28]



# train_set = trainT[trainT.date_block_num<=28]
# valid_set = trainT[trainT.date_block_num>28]

# def split_sequence(sequence, n_steps):
#     x, y = list(), list()
#     for i in range(len(sequence)):
        
#         end_ix = i + n_steps
        
#         if end_ix > len(sequence)-1:
#             break
#         seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
#         x.append(seq_x)
#         y.append(seq_y)
#     return array(x), array(y)

# raw_seq = [10,20,30,40,50,60,70,80,90]
# n_steps = 3
# train_x,train_y = split_sequence(train_set.item_cnt_month.values,n_steps)
# valid_x,valid_y = split_sequence(valid_set.item_cnt_month.values,n_steps)





class ElecDataset(Dataset):
    def __init__(self,feature,target):
        self.feature = feature
        self.target = target
    
    def __len__(self):
        return len(self.feature)
    
    def __getitem__(self,idx):
        item = self.feature[idx]
        label = self.target[idx]
        
        return item,label
class CNN_ForecastNet(nn.Module):
    def __init__(self):
        super(CNN_ForecastNet,self).__init__()
        self.conv1d = nn.Conv1d(3,64,kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(64*2,50)
        self.fc2 = nn.Linear(50,1)
        
    def forward(self,x):
        x = self.conv1d(x)
        x = self.relu(x)
        x = x.view(-1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        return x


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model = CNN_ForecastNet().to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
# criterion = nn.MSELoss()



# train = ElecDataset(train_x.reshape(train_x.shape[0],train_x.shape[1],1),train_y)
# valid = ElecDataset(valid_x.reshape(valid_x.shape[0],valid_x.shape[1],1),valid_y)
# train_loader = torch.utils.data.DataLoader(train,batch_size=2,shuffle=False)
# valid_loader = torch.utils.data.DataLoader(train,batch_size=2,shuffle=False)

# train_losses = []
# valid_losses = []
# def Train():
    
#     running_loss = .0
    
#     model.train()
    
#     for idx, (inputs,labels) in enumerate(train_loader):
#         inputs = inputs.to(device,dtype=torch.float32)
#         labels = labels.to(device,dtype=torch.float32)
#         optimizer.zero_grad()
#         preds = model(inputs.float())
#         loss = criterion(preds,labels)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss
        
#     train_loss = running_loss/len(train_loader)
#     train_losses.append(train_loss.detach().numpy())
    
#     print(f'train_loss {train_loss}')
    
# def Valid():
#     running_loss = .0
    
#     model.eval()
    
#     with torch.no_grad():
#         for idx, (inputs, labels) in enumerate(valid_loader):
#             inputs = inputs.to(device)
#             labels = labels.to(device)
#             optimizer.zero_grad()
#             preds = model(inputs.float())
#             loss = criterion(preds,labels)
#             running_loss += loss
            
#         valid_loss = running_loss/len(valid_loader)
#         valid_losses.append(valid_loss.detach().numpy())
#         print(f'valid_loss {valid_loss}')

# epochs = 200
# for epoch in range(epochs):
#     print('epochs {}/{}'.format(epoch+1,epochs))
#     Train()
#     Valid()
#     gc.collect()


def ModelPerItemAndPerShop(shop,item,train):
    trainT = train[train.item_id==item]



    train_set = trainT[trainT.date_block_num<=28]
    valid_set = trainT[trainT.date_block_num>28]
    
    def split_sequence(sequence, n_steps):
        x, y = list(), list()
        for i in range(len(sequence)):
            
            end_ix = i + n_steps
            
            if end_ix > len(sequence)-1:
                break
            seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
            x.append(seq_x)
            y.append(seq_y)
        return array(x), array(y)
    
    #raw_seq = [10,20,30,40,50,60,70,80,90]
    n_steps = 3
    train_x,train_y = split_sequence(train_set.item_cnt_month.values,n_steps)
    valid_x,valid_y = split_sequence(valid_set.item_cnt_month.values,n_steps)
    

    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CNN_ForecastNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.MSELoss()
    
    
    
    train = ElecDataset(train_x.reshape(train_x.shape[0],train_x.shape[1],1),train_y)
    #valid = ElecDataset(valid_x.reshape(valid_x.shape[0],valid_x.shape[1],1),valid_y)
    train_loader = torch.utils.data.DataLoader(train,batch_size=2,shuffle=False)
    valid_loader = torch.utils.data.DataLoader(train,batch_size=2,shuffle=False)
    
    train_losses = []
    valid_losses = []
    def Train():
        running_loss = .0
        
        model.train()
        
        for idx, (inputs,labels) in enumerate(train_loader):
            inputs = inputs.to(device,dtype=torch.float32)
            labels = labels.to(device,dtype=torch.float32)
            optimizer.zero_grad()
            preds = model(inputs.float())
            loss = criterion(preds,labels)
            loss.backward()
            optimizer.step()
            running_loss += loss
            
        train_loss = running_loss/len(train_loader)
        train_losses.append(train_loss.detach().numpy())
        
        #print(f'train_loss {train_loss}')
        return train_loss
        
    def Valid():
        running_loss = .0
        
        model.eval()
        
        with torch.no_grad():
            for idx, (inputs, labels) in enumerate(valid_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                preds = model(inputs.float())
                loss = criterion(preds,labels)
                running_loss += loss
                
            valid_loss = running_loss/len(valid_loader)
            valid_losses.append(valid_loss.detach().numpy())
            #print(f'valid_loss {valid_loss}')
            return valid_loss

    epochs = 200
    for epoch in range(epochs):
        #print('epochs {}/{}'.format(epoch+1,epochs))
        train_loss = Train()
        valid_loss = Valid()
        gc.collect()
    
    print(f'item {item}')
    print(f'train_loss {train_loss}')
    print(f'valid_loss {valid_loss}')
    return train_loss,valid_loss



losses_list = []
Current=0
lenghtT = len(train['item_id'].unique()[1:10])
for itemId in train['item_id'].unique()[1:10]:
    Current +=1
    train_loss,valid_loss = ModelPerItemAndPerShop(59,itemId,train)
    losses_list.append([59,itemId,train_loss.item(),valid_loss.item()])
    print(f'{Current} out of {lenghtT}')



losses_list = np.array(losses_list)

mean_train_loss = mean(losses_list[:,2])
mean_valid_loss = mean(losses_list[:,3])
max_train_loss = max(losses_list[:,2])
max_valid_loss = max(losses_list[:,3])

print(f'mean_train_loss {mean_train_loss}')
print(f'mean_valid_loss {mean_valid_loss}')
print(f'max_train_loss {max_train_loss}')
print(f'max_valid_loss {max_valid_loss}')

