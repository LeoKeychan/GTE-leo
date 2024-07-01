import numpy as np
import pickle
import torch
import time
from utils import scipy_sparse_mat_to_torch_sparse_tensor, metrics
from my_parser import args
import matplotlib.pyplot as plt

device = 'cpu'
if args.device == 'cuda':
    device = 'cuda:' + args.cuda

# 迭代次数，通过命令行参数设置    
k = args.k  

# 数据加载与处理
f = open('data/'+args.data+'/train_mat.pkl','rb')
train = pickle.load(f)  # 加载训练集数据
f = open('data/'+args.data+'/test_mat.pkl','rb')
test = pickle.load(f)  # 加载测试集数据

# 初始化测试集标签，存储每个用户的物品交互信息
test_labels = [[] for i in range(test.shape[0])]
for i in range(len(test.data)):
    row = test.row[i]  # 用户索引
    col = test.col[i]  # 物品索引
    test_labels[row].append(col)
print('Data loaded and processed.')

start_time = time.time()

# 获取用户数和物品数
n_u, n_i = train.shape
# 物品的初始表示为单位矩阵
item_rep = torch.eye(n_i).to(device)
# 用户的初始表示为零矩阵
user_rep = torch.zeros(n_u,n_i).to(device)

# 处理邻接矩阵
adj = scipy_sparse_mat_to_torch_sparse_tensor(train).coalesce().to(device)

# 图上的迭代表示传播
for i in range(k):
    print("Running layer", i)
    # 使用邻接矩阵更新用户表示
    user_rep_temp = torch.sparse.mm(adj,item_rep) + user_rep  
    # 使用邻接矩阵的转置更新物品表示
    item_rep_temp = torch.sparse.mm(adj.transpose(0,1),user_rep) + item_rep  
    user_rep = user_rep_temp
    item_rep = item_rep_temp

# 评估
pred = user_rep.cpu().numpy()

train_csr = (train!=0).astype(np.float32)

batch_user = 256  # 批处理大小
test_uids = np.array([i for i in range(test.shape[0])])
batch_no = int(np.ceil(len(test_uids)/batch_user))  # 计算批次数

# 存储每个批次的评估指标
batch_recall_20 = []
batch_ndcg_20 = []
batch_recall_40 = []
batch_ndcg_40 = []

all_recall_20 = 0
all_ndcg_20 = 0
all_recall_40 = 0
all_ndcg_40 = 0
for batch in range(batch_no):
    start = batch*batch_user
    end = min((batch+1)*batch_user,len(test_uids))

    preds = pred[start:end]
    # 用于过滤已有交互的评分
    mask = train_csr[start:end].toarray()
    # 应用掩码
    preds = preds * (1-mask)
    # 获取推荐的物品索引
    predictions = (-preds).argsort()
    
    #top@20
    recall_20, ndcg_20 = metrics(test_uids[start:end],predictions,20,test_labels)
    #top@40
    recall_40, ndcg_40 = metrics(test_uids[start:end],predictions,40,test_labels)

    batch_recall_20.append(recall_20)
    batch_ndcg_20.append(ndcg_20)
    batch_recall_40.append(recall_40)
    batch_ndcg_40.append(ndcg_40)

    all_recall_20+=recall_20
    all_ndcg_20+=ndcg_20
    all_recall_40+=recall_40
    all_ndcg_40+=ndcg_40
    print('batch',batch,'recall@20',recall_20,'ndcg@20',ndcg_20,'recall@40',recall_40,'ndcg@40',ndcg_40)
print('-------------------------------------------')
print('recall@20',all_recall_20/batch_no,'ndcg@20',all_ndcg_20/batch_no,'recall@40',all_recall_40/batch_no,'ndcg@40',all_ndcg_40/batch_no)

end_time = time.time()
print("Total running time (seconds):", end_time-start_time)

# 绘图
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(batch_recall_20, label='Recall@20')
plt.plot(batch_recall_40, label='Recall@40')
plt.title('Recall over batches ' + '(' + args.data + ')')
plt.xlabel('Batch')
plt.ylabel('Recall')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(batch_ndcg_20, label='NDCG@20')
plt.plot(batch_ndcg_40, label='NDCG@40')
plt.title('NDCG over batches ' + '(' + args.data + ')')
plt.xlabel('Batch')
plt.ylabel('NDCG')
plt.legend()

plt.tight_layout()
plt.savefig('result/'+args.data+'_result.png')