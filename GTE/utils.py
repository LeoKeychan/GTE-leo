import torch
import numpy as np

def scipy_sparse_mat_to_torch_sparse_tensor(sparse_mx):
    # 将 SciPy 稀疏矩阵转换为 COOrdinate format（坐标格式）
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    # 从稀疏矩阵中提取非零元素的索引
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    # 从稀疏矩阵中提取非零元素的值
    values = torch.from_numpy(sparse_mx.data)
    # 创建张量的形状
    shape = torch.Size(sparse_mx.shape)
    # 返回一个 PyTorch 稀疏张量
    return torch.sparse.FloatTensor(indices, values, shape)

def metrics(uids, predictions, topk, test_labels):
    user_num = 0  # 记录参与计算指标的用户数量
    all_recall = 0  # 累计召回率
    all_ndcg = 0  # 累计NDCG
    for i in range(len(uids)):
        uid = uids[i]  # 当前用户ID
        # 模型对该用户的前k个推荐
        prediction = list(predictions[i][:topk])
        # 实际用户感兴趣的物品列表
        label = test_labels[uid]
        # 确保用户有实际感兴趣的物品
        if len(label)>0:
            hit = 0  # 记录命中数
            # 计算理想情况下的最大DCG
            idcg = np.sum([np.reciprocal(np.log2(loc + 2)) for loc in range(min(topk, len(label)))])
            dcg = 0  # 累计折扣累积增益
            for item in label:
                if item in prediction:
                    hit += 1
                    loc = prediction.index(item)
                    dcg = dcg + np.reciprocal(np.log2(loc+2))
            all_recall = all_recall + hit/len(label)  # 更新累计召回率
            all_ndcg = all_ndcg + dcg/idcg  # 更新累计NDCG
            user_num += 1  # 更新参与计算的用户数
    # 返回平均召回率和平均NDCG
    return all_recall/user_num, all_ndcg/user_num

