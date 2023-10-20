import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

def topk_(matrix, K, axis=1):
    if axis == 0:
        row_index = np.arange(matrix.shape[1 - axis])
        topk_index = np.argpartition(-matrix, K, axis=axis)[0:K, :]
        topk_data = matrix[topk_index, row_index]
        topk_index_sort = np.argsort(-topk_data,axis=axis)
        topk_data_sort = topk_data[topk_index_sort,row_index]
        topk_index_sort = topk_index[0:K,:][topk_index_sort,row_index]
    else:
        column_index = np.arange(matrix.shape[1 - axis])[:, None]
        topk_index = np.argpartition(-matrix, K, axis=axis)[:, 0:K]
        topk_data = matrix[column_index, topk_index]
        topk_index_sort = np.argsort(-topk_data, axis=axis)
        topk_data_sort = topk_data[column_index, topk_index_sort]
        topk_index_sort = topk_index[:,0:K][column_index,topk_index_sort]
    return topk_data_sort, topk_index_sort

def knn(x, k):
    # input:
            # x:点云数据，（N,3）
            # k：最近邻点的数量
    # output：
            # distance：距离（n，k）
            # idx：索引(n,k)
    #x:(3,n)
    # print("x.shape:", x.shape)
    # print("x.transpose(1,0):",x.transpose(1,0).shape)
    inner = -2*np.dot(x, x.transpose(1, 0))#(n,n)
    # print("inner:",inner.shape)
    xx = np.expand_dims(np.sum(x**2, axis=1), axis=1)#(n,1)
    # print("xx:",xx.shape)
    # print("xx(1,0):",xx.transpose(1,0).shape)
    # print("x:",x)
    # print("xx:",xx)
    pairwise_distance = -xx - inner - xx.transpose(1, 0)#(n,n)
    # print("pairwise_distance:",pairwise_distance.shape)
    distance, idx = topk_(pairwise_distance, K=k, axis=1)   # (n,k)
    # print("distance:",distance.shape)
    # print("idx:",idx.shape)
    return distance, idx

""" def knn(points, k):
    
    # 计算每个点的k个最近邻点
    # Args:
    #     points: 点云数据，形状为(b, 3, N)，b为batch大小，N为点的数量
    #     k: 最近邻点的数量
    # Returns:
    #     distances: 每个点到其最近邻点的距离，形状为(b, N, k)
    #     indices: 每个点的最近邻点的索引，形状为(b, N, k)
    
    b, _, N = points.size()

    # 计算每对点之间的距离
    distances = torch.cdist(points, points)

    # 将对角线上的距离设置为无穷大，以排除每个点与自身的距离
    # distances.fill_diagonal_(float('inf'))

    # 对距离进行排序，获取每个点的k个最近邻点的索引
    _, indices = torch.topk(distances, k, dim=-1, largest=False)

    # 根据索引获取每个点的k个最近邻点的距离
    distances = torch.gather(distances, -1, indices)

    return distances, indices """

""" def saliency():
    model.eval()
    data_var = Variable(data.permute(0,2,1), requires_grad=True)
    logits = model(data_var)
    loss = cal_loss(logits, label, smoothing=False)
    loss.backward()
    grad = data_var.grad.data#(32,3,1024)
    opt.zero_grad()
    # print("grad:",grad.shape)
    # Change gradients into spherical axis and compute r*dL/dr
    sphere_core = torch.median(data, dim=1, keepdim=True)[0]#(32,1,1024)
    # print("sphere_core:",sphere_core.shape)
    sphere_r = torch.sqrt(torch.sum(torch.square(data - sphere_core), dim=2))  # BxN(32,1024)
    # print("sphere_r:",sphere_r.shape)
    sphere_axis = data - sphere_core  # BxNx3(32,1024,3)
    # print("sphere_axis:",sphere_axis.shape)

    sphere_map = torch.mul(torch.sum(torch.mul(grad.permute(0,2,1), sphere_axis), dim=2), torch.pow(sphere_r, args.power))
    # print("sphere_map:",sphere_map.shape)
    # saliency_map = spherealiency.compute_saliency(data)  # Compute saliency map
    saliency = sphere_map.to(device)  # Convert saliency map to torch tensor
    ###end """

def density_saliency(points, k=20, sigma=0.1):
    """
    计算点云中每个点的密度作为显著性分数
    Args:
        points: 点云数据，形状为(N,3)，N为点的数量
        k: 近邻点的数量
        sigma: 高斯核的标准差
    Returns:
        saliency_scores: 每个点的显著性分数，形状为(N, 1)
    """
    
    # print("points:",points.shape)
    N = points.shape[0]
    # 计算每个点与其k近邻点之间的欧氏距离
    distances, _ = knn(points, k)#(n,k)
    # print("distance:",distances.shape)

    # 计算每个点的高斯核密度估计
    # kernel_distances = distances[:,:, 1:]#(n,k)
    kernel_weights = np.exp(-np.power(distances, 2) / (2 * sigma**2))#(n,k)
    density = kernel_weights.sum(axis=1)#(n)
    # print("density:",density.shape)

    # 归一化密度作为显著性分数
    saliency_scores = density / density.max()#(N)
    # print("saliency_scores:",saliency_scores.shape)

    return saliency_scores

def compute_density(points, k):
    """
    计算每个点的密度
    Args:
        points: 点云数据，形状为(b, 3, N)，b为batch大小，N为点的数量
        k: 最近邻点的数量
    Returns:
        densities: 每个点的密度，形状为(b, N)
    """
    b, _, N = points.size()

    # 寻找每个点的k近邻点
    _, indices = knn(points.view(b*N, 3), k)

    # 将indices的形状转换为(b, N, k)
    indices = indices.view(b, N, k)

    # 计算每个点的密度
    densities = torch.sum(indices, dim=2)

    return densities

def compute_salience_scores(points, k):
    """
    计算每个点的显著性分数
    Args:
        points: 点云数据，形状为(b, 3, N)，b为batch大小，N为点的数量
        k: 最近邻点的数量
    Returns:
        scores: 每个点的显著性分数，形状为(b, N)
    """
    densities = compute_density(points, k)

    # 归一化密度
    mean_density = torch.mean(densities, dim=1, keepdim=True)
    std_density = torch.std(densities, dim=1, keepdim=True)
    normalized_densities = (densities - mean_density) / std_density

    # 计算显著性分数
    scores = F.softmax(normalized_densities, dim=1)

    return scores

if __name__ == "__main__":
    # 示例用法
    points = np.random.rand(32,3)  #(3,n)
    saliency_scores = density_saliency(points)
    print(saliency_scores)
    saliency = np.expand_dims(saliency_scores, axis=1).repeat(4, axis=-1).transpose(1,0)
    print("saliency:",saliency.shape)
    # points = torch.randn(2, 3, 1000)  # 假设有2个点云数据，每个点云有1000个点
    # k = 10  # 设置k值为10
    # distances, indices = knn(points, k)
    # print(distances.shape)
    # print(indices.shape)