import torch


def _init_centroids_plus_plus(x: torch.Tensor, k: int, generator: torch.Generator) -> torch.Tensor:
    n, d = x.shape
    centroids = torch.empty((k, d), device=x.device, dtype=x.dtype)
    first_idx = torch.randint(0, n, (1,), generator=generator, device=x.device).item()
    centroids[0] = x[first_idx]

    min_dist_sq = torch.sum((x - centroids[0]) ** 2, dim=1)
    for c_idx in range(1, k):
        probs = min_dist_sq / torch.sum(min_dist_sq).clamp_min(1e-12)
        next_idx = torch.multinomial(probs, 1, generator=generator).item()
        centroids[c_idx] = x[next_idx]
        dist_sq = torch.sum((x - centroids[c_idx]) ** 2, dim=1)
        min_dist_sq = torch.minimum(min_dist_sq, dist_sq)

    return centroids


@torch.no_grad()
def kmeans_gpu(
    x: torch.Tensor,
    n_clusters: int,
    max_iter: int = 100,
    tol: float = 1e-4,
    seed: int = 42,
):
    """
    x: shape (N, D)，建议已放在 cuda 上，如 x = x.cuda()
    返回: labels (N,), centroids (K, D)
    """
    device = x.device
    n, d = x.shape
    generator = torch.Generator(device=device).manual_seed(seed)

    centroids = _init_centroids_plus_plus(x, n_clusters, generator)
    labels = torch.zeros(n, dtype=torch.long, device=device)

    for _ in range(max_iter):
        # 计算 (N, K) 欧氏距离矩阵
        dists = torch.cdist(x, centroids, p=2.0)
        new_labels = torch.argmin(dists, dim=1)

        new_centroids = torch.zeros_like(centroids)
        counts = torch.bincount(new_labels, minlength=n_clusters).to(x.dtype)

        # 累加各簇样本
        new_centroids.index_add_(0, new_labels, x)

        # 处理空簇：随机用现有样本重新初始化
        empty_mask = counts == 0
        if torch.any(empty_mask):
            num_empty = int(empty_mask.sum().item())
            rand_idx = torch.randint(0, n, (num_empty,), generator=generator, device=device)
            new_centroids[empty_mask] = x[rand_idx]
            counts[empty_mask] = 1.0

        new_centroids = new_centroids / counts.unsqueeze(1)

        # 判断质心最大移动距离是否收敛
        shift = torch.max(torch.norm(new_centroids - centroids, dim=1))
        centroids = new_centroids
        labels = new_labels

        if shift <= tol:
            break

    return labels, centroids