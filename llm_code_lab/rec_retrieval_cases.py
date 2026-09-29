"""Independent small CPU oracles, cases and teaching mutations for R11–R19."""
import math

import torch
import torch.nn.functional as F


IDS = {
    'binary_auc', 'group_auc', 'ranking_metrics', 'calibration_error',
    'embedding_mean_pool', 'cosine_score_matrix', 'bpr_loss',
    'inbatch_softmax_loss', 'sampled_softmax_loss',
}


def _pair_auc(labels, scores):
    """Quadratic pair enumeration deliberately differs from the sorting solution."""
    positive = scores[labels == 1][:, None]
    negative = scores[labels == 0][None, :]
    wins = (positive > negative).to(scores.dtype)
    ties = (positive == negative).to(scores.dtype)
    return (wins + 0.5 * ties).mean()


def reference(pid, *args):
    if pid == 'binary_auc':
        return _pair_auc(*args)
    if pid == 'group_auc':
        labels, scores, group_ids = args
        parts, count = [], 0
        for group in sorted(set(group_ids.tolist())):
            selected = group_ids == group
            y, s = labels[selected], scores[selected]
            if (y == 0).any() and (y == 1).any():
                parts.append(_pair_auc(y, s) * y.numel())
                count += y.numel()
        return torch.stack(parts).sum() / count
    if pid == 'ranking_metrics':
        scores, relevance, k = args
        recalls, ndcgs = [], []
        for score_row, rel_row in zip(scores, relevance):
            rel = rel_row.tolist()
            order = sorted(range(len(rel)), key=lambda j: (-score_row[j].item(), j))[:k]
            n_relevant = sum(r > 0 for r in rel)
            recall = sum(rel[j] > 0 for j in order) / n_relevant if n_relevant else 0.0
            dcg = sum((2 ** rel[j] - 1) / math.log2(rank + 2) for rank, j in enumerate(order))
            ideal = sorted(rel, reverse=True)[:k]
            idcg = sum((2 ** r - 1) / math.log2(rank + 2) for rank, r in enumerate(ideal))
            recalls.append(recall)
            ndcgs.append(dcg / idcg if idcg else 0.0)
        return scores.new_tensor(recalls), scores.new_tensor(ndcgs)
    if pid == 'calibration_error':
        probabilities, labels, n_bins = args
        index = (probabilities * n_bins).floor().long().clamp_max(n_bins - 1)
        total = probabilities.new_zeros(())
        for bucket in range(n_bins):
            selected = index == bucket
            count = int(selected.sum().item())
            if count:
                gap = (probabilities[selected].mean() - labels[selected].to(probabilities.dtype).mean()).abs()
                total = total + count * gap / labels.numel()
        return total
    if pid == 'embedding_mean_pool':
        table, ids, mask = args
        rows = []
        for row_ids, row_mask in zip(ids, mask):
            selected = row_ids[row_mask]
            if selected.numel():
                # Explicit per-occurrence lookup retains duplicate-ID accumulation.
                rows.append(torch.stack([table[int(item)] for item in selected]).mean(0))
            else:
                rows.append(table.sum(0) * 0)
        return torch.stack(rows)
    if pid == 'cosine_score_matrix':
        users, items, temperature, eps = args
        u = F.normalize(users, p=2, dim=1, eps=eps)
        v = F.normalize(items, p=2, dim=1, eps=eps)
        return torch.einsum('bd,md->bm', u, v) / temperature
    if pid == 'bpr_loss':
        return -F.logsigmoid(args[0] - args[1]).mean()
    if pid == 'inbatch_softmax_loss':
        users, items, temperature = args
        logits = F.linear(users, items) / temperature
        return F.cross_entropy(logits, torch.arange(users.shape[0], device=users.device))
    if pid == 'sampled_softmax_loss':
        users, items, item_ids, sampling_probs, temperature = args
        losses = []
        # Per-query candidate lists avoid reusing the vectorized masking solution.
        for i in range(users.shape[0]):
            allowed = [j for j in range(items.shape[0]) if j == i or item_ids[j] != item_ids[i]]
            logits = torch.stack([
                torch.dot(users[i], items[j]) / temperature - sampling_probs[j].log()
                for j in allowed
            ])
            target = torch.tensor([allowed.index(i)], dtype=torch.long, device=users.device)
            losses.append(F.cross_entropy(logits.unsqueeze(0), target))
        return torch.stack(losses).mean()
    raise KeyError(pid)


def make_cases(pid, seed=1729):
    """The first two cases are visible in Run; Submit evaluates both dtypes."""
    if pid not in IDS:
        raise KeyError(pid)
    generator = torch.Generator().manual_seed(seed)
    out = []

    def rand(shape, dtype, scale=1.0):
        return torch.randn(shape, dtype=dtype, generator=generator) * scale

    for dtype in (torch.float64, torch.float32):
        suffix = str(dtype).split('.')[-1]

        def tensor(data):
            return torch.tensor(data, dtype=dtype)

        def integer(data):
            return torch.tensor(data, dtype=torch.long)

        def add(name, *args, grad=False, grad_paths=None):
            case = {'name': name + ' · ' + suffix, 'args': args, 'grad': grad}
            if grad_paths is not None:
                case['grad_paths'] = tuple(grad_paths)
            out.append(case)

        if pid == 'binary_auc':
            add('四样本基础排序', integer([0, 1, 0, 1]), tensor([0.1, 0.4, 0.35, 0.8]))
            add('全部同分应为二分之一', integer([0, 1, 1, 0]), tensor([2, 2, 2, 2]))
            add('完美排序', integer([1, 0, 1, 0]), tensor([9, -8, 4, -2]))
            add('完全逆序', integer([1, 1, 0, 0, 0]), tensor([-2, -1, 0, 1, 3]))
            add('多个同分组与原顺序', integer([1, 0, 1, 0, 0, 1]), tensor([1, 1, 2, 2, 2, 3]))
            add('单正例不均衡', integer([0, 0, 1, 0, 0, 0, 0]), tensor([0, 2, 2, -1, 4, 5, 2]))
            add('单负例随机分数', integer([1, 1, 1, 1, 0, 1, 1]), rand((7,), dtype))
            add('非连续切片', integer([0, 1, 1, 0, 1, 0, 0]), rand((7, 2), dtype)[:, 0])
        elif pid == 'group_auc':
            add('组大小不同且排序相反', integer([0, 1, 0, 1, 1, 1]), tensor([0, 1, 2, 0, 0, 0]), integer([10, 10, 20, 20, 20, 20]))
            add('单类组不能进入分母', integer([0, 1, 0, 0, 0]), tensor([0, 1, 9, 8, 7]), integer([5, 5, 99, 99, 99]))
            add('交错负ID与同分组', integer([1, 0, 0, 1, 1, 0, 0]), tensor([1, 1, 1, 2, 1, 1, 3]), integer([-7, 20, -7, 20, -7, 20, 20]))
            add('组内全同分', integer([0, 1, 0, 1, 1]), tensor([5, 5, -2, -2, -2]), integer([0, 0, 3, 3, 3]))
            add('仅一有效组并含全正全负', integer([1, 1, 0, 1, 0, 0, 0]), tensor([0, 1, 3, 2, 7, 8, 9]), integer([30, 30, 10, 10, -1, -1, -1]))
            add('每组仅一对且全局分布漂移', integer([0, 1, 0, 1, 0, 1]), tensor([100, 101, -100, -99, 0, -1]), integer([9, 9, 8, 8, 7, 7]))
            add('组内同分按半分', integer([0, 1, 1, 0, 1, 0, 0, 1]), tensor([0, 0, 2, 1, 1, 1, 3, 4]), integer([1, 1, 1, 8, 8, 8, 8, 8]))
            add('非连续分数与随机排序', integer([0, 1, 1, 0, 0, 1, 0, 1]), rand((8, 2), dtype)[:, 1], integer([2, 2, 9, 9, 5, 5, 5, 5]))
        elif pid == 'ranking_metrics':
            add('分级相关性与IDCG', tensor([[3, 2, 1]]), integer([[0, 2, 1]]), 2)
            add('同分按原索引与K等于一', tensor([[1, 1, 1]]), integer([[0, 1, 2]]), 1)
            add('K超过候选数量', tensor([[1, 3, 2], [3, 2, 1]]), integer([[4, 0, 1], [0, 0, 0]]), 8)
            add('每个query独立排序', tensor([[1, 9, 3, 2], [4, 3, 2, 1]]), integer([[1, 0, 3, 0], [0, 4, 1, 2]]), 2)
            add('部分同分且等级不同', tensor([[3, 3, 2, 2, 1], [0, 0, 0, 0, 0]]), integer([[0, 4, 1, 2, 3], [3, 0, 2, 1, 4]]), 3)
            add('所有候选相关但只召回一部分', tensor([[2, 1, 4, 3]]), integer([[1, 1, 1, 1]]), 2)
            add('单候选含无相关query', tensor([[0], [8], [-2]]), integer([[1], [0], [4]]), 1)
            add('非连续多query', rand((7, 3), dtype).T, integer([[0, 1, 0, 3, 1, 2, 4], [4, 2, 1, 0, 0, 1, 3], [0, 0, 0, 0, 0, 0, 0]]), 4)
        elif pid == 'calibration_error':
            add('左右端点及桶边界', tensor([0, 0.25, 0.5, 1]), integer([0, 1, 0, 1]), 2)
            add('单桶残差抵消', tensor([0.25, 0.25, 0.75, 0.75]), integer([0, 1, 0, 1]), 1)
            add('空桶与不等桶样本数', tensor([0.05, 0.06, 0.07, 0.9]), integer([0, 1, 0, 1]), 10)
            add('概率零一完全校准', tensor([0, 1, 1, 0, 1]), integer([0, 1, 1, 0, 1]), 10)
            add('概率零一完全反向', tensor([0, 1, 1, 0]), integer([1, 0, 0, 1]), 4)
            add('精确二进制桶边界', tensor([0.125, 0.25, 0.5, 0.75, 0.875, 1]), integer([1, 0, 1, 0, 1, 0]), 8)
            add('单样本', tensor([0.2]), integer([1]), 20)
            add('非连续概率', torch.sigmoid(rand((9, 2), dtype))[:, 0], integer([0, 1, 1, 0, 0, 1, 0, 1, 0]), 3)
        elif pid == 'embedding_mean_pool':
            table = tensor([[1, 2], [3, 4], [5, 6]])
            add('mask分母是有效长度', table, integer([[0, 1, 2]]), torch.tensor([[True, False, True]]), grad=True, grad_paths=('input[0]',))
            add('全padding输出与梯度', table, integer([[0, 1, 2]]), torch.zeros((1, 3), dtype=torch.bool), grad=True, grad_paths=('input[0]',))
            add('重复ID需要梯度累加', rand((5, 4), dtype), integer([[2, 2, 1, 2], [1, 4, 1, 4]]), torch.ones((2, 4), dtype=torch.bool), grad=True, grad_paths=('input[0]',))
            add('混合有效长度与单有效项', rand((7, 3), dtype), integer([[1, 2, 3, 4], [0, 5, 6, 1], [3, 2, 4, 0]]), torch.tensor([[True, False, False, False], [True, True, True, False], [False, False, False, False]]), grad=True, grad_paths=('input[0]',))
            add('空历史L等于零', rand((4, 3), dtype), torch.empty((2, 0), dtype=torch.long), torch.empty((2, 0), dtype=torch.bool), grad=True, grad_paths=('input[0]',))
            add('非连续embedding表', rand((5, 8), dtype).T, integer([[0, 7, 3], [2, 2, 6]]), torch.tensor([[True, False, True], [True, True, False]]), grad=True, grad_paths=('input[0]',))
            add('一维embedding且零ID有效', rand((4, 1), dtype), integer([[0, 0, 3], [3, 2, 0]]), torch.ones((2, 3), dtype=torch.bool), grad=True, grad_paths=('input[0]',))
            add('masked大值不能进入均值', tensor([[10000, -10000], [1, 2], [3, 6]]), integer([[0, 1, 2], [0, 0, 2]]), torch.tensor([[False, True, True], [False, False, True]]), grad=True, grad_paths=('input[0]',))
        elif pid == 'cosine_score_matrix':
            add('非单位向量与温度', tensor([[3, 4]]), tensor([[3, 4], [-4, 3]]), 0.5, 1e-12, grad=True)
            add('零向量与矩形分数矩阵', tensor([[0, 0], [1, 0]]), tensor([[1, 2], [3, 4], [0, 0]]), 1.0, 1e-3, grad=True)
            add('随机B不等于M', rand((3, 5), dtype), rand((7, 5), dtype), 0.2, 1e-12, grad=True)
            add('非连续双侧输入', rand((4, 3), dtype).T, rand((4, 6), dtype).T, 2.0, 1e-5, grad=True)
            add('范数小于epsilon', tensor([[0.0002, 0], [0.0001, -0.0001]]), tensor([[1, 2], [0.0001, 0.0002]]), 0.8, 1e-3, grad=True)
            add('特征维一', tensor([[2], [-3], [0]]), tensor([[-5], [4]]), 0.3, 1e-2, grad=True)
            add('整体缩放不改变余弦', rand((2, 3), dtype, 100), rand((4, 3), dtype, 0.02), 1.0, 1e-12, grad=True)
            add('全零两侧保持计算图', torch.zeros((2, 3), dtype=dtype), torch.zeros((4, 3), dtype=dtype), 1.5, 1e-3, grad=True)
        elif pid == 'bpr_loss':
            add('同分零点梯度', tensor([0, 0]), tensor([0, 0]), grad=True)
            add('极端分差不溢出', tensor([1000, -1000]), tensor([0, 0]), grad=True)
            add('混合胜负与batch均值', tensor([3, -1, 2, 0]), tensor([1, 2, 2, -3]), grad=True)
            add('单对', tensor([0.7]), tensor([-0.3]), grad=True)
            add('共同平移不改变损失', tensor([1003, 999, 1002]), tensor([1001, 1002, 1002]), grad=True)
            add('非连续两侧分数', rand((9, 2), dtype)[:, 0], rand((9, 3), dtype)[:, 1], grad=True)
            add('正例全部更差', tensor([-4, -3, -2, -1]), tensor([0, 1, 2, 3]), grad=True)
            add('随机不同尺度', rand((7,), dtype, 4), rand((7,), dtype, 2), grad=True)
        elif pid == 'inbatch_softmax_loss':
            add('正交两对正例', torch.eye(2, dtype=dtype), torch.eye(2, dtype=dtype), 1.0, grad=True)
            add('单样本没有负例', tensor([[2, -1, 3]]), tensor([[1, 4, -2]]), 0.1, grad=True)
            add('非对称分数区分CE方向', tensor([[1, 2], [3, -1], [0, 2]]), tensor([[2, 0], [1, 3], [-1, 1]]), 0.7, grad=True)
            add('大logits稳定计算', tensor([[30, 0], [0, 30], [-30, 0]]), tensor([[30, 0], [30, 0], [0, 30]]), 1.0, grad=True)
            add('非连续用户与物品向量', rand((5, 4), dtype).T, rand((5, 4), dtype).T, 0.2, grad=True)
            add('所有分数为零但用户梯度不必零', torch.zeros((3, 4), dtype=dtype), rand((3, 4), dtype), 2.0, grad=True)
            add('一维非单位embedding', tensor([[1], [2], [-3], [4]]), tensor([[2], [-1], [1], [3]]), 3.0, grad=True)
            add('随机批次与默认温度', rand((7, 3), dtype, 0.5), rand((7, 3), dtype, 0.5), 0.1, grad=True)
        elif pid == 'sampled_softmax_loss':
            paths = ('input[0]', 'input[1]')
            add('非均匀Q修正包含正例', tensor([[0], [0]]), tensor([[0], [0]]), integer([10, 20]), tensor([0.25, 0.5]), 1.0, grad=True, grad_paths=paths)
            add('同ID只保留对角正例', tensor([[2, 1], [-1, 3]]), tensor([[1, 2], [1, 2]]), integer([10, 10]), tensor([0.25, 0.25]), 0.7, grad=True, grad_paths=paths)
            add('部分重复及非均匀Q', tensor([[1, 2], [3, -1], [0, 2], [2, 0]]), tensor([[2, 0], [1, 3], [2, 0], [-1, 2]]), integer([5, 8, 5, 9]), tensor([0.1, 0.3, 0.1, 0.05]), 0.7, grad=True, grad_paths=paths)
            add('单样本且Q很小', tensor([[3, -2, 1]]), tensor([[1, 2, -3]]), integer([-7]), tensor([1e-8]), 0.1, grad=True, grad_paths=paths)
            add('全部独立ID与均匀Q', rand((4, 3), dtype), rand((4, 3), dtype), integer([11, 22, 33, 44]), tensor([0.02, 0.02, 0.02, 0.02]), 2.0, grad=True, grad_paths=paths)
            add('非连续向量与极小采样概率', rand((5, 3), dtype).T, rand((5, 3), dtype).T, integer([1, 2, 3]), tensor([1e-12, 0.1, 0.8]), 0.2, grad=True, grad_paths=paths)
            add('大logits和正例假负例mask', tensor([[30, 0], [0, 30], [-30, 0]]), tensor([[30, 0], [30, 0], [0, 30]]), integer([4, 4, 7]), tensor([0.4, 0.4, 0.05]), 1.0, grad=True, grad_paths=paths)
            add('其他ID重复负例仍按列计数', tensor([[1, 0], [0, 1], [1, 1], [-1, 2]]), tensor([[1, 2], [3, 1], [3, 1], [3, 1]]), integer([10, 20, 20, 20]), tensor([0.2, 0.3, 0.3, 0.3]), 1.5, grad=True, grad_paths=paths)
    return out


def mutations(bank):
    """One or more characteristic implementation errors per new exercise."""
    result = {}

    def replace(name, pid, old, new):
        solution = bank[pid]['solution']
        if old not in solution:
            raise AssertionError(f'Mutation anchor missing: {name}')
        result[name] = (pid, solution.replace(old, new))

    replace('rec_auc_ties_are_wins', 'binary_auc', '0.5 * negatives', '1.0 * negatives')
    replace('rec_gauc_invalid_groups_in_denominator', 'group_auc', 'return weighted / total', 'return weighted / labels.numel()')
    replace('rec_recall_uses_k_denominator', 'ranking_metrics', '/ relevant_count.clamp_min(1)', '/ k')
    replace('rec_ndcg_linear_gain', 'ranking_metrics', 'torch.pow(2.0, chosen.to(scores.dtype)) - 1', 'chosen.to(scores.dtype)')
    result['rec_ece_individual_absolute_errors'] = ('calibration_error', '''import torch
def calibration_error(probabilities, labels, n_bins=10):
    return (probabilities - labels.to(probabilities.dtype)).abs().mean()
''')
    replace('rec_pool_uses_padded_length', 'embedding_mean_pool',
            'mask.sum(dim=1, keepdim=True).clamp_min(1).to(table.dtype)',
            'table.new_full((ids.shape[0], 1), max(ids.shape[1], 1))')
    replace('rec_cosine_missing_temperature', 'cosine_score_matrix', '/ temperature', '')
    replace('rec_bpr_reversed_difference', 'bpr_loss', 'neg_scores - pos_scores', 'pos_scores - neg_scores')
    replace('rec_inbatch_column_softmax', 'inbatch_softmax_loss', 'torch.logsumexp(scores, dim=1)', 'torch.logsumexp(scores, dim=0)')
    replace('rec_sampled_missing_logq', 'sampled_softmax_loss', ' - sampling_probs.log().unsqueeze(0)', '')
    replace('rec_sampled_keeps_accidental_hits', 'sampled_softmax_loss', 'allowed = (~same_id) | diagonal', 'allowed = torch.ones_like(same_id)')
    return result
