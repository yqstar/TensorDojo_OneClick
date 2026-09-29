"""Independent references and boundary cases for LLM decoding and training."""
import math

import torch
import torch.nn.functional as F


IDS = {
    'temperature_softmax', 'topk_sampling_probs', 'topp_sampling_probs',
    'repetition_penalty', 'causal_lm_loss', 'sequence_log_probs',
    'kl_distillation_loss', 'dpo_loss', 'reward_pairwise_loss',
    'adamw_step', 'gradient_accumulation', 'token_perplexity',
}


def reference(pid, *args):
    if pid == 'temperature_softmax':
        logits, temperature, dim = args
        return F.softmax(logits / temperature, dim=dim)
    if pid in ('topk_sampling_probs', 'topp_sampling_probs'):
        logits, threshold, temperature = args
        rows = []
        for row in logits:
            order = sorted(range(row.numel()), key=lambda index: (-row[index].item(), index))
            if pid == 'topk_sampling_probs':
                selected = order[:min(threshold, row.numel())]
            elif threshold == 1:
                selected = order
            else:
                probabilities = F.softmax(row / temperature, dim=0)
                selected = []
                mass = row.new_zeros(())
                for index in order:
                    selected.append(index)
                    mass = mass + probabilities[index]
                    if mass.item() >= threshold:
                        break
            selected = torch.tensor(selected, dtype=torch.long, device=row.device)
            values = F.softmax(row[selected] / temperature, dim=0)
            result = torch.zeros_like(row)
            result[selected] = values
            rows.append(result)
        return torch.stack(rows)
    if pid == 'repetition_penalty':
        logits, token_ids, history_mask, penalty = args
        result = logits.clone()
        for b in range(logits.shape[0]):
            seen = set(token_ids[b][history_mask[b]].tolist())
            for token in seen:
                value = logits[b, token]
                result[b, token] = value * penalty if value < 0 else value / penalty
        return result
    if pid == 'causal_lm_loss':
        logits, labels, ignore_index = args
        prediction, target = logits[:, :-1], labels[:, 1:]
        selected = target != ignore_index
        if not selected.any():
            return logits.sum() * 0
        # Boolean-select only valid aligned tokens; official CE is the oracle.
        return F.cross_entropy(prediction[selected], target[selected], reduction='mean')
    if pid == 'sequence_log_probs':
        logits, labels, mask = args
        outputs = []
        for row, target, valid in zip(logits, labels, mask):
            if valid.any():
                outputs.append(-F.cross_entropy(row[valid], target[valid], reduction='sum'))
            else:
                outputs.append(row.sum() * 0)
        return torch.stack(outputs)
    if pid == 'kl_distillation_loss':
        student, teacher, temperature = args
        return F.kl_div(F.log_softmax(student / temperature, dim=1),
                        F.log_softmax(teacher / temperature, dim=1),
                        reduction='batchmean', log_target=True) * temperature ** 2
    if pid == 'dpo_loss':
        chosen, rejected, ref_chosen, ref_rejected, beta = args
        # Form the implicit reward of each response before taking a pair gap.
        chosen_reward = beta * (chosen - ref_chosen)
        rejected_reward = beta * (rejected - ref_rejected)
        return -F.logsigmoid(chosen_reward - rejected_reward).mean()
    if pid == 'reward_pairwise_loss':
        chosen, rejected, margin = args
        return -F.logsigmoid(chosen - rejected - margin).mean()
    if pid == 'adamw_step':
        p, g, m, v, step, lr, beta1, beta2, eps, weight_decay = args
        parameter = torch.nn.Parameter(p.detach().clone())
        optimizer = torch.optim.AdamW([parameter], lr=lr, betas=(beta1, beta2),
                                      eps=eps, weight_decay=weight_decay, foreach=False)
        optimizer.state[parameter] = {
            'step': torch.tensor(float(step - 1)),
            'exp_avg': m.detach().clone(),
            'exp_avg_sq': v.detach().clone(),
        }
        parameter.grad = g.detach().clone()
        optimizer.step()
        state = optimizer.state[parameter]
        return parameter.detach().clone(), state['exp_avg'].clone(), state['exp_avg_sq'].clone()
    if pid == 'gradient_accumulation':
        gradients, sizes = args
        # Represent each microbatch mean as that many identical sample gradients.
        samples = [gradient.unsqueeze(0).expand(count, *gradient.shape)
                   for gradient, count in zip(gradients, sizes)]
        return torch.cat(samples, dim=0).mean(dim=0)
    if pid == 'token_perplexity':
        nll, mask = args
        selected = nll[mask]
        return selected.mean().exp() if selected.numel() else nll.new_ones(())
    raise KeyError(pid)


def make_cases(pid, seed=1729):
    if pid not in IDS:
        raise KeyError(pid)
    generator = torch.Generator().manual_seed(seed)
    out = []
    differentiable = {
        'temperature_softmax': ('input[0]',),
        'causal_lm_loss': ('input[0]',),
        'sequence_log_probs': ('input[0]',),
        'kl_distillation_loss': ('input[0]',),
        'dpo_loss': ('input[0]', 'input[1]'),
        'reward_pairwise_loss': ('input[0]', 'input[1]'),
    }

    def rand(shape, dtype, scale=1.0):
        return torch.randn(shape, generator=generator, dtype=dtype) * scale

    for dtype in (torch.float64, torch.float32):
        suffix = str(dtype).split('.')[-1]

        def tensor(values):
            return torch.tensor(values, dtype=dtype)

        def integer(values):
            return torch.tensor(values, dtype=torch.long)

        def boolean(values):
            return torch.tensor(values, dtype=torch.bool)

        def add(name, *args, connected=False):
            case = {'name': name + ' · ' + suffix, 'args': args, 'grad': pid in differentiable}
            if pid in differentiable:
                case['grad_paths'] = differentiable[pid]
                if connected:
                    case['require_connected_paths'] = differentiable[pid]
            out.append(case)

        if pid == 'temperature_softmax':
            add('温度开平方概率比', tensor([0, math.log(4)]), 2.0, -1)
            add('同分低温仍均匀', tensor([[5, 5, 5]]), 0.1, 1)
            add('极端logits', tensor([[1000, 999, -1000], [-1000, 0, 1000]]), 0.5, -1)
            add('沿batch维归一化', rand((3, 5), dtype), 1.7, 0)
            add('非连续多维输入', rand((2, 5, 3), dtype).transpose(1, 2), 0.7, 1)
            add('单元素归一化维', rand((4, 1), dtype), 3.0, -1)
        elif pid == 'topk_sampling_probs':
            add('只在前k项内归一化', tensor([[0, math.log(2), math.log(4)]]), 2, 1.0)
            add('同分只留一个且索引稳定', tensor([[1, 1, 1]]), 1, 1.0)
            add('k大于词表且非一温度', tensor([[3, -2, 1], [0, 0, 0]]), 9, 2.0)
            add('边界多个同分仍恰好k项', tensor([[2, 1, 1, 1, -2], [1, 1, 3, 3, 3]]), 3, 0.7)
            add('极端logits和单词表', tensor([[1000], [-1000]]), 2, 0.5)
            add('非连续batch独立截断', rand((7, 3), dtype).T, 4, 1.3)
        elif pid == 'topp_sampling_probs':
            add('保留跨过阈值的token', tensor([[0.5, 0.3, 0.2]]).log(), 0.6, 1.0)
            add('精确等于阈值即停止', torch.zeros((1, 4), dtype=dtype), 0.5, 1.0)
            add('阈值很小至少保留一个', tensor([[1, 2, 3], [1, 1, 1]]), 0.01, 1.0)
            add('p等于一保留全部', tensor([[0, -2, -4], [1000, 0, -1000]]), 1.0, 0.5)
            add('温度改变nucleus大小', tensor([[2, 1, 0], [0, 0, 0]]), 0.8, 3.0)
            add('同分边界稳定原索引', tensor([[0, 2, 2, 2, 0], [2, 0, 2, 0, 2]]), 0.5, 0.3)
            add('非连续与每行不同候选数', rand((7, 3), dtype).T, 0.76, 0.8)
            add('单token词表', tensor([[1], [-1000]]), 0.4, 2.0)
        elif pid == 'repetition_penalty':
            add('正负号与重复ID只罚一次', tensor([[2, -3, 0, 4]]), integer([[0, 1, 1]]), boolean([[1, 1, 1]]), 2.0)
            add('全mask保持原值', tensor([[2, -3, 0, 4]]), integer([[0, 1, 1]]), boolean([[0, 0, 0]]), 2.0)
            add('不同batch历史与有效mask', tensor([[1, -2, 3, -4], [5, -6, 0, 8]]), integer([[0, 2, 1, 3], [0, 2, 1, 3]]), boolean([[1, 0, 1, 0], [0, 1, 0, 1]]), 1.2)
            add('空历史', rand((2, 5), dtype), torch.empty((2, 0), dtype=torch.long), torch.empty((2, 0), dtype=torch.bool), 3.0)
            add('penalty一恒等', rand((3, 4), dtype), integer([[0, 1], [2, 3], [0, 3]]), torch.ones((3, 2), dtype=torch.bool), 1.0)
            add('非连续logits', rand((5, 3), dtype).T, integer([[1, 1, 4], [2, 0, 2], [3, 4, 1]]), boolean([[1, 1, 1], [1, 0, 1], [0, 1, 1]]), 1.7)
        elif pid == 'causal_lm_loss':
            add('均匀词表的右移目标', torch.zeros((1, 3, 2), dtype=dtype), integer([[1, 0, 1]]), -100)
            add('全部忽略需图连零', rand((1, 3, 2), dtype), integer([[1, -100, -100]]), -100, connected=True)
            add('明显错位目标与末位无梯度', tensor([[[5, 0, -2], [0, 5, -2], [-2, 0, 5], [8, 8, 8]]]), integer([[2, 0, 1, 2]]), -100)
            add('不同序列有效token数', rand((2, 4, 5), dtype), integer([[0, 1, -100, -100], [2, 3, 4, 1]]), -100)
            add('T等于一无预测目标', rand((2, 1, 4), dtype), integer([[0], [-100]]), -100, connected=True)
            add('非连续与自定义忽略标签', rand((2, 5, 4), dtype).transpose(1, 2), integer([[2, -9, 3, 1], [-9, 0, 2, -9]]), -9)
            add('极大logits且首标签忽略', tensor([[[1000, -1000], [-1000, 1000], [0, 0]]]), integer([[-100, 1, 1]]), -100)
        elif pid == 'sequence_log_probs':
            add('按有效token求和而非均值', torch.zeros((1, 3, 2), dtype=dtype), integer([[0, 1, 0]]), boolean([[1, 1, 0]]))
            add('无效标签越界也安全', rand((2, 3, 4), dtype), integer([[999, -100, 999], [1, -999, 2]]), boolean([[0, 0, 0], [1, 0, 1]]), connected=True)
            add('输入已对齐不能再shift', tensor([[[4, 0, -3], [0, -3, 4], [-3, 4, 0]]]), integer([[0, 2, 1]]), torch.ones((1, 3), dtype=torch.bool))
            add('空序列输出B个图连零', torch.empty((3, 0, 5), dtype=dtype), torch.empty((3, 0), dtype=torch.long), torch.empty((3, 0), dtype=torch.bool), connected=True)
            add('非连续与不同长度', rand((2, 5, 4), dtype).transpose(1, 2), integer([[0, 1, 3, 2], [4, 2, 1, 0]]), boolean([[1, 0, 0, 0], [1, 1, 1, 1]]))
            add('极端logits不能先概率再log', tensor([[[1000, -1000], [-1000, 1000]]]), integer([[1, 0]]), torch.ones((1, 2), dtype=torch.bool))
            add('单类词表log概率为零', rand((2, 3, 1), dtype), integer([[0, 0, 0], [0, 0, 0]]), boolean([[1, 0, 1], [1, 1, 1]]), connected=True)
        elif pid == 'kl_distillation_loss':
            same = tensor([[1, 2, 3]])
            add('教师学生相同', same, same.clone(), 2.0)
            add('非对称KL方向', tensor([[0, 0]]), tensor([[math.log(3), 0]]), 1.0)
            add('非一温度必须乘T平方', rand((3, 5), dtype), rand((3, 5), dtype, 2), 3.0)
            add('极端教师概率下溢', tensor([[1000, -1000, 0], [0, -1000, 1000]]), tensor([[-1000, 1000, 0], [1000, 0, -1000]]), 0.5)
            add('非连续和按batchmean归约', rand((7, 2), dtype).T, rand((7, 2), dtype).T, 1.7)
            add('单类词表保持学生零梯度', rand((4, 1), dtype), rand((4, 1), dtype), 2.0, connected=True)
        elif pid == 'dpo_loss':
            add('策略与参考偏好差相等', tensor([-2]), tensor([-4]), tensor([-2]), tensor([-4]), 0.1)
            add('策略增强chosen偏好', tensor([-1]), tensor([-3]), tensor([-2]), tensor([-2]), 1.0)
            add('非零参考差与batch均值', tensor([-1, -8, -5]), tensor([-3, -4, -5]), tensor([-2, -5, -1]), tensor([-6, -7, -5]), 0.3)
            add('极端偏好差稳定性', tensor([-1, -2001]), tensor([-2001, -1]), tensor([-1, -1]), tensor([-1, -1]), 2.0)
            add('非连续概率和', -rand((7, 2), dtype).abs()[:, 0], -rand((7, 2), dtype).abs()[:, 1], -rand((7, 2), dtype).abs()[:, 0], -rand((7, 2), dtype).abs()[:, 1], 0.7)
            add('参考差抵消策略优势', tensor([-2, -3]), tensor([-5, -9]), tensor([-1, -2]), tensor([-6, -7]), 0.05)
        elif pid == 'reward_pairwise_loss':
            add('同分梯度与log二', tensor([0, 0]), tensor([0, 0]), 0.0)
            add('margin抵消已有优势', tensor([2]), tensor([0]), 2.0)
            add('极端奖励差稳定', tensor([1000, -1000]), tensor([0, 0]), 0.5)
            add('非连续输入', rand((7, 2), dtype)[:, 0], rand((7, 3), dtype)[:, 1], 1.3)
            add('多样本归约', tensor([3, -2, 1, 5]), tensor([0, 4, 1, 2]), 0.7)
            add('奖励共同平移', tensor([1003, 998, 1001]), tensor([1000, 1004, 1001]), 0.0)
        elif pid == 'adamw_step':
            def add_step(name, p, g, m, v, step=1, lr=0.1, b1=0.9, b2=0.999, eps=1e-8, wd=0.01):
                add(name, p, g, m, v, step, lr, b1, b2, eps, wd)

            add_step('零梯度仅解耦衰减', tensor([2]), tensor([0]), tensor([0]), tensor([0]), wd=0.5)
            add_step('学习率零仍更新状态', tensor([2, -1]), tensor([1, -2]), tensor([0.5, 0.3]), tensor([0.2, 0.7]), step=4, lr=0)
            add_step('首步偏差修正和非零衰减', tensor([1, -3, 2]), tensor([0.5, -0.2, 0]), torch.zeros(3, dtype=dtype), torch.zeros(3, dtype=dtype), wd=0.2)
            add_step('非零旧状态与不同beta', rand((2, 3), dtype), rand((2, 3), dtype), rand((2, 3), dtype, 0.2), rand((2, 3), dtype).square(), step=17, b1=0.7, b2=0.8, wd=0.1)
            add_step('epsilon必须在平方根外', tensor([1, -1]), tensor([1e-4, -2e-4]), torch.zeros(2, dtype=dtype), torch.zeros(2, dtype=dtype), b1=0.0, b2=0.0, eps=0.01, wd=0)
            add_step('非连续矩阵状态', rand((5, 3), dtype).T, rand((5, 3), dtype).T, rand((5, 3), dtype).T, rand((5, 3), dtype).square().T, step=3, wd=0.7)
            p, m, v = tensor([2, -1, 0.5]), torch.zeros(3, dtype=dtype), torch.zeros(3, dtype=dtype)
            for step, gradient in enumerate((tensor([1, -2, 0]), tensor([-0.5, 0.3, 1]), tensor([0.1, -1, -0.7])), 1):
                arguments = (p, gradient, m, v, step, 0.03, 0.8, 0.9, 1e-5, 0.15)
                add('连续轨迹状态第' + str(step) + '步', *arguments)
                p, m, v = reference(pid, *arguments)
        elif pid == 'gradient_accumulation':
            add('不等microbatch加权', (tensor([1, 3]), tensor([5, 7])), (1, 3))
            add('等大batch退化均值', (tensor([2]), tensor([8])), (2, 2))
            add('标量参数与三个batch', (tensor(2), tensor(-4), tensor(7)), (3, 1, 2))
            add('单microbatch数值不变', (rand((2, 3, 4), dtype),), (7,))
            add('非连续多维参数', (rand((5, 3), dtype).T, rand((5, 3), dtype).T, rand((5, 3), dtype).T), (2, 7, 1))
            add('相反梯度按数量抵消', (tensor([3, -6]), tensor([-1, 2])), (1, 3))
        elif pid == 'token_perplexity':
            add('先均值NLL再exp', tensor([[math.log(2), math.log(8)]]), boolean([[1, 1]]))
            add('全mask约定返回一', tensor([[5, 7]]), boolean([[0, 0]]))
            add('不同序列有效数量', tensor([[0, 10, 10], [1, 2, 3]]), boolean([[1, 0, 0], [1, 1, 1]]))
            add('空时间维', torch.empty((3, 0), dtype=dtype), torch.empty((3, 0), dtype=torch.bool))
            add('高但有限NLL', tensor([[50, 0], [40, 30]]), boolean([[1, 0], [1, 1]]))
            add('非连续与单个有效token', rand((5, 3), dtype).abs().T, boolean([[0, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0]]))
    return out


def mutations(bank):
    result = {}

    def replace(name, pid, old, new):
        code = bank[pid]['solution']
        if old not in code:
            raise AssertionError('Missing mutation anchor: ' + name)
        result[name] = (pid, code.replace(old, new))

    replace('llm_temperature_multiplied', 'temperature_softmax', '/ temperature', '* temperature')
    replace('llm_topk_unnormalized', 'topk_sampling_probs',
            'probabilities = probabilities / probabilities.sum(1, keepdim=True)',
            'probabilities = probabilities / (probabilities.sum(1, keepdim=True) + 1)')
    result['llm_topk_includes_all_boundary_ties'] = ('topk_sampling_probs', '''import torch
def topk_sampling_probs(logits, k, temperature=1.0):
    threshold = torch.topk(logits, min(k, logits.shape[1]), dim=1).values[:, -1:]
    return torch.softmax((logits / temperature).masked_fill(logits < threshold, float('-inf')), dim=1)
''')
    replace('llm_topp_drops_crossing_token', 'topp_sampling_probs',
            'else previous < top_p', 'else probabilities.cumsum(1) <= top_p')
    replace('llm_topp_keeps_extra_at_exact_threshold', 'topp_sampling_probs', 'else previous < top_p', 'else previous <= top_p')
    replace('llm_repetition_wrong_negative_sign', 'repetition_penalty',
            'torch.where(logits < 0, logits * penalty, logits / penalty)', 'logits / penalty')
    replace('llm_causal_no_shift', 'causal_lm_loss', 'logits[:, :-1], labels[:, 1:]', 'logits, labels')
    replace('llm_causal_counts_padding', 'causal_lm_loss', 'valid.sum().clamp_min(1)', 'max(valid.numel(), 1)')
    replace('llm_sequence_means_instead_of_sums', 'sequence_log_probs',
            'return (selected * mask.to(logits.dtype)).sum(dim=1)',
            'return (selected * mask.to(logits.dtype)).sum(dim=1) / mask.sum(dim=1).clamp_min(1)')
    replace('llm_distillation_missing_t_squared', 'kl_distillation_loss', 'per_row.mean() * temperature ** 2', 'per_row.mean()')
    replace('llm_distillation_reverse_kl', 'kl_distillation_loss',
            'log_q.exp() * (log_q - log_p)', 'log_p.exp() * (log_p - log_q)')
    replace('llm_dpo_ignores_reference', 'dpo_loss', 'beta * (policy_gap - reference_gap)', 'beta * policy_gap')
    replace('llm_reward_wrong_margin_sign', 'reward_pairwise_loss',
            'rejected_rewards - chosen_rewards + margin', 'rejected_rewards - chosen_rewards - margin')
    replace('llm_adamw_decay_coupled_to_grad', 'adamw_step',
            '    new_m = beta1 * exp_avg', '    grad = grad + weight_decay * param\n    new_m = beta1 * exp_avg')
    replace('llm_adamw_missing_decay', 'adamw_step', 'param * (1 - lr * weight_decay)', 'param')
    result['llm_accumulation_equal_batch_weights'] = ('gradient_accumulation', '''import torch
def gradient_accumulation(microbatch_grads, batch_sizes):
    return torch.stack(microbatch_grads).mean(0)
''')
    result['llm_perplexity_average_of_exp'] = ('token_perplexity', '''import torch
def token_perplexity(token_nll, mask):
    if not mask.any():
        return token_nll.new_ones(())
    return token_nll[mask].exp().mean()
''')
    return result
