"""Independent finite CPU cases for reinforcement-learning components."""
import torch
import torch.nn.functional as F

IDS = {
    'discounted_returns', 'td0_target', 'gae_advantages', 'normalize_advantages',
    'reinforce_loss', 'categorical_entropy', 'importance_sampling_ratio',
    'ppo_clipped_loss', 'ppo_value_loss', 'dqn_loss', 'double_dqn_target',
    'polyak_update', 'value_iteration_step', 'c51_projection',
    'grpo_advantages', 'grpo_loss',
}


def reference(pid, *args):
    if pid == 'discounted_returns':
        rewards, terminated, bootstrap, gamma = args
        # Enumerate each future reward rather than reusing backward recursion.
        rows = []
        for t in range(rewards.shape[0]):
            total = torch.zeros_like(bootstrap)
            discount = torch.ones_like(bootstrap)
            for k in range(t, rewards.shape[0]):
                total = total + discount * rewards[k]
                discount = discount * gamma * (~terminated[k]).to(rewards.dtype)
            rows.append(total + discount * bootstrap)
        return torch.stack(rows)
    if pid == 'td0_target':
        reward, next_value, terminal, gamma = args
        return torch.where(terminal, reward, reward + gamma * next_value)
    if pid == 'gae_advantages':
        rewards, values, next_values, terminated, truncated, gamma, lam = args
        delta = rewards - values + torch.where(terminated, torch.zeros_like(next_values), gamma * next_values)
        rows = []
        # Finite weighted sum of TD residuals with independent boundary products.
        for t in range(rewards.shape[0]):
            total = torch.zeros_like(values[0])
            weight = torch.ones_like(values[0])
            for k in range(t, rewards.shape[0]):
                total = total + weight * delta[k]
                weight = torch.where(terminated[k] | truncated[k], torch.zeros_like(weight), weight * gamma * lam)
            rows.append(total)
        advantage = torch.stack(rows)
        return advantage, advantage + values
    if pid == 'normalize_advantages':
        advantages, eps = args
        return (advantages - advantages.mean()) / (torch.std(advantages, unbiased=False) + eps)
    if pid == 'reinforce_loss':
        logits, actions, returns = args
        return (F.cross_entropy(logits, actions, reduction='none') * returns).mean()
    if pid == 'categorical_entropy':
        return torch.distributions.Categorical(logits=args[0]).entropy()
    if pid == 'importance_sampling_ratio':
        new, old = args
        # A common centering constant prevents underflow in the quotient oracle.
        center = torch.maximum(new, old)
        return (new - center).exp() / (old - center).exp()
    if pid == 'ppo_clipped_loss':
        new, old, advantage, epsilon = args
        ratio = (new - old).exp()
        pessimistic = torch.where(advantage >= 0, ratio.clamp(max=1 + epsilon), ratio.clamp(min=1 - epsilon))
        return -(pessimistic * advantage).mean()
    if pid == 'ppo_value_loss':
        new, old, returns, epsilon = args
        # Strict inequalities preserve clamp's derivative 1 at either endpoint.
        clipped = torch.where(new < old - epsilon, old - epsilon,
                              torch.where(new > old + epsilon, old + epsilon, new))
        errors = torch.stack(((new - returns).square(), (clipped - returns).square()))
        return errors.amax(dim=0).mean() / 2
    if pid == 'dqn_loss':
        q, actions, reward, next_value, terminal, gamma = args
        predicted = (q * F.one_hot(actions, q.shape[1]).to(q.dtype)).sum(-1)
        target = torch.where(terminal, reward, reward + gamma * next_value)
        return F.smooth_l1_loss(predicted, target, beta=1.0)
    if pid == 'double_dqn_target':
        rewards, online, target, terminal, gamma = args
        outputs = []
        for i in range(rewards.numel()):
            # Python max keeps the first action on ties, independently of torch.argmax.
            action = max(range(online.shape[1]), key=lambda a: online[i, a].item())
            outputs.append(rewards[i] if terminal[i] else rewards[i] + gamma * target[i, action])
        return torch.stack(outputs)
    if pid == 'polyak_update':
        online, target, tau = args
        return torch.lerp(target, online, tau)
    if pid == 'value_iteration_step':
        values, transition, rewards, terminal, gamma = args
        result = []
        for s in range(values.numel()):
            action_values = []
            for a in range(transition.shape[1]):
                total = values.new_zeros(())
                for nxt in range(values.numel()):
                    continuation = values.new_zeros(()) if terminal[s, a, nxt] else gamma * values[nxt]
                    total = total + transition[s, a, nxt] * (rewards[s, a, nxt] + continuation)
                action_values.append(total)
            result.append(torch.stack(action_values).amax())
        return torch.stack(result)
    if pid == 'c51_projection':
        probabilities, rewards, terminal, support, gamma = args
        delta = (support[-1] - support[0]) / (support.numel() - 1)
        moved = rewards[:, None] + torch.where(terminal[:, None], 0., gamma * support[None, :])
        moved = moved.clamp(min=support[0], max=support[-1])
        # Triangular interpolation kernels avoid floor/ceil/scatter and retain exact atoms.
        distance = (moved[:, :, None] - support[None, None, :]).abs() / delta
        interpolation = (1 - distance).clamp(min=0, max=1)
        return torch.einsum('bj,bji->bi', probabilities, interpolation)
    if pid == 'grpo_advantages':
        rewards, eps = args
        return torch.stack([(row - row.mean()) / (torch.std(row, unbiased=False) + eps) for row in rewards])
    if pid == 'grpo_loss':
        new, old, ref, advantage, mask, epsilon, beta = args
        sequences = []
        for i in range(new.shape[0]):
            selected = mask[i]
            if not selected.any():
                continue
            ratios = (new[i, selected] - old[i, selected]).exp()
            pessimistic = ratios.clamp(max=1 + epsilon) if advantage[i] >= 0 else ratios.clamp(min=1 - epsilon)
            difference = ref[i, selected] - new[i, selected]
            kl = difference.exp() - 1 - difference
            sequences.append((-pessimistic * advantage[i] + beta * kl).mean())
        return torch.stack(sequences).mean() if sequences else new.sum() * 0
    raise KeyError(pid)


def make_cases(pid, seed=1729):
    if pid not in IDS:
        raise KeyError(pid)
    g = torch.Generator().manual_seed(seed)
    cases = []
    for dtype in (torch.float64, torch.float32):
        suffix = ' · ' + str(dtype).split('.')[-1]
        def tensor(data):
            return torch.tensor(data, dtype=dtype)
        def boolean(data):
            return torch.tensor(data, dtype=torch.bool)
        def integer(data):
            return torch.tensor(data, dtype=torch.long)
        def rand(*shape, scale=1.):
            return torch.randn(shape, generator=g, dtype=dtype) * scale
        def add(name, *args, grad=False, paths=('input[0]',), connected=False):
            item = {'name': name + suffix, 'args': args, 'grad': grad}
            if grad:
                item['grad_paths'] = paths
            if connected:
                item['require_connected_paths'] = paths
            cases.append(item)

        if pid == 'discounted_returns':
            add('两步带bootstrap', tensor([[1],[2]]), boolean([[False],[False]]), tensor([4]), .5)
            add('终止断开后续episode', tensor([[1],[2],[10]]), boolean([[False],[True],[False]]), tensor([6]), .5)
            add('单时间步并行环境', tensor([[1,-2,3]]), boolean([[True,False,False]]), tensor([99,4,-2]), .9)
            add('gamma零只看当步', rand(4,2), torch.zeros(4,2,dtype=torch.bool), rand(2), 0.)
            add('gamma一累计', rand(4,3), boolean([[False,False,True],[True,False,False],[False,True,False],[False,False,False]]), rand(3), 1.)
            add('所有步都终止', rand(3,2), torch.ones(3,2,dtype=torch.bool), rand(2), .95)
            add('非连续奖励', rand(3,5).T, torch.zeros(5,3,dtype=torch.bool), rand(6)[::2], .7)
            add('最后一步终止忽略bootstrap', rand(5,2), boolean([[False,False],[False,False],[False,False],[False,False],[True,True]]), tensor([100,-100]), .99)
        elif pid == 'td0_target':
            add('混合终止手算', tensor([1,2]), tensor([10,20]), boolean([False,True]), .5)
            add('多维target', rand(2,3), rand(2,3), boolean([[True,False,True],[False,False,True]]), .9)
            add('标量非终止', tensor(2), tensor(-3), boolean(False), .8)
            add('全部终止', rand(4), rand(4,scale=100), torch.ones(4,dtype=torch.bool), .99)
            add('gamma零', rand(3,2), rand(3,2), torch.zeros(3,2,dtype=torch.bool), 0.)
            add('gamma一和负价值', tensor([-2,3]), tensor([-4,-1]), boolean([False,False]), 1.)
            add('非连续输入', rand(5,3).T, rand(5,3).T, torch.zeros(3,5,dtype=torch.bool), .7)
            add('零奖励保留bootstrap', torch.zeros(3,dtype=dtype), tensor([1,-2,0]), boolean([False,False,True]), .9)
        elif pid == 'gae_advantages':
            add('两步trace手算', tensor([[1],[2]]), tensor([[0],[0]]), tensor([[0],[0]]),
                boolean([[False],[False]]), boolean([[False],[False]]), 1., .5)
            add('截断仍bootstrap但不串episode', tensor([[1],[20]]), tensor([[2],[3]]), tensor([[4],[8]]),
                boolean([[False],[False]]), boolean([[True],[False]]), .5, .9)
            add('真正终止禁bootstrap', tensor([[1],[20]]), tensor([[2],[3]]), tensor([[100],[8]]),
                boolean([[True],[False]]), boolean([[False],[False]]), .5, .9)
            add('lambda零等于TD残差', rand(3,2), rand(3,2), rand(3,2), torch.zeros(3,2,dtype=torch.bool), torch.zeros(3,2,dtype=torch.bool), .9, 0.)
            add('lambda一与窗口尾bootstrap', rand(4,2), rand(4,2), rand(4,2), torch.zeros(4,2,dtype=torch.bool), torch.zeros(4,2,dtype=torch.bool), .95, 1.)
            add('并行混合边界含双True', rand(4,3), rand(4,3), rand(4,3),
                boolean([[False,False,True],[True,False,False],[False,False,False],[True,False,False]]),
                boolean([[False,True,True],[False,False,False],[False,False,True],[False,True,False]]), .8, .7)
            add('gamma零与单步', rand(1,3), rand(1,3), rand(1,3), boolean([[False,True,False]]), boolean([[True,False,False]]), 0., .95)
            add('非连续时间张量', rand(2,5).T, rand(2,5).T, rand(2,5).T,
                torch.zeros(5,2,dtype=torch.bool), boolean([[False,False],[False,True],[False,False],[True,False],[False,False]]), .99, .93)
        elif pid == 'normalize_advantages':
            for name,a,eps in [
                ('两元素显式eps',tensor([1,3]),1.), ('全部维度统一归一化',tensor([[1,2],[10,20]]),1e-8),
                ('单元素标量',tensor(7),1e-6), ('常量二维',torch.full((3,4),5.,dtype=dtype),1e-8),
                ('非连续输入',rand(5,3).T,1e-5), ('小方差epsilon主导',tensor([1.,1.0001,1.0002]),.1),
                ('全负优势',tensor([-7,-2,-5,-1]),1e-6), ('零均值三维',rand(2,3,4),1e-3)]:
                add(name,a,eps)
        elif pid == 'reinforce_loss':
            add('均匀策略正回报',tensor([[0,0]]),integer([1]),tensor([2]),grad=True)
            add('负回报反转梯度',tensor([[0,0]]),integer([0]),tensor([-2]),grad=True)
            add('不同动作与回报',rand(4,3),integer([0,2,1,2]),tensor([1,-2,0,3]),grad=True)
            add('单动作零logprob',rand(3,1),integer([0,0,0]),tensor([1,-2,4]),grad=True)
            add('极端logits',tensor([[1000,-1000],[-1000,1000],[1000,1000]]),integer([1,0,1]),tensor([1,-1,2]),grad=True)
            add('非连续策略logits',rand(5,3).T,integer([0,4,2]),tensor([2,3,-1]),grad=True)
            add('零回报保留图',rand(4,2),integer([0,1,1,0]),torch.zeros(4,dtype=dtype),grad=True,connected=True)
            add('重复动作但逐样本平均',rand(5,4),integer([2,2,2,2,2]),tensor([1,2,3,4,5]),grad=True)
        elif pid == 'categorical_entropy':
            for name,z in [('均匀两动作',torch.zeros(2,2,dtype=dtype)),('非均匀分布',tensor([[0,1,-2],[2,0,1]])),
                           ('单动作',rand(3,1)),('极端logits',tensor([[1000,-1000],[-1000,1000]])),
                           ('大共同偏移',tensor([[1000,1000,1000],[-1000,-1001,-999]])),
                           ('非连续logits',rand(5,3).T),('单batch多动作',rand(1,7)),('负logits小差异',rand(4,3,scale=.01)-10)]:
                add(name,z,grad=True)
        elif pid == 'importance_sampling_ratio':
            pairs = [('概率比手算',tensor([-.916290731874155]),tensor([-1.6094379124341003])),
                     ('共同极小概率',tensor([-1000.,-1001.]),tensor([-1000.,-1000.])),
                     ('标量输入',tensor(-2.),tensor(-3.)),('二维两方向',tensor([[-1.,-3.],[-5.,-2.]]),tensor([[-2.,-2.],[-4.,-4.]])),
                     ('极端允许log比',tensor([-10.,-70.]),tensor([-40.,-40.])),
                     ('非连续输入',-rand(5,3).abs().T-3,-rand(5,3).abs().T-3),
                     ('新旧相等',tensor([-1.,-2.,-8.]),tensor([-1.,-2.,-8.])),
                     ('三维比率',-rand(2,3,2).abs()-2,-rand(2,3,2).abs()-2)]
            for name,new,old in pairs:add(name,new,old,grad=True)
        elif pid == 'ppo_clipped_loss':
            configurations = [
                ('正优势裁上限',[1.5],[2.],.2),('负优势裁下限',[.5],[-2.],.2),
                ('两符号不利方向不裁掉',[.5,1.5],[2.,-2.],.2),
                ('区间内非单位比率',[.9,1.1,1.],[1.,-2.,3.],.2),
                ('零优势梯度',[.5,1.,1.8],[0.,0.,0.],.1),
                ('不同裁剪宽度',[.6,.9,1.1,1.4],[2.,-1.,-3.,1.],.35),
                ('非连续一维',[.7,1.,1.6,.95],[1.,3.,-2.,-1.],.15),
                ('精确ratio一与平衡优势',[1.,1.,1.,1.],[1.,-1.,2.,-2.],.2)]
            for i,(name,r,a,eps) in enumerate(configurations):
                old=torch.full((len(r),),-3.,dtype=dtype);new=old+tensor(r).log();adv=tensor(a)
                if i==6:
                    new=torch.stack((new,new),-1)[:,0];old=torch.stack((old,old),-1)[:,0]
                add(name,new,old,adv,eps,grad=True,connected=i==4)
        elif pid == 'ppo_value_loss':
            data = [('基础平方误差',[0.],[0.],[2.],.2),('有利变化被clip',[1.],[0.],[2.],.2),
                    ('不利变化保留原误差',[2.,-2.],[0.,0.],[0.,0.],.2),
                    ('旧value非零中心',[3.,-1.],[2.5,-.8],[5.,-3.],.3),
                    ('区间内新值',[.1,-.1],[0.,0.],[1.,-1.],.2),
                    ('完全命中return',[1.,-2.],[1.,-2.],[1.,-2.],.1),
                    ('非连续输入',[3.,1.,-4.],[1.,1.,-2.],[0.,2.,-1.],.5),
                    ('不同尺度epsilon',[2.,-3.,1.],[0.,0.,0.],[4.,-1.,2.],2.)]
            for i,(name,new,old,target,eps) in enumerate(data):
                z=tensor(new)
                if i==6:z=torch.stack((z,z),-1)[:,0]
                add(name,z,tensor(old),tensor(target),eps,grad=True)
        elif pid == 'dqn_loss':
            add('选中动作不是最大Q',tensor([[0,5]]),integer([0]),tensor([2]),tensor([99]),boolean([True]),.9,grad=True)
            add('Huber平方区间',tensor([[.5]]),integer([0]),tensor([0]),tensor([0]),boolean([False]),.99,grad=True)
            add('混合终止与误差大小',rand(4,3),integer([0,2,1,2]),tensor([1,-2,0,3]),tensor([2,10,-3,4]),boolean([False,True,False,True]),.5,grad=True)
            add('Huber边界正负一',tensor([[1],[-1],[0]]),integer([0,0,0]),tensor([0,0,0]),tensor([0,0,0]),boolean([False,False,False]),1.,grad=True)
            add('gamma零忽略未来',rand(3,4),integer([3,1,2]),rand(3),rand(3,scale=20),boolean([False,False,False]),0.,grad=True)
            add('非连续Q矩阵',rand(5,3).T,integer([1,4,0]),rand(3),rand(3),boolean([False,True,False]),.9,grad=True)
            add('大TD误差线性区',tensor([[1000,-1000],[-1000,1000]]),integer([0,0]),tensor([0,0]),tensor([0,0]),boolean([True,True]),.9,grad=True)
            add('零TD误差图连接',torch.zeros(2,3,dtype=dtype),integer([1,2]),torch.zeros(2,dtype=dtype),torch.zeros(2,dtype=dtype),boolean([False,True]),.9,grad=True,connected=True)
        elif pid == 'double_dqn_target':
            add('online与target最优不同',tensor([1]),tensor([[5,1]]),tensor([[2,10]]),boolean([False]),.5)
            add('并列选最小索引',tensor([0,1]),tensor([[3,3],[4,4]]),tensor([[2,10],[5,-8]]),boolean([False,False]),1.)
            add('混合终止',rand(4),rand(4,3),rand(4,3),boolean([False,True,True,False]),.9)
            add('单动作',rand(3),rand(3,1),rand(3,1),boolean([False,False,True]),.7)
            add('gamma零',rand(3),rand(3,4),rand(3,4),boolean([False,False,False]),0.)
            add('非连续网络输出',rand(3),rand(5,3).T,rand(5,3).T,boolean([False,True,False]),.95)
            add('全部终止',rand(4),rand(4,2),rand(4,2,scale=100),torch.ones(4,dtype=torch.bool),1.)
            add('所有Q为负',tensor([-1,-2]),tensor([[-5,-2,-3],[-1,-1,-4]]),tensor([[-10,-8,-2],[-3,-9,-1]]),boolean([False,False]),.8)
        elif pid == 'polyak_update':
            for name,online,target,tau in [
                ('tau方向手算',tensor([10]),tensor([2]),.25),('tau零保持旧target',rand(2,3),rand(2,3),0.),
                ('tau一复制online',rand(3,2),rand(3,2),1.),('标量参数',tensor(3),tensor(-2),.1),
                ('非连续权重',rand(5,3).T,rand(5,3).T,.005),('正负混合',tensor([-2,4,0]),tensor([3,-1,5]),.8),
                ('online等于target',torch.ones(2,3,dtype=dtype),torch.ones(2,3,dtype=dtype),.4),('三维参数',rand(2,3,2),rand(2,3,2),.5)]:
                add(name,online,target,tau)
        elif pid == 'value_iteration_step':
            add('单状态手算',tensor([4]),tensor([[[1]]]),tensor([[[2]]]),boolean([[[False]]]),.5)
            add('先期望再选动作',tensor([0,10]),tensor([[[.5,.5],[1,0]],[[0,1],[.25,.75]]]),
                tensor([[[0,0],[6,6]],[[1,1],[0,0]]]),torch.zeros(2,2,2,dtype=torch.bool),1.)
            for i,(s,a) in enumerate([(3,2),(2,3),(4,1),(3,3),(2,2),(3,2)]):
                probabilities=torch.softmax(rand(s,a,s),dim=-1)
                values,reward=rand(s),rand(s,a,s)
                terminal=torch.zeros(s,a,s,dtype=torch.bool)
                if i==0:terminal[0,:,1]=True;terminal[-1,:,:]=True
                if i==1:terminal.fill_(True)
                if i==4:
                    probabilities=torch.softmax(rand(s,s,a),dim=1).transpose(1,2)
                    reward=rand(s,s,a).transpose(1,2);values=rand(s*2)[::2]
                add(['混合终止transition','全终止只看奖励','单动作期望','gamma零',
                     '非连续转移和reward','负价值多动作'][i],values,probabilities,reward,terminal,0. if i==3 else .9)
        elif pid == 'c51_projection':
            support=tensor([-1,0,1])
            add('恰落atom保持质量',tensor([[.2,.3,.5]]),tensor([0]),boolean([False]),support,1.)
            add('terminal落在两atom之间',tensor([[.1,.7,.2]]),tensor([.5]),boolean([True]),support,.99)
            add('上下界截断',tensor([[.2,.3,.5],[.4,.4,.2]]),tensor([10,-10]),boolean([False,True]),support,.9)
            add('gamma零退化点质量插值',tensor([[.2,.3,.5],[.3,.5,.2]]),tensor([-.25,.75]),boolean([False,False]),support,0.)
            add('两个atom最小支撑',tensor([[.4,.6],[.8,.2]]),tensor([.5,1.]),boolean([False,True]),tensor([0,2]),.5)
            add('随机七atom概率',torch.softmax(rand(3,7),-1),tensor([.3,-.8,1.2]),boolean([False,True,False]),torch.linspace(-3,3,7,dtype=dtype),.9)
            add('非连续分布与非零支撑中心',torch.softmax(rand(6,3),0).T,rand(3),boolean([False,True,False]),torch.linspace(-3,7,6,dtype=dtype),.7)
            add('点质量与压缩后恰落atom',tensor([[1,0,0,0,0],[0,0,1,0,0],[0,0,0,0,1]]),tensor([0,0,0]),
                boolean([False,False,False]),tensor([-2,-1,0,1,2]),.5)
        elif pid == 'grpo_advantages':
            for name,reward,eps in [
                ('同prompt组内手算',tensor([[1,3],[10,10]]),1.),('不同prompt均值尺度',tensor([[0,2,4],[100,101,102]]),1e-6),
                ('每组一条回答',tensor([[2],[7]]),1e-6),('常量组',torch.ones(3,4,dtype=dtype),1e-8),
                ('非连续奖励',rand(5,3).T,1e-4),('微小奖励差异与eps',tensor([[1,1.0001,1.0002],[2,2.0001,2.0002]]),.1),
                ('负奖励组',tensor([[-7,-2,-5],[-1,-8,-3]]),1e-6),('单prompt多回答',rand(1,7),1e-3)]:
                add(name,reward,eps)
        elif pid == 'grpo_loss':
            new=torch.full((2,3),-2.,dtype=dtype)
            add('新旧参考一致',new,new.clone(),new.clone(),tensor([1,-1]),torch.ones(2,3,dtype=torch.bool),.2,.04,grad=True,connected=True)
            add('先序列平均再batch平均',new,new.clone(),new.clone(),tensor([2,-1]),boolean([[True,False,False],[True,True,True]]),.2,0.,grad=True,connected=True)
            add('全部padding保留零梯度连接',-rand(2,3).abs()-2,-rand(2,3).abs()-2,-rand(2,3).abs()-2,tensor([1,-2]),torch.zeros(2,3,dtype=torch.bool),.2,.04,grad=True,connected=True)
            add('空序列不进入外层分母',tensor([[-2,-3,-4],[-3,-2,-1]]),tensor([[-2,-2,-4],[-3,-3,-2]]),
                tensor([[-3,-2,-4],[-3,-2,-1]]),tensor([8,-1]),boolean([[False,False,False],[True,False,True]]),.2,.2,grad=True,connected=True)
            old=torch.full((2,3),-3.,dtype=dtype)
            add('正负优势裁剪方向',old+tensor([[1.5,.5,1.],[.5,1.5,1.]]).log(),old,old-.2,tensor([2,-2]),torch.ones(2,3,dtype=torch.bool),.2,.03,grad=True,connected=True)
            add('beta零不混入KL',tensor([[-3,-2],[-2,-3]]),tensor([[-2,-3],[-3,-2]]),tensor([[-8,-7],[-9,-6]]),tensor([1,-3]),boolean([[True,True],[True,False]]),.1,0.,grad=True,connected=True)
            add('非连续log概率与间隔mask',-rand(4,3).abs().T-2,-rand(4,3).abs().T-2,-rand(4,3).abs().T-2,rand(3),
                boolean([[True,False,True,False],[True,True,True,True],[False,True,False,False]]),.15,.1,grad=True,connected=True)
            add('KL方向大比率与近零差值',tensor([[-25.,-3.],[-3.,-3.]]),tensor([[-25.,-3.],[-3.,-3.]]),
                tensor([[-5.,-2.99999],[-3.00001,-3.]]),tensor([0.,0.]),torch.ones(2,2,dtype=torch.bool),.2,.04,grad=True,connected=True)
    return cases


def mutations(bank):
    specifications = {
        'rl_returns_ignores_terminal': ('discounted_returns', '(~terminated[t]).to(rewards.dtype)', 'torch.ones_like(rewards[t])'),
        'rl_td0_bootstrap_terminal': ('td0_target', ' * (~terminated).to(rewards.dtype)', ''),
        'rl_gae_truncation_kills_bootstrap': ('gae_advantages', 'next_values * (~terminated).to(values.dtype)', 'next_values * (~(terminated | truncated)).to(values.dtype)'),
        'rl_normalize_sample_variance': ('normalize_advantages', 'centered.square().mean().sqrt()', 'advantages.std(unbiased=True)'),
        'rl_reinforce_missing_negative': ('reinforce_loss', 'return -(chosen * returns).mean()', 'return (chosen * returns).mean()'),
        'rl_entropy_wrong_sign': ('categorical_entropy', 'return -(log_probs.exp() * log_probs).sum(-1)', 'return (log_probs.exp() * log_probs).sum(-1)'),
        'rl_ratio_reversed': ('importance_sampling_ratio', '(new_log_probs - old_log_probs).exp()', '(old_log_probs - new_log_probs).exp()'),
        'rl_ppo_actor_optimistic_max': ('ppo_clipped_loss', 'torch.minimum(', 'torch.maximum('),
        'rl_ppo_value_min_error': ('ppo_value_loss', 'torch.maximum(', 'torch.minimum('),
        'rl_dqn_max_instead_of_action': ('dqn_loss', 'q_values.gather(1, actions[:, None]).squeeze(-1)', 'q_values.max(dim=1).values'),
        'rl_ddqn_target_selects_itself': ('double_dqn_target', 'online_next_q.argmax(dim=-1)', 'target_next_q.argmax(dim=-1)'),
        'rl_polyak_tau_reversed': ('polyak_update', '(1 - tau) * target + tau * online', 'tau * target + (1 - tau) * online'),
        'rl_value_iteration_max_next_state': ('value_iteration_step', '(transition * target).sum(dim=-1)', '(transition * target).max(dim=-1).values'),
        'rl_c51_exact_atom_loses_mass': ('c51_projection', ' + (lower == upper).to(position.dtype)', ''),
        'rl_grpo_advantages_mix_prompts': ('grpo_advantages', 'rewards.mean(dim=1, keepdim=True)', 'rewards.mean()'),
        'rl_grpo_loss_global_token_mean': ('grpo_loss', 'return per_sequence.sum() / (lengths > 0).sum().clamp_min(1)', 'return (token_loss * valid).sum() / valid.sum().clamp_min(1)'),
    }
    result = {}
    for name,(pid,before,after) in specifications.items():
        code = bank[pid]['solution']
        if before not in code:
            raise AssertionError('Missing mutation anchor: '+name)
        result[name] = (pid, code.replace(before, after))
    # Separate KL direction and all-padding graph regressions protect GRPO's distinctive contract.
    pid = 'grpo_loss'
    code = bank[pid]['solution']
    result['rl_grpo_kl_direction'] = (pid, code.replace('difference = reference_log_probs - new_log_probs',
                                                        'difference = new_log_probs - reference_log_probs'))
    result['rl_grpo_padding_detached_zero'] = (pid, code.replace('    ratio = (new_log_probs - old_log_probs).exp()',
        '    if not mask.any():\n        return torch.zeros((), dtype=new_log_probs.dtype, device=new_log_probs.device, requires_grad=True)\n'
        '    ratio = (new_log_probs - old_log_probs).exp()'))
    return result
