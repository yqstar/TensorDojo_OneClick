# TensorDojo

TensorDojo 是一个本地运行的 PyTorch 手写题训练场，面向大模型、推荐算法和强化学习面试练习。当前包含 **90 道题**：大模型 42 道、推荐算法 32 道、强化学习 16 道。

每道题包含中文题面、公式、输入输出约定、提示、易错点、参考实现和本地测试。提交的代码由本机 Python / PyTorch 执行，并检查返回契约、数值、输入不变性，以及适用题目的自动求导结果。

## 快速开始

需要 64 位 Python 3.10 或更高版本，推荐 Python 3.11。首次安装 PyTorch 时需要联网。

- macOS：双击 `Start.command`
- Windows：双击 `Start.bat`
- Linux：运行 `bash Start.command`

服务默认打开 `http://127.0.0.1:8765`。停止服务时，在启动终端按 `Ctrl+C`。

详细的环境兼容、离线缓存和故障处理说明见 [双击启动说明](README_双击启动.md)。题目清单、判题口径和扩展接口见 [题库说明](llm_code_lab/README.md)。

## 题目方向

- 大模型：Transformer 基础、LoRA、位置编码、Attention、KV Cache、MoE、生成采样、Causal LM Loss、蒸馏、DPO 和 AdamW。
- 推荐算法：逻辑回归、MSE / BCE / CE、优化器、AUC / GAUC、召回、排序、DIN、SASRec、MMoE、ESMM 和 PLE。
- 强化学习：折扣回报、TD、GAE、REINFORCE、PPO、DQN、Double DQN、C51 和 GRPO。

## 验证

```bash
python launch.py --check
```

当前题库的 90 道参考实现共通过 1,252 组测试，并能拒绝 94 种针对核心考点设计的错误实现。详细结果见 [大模型与强化学习题库扩充验收](TEST_REPORT_大模型与强化学习.md)。

## 项目边界

这是可信个人代码的本地练习器，不是恶意代码安全沙箱。判题子进程仍拥有当前用户的文件和网络权限，请勿运行来源不明的代码，也不要把当前判题服务直接暴露到公网。

浏览器中的草稿、笔记和进度保存在 localStorage。更新项目后，只要继续使用相同浏览器和相同地址，原有题目的记录会按稳定 ID 继续读取。

## License

见 [LICENSE](llm_code_lab/LICENSE)。
