# TensorDojo · 本地双击启动版

当前共 **90 道 PyTorch 手写题**：大模型 42 道、推荐算法 32 道、强化学习 16 道。覆盖逻辑回归、MSE / CE、召回排序、LoRA、MoE、生成采样、DPO、GAE、PPO 与 GRPO 等模块。

本轮追加 24 道大模型题和 16 道强化学习题，沿用 P0 / P1 / P2 练习优先级；页面支持三个方向的筛选与学习路径，每题包含题面、提示、参考实现和真实判题用例。完整题单见 [题库 README](llm_code_lab/README.md)。

**本 ZIP 是源码＋启动器，不含 Python 安装包、torch wheel 或已经装好的 Python 环境，也不是 .app / .exe 成品。**

## 最简单的使用方式

先将整个 ZIP 解压到一个可写文件夹，不要在压缩包内直接运行，也不要只复制 HTML。

电脑需要已安装 64 位 Python；自动安装分支建议 Python 3.11。已有能运行本题库的 PyTorch 环境也可以复用。

| 系统 | 日常入口 |
| --- | --- |
| macOS | 双击 `Start.command` |
| Windows | 双击 `Start.bat` |
| Linux | 终端运行 `bash Start.command`；文件管理器的双击行为视发行版设置而定 |

入口依次执行：检测已有环境 → 缺少 torch 时创建项目环境并安装 → 首次题库自检 → 启动本地判题服务 → 自动调用浏览器打开界面。

首次缺依赖时需要联网。之后，保留已经安装好的环境和项目，仍然双击同一个入口，不必反复输入命令或重新安装。

正常情况下浏览器会打开 `http://127.0.0.1:8765`。系统没有可用的默认浏览器时，手动复制终端显示的地址打开。请不要在 `file://` 页面提交代码。

练习期间保留启动终端；在该终端按 Ctrl+C 停止服务。关闭浏览器标签不代表退出判题服务。本版没有后台常驻、开机启动或自动保活功能。

## 为什么不是双击 HTML 本身？

裸 HTML 在浏览器中负责界面。它不能自行安装或启动本机 Python/PyTorch；已经把 torch 下载到磁盘，也不会自动启动一个 Python 判题进程。

本版采用相反的启动顺序：**先由本机启动器开启 Python，再由 Python 打开 HTML 页面**。界面仍然是原来的网页，但不需要用户手动输入启动服务的命令。

浏览器与本机程序交互需要专门桥接，例如本工具的本地服务，或另行安装的扩展/原生程序。不会通过关闭浏览器安全设置来绕过这个边界。

## Torch 的安装策略

优先复用项目 `.tensor-dojo-venv` 中的 torch，其次检查运行启动器的 Python 环境。会执行一个 ReLU 梯度探测，并要求存在 SDPA 接口；环境或题库变化时，再执行完整题库自检。

可用的现有环境不被升级、降级或写入新的软件包。没有可用环境时，创建项目目录内的 `.tensor-dojo-venv`，仅向该环境安装。

| 自动安装分支 | 本包固定方案 | Python 自动安装分支 |
| --- | --- | --- |
| Windows x86-64 / Linux x86-64 | torch 2.10.0，官方 CPU wheel 索引 | 3.10–3.13 |
| macOS Apple Silicon，原生 arm64 Python | torch 2.10.0，PyPI | 3.10–3.13 |
| macOS Intel，x86-64 Python | torch 2.2.2 + NumPy 1.26.4，PyPI | 3.10–3.12 |

2.10.0 是此项目已经回归测试过的基线，不宣称是最新版本。Intel Mac 分支是旧环境兼容预设，并非推荐用于新的生产服务；不得加载来源不明的模型文件或代码。

**兼容安装预设不等于已在相应系统实测。** 所有平台还受操作系统版本、Python 架构、wheel 的系统要求、可用运行库和下载网络限制影响。原启动器曾在 Linux / Python 3.13.5 / torch 2.10.0+cpu 验证；当前扩展题库在 macOS Intel / Python 3.11.15 / torch 2.2.2 验证，见 [大模型与强化学习测试报告](TEST_REPORT_大模型与强化学习.md)。

本题库只做小规模 CPU 正确性判题，不需要 CUDA、GPU、torchvision、torchaudio 或大模型权重。本版不会下载这些额外包。

新安装的依赖使用固定 torch 版本，但没有锁定所有传递依赖；不能称为跨机器完全可复现的软件环境。

## 先下载 Torch，再离线使用

在有网络的目标电脑上，双击：

- macOS：`Download_Torch.command`
- Windows：`Download_Torch.bat`

这会运行 `pip download`，下载当前系统、Python 版本和架构对应的 torch wheel 及必要依赖，存入 `torch_wheels/<平台与 Python 标记>/`，并生成文件校验记录。此步骤只下载，不会安装 Python，也不会立即创建判题服务。

之后正常双击 `Start`。缺少可用 torch 时，启动器检测到匹配缓存，会优先从缓存安装到项目环境，不访问软件包索引。缓存缺失或损坏会报告错误，不会静默忽略损坏文件。

严格离线启动可使用：

```bash
python launch.py --offline
```

`--offline` 只限制本启动器的依赖安装来源；它不是用户代码的网络沙箱。已安装的环境可以直接复用，即使没有 wheel 缓存。

离线 wheel 不是跨平台通用包。本功能以“同一台电脑、同一 Python 小版本”的预下载为目标；不能把 Linux 的缓存当作 Windows 或 Mac 的安装包，也不能直接复制一个 venv 当作通用便携环境。

文件 SHA-256 记录用来发现损坏，并非签名或独立的软件供应链信任验证。

## 常见问题

### 没有安装 Python

先从 Python 官方网站安装 64 位 Python，建议 3.11。本启动器不会自动安装 Python，也不请求管理员权限。

### macOS 拦截或没有执行权限

先检查文件来源与脚本内容，再按 macOS 对可信下载程序的提示处理；不要关闭整个系统的安全检查。

如果解压工具丢失执行权限，可在项目目录执行一次：

```bash
chmod +x Start.command Download_Torch.command
```

也可在 Terminal 中运行 `bash Start.command`。首次授权/权限处理是可能需要的，并非保证所有 Mac 下载后都无需任何额外操作。

### 想指定已有 Conda / Python 环境

在该环境中运行：

```bash
python launch.py
```

高级使用也可将 `TENSORDOJO_PYTHON` 环境变量设为目标 Python 可执行文件的完整路径。启动器只检查选中的解释器和项目环境，不扫描整台电脑寻找全部 Conda 环境。

### 已有项目正在运行

同一项目、同一端口且服务就绪时，再次启动会打开已有服务，不重复启动 Python 后端。

**更新题库后需要重启旧服务。** 在原启动终端按 `Ctrl+C`，再双击 `Start.command`（或运行 `bash Start.command`），刷新原来的 `http://127.0.0.1:8765` 页面，左侧应显示“90 题”。在同一浏览器、同一地址下，原有题目的草稿和进度继续保留。

如果端口被其他项目或原版服务占用，会明确报错，不自动换端口。可关闭旧服务后重新启动；高级用法为 `python launch.py --port 8766`。浏览器进度按来源隔离，换端口需使用页面的导出/导入迁移。

### 安装中断或环境不完整

先重试。启动器不会自动删除已有环境目录；必要时仅移走 `.tensor-dojo-venv` 后重新准备。不要删除整个项目或你的进度备份。

### 不希望看到终端，连 Python 都不想装

这需要针对目标系统构建完整桌面分发包，打包 Python、torch 原生库、判题入口及 HTML，例如 .app / .exe。**本 ZIP 不是这种全内置成品。**

打包时还要正确处理判题子进程：冻结程序中的 `sys.executable` 不再是一个普通 Python 解释器，不能不改代码就继续用它执行 `judge.py`。本包没有声称完成这项改造。

## 目录

```text
TensorDojo_OneClick/
├── Start.command / Start.bat              日常入口
├── Download_Torch.command / .bat         可选：仅下载依赖
├── launch.py                             环境检查、安装、启动
├── README_双击启动.md
├── TEST_REPORT_本轮.md
├── TEST_REPORT_推荐题库.md
├── TEST_REPORT_大模型与强化学习.md
├── tests/test_launcher.py
├── tests/test_recommendations.py
├── tests/test_advanced_tracks.py
└── llm_code_lab/                          90 题前端、题库、判题核心
```

运行后可能新增 `.tensor-dojo-venv/`、`.validation.json`、`torch_wheels/`；这些均没有预装在本次 ZIP 里。

## 安全与数据

仅运行可信的个人练习代码，不是恶意代码安全沙箱。判题代码拥有当前用户的文件和网络权限；子进程与超时不能代替权限隔离。本地服务只监听 127.0.0.1，保留原 Host、Origin 和随机 token 检查。

草稿和进度仍在浏览器 localStorage，定期从页面导出备份。不要假设关闭终端、复制项目目录或保存 HTML 就保存了浏览器中的进度。

## 参考资料

- PyTorch 安装及历史版本：https://pytorch.org/get-started/locally/ ，https://pytorch.org/get-started/previous-versions/
- PyTorch 2.2.2 的官方发布 wheel：https://pypi.org/project/torch/2.2.2/
- Python venv：https://docs.python.org/3/library/venv.html
- pip download：https://pip.pypa.io/en/stable/cli/pip_download/
- 浏览器与本机程序桥接：https://developer.mozilla.org/en-US/docs/Mozilla/Add-ons/WebExtensions/Native_messaging
- PyInstaller 打包与平台约束：https://pyinstaller.org/en/stable/operating-mode.html
- Mac 下载程序安全提示：https://support.apple.com/en-us/102445
