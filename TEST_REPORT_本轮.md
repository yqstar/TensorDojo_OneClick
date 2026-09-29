# 本轮验证记录

日期：2026-09-29。目标：在原 TensorDojo 本地版本上增加依赖准备与双击启动入口，而非改造 Colab 版。

## 实际运行环境

Linux x86-64，Python 3.13.5，torch 2.10.0+cpu。

## 实际执行通过

1. `python launch.py --check`：复用已有 PyTorch，计算探测成功，执行完整题库自检。
2. 原题库参考实现：18 题、206 组算例全部通过；9 类故意错误实现全部检出。
3. `python -m unittest discover -s tests -v`：18 项测试全部通过。
4. `bash Start.command --offline --check`：通过 shell 入口复用环境、自检，不安装依赖。
5. Python 文件编译检查与 Bash 脚本语法检查通过。

18 项测试覆盖：平台安装方案、拒绝不兼容 Python/架构、离线缓存缺失和损坏、已有环境不执行 pip、环境验证记录复用、本轮 launcher 真正启动本地服务、健康检查与项目识别、重复启动、浏览器打开参数、HTML 服务、真实 ReLU 提交 12/12、跨域请求拒绝、端口冲突。

其中平台方案、离线缓存和浏览器打开参数使用模拟测试；HTTP 服务和 ReLU 提交运行的是真实 Python / PyTorch。离线缓存测试使用伪造的测试 fixture 验证文件校验逻辑，不是安装真实 torch wheel。

## 尚未实际验证

- Windows 原生 `.bat` 双击和安装。
- macOS Finder 双击、Gatekeeper 权限、Intel/Apple Silicon 的 torch 导入与安装。
- 从无 torch 的干净环境联网下载和安装全部依赖。
- 从真实预下载 torch wheel 完成首次离线安装。
- 原生桌面默认浏览器的启动及 GUI 交互；本轮浏览器打开测试使用 mock 验证参数。
- Python 3.10 / 3.11 / 3.12 以及 torch 2.2.2 的整套运行。

ZIP 未包含 Python、torch 二进制、wheel 缓存或预装虚拟环境，不应称为完整离线桌面安装包。
