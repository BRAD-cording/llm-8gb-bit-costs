# 实验环境事实（可直接用作论文的 "Execution Stack" 附录）

> 这份文件的存在本身就是一个贡献点：多篇批评指出量化论文普遍**不报告执行栈**，导致结果不可复现。
> 我们从第一周起就把栈写死并随文提交。

## 1. 主机 A：实验笔记本

| 项 | 值 | 获取方式 |
|---|---|---|
| CPU | i9-14700HX | 用户提供 |
| 内存 | 32 GB DDR5 | 用户提供 |
| GPU | **NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB** | `nvidia-smi` |
| 驱动 | **592.82** | `nvidia-smi --query-gpu=driver_version` |
| 计算能力 | **sm_89 (Ada)** | `torch.cuda.get_device_properties(0)` |
| SM 数 | **24** | 同上 |
| CUDA | torch 自带 **cu124** | `torch.version.cuda` |
| PyTorch | **2.6.0+cu124** | 已装 |
| Python | **3.13.12** | 已装 |
| 磁盘 | C: 448 GB 可用 | `DriveInfo` |

**实测（见 `results/probe_4060.json`）**
- copy 带宽 **161.9 GB/s**；GEMV 有效带宽 **179.8 GB/s**
- FP16 matmul 有效算力：bs=1 → 0.21 TFLOP/s；bs=128 → **22.36 TFLOP/s**
- 分配器额外开销 $\rho_{alloc}$ ≈ 0（单次大分配；真实 $\rho$ 需加载模型后测）

## 2. 主机 B：Jetson Orin Nano Super 8GB

*待 SSH 接入后填充：L4T 版本、JetPack 版本、CUDA 版本、最大/最小功耗模式（`nvpmodel`）、`tegrastats` 采样口径。*
**这些字段若不填，Jetson 侧的数字在审稿中不可信。**

## 3. 网络与包管理（中国网络环境下的可复现性事实）

| 端点 | 状态 | 说明 |
|---|---|---|
| `pypi.org` | **不可达**（TLS handshake reset） | 不是我们的配置问题，是链路层拦截 |
| `pypi.tuna.tsinghua.edu.cn` | 可达，0.7 s | **本项目的默认索引** |
| `mirrors.aliyun.com/pypi` | 可达，0.3 s | 备用 |
| `mirrors.ustc.edu.cn/pypi` | 可达，0.4 s | 备用 |
| `huggingface.co` | **DNS 解析失败** | 模型不能走 HF |
| `hf-mirror.com` | **不可达** | 也不可用 |
| `modelscope.cn` | 可达，0.2 s | **本项目的模型来源** |

**因此本项目的安装命令固定为：**
```powershell
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple `
  --trusted-host pypi.tuna.tsinghua.edu.cn <packages>
```
模型权重一律经 ModelScope 获取，**不走 HuggingFace**。

> 论文附录应写明这一点：如果审稿人按 README 去 HF 拉模型会失败。这不是缺陷，是必须披露的环境约束。

## 4. 已知工具链限制（要在 Limitations 里诚实写）

- **Windows + Python 3.13**：`vLLM`、`autoawq`、`auto-gptq`、部分 Triton 路径**不可用或极不稳定**。
  → 主线是 **PyTorch fake-quant 参考实现 + bnb 真实量化**；`llama.cpp` 未纳入（见下）。
- **bnb 的 int8 路径（LLM.int8）比 4-bit 还慢**：实测 4k 上下文 TTFT 520.9 ms，
  而 nf4 是 404.2 ms、fp16 是 355.9 ms。这是 bnb 的逐 matmul 离群值分解导致的，
  不是 8-bit 算术的性质。论文 §6.2 引用了这一点，用来说明"位宽不决定时延"。
- 无自研 CUDA kernel：本文的容量收益来自**分配**，不是 kernel 优化。

## 4a. 官方 gptqmodel 的安装约束（2026-10-21 实测）

| 环境 | 结论 |
|---|---|
| Windows + Python **3.13** | **可用**（前提是先修环境，见下）。最初报的 `pkgutil.ImpImporter` **不是 Python 版本问题**，而是该环境的 `pkg_resources` 老旧：`setuptools` 自带的 `pkg_resources/__init__.py:2172` 在 import 时就引用 `pkgutil.ImpImporter`。`pip install -U "setuptools>=75"` 之后 `python setup.py --version` 直接成功 |
| Windows + Python **3.11** | 也可行（`pkgutil.ImpImporter` 仍在），本机已装 3.11.9，但最终没用上 |
| WSL2 Ubuntu 22.04 | **本机不可用**，见 4c |
| Jetson (Ubuntu 22.04, Py3.10) | **可用**，gptqmodel **4.2.5**（7.6 需要 transformers 5.x，而 5.x 要 torch ≥2.6，本机 torch 是 2.5.0a0） |

Windows 上还需要（都已落地为仓库里的脚本）：

- `--no-build-isolation` 从 sdist 安装；
- **`tools/gptq_compat.py`**：把 `no_init_weights` 从 `transformers.initialization`
  重新导出到 `transformers.modeling_utils`（5.x 里移位了），并关闭 `torch._dynamo`
  ——Windows 没有 Triton，gptqmodel 的量化循环调用 `torch.compile` 会直接抛错；
- `backend=BACKEND.TORCH`（否则 AUTO 会选到需要 Triton 的内核）。

Jetson 上则需要：`--no-build-isolation`、`python-pcre`、
以及垫片 `artifacts/pcre_shim.py`（gptqmodel 用到 `pcre.Flag` 等，旧版 python-pcre 没有；
装到用户 site-packages，文件名为 pcre.py）、`defuser`、`random_word`、`device_smi`。
全部细节见 `results/GPTQ_OFFICIAL.md`。

## 4c. WSL2 在本机不可用（2026-10-21）

`wsl -d Ubuntu-22.04` 报 `Wsl/Service/CreateInstance/CreateVm/HCS/HCS_E_SERVICE_NOT_AVAILABLE`。
排查：

- WSL 版本 2.6.3.0、内核 6.6.87.2 都已安装，发行版 `Ubuntu-22.04` 存在但处于 Stopped。
- **`vmcompute`（Hyper-V Host Compute Service）不存在**：`sc query vmcompute` 返回 1060。
- 查询 Windows 功能状态需要管理员权限，本会话不是管理员，审批提示也已禁用。

**修复需要管理员**（然后在重启后生效）：

```powershell
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
# 重启，然后
wsl --update
```

因此"用 WSL2 装 gptqmodel"这条路**当前不可行**，不是配置问题，是虚拟化平台功能缺失。


## 4b. LaTeX 工具链（2026-10-20 补记）

官方 `basic-miktex-*.exe` 安装器在这个会话里**卡死**：它以 0% CPU 空转，
既不报错也不创建安装目录。原因是它需要 UAC 提权，而本会话无法应答批准对话框。

可用的路线是 **portable 安装**，全程无需管理员权限：

```powershell
# 1. 取安装器（清华 CTAN 镜像可达；官方站点与 GitHub 均被墙）
python tools/http_get.py `
  "https://mirrors.tuna.tsinghua.edu.cn/CTAN/systems/win32/miktex/setup/windows-x64/miktexsetup-5.5.0+1763023-x64.zip" `
  C:\texinstall\miktexsetup.zip
Expand-Archive C:\texinstall\miktexsetup.zip -DestinationPath C:\texinstall\setup

# 2. 先下载包集合，再安装（`install` 只认本地包目录，不认 --remote-package-repository）
& C:\texinstall\setup\miktexsetup_standalone.exe download `
  --remote-package-repository="https://mirrors.tuna.tsinghua.edu.cn/CTAN/systems/win32/miktex/tm/packages/" `
  --local-package-repository=C:\texpkgs --package-set=basic

& C:\texinstall\setup\miktexsetup_standalone.exe install `
  --portable=C:\texportable --local-package-repository=C:\texpkgs --package-set=basic
```

**坑**：`--portable` 与 `--shared` 互斥；`install` 不接受 `--remote-package-repository`；
选项名是 `--remote-package-repository` 而不是 `--repository`。

编译：
```powershell
cd paper
C:\texportable\texmfs\install\miktex\bin\x64\bibtex main
C:\texportable\texmfs\install\miktex\bin\x64\pdflatex main   # 跑两遍
```

字体与 IEEEtran 会在首次编译时自动从镜像安装（需 `initexmf --set-config-value="[MPM]AutoInstall=1"`）。

## 5. 复现清单（投稿时随 artifact 提交）

- [x] `docs/requirements-freeze.txt`（211 个包的完整快照）
- [ ] 本文件（机器可读 JSON 版）
- [ ] `results/probe_*.json`（两机）
- [ ] MiKTeX 版本与 CTAN 镜像 URL（MiKTeX 26.5 portable，清华镜像）
- [ ] Jetson 的 `nvpmodel -q` 与 `tegrastats` 原始日志

**关键版本**（完整列表见 `requirements-freeze.txt`）：

```
torch==2.6.0+cu124        transformers==5.20.2
bitsandbytes==0.50.2      datasets==5.1.0
accelerate==1.15.0        numpy==2.4.2
PyMuPDF==1.27.2.2         matplotlib==3.10.8
```

> 注意 `torch` 的后缀 `+cu124` 是必需的：CPU 版本会让
> `device_probe.py` 与全部时延测量失效。
- [ ] 所有随机种子
- [ ] 每个实验的 wall-clock 与当时的 GPU 温度/功耗（证明没有降频污染）
