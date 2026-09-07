# TTT conda 环境完整报告

采集日期：2026-09-07；报告生成时间：`2026-09-07T22:18:59+00:00`。目标环境：`/home/da839/.conda/envs/TTT`。

## 1. 核心结论

该环境以 **Python 3.11.15 + PyTorch 2.9.1+cu126（CUDA 12.6 构建）** 为核心，包含 Transformers、vLLM、DeepSpeed、FlashAttention、FlashInfer 和 Triton。读取到 **231 个 Python distribution** 和 **34 条 conda 包记录**；两种统计口径存在重叠，不能相加作为独立包总数。

PyTorch 成功导入，CPU 小矩阵乘法通过；cuDNN 返回 `91002`（9.10.2），NCCL 返回 `2.27.5`。当前登录节点无法初始化 NVIDIA 驱动/NVML，`torch.cuda.is_available()` 为 `False`、设备数为 `0`，因此本次没有验证 GPU 运算。

`pip check` 发现 3 处依赖声明冲突：torchaudio、torchvision、vLLM 都要求 `torch==2.9.0`，当前安装 `torch==2.9.1+cu126`。这证明安装版本不满足这些包的声明，尚不能据此判断每个功能是否可运行。本报告仅做盘点，没有调整环境。

## 2. 环境与采集范围

| 项目 | 实测值 |
| --- | --- |
| 环境名称 | TTT |
| 环境路径 | /home/da839/.conda/envs/TTT |
| Python 可执行文件 | /home/da839/.conda/envs/TTT/bin/python |
| Python 完整版本 | 3.11.15 (main, Mar 11 2026, 17:20:07) [GCC 14.3.0] |
| Conda 管理工具 | 24.1.2（来自 /home/da839/anaconda3） |
| 操作系统/架构 | Linux-5.14.0-570.119.1.el9_6.x86_64-x86_64-with-glibc2.34 |
| 采集节点 | login1.bouchet.ycrc.yale.edu |
| 采集方式 | 目标 Python 的 importlib.metadata、conda-meta/*.json、激活后的 shell、轻量 PyTorch 检查和 pip check |
| 计算资源 | 登录节点；没有 SLURM_JOB_ID，squeue --me 未返回作业；没有提交 GPU 作业 |

Python 包列表来自目标解释器可见的 distribution 元数据；conda 表来自本地安装记录。完整枚举不等于逐包运行验证。未导入测试 vLLM、DeepSpeed、FlashAttention、FlashInfer，也未编译扩展或下载模型。

## 3. 主要 Python 包

| 类别 | 包 | 安装版本 |
| --- | --- | --- |
| 深度学习核心 | torch | 2.9.1+cu126 |
| 深度学习核心 | torchvision | 0.24.0 |
| 深度学习核心 | torchaudio | 2.9.0 |
| 深度学习核心 | torchdata | 0.11.0 |
| 深度学习核心 | triton | 3.5.1 |
| 模型与训练 | transformers | 4.57.6 |
| 模型与训练 | accelerate | 1.14.0 |
| 模型与训练 | peft | 0.19.1 |
| 模型与训练 | deepspeed | 0.19.3 |
| 模型与训练 | datasets | 4.8.4 |
| 模型与训练 | huggingface-hub | 0.36.2 |
| 模型与训练 | tokenizers | 0.22.2 |
| 模型与训练 | safetensors | 0.8.0 |
| 推理与 attention | vllm | 0.12.0 |
| 推理与 attention | flash-attn | 2.8.3.post1 |
| 推理与 attention | flashinfer-python | 0.5.3 |
| 计算与分布式 | numpy | 2.2.6 |
| 计算与分布式 | scipy | 1.17.1 |
| 计算与分布式 | pandas | 3.0.2 |
| 计算与分布式 | scikit-learn | 未安装 |
| 计算与分布式 | cupy-cuda12x | 14.1.1 |
| 计算与分布式 | ray | 2.56.1 |
| 实验与工具 | wandb | 0.28.1 |
| 实验与工具 | tensorboard | 未安装 |
| 实验与工具 | matplotlib | 3.11.1 |
| 实验与工具 | pytest | 9.1.1 |
| 实验与工具 | ninja | 1.13.0 |
| 实验与工具 | mfspd | 0.1.0 |

## 4. CUDA、cuDNN、NCCL 与工具链

### 4.1 CUDA 各层的实际状态

| 层级 | 实测值 | 含义 / 边界 |
| --- | --- | --- |
| PyTorch 安装版本 | 2.9.1+cu126 | cu126 为该 PyTorch 构建的 CUDA 标记 |
| torch.version.cuda | 12.6 | PyTorch 编译时 CUDA 版本 |
| CUDA runtime Python 包 | 12.6.77 | 环境安装的 CUDA runtime 组件版本 |
| cuDNN Python 包 | 9.10.2.21 | 安装包版本 |
| torch.backends.cudnn.version() | 91002 | 运行时版本查询结果：9.10.2 |
| torch.backends.cudnn.is_available() | True | 库支持可查询；不代表当前能执行 GPU 运算 |
| torch.cuda.nccl.version() | 2.27.5 | NCCL 版本查询成功；未测试多卡通信 |
| 激活 TTT 后 nvcc | PATH 中未找到 | 当前 shell 没有可用的 CUDA 编译器 |
| CUDA_HOME / CUDA_PATH | 均未设置 | 没有通过这些变量指定 toolkit |
| NVIDIA 驱动版本 / GPU 型号 | 当前节点未能获取 | nvidia-smi 无法与驱动通信 |
| torch.cuda.is_available() | False | 当前进程 CUDA 不可用 |
| torch.cuda.device_count() | 0 | 当前未检测到 GPU |

`cuda-python==13.3.1`、`cuda-bindings==13.3.1` 和 `nvidia-cuda-nvdisasm==13.3.73` 也存在于环境中。这些组件的 13.x 包版本不能用来把 PyTorch 的 CUDA 版本改写成 13.x；本次 PyTorch 明确报告 CUDA 12.6。未验证 CUDA 13.x 绑定、CuPy 或其他扩展在 GPU 节点上的兼容性。

### 4.2 全部 CUDA / NVIDIA / 加速扩展相关包

| 包 | 安装版本 |
| --- | --- |
| cuda-bindings | 13.3.1 |
| cuda-core | 1.0.1 |
| cuda-pathfinder | 1.6.0 |
| cuda-python | 13.3.1 |
| cupy-cuda12x | 14.1.1 |
| deepspeed | 0.19.3 |
| flash_attn | 2.8.3.post1 |
| flashinfer-python | 0.5.3 |
| nvidia-cublas-cu12 | 12.6.4.1 |
| nvidia-cuda-cupti-cu12 | 12.6.80 |
| nvidia-cuda-nvdisasm | 13.3.73 |
| nvidia-cuda-nvrtc-cu12 | 12.6.77 |
| nvidia-cuda-runtime-cu12 | 12.6.77 |
| nvidia-cudnn-cu12 | 9.10.2.21 |
| nvidia-cudnn-frontend | 1.26.0 |
| nvidia-cufft-cu12 | 11.3.0.4 |
| nvidia-cufile-cu12 | 1.11.1.6 |
| nvidia-curand-cu12 | 10.3.7.77 |
| nvidia-cusolver-cu12 | 11.7.1.2 |
| nvidia-cusparse-cu12 | 12.5.4.2 |
| nvidia-cusparselt-cu12 | 0.7.1 |
| nvidia-cutlass-dsl | 4.6.1 |
| nvidia-cutlass-dsl-libs-base | 4.6.1 |
| nvidia-cutlass-dsl-libs-core | 4.6.1 |
| nvidia-cutlass-dsl-libs-cu12 | 4.6.1 |
| nvidia-ml-py | 13.610.43 |
| nvidia-nccl-cu12 | 2.27.5 |
| nvidia-nvjitlink-cu12 | 12.6.85 |
| nvidia-nvshmem-cu12 | 3.3.20 |
| nvidia-nvtx-cu12 | 12.6.77 |
| torch | 2.9.1+cu126 |
| torchaudio | 2.9.0 |
| torchvision | 0.24.0 |
| triton | 3.5.1 |
| vllm | 0.12.0 |

### 4.3 激活 TTT 后的 shell 状态

| 变量或命令 | 值 |
| --- | --- |
| CONDA_DEFAULT_ENV | TTT |
| CONDA_PREFIX | /home/da839/.conda/envs/TTT |
| CUDA_HOME | 未设置 |
| CUDA_PATH | 未设置 |
| CUDA_VISIBLE_DEVICES | 未设置 |
| LD_LIBRARY_PATH | 未设置 |
| LOADEDMODULES | StdEnv |
| python | /home/da839/.conda/envs/TTT/bin/python |
| pip | /home/da839/.conda/envs/TTT/bin/pip |
| conda | /home/da839/anaconda3/condabin/conda |
| nvcc | PATH 中未找到 |
| nvidia-smi | /usr/bin/nvidia-smi |
| gcc / g++ | /usr/bin/gcc / /usr/bin/g++ |
| cmake | PATH 中未找到 |
| ninja | /home/da839/.conda/envs/TTT/bin/ninja |

环境目录下未发现 conda 的 `activate.d` / `deactivate.d` 脚本。当前 shell 只加载 `StdEnv`；module 列表提供的 CUDA toolkit 包括 `CUDA/12.0.0`、`CUDA/12.1.1`、`CUDA/12.6.0`、`CUDA/12.8.0`、`CUDA/12.9.1`，它们是集群可选模块，不代表已加载到 TTT。未执行 module load 或 nvcc 编译。

### 4.4 PyTorch 构建信息

| 项目 | 值 |
| --- | --- |
| 构建用 GCC | 13.3 |
| C++ 标准 | 201703（C++17） |
| C++11 ABI | True |
| BLAS | Intel oneAPI MKL 2024.2 |
| MKL-DNN | 3.7.1 |
| OpenMP | 201511（4.5） |
| CPU capability usage | AVX512 |
| CUDA / cuDNN / NCCL | 构建中启用 |
| ROCm / MPI / XPU | 构建中未启用 |

上述编译器版本来自 PyTorch 构建记录，不是对当前 `/usr/bin/gcc` 版本的测量。

## 5. 依赖一致性与已完成检查

| 检查 | 结果 |
| --- | --- |
| 目标 Python 启动 | 通过 |
| PyTorch 导入 | 通过 |
| 2×2 CPU 矩阵乘法 | 通过，结果为 [[2.0, 2.0], [2.0, 2.0]] |
| cuDNN / NCCL 版本查询 | 成功 |
| GPU / CUDA 运算 | 未验证：当前登录节点无法使用 GPU |
| pip check | 退出码 1，发现以下 3 项声明冲突 |

```text
torchaudio 2.9.0 has requirement torch==2.9.0, but you have torch 2.9.1+cu126.
torchvision 0.24.0 has requirement torch==2.9.0, but you have torch 2.9.1+cu126.
vllm 0.12.0 has requirement torch==2.9.0, but you have torch 2.9.1+cu126.
```

`mfspd==0.1.0` 是 editable 安装；仅凭版本号不能重建其本地源代码状态。下方完整包表是版本盘点，不是经过重建验证的 lockfile。

## 6. 全部 Python distribution（231 项）

按包名排序。安装器取自 distribution 的 `INSTALLER`；`unknown` 表示该字段缺失，不能据此断言包来源。统计：pip 227 项、conda 2 项、unknown 2 项。

| 序号 | 包名 | 版本 | 安装器 | 备注 |
| --- | --- | --- | --- | --- |
| 1 | absl-py | 2.4.0 | pip |  |
| 2 | accelerate | 1.14.0 | pip |  |
| 3 | aiohappyeyeballs | 2.6.1 | pip |  |
| 4 | aiohttp | 3.13.5 | pip |  |
| 5 | aiosignal | 1.4.0 | pip |  |
| 6 | annotated-doc | 0.0.4 | pip |  |
| 7 | annotated-types | 0.8.0 | pip |  |
| 8 | anthropic | 0.71.0 | pip |  |
| 9 | antlr4-python3-runtime | 4.9.3 | pip |  |
| 10 | anyio | 4.12.1 | pip |  |
| 11 | apache-tvm-ffi | 0.1.12 | pip |  |
| 12 | astor | 0.8.1 | pip |  |
| 13 | attrs | 26.1.0 | pip |  |
| 14 | beautifulsoup4 | 4.15.0 | pip |  |
| 15 | blake3 | 1.0.9 | pip |  |
| 16 | cachetools | 7.1.6 | pip |  |
| 17 | cbor2 | 6.1.3 | pip |  |
| 18 | certifi | 2026.2.25 | pip |  |
| 19 | cffi | 2.1.1 | pip |  |
| 20 | charset-normalizer | 3.4.7 | pip |  |
| 21 | click | 8.3.1 | pip |  |
| 22 | cloudpickle | 3.1.2 | pip |  |
| 23 | codetiming | 1.4.0 | pip |  |
| 24 | compressed-tensors | 0.12.2 | pip |  |
| 25 | contourpy | 1.3.3 | pip |  |
| 26 | cryptography | 50.0.0 | pip |  |
| 27 | cuda-bindings | 13.3.1 | pip |  |
| 28 | cuda-core | 1.0.1 | pip |  |
| 29 | cuda-pathfinder | 1.6.0 | pip |  |
| 30 | cuda-python | 13.3.1 | pip |  |
| 31 | cupy-cuda12x | 14.1.1 | pip |  |
| 32 | cycler | 0.12.1 | pip |  |
| 33 | datasets | 4.8.4 | pip |  |
| 34 | deepspeed | 0.19.3 | pip |  |
| 35 | depyf | 0.20.0 | pip |  |
| 36 | detect-installer | 0.1.0 | pip |  |
| 37 | dill | 0.4.1 | pip |  |
| 38 | diskcache | 5.6.3 | pip |  |
| 39 | distro | 1.9.0 | pip |  |
| 40 | dnspython | 2.8.0 | pip |  |
| 41 | docstring_parser | 0.18.0 | pip |  |
| 42 | einops | 0.8.2 | pip |  |
| 43 | email-validator | 2.3.0 | pip |  |
| 44 | fastapi | 0.140.0 | pip |  |
| 45 | fastapi-cli | 0.0.32 | pip |  |
| 46 | fastapi-cloud-cli | 0.22.2 | pip |  |
| 47 | fastar | 0.11.0 | pip |  |
| 48 | filelock | 3.25.2 | pip |  |
| 49 | flash_attn | 2.8.3.post1 | pip |  |
| 50 | flashinfer-python | 0.5.3 | pip |  |
| 51 | fonttools | 4.63.0 | pip |  |
| 52 | frozenlist | 1.8.0 | pip |  |
| 53 | fsspec | 2026.2.0 | pip |  |
| 54 | gdown | 6.1.0 | pip |  |
| 55 | gguf | 0.19.0 | pip |  |
| 56 | h11 | 0.16.0 | pip |  |
| 57 | hf-xet | 1.4.3 | pip |  |
| 58 | hjson | 3.1.0 | pip |  |
| 59 | httpcore | 1.0.9 | pip |  |
| 60 | httptools | 0.8.0 | pip |  |
| 61 | httpx | 0.28.1 | pip |  |
| 62 | huggingface_hub | 0.36.2 | pip |  |
| 63 | hydra-core | 1.3.4 | pip |  |
| 64 | id | 1.6.1 | pip |  |
| 65 | idna | 3.11 | pip |  |
| 66 | importlib_metadata | 9.0.0 | pip |  |
| 67 | iniconfig | 2.3.0 | pip |  |
| 68 | interegular | 0.3.3 | pip |  |
| 69 | Jinja2 | 3.1.6 | pip |  |
| 70 | jiter | 0.16.0 | pip |  |
| 71 | jmespath | 1.1.0 | pip |  |
| 72 | joblib | 1.5.3 | pip |  |
| 73 | jsonschema | 4.26.0 | pip |  |
| 74 | jsonschema-specifications | 2025.9.1 | pip |  |
| 75 | kernels | 0.9.0 | pip |  |
| 76 | kernels-data | 0.16.0 | pip |  |
| 77 | kiwisolver | 1.5.0 | pip |  |
| 78 | lark | 1.2.2 | pip |  |
| 79 | latex2sympy2_extended | 1.11.0 | pip |  |
| 80 | llguidance | 1.3.0 | pip |  |
| 81 | llvmlite | 0.44.0 | pip |  |
| 82 | lm-format-enforcer | 0.11.3 | pip |  |
| 83 | loguru | 0.7.3 | pip |  |
| 84 | markdown-it-py | 4.0.0 | pip |  |
| 85 | MarkupSafe | 3.0.3 | pip |  |
| 86 | math-verify | 0.9.0 | pip |  |
| 87 | matplotlib | 3.11.1 | pip |  |
| 88 | mdurl | 0.1.2 | pip |  |
| 89 | mfspd | 0.1.0 | pip | editable 本地安装 |
| 90 | mistral_common | 1.11.7 | pip |  |
| 91 | model-hosting-container-standards | 0.1.16 | pip |  |
| 92 | mpmath | 1.3.0 | pip |  |
| 93 | msgpack | 1.2.1 | pip |  |
| 94 | msgspec | 0.21.1 | pip |  |
| 95 | multidict | 6.7.1 | pip |  |
| 96 | multiprocess | 0.70.19 | pip |  |
| 97 | networkx | 3.6.1 | pip |  |
| 98 | ninja | 1.13.0 | pip |  |
| 99 | nltk | 3.9.4 | pip |  |
| 100 | numba | 0.61.2 | pip |  |
| 101 | numpy | 2.2.6 | pip |  |
| 102 | nvidia-cublas-cu12 | 12.6.4.1 | pip |  |
| 103 | nvidia-cuda-cupti-cu12 | 12.6.80 | pip |  |
| 104 | nvidia-cuda-nvdisasm | 13.3.73 | pip |  |
| 105 | nvidia-cuda-nvrtc-cu12 | 12.6.77 | pip |  |
| 106 | nvidia-cuda-runtime-cu12 | 12.6.77 | pip |  |
| 107 | nvidia-cudnn-cu12 | 9.10.2.21 | pip |  |
| 108 | nvidia-cudnn-frontend | 1.26.0 | pip |  |
| 109 | nvidia-cufft-cu12 | 11.3.0.4 | pip |  |
| 110 | nvidia-cufile-cu12 | 1.11.1.6 | pip |  |
| 111 | nvidia-curand-cu12 | 10.3.7.77 | pip |  |
| 112 | nvidia-cusolver-cu12 | 11.7.1.2 | pip |  |
| 113 | nvidia-cusparse-cu12 | 12.5.4.2 | pip |  |
| 114 | nvidia-cusparselt-cu12 | 0.7.1 | pip |  |
| 115 | nvidia-cutlass-dsl | 4.6.1 | pip |  |
| 116 | nvidia-cutlass-dsl-libs-base | 4.6.1 | pip |  |
| 117 | nvidia-cutlass-dsl-libs-core | 4.6.1 | pip |  |
| 118 | nvidia-cutlass-dsl-libs-cu12 | 4.6.1 | pip |  |
| 119 | nvidia-ml-py | 13.610.43 | pip |  |
| 120 | nvidia-nccl-cu12 | 2.27.5 | pip |  |
| 121 | nvidia-nvjitlink-cu12 | 12.6.85 | pip |  |
| 122 | nvidia-nvshmem-cu12 | 3.3.20 | pip |  |
| 123 | nvidia-nvtx-cu12 | 12.6.77 | pip |  |
| 124 | omegaconf | 2.3.1 | pip |  |
| 125 | openai | 2.48.0 | pip |  |
| 126 | openai-harmony | 0.0.8 | pip |  |
| 127 | opencv-python-headless | 5.0.0.93 | pip |  |
| 128 | orjson | 3.11.9 | pip |  |
| 129 | outlines_core | 0.2.11 | pip |  |
| 130 | packaging | 25.0 | conda |  |
| 131 | pandas | 3.0.2 | pip |  |
| 132 | partial-json-parser | 0.2.1.1.post7 | pip |  |
| 133 | peft | 0.19.1 | pip |  |
| 134 | pillow | 12.3.0 | pip |  |
| 135 | pip | 26.0.1 | conda |  |
| 136 | platformdirs | 4.11.0 | pip |  |
| 137 | pluggy | 1.6.0 | pip |  |
| 138 | prettytable | 3.18.0 | pip |  |
| 139 | prometheus-fastapi-instrumentator | 8.0.2 | pip |  |
| 140 | prometheus_client | 0.26.0 | pip |  |
| 141 | propcache | 0.4.1 | pip |  |
| 142 | protobuf | 6.33.6 | pip |  |
| 143 | psutil | 7.2.2 | pip |  |
| 144 | pwinput | 1.0.3 | pip |  |
| 145 | py-cpuinfo | 9.0.0 | pip |  |
| 146 | pyarrow | 23.0.1 | pip |  |
| 147 | pyasn1 | 0.6.4 | pip |  |
| 148 | pybase64 | 1.4.3 | pip |  |
| 149 | pybind11 | 3.0.4 | pip |  |
| 150 | pycountry | 26.2.16 | pip |  |
| 151 | pycparser | 3.0 | pip |  |
| 152 | pydantic | 2.13.4 | pip |  |
| 153 | pydantic-extra-types | 2.11.1 | pip |  |
| 154 | pydantic-settings | 2.14.2 | pip |  |
| 155 | pydantic_core | 2.46.4 | pip |  |
| 156 | pyecharts | 2.1.0 | pip |  |
| 157 | Pygments | 2.19.2 | pip |  |
| 158 | PyJWT | 2.13.0 | pip |  |
| 159 | pylatexenc | 2.11 | pip |  |
| 160 | pyOpenSSL | 26.4.0 | pip |  |
| 161 | pyparsing | 3.3.2 | pip |  |
| 162 | PySocks | 1.7.1 | pip |  |
| 163 | pytest | 9.1.1 | pip |  |
| 164 | python-dateutil | 2.9.0.post0 | pip |  |
| 165 | python-dotenv | 1.2.2 | pip |  |
| 166 | python-json-logger | 4.1.0 | pip |  |
| 167 | python-multipart | 0.0.32 | pip |  |
| 168 | pyvers | 0.1.0 | pip |  |
| 169 | PyYAML | 6.0.3 | pip |  |
| 170 | pyzmq | 27.1.0 | pip |  |
| 171 | ray | 2.56.1 | pip |  |
| 172 | referencing | 0.37.0 | pip |  |
| 173 | regex | 2026.4.4 | pip |  |
| 174 | requests | 2.33.1 | pip |  |
| 175 | rfc3161-client | 1.0.8 | pip |  |
| 176 | rfc8785 | 0.1.4 | pip |  |
| 177 | rich | 14.3.3 | pip |  |
| 178 | rich-toolkit | 0.20.3 | pip |  |
| 179 | rignore | 0.8.0 | pip |  |
| 180 | rouge_score | 0.1.2 | pip |  |
| 181 | rpds-py | 2026.6.3 | pip |  |
| 182 | safetensors | 0.8.0 | pip |  |
| 183 | scipy | 1.17.1 | pip |  |
| 184 | securesystemslib | 1.4.0 | pip |  |
| 185 | sentencepiece | 0.2.2 | pip |  |
| 186 | sentry-sdk | 2.66.1 | pip |  |
| 187 | setproctitle | 1.3.7 | pip |  |
| 188 | setuptools | 80.10.2 | unknown |  |
| 189 | shellingham | 1.5.4 | pip |  |
| 190 | sigstore | 4.5.0 | pip |  |
| 191 | sigstore-models | 0.0.6 | pip |  |
| 192 | sigstore-rekor-types | 0.0.18 | pip |  |
| 193 | simplejson | 4.1.1 | pip |  |
| 194 | six | 1.17.0 | pip |  |
| 195 | sniffio | 1.3.1 | pip |  |
| 196 | soupsieve | 2.9.1 | pip |  |
| 197 | starlette | 1.3.1 | pip |  |
| 198 | supervisor | 4.3.0 | pip |  |
| 199 | swanlab | 0.9.1 | pip |  |
| 200 | sympy | 1.14.0 | pip |  |
| 201 | tabulate | 0.10.0 | pip |  |
| 202 | tensordict | 0.10.0 | pip |  |
| 203 | tiktoken | 0.13.0 | pip |  |
| 204 | tokenizers | 0.22.2 | pip |  |
| 205 | tomlkit | 0.15.1 | pip |  |
| 206 | torch | 2.9.1+cu126 | pip |  |
| 207 | torchaudio | 2.9.0 | pip |  |
| 208 | torchdata | 0.11.0 | pip |  |
| 209 | torchvision | 0.24.0 | pip |  |
| 210 | tqdm | 4.67.3 | pip |  |
| 211 | transformers | 4.57.6 | pip |  |
| 212 | triton | 3.5.1 | pip |  |
| 213 | tuf | 7.0.0 | pip |  |
| 214 | typer | 0.24.1 | pip |  |
| 215 | typing-inspection | 0.4.2 | pip |  |
| 216 | typing_extensions | 4.15.0 | pip |  |
| 217 | urllib3 | 2.6.3 | pip |  |
| 218 | uvicorn | 0.51.0 | pip |  |
| 219 | uvloop | 0.22.1 | pip |  |
| 220 | vllm | 0.12.0 | pip |  |
| 221 | wandb | 0.28.1 | pip |  |
| 222 | watchdog | 6.0.0 | pip |  |
| 223 | watchfiles | 1.2.0 | pip |  |
| 224 | wcwidth | 0.8.2 | pip |  |
| 225 | websockets | 16.1.1 | pip |  |
| 226 | wheel | 0.46.3 | unknown |  |
| 227 | wrapt | 2.2.2 | pip |  |
| 228 | xgrammar | 0.1.27 | pip |  |
| 229 | xxhash | 3.6.0 | pip |  |
| 230 | yarl | 1.23.0 | pip |  |
| 231 | zipp | 4.1.0 | pip |  |

## 7. 全部 conda 包记录（34 项）

以下为 conda 安装记录；它们覆盖 Python 解释器、系统库和部分 Python 工具。Python 表与本表有重叠。全部记录来自 Anaconda defaults 的 `https://repo.anaconda.com/pkgs/main`，具体子目录见表。

| 包名 | 版本 | 构建号 | 子目录 |
| --- | --- | --- | --- |
| _libgcc_mutex | 0.1 | main | linux-64 |
| _openmp_mutex | 5.1 | 1_gnu | linux-64 |
| bzip2 | 1.0.8 | h5eee18b_6 | linux-64 |
| ca-certificates | 2025.12.2 | h06a4308_0 | linux-64 |
| ld_impl_linux-64 | 2.44 | h9e0c5a2_3 | linux-64 |
| libexpat | 2.7.4 | h7354ed3_0 | linux-64 |
| libffi | 3.4.4 | h6a678d5_1 | linux-64 |
| libgcc | 15.2.0 | h69a1729_7 | linux-64 |
| libgcc-ng | 15.2.0 | h166f726_7 | linux-64 |
| libgomp | 15.2.0 | h4751f2c_7 | linux-64 |
| libnsl | 2.0.0 | h5eee18b_0 | linux-64 |
| libstdcxx | 15.2.0 | h39759b7_7 | linux-64 |
| libstdcxx-ng | 15.2.0 | hc03a8fd_7 | linux-64 |
| libuuid | 1.41.5 | h5eee18b_0 | linux-64 |
| libxcb | 1.17.0 | h9b100fa_0 | linux-64 |
| libzlib | 1.3.1 | hb25bd0a_0 | linux-64 |
| ncurses | 6.5 | h7934f7d_0 | linux-64 |
| openssl | 3.5.5 | h1b28b03_0 | linux-64 |
| packaging | 25.0 | py311h06a4308_1 | linux-64 |
| pip | 26.0.1 | pyhc872135_0 | noarch |
| pthread-stubs | 0.3 | h0ce48e5_1 | linux-64 |
| python | 3.11.15 | h741d88c_0 | linux-64 |
| readline | 8.3 | hc2a1206_0 | linux-64 |
| setuptools | 80.10.2 | py311h06a4308_0 | linux-64 |
| sqlite | 3.51.2 | h3e8d24a_0 | linux-64 |
| tk | 8.6.15 | h54e0aa7_0 | linux-64 |
| tzdata | 2026a | he532380_0 | noarch |
| wheel | 0.46.3 | py311h06a4308_0 | linux-64 |
| xorg-libx11 | 1.8.12 | h9b100fa_1 | linux-64 |
| xorg-libxau | 1.0.12 | h9b100fa_0 | linux-64 |
| xorg-libxdmcp | 1.1.5 | h9b100fa_0 | linux-64 |
| xorg-xorgproto | 2024.1 | h5eee18b_1 | linux-64 |
| xz | 5.8.2 | h448239c_0 | linux-64 |
| zlib | 1.3.1 | hb25bd0a_0 | linux-64 |

## 8. 采集命令与原始 PyTorch 输出

以下命令用于重新盘点；本报告的完整包表以本地元数据读取结果为准。

```bash
source /home/da839/anaconda3/etc/profile.d/conda.sh
conda activate TTT
python --version
python -m pip list
python -m pip check
conda list -p /home/da839/.conda/envs/TTT
command -v nvcc
nvidia-smi
module -t list
module -t avail CUDA
```

原始 PyTorch 检查输出（包括当前节点的 NVML 警告）：

```text
/home/da839/.conda/envs/TTT/lib/python3.11/site-packages/torch/cuda/__init__.py:827: UserWarning: Can't initialize NVML
  warnings.warn("Can't initialize NVML")
{
  "torch": "2.9.1+cu126",
  "torch_path": "/home/da839/.conda/envs/TTT/lib/python3.11/site-packages/torch/__init__.py",
  "cuda_build": "12.6",
  "cuda_available": false,
  "device_count": 0,
  "cudnn_version": 91002,
  "cudnn_available": true,
  "cxx11_abi": true,
  "cpu_result": [
    [
      2.0,
      2.0
    ],
    [
      2.0,
      2.0
    ]
  ],
  "nccl_version": [
    2,
    27,
    5
  ]
}
PyTorch built with:
  - GCC 13.3
  - C++ Version: 201703
  - Intel(R) oneAPI Math Kernel Library Version 2024.2-Product Build 20240605 for Intel(R) 64 architecture applications
  - Intel(R) MKL-DNN v3.7.1 (Git Hash 8d263e693366ef8db40acc569cc7d8edf644556d)
  - OpenMP 201511 (a.k.a. OpenMP 4.5)
  - LAPACK is enabled (usually provided by MKL)
  - NNPACK is enabled
  - CPU capability usage: AVX512
  - Build settings: BLAS_INFO=mkl, BUILD_TYPE=Release, COMMIT_SHA=5811a8d7da873dd699ff6687092c225caffcf1bb, CUDA_VERSION=12.6, CUDNN_VERSION=9.10.2, CXX_COMPILER=/opt/rh/gcc-toolset-13/root/usr/bin/c++, CXX_FLAGS= -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -DNDEBUG -DUSE_KINETO -DLIBKINETO_NOROCTRACER -DLIBKINETO_NOXPUPTI=ON -DUSE_FBGEMM -DUSE_PYTORCH_QNNPACK -DUSE_XNNPACK -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -O2 -fPIC -DC10_NODEPRECATED -Wall -Wextra -Werror=return-type -Werror=non-virtual-dtor -Werror=range-loop-construct -Werror=bool-operation -Wnarrowing -Wno-missing-field-initializers -Wno-unknown-pragmas -Wno-unused-parameter -Wno-strict-overflow -Wno-strict-aliasing -Wno-stringop-overflow -Wsuggest-override -Wno-psabi -Wno-error=old-style-cast -faligned-new -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Wno-dangling-reference -Wno-error=dangling-reference -Wno-stringop-overflow, LAPACK_INFO=mkl, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, TORCH_VERSION=2.9.1, USE_CUDA=ON, USE_CUDNN=ON, USE_CUSPARSELT=1, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_GLOO=ON, USE_MKL=ON, USE_MKLDNN=ON, USE_MPI=OFF, USE_NCCL=1, USE_NNPACK=ON, USE_OPENMP=ON, USE_ROCM=OFF, USE_ROCM_KERNEL_ASSERT=OFF, USE_XCCL=OFF, USE_XPU=OFF,
```

本报告反映 2026-09-07 的环境快照；没有安装、升级或卸载包，也没有修改 CUDA 模块设置。
