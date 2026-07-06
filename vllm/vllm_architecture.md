# vLLM 源码架构分析

> vLLM (v0.x) — 高效大模型推理引擎
> 源码路径: `Project/vLLM/vllm/`
> 更新时间: 2026-07-06

---

## 目录

1. [项目总览](#1-项目总览)
2. [顶层核心数据结构和配置](#2-顶层核心数据结构和配置)
3. [引擎层 (engine/)](#3-引擎层-engine)
4. [V1 新一代引擎 (v1/)](#4-v1-新一代引擎-v1)
5. [配置系统 (config/)](#5-配置系统-config)
6. [服务入口 (entrypoints/)](#6-服务入口-entrypoints)
7. [模型执行核心 (model_executor/)](#7-模型执行核心-model_executor)
8. [C++/CUDA 底层算子 (csrc/)](#8-ccuda-底层算子-csrc)
9. [分布式通信 (distributed/)](#9-分布式通信-distributed)
10. [多模态处理 (multimodal/)](#10-多模态处理-multimodal)
11. [LoRA 适配器 (lora/)](#11-lora-适配器-lora)
12. [推理和工具解析器](#12-推理和工具解析器)
13. [分词器 (tokenizers/)](#13-分词器-tokenizers)
14. [编译优化 (compilation/)](#14-编译优化-compilation)
15. [硬件平台抽象 (platforms/)](#15-硬件平台抽象-platforms)
16. [测试体系 (tests/)](#16-测试体系-tests)
17. [工具模块汇总 (utils/ 等)](#17-工具模块汇总-utils-等)
18. [架构总图](#18-架构总图)

---

## 1. 项目总览

### 1.1 项目定位

vLLM 是一个快速、易用、低成本的大模型推理和服务库，最初由 UC Berkeley **Sky Computing Lab** 开发，现已成为最活跃的开源 AI 推理项目之一（2000+ 贡献者）。

### 1.2 顶层目录结构

```
vllm/                         # 项目根目录
├── vllm/                     # Python 主包（核心源码）
│   ├── __init__.py           # 公共导出（惰性导入）
│   ├── version.py            # 版本信息
│   ├── connections.py        # 连接管理
│   ├── envs.py               # 环境变量
│   ├── env_override.py       # 环境变量覆盖
│   ├── exceptions.py         # 异常定义
│   ├── forward_context.py    # 前向上下文
│   ├── logger.py             # 日志
│   ├── sampling_params.py    # 采样参数
│   ├── pooling_params.py     # 池化参数
│   ├── logits_process.py     # Logits 处理
│   ├── logprobs.py           # Logprob 数据结构
│   ├── sequence.py           # 序列/中间张量
│   ├── outputs.py            # 输出结构
│   ├── inputs.py             # 输入格式
│   ├── scalar_type.py        # 标量类型
│   ├── model_inspection.py   # 模型检查
│   ├── tasks.py              # 任务类型定义
│   ├── engine/               # 引擎层（薄包装）
│   ├── v1/                   # V1 新一代引擎
│   ├── config/               # 统一配置系统
│   ├── entrypoints/          # 服务入口（最大目录）
│   ├── model_executor/       # 模型执行核心
│   ├── distributed/          # 分布式通信
│   ├── multimodal/           # 多模态处理
│   ├── lora/                 # LoRA 适配器
│   ├── compilation/          # 编译优化
│   ├── platforms/            # 硬件平台抽象
│   ├── parser/               # 结构化输出解析器
│   ├── tool_parsers/         # 工具调用解析器
│   ├── reasoning/            # 推理解析器
│   ├── tokenizers/           # 分词器
│   ├── kernels/              # 自定义内核
│   ├── models/               # 平台定制模型
│   ├── renderers/            # 渲染/反渲染
│   ├── profiler/             # 性能分析
│   ├── ray/                  # Ray 集成
│   ├── tracing/              # 链路追踪
│   ├── transformers_utils/   # Transformers 工具
│   ├── triton_utils/         # Triton 工具
│   ├── cute_utils/           # CuTe 工具
│   ├── ir/                   # 中间表示
│   ├── logging_utils/        # 日志工具
│   ├── usage/                # 使用统计
│   ├── plugins/              # 插件系统
│   ├── assets/               # 多模态资产
│   └── utils/                # 通用工具库
├── csrc/                     # C++/CUDA 源码
├── tests/                    # 测试套件
├── benchmarks/               # 基准测试
├── docs/                     # 文档
├── examples/                 # 示例
├── requirements/             # 依赖管理
├── scripts/                  # 构建/部署脚本
├── cmake/                    # CMake 构建
├── rust/                     # Rust 扩展
├── docker/                   # Docker 配置
├── tools/                    # 开发工具
├── .buildkite/               # CI 配置
├── pyproject.toml            # 项目元数据
├── CMakeLists.txt            # CMake 构建配置
├── setup.py                  # 安装脚本
└── README.md                 # 项目说明
```

### 1.3 核心设计理念

| 理念 | 说明 |
|------|------|
| **模块化** | 每个子系统都设计为可独立替换的模块 |
| **可插拔** | 注意力后端、线性层内核、MoE 实现都有多种替代方案 |
| **自动适配** | 平台检测完全自动化，模型架构通过注册表动态加载 |
| **双 ABI** | 旧版 ABI（CPU 兼容）和 Stable ABI（CUDA，vLLM 正向此迁移） |

---

## 2. 顶层核心数据结构和配置

### 2.1 根目录关键文件

| 文件 | 核心类/内容 | 说明 |
|------|------------|------|
| `__init__.py` | `LLM`, `LLMEngine`, `SamplingParams` 等 | 公共 API 导出，采用惰性导入策略 |
| `sampling_params.py` | `SamplingParams`, `StructuredOutputsParams` | 采样参数（temperature, top_p, top_k），结构化输出配置 |
| `pooling_params.py` | `PoolingParams` | 池化任务参数 |
| `logits_process.py` | `LogitsProcessor`, `NoBadWordsLogitsProcessor` | Logits 处理器（禁止词表、偏置等） |
| `logprobs.py` | `Logprob`, `FlatLogprobs`, `PromptLogprobs` | Logprob 数据结构，支持 OpenAI 兼容格式 |
| `sequence.py` | `IntermediateTensors` | 流水线中间状态（hidden states, residuals） |
| `outputs.py` | `CompletionOutput`, `PoolingOutput`, `RequestOutput` | 请求输出结构（文本、token IDs、logprobs） |
| `inputs.py` | `PromptType`, `TextPrompt`, `TokensPrompt` | 输入格式定义 |
| `scalar_type.py` | `ScalarType` | 自定义标量类型（sub-byte 数据类型） |
| `tasks.py` | `SupportedTask`, `PoolingTask` | 支持的任务类型（generate, embed, classify, score） |

### 2.2 惰性导入机制

`__init__.py` 使用 `MODULE_ATTRS` 字典映射，在 `__getattr__` 中动态导入，避免一次性加载所有依赖。

---

## 3. 引擎层 (engine/)

**路径**: `vllm/vllm/engine/`

当前版本中，`engine/` 是一个**薄包装层**，实际实现已迁移到 `v1/engine/`。

| 文件 | 说明 |
|------|------|
| `arg_utils.py` | `EngineArgs`, `AsyncEngineArgs` — 解析命令行参数为 `VllmConfig` |
| `protocol.py` | `EngineClient` 抽象基类 — 定义 generate/encode/abort/健康检查接口 |
| `async_llm_engine.py` | 一行重定向: `AsyncLLMEngine = AsyncLLM` (来自 v1) |
| `llm_engine.py` | 一行重定向: `LLMEngine = V1LLMEngine` (来自 v1) |

---

## 4. V1 新一代引擎 (v1/)

**路径**: `vllm/vllm/v1/`

vLLM 采用全新的 V1 架构，基于 **ZeroMQ** 进程间通信。

### 4.1 目录结构

```
v1/
├── __init__.py
├── engine/               # V1 引擎核心
│   ├── async_llm.py      # 异步引擎
│   ├── llm_engine.py     # 同步引擎
│   ├── core.py           # 核心循环（ZeroMQ 通信）
│   ├── core_client.py    # 核心客户端
│   ├── coordinator.py    # 协调器
│   ├── detokenizer.py    # 去分词器
│   ├── input_processor.py # 输入处理器
│   └── output_processor.py # 输出处理器
├── executor/             # 分布式执行器
│   ├── abstract.py       # Executor 抽象基类
│   ├── uniproc_executor.py # 单进程
│   ├── multiproc_executor.py # 多进程
│   ├── ray_executor.py   # Ray 分布式
│   ├── ray_executor_v2.py # Ray V2
│   └── ray_utils.py     # Ray 工具
├── core/                 # 调度核心
│   ├── sched/            # 调度器
│   │   ├── scheduler.py  # 调度器
│   │   ├── async_scheduler.py # 异步调度器
│   │   └── request_queue.py # 请求队列
│   ├── block_pool.py     # 块池
│   ├── kv_cache_manager.py # KV 缓存管理
│   ├── kv_cache_coordinator.py # KV 缓存协调
│   └── kv_cache_utils.py # KV 缓存工具
├── attention/            # 注意力后端（可插拔）
│   ├── backend.py        # AttentionBackend 抽象基类
│   ├── selector.py       # 后端选择器
│   ├── backends/         # 具体后端实现
│   │   ├── flash_attn.py
│   │   ├── flash_infer.py
│   │   ├── triton_attention.py
│   │   ├── mamba.py
│   │   ├── cpu_attn.py
│   │   ├── rocm_attn.py
│   │   └── mla/          # MLA 注意力专用
│   └── ops/              # 注意力算子
├── worker/               # 工作器
│   ├── gpu_model_runner.py # GPU 模型运行器
│   ├── gpu_worker.py
│   ├── cpu_model_runner.py
│   ├── cpu_worker.py
│   ├── xpu_model_runner.py
│   ├── xpu_worker.py
│   └── gpu/              # GPU 专用子模块
│       ├── model_state.py
│       ├── sampler.py
│       └── spec_decode/
├── sample/               # 采样模块
│   ├── sampler.py
│   ├── rejection_sampler.py
│   └── logits_processor/
├── metrics/              # 指标统计
│   ├── stats.py
│   ├── prometheus.py
│   ├── loggers.py
│   └── perf.py
├── kv_offload/           # KV 缓存卸载
│   ├── cpu_offload.py    # CPU 卸载（LRU/ARC）
│   ├── tiering/          # 分层卸载（文件系统/对象存储/P2P）
│   └── simple_kv_offload/
├── spec_decode/          # 推测解码
│   ├── eagle/            # EAGLE 推测
│   ├── medusa/           # Medusa 推测
│   ├── dflash/           # DFlash
│   ├── draft_model/
│   ├── ngram/
│   └── suffix/
├── structured_output/    # 结构化输出
│   ├── outlines/
│   ├── lmfe/
│   ├── guidance/
│   └── xgrammar/
├── pool/                 # 池化
│   └── late_interaction/
├── cudagraph_dispatcher.py
├── request.py
├── outputs.py
├── serial_utils.py
└── utils.py
```

### 4.2 V1 架构改进

- **ZeroMQ IPC**: 引擎核心基于 ZeroMQ 进行进程间通信
- **全新调度器**: `core/sched/` 支持异步调度
- **可插拔注意力**: 统一的 `AttentionBackend` 接口
- **KV 缓存卸载**: 支持 CPU/分层存储/P2P 卸载
- **弹性专家并行**: 动态扩展/收缩专家
- **完善的推测解码**: EAGLE、Medusa、dFlash、N-gram 等

---

## 5. 配置系统 (config/)

**路径**: `vllm/vllm/config/`

所有子配置聚合到统一的 **`VllmConfig`** 中。

| 文件 | 配置类 | 说明 |
|------|--------|------|
| `vllm.py` | `VllmConfig` | **顶级配置容器**，聚合所有子配置 |
| `model.py` | `ModelConfig` | 模型架构、dtype、tokenizer |
| `cache.py` | `CacheConfig` | KV 缓存（块大小、预算、前缀缓存算法） |
| `attention.py` | `AttentionConfig` | 注意力机制（sliding window 等） |
| `scheduler.py` | `SchedulerConfig` | 调度策略、最大并行数、交换策略 |
| `parallel.py` | `ParallelConfig` | 分布式并行（TP/PP/DP 规模、后端） |
| `load.py` | `LoadConfig` | 权重加载策略 |
| `quantization.py` | `QuantizationConfigArgs` | 量化参数 |
| `lora.py` | `LoRAConfig` | LoRA 适配器 |
| `speculative.py` | `SpeculativeConfig` | 推测解码 |
| `multimodal.py` | `MultiModalConfig` | 多模态输入 |
| `offload.py` | `OffloadConfig` | CPU/UVA 卸载 |
| `kv_transfer.py` | `KVTransferConfig` | 跨节点 KV 传输（PD 分离） |
| `compilation.py` | `CompilationConfig` | torch.compile / CUDA Graph |
| `diffusion.py` | `DiffusionConfig` | 扩散模型 |
| `profiler.py` | `ProfilerConfig` | 性能分析 |
| `mamba.py` | `MambaConfig` | Mamba SSM |
| `pooler.py` | `PoolerConfig` | 池化层 |
| `kernel.py` | `KernelConfig` | 内核选择（线性层后端、MoE 后端） |
| `reasoning.py` | `ReasoningConfig` | 推理/思维链 |
| `speech_to_text.py` | `SpeechToTextConfig` | 语音识别 |
| `structured_outputs.py` | `StructuredOutputsConfig` | 结构化输出 |
| `observability.py` | `ObservabilityConfig` | 可观测性 |
| `weight_transfer.py` | `WeightTransferConfig` | 权重传输 |
| `model_arch.py` | — | 模型架构常量定义 |

---

## 6. 服务入口 (entrypoints/)

**路径**: `vllm/vllm/entrypoints/`

这是 vLLM 最大的目录，包含多种接入方式。

### 6.1 目录结构

```
entrypoints/
├── llm.py              # Python 直接调用入口 LLM 类
├── chat_utils.py       # 聊天模板工具
├── launcher.py         # HTTP 服务器启动器（uvicorn）
├── offline_utils.py    # 离线推理工具
├── grpc_server.py      # gRPC 服务器
├── openai/             # OpenAI 兼容 API
│   ├── api_server.py   # FastAPI 主入口
│   ├── cli_args.py     # CLI 参数
│   ├── run_batch.py    # 批处理
│   ├── chat_completion/ # Chat Completions (protocol, serving, api_router)
│   ├── completion/     # Completions
│   ├── engine/         # 引擎协议
│   ├── models/         # 模型列表
│   └── responses/      # Responses API
├── anthropic/          # Anthropic Messages API
│   ├── protocol.py     # 协议模型
│   ├── serving.py      # 服务处理
│   └── api_router.py   # 路由
├── cli/                # 命令行工具
│   ├── main.py         # 主入口（serve, bench, launch 等）
│   ├── serve.py        # vllm serve
│   ├── launch.py       # vllm launch
│   ├── run_batch.py    # vllm run_batch
│   ├── openai.py       # vllm openai
│   ├── collect_env.py  # 环境收集
│   └── benchmark/      # 基准测试子命令
├── mcp/                # MCP 工具服务器
│   ├── tool.py         # Tool 抽象基类
│   └── tool_server.py  # MCP 工具服务器
├── generate/           # 文本生成入口
├── pooling/            # 池化入口（embed, classify, score）
├── serve/              # 服务器基础设施
│   ├── engine/         # 引擎服务
│   ├── lora/           # LoRA 管理
│   └── tokenize/       # 分词服务
├── scale_out/          # 偏出推理
├── speech_to_text/     # 语音服务
└── anthropic/          # Anthropic API
```

### 6.2 子模块要点

**MCP 工具服务器** (`mcp/`):
- `Tool` 抽象基类 + `HarmonyBrowserTool`, `HarmonyPythonTool`
- `ToolServer` — 通过 SSE 连接 MCP 服务器，适配为 Harmony 工具

**Anthropic API** (`anthropic/`):
- 支持 text, image, tool_use, thinking 等多种 content block
- 将 Anthropic Messages API 请求转换为 OpenAI 兼容格式

**OpenAI API** (`openai/`):
- 完整的 Chat Completions / Completions / Responses API
- 支持流式、工具调用、结构化输出
- 模型列表管理、批处理、DP supervisor

---

## 7. 模型执行核心 (model_executor/)

**路径**: `vllm/vllm/model_executor/`

### 7.1 模块结构

```
model_executor/
├── layers/             # 神经网络层
│   ├── attention/      # 注意力层实现
│   ├── fused_moe/      # MoE 融合层
│   ├── fla/            # 快速线性注意力
│   ├── activation.py   # 激活函数
│   ├── conv.py         # 卷积层
│   ├── layernorm.py    # 归一化层
│   ├── linear.py       # 线性层
│   ├── mla.py          # MLA 辅助层
│   └── resampler.py    # 重采样层
├── models/             # 120+ 种模型架构
│   ├── registry.py     # 模型注册中心（200+ 架构）
│   ├── llama.py        # LLaMA 系列
│   ├── qwen2.py        # Qwen2
│   ├── qwen3.py        # Qwen3
│   ├── deepseek_v32.py
│   ├── gemma3.py
│   ├── gemma4.py
│   ├── mixtral.py
│   ├── chatglm.py
│   ├── internvl.py
│   └── ... (100+ 更多)
├── kernels/            # 高性能内核
│   ├── attention/      # 注意力内核
│   ├── linear/         # 线性层内核（13+ 后端）
│   │   ├── mixed_precision/ # Marlin, EXL2, Cutlass, Machete...
│   │   ├── mxfp4/      # MXFP4
│   │   ├── mxfp8/      # MXFP8
│   │   ├── nvfp4/      # NVFP4
│   │   └── scaled_mm/  # 缩放矩阵乘法
│   ├── mhc/            # 多头卷积内核
│   └── fused_moe/      # MoE 内核
├── offloader/          # 权重卸载
├── warmup/             # 预热
├── custom_op.py        # 自定义操作
├── parameter.py        # 参数类型
└── utils.py            # 工具函数
```

### 7.2 注意力层概览

| 文件/类 | 说明 |
|---------|------|
| `Attention` | 标准多头注意力 |
| `CrossAttention` | 交叉注意力（编码器-解码器） |
| `MLAAttention` | 多头潜在注意力（DeepSeek MLA） |
| `ChunkedLocalAttention` | 分块局部注意力 |
| `EncoderOnlyAttention` | 仅编码器注意力 |
| `MMEncoderAttention` | 多模态编码器注意力 |
| `RSWAAttention` | 循环滑动窗口注意力 |
| `StaticSinkAttention` | 静态汇点注意力 |
| `PrefillPrefixLMAttention` | 前缀 LM 预填充注意力 |

### 7.3 模型数量统计

| 模型家族 | 文件数 | 代表模型 |
|---------|--------|---------|
| LLaMA | llama.py, llama4.py, mllama4.py | LLaMA 3/4 |
| Qwen | qwen2.py, qwen2_vl.py, qwen2_moe.py, qwen3.py, qwen3_vl.py | Qwen2/3 |
| DeepSeek | deepseek_v32.py, deepseek_v4.py | DeepSeek V3/V4 |
| Gemma | gemma.py, gemma2.py, gemma3.py, gemma3n.py, gemma4.py | Gemma 系列 |
| Mistral | mistral.py, mistral3.py, mixtral.py | Mistral/Mixtral |
| Phi | phi.py, phi3.py, phi3v.py, phi4mm.py | Phi 系列 |
| ChatGLM | chatglm.py, glm.py, glm4.py, glm4v.py | ChatGLM/GLM |
| **总计** | **120+ 文件** | **200+ 已注册架构** |

---

## 8. C++/CUDA 底层算子 (csrc/)

**路径**: `vllm/csrc/`

这是 vLLM 性能的关键所在，实现了所有高性能 GPU/CPU 内核。

### 8.1 整体架构

```
csrc/
├── attention/              # Attention 内核模板
│   ├── attention_generic.cuh
│   ├── dtype_bfloat16.cuh  # bf16 特化
│   ├── dtype_float16.cuh   # fp16 特化
│   ├── dtype_float32.cuh   # fp32 特化
│   └── dtype_fp8.cuh       # fp8 特化
├── libtorch_stable/        # Stable ABI 新版实现
│   ├── attention/          # 高级 Attention
│   │   ├── mla/            # Blackwell MLA 解码
│   │   └── merge_attn_states.cu
│   ├── quantization/       # 量化内核套件
│   │   ├── marlin/         # Marlin GEMM (GPTQ/AWQ)
│   │   ├── gptq/           # GPTQ (2/3/4/8-bit)
│   │   ├── awq/            # AWQ GEMM
│   │   ├── cutlass_w4a8/   # W4A8 CUTLASS
│   │   ├── w8a8/cutlass/   # W8A8（SM75-SM120 全覆盖）
│   │   ├── fp4/            # NVFP4/MXFP4
│   │   ├── machete/        # Hopper 混合精度 GEMM
│   │   └── fused_kernels/  # 融合量化
│   ├── moe/                # MoE 内核
│   │   ├── marlin_moe_wna16/
│   │   ├── grouped_topk_kernels.cu
│   │   ├── topk_softmax_kernels.cu
│   │   └── dsv3_router_gemm.cu
│   ├── cache_kernels.cu    # KV 缓存操作
│   ├── custom_all_reduce.cu # 自定义 AllReduce
│   ├── layernorm_kernels.cu
│   ├── pos_encoding_kernels.cu
│   ├── sampler.cu
│   └── activation_kernels.cu
├── cpu/                    # CPU 内核
│   ├── cpu_attn.cpp        # CPU Attention
│   ├── cpu_fused_moe.cpp   # CPU MoE
│   ├── micro_gemm/         # AMX/NEON/RVV 微 GEMM
│   └── sgl-kernels/        # SGL 内核
├── quantization/           # 量化工具
│   └── w8a8/fp8/           # FP8 工具（AMD/NVIDIA）
├── rocm/                   # AMD GPU 内核
│   ├── attention.cu
│   ├── q_gemm_rdna3.cu
│   └── moe_q_gemm_rdna3.cu
├── cutlass_extensions/     # CUTLASS 扩展
├── quickreduce/            # 快速 AllReduce
├── ops.h                   # 旧版 ABI op 声明
├── torch_bindings.cpp      # PyTorch 绑定入口
└── dispatch_utils.h        # 调度工具
```

### 8.2 支持的 GPU 架构

| 架构 | SM 版本 | 量化支持 |
|------|---------|---------|
| Turing | SM75 | INT8 |
| Ampere | SM80/SM89 | FP8, INT8, W4A8 |
| Hopper | SM90 | FP8, INT8, NVFP4, W4A8 |
| Blackwell | SM100/SM120 | NVFP4, MXFP4, MLA |
| RDNA3 | AMD | W4A16, FP8 |

### 8.3 主要算子分类

| 类别 | 算子 |
|------|------|
| **量化 GEMM** | marlin_gemm, gptq_gemm, awq_gemm, cutlass_scaled_mm, machete_mm, cutlass_w4a8_mm |
| **FP4** | cutlass_scaled_fp4_mm, cutlass_mxfp4_group_mm, nvfp4_scaled_mm |
| **归一化** | rms_norm, fused_add_rms_norm, rms_norm_dynamic_per_token_quant |
| **激活** | silu_and_mul, gelu_and_mul, fatrelu_and_mul, swigluoai_and_mul |
| **位置编码** | rotary_embedding, fused_qk_norm_rope |
| **缓存** | swap_blocks, reshape_and_cache, concat_and_cache_mla |
| **采样** | top_k_softmax, persistent_topk, cooperative_topk |
| **通信** | custom_all_reduce, init_custom_ar |
| **MoE** | cutlass_moe_mm, topk_softmax, grouped_topk |

---

## 9. 分布式通信 (distributed/)

**路径**: `vllm/vllm/distributed/`

| 文件/子目录 | 说明 |
|------------|------|
| `parallel_state.py` | 分布式状态管理（world_size, rank, TP/PP/DP 进程组） |
| `communication_op.py` | 通信算子（allreduce, broadcast 等） |
| `utils.py` | 分布式工具函数 |
| `kv_events.py` | KV 缓存事件发布（跨节点 KV 传输） |
| `nixl_utils.py` | NVIDIA NIXL 通信工具 |
| `device_communicators/` | 设备间通信实现（14 个文件） |
| ├── `pynccl.py` | NCCL 包装器 |
| ├── `custom_all_reduce.py` | 自定义 AllReduce |
| ├── `shm_broadcast.py` | 共享内存广播 |
| ├── `cuda_communicator.py` | CUDA 通信器 |
| ├── `cpu_communicator.py` | CPU 通信器 |
| ├── `xpu_communicator.py` | XPU 通信器 |
| ├── `ray_communicator.py` | Ray 通信器 |
| └── `flashinfer_all_reduce.py` | FlashInfer AllReduce |
| `ec_transfer/` | 纠删码传输（弹性推理） |
| `elastic_ep/` | 弹性专家并行 |
| `eplb/` | 专家并行负载均衡 |
| `kv_transfer/` | KV 缓存跨节点传输（PD 分离） |
| │ └── backends/ | mooncake, nixl, hf3fs, offloading, lmcache, flexkv, moriio |
| `weight_transfer/` | 权重传输（RL 训练权重同步） |

### 并行策略

| 策略 | 缩写 | 说明 |
|------|------|------|
| Tensor Parallelism | TP | 张量并行 |
| Pipeline Parallelism | PP | 流水线并行 |
| Data Parallelism | DP | 数据并行 |
| Expert Parallelism | EP | 专家并行 |
| Context Parallelism | CP | 上下文并行 |

---

## 10. 多模态处理 (multimodal/)

**路径**: `vllm/vllm/multimodal/`

```
multimodal/
├── inputs.py           # 输入类型（ImageItem, VideoItem, AudioItem 等）
├── parse.py            # 多模态数据解析器
├── registry.py         # 多模态处理器注册中心
├── audio.py            # 音频处理（重采样、声道缩减）
├── image.py            # 图像处理（缩放、EXIF、RGBA 转换）
├── video.py            # 视频处理（帧提取）
├── cache.py            # 处理器缓存
├── encoder_budget.py   # 编码预算估算
├── evs.py              # 高效视频摘要（token 剪枝）
├── gpu_ipc_memory.py   # GPU IPC 内存池
├── hasher.py           # 多模态数据哈希
├── utils.py            # 工具函数
├── media/              # 媒体 IO 层
│   ├── base.py         # MediaIO 抽象基类
│   ├── connector.py    # 媒体连接器（HTTP/HTTPS/Data URI）
│   ├── image.py        # 图像加载/编码
│   ├── audio.py        # 音频加载/编码
│   └── video.py        # 视频加载
└── processing/         # 多模态处理管线
    ├── context.py      # 处理上下文
    ├── dummy_inputs.py # 虚拟输入构建
    ├── inputs.py       # 处理器输入封装
    └── processor.py    # 核心多模态处理器
```

---

## 11. LoRA 适配器 (lora/)

**路径**: `vllm/vllm/lora/`

```
lora/
├── request.py          # LoRARequest（name, id, path）
├── lora_model.py       # LoRAModel 加载器
├── lora_weights.py     # LoRALayerWeights（lora_a, lora_b, scaling）
├── model_manager.py    # LoRAModelManager + LRU 缓存
├── worker_manager.py   # Worker 端 LoRA 管理
├── resolver.py         # LoRA 解析器（本地文件系统 / S3）
├── peft_helper.py      # PEFT 配置解析
├── utils.py            # 模块替换、目标匹配
├── layers/             # LoRA 层实现
│   ├── base.py         # BaseLayerWithLoRA 基类
│   ├── base_linear.py  # Column/Row Parallel LoRA
│   ├── fused_moe.py    # MoE LoRA
│   ├── logits_processor.py
│   └── utils.py
├── ops/                # LoRA 算子
│   ├── torch_ops/      # PyTorch 实现（shrink, expand）
│   ├── triton_ops/     # Triton 实现（含 FP8 变体）
│   └── xpu_ops/        # XPU 实现
└── punica_wrapper/     # Punica 包装器（GPU/CPU/XPU）
```

---

## 12. 推理和工具解析器

vLLM 对主流模型提供了完善的推理内容提取和工具调用解析，分为三个层次：

### 12.1 推理解析器 (reasoning/)

**路径**: `vllm/vllm/reasoning/`

| 文件 | 模型 |
|------|------|
| `abs_reasoning_parsers.py` | `ReasoningParser` 抽象基类 |
| `basic_parsers.py` | 基础实现 |
| `deepseek_r1_reasoning_parser.py` | DeepSeek-R1 |
| `deepseek_v3_reasoning_parser.py` | DeepSeek-V3 |
| `deepseek_v4_engine_reasoning_parser.py` | DeepSeek-V4 |
| `qwen3_engine_reasoning_parser.py` | Qwen3 |
| `gemma4_engine_reasoning_parser.py` | Gemma 4 |
| `minimax_m3_reasoning_parser.py` | MiniMax M3 |
| `mistral_reasoning_parser.py` | Mistral |
| `olmo3_reasoning_parser.py` | OLMo 3 |
| `kimi_k2_reasoning_parser.py` | Kimi K2 |
| `step3_reasoning_parser.py`, `step3p5_reasoning_parser.py` | Step 3/3.5 |
| `cohere_command_reasoning_parser.py` | Cohere Command |
| `granite_reasoning_parser.py` | Granite |
| `internlm2_reasoning_parser.py` | InternLM 2 |
| `nemotron_v3_engine_reasoning_parser.py` | Nemotron V3 |
| `poolside_v1_reasoning_parser.py` | Poolside V1 |
| ... (共 20+ 个解析器) | |

### 12.2 工具调用解析器 (tool_parsers/)

**路径**: `vllm/vllm/tool_parsers/`

| 文件 | 模型 |
|------|------|
| `abstract_tool_parser.py` | `ToolParser` 抽象基类 |
| `llama_tool_parser.py` | LLaMA |
| `hermes_tool_parser.py` | Hermes |
| `mistral_tool_parser.py` | Mistral |
| `deepseekv3_tool_parser.py` | DeepSeek-V3 |
| `deepseekv32_engine_tool_parser.py` | DeepSeek-V3.2 |
| `deepseekv4_engine_tool_parser.py` | DeepSeek-V4 |
| `qwen3_engine_tool_parser.py` | Qwen3 |
| `gemma4_engine_tool_parser.py` | Gemma 4 |
| `glm47_moe_tool_parser.py` | GLM-4-7B MoE |
| `granite_tool_parser.py`, `granite4_tool_parser.py` | Granite |
| `kimi_k2_tool_parser.py` | Kimi K2 |
| `minimax_m2_tool_parser.py`, `minimax_m3_tool_parser.py` | MiniMax |
| `olmo3_tool_parser.py` | OLMo 3 |
| `step3_tool_parser.py`, `step3p5_tool_parser.py` | Step |
| `xlam_tool_parser.py` | XLAM |
| `rust_tool_parser.py` | Rust 风格 |
| `pythonic_tool_parser.py` | Pythonic 风格 |
| ... (共 30+ 个解析器) | |

### 12.3 结构化输出解析器 (parser/)

**路径**: `vllm/vllm/parser/`

```
parser/
├── abstract_parser.py  # Parser 抽象基类
├── parser_manager.py   # 统一解析器管理器
├── harmony.py          # OpenAI Harmony 协议
├── utils.py            # 工具函数
├── metrics.py          # 性能指标
├── deepseek_v32.py     # DeepSeek-V3.2
├── deepseek_v4.py      # DeepSeek-V4
├── gemma4.py           # Gemma 4
├── qwen3.py            # Qwen3
├── mistral.py          # Mistral
├── kimi_k2.py          # Kimi K2
├── engine/             # 流式解析引擎
│   ├── events.py       # 语义事件定义
│   ├── incremental_lexer.py # 增量词法分析
│   ├── parser_engine.py # 核心解析引擎
│   ├── token_id_scanner.py # Token ID 扫描
│   └── adapters.py     # 模型适配器
└── ...
```

---

## 13. 分词器 (tokenizers/)

**路径**: `vllm/vllm/tokenizers/`

| 文件 | 说明 |
|------|------|
| `protocol.py` | `TokenizerLike` 协议接口 |
| `registry.py` | 分词器注册中心 |
| `hf.py` | HuggingFace 分词器（线程安全池化） |
| `mistral.py` | Mistral 分词器 |
| `deepseek_v32.py` | DeepSeek-V3.2 分词器 |
| `deepseek_v32_encoding.py` | DeepSeek-V3.2 编码实现 |
| `deepseek_v4.py` | DeepSeek-V4 分词器 |
| `deepseek_v4_encoding.py` | DeepSeek-V4 编码（支持 tool calling / thinking mode） |
| `fastokens.py` | fastokens 后端补丁（替换 Rust 分词器） |
| `detokenizer_utils.py` | 解码工具函数 |
| `kimi_audio.py` | Kimi 音频分词器 |

---

## 14. 编译优化 (compilation/)

**路径**: `vllm/vllm/compilation/`

### 14.1 编译系统结构

```
compilation/
├── backends.py             # torch.compile 后端封装
├── compiler_interface.py   # 编译器接口抽象
├── caching.py              # 编译缓存
├── codegen.py              # 执行代码生成
├── cuda_graph.py           # CUDA Graph 捕获/重放
├── breakable_cudagraph.py  # 可中断 CUDA Graph
├── base_static_graph.py    # 静态图基类
├── wrapper.py              # 编译包装器
├── decorators.py           # 编译装饰器
├── partition_rules.py      # 分区规则
├── piecewise_backend.py    # 分段编译后端
└── passes/                 # 编译 passes
    ├── fusion/             # 算子融合 passes
    │   ├── act_quant_fusion.py       # Activation + Quantization
    │   ├── allreduce_rms_fusion.py   # AllReduce + RMS Norm
    │   ├── attn_quant_fusion.py      # Attention + Quantization
    │   ├── collective_fusion.py      # 集合通信融合
    │   ├── mla_attn_quant_fusion.py  # MLA + Quantization
    │   ├── mla_rope_kvcache_cat_fusion.py
    │   ├── qk_norm_rope_fusion.py
    │   ├── rms_quant_fusion.py
    │   ├── rocm_aiter_fusion.py
    │   ├── rope_kvcache_fusion.py
    │   └── sequence_parallelism.py
    ├── ir/                 # IR 级别 passes
    │   ├── clone_elimination.py
    │   ├── inplace_functionalization.py
    │   └── lowering_pass.py
    └── utility/            # 工具 passes
        ├── noop_elimination.py
        ├── split_coalescing.py
        └── post_cleanup.py
```

### 14.2 核心功能

- **torch.compile**: 支持 eager + inductor 混合编译
- **CUDA Graph**: 支持完整图和可中断图
- **算子融合**: 10+ 种融合模式（Act+Quant, AllReduce+RMSNorm, QK Norm+RoPE 等）
- **IR 优化**: 消除冗余 clone、inplace 函数化、IR 降低

---

## 15. 硬件平台抽象 (platforms/)

**路径**: `vllm/vllm/platforms/`

| 文件 | 平台 | 说明 |
|------|------|------|
| `interface.py` | — | `Platform` 抽象基类，定义设备管理、注意力、内存接口 |
| `cuda.py` | NVIDIA GPU | `CudaPlatform` |
| `rocm.py` | AMD GPU | `RocmPlatform` |
| `tpu.py` | Google TPU | `TpuPlatform` |
| `xpu.py` | Intel XPU | `XPUPlatform` |
| `cpu.py` | 通用 CPU | `CpuPlatform` |
| `zen_cpu.py` | AMD Zen CPU | `ZenCpuPlatform`（AVX-512 + zenT orch） |
| `__init__.py` | — | **自动检测**: 运行时检测 CUDA/ROCm/TPU/XPU/CPU |

### 自动检测流程

```
检查插件 → 尝试 TPU → CUDA → ROCm → XPU → CPU
```

---

## 16. 测试体系 (tests/)

**路径**: `vllm/tests/`

```
tests/
├── basic_correctness/  # 基础正确性（CPU offload, 内存）
├── benchmarks/         # 基准测试（延迟、吞吐量、参数扫描）
├── compile/            # torch.compile 测试
│   ├── correctness_e2e/
│   ├── fullgraph/
│   ├── fusions_e2e/
│   └── passes/
├── config/             # 配置测试
├── cuda/               # CUDA 特定测试
├── detokenizer/        # 分词器测试
├── distributed/        # 分布式测试（20+ 文件）
├── engine/             # 引擎测试
├── entrypoints/        # API 入口测试
├── evals/              # 模型评估（GPQA, GSM8K, MRCR）
├── kernels/            # 内核测试（最丰富）
│   ├── attention/      # FlashAttention, FlashInfer, MLA 等
│   ├── core/           # RoPE, LayerNorm, 激活等
│   ├── moe/            # MoE 内核（15+ 文件）
│   └── quantization/   # 量化内核测试
├── lora/               # LoRA 测试
├── models/             # 模型测试
├── multimodal/         # 多模态测试
├── parser/             # 解析器测试
├── quantization/       # 量化集成测试
├── reasoning/          # 推理解析器测试
└── samplers/           # 采样器测试
```

---

## 17. 工具模块汇总 (utils/ 等)

### 17.1 utils/ 通用工具库

**路径**: `vllm/vllm/utils/`

| 文件 | 功能 |
|------|------|
| `async_utils.py` | 异步工具（resolve, merge_async_iterators） |
| `cache.py` | 通用缓存（LRU） |
| `collection_utils.py` | 集合工具（is_list_of, groupby） |
| `func_utils.py` | 函数工具 |
| `gc_utils.py` | 垃圾回收 |
| `hashing.py` | 哈希工具 |
| `import_utils.py` | 导入工具（LazyLoader, resolve_obj_by_qualname） |
| `jsontree.py` | JSON 树操作 |
| `math_utils.py` | 数学工具 |
| `mem_utils.py` | 内存工具 |
| `nccl.py` | NCCL 工具 |
| `network_utils.py` | 网络工具（端口检测、IP 验证） |
| `numa_utils.py` | NUMA 工具 |
| `registry.py` | 扩展管理器（插件注册框架） |
| `torch_utils.py` | PyTorch 工具 |
| `hpc.py` | HPC 工具（CPU pinning） |

### 17.2 其他工具模块

| 模块 | 功能 |
|------|------|
| `triton_utils/` | Triton 内存分配、强制首次配置 |
| `transformers_utils/` | HuggingFace Transformers 工具（配置、处理器、S3） |
| `logging_utils/` | 访问日志过滤、输入转储、计时 |
| `tracing/` | OpenTelemetry 链路追踪 |
| `profiler/` | 逐层性能分析 |
| `ir/` | 中间表示（算子定义、容差） |
| `cute_utils/` | CuTe（CUDA 模板库）工具 |
| `plugins/` | 插件系统（IO 处理器、LoRA 解析器） |

---

## 18. 架构总图

### 18.1 数据流

```
用户
  │
  ├── Python API: entrypoints/llm.py (LLM 类)
  │
  ├── HTTP API: entrypoints/openai/ (FastAPI)
  │   ├── OpenAI Chat/Completions API
  │   ├── Anthropic Messages API
  │   └── gRPC Server
  │
  └── CLI: entrypoints/cli/ (vllm serve)
        │
        ▼
  engine/protocol.py (EngineClient 接口)
        │
        ▼
  v1/engine/ (V1 引擎核心)
        │
        ├── v1/engine/core.py (ZeroMQ IPC 核心循环)
        │       │
        │       ├── v1/core/sched/ (调度器)
        │       ├── v1/worker/ (模型运行器)
        │       │       │
        │       │       ├── model_executor/models/ (120+ 模型)
        │       │       ├── model_executor/layers/ (注意力/MoE/归一化)
        │       │       └── model_executor/kernels/ (高性能内核)
        │       │       │       └── csrc/ (C++/CUDA 底层算子)
        │       │       │
        │       │       └── v1/spec_decode/ (推测解码)
        │       │
        │       ├── v1/sample/ (采样器)
        │       ├── v1/attention/ (可插拔注意力后端)
        │       └── v1/kv_offload/ (KV 缓存卸载)
        │
        └── v1/executor/ (分布式执行器)
                │
                └── distributed/ (分布式通信)
```

### 18.2 模块依赖关系

```
┌─────────────────────────────────────────────────────────────┐
│                    config/ (VllmConfig)                      │
│                  全系统统一配置容器                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 platforms/ (硬件平台抽象)                    │
│        CUDA / ROCm / TPU / XPU / CPU 自动检测              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────┐   ┌────────────┐   ┌─────────────────────────┐
│ entrypoints│──▶│ v1/engine  │──▶│  v1/worker + model_exec │
│ (API入口)  │   │ (引擎核心)  │   │  (模型加载/执行)        │
└────────────┘   └─────┬──────┘   └──────────┬──────────────┘
                       │                     │
                       ▼                     ▼
              ┌────────────────┐   ┌──────────────────────────┐
              │ v1/executor    │   │  csrc/ (C++/CUDA 算子)   │
              │ (分布式执行器)  │   │  ┌────────────────────┐  │
              └───────┬────────┘   │  │ attention/         │  │
                      │            │  │ quantization/      │  │
                      ▼            │  │ moe/               │  │
              ┌────────────────┐   │  │ cpu/               │  │
              │ distributed/   │   │  │ rocm/              │  │
              │ (通信)         │   │  │ cutlass_extensions/│  │
              └────────────────┘   │  └────────────────────┘  │
                                   └──────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    辅助模块 (可插拔)                         │
│  multimodal/ │  lora/ │  parser/ │  tool_parsers/          │
│  reasoning/  │  compilation/  │  tokenizers/              │
│  inputs/ │  outputs/ │  sampling_params/                   │
└─────────────────────────────────────────────────────────────┘
```

### 18.3 核心设计要点总结

| 方面 | 设计决策 |
|------|---------|
| **引擎架构** | V1 基于 ZeroMQ IPC，支持同步/异步引擎 |
| **调度** | 全新调度器支持连续批处理、chunked prefill、prefix caching |
| **注意力** | PagedAttention + 可插拔后端（FlashAttention/FlashInfer/Triton/MLA） |
| **量化** | 10+ 种量化方案，覆盖 INT4/INT8/FP8/FP4 |
| **分布式** | TP/PP/DP/EP/CP 全面支持，自定义 AllReduce |
| **模型支持** | 120+ 模型实现，200+ 已注册架构 |
| **多模态** | 图片/音频/视频全面支持，高效视频 token 剪枝 |
| **服务** | OpenAI/Anthropic/MCP/gRPC 多协议入口 |
| **平台** | 自动检测 CUDA/ROCm/TPU/XPU/CPU |
| **编译** | torch.compile + CUDA Graph + 10+ 算子融合 |
| **推测解码** | EAGLE/Medusa/dFlash/N-gram/Suffix 多种策略 |
