# Post-Training 论文报告生成脚手架

精简 RAG 方案：**摘要卡 + 大纲驱动**，替代 GraphRAG + 多 Agent。适合 ~40 篇论文规模。

## 架构

```
PDF (Info/) ──Docling──▶ Markdown ──LLM──▶ 摘要卡(JSON) ──┐
                                                          ├─ 大纲 → 分章检索写作 → 合成报告
Markdown chunks ──SentenceTransformer──▶ numpy 向量库 ──┘
```

## 三类报告

| 类型 | 策略 |
|---|---|
| 技术截面报告 | 摘要卡按 `category` 聚类，每类一章 |
| 技术演进报告 | 按 `year` 排序，沿 `improves_over` 字段画演进链 |
| 学习路线报告 | 用 `prerequisites` 字段做拓扑排序，前置依赖先讲 |

## 快速开始

```powershell
cd d:\objects\LLM\llm_note\post_training\rag_scaffold

# 1. 安装依赖（uv 会自动创建 .venv）
uv sync

# 2. 配置 LLM API Key
cp .env.example .env
# 编辑 .env 填入 LLM_API_KEY

# 3. 建库（解析 PDF + 生成摘要卡 + 入向量库，约 30-60 分钟）
uv run python build_index.py

# 4. 生成报告
uv run python generate_report.py --type evolution --need "梳理 2023-2026 年对齐技术从 DPO 到 GRPO 的演进脉络及源码实现思路"
uv run python generate_report.py --type cross_section --need "对比当前主流偏好优化方法 (DPO/SimPO/KTO/ORPO)"
uv run python generate_report.py --type learning_path --need "为零基础工程师设计大模型后训练学习路线"
```

## 目录结构

```
rag_scaffold/
├── pyproject.toml          # uv 依赖
├── .env                    # LLM 配置（自行创建）
├── config.py               # 路径与参数
├── llm_client.py           # OpenAI 兼容客户端
├── prompts.py              # 4 个 Prompt 模板
├── utils.py                # 文件名解析 / Markdown 切分
├── vector_store.py         # 轻量向量库 (sentence-transformers + numpy)
├── build_index.py          # 阶段 A：建库
├── generate_report.py      # 阶段 B：报告生成
└── data/                   # 自动生成
    ├── markdown/           # Docling 输出
    ├── cards/              # 摘要卡 JSON
    ├── vector_store/       # 向量库 (.npy + .json)
    └── reports/            # 生成的报告
```

## 设计要点

1. **摘要卡替代 GraphRAG**：每篇论文一张 JSON 卡，含 `improves_over` / `prerequisites` 字段。40 张卡可整体塞进 LLM 上下文做全局规划，无需图数据库。
2. **大纲驱动而非检索驱动**：先生成带论文引用约束的大纲，再分章检索写作，避免传统 RAG 的碎片化。
3. **元数据硬过滤**：文件名年份作为 Chroma metadata，演进报告按年份排序检索。
4. **缓存可断点续跑**：Markdown 和摘要卡都落盘，重跑只处理缺失项。

## 故障排查

- **Docling 解析慢/失败**：首次运行会下载模型 (~1GB)，后续走缓存。个别 PDF 失败会跳过并记录到 `data/failed.txt`。
- **LLM 返回非 JSON**：`llm_client.py` 已加重试 + JSON 提取，仍失败时检查模型是否支持 JSON mode。
- **嵌入模型下载慢**：`paraphrase-multilingual-MiniLM-L12-v2` 约 120MB，可换 `BAAI/bge-m3`（在 `config.py` 改）质量更好但更大。
- **向量库说明**：用 sentence-transformers + numpy 自建（`vector_store.py`），接口对齐 chromadb 子集。40 篇规模（~几千 chunks）余弦检索足够快，且避免了 chromadb 在 Windows + uv 下的 pywin32 wheel 问题。如需扩展到上千篇，可换回 chromadb 或 faiss。
