# 文档提取、切分、向量化问题反思

> 基于 43 篇 post-training 论文（PDF）的实际建库过程，记录遇到的问题、根因分析与解决方案。

---

## 一、依赖安装问题

### 1.1 transformers 版本过低导致 ImportError

**现象**：
```
ImportError: cannot import name 'AutoModelForImageTextToText' from 'transformers'
```

**根因**：docling 2.110.0 依赖较新的 transformers（含 `AutoModelForImageTextToText`），但 `pyproject.toml` 只写了 `docling>=2.5.0`，依赖解析时装了旧版 transformers。

**解决**：手动升级 `transformers>=4.50.0`。

**反思**：`pyproject.toml` 对 docling 的版本约束过松（`>=2.5.0`），应固定到实际测试过的版本（如 `docling==2.110.0`），避免依赖链上的间接版本漂移。

### 1.2 sympy/mpmath 依赖断裂

**现象**：
```
ImportError: SymPy now depends on mpmath as an external library.
```

**根因**：torch → sympy → mpmath 依赖链中，mpmath 未被自动拉入 venv。`uv pip install mpmath` 显示"Checked 1 package"（缓存命中）但实际没装进 venv。

**解决**：`uv pip install sympy mpmath --force-reinstall` 强制重装。

**反思**：uv 的缓存机制有时会"假命中"——缓存里有但 venv 里没装上。遇到 `Checked 1 package in 5ms` 但问题仍在时，必须加 `--force-reinstall`。

### 1.3 跨盘缓存导致安装极慢

**现象**：`uv sync` 跑了 40+ 分钟仍在 copy 阶段。

**根因**：uv 默认缓存在 C 盘（`%LOCALAPPDATA%\uv\cache`），但 venv 在 D 盘。跨盘 copy ~3GB 依赖（torch + docling + sentence-transformers）需要逐文件复制，而同盘则可用硬链接秒级完成。

**解决**：设置 `UV_CACHE_DIR=D:\objects\.uv_cache`（注意 D:\ 根目录无写入权限，需用子目录）。

**反思**：大型依赖（torch ~800MB）务必让缓存和 venv 同盘。小型项目（~100MB 依赖）跨盘无感知，大型项目暴露问题。

---

## 二、PDF 解析问题（Docling）

### 2.1 std::bad_alloc 内存崩溃

**现象**：
```
Stage preprocess failed for run 1, pages [16]: std::bad_alloc
Stage preprocess failed for run 1, pages [17]: std::bad_alloc
...
Producer failed for run 1: [WinError -536870904] Windows Error 0xe0000008
```

**根因**：Docling 的深度学习 pipeline 对高分辨率页面预处理时内存分配失败。91 页的长论文尤其严重——前 15 页跑完后内存耗尽，后续每页都 bad_alloc。

**演进解决过程**：

| 尝试 | 代码 | 结果 |
|---|---|---|
| 第 1 次 | `except (RuntimeError, MemoryError)` 捕获 | 失败——实际抛的是 `OSError`，没捕获到 |
| 第 2 次 | `except (RuntimeError, MemoryError, OSError)` | 捕获了，但 Docling 在抛异常前大量打印 bad_alloc 日志，耗时很长 |
| 第 3 次 | 预检：>10MB 或 >30 页直接跳过 Docling，用 PyMuPDF | 成功——彻底避免 OOM |

**反思**：
1. `std::bad_alloc` 是 C++ 层的异常，Python 层包装成 `OSError` 而非 `RuntimeError`。异常类型预判错误导致第一次修复无效。
2. **预防优于兜底**：与其等 Docling 崩了再回退，不如先判断 PDF 大小/页数直接选路径。预检 + 兜底双保险才可靠。

### 2.2 PdfPipelineOptions API 变更

**现象**：
```
'PdfPipelineOptions' object has no attribute 'backend'
```

**根因**：尝试用 `PdfPipelineOptions(do_ocr=False)` 关闭 OCR 以消除 `RapidOCR returned empty result!` 警告，但 docling 2.110.0 的 API 已变更，`PdfPipelineOptions` 的构造方式与文档不符。

**解决**：去掉自定义 pipeline_options，用默认 `DocumentConverter()`。OCR 警告是非致命的（只是日志噪音），不影响文本提取。

**反思**：不要为了消除无害的警告而引入新的依赖风险。优先用默认配置，只在确有性能/质量影响时才调整。

### 2.3 PyMuPDF 兜底丢失章节结构

**现象**：大 PDF（>10MB 或 >30 页）走 PyMuPDF 兜底后，`split_markdown_by_header()` 找不到 `#` header，整篇被当一个 chunk 滑窗切成 1200 字符段，丢失章节边界。

**根因**：PyMuPDF 的 `page.get_text()` 返回纯文本，不生成 Markdown header。

**影响**：检索质量下降——用户查询"DPO 损失函数"时，可能命中的是包含多个不相关段落的 1200 字符滑窗，而非干净的"DPO 方法"章节。

**未解决**：当前接受这个降级。如果后续需要改善，可以用 PyMuPDF 的 `get_text("dict")` 提取字体大小信息，推断标题层级后手动补 `#` header。

---

## 三、向量库问题

### 3.1 ChromaDB 在 Windows + uv 下不可用

**现象**：chromadb 间接依赖的 pywin32 wheel 校验失败。

**解决**：放弃 chromadb，用 sentence-transformers + numpy 自建轻量向量库（`vector_store.py`），接口对齐 chromadb 子集（add/query/get/delete/count）。

**反思**：40 篇论文 ~2600 chunks 的规模下，numpy 余弦相似度检索完全够用（全量矩阵乘法 <100ms）。chromadb 的 HNSW 索引在千级规模下没有优势，反而引入了 pywin32/SQLite/ONNX 等额外依赖风险。**工具选择应匹配数据规模**。

### 3.2 缓存的摘要卡缺少 category 字段

**现象**：向量库 metadata 里部分 chunk 的 category 为 "other"，但实际论文应该有更精确的分类。

**根因**：`build_index.py` 的 `extract_card()` 对新提取的 card 用 `setdefault` 兜底字段，但**缓存命中的旧 card 直接 `json.loads` 使用，不走 setdefault**。早期 Prompt 版本生成的 card 可能缺 category 字段。

**解决思路**（未改代码）：删除旧 card 缓存重新提取，或在缓存命中分支也加 setdefault 兜底。

**反思**：缓存读取路径和数据生成路径的字段保证逻辑不一致，是典型的缓存陷阱。应该统一用 `normalize_card(card)` 函数处理两条路径的输出。

---

## 四、切分策略问题

### 4.1 当前切分策略

```python
split_markdown_by_header(md):
  1. 按 # / ## / ### header 切 section
  2. section <= 1200 字符 → 直接作为一个 chunk
  3. section > 1200 字符 → 滑窗切（1200 字符窗口，150 字符重叠）
```

### 4.2 存在的问题

| 问题 | 影响 | 严重程度 |
|---|---|---|
| header 层级不区分：`##` 和 `###` 都触发切分 | `###` 子节与其父 `##` 节分离，丢失上下文 | 中 |
| 滑窗切分在段落中间断开 | chunk 可能从段落中间开始或结束，语义不完整 | 中 |
| PyMuPDF 兜底的纯文本无 header | 整篇当一个 section，滑窗切成无结构碎片 | 高（影响 ~30% 的论文） |

### 4.3 改进方向（未实施）

- 按层级切分：只在 `##` 级别切，`###` 级别保留在父 chunk 内
- 段落感知切分：滑窗在段落边界处对齐，不在段落中间断开
- PyMuPDF 兜底时用字体大小推断标题：`page.get_text("dict")` 返回字体信息

---

## 五、问题统计

### 5.1 建库结果

| 指标 | 数值 |
|---|---|
| PDF 总数 | 43 |
| 成功建库 | 43（全部成功） |
| Docling 解析 | ~30 篇（<10MB 且 ≤30 页的） |
| PyMuPDF 兜底 | ~13 篇（大 PDF 或多页 PDF） |
| 生成 chunks | 2616 |
| 生成摘要卡 | 43 |

### 5.2 问题分类

| 类别 | 问题数 | 已解决 | 未解决 |
|---|---|---|---|
| 依赖安装 | 3 | 3 | 0 |
| PDF 解析 | 3 | 2 | 1（PyMuPDF 丢结构） |
| 向量库 | 2 | 1 | 1（card 字段不一致） |
| 切分策略 | 3 | 0 | 3（改进方向） |

---

## 六、经验总结

### 6.1 依赖管理

1. **固定关键依赖版本**：docling/torch/transformers 的 API 在小版本间就会变，`>=` 约束不够安全
2. **大型依赖同盘安装**：torch ~800MB 必须走硬链接，跨盘 copy 是性能杀手
3. **uv 缓存假命中**：`Checked 1 package in 5ms` 不代表真的装上了，必要时 `--force-reinstall`

### 6.2 异常处理

1. **C++ 异常的 Python 包装类型不可预判**：`std::bad_alloc` 可能是 `RuntimeError`/`OSError`/`MemoryError`，用 `except Exception` 兜底最安全
2. **预防优于兜底**：预检（文件大小/页数）比异常捕获更可靠——避免 Docling 在崩溃前浪费大量时间打印日志
3. **兜底方案要考虑数据质量降级**：PyMuPDF 兜底成功但不保留 Markdown 结构，影响下游切分

### 6.3 缓存设计

1. **缓存读取和生成路径的字段保证要一致**：新数据走 `setdefault`，旧缓存也要走同一套 normalize 逻辑
2. **缓存失效策略要清晰**：改了 Prompt/解析逻辑后，旧缓存可能过期，需要有选择性地清除（删 .md → 重解析，删 .json → 重抽卡）

### 6.4 工具选择

1. **匹配数据规模**：40 篇论文不需要 chromadb/GraphRAG/Neo4j，numpy + 摘要卡足够
2. **减少依赖链深度**：chromadb → pywin32 → Windows wheel 问题，自建 numpy 向量库零外部依赖
3. **默认配置优先**：不要为消除无害警告（如 RapidOCR empty result）而调整 API，引入新风险
