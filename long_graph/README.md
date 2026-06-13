# long_graph

## Agent 生成文件说明

本目录下以下文件由 AI Agent 自动生成：

| 文件 | 说明 |
|------|------|
| `learning.md` | LangGraph 专业学习指南（20 节，约 1450 行） |
| `log.md` | 学习指南迭代评分日志（第1轮 → 第2轮） |

## 生成环境

- **模型**：DeepSeek v4 Pro
- **Token 消耗**：较多，两百万左右
- **运行环境**：TRAE IDE（原生）
- **工作流程**：
  - 查阅 LangGraph 官方文档及社区资料
  - 生成初版 → 隔离打分（7.5/10，不合格）
  - 根据不足迭代 → 再打分（9.2/10，合格）
  - 第1轮不足记入 `log.md`，终版写入 `learning.md`
- **循环与检查**：全部由 TRAE 原生完成（2 轮生成-评分-迭代循环）

## 目录结构

```
long_graph/
├── AGENTS.md      # 任务定义（人工编写）
├── README.md       # 本文件（Agent 生成）
├── learning.md     # 学习指南（Agent 生成）
└── log.md          # 评分日志（Agent 生成）
```
