# LangGraph 专业学习指南

> **定位**：LangGraph 是由 LangChain 团队开发的开源低层级编排框架与运行时，专为构建、管理和部署长时间运行的、有状态的 AI Agent 而设计。
> **版本基准**：v1.x（2025年10月发布 v1.0），包含 LangSmith Deployment（原 LangGraph Platform）GA
> **官方文档**：https://docs.langchain.com/oss/python/langgraph/overview

---

## 目录

1. [LangGraph 概述与生态定位](#1-langgraph-概述与生态定位)
2. [核心理念：用图思考 Agent](#2-核心理念用图思考-agent)
3. [两种 API：Graph API 与 Functional API](#3-两种-api-graph-api-与-functional-api)
4. [快速入门：构建第一个 Agent](#4-快速入门构建第一个-agent)
5. [State（状态）深度解析](#5-state状态深度解析)
6. [Nodes 与 Edges 详解](#6-nodes-与-edges-详解)
7. [工作流与 Agent 设计模式](#7-工作流与-agent-设计模式)
8. [持久化与记忆系统](#8-持久化与记忆系统)
9. [Human-in-the-Loop（人机协同）](#9-human-in-the-loop人机协同)
10. [Subgraphs 与 Multi-Agent 系统](#10-subgraphs-与-multi-agent-系统)
11. [LangGraph + GraphRAG：知识图谱增强](#11-langgraph--graphrag知识图谱增强)
12. [Streaming（流式处理）](#12-streaming流式处理)
13. [容错与错误处理](#13-容错与错误处理)
14. [安全与护栏（Guardrails）](#14-安全与护栏guardrails)
15. [测试与评估（Testing & Evaluation）](#15-测试与评估testing--evaluation)
16. [性能优化与成本控制](#16-性能优化与成本控制)
17. [LangGraph Platform / Agent Server](#17-langgraph-platform--agent-server)
18. [生产部署](#18-生产部署)
19. [监控与可观测性（LangSmith）](#19-监控与可观测性langsmith)
20. [最佳实践](#20-最佳实践)

---

## 1. LangGraph 概述与生态定位

### 1.1 LangGraph 是什么

LangGraph 是一个**低层级的 Agent 编排运行时**，它让你像画流程图一样定义 AI Agent 的行为。每个步骤是一个节点（Node），步骤之间的决策和跳转是边（Edge），所有节点通过共享状态（State）通信。

它由 LangChain 团队开发，灵感来自 Google 的 **Pregel**（图计算框架）和 **Apache Beam**（数据处理管道），其公开接口受 **NetworkX** 影响。

### 1.2 LangChain 生态中的位置

```
┌──────────────────────────────────────────────────┐
│          Deep Agents (Agent Harness)               │  ← 规划、子Agent、文件系统、上下文管理
├──────────────────────────────────────────────────┤
│          LangChain (Agent Framework)               │  ← 模型集成、工具、Agent 循环
├──────────────────────────────────────────────────┤
│          LangGraph (Orchestration Runtime)         │  ← 持久执行、流式、HITL、持久化、子图
├──────────────────────────────────────────────────┤
│          LangSmith (Platform)                      │  ← Tracing、Evaluation、Deployment
├──────────────────────────────────────────────────┤
│          LangSmith Engine + Fleet                  │  ← 自动问题检测、无代码 Agent 构建
└──────────────────────────────────────────────────┘
```

- **LangGraph 不依赖 LangChain** — 可以与原生 OpenAI/Anthropic 客户端配合使用
- **LangGraph Platform** — 已重命名为 LangSmith Deployment，提供托管与自托管两种部署方式

### 1.3 核心优势

| 能力 | 说明 |
|------|------|
| **持久执行（Durable Execution）** | Agent 崩溃后从中断点自动恢复 |
| **Human-in-the-Loop** | 任意节点前后插入人工审批/修改，通过 `interrupt()` 函数 |
| **全面记忆** | Checkpointer（短期/线程级）+ Store（长期/跨线程）双层架构 |
| **流式处理** | 7 种流模式：values / updates / messages / custom / checkpoints / tasks / debug |
| **生产级部署** | 自托管 Docker/K8s + 云部署，1-Click GitHub 集成 |
| **容错机制** | RetryPolicy、Durability Mode（exit/async/sync）、error_handler |

### 1.4 谁在用

Klarna、Uber、J.P. Morgan、Replit、LinkedIn、GitLab、Elastic、Elix 等企业，近 400 家公司在生产环境中使用。

---

## 2. 核心理念：用图思考 Agent

### 2.1 设计思维五步法

**Step 1: 分解工作流**
- 每个步骤 = 一个节点，画出连线

**Step 2: 识别步骤类型**
- **LLM 步骤**：理解、分析、生成、推理
- **数据步骤**：检索外部信息
- **动作步骤**：外部操作（发邮件、创工单）
- **用户输入步骤**：人工介入

**Step 3: 设计 State**
- 原则：**State 中存储原始数据，Prompt 格式化在节点内按需完成**

**Step 4: 构建节点函数**
- 每个节点：`(state) -> state_update` 或 `-> Command(goto=...)`
- 分类处理错误

**Step 5: 串联组装**
- 添加节点 → 添加边 → 条件边 → 编译

### 2.2 状态设计核心原则

> **在 State 中存储原始数据，在节点内部按需格式化**

```
State 中应存储：                   State 中不应存储：
├── 原始邮件内容                    ├── 格式化后的 Prompt 文本
├── 分类结果（结构化 dict）          ├── 拼接后的上下文字符串
├── 搜索返回的原始文本                ├── 模板
├── 草稿回复文本                    └── 指令
├── 执行元数据
└── 循环计数
```

---

## 3. 两种 API：Graph API 与 Functional API

### 3.1 Graph API

通过显式定义节点和边构建工作流。编译器支持 `checkpointer`, `store`, `interrupt_before/after`。

```python
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition

class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int

builder = StateGraph(MessagesState)
builder.add_node("llm_call", llm_call)
builder.add_node("tool_node", ToolNode(tools))
builder.add_edge(START, "llm_call")
builder.add_conditional_edges("llm_call", tools_condition)
builder.add_edge("tool_node", "llm_call")
agent = builder.compile(checkpointer=checkpointer)
```

### 3.2 Functional API

通过 `@entrypoint` 和 `@task` 装饰器，使用标准 Python 控制流。

```python
from langgraph.func import entrypoint, task
from langgraph.graph import add_messages

@task
def call_llm(messages: list[BaseMessage]):
    return model_with_tools.invoke(messages)

@task
def call_tool(tool_call: ToolCall):
    return tools_by_name[tool_call["name"]].invoke(tool_call)

@entrypoint()
def agent(messages: list[BaseMessage]):
    response = call_llm(messages).result()
    while response.tool_calls:
        results = [call_tool(tc).result() for tc in response.tool_calls]
        messages = add_messages(messages, [response, *results])
        response = call_llm(messages).result()
    return add_messages(messages, response)
```

### 3.3 选择建议

| 场景 | 推荐 API |
|------|---------|
| 复杂分支、多路径、子图、Multi-Agent | Graph API |
| HITL 嵌入式控制 | Graph API |
| 简单工具调用循环 | Functional API |
| 传统 Python 风格 | Functional API |

---

## 4. 快速入门：构建第一个 Agent

### 4.1 安装

```bash
pip install -U langgraph langchain langchain-openai
# 生产环境：
pip install langgraph-checkpoint-sqlite  # 或 langgraph-checkpoint-postgres
```

### 4.2 完整示例：计算器 Agent

```python
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain.messages import AnyMessage, SystemMessage, ToolMessage, HumanMessage
from typing_extensions import TypedDict, Annotated
from typing import Literal
import operator
from langgraph.graph import StateGraph, START, END

# Step 1: 工具和模型
model = init_chat_model("claude-sonnet-4-6", temperature=0)

@tool def multiply(a: int, b: int) -> int: return a * b
@tool def add(a: int, b: int) -> int: return a + b
@tool def divide(a: int, b: int) -> float: return a / b

tools = [add, multiply, divide]
tools_by_name = {t.name: t for t in tools}
model_with_tools = model.bind_tools(tools)

# Step 2: 状态
class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int

# Step 3: LLM 节点
def llm_call(state: MessagesState):
    return {
        "messages": [model_with_tools.invoke(
            [SystemMessage(content="You are a helpful math assistant.")] + state["messages"]
        )],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }

# Step 4: 工具节点
def tool_node(state: MessagesState):
    result = []
    for tc in state["messages"][-1].tool_calls:
        tool = tools_by_name[tc["name"]]
        observation = tool.invoke(tc["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tc["id"]))
    return {"messages": result}

# Step 5: 路由
def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    return "tool_node" if state["messages"][-1].tool_calls else END

# Step 6: 构建
builder = StateGraph(MessagesState)
builder.add_node("llm_call", llm_call)
builder.add_node("tool_node", tool_node)
builder.add_edge(START, "llm_call")
builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
builder.add_edge("tool_node", "llm_call")
agent = builder.compile()

# 调用
result = agent.invoke({"messages": [HumanMessage(content="Add 3 and 4, multiply by 2")]})
for m in result["messages"]:
    m.pretty_print()
```

---

## 5. State（状态）深度解析

### 5.1 State 本质

State 是 `TypedDict`（或 dataclass），在所有节点间共享。

```python
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]  # Reducer: 追加
    user_id: str                                           # 覆盖
    search_results: list[str] | None
    classification: dict | None
    cycle_count: Annotated[int, lambda x, y: x + y]       # 自定义 Reducer
```

### 5.2 Reducer 机制

| 注解方式 | 行为 | 适用场景 |
|----------|------|---------|
| 无注解 | 覆盖（最后写入胜出） | 单一值字段 |
| `Annotated[list, operator.add]` | 追加合并 | 消息列表 |
| `Annotated[list, custom_fn]` | 自定义合并 | 需裁剪/去重/限制的列表 |

```python
# 生产级 Reducer：限制消息历史为最近 50 条
def trimmed_add(left: list[BaseMessage], right: list[BaseMessage] | BaseMessage) -> list[BaseMessage]:
    if isinstance(right, BaseMessage):
        right = [right]
    merged = left + right if left else right
    return merged[-50:]

class ConversationState(TypedDict):
    messages: Annotated[list[BaseMessage], trimmed_add]
```

### 5.3 Command：统一状态更新与路由

```python
from langgraph.types import Command

def classify(state: State) -> Command[Literal["search", "human", "reply"]]:
    classification = llm.invoke(...)
    if classification["urgency"] == "critical":
        return Command(update={"classification": classification}, goto="human")
    return Command(update={"classification": classification}, goto="search")
```

### 5.4 Context API（v0.6+）

```python
from dataclasses import dataclass
from langgraph.runtime import Runtime

@dataclass
class Context:
    user_id: str
    db_connection: str
    tenant_id: str

def node(state: State, runtime: Runtime[Context]):
    user_id = runtime.context.user_id
    tenant_data = runtime.store.get(("tenants", runtime.context.tenant_id), "config")
    runtime.stream_writer({"status": "processing..."})
```

---

## 6. Nodes 与 Edges 详解

### 6.1 节点类型

| 类型 | 说明 | 实现 |
|------|------|------|
| 普通节点 | 纯函数 | `def node(state) -> dict` |
| LLM 节点 | 调用模型 | `model.invoke(messages)` |
| 工具节点 | 执行工具 | `ToolNode(tools)` |
| 子图节点 | 嵌套图 | `builder.add_node("sub", compiled_subgraph)` |
| 条件路由 | 路由决策 | `add_conditional_edges` 或 `Command(goto=...)` |

### 6.2 边的类型

```python
# 普通边
builder.add_edge("node_a", "node_b")

# 条件边
def route(state: State) -> Literal["path_a", "path_b", END]:
    return "path_a" if state["score"] > 0.8 else "path_b"

builder.add_conditional_edges("source", route, {"path_a": "node_a", "path_b": "node_b"})

# Command 动态路由（逐次决策，更灵活）
def node(state: State) -> Command[Literal["path_a", END]]:
    return Command(update={...}, goto="path_a" if cond else END)
```

### 6.3 Send API：动态扇出

```python
from langgraph.types import Send

def assign_workers(state: State):
    return [Send("worker", {"section": s}) for s in state["sections"]]

builder.add_conditional_edges("orchestrator", assign_workers, ["worker"])
```

### 6.4 节点配置

```python
builder.add_node(
    "api_call",
    call_api,
    retry_policy=RetryPolicy(max_attempts=3, initial_interval=1.0, backoff_factor=2.0),
    timeout=30,
    cache_policy=CachePolicy(key_fn=lambda s: hash(s["query"]), ttl=300)
)
```

---

## 7. 工作流与 Agent 设计模式

### 7.1 Prompt Chaining（提示词链）

```
[Input] → [Generate] → [Validate] → [Improve] → [Polish] → [Output]
```

类似管道模式，每步验证后决定是否继续。

### 7.2 Parallelization（并行化）

```
               ┌──→ [LLM 1: Joke]
[START] ──→  ├──→ [LLM 2: Story]  ──→ [Aggregator] → [END]
               └──→ [LLM 3: Poem]
```

所有分支从 START 出发，并行执行后汇聚到 Aggregator。

### 7.3 Routing（路由）

```
                           ┌──→ [Story Writer]
[Input] → [Router/LLM] ──┼──→ [Joke Writer]
                           └──→ [Poem Writer]
```

使用 `with_structured_output()` 让 LLM 输出结构化路由决策。

### 7.4 Orchestrator-Worker

```
[Orchestrator] → [Send Workers] → [Worker 1] ┐
                                   [Worker 2] ├→ [Synthesizer] → [Output]
                                   [Worker 3] ┘
```

Worker 使用 `completed_sections: Annotated[list, operator.add]` 并行写入共享结果。

### 7.5 Agent 循环（ReAct）

```
[Agent/LLM] ←──→ [Tool Node]
     │ (无工具调用)
     ↓
   [END]
```

LangGraph 的 `create_react_agent` 预置了这一经典模式。

---

## 8. 持久化与记忆系统

### 8.1 双层架构

| 维度 | Checkpointer | Store |
|------|-------------|-------|
| 持久化内容 | 图状态快照 | 应用键值数据 |
| 作用域 | 单线程 | 跨线程 |
| 记忆类型 | 短期（线程级） | 长期（跨线程） |
| 用途 | 对话连续性、HITL、时间旅行、容错 | 用户偏好、共享知识 |

### 8.2 Checkpointer

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.postgres import PostgresSaver

# 开发：内存
checkpointer = InMemorySaver()
# 单机生产：SQLite
checkpointer = SqliteSaver.from_conn_string("checkpoints.db")
# 集群生产：Postgres
checkpointer = PostgresSaver.from_conn_string("postgresql://user:pass@host:5432/db")

graph = builder.compile(checkpointer=checkpointer, durability="async")
config = {"configurable": {"thread_id": "user-session-123"}}
```

### 8.3 时间旅行

```python
# 查看历史
history = list(graph.get_state_history(config))
# 从历史检查点恢复
graph.update_state(config, values=history[-3].values)
# 获取当前状态
snapshot = graph.get_state(config)
print(snapshot.values, snapshot.next, snapshot.interrupts)
```

### 8.4 Store（长期记忆）

```python
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()
graph = builder.compile(checkpointer=checkpointer, store=store)

def remember_user(state: State, runtime: Runtime):
    # 写入
    runtime.store.put(("users", state["user_id"]), "preferences", {"lang": "zh"})
    # 读取
    prefs = runtime.store.get(("users", state["user_id"]), "preferences")
    # 搜索（语义搜索）
    results = runtime.store.search(("knowledge",), query="如何重置密码", limit=5)
```

---

## 9. Human-in-the-Loop（人机协同）

### 9.1 interrupt()：暂停等待

```python
from langgraph.types import interrupt, Command

def review_draft(state: State):
    if state["urgency"] == "critical":
        decision = interrupt({
            "message": "请审核此关键回复",
            "draft": state["draft_response"],
        })
        if decision.get("approved"):
            return {"response": state["draft_response"]}
        return {"response": decision.get("edited_response")}
    return {"response": state["draft_response"]}
```

### 9.2 恢复执行

```python
# 驱动到中断点
stream = graph.stream_events(input_data, config=config, version="v3")
output = stream.output

# 查看中断
for i in stream.interrupts:
    print(i.value)

# 恢复（批准）
resumed = graph.stream_events(Command(resume=True), config=config, version="v3")
# 恢复（带修改）
resumed = graph.stream_events(Command(resume={"approved": True}), config=config)
```

### 9.3 编译时配置

```python
graph = builder.compile(
    checkpointer=checkpointer,
    interrupt_before=["send_reply", "delete_data"],  # 关键操作前暂停
    interrupt_after=["classify_intent"],              # 分类后检查
)
```

---

## 10. Subgraphs 与 Multi-Agent 系统

### 10.1 子图通信模式

| 模式 | 状态 Schema | 实现 |
|------|------------|------|
| 节点内调用 | 不同 | 包装函数做状态转换 |
| 作为节点添加 | 共享 | `builder.add_node("sub", compiled_subgraph)` |

### 10.2 子图持久化模式

| 模式 | checkpointer= | 行为 |
|------|--------------|------|
| Per-invocation | `None` | 每次全新开始，支持 interrupt（推荐） |
| Per-thread | `True` | 跨调用累积状态 |
| Stateless | `False` | 无检查点，纯函数调用 |

### 10.3 Multi-Agent 三大架构模式

#### 模式 1：Supervisor（中央管理者）

```
[Supervisor] ──→ [Agent A] ──→ [Supervisor]
              ──→ [Agent B] ──→ [Supervisor]
              ──→ [Agent C] ──→ [Supervisor]
                        └──→ [END]
```

Supervisor 作为一个路由 LLM，根据上下文决定哪个子 Agent 处理下一步。

```python
class Router(TypedDict):
    next: Literal["billing_agent", "tech_agent", "FINISH"]

supervisor = llm.with_structured_output(Router)

def supervisor_node(state: State) -> Command[Literal["billing", "tech", END]]:
    decision = supervisor.invoke(state["messages"])
    if decision["next"] == "FINISH":
        return Command(goto=END)
    return Command(goto=decision["next"])
```

#### 模式 2：Hierarchical Teams（层级团队）

```
[Team Lead] ──→ [Sub-team Lead] ──→ [Worker 1]
                                   ──→ [Worker 2]
              ──→ [Sub-team Lead] ──→ [Worker 3]
```

通过子图嵌套实现多级 Agent 结构。高层 Agent 将子 Agent 封装为工具调用。

#### 模式 3：Shared Scratchpad（共享工作空间）

多个 Agent 通过共享 State + Store 协作，无需中心调度器，适用于松耦合协作场景。

### 10.4 子 Agent 作为工具（推荐模式）

```python
from langchain.agents import create_agent

billing_agent = create_agent(
    model="gpt-5",
    tools=[billing_tool],
    prompt="You are a billing expert.",
)

@tool
def ask_billing_expert(question: str) -> str:
    """Ask billing expert. Use for ALL billing/payment questions."""
    response = billing_agent.invoke(
        {"messages": [{"role": "user", "content": question}]}
    )
    return response["messages"][-1].content

router_agent = create_agent(
    model="gpt-5",
    tools=[ask_billing_expert, ask_tech_expert],
    prompt="Delegate questions to the appropriate expert.",
    checkpointer=MemorySaver(),
)
```

---

## 11. LangGraph + GraphRAG：知识图谱增强

### 11.1 为什么需要 GraphRAG

传统向量 RAG 擅长语义相似性搜索，但无法处理多跳推理和关系查询。

```
问题："谁和小明在同一个项目工作过后又转到了其他部门？"

向量 RAG：可能找到"小明"相关文档，但无法追踪关系链
GraphRAG：小明 → 任职于 → 项目 X → 成员 → 张三 → 转岗 → 部门 Y
```

### 11.2 GraphRAG 架构

```
+------------+    +-------------------+    +-----------------+
| 原始数据   |──→ | LLM 实体/关系提取  |──→ | 知识图谱存储    |
+------------+    +-------------------+    +-------+---------+
                                                   │
    用户查询 ──→ LLM 查询翻译 ──→ 图查询（Cypher/SPARQL）──┤
                                                   │
    增强生成 ←── LLM ←── 子图结果 + 原始查询           │
```

### 11.3 LangGraph + GraphRAG 的实现方式

**方式一：Graph Database 作为工具**

```python
from langchain_community.graphs import Neo4jGraph

graph_db = Neo4jGraph(url="bolt://localhost:7687", username="neo4j", password="password")

@tool
def query_knowledge_graph(cypher_query: str) -> str:
    """Execute a Cypher query on the knowledge graph."""
    result = graph_db.query(cypher_query)
    return json.dumps(result, ensure_ascii=False)

@tool
def semantic_search(query: str) -> str:
    """Vector similarity search in document store."""
    results = vector_store.similarity_search(query, k=5)
    return "\n\n".join([doc.page_content for doc in results])
```

**方式二：Hybrid GraphRAG Agent 完整流程**

```python
class GraphRAGState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    query_type: str          # "graph" | "semantic" | "hybrid"
    graph_results: str | None
    semantic_results: str | None
    final_answer: str | None

def query_router(state: GraphRAGState) -> Command:
    """判断用图查询还是向量查询"""
    router = llm.with_structured_output(QueryType)
    decision = router.invoke(f"Query: {state['messages'][-1].content}")
    return Command(
        update={"query_type": decision.type},
        goto="graph_search" if decision.type == "graph" else "semantic_search"
    )

def graph_search(state: GraphRAGState):
    llm_with_cypher = llm.with_structured_output(CypherQuery)
    cypher = llm_with_cypher.invoke(f"Generate Cypher: {state['messages'][-1].content}")
    result = graph_db.query(cypher.query)
    return {"graph_results": json.dumps(result, ensure_ascii=False)}

def synthesize(state: GraphRAGState):
    context = []
    if state.get("graph_results"):
        context.append(f"知识图谱查询结果:\n{state['graph_results']}")
    if state.get("semantic_results"):
        context.append(f"文档检索结果:\n{state['semantic_results']}")
    answer = llm.invoke(f"Context:\n{chr(10).join(context)}\n\nQuestion: {state['messages'][-1].content}")
    return {"final_answer": answer.content}
```

### 11.4 GraphRAG Pipeline 四阶段

1. **文本切分** → TextUnits（保留实体边界）
2. **图谱构建** → LLM 实体提取 → 关系提取 → 声明提取 → 知识图谱
3. **社区检测与摘要** → Leiden 算法检测图社区 → 生成社区摘要
4. **查询检索** → 图遍历 + 社区上下文 → 增强生成

---

## 12. Streaming（流式处理）

### 12.1 流模式

| 模式 | 输出内容 | 使用场景 |
|------|---------|---------|
| `"values"` | 每步后完整 State | 获取全量状态 |
| `"updates"` | 每步增量更新 | 追踪变化 |
| `"messages"` | Token 级 LLM 输出 | 打字机效果 |
| `"custom"` | 自定义输出 | 业务事件 |
| `"checkpoints"` | 检查点事件 | 监控持久化 |
| `"tasks"` | 任务生命周期 | 调试调度 |
| `"debug"` | 详细调试信息 | 深度排查 |

### 12.2 使用示例

```python
# 增量更新
for chunk in graph.stream(input_data, config=config, stream_mode="updates"):
    print(chunk)

# Token 级流 + 增量更新（组合模式）
for chunk in graph.stream(input_data, config=config, stream_mode=["updates", "messages"]):
    if isinstance(chunk, tuple):
        (mode, data) = chunk
        if mode == "messages":
            print(data.content, end="", flush=True)

# 事件流（v3 最丰富）
for event in graph.stream_events(input_data, config=config, version="v3"):
    if event["event"] == "on_chat_model_stream":
        print(event["data"]["chunk"].content, end="")
    elif event["event"] == "on_custom_event":
        print(f"[Custom] {event['data']}")
```

---

## 13. 容错与错误处理

### 13.1 错误分类与处理策略

| 错误类型 | 处理方式 | 实现 |
|----------|---------|------|
| **瞬态错误**（网络/限流） | 自动重试 | `RetryPolicy` |
| **LLM 可恢复**（工具失败） | 写入错误信息 → 循环回 LLM | `Command(goto="agent")` |
| **用户可修复**（缺信息） | 暂停等待 | `interrupt()` |
| **重试耗尽后可恢复** | 补偿/降级 | `error_handler`（v1.2+） |
| **意外错误** | 向上冒泡 | 不捕获 |

### 13.2 RetryPolicy

```python
builder.add_node(
    "api_call",
    call_external_api,
    retry_policy=RetryPolicy(
        max_attempts=3,
        initial_interval=1.0,
        backoff_factor=2.0,           # 1s → 2s → 4s
        retry_on=(ConnectionError, TimeoutError)
    ),
    timeout=30,
)
```

### 13.3 LLM 自恢复

```python
def execute_tool(state: State) -> Command[Literal["agent", "execute_tool"]]:
    try:
        result = run_tool(state["tool_call"])
        return Command(update={"tool_result": result}, goto="agent")
    except ToolError as e:
        return Command(
            update={"tool_result": f"Tool error: {str(e)}"},
            goto="agent"  # 回到 agent 让 LLM 调整策略
        )
```

### 13.4 Durability Mode

| 模式 | 检查点保存时机 | 持久性 | 性能 |
|------|-------------|--------|------|
| `"exit"` | 仅图退出时 | 低 | 最快 |
| `"async"` | 异步保存（下一步执行时） | 中 | 快 |
| `"sync"` | 同步保存（下一步执行前） | 高 | 较慢 |

```python
graph = builder.compile(checkpointer=checkpointer, durability="async")
```

### 13.5 常见故障排查

| 症状 | 可能原因 | 排查方法 |
|------|---------|---------|
| 无限循环 | 条件边逻辑错误、LLM 反复工具调用 | 设置 `cycle_count` 上限、检查 `tools_condition` |
| State 丢失 | Reducer 冲突、覆盖而非追加 | 检查 `Annotated[...]` 注解 |
| 检查点反序列化失败 | Schema 变更 | 使用版本化的 State Schema |
| 并发状态冲突 | 多 Worker 写入同一 key | 使用 `Annotated[list, operator.add]` 合并 |

---

## 14. 安全与护栏（Guardrails）

### 14.1 LangGraph 安全攻击面

根据 Gravitee 2026 报告，**88% 的 Agent 团队报告过安全事件**。LangGraph 特有的攻击面：

| 攻击面 | 风险 | 防护措施 |
|--------|------|---------|
| **State 操纵** | 恶意工具返回篡改 State | 输出验证、State Schema 约束 |
| **凭证泄露** | 所有节点共享同一凭证上下文 | 按节点隔离凭证 |
| **Agent-to-Agent 信任升级** | 子 Agent 获取父 Agent 全部权限 | 能力范围 Token、最小权限 |
| **MCP 通配符权限** | 所有 Agent 可调用所有工具 | 显式工具许可列表 |
| **检查点投毒** | 恶意数据持久化到检查点 | 检查点数据校验 |

### 14.2 三层护栏体系

```
┌─────────────────────────────────────────┐
│            输入护栏 (Input Guardrails)     │
│  - Prompt 注入检测                         │
│  - 输入长度/内容限制                        │
│  - 敏感信息过滤                            │
├─────────────────────────────────────────┤
│          工具调用护栏 (Tool Guardrails)      │
│  - 高危工具需 HITL 审批                     │
│  - 金额/数量上限                            │
│  - 工具参数校验                            │
│  - 速率限制                               │
├─────────────────────────────────────────┤
│            输出护栏 (Output Guardrails)     │
│  - 内部信息泄露检测                         │
│  - 不当内容过滤                            │
│  - 幻觉/事实核查                           │
└─────────────────────────────────────────┘
```

### 14.3 实现示例

```python
# 输入护栏
def input_guardrail(state: State) -> Command[Literal["agent", "block"]]:
    content = state["messages"][-1].content
    # Prompt 注入检测
    if any(phrase in content.lower() for phrase in [
        "ignore previous instructions",
        "system prompt",
        "you are now"
    ]):
        return Command(
            update={"messages": [{"role": "ai", "content": "检测到不安全输入"}]},
            goto=END
        )
    # 敏感信息过滤
    if re.search(r'\b\d{16}\b', content):  # 信用卡号
        return Command(
            update={"messages": [{"role": "ai", "content": "请勿输入信用卡信息"}]},
            goto=END
        )
    return Command(goto="agent")

# 工具护栏
@tool
def delete_data(record_id: str) -> str:
    """Delete a record. Requires human approval."""
    # 使用 interrupt 强制人工确认
    approval = interrupt({
        "action": "delete_data",
        "record_id": record_id,
        "message": "确认删除此记录？此操作不可逆。"
    })
    if not approval.get("confirmed"):
        return "操作已取消"
    return db.delete(record_id)

# 输出护栏
def output_guardrail(state: State) -> dict:
    """检查输出是否包含内部信息"""
    response = state.get("response", "")
    forbidden = ["内部系统", "API_KEY", "数据库密码"]
    for keyword in forbidden:
        if keyword.lower() in response.lower():
            return {"response": "回答包含敏感信息，已被过滤"}
    return {}
```

### 14.4 凭证隔离最佳实践

```python
# 按节点隔离凭证（不要在全局共享 credentials）
def billing_node(state: State):
    # 仅在需要时获取特定凭证
    billing_api_key = os.environ["BILLING_API_KEY"]
    return call_billing_api(billing_api_key, state["data"])

def email_node(state: State):
    # 不同的凭证用于不同的节点
    email_api_key = os.environ["EMAIL_API_KEY"]
    return send_email(email_api_key, state["data"])
```

---

## 15. 测试与评估（Testing & Evaluation）

### 15.1 测试金字塔

```
              ┌───────────┐
              │  E2E 测试  │  ← 完整 Agent 工作流
              ├───────────┤
              │  集成测试  │  ← Node 间交互、路由正确性
              ├───────────┤
              │  单元测试  │  ← 单个 Node 函数、Reducer、State
              └───────────┘
```

### 15.2 单元测试

```python
import pytest

def test_tool_execution():
    """测试工具节点是否正确处理 tool_calls"""
    state = {
        "messages": [
            AIMessage(content="", tool_calls=[{
                "name": "add", "args": {"a": 3, "b": 4}, "id": "call_1"
            }])
        ],
        "llm_calls": 0
    }
    result = tool_node(state)
    assert len(result["messages"]) == 1
    assert result["messages"][0].content == "7"

def test_reducer_trimming():
    """测试自定义 Reducer 是否限制消息数量"""
    messages = [HumanMessage(content=f"msg_{i}") for i in range(60)]
    result = trimmed_add([], messages)
    assert len(result) == 50
    assert result[0].content == "msg_10"  # 只保留最近 50 条

def test_routing_logic():
    """测试条件路由"""
    state = {"messages": [AIMessage(content="", tool_calls=[{...}])]}
    assert should_continue(state) == "tool_node"

    state = {"messages": [AIMessage(content="Done!")]}
    assert should_continue(state) == END
```

### 15.3 集成测试（完整图执行）

```python
def test_calculator_agent_flow():
    """端到端测试计算器 Agent"""
    result = agent.invoke({
        "messages": [HumanMessage(content="What is 3 + 4?")]
    })
    # 验证最终回答包含正确结果
    final_message = result["messages"][-1].content
    assert "7" in final_message

def test_multi_step_reasoning():
    """测试多步骤推理"""
    result = agent.invoke({
        "messages": [HumanMessage(content="3 + 4, then multiply by 2")]
    })
    final_content = result["messages"][-1].content
    assert "14" in final_content

def test_graph_state_persistence():
    """测试状态持久化与恢复"""
    config = {"configurable": {"thread_id": "test-1"}}
    agent.invoke(
        {"messages": [HumanMessage(content="My name is Alice")]},
        config=config
    )
    result = agent.invoke(
        {"messages": [HumanMessage(content="What's my name?")]},
        config=config
    )
    assert "Alice" in result["messages"][-1].content
```

### 15.4 使用 Promptfoo 进行 Agent 评估

```yaml
# promptfooconfig.yaml
description: "LangGraph Agent Evaluation"
prompts:
  - "What is 3 + 4?"
  - "Calculate 10 * 5 / 2"
  - "What is the capital of France?"

providers:
  - id: langgraph
    config:
      graph: agent
      model: gpt-5

tests:
  - vars:
      question: "What is 3 + 4?"
    assert:
      - type: contains
        value: "7"
      - type: not-contains
        value: "error"

  - vars:
      question: "What is 10 * 5 / 2?"
    assert:
      - type: javascript
        value: "output.includes('25')"

redteam:
  plugins:
    - prompt-injection
    - harmful-content
    - pii-leak
```

### 15.5 LangSmith 评估

```python
from langsmith import evaluate, Client

# 创建评估数据集
dataset = client.create_dataset(
    "agent-eval",
    description="Agent evaluation test cases"
)
client.create_examples(
    inputs=[
        {"question": "3 + 4 = ?"},
        {"question": "复杂的多步骤计算..."},
    ],
    outputs=[
        {"answer": "7"},
        {"answer": "..."},
    ],
    dataset_id=dataset.id,
)

# 运行评估
results = evaluate(
    lambda x: agent.invoke({"messages": [HumanMessage(content=x["question"])]}),
    data=dataset.name,
    evaluators=[
        "correctness",       # 基础正确性
        "criteria",          # 自定义标准
        "labeled_criteria",  # 标注对比
    ],
)
```

---

## 16. 性能优化与成本控制

### 16.1 性能优化策略

| 策略 | 方法 | 效果 |
|------|------|------|
| **消息裁剪** | 自定义 Reducer 保留最近 N 条 | 降低 Token 消耗 30-60% |
| **对话摘要** | 超过阈值时 LLM 摘要取代历史 | 保持上下文同时节省 Token |
| **缓存 LLM 响应** | 相同查询命中缓存 | 响应时间降低 80%+ |
| **并行执行** | 无依赖节点并行 | 整体延迟降低 40-70% |
| **异步模式** | `"async"` 耐久性 | 吞吐量提升 2-5x |
| **连接池复用** | HTTP/数据库连接池 | 降低连接开销 |

### 16.2 消息历史管理

```python
# 策略 1: 消息数量裁剪
def limited_add(left: list, right: list | BaseMessage) -> list:
    if isinstance(right, BaseMessage):
        right = [right]
    merged = left + right if left else right
    return merged[-20:]  # 最多 20 条

# 策略 2: Token 数量限制 + 摘要
def summarize_if_needed(state: State) -> dict:
    """当消息超过 N 条时，用 LLM 摘要早期消息"""
    if len(state["messages"]) > 10:
        early_msgs = state["messages"][:5]
        summary = llm.invoke(f"Summarize: {early_msgs}").content
        # 保留摘要 + 最近 5 条消息
        return {
            "messages": [SystemMessage(content=summary)] + state["messages"][-5:]
        }
    return {}
```

### 16.3 缓存策略

```python
from langgraph.types import CachePolicy

builder.add_node(
    "search",
    search_fn,
    cache_policy=CachePolicy(
        key_fn=lambda state: hash(state["query"]),
        ttl=3600  # 1 小时过期
    )
)
```

### 16.4 成本监控

```python
class CostTrackerState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    total_tokens: Annotated[int, operator.add]
    total_cost: Annotated[float, operator.add]

MODEL_PRICING = {
    "gpt-5-mini": {"input": 0.15, "output": 0.60},  # 每 1M tokens 价格
    "claude-sonnet-4": {"input": 3.00, "output": 15.00},
}

def track_cost(state: CostTrackerState, response: AIMessage):
    usage = response.response_metadata.get("token_usage", {})
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)

    model = response.response_metadata.get("model_name", "gpt-5-mini")
    pricing = MODEL_PRICING.get(model, MODEL_PRICING["gpt-5-mini"])
    cost = (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000

    return {
        "total_tokens": input_tokens + output_tokens,
        "total_cost": cost
    }
```

---

## 17. LangGraph Platform / Agent Server

### 17.1 概述

LangGraph Platform（已更名为 **LangSmith Deployment**）是专门为长时间运行、有状态 Agent 构建的基础设施和部署管理平台。

- **1-Click 部署**：GitHub 集成，选择仓库即可上线
- **30+ API 端点**：Assistants / Threads / Runs / Cron / Webhooks
- **水平扩缩**：处理突发流量和长时间运行任务
- **内置持久化**：PostgreSQL + Redis 自动配置

### 17.2 Agent Server 核心概念

| 概念 | 说明 |
|------|------|
| **Assistant** | 图 + 特定配置 = 一个 Agent 实例 |
| **Thread** | 一次对话会话，对应 `thread_id` |
| **Run** | Thread 中的一次执行 |
| **Cron Job** | 定时触发 Agent 执行（仅 Enterprise） |
| **Webhook** | 事件驱动的 Agent 触发 |

### 17.3 版本对比

| 功能 | Lite（免费） | Enterprise |
|------|-------------|-----------|
| 节点执行量/年 | 100 万 | 无限制 |
| Cron Jobs | ❌ | ✅ |
| 自定义认证 | ❌ | ✅ |
| 部署选项 | 独立容器 | Cloud SaaS / 自托管 / 混合 |

### 17.4 Agent Server API 示例

```python
# 创建 Assistant
POST /assistants
{
    "graph_id": "customer-support",
    "config": {"model": "gpt-5-mini"}
}

# 创建 Thread
POST /threads
{"metadata": {"user_id": "user-123"}}

# 执行 Run
POST /threads/{thread_id}/runs
{
    "assistant_id": "asst_xxx",
    "input": {"messages": [{"role": "user", "content": "退款请求"}]}
}

# 流式输出
POST /threads/{thread_id}/runs/stream
{"assistant_id": "asst_xxx", "input": {...}}

# 查看 Run 状态
GET /threads/{thread_id}/runs/{run_id}

# 恢复中断的 Run
POST /threads/{thread_id}/runs/{run_id}/resume
{"decision": {"approved": true}}
```

### 17.5 langgraph.json 应用结构

```json
{
    "python_version": "3.12",
    "dependencies": ["."],
    "graphs": {
        "customer-support": "./src/graph.py:agent",
        "data-analysis": "./src/analysis.py:analyst"
    },
    "env": ".env"
}
```

### 17.6 RemoteGraph：远程调用

```python
from langgraph.remote_graph import RemoteGraph

remote_agent = RemoteGraph(
    graph_id="customer-support",
    url="https://your-deployment.langchain.com",
    api_key=os.environ["LANGSMITH_API_KEY"]
)

result = remote_agent.invoke(
    {"messages": [{"role": "user", "content": "退款申请"}]},
    config={"configurable": {"thread_id": "thread-1"}}
)
```

---

## 18. 生产部署

### 18.1 自托管（Docker Compose）

```yaml
services:
  langgraph-redis:
    image: redis:7
    healthcheck:
      test: redis-cli ping
      interval: 5s
  langgraph-postgres:
    image: postgres:16
    environment:
      POSTGRES_DB: postgres
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: pg_isready -U postgres
      interval: 5s
  langgraph-api:
    image: ${IMAGE_NAME}
    ports:
      - "8123:8000"
    depends_on:
      langgraph-redis:
        condition: service_healthy
      langgraph-postgres:
        condition: service_healthy
    environment:
      REDIS_URI: redis://langgraph-redis:6379
      DATABASE_URI: postgresql://postgres:postgres@langgraph-postgres:5432/postgres
      LANGSMITH_API_KEY: ${LANGSMITH_API_KEY}
      LANGGRAPH_CLOUD_LICENSE_KEY: ${LANGGRAPH_CLOUD_LICENSE_KEY}

volumes:
  pgdata:
```

### 18.2 Kubernetes

使用官方 Helm Chart 部署，支持 HPA 自动扩缩容。

### 18.3 环境变量

| 变量 | 必填 | 说明 |
|------|------|------|
| `REDIS_URI` | 是 | Redis 连接串（pub-sub 流式） |
| `DATABASE_URI` | 是 | Postgres 连接串（状态持久化） |
| `LANGSMITH_API_KEY` | 是 | LangSmith API Key |
| `LANGGRAPH_CLOUD_LICENSE_KEY` | Enterprise | License 密钥 |
| `LANGSMITH_ENDPOINT` | 自托管 | 自托管 LangSmith 地址 |
| `POSTGRES_URI` | Docker | 同 DATABASE_URI |

### 18.4 健康检查

```bash
curl http://0.0.0.0:8123/ok
# {"ok": true}
```

### 18.5 扩缩容配置

```yaml
autoscaling:
  min_instances: 3
  max_instances: 10
  target_cpu_utilization: 70
  metrics:
    - type: concurrent_runs
      target: 100
  cooldown: 300
```

---

## 19. 监控与可观测性（LangSmith）

### 19.1 自动追踪

```bash
# 设置环境变量即可自动追踪
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=ls_...
LANGSMITH_PROJECT=my-project
```

### 19.2 LangSmith Engine

- 自动监控 Agent trace，检测异常模式
- 发现循环、路径错误、State 异常
- 提出修复建议，可直接创建 Pull Request

### 19.3 Studio

LangGraph Studio 提供：
- 图结构可视化（Mermaid 图）
- 实时执行路径追踪和回放
- 逐节点状态检查
- 中断点设置与恢复
- 支持分支逻辑重放

```python
# 在 Jupyter 中显示图结构
from IPython.display import Image, display
display(Image(agent.get_graph().draw_mermaid_png()))
```

### 19.4 分布式追踪

```python
import langsmith as ls
from langsmith import traceable

@traceable(run_type="tool", name="knowledge_graph_query")
def query_kg(question: str) -> str:
    # 自动追踪为 LangSmith trace 中的一个 span
    result = graph_db.query(question)
    return result
```

---

## 20. 最佳实践

### 20.1 State 设计

- **存储原始数据** — 格式化在节点内完成
- **使用 Reducer 管理列表** — `operator.add` 追加，自定义 Reducer 裁剪
- **避免 State 膨胀** — 仅存储跨步骤需要的数据

### 20.2 错误处理

- **不为所有操作加 try-catch** — 信赖 LangGraph 的重试/恢复机制
- **分类处理错误** — 瞬态重试 / LLM 恢复 / HITL / 冒泡
- **关键节点配置 RetryPolicy** — API 调用、数据库查询

### 20.3 性能

- **`"async"` 耐久性** — 平衡性能与可靠性
- **消息裁剪** — 生产环境必须限制消息历史长度
- **并行化** — 无依赖节点应并行执行

### 20.4 安全

- **按节点隔离凭证** — 不同节点用不同 API Key
- **高危操作强制 HITL** — 删除、批量操作、支付相关
- **输入/输出护栏** — Prompt 注入检测、敏感信息过滤
- **MCP 工具许可白名单** — 显式声明可调用的工具

### 20.5 测试与部署

- **始终编译 checkpointer** — 即使开发阶段用 `InMemorySaver`
- **LangSmith 从开发阶段接入** — 不要上线才加
- **每个用户会话独用 thread_id**
- **CI/CD 中加入 Agent 测试** — Promptfoo + LangSmith 评估

### 20.6 Prebuilt 组件速查

```python
from langgraph.prebuilt import (
    create_react_agent,    # 快速创建 ReAct Agent
    ToolNode,              # 自动执行工具调用
    tools_condition,       # 标准工具路由条件
)

# 最简 Agent
agent = create_react_agent(model, tools, prompt="You are helpful.")
result = agent.invoke({"messages": [{"role": "user", "content": "Hi!"}]})
```

### 20.7 框架对比

| 维度 | LangGraph | CrewAI | AutoGen |
|------|-----------|--------|---------|
| 定位 | 底层编排运行时 | 角色扮演 Agent | 对话式 Multi-Agent |
| 灵活性 | 极高 | 中 | 中 |
| 学习曲线 | 中高 | 低 | 中 |
| 持久化 | 原生支持 | 有限 | 有限 |
| HITL | 深度集成 | 有限 | 有限 |
| 生产就绪度 | 高（Klarna/Uber） | 中 | 中 |

---

## 参考资源

- **官方文档**：https://docs.langchain.com/oss/python/langgraph/overview
- **GitHub**：https://github.com/langchain-ai/langgraph
- **LangChain Academy**：https://academy.langchain.com/courses/intro-to-langgraph
- **API 参考**：https://reference.langchain.com/python/langgraph/
- **LangSmith**：https://www.langchain.com/langsmith
- **LangSmith Deployment**：https://docs.langchain.com/langsmith/deployment
- **How-to 指南**：https://langchain-ai.github.io/langgraph/guides/
- **LangGraph Server 概念**：https://langchain-ai.github.io/langgraph/concepts/langgraph_server/

---

> **总结**：LangGraph 是构建生产级 Agent 的核心基础设施。它通过有向图建模 Agent 行为，用 State + Reducer 管理记忆，用 Checkpointer + Store 实现双层持久化，用 `interrupt()` 实现人机协同，用 Subgraph 支持多 Agent 协作，用 Agent Server 实现生产部署。结合 GraphRAG 知识图谱增强、多层安全护栏、系统化测试评估，LangGraph 构成了企业级 AI Agent 开发的完整技术栈。
