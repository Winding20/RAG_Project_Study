# RAG 文档问答系统

基于 RAG（检索增强生成）的本地知识库问答系统。
导入文档后，可以基于文档内容提问，减少大模型幻觉。

## 项目背景

大模型本身不知道你本地文档的内容，如果直接问，它可能编造答案。
RAG 的思路是：先从文档里找到相关内容，再让大模型基于这些内容回答。

## 核心原理

```
准备阶段：文档 → 加载 → 切片 → 向量化 → 存入向量库
提问阶段：问题 → 向量化 → 检索相似片段 → 拼入prompt → 大模型回答
```

## 技术栈

- Python 3.8+
- sentence-transformers（文字转向量）
- chromadb（向量数据库）
- requests（调用大模型）
- 智谱 GLM-4（大模型）

## 功能

- 读取本地 txt 文档
- 文本切片（支持重叠）
- 向量化入库
- 相似度检索
- 基于检索结果回答，减少幻觉

## 安装与运行

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

```bash
export ZHIPU_API_KEY="你的智谱API Key"
```

### 3. 准备知识库

编辑 `knowledge.txt`，填入你的文档内容。

### 4. 运行

```bash
python3 rag_project.py
```

## 运行示例

```
✅ 入库完成，共 3 块

👤 你: 怎么退款
🤖 Agent: 根据资料，退款流程是：登录账户，进入订单页，点击申请退款...

👤 你: 你们年假多少天
🤖 Agent: 资料中没有提到相关信息。
```

## 项目结构

```
rag-doc-qa/
├── rag_project.py      # 主程序
├── knowledge.txt       # 知识库文档
├── requirements.txt    # 依赖清单
├── README.md           # 项目说明
└── .gitignore          # Git忽略规则
```

## 学习收获

- 理解 RAG 的完整流程：加载 → 切片 → 向量化 → 检索 → 生成
- 掌握 Embedding 模型的用法
- 掌握 Chroma 向量库的增删查
- 理解"检索 + 生成"如何减少大模型幻觉

## License

MIT