import chromadb
import requests
from sentence_transformers import SentenceTransformer

# ============ 配置 ============
import os
import chromadb
import requests
from sentence_transformers import SentenceTransformer

# ============ 配置 ============
API_KEY = os.environ.get("ZHIPU_API_KEY")
if not API_KEY:
    raise ValueError("❌ 请设置环境变量 ZHIPU_API_KEY")

URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "my_docs"

# ============ 1. 加载模型 + Chroma ============
model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(COLLECTION_NAME)


# ============ 2. 文档切片 ============
def split_text(text, chunk_size=300, overlap=50):
    """按固定字数切片，相邻块有重叠"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


# ============ 3. 入库 ============
def build_knowledge_base(file_path):
    """读取文档 → 切片 → 向量化 → 存Chroma"""
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = split_text(text)
    vectors = model.encode(chunks)

    ids = [f"doc_{i}" for i in range(len(chunks))]
    collection.add(
        documents=chunks,
        embeddings=vectors.tolist(),
        ids=ids
    )
    print(f"✅ 入库完成，共 {len(chunks)} 块")


# ============ 4. 检索 + 回答 ============
def ask(question):
    """提问 → 检索 → 拼prompt → 大模型回答"""
    # 4.1 问题转向量
    question_vector = model.encode([question])

    # 4.2 去Chroma检索
    results = collection.query(
        query_embeddings=question_vector.tolist(),
        n_results=3
    )
    retrieved_docs = results["documents"][0]

    # 4.3 拼prompt
    context = "\n".join(retrieved_docs)
    prompt = f"""请根据以下资料回答问题，不要编造。

资料：
{context}

问题：{question}
"""

    # 4.4 发给大模型
    resp = requests.post(
        url=URL,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "glm-4-flash",
            "messages": [{"role": "user", "content": prompt}]
        }
    )
    data = resp.json()
    return data["choices"][0]["message"]["content"]


# ============ 5. 主程序 ============
if __name__ == "__main__":
    # 第一次运行：建知识库
    build_knowledge_base("knowledge.txt")

    # 循环提问
    while True:
        q = input("\n👤 你: ").strip()
        if q in ["exit", "quit", "退出"]:
            break
        if not q:
            continue
        answer = ask(q)
        print(f"🤖 Agent: {answer}")