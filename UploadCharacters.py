# seed_characters.py
# -*- coding: utf-8 -*-
import os
import sys
import re
from typing import List, Dict

from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable
from google import genai

# ========= 环境变量 =========
load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI", "").strip()
NEO4J_USER = os.getenv("NEO4J_USER", "").strip()
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "").strip()
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "").strip() or None  # 可选

# 用于嵌入
GEMINI_PROJECT = os.getenv("GEMINI_PROJECT_ID", "").strip()
GEMINI_LOCATION = os.getenv("GEMINI_LOCATION", "us-central1").strip()
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-004").strip()

if not (NEO4J_URI and NEO4J_USER and NEO4J_PASSWORD):
    print("请在 .env 中设置 NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD")
    sys.exit(1)

if not GEMINI_PROJECT:
    print("请在 .env 中设置 GEMINI_PROJECT_ID（用于生成向量）")
    sys.exit(1)

# ========= 客户端 =========
genai_client = genai.Client(vertexai=True, project=GEMINI_PROJECT, location=GEMINI_LOCATION)

def embed_text(text: str) -> List[float]:
    """调用嵌入模型，把文本转成向量。失败时返回空列表。"""
    try:
        text = (text or "").strip()
        if not text:
            return []
        resp = genai_client.models.embed_content(model=EMBED_MODEL, contents=text)
        # 兼容不同字段
        if hasattr(resp, "embeddings") and resp.embeddings:
            e0 = resp.embeddings[0]
            vec = getattr(e0, "values", None) or getattr(e0, "embedding", None)
        else:
            vec = getattr(resp, "values", None)
        return list(vec) if vec else []
    except Exception as e:
        print(f"[warn] embed failed: {e}")
        return []

# ========= 连接 =========
def _bolt_fallback(uri: str) -> str:
    return (uri.replace("neo4j+ssc://", "bolt+ssc://")
               .replace("neo4j+s://", "bolt+s://")
               .replace("neo4j://", "bolt://", 1))

def get_driver():
    try:
        drv = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        drv.verify_connectivity()
        return drv
    except ServiceUnavailable as e:
        if "routing" in str(e).lower():
            bolt_uri = _bolt_fallback(NEO4J_URI)
            drv = GraphDatabase.driver(bolt_uri, auth=(NEO4J_USER, NEO4J_PASSWORD))
            drv.verify_connectivity()
            print(f"[info] 路由失败，已从 {NEO4J_URI} 降级为 {bolt_uri}")
            return drv
        raise

# ========= 预设数据 =========
DEFAULT_WORLD = "格雷雾港·低魔中世纪小镇，海雾、盗贼行会与地底封印的流言。"

PRESETS = [
    {"name": "小明",   "persona": "油嘴滑舌、爱逗趣，表面幽默内心细腻。",         "worldview": DEFAULT_WORLD, "desire": 0.6},
    {"name": "小红",   "persona": "外冷内热、毒舌又别扭，刀子嘴豆腐心。",         "worldview": DEFAULT_WORLD, "desire": 0.4},
    {"name": "老板娘", "persona": "稳重谨慎、爱打听，掌控场面高手。",           "worldview": DEFAULT_WORLD, "desire": 0.5},
    {"name": "猎魔人", "persona": "冷峻寡言，目标明确，不喜赘述。",             "worldview": DEFAULT_WORLD, "desire": 0.35},
    {"name": "学院学者","persona": "好奇心过盛、讲究证据，常用比喻解释复杂问题。", "worldview": DEFAULT_WORLD, "desire": 0.25},
]

# 每个角色的“自述/偏好类”记忆（会转成『<名字>曾说：...』）
PERSONA_MEMORIES: Dict[str, List[str]] = {
    # "小明": [
    #     "我嘴上爱开玩笑，但认真起来比谁都真诚。",
    #     "我最拿手的是逗笑别人，尤其是心情不好的时候。",
    # ],
    # "小红": [
    #     "我讨厌被看穿的感觉。",
    #     "不喜欢花里胡哨的东西，实在才重要。"
    # ],
    # "老板娘": [
    #     "酒馆里的一句闲话，能比官府告示更快传遍全城。",
    #     "做生意要讲信誉，但也要懂规矩。",
    # ],
    # "猎魔人": [
    #     "我从不夸下海口，只给结果。",
    #     "血月临近时，不要轻信任何承诺。",
    # ],
    # "学院学者": [
    #     "未经证实的说法，只能算假设。",
    #     "记录和证据，让我们不至于迷失在流言里。",
    # ],
}

# “情报/传闻/档案”类记忆（可共享）
# audience 表示知道这条情报的角色列表（名字需在 PRESETS 中）
SHARED_INTEL = [
    {
        "content": "【传闻】封印古井在今春出现不明震动，夜里会发出低鸣。",
        "audience": ["小明", "老板娘", "猎魔人", "学院学者"],
    },
    {
        "content": "【情报】盗贼行会近期与外港走私船接头，时间多在浓雾夜。",
        "audience": ["老板娘", "猎魔人"],
    },
    {
        "content": "【档案】城北旧塔地下曾存放学院禁书《暗影律》，百年前被转移。",
        "audience": ["学院学者"],
    },
    {
        "content": "【流言】海上会唱歌的雾灯会迷惑船只，只有银铃声能让人清醒。",
        "audience": ["小明", "小红"],
    },
    {
        "content": "【传闻】市集里出现陌生炼金商贩，收购稀有血石，高价。",
        "audience": ["小红", "老板娘"],
    },
]

# ========= Cypher =========
INIT_SCHEMA = [
    "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Character) REQUIRE c.name IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (rb:RecentBuffer) REQUIRE rb.id IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (m:Memory) REQUIRE m.content IS UNIQUE",
]

UPSERT_CHARACTER_AND_RB = """
MERGE (c:Character {name:$name})
  ON CREATE SET c.created_at = datetime()
SET  c.persona    = $persona,
     c.worldview  = $worldview,
     c.desire     = $desire,
     c.updated_at = datetime()
WITH c
MERGE (rb:RecentBuffer {id:$rb_id})
  ON CREATE SET rb.lines_json = "[]", rb.created_at = datetime()
MERGE (c)-[:HAS_RECENT_BUFFER]->(rb)
"""

UPSERT_MEMORY = """
MERGE (m:Memory {content:$content})
  ON CREATE SET m.embedding = $embedding
"""

LINK_AWARE = """
UNWIND $names AS n
MATCH (c:Character {name:n})
MATCH (m:Memory {content:$content})
MERGE (c)-[:AWARE_OF]->(m)
"""

def rb_id_for(name: str) -> str:
    return f"rb::{name}"

def ensure_vector_index(session, dims: int):
    """按需创建 Memory 向量索引（支持的 Neo4j 版本才会成功；失败不报错）。"""
    try:
        session.run(f"""
        CREATE VECTOR INDEX mem_vec IF NOT EXISTS
        FOR (m:Memory) ON (m.embedding)
        OPTIONS {{
          indexConfig: {{
            `vector.dimensions`: {int(dims)},
            `vector.similarity_function`: 'cosine'
          }}
        }}
        """)
    except Exception as e:
        print(f"[info] 创建向量索引失败（可能版本不支持）：{e}")

def norm_persona_memory(name: str, text: str) -> str:
    """规范化人物记忆，统一加主语：『<名字>曾说：...』"""
    t = (text or "").strip().strip('“”"\'')
    if re.match(rf"^{re.escape(name)}\s*(曾说|说)\s*[:：]", t):
        return t
    return f"{name}曾说：{t}"

def main():
    driver = get_driver()
    with driver.session(database=NEO4J_DATABASE) as session:
        # 1) 约束
        for stmt in INIT_SCHEMA:
            session.run(stmt)

        # 2) 写入预设角色 & RecentBuffer
        for p in PRESETS:
            params = {
                "name": p["name"],
                "persona": p.get("persona", ""),
                "worldview": p.get("worldview", ""),
                "desire": float(p.get("desire", 0.3)),
                "rb_id": rb_id_for(p["name"]),
            }
            session.run(UPSERT_CHARACTER_AND_RB, **params)
            print(f"[ok] upsert Character: {p['name']} & RB({rb_id_for(p['name'])})")

        # 3) 写入“人物自述/偏好类”记忆（每人独有）
        first_vec_dims = None
        for name, mems in PERSONA_MEMORIES.items():
            for m in mems:
                content = norm_persona_memory(name, m)
                vec = embed_text(content)
                if vec and first_vec_dims is None:
                    first_vec_dims = len(vec)
                    ensure_vector_index(session, first_vec_dims)
                session.run(UPSERT_MEMORY, content=content, embedding=vec)
                session.run(LINK_AWARE, names=[name], content=content)
                print(f"  [mem] {name} <- {content[:28]}...")

        # 4) 写入“情报/传闻/档案”类（可多人共享）
        for item in SHARED_INTEL:
            content = (item["content"] or "").strip().strip('“”"\'')
            vec = embed_text(content)
            if vec and first_vec_dims is None:
                first_vec_dims = len(vec)
                ensure_vector_index(session, first_vec_dims)
            session.run(UPSERT_MEMORY, content=content, embedding=vec)
            # 过滤掉预设外的名字（避免拼错导致失败）
            audience = [n for n in item.get("audience", []) if any(n == p["name"] for p in PRESETS)]
            if audience:
                session.run(LINK_AWARE, names=audience, content=content)
            print(f"  [intel] {','.join(audience)} <- {content[:28]}...")

        # 5) 校验读取
        res = session.run("""
            MATCH (c:Character)-[:HAS_RECENT_BUFFER]->(rb:RecentBuffer)
            RETURN c.name AS name, c.desire AS desire, rb.id AS rb_id
            ORDER BY c.name
        """)
        rows = list(res)
        print("\n角色写入结果：")
        for r in rows:
            print(f" - {r['name']} (desire={r['desire']}) rb={r['rb_id']}")

        mem_cnt = session.run("MATCH (m:Memory) RETURN count(m) AS n").single()["n"]
        rel_cnt = session.run("MATCH (:Character)-[:AWARE_OF]->(:Memory) RETURN count(*) AS n").single()["n"]
        print(f"\nMemory 节点：{mem_cnt}  条，AWARE_OF 关系：{rel_cnt} 条。")

    driver.close()
    print("\n完成。你可以在应用里直接使用这些角色与记忆。")

if __name__ == "__main__":
    main()
