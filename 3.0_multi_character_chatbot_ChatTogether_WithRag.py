# -*- coding: utf-8 -*-
import os
import re
import json
import random
from typing import List, Dict, Optional

import streamlit as st
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable
from dotenv import load_dotenv
from google import genai

# ===================== 环境与配置 =====================
load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "").strip() or None  # 可选

GEMINI_PROJECT = os.getenv("GEMINI_PROJECT_ID")
GEMINI_LOCATION = os.getenv("GEMINI_LOCATION", "us-central1")
GEMINI_MODEL = os.getenv("GEMINI_MODEL_STD", "gemini-2.0-flash-lite")

# 嵌入模型（用于 Memory 与临时查询）
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-004")

# UI 与记忆窗口
CONV_UI_MAX_TURNS = 100             # 前端展示最多多少条
RB_MAX_TURNS_PER_ROLE = 30          # 每角色 RecentBuffer 的滚动窗口大小
PROTAGONIST_NAME = "主角（你）"
PROTAGONIST_DEFAULT_DESIRE = 0.5

# ——给模型看的主角别名（避免“你”带来的指代混淆）
PROTAGONIST_PROMPT_ALIAS = "主角"  # 你也可以改成“玩家”

# 检索参数
RETR_TOPK = 5
RETR_WEIGHTS = [0.5, 0.3, 0.2]  # 最近 -> 更早

# ===================== 小工具 =====================
def role_alias_for_prompt(name: str) -> str:
    """将‘主角（你）’映射为给模型看的别名，其它名字保持不变。"""
    return PROTAGONIST_PROMPT_ALIAS if name == PROTAGONIST_NAME else name

def map_turn_for_prompt(turn: Dict) -> Dict:
    """把 turn 的角色名替换为给模型看的别名。"""
    return {"role": role_alias_for_prompt(turn.get("role", "")),
            "content": turn.get("content", "")}

def sanitize_line(raw: str, speaker_name: str, all_names: List[str]):
    """去名字前缀/引号/长省略号，仅保留台词本身。空则返回空字符串。"""
    if not raw:
        return ""
    text = raw.strip()
    text = text.strip('“”"\'').strip()
    if all_names:
        name_pat = "|".join(re.escape(n) for n in sorted(set(all_names), key=len, reverse=True))
        for _ in range(2):
            text = re.sub(rf'^(?:{name_pat})\s*[:：]\s*', "", text)
    text = re.sub(r'[。！？!?…]{3,}', '…', text)
    return text

def clamp_tail(lst, n: int):
    return lst[-n:] if len(lst) > n else lst

def weighted_next_speaker(weights: dict, exclude: Optional[str] = None) -> str:
    """按权重随机下一位说话者，可选排除上一位避免连说。"""
    items = [(k, max(0.0, float(v))) for k, v in weights.items() if (max(0.0, float(v)) > 0.0 and k != exclude)]
    if not items:
        items = [(k, max(0.0, float(v))) for k, v in weights.items() if max(0.0, float(v)) > 0.0]
    if not items:
        return random.choice(list(weights.keys()))
    total = sum(w for _, w in items)
    r = random.uniform(0, total)
    acc = 0.0
    for name, w in items:
        acc += w
        if r <= acc:
            return name
    return items[-1][0]

def make_conv_id(participants: List[str], title: str) -> str:
    """统一生成 conv_id（包含主角，参与者去重排序，标题规范化）"""
    names = sorted(set(participants + [PROTAGONIST_NAME]))
    title_norm = re.sub(r"\s+", " ", (title or "").strip()) or "untitled"
    return f"conv::{title_norm}::{'|'.join(names)}"

def format_query_text(turn: Dict, speaker_pool: List[str]) -> str:
    """
    历史回合 -> 临时查询文本（用于向量检索）；
    这里用别名喂给模型，避免把“主角（你）”误解为当前“你”。
    """
    orig_role = turn.get("role", "")
    role = role_alias_for_prompt(orig_role)
    others = [role_alias_for_prompt(n) for n in speaker_pool if n != orig_role]
    others_str = "/".join(others) if others else "众人"
    content = turn.get("content", "")
    return f"{role}对全体，{others_str}说：{content}".strip()

def parse_memory_triggers(text: str) -> List[str]:
    """
    从模型回复中提取 Memory 触发：
    [[MEMORY]] <文本> [[/MEMORY]]
    允许多条
    """
    if not text:
        return []
    try:
        pattern = r"\[\[MEMORY\]\]\s*(.+?)\s*\[\[/MEMORY\]\]"
        return [m.strip() for m in re.findall(pattern, text, flags=re.S)]
    except Exception:
        return []

_MEM_BLOCK_RE = re.compile(r"\s*\[\[MEMORY\]\]\s*.+?\s*\[\[/MEMORY\]\]\s*", flags=re.S)

def strip_memory_blocks(text: str) -> str:
    """移除所有 [[MEMORY]]...[[/MEMORY]] 段，并压缩空白。"""
    if not text:
        return ""
    cleaned = _MEM_BLOCK_RE.sub(" ", text)
    return re.sub(r"\s+\n|\n\s+", "\n", cleaned).strip()

def normalize_memory_content(raw: str, speaker_name: str) -> str:
    """把 LLM 触发的记忆文本规范为『<说话者>曾说：<内容>』"""
    if not raw:
        return f"{speaker_name}曾说：……"
    text = raw.strip().strip('“”"\'').strip()
    if re.match(rf"^{re.escape(speaker_name)}\s*(曾说|说)\s*[:：]", text):
        return text
    return f"{speaker_name}曾说：{text}"

# ===================== Neo4j 存储层（Conversation 单节点仅存 history_json + Memory） =====================
def _bolt_fallback(uri: str) -> str:
    return (uri.replace("neo4j+ssc://", "bolt+ssc://")
               .replace("neo4j+s://", "bolt+s://")
               .replace("neo4j://", "bolt://", 1))

class Neo4jStore:
    """
    模型：
      (:Character {name, persona, worldview, desire})
        -[:HAS_RECENT_BUFFER]-> (:RecentBuffer {id, lines_json})

      (:Conversation {id, title, history_json, created_at, updated_at})
      (:Character)-[:PARTICIPATED_IN {role:"npc"|"protagonist"}]->(:Conversation)

      (:Memory {content, embedding})
      (:Character)-[:AWARE_OF]->(:Memory)
      (:Conversation)-[:MENTIONED_MEMORY]->(:Memory)
    """
    MEM_INDEX = "mem_vec"  # 向量索引名（若版本不支持，会静默失败）
    mem_index_ready = False
    mem_index_dims = None

    def __init__(self):
        self.driver = self._mk_driver()
        self._init_schema()

    def _mk_driver(self):
        if not (NEO4J_URI and NEO4J_USER and NEO4J_PASSWORD):
            raise RuntimeError("Missing Neo4j env vars")
        try:
            drv = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
            drv.verify_connectivity()
            return drv
        except ServiceUnavailable as e:
            if "routing" in str(e).lower():
                bolt_uri = _bolt_fallback(NEO4J_URI)
                drv = GraphDatabase.driver(bolt_uri, auth=(NEO4J_USER, NEO4J_PASSWORD))
                drv.verify_connectivity()
                st.info(f"Neo4j 路由失败，已自动从 {NEO4J_URI} 降级为 {bolt_uri}")
                return drv
            raise

    def _init_schema(self):
        with self.driver.session(database=NEO4J_DATABASE) as s:
            s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Character) REQUIRE c.name IS UNIQUE")
            s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (rb:RecentBuffer) REQUIRE rb.id IS UNIQUE")
            s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (conv:Conversation) REQUIRE conv.id IS UNIQUE")
            s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (m:Memory) REQUIRE m.content IS UNIQUE")
        # 向量索引在第一次拿到 embedding 维度后再尝试创建

    # ----- Memory 向量索引管理 -----
    def ensure_mem_index(self, dims: int):
        """按需创建 Memory 的向量索引；低版本不支持会静默失败。"""
        if self.mem_index_ready and self.mem_index_dims == dims:
            return
        try:
            with self.driver.session(database=NEO4J_DATABASE) as s:
                s.run(f"""
                CREATE VECTOR INDEX {self.MEM_INDEX} IF NOT EXISTS
                FOR (m:Memory) ON (m.embedding)
                OPTIONS {{
                  indexConfig: {{
                    `vector.dimensions`: {int(dims)},
                    `vector.similarity_function`: 'cosine'
                  }}
                }}
                """)
            self.mem_index_ready = True
            self.mem_index_dims = int(dims)
        except Exception:
            self.mem_index_ready = False
            self.mem_index_dims = None

    # ----- 角色 & RecentBuffer -----
    def ensure_character(self, name: str, persona: str = "", worldview: str = "", desire: float = 0.3):
        with self.driver.session(database=NEO4J_DATABASE) as s:
            s.run("""
                MERGE (c:Character {name:$name})
                  ON CREATE SET c.created_at = datetime()
                SET c.persona=$persona, c.worldview=$worldview, c.desire=$desire, c.updated_at=datetime()
            """, name=name, persona=persona, worldview=worldview, desire=float(desire))
            self.ensure_role_recentbuffer(name)

    def get_characters(self):
        with self.driver.session(database=NEO4J_DATABASE) as s:
            res = s.run("""
                MATCH (c:Character)
                RETURN c.name AS name, c.persona AS persona, c.worldview AS worldview, coalesce(c.desire,0.3) AS desire
                ORDER BY c.name
            """)
            # 一次性取完，避免二次迭代导致 ResultConsumedError
            return res.data()

    def update_character_desire(self, name: str, desire: float):
        with self.driver.session(database=NEO4J_DATABASE) as s:
            s.run("MATCH (c:Character {name:$n}) SET c.desire=$d, c.updated_at=datetime()", n=name, d=float(desire))

    def _rb_id(self, name: str) -> str:
        return f"rb::{name}"

    def ensure_role_recentbuffer(self, name: str):
        with self.driver.session(database=NEO4J_DATABASE) as s:
            s.run("""
                MERGE (c:Character {name:$name})
                MERGE (rb:RecentBuffer {id:$rb})
                  ON CREATE SET rb.lines_json="[]", rb.created_at=datetime()
                MERGE (c)-[:HAS_RECENT_BUFFER]->(rb)
            """, name=name, rb=self._rb_id(name))

    def get_role_recentbuffer(self, name: str) -> List[Dict]:
        with self.driver.session(database=NEO4J_DATABASE) as s:
            rec = s.run("""
                MATCH (:Character {name:$name})-[:HAS_RECENT_BUFFER]->(rb:RecentBuffer {id:$rb})
                RETURN rb.lines_json AS j
            """, name=name, rb=self._rb_id(name)).single()
        if rec and rec["j"]:
            try:
                return json.loads(rec["j"])
            except Exception:
                return []
        return []

    def append_turn_to_roles_recent(self, role_names: List[str], turn: Dict, keep: int = RB_MAX_TURNS_PER_ROLE):
        """把一条新 turn 追加到这些角色的 RecentBuffer（只更新本会话的参与者）。"""
        with self.driver.session(database=NEO4J_DATABASE) as s:
            for n in role_names:
                rec = s.run("""
                    MERGE (c:Character {name:$name})
                    MERGE (rb:RecentBuffer {id:$rb})
                      ON CREATE SET rb.lines_json="[]", rb.created_at=datetime()
                    MERGE (c)-[:HAS_RECENT_BUFFER]->(rb)
                    RETURN rb.lines_json AS j
                """, name=n, rb=self._rb_id(n)).single()
                lines = []
                if rec and rec["j"]:
                    try:
                        lines = json.loads(rec["j"]) or []
                    except Exception:
                        lines = []
                lines.append(turn)
                lines = clamp_tail(lines, keep)
                j = json.dumps(lines, ensure_ascii=False)
                s.run("""
                    MATCH (:Character {name:$name})-[:HAS_RECENT_BUFFER]->(rb:RecentBuffer {id:$rb})
                    SET rb.lines_json=$j, rb.updated=datetime()
                """, name=n, rb=self._rb_id(n), j=j)

    # ----- Conversation（单节点仅存 history_json） + 参与关系 -----
    def ensure_conversation_with_participants(self, conv_id: str, title: str, participants: List[Dict]):
        with self.driver.session(database=NEO4J_DATABASE) as s:
            s.run("""
                MERGE (conv:Conversation {id:$cid})
                  ON CREATE SET conv.created_at = datetime()
                SET conv.title = coalesce($title, conv.title),
                    conv.updated_at = datetime()
            """, cid=conv_id, title=title or "")
            s.run("""
                UNWIND $ps AS p
                MERGE (c:Character {name:p.name})
                MERGE (conv:Conversation {id:$cid})
                MERGE (c)-[r:PARTICIPATED_IN]->(conv)
                SET r.role = p.role_tag
            """, cid=conv_id, ps=participants)

    def set_conversation_history(self, conv_id: str, turns: List[Dict]):
        j = json.dumps(turns, ensure_ascii=False)
        with self.driver.session(database=NEO4J_DATABASE) as s:
            s.run("""
                MERGE (conv:Conversation {id:$cid})
                SET conv.history_json = $j,
                    conv.updated_at = datetime()
            """, cid=conv_id, j=j)

    def get_conversation_history(self, conv_id: str) -> List[Dict]:
        with self.driver.session(database=NEO4J_DATABASE) as s:
            rec = s.run("MATCH (conv:Conversation {id:$cid}) RETURN conv.history_json AS j", cid=conv_id).single()
        if rec and rec["j"]:
            try:
                return json.loads(rec["j"])
            except Exception:
                return []
        return []

    # ----- Memory Upsert & Link -----
    def upsert_memory(self, content: str, embedding: List[float]):
        """基于 content 幂等创建 Memory；首次创建时写入 embedding。"""
        if embedding and isinstance(embedding, list):
            self.ensure_mem_index(len(embedding))
        with self.driver.session(database=NEO4J_DATABASE) as s:
            s.run("""
                MERGE (m:Memory {content:$content})
                  ON CREATE SET m.embedding = $embedding
            """, content=content, embedding=embedding)

    def link_awareness(self, role_names: List[str], mem_content: str):
        """将一组角色连到指定 Memory（AWARE_OF）。"""
        with self.driver.session(database=NEO4J_DATABASE) as s:
            s.run("""
                UNWIND $names AS n
                MATCH (c:Character {name:n})
                MATCH (m:Memory {content:$content})
                MERGE (c)-[:AWARE_OF]->(m)
            """, names=role_names, content=mem_content)

    def link_conversation_to_memory(self, conv_id: str, mem_content: str):
        """为会话与记忆建立有向边：(:Conversation {id})-[:MENTIONED_MEMORY]->(:Memory {content})"""
        with self.driver.session(database=NEO4J_DATABASE) as s:
            s.run("""
                MATCH (conv:Conversation {id:$cid})
                MATCH (m:Memory {content:$content})
                MERGE (conv)-[:MENTIONED_MEMORY]->(m)
            """, cid=conv_id, content=mem_content)

    # ----- Memory 检索（sum of 3 queries with weights） -----
    def query_memories_sum_scores(self, speaker: str, q_embs: List[List[float]], weights: List[float], k: int):
        """
        在 speaker 可见的 Memory 中做向量检索。
        q_embs 最多 3 个，分别按 weights 加权后对同一 node 求和。
        返回：[{content, total_score}]
        """
        if not q_embs:
            return []
        if not self.mem_index_ready:
            return []
        payloads = [(q, w) for q, w in zip(q_embs, weights) if isinstance(q, list) and q]
        if not payloads:
            return []
        parts = []
        params = {"speaker": speaker, "k": int(k)}
        for i, (q, w) in enumerate(payloads, 1):
            qi = f"q{i}"
            wi = f"w{i}"
            params[qi] = q
            params[wi] = float(w)
            parts.append(f"""
CALL db.index.vector.queryNodes('{self.MEM_INDEX}', $k, ${qi}) YIELD node, score
MATCH (:Character {{name:$speaker}})-[:AWARE_OF]->(node)
RETURN node, score*${wi} AS part
""")
        unioned = "\nUNION ALL\n".join(parts)
        cypher = f"""
CALL {{
{unioned}
}}
RETURN node.content AS content, sum(part) AS total_score
ORDER BY total_score DESC
LIMIT $k
"""
        with self.driver.session(database=NEO4J_DATABASE) as s:
            try:
                res = s.run(cypher, **params)
                # 一次性取完
                return res.data()
            except Exception:
                return []

# ===================== Google GenAI 客户端（文本 & 嵌入） =====================
client = genai.Client(vertexai=True, project=GEMINI_PROJECT, location=GEMINI_LOCATION)

def embed_text(text: str) -> List[float]:
    """调用嵌入模型，把文本转成向量。失败时返回空列表。"""
    try:
        if not text or not text.strip():
            return []
        resp = client.models.embed_content(model=EMBED_MODEL, contents=text.strip())
        vec = None
        if hasattr(resp, "embeddings") and resp.embeddings:
            emb0 = resp.embeddings[0]
            vec = getattr(emb0, "values", None) or getattr(emb0, "embedding", None)
        if vec is None and hasattr(resp, "values"):
            vec = resp.values
        return list(vec) if vec else []
    except Exception:
        return []

# ===================== 生成逻辑（加入 Memory 检索 + 触发） =====================
def build_memory_blocks_for_speaker(speaker_name: str, recent_turns: List[Dict], speaker_pool: List[str]) -> List[str]:
    """把最近三条回合→临时查询文本→向量→检索 speaker 可见的 Memory。"""
    turns = list(reversed(recent_turns[-3:]))  # [last, last-1, last-2]
    q_texts = [format_query_text(t, speaker_pool) for t in turns]
    q_embs = [embed_text(qt) for qt in q_texts]
    rows = store.query_memories_sum_scores(speaker_name, q_embs, RETR_WEIGHTS, RETR_TOPK)
    return [r["content"] for r in rows]

def join_memory_lines(mem_lines: List[str], max_n: int = 3) -> str:
    if not mem_lines:
        return ""
    mem_lines = mem_lines[:max_n]
    return "\n".join(f"- {m}" for m in mem_lines)

def integrate_memory_into_prompt(base_prompt: str, mem_lines: List[str]) -> str:
    if not mem_lines:
        return base_prompt
    mem_block = join_memory_lines(mem_lines)
    prefix = "你已知的关键信息（可能有用）：\n" + f"{mem_block}\n\n"
    return prefix + base_prompt

def handle_memory_triggers_from_reply(reply_text: str, audience_names: List[str], speaker_name: str, conv_id: str):
    """
    解析回复中的 Memory 触发并写库：
      - 统一规范为『<speaker>曾说：<内容>』
      - MERGE (:Memory {content}) ON CREATE SET embedding
      - audience_names（本场所有在场者+主角）均连上 AWARE_OF
      - (:Conversation {id: conv_id})-[:MENTIONED_MEMORY]->(:Memory)
    """
    triggered = parse_memory_triggers(reply_text)
    if not triggered:
        return
    for raw_content in triggered:
        content = normalize_memory_content(raw_content, speaker_name)
        emb = embed_text(content)
        store.upsert_memory(content, emb)
        store.link_awareness(audience_names, content)
        store.link_conversation_to_memory(conv_id, content)

# ===================== 文本生成（加入 Memory 注入 & 触发机制） =====================
def _ctx_block_from_rb(rb_history: List[Dict]) -> str:
    if not rb_history:
        return "（无历史；请以自然口吻开场，先抛出一个与话题提示相关的轻松问题。）"
    safe = [map_turn_for_prompt(t) for t in rb_history]
    return "\n".join([f"{t['role']}: {t['content']}" for t in safe])

def gen_with_role_recentbuffer(
    speaker_name: str,
    speaker_persona: str,
    worldview: str,
    rb_history: List[Dict],
    topic_hint: str = "",
    mem_lines: Optional[List[str]] = None
) -> str:
    ctx = _ctx_block_from_rb(rb_history)
    last_role = rb_history[-1]["role"] if rb_history else ""
    last_text = rb_history[-1]["content"] if rb_history else ""
    last_role_alias = role_alias_for_prompt(last_role)
    base = (
        f"世界观：{worldview or '（未提供）'}\n"
        f"你的身份：{speaker_name}（人设：{speaker_persona or '（未提供）'}）\n"
        f"注意：下方片段中的“{PROTAGONIST_PROMPT_ALIAS}”指玩家，不是你。\n"
        f"话题提示：{topic_hint or '（无）'}\n\n"
        f"== 你参与的最近对话片段（含所有人台词） ==\n{ctx}\n"
        f"最近一条：{last_role_alias or '（无）'}：{last_text or '（无）'}\n\n"
        f"只输出你的台词，不要名字前缀，不要加引号；优先自然回应最近一条，也可按人设转移但要自洽。\n"
        f"【记忆触发】仅当出现**新且长期重要**的信息，且不与“你已知的关键信息”重复时，"
        f"在最后一行输出：[[MEMORY]] 一句话 [[/MEMORY]]（最多一条，否则不要输出）。"
    )
    prompt = integrate_memory_into_prompt(base, mem_lines or [])
    try:
        resp = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        return (getattr(resp, "text", "") or "").strip()
    except Exception as e:
        return f"(模型调用出错：{e})"

def suggest_for_protagonist(
    protagonist_persona: str,
    worldview: str,
    rb_history: List[Dict],
    topic_hint: str = "",
    mem_lines: Optional[List[str]] = None
) -> str:
    ctx = _ctx_block_from_rb(rb_history)
    last_role = rb_history[-1]["role"] if rb_history else ""
    last_text = rb_history[-1]["content"] if rb_history else ""
    last_role_alias = role_alias_for_prompt(last_role)
    base = (
        f"世界观：{worldview or '（未提供）'}\n"
        f"玩家角色：{PROTAGONIST_NAME}（人设：{protagonist_persona or '（未提供）'}）\n"
        f"话题提示：{topic_hint or '（无）'}\n\n"
        f"== 你参与的最近片段 ==\n{ctx}\n"
        f"最近一条：{last_role_alias or '（无）'}：{last_text or '（无）'}\n\n"
        f"给出一条建议台词（第一人称“我”，1-2句；不要加引号/名字前缀）。\n"
        f"【记忆触发】仅当出现**新且长期重要**的信息，且不与“你已知的关键信息”重复时，"
        f"在最后一行输出：[[MEMORY]] 一句话 [[/MEMORY]]（最多一条，否则不要输出）。"
    )
    prompt = integrate_memory_into_prompt(base, mem_lines or [])
    try:
        resp = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        return (getattr(resp, "text", "") or "").strip()
    except Exception as e:
        return f"(建议生成出错：{e})"

# ===================== Streamlit UI =====================
st.set_page_config(page_title="多角色聊天 · Conversation+Memory", page_icon="🤖", layout="centered")
st.title("🤖 多角色多人聊天（Conversation 单节点 + Memory 检索 + 触发写入）")
st.caption(f"BUILD: 2025-08-17 · CONV_UI_MAX={CONV_UI_MAX_TURNS} · RB_MAX={RB_MAX_TURNS_PER_ROLE} · EMBED_MODEL={EMBED_MODEL}")

store = Neo4jStore()

# ---- 载入/刷新角色 ----
if "characters" not in st.session_state:
    st.session_state.characters = store.get_characters()
if st.button("刷新角色列表", key="btn_refresh_chars"):
    st.session_state.characters = store.get_characters()

# 初始化会话锁状态
if "conv_locked" not in st.session_state:
    st.session_state.conv_locked = False
if "conv_id" not in st.session_state:
    st.session_state.conv_id = None
if "locked_participants" not in st.session_state:
    st.session_state.locked_participants = []

# 不把主角放进 NPC 选择里
npc_chars = [c for c in st.session_state.characters if c["name"] != PROTAGONIST_NAME]
char_names = [c["name"] for c in npc_chars]

# ---- 选择参与者 + 标题（未锁定时才有效）----
participants_selected = st.multiselect(
    "选择参与聊天的 NPC（可多选）",
    options=char_names,
    default=char_names[:2] if (not st.session_state.conv_locked and len(char_names) >= 2) else [],
    key="ms_participants"
)
conv_title = st.text_input("（可选）会话标题", value="", key="ti_title")

# ---- 主角与话题 ----
st.markdown("### 主角与话题")
with st.container():
    st.text(f"{PROTAGONIST_NAME} 的发言欲望固定为 {PROTAGONIST_DEFAULT_DESIRE}。")
    protagonist_persona = st.text_input("主角人设", value="果断、善良、略带调皮的冒险者", key="ti_protagonist")
    protagonist_worldview = st.text_input("共同世界观（为空默认取第一个 NPC 的世界观）", value="", key="ti_world")
topic_hint = st.text_input("（可选）话题提示/场景描述", value="", key="ti_topic")

# ---- 发言欲望 ----
st.markdown("### 发言欲望（0.0—1.0）")
if "desires" not in st.session_state:
    st.session_state.desires = {}
participants_for_scene = st.session_state.locked_participants if st.session_state.conv_locked else participants_selected
for name in participants_for_scene:
    c = next((c for c in npc_chars if c["name"] == name), None)
    if not c:
        continue
    cur = float(c.get("desire", 0.3))
    st.session_state.desires[name] = st.slider(
        f"{name} 的发言欲望", 0.0, 1.0, value=cur, step=0.05, key=f"sl_des_{name}"
    )
if participants_for_scene and st.button("保存欲望", key="btn_save_desires"):
    for name in participants_for_scene:
        store.update_character_desire(name, float(st.session_state.desires.get(name, 0.3)))
    st.session_state.characters = store.get_characters()
    st.success("已保存。")

# ---- 会话首次创建：延迟到第一次真正发言时 ----
def create_conv_if_needed():
    if st.session_state.conv_locked:
        return
    if not participants_selected:
        st.warning("请先选择至少一个 NPC 才能开始对话。")
        return
    cid = make_conv_id(participants_selected, conv_title)
    st.session_state.conv_id = cid
    st.session_state.conv_locked = True
    st.session_state.locked_participants = list(participants_selected)
    store.ensure_character(PROTAGONIST_NAME, protagonist_persona, protagonist_worldview or "", PROTAGONIST_DEFAULT_DESIRE)
    for n in st.session_state.locked_participants:
        store.ensure_character(n)
    participants_payload = (
        [{"name": n, "role_tag": "npc"} for n in st.session_state.locked_participants] +
        [{"name": PROTAGONIST_NAME, "role_tag": "protagonist"}]
    )
    store.ensure_conversation_with_participants(cid, conv_title, participants_payload)
    st.session_state.chat_history = store.get_conversation_history(cid) or []

def reset_conversation():
    for k in ["conv_locked", "conv_id", "history_loaded_for", "chat_history",
              "awaiting_player_input", "player_suggested_line", "player_line_input",
              "player_memory_trigger_raw", "locked_participants"]:
        st.session_state.pop(k, None)
    st.session_state.conv_locked = False
    st.session_state.locked_participants = []

col_lock, col_new = st.columns(2)
with col_lock:
    if st.button("开始对话 / 锁定参与者", key="btn_lock"):
        create_conv_if_needed()
        st.rerun()
with col_new:
    if st.button("另起新会话", key="btn_new"):
        reset_conversation()
        st.rerun()

if st.session_state.conv_locked:
    st.info(f"已锁定会话：{st.session_state.conv_id}\n"
            f"参与者：{', '.join(st.session_state.locked_participants)}\n"
            f"如要变更参与者，请点击“另起新会话”。")

# ---- 读取会话历史（仅锁定后）----
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if st.session_state.conv_locked and st.session_state.conv_id:
    if st.session_state.get("history_loaded_for") != st.session_state.conv_id:
        st.session_state.chat_history = store.get_conversation_history(st.session_state.conv_id) or []
        st.session_state.history_loaded_for = st.session_state.conv_id

# ---- 展示历史 ----
st.markdown("### 对话历史")
for t in st.session_state.chat_history[-CONV_UI_MAX_TURNS:]:
    st.chat_message("assistant").markdown(f"**{t.get('role','角色')}**：{t.get('content','')}")

# ---- 新建角色 ----
with st.expander("＋ 新建角色"):
    with st.form("add_char_form"):
        name = st.text_input("角色名", key="ti_new_name")
        persona = st.text_area("人设", key="ta_new_persona")
        worldview = st.text_area("世界观", key="ta_new_world")
        desire = st.slider("初始发言欲望", 0.0, 1.0, value=0.3, step=0.05, key="sl_new_desire")
        submitted = st.form_submit_button("添加角色")
        if submitted and name and persona:
            store.ensure_character(name, persona, worldview, desire)
            st.session_state.characters = store.get_characters()
            st.success(f"角色 {name} 已添加！")

# ---- 控制区 ----
st.markdown("### 控制")
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    run_once = st.button("生成下一句", key="btn_run_once")
with c2:
    clear_btn = st.button("清空对话（仅本会话）", key="btn_clear")
with c3:
    run_multi_skip = st.button("连发 3 句（跳过主角）", key="btn_multi_skip")
with c4:
    run_multi_allow = st.button("连发 3 句（允许主角）", key="btn_multi_allow")
with c5:
    reroll_btn = st.button("换个说话者", key="btn_reroll")

# ---- 活跃参与者（含主角）----
scene_participants = st.session_state.locked_participants if st.session_state.conv_locked else participants_selected
active_participants = []
for name in scene_participants:
    c = next((c for c in npc_chars if c["name"] == name), None)
    if not c:
        continue
    active_participants.append({
        "name": c["name"],
        "persona": c.get("persona", ""),
        "worldview": c.get("worldview", ""),
        "desire": float(st.session_state.desires.get(name, c.get("desire", 0.3)))
    })
active_participants.append({
    "name": PROTAGONIST_NAME,
    "persona": protagonist_persona,
    "worldview": protagonist_worldview or (active_participants[0]["worldview"] if active_participants else ""),
    "desire": PROTAGONIST_DEFAULT_DESIRE
})

def current_worldview() -> str:
    return protagonist_worldview or (active_participants[0]["worldview"] if active_participants else "")

def all_names_in_scene():
    # 增补“主角”别名，保证 sanitize_line 能去掉“主角:”这类前缀
    names = [p["name"] for p in active_participants]
    if PROTAGONIST_NAME in names and PROTAGONIST_PROMPT_ALIAS != PROTAGONIST_NAME:
        names.append(PROTAGONIST_PROMPT_ALIAS)
    return names

def persist_everything(speaker: str, text: str, raw_reply_for_trigger: Optional[str] = None):
    """
    把新 turn 写入：首次发言时创建会话；随后写 Conversation.history_json；
    并更新每位参与者的 RecentBuffer。
    同时处理 LLM 触发的 Memory（如有）。
    要求：传入的 text 必须是已剔除 MEMORY 段后的最终台词。
    """
    create_conv_if_needed()
    if not st.session_state.conv_locked:
        return

    if text and text.strip():
        turn = {"role": speaker, "content": text.strip()}
        st.session_state.chat_history.append(turn)
        store.set_conversation_history(st.session_state.conv_id, st.session_state.chat_history)
        roles_to_update = list(dict.fromkeys(st.session_state.locked_participants + [PROTAGONIST_NAME]))
        store.append_turn_to_roles_recent(roles_to_update, turn, keep=RB_MAX_TURNS_PER_ROLE)

    # Memory 触发（即使没有可显示台词也会写 Memory）
    if raw_reply_for_trigger:
        audience = list(dict.fromkeys(st.session_state.locked_participants + [PROTAGONIST_NAME]))
        handle_memory_triggers_from_reply(
            raw_reply_for_trigger,
            audience,
            speaker,
            st.session_state.conv_id  # 让会话指向记忆
        )

# ===================== 生成一步（含 Memory 检索注入） =====================
def one_step_generate(skip_protagonist: bool = False, exclude_name: Optional[str] = None):
    if not scene_participants:
        st.warning("请至少选择一个 NPC。")
        return

    weights = {p["name"]: float(p["desire"]) for p in active_participants}
    last_speaker = st.session_state.chat_history[-1]["role"] if st.session_state.chat_history else None
    if skip_protagonist:
        weights[PROTAGONIST_NAME] = 0.0
    if exclude_name:
        weights[exclude_name] = 0.0

    speaker_name = weighted_next_speaker(weights, exclude=last_speaker)
    sp = next(p for p in active_participants if p["name"] == speaker_name)

    rb_history = store.get_role_recentbuffer(sp["name"])

    mem_lines = build_memory_blocks_for_speaker(
        speaker_name, st.session_state.chat_history, [p["name"] for p in active_participants]
    )

    if sp["name"] == PROTAGONIST_NAME:
        sugg_raw = suggest_for_protagonist(
            protagonist_persona=protagonist_persona,
            worldview=current_worldview(),
            rb_history=rb_history,
            topic_hint=topic_hint,
            mem_lines=mem_lines
        )
        # 展示给玩家的建议需要去掉 MEMORY 段
        clean_sugg = strip_memory_blocks(sugg_raw)
        st.session_state.awaiting_player_input = True
        st.session_state.player_suggested_line = sanitize_line(clean_sugg, PROTAGONIST_NAME, all_names_in_scene())
        st.session_state.player_line_input = st.session_state.player_suggested_line
        # 保存原始建议文本，仅用于触发记忆
        st.session_state.player_memory_trigger_raw = sugg_raw
        return "player"

    # NPC：生成 + 触发记忆；把剔除 MEMORY 后的文本写入
    raw = gen_with_role_recentbuffer(
        speaker_name=sp["name"],
        speaker_persona=sp.get("persona", ""),
        worldview=current_worldview(),
        rb_history=rb_history,
        topic_hint=topic_hint,
        mem_lines=mem_lines
    )
    cleaned = strip_memory_blocks(raw)
    line = sanitize_line(cleaned, sp["name"], all_names_in_scene()).strip()

    # 若只包含 MEMORY 而无台词，则仅写 Memory，不入历史
    if not line:
        audience = list(dict.fromkeys(st.session_state.locked_participants + [PROTAGONIST_NAME]))
        handle_memory_triggers_from_reply(raw, audience, sp["name"], st.session_state.conv_id)
        return "npc"

    persist_everything(sp["name"], line, raw_reply_for_trigger=raw)
    return "npc"

# ===================== 触发按钮逻辑 =====================
if reroll_btn:
    one_step_generate(skip_protagonist=True, exclude_name=PROTAGONIST_NAME)
    st.rerun()

if run_once:
    if not st.session_state.get("awaiting_player_input"):
        one_step_generate()
    st.rerun()

if run_multi_skip:
    for _ in range(3):
        one_step_generate(skip_protagonist=True)
    st.rerun()

if run_multi_allow:
    steps = 0
    while steps < 3:
        res = one_step_generate()
        if res == "player":
            break
        steps += 1
    st.rerun()

if clear_btn:
    st.session_state.chat_history = []
    if st.session_state.conv_locked and st.session_state.conv_id:
        store.set_conversation_history(st.session_state.conv_id, [])
    for k in ["awaiting_player_input", "player_suggested_line", "player_line_input", "player_memory_trigger_raw"]:
        st.session_state.pop(k, None)
    st.rerun()

# ---- 玩家输入区（主角）----
if st.session_state.get("awaiting_player_input"):
    st.markdown("### 轮到你发言（作为：**主角（你）**）")
    st.session_state.player_line_input = st.text_area(
        "你的台词（可编辑）",
        value=st.session_state.get("player_line_input", st.session_state.get("player_suggested_line", "")),
        height=120,
        key="ta_player_line"
    )
    cA, cB, cC = st.columns(3)
    with cA:
        send_player = st.button("发送", key="btn_player_send")
    with cB:
        regen_sugg = st.button("换一个建议", key="btn_player_regen")
    with cC:
        skip_player = st.button("换个说话者", key="btn_player_skip")

    if regen_sugg:
        mem_lines_me = build_memory_blocks_for_speaker(
            PROTAGONIST_NAME, st.session_state.chat_history, [p["name"] for p in active_participants]
        )
        new_sugg_raw = suggest_for_protagonist(
            protagonist_persona=protagonist_persona,
            worldview=current_worldview(),
            rb_history=store.get_role_recentbuffer(PROTAGONIST_NAME),
            topic_hint=topic_hint,
            mem_lines=mem_lines_me
        )
        clean_sugg = strip_memory_blocks(new_sugg_raw)
        st.session_state.player_suggested_line = sanitize_line(clean_sugg, PROTAGONIST_NAME, all_names_in_scene())
        st.session_state.player_line_input = st.session_state.player_suggested_line
        st.session_state.player_memory_trigger_raw = new_sugg_raw
        st.rerun()

    if skip_player:
        st.session_state.awaiting_player_input = False
        for k in ["player_suggested_line", "player_line_input", "player_memory_trigger_raw"]:
            st.session_state.pop(k, None)
        one_step_generate(skip_protagonist=True)
        st.rerun()

    if send_player:
        typed_raw = st.session_state.get("player_line_input", "")
        cleaned = strip_memory_blocks(typed_raw)
        line = sanitize_line(cleaned, PROTAGONIST_NAME, all_names_in_scene()).strip()

        # 组合触发文本：玩家输入 + 建议原文（若存在）
        trigger_blob = (st.session_state.get("player_memory_trigger_raw", "") or "") + "\n" + (typed_raw or "")

        if not line:
            # 仅包含 MEMORY，写入记忆但不入历史
            audience = list(dict.fromkeys(st.session_state.locked_participants + [PROTAGONIST_NAME]))
            handle_memory_triggers_from_reply(trigger_blob, audience, PROTAGONIST_NAME, st.session_state.conv_id)
        else:
            persist_everything(PROTAGONIST_NAME, line, raw_reply_for_trigger=trigger_blob)

        st.session_state.awaiting_player_input = False
        for k in ["player_suggested_line", "player_line_input", "player_memory_trigger_raw"]:
            st.session_state.pop(k, None)
        st.rerun()

# ---- 维护工具（可选）：清理空会话 ----
with st.expander("🧹 清理工具"):
    if st.button("删除“空会话”（history_json 为 NULL 或 '[]'）", key="btn_cleanup"):
        with store.driver.session(database=NEO4J_DATABASE) as s:
            s.run("""
                MATCH (conv:Conversation)
                WHERE conv.history_json IS NULL OR conv.history_json = "[]"
                DETACH DELETE conv
            """)
        st.success("已清理空会话。")

# 运行备注：
# streamlit run d:/NewProjects/LowCostChattingBot/npc-graph-chat/3.0_multi_character_chatbot_ChatTogether_WithRag.py
