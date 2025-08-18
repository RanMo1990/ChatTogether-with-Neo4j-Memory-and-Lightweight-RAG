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
CONV_UI_MAX_TURNS = 100
RB_MAX_TURNS_PER_ROLE = 30
PROTAGONIST_NAME = "主角（你）"
PROTAGONIST_DEFAULT_DESIRE = 0.5
NARRATOR_NAME = "旁白"

# 检索参数
RETR_TOPK = 5
RETR_WEIGHTS = [0.5, 0.3, 0.2]  # 最近 -> 更早

# 剧情推进/动作偏置参数
ACTION_BIAS_TURNS = 2          # 最近连续多少轮没有动作后，强烈要求本轮给动作
ACTION_BIAS_FORCE_TEXT = "【剧情推进】最近几轮缺少真实动作，请给出明确行动并务必追加 [[ACT]] 动作简述 [[/ACT]]。"
ACTION_BIAS_SOFT_TEXT  = "若能推进剧情，优先给出可执行行动，并在末行追加 [[ACT]] 动作简述 [[/ACT]]。"

# ===================== 小工具 =====================
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
    names = sorted(set(participants + [PROTAGONIST_NAME]))
    title_norm = re.sub(r"\s+", " ", (title or "").strip()) or "untitled"
    return f"conv::{title_norm}::{'|'.join(names)}"

def format_query_text(turn: Dict, speaker_pool: List[str]) -> str:
    role = turn.get("role", "")
    content = turn.get("content", "")
    others = [n for n in speaker_pool if n != role]
    others_str = "/".join(others) if others else "众人"
    return f"{role}对全体，{others_str}说：{content}".strip()

# ====== 记忆触发块 ======
def parse_memory_triggers(text: str) -> List[str]:
    if not text:
        return []
    try:
        pattern = r"\[\[MEMORY\]\]\s*(.+?)\s*\[\[/MEMORY\]\]"
        return [m.strip() for m in re.findall(pattern, text, flags=re.S)]
    except Exception:
        return []

_MEM_BLOCK_RE = re.compile(r"\s*\[\[MEMORY\]\]\s*.+?\s*\[\[/MEMORY\]\]\s*", flags=re.S)

def strip_memory_blocks(text: str) -> str:
    if not text:
        return ""
    cleaned = _MEM_BLOCK_RE.sub(" ", text)
    return re.sub(r"\s+\n|\n\s+", "\n", cleaned).strip()

def normalize_memory_content(raw: str, speaker_name: str) -> str:
    if not raw:
        return f"{speaker_name}曾说：……"
    text = raw.strip().strip('“”"\'').strip()
    if re.match(rf"^{re.escape(speaker_name)}\s*(曾说|说)\s*[:：]", text):
        return text
    return f"{speaker_name}曾说：{text}"

# ====== 动作触发块 ======
def parse_action_triggers(text: str) -> List[str]:
    if not text:
        return []
    try:
        pattern = r"\[\[(?:ACT|ACTION)\]\]\s*(.+?)\s*\[\[/(?:ACT|ACTION)\]\]\s*"
        return [m.strip() for m in re.findall(pattern, text, flags=re.S)]
    except Exception:
        return []

_ACT_BLOCK_RE = re.compile(r"\s*\[\[(?:ACT|ACTION)\]\]\s*.+?\s*\[\[/(?:ACT|ACTION)\]\]\s*", flags=re.S)

def strip_action_blocks(text: str) -> str:
    if not text:
        return ""
    cleaned = _ACT_BLOCK_RE.sub(" ", text)
    return re.sub(r"\s+\n|\n\s+", "\n", cleaned).strip()

# ===================== Neo4j 存储层 =====================
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
      (:Character)-[:PARTICIPATED_IN {role}]->(:Conversation)
      (:Memory {content, embedding})
      (:Character)-[:AWARE_OF]->(:Memory)
      (:Conversation)-[:MENTIONED_MEMORY]->(:Memory)
    """
    MEM_INDEX = "mem_vec"
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

    def ensure_mem_index(self, dims: int):
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

    # 角色 & RB
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

    # 会话 & 参与
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

    # Memory
    def upsert_memory(self, content: str, embedding: List[float]):
        if embedding and isinstance(embedding, list):
            self.ensure_mem_index(len(embedding))
        with self.driver.session(database=NEO4J_DATABASE) as s:
            s.run("""
                MERGE (m:Memory {content:$content})
                  ON CREATE SET m.embedding = $embedding
            """, content=content, embedding=embedding)

    def link_awareness(self, role_names: List[str], mem_content: str):
        with self.driver.session(database=NEO4J_DATABASE) as s:
            s.run("""
                UNWIND $names AS n
                MATCH (c:Character {name:n})
                MATCH (m:Memory {content:$content})
                MERGE (c)-[:AWARE_OF]->(m)
            """, names=role_names, content=mem_content)

    def link_conversation_to_memory(self, conv_id: str, mem_content: str):
        with self.driver.session(database=NEO4J_DATABASE) as s:
            s.run("""
                MATCH (conv:Conversation {id:$cid})
                MATCH (m:Memory {content:$content})
                MERGE (conv)-[:MENTIONED_MEMORY]->(m)
            """, cid=conv_id, content=mem_content)

    # Memory 检索
    def query_memories_sum_scores(self, speaker: str, q_embs: List[List[float]], weights: List[float], k: int):
        if not q_embs or not self.mem_index_ready:
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
                return res.data()
            except Exception:
                return []

# ===================== Google GenAI =====================
client = genai.Client(vertexai=True, project=GEMINI_PROJECT, location=GEMINI_LOCATION)

def embed_text(text: str) -> List[float]:
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
    triggered = parse_memory_triggers(reply_text)
    if not triggered:
        return
    for raw_content in triggered:
        content = normalize_memory_content(raw_content, speaker_name)
        emb = embed_text(content)
        store.upsert_memory(content, emb)
        store.link_awareness(audience_names, content)
        store.link_conversation_to_memory(conv_id, content)

# ===================== 旁白与动作旁白 =====================
def _ctx_block_from_rb(rb_history: List[Dict]) -> str:
    if not rb_history:
        return "（无历史；请以自然口吻开场，先抛出一个与话题提示相关的轻松问题。）"
    return "\n".join([f"{t.get('role','')}: {t.get('content','')}" for t in rb_history])

def gen_opening_narration(worldview: str, topic_hint: str, participants: List[str]) -> str:
    base = (
        f"世界观：{worldview or '（未提供）'}\n"
        f"到场人物：{', '.join(participants)}\n"
        f"场景提示：{topic_hint or '（无）'}\n\n"
        "请写一段开场旁白（2-4句），第三人称、气氛清晰，给出一个短期目标或紧张因子；"
        "如出现重要背景/计划/地点等可长期记忆的信息，可在末行输出：[[MEMORY]] 一句话 [[/MEMORY]]（可选且最多一条）。"
    )
    try:
        resp = client.models.generate_content(model=GEMINI_MODEL, contents=base)
        return (getattr(resp, "text", "") or "").strip()
    except Exception as e:
        return f"(旁白生成出错：{e})"

def gen_action_narration(actor_name: str, action_text: str, worldview: str, rb_history: List[Dict]) -> str:
    ctx = _ctx_block_from_rb(rb_history)
    prompt = (
        f"世界观：{worldview or '（未提供）'}\n"
        f"最近片段（含在场台词）：\n{ctx}\n\n"
        f"现在，{actor_name} 执行动作：{action_text}\n"
        "请用第三人称写出该动作的过程与结果（1-3句），不要编造台词；"
        "如该动作导致重要发现/承诺/计划等，可在末行额外输出：[[MEMORY]] 一句话 [[/MEMORY]]（可选，最多一条）。"
    )
    try:
        resp = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        return (getattr(resp, "text", "") or "").strip()
    except Exception as e:
        return f"(动作旁白生成出错：{e})"

def handle_action_triggers_from_reply(reply_text: str, actor_name: str, worldview: str, conv_id: str) -> bool:
    """解析 [[ACT]]... 生成动作旁白；返回是否出现动作。"""
    acts = parse_action_triggers(reply_text)
    if not acts:
        return False
    for act in acts:
        rb_for_narr = store.get_role_recentbuffer(actor_name)
        narration_raw = gen_action_narration(actor_name, act, worldview, rb_for_narr)
        nar_clean = strip_action_blocks(strip_memory_blocks(narration_raw)).strip()
        if not nar_clean:
            continue
        nar_turn = {"role": NARRATOR_NAME, "content": nar_clean}
        st.session_state.chat_history.append(nar_turn)
        store.set_conversation_history(st.session_state.conv_id, st.session_state.chat_history)
        roles_to_update = list(dict.fromkeys(st.session_state.locked_participants + [PROTAGONIST_NAME]))
        store.append_turn_to_roles_recent(roles_to_update, nar_turn, keep=RB_MAX_TURNS_PER_ROLE)
        audience = list(dict.fromkeys(st.session_state.locked_participants + [PROTAGONIST_NAME]))
        handle_memory_triggers_from_reply(narration_raw, audience, NARRATOR_NAME, conv_id)
    return True

# ===================== 推进提示（核心改动） =====================
def action_bias_text() -> str:
    turns = int(st.session_state.get("turns_since_last_act", 999))
    return ACTION_BIAS_FORCE_TEXT if turns >= ACTION_BIAS_TURNS else ACTION_BIAS_SOFT_TEXT

STORY_DRIVE_RULES = (
    "请满足下列其一（优先靠前项）：\n"
    "1) 提出或回答一个推动剧情的**关键问题**；\n"
    "2) 提出一个**可执行的下一步提议**（谁/何时/何地/做什么）；\n"
    "3) 直接**执行动作**并在末行追加 [[ACT]] 动作简述 [[/ACT]]。\n"
    "避免空洞寒暄和重复。"
)

ACT_USAGE_TIP = "动作标记写法：[[ACT]] 我做了什么（简短祈使/陈述） [[/ACT]]"

# ===================== 文本生成（加入 Memory 注入 & 动作偏置） =====================
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
    base = (
        f"世界观：{worldview or '（未提供）'}\n"
        f"你的身份：{speaker_name}（人设：{speaker_persona or '（未提供）'}）\n"
        f"话题提示：{topic_hint or '（无）'}\n\n"
        f"== 你参与的最近对话片段（含所有人台词） ==\n{ctx}\n"
        f"最近一条：{last_role or '（无）'}：{last_text or '（无）'}\n\n"
        f"{STORY_DRIVE_RULES}\n"
        f"{action_bias_text()}\n"
        f"只输出你的台词（1-2句），不要名字前缀，不要加引号；保持自然、自洽。\n"
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
    base = (
        f"世界观：{worldview or '（未提供）'}\n"
        f"玩家角色：{PROTAGONIST_NAME}（人设：{protagonist_persona or '（未提供）'}）\n"
        f"话题提示：{topic_hint or '（无）'}\n\n"
        f"== 你参与的最近片段 ==\n{ctx}\n"
        f"最近一条：{last_role or '（无）'}：{last_text or '（无）'}\n\n"
        f"{STORY_DRIVE_RULES}\n"
        f"{action_bias_text()}\n"
        f"给出一条建议台词（第一人称“我”，1-2句；不要加引号/名字前缀）。"
    )
    prompt = integrate_memory_into_prompt(base, mem_lines or [])
    try:
        resp = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        return (getattr(resp, "text", "") or "").strip()
    except Exception as e:
        return f"(建议生成出错：{e})"

# —— 主角动作灵感（UI按钮使用） ——
def gen_player_action_ideas(worldview: str, rb_history: List[Dict], topic_hint: str = "") -> list:
    ctx = _ctx_block_from_rb(rb_history)
    prompt = (
        f"世界观：{worldview or '（未提供）'}\n"
        f"最近片段：\n{ctx}\n\n"
        f"话题提示：{topic_hint or '（无）'}\n"
        "给出 3 条可推动剧情的**具体动作灵感**（每条不超过15字，勿加句号，勿含台词），只用列表：\n"
        "- 示例动作A\n- 示例动作B\n- 示例动作C"
    )
    try:
        resp = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        txt = (getattr(resp, "text", "") or "").strip()
    except Exception:
        txt = ""
    ideas = [re.sub(r"^[-•]\s*", "", line).strip() for line in txt.splitlines() if line.strip()]
    return [i for i in ideas if i][:3]

# ===================== Streamlit UI =====================
st.set_page_config(page_title="多角色聊天 · 推进+动作+旁白", page_icon="🤖", layout="centered")
st.title("🤖 多角色多人聊天（Memory 检索/触发 + 旁白/动作旁白 + 剧情推进偏置）")

store = Neo4jStore()

# 计数：距离上次动作的轮数
if "turns_since_last_act" not in st.session_state:
    st.session_state.turns_since_last_act = 999

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

# ---- 会话首次创建 ----
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

    # 开场旁白
    opening = gen_opening_narration(
        worldview=protagonist_worldview or "",
        topic_hint=topic_hint,
        participants=[PROTAGONIST_NAME] + st.session_state.locked_participants
    )
    opening_clean = strip_action_blocks(strip_memory_blocks(opening))
    if opening_clean:
        nar_turn = {"role": NARRATOR_NAME, "content": opening_clean}
        st.session_state.chat_history.append(nar_turn)
        store.set_conversation_history(st.session_state.conv_id, st.session_state.chat_history)
        roles_to_update = list(dict.fromkeys(st.session_state.locked_participants + [PROTAGONIST_NAME]))
        store.append_turn_to_roles_recent(roles_to_update, nar_turn, keep=RB_MAX_TURNS_PER_ROLE)
        audience = list(dict.fromkeys(st.session_state.locked_participants + [PROTAGONIST_NAME]))
        handle_memory_triggers_from_reply(opening, audience, NARRATOR_NAME, st.session_state.conv_id)
    # 开始计数（无动作）
    st.session_state.turns_since_last_act = st.session_state.turns_since_last_act + 1

def reset_conversation():
    for k in ["conv_locked", "conv_id", "history_loaded_for", "chat_history",
              "awaiting_player_input", "player_suggested_line", "player_line_input",
              "player_memory_trigger_raw", "locked_participants", "player_action_ideas"]:
        st.session_state.pop(k, None)
    st.session_state.conv_locked = False
    st.session_state.locked_participants = []
    st.session_state.turns_since_last_act = 999

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
    st.info(
        f"已锁定会话：{st.session_state.conv_id}\n"
        f"参与者：{', '.join(st.session_state.locked_participants)}\n"
        f"自上次动作起已过：{st.session_state.turns_since_last_act} 轮"
    )

# ---- 读取会话历史 ----
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
    return [p["name"] for p in active_participants]

def persist_everything(speaker: str, text: str, raw_reply_for_trigger: Optional[str] = None):
    """
    写入历史 & RB；同时处理记忆/动作；更新 turns_since_last_act。
    传入 text 必须是剔除 MEMORY/ACT 后的台词。
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

    did_act = False
    if raw_reply_for_trigger:
        audience = list(dict.fromkeys(st.session_state.locked_participants + [PROTAGONIST_NAME]))
        handle_memory_triggers_from_reply(raw_reply_for_trigger, audience, speaker, st.session_state.conv_id)
        did_act = handle_action_triggers_from_reply(raw_reply_for_trigger, speaker, current_worldview(), st.session_state.conv_id)

    # 更新计数
    st.session_state.turns_since_last_act = 0 if did_act else st.session_state.turns_since_last_act + 1

# ===================== 生成一步（含 Memory/Action 注入） =====================
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
    mem_lines = build_memory_blocks_for_speaker(speaker_name, st.session_state.chat_history, all_names_in_scene())

    if sp["name"] == PROTAGONIST_NAME:
        sugg_raw = suggest_for_protagonist(
            protagonist_persona=protagonist_persona,
            worldview=current_worldview(),
            rb_history=rb_history,
            topic_hint=topic_hint,
            mem_lines=mem_lines
        )
        clean_sugg = strip_action_blocks(strip_memory_blocks(sugg_raw))
        st.session_state.awaiting_player_input = True
        st.session_state.player_suggested_line = sanitize_line(clean_sugg, PROTAGONIST_NAME, all_names_in_scene())
        st.session_state.player_line_input = st.session_state.player_suggested_line
        st.session_state.player_memory_trigger_raw = sugg_raw  # 触发原文
        return "player"

    raw = gen_with_role_recentbuffer(
        speaker_name=sp["name"],
        speaker_persona=sp.get("persona", ""),
        worldview=current_worldview(),
        rb_history=rb_history,
        topic_hint=topic_hint,
        mem_lines=mem_lines
    )
    cleaned = strip_action_blocks(strip_memory_blocks(raw))
    line = sanitize_line(cleaned, sp["name"], all_names_in_scene()).strip()

    if not line:
        audience = list(dict.fromkeys(st.session_state.locked_participants + [PROTAGONIST_NAME]))
        handle_memory_triggers_from_reply(raw, audience, sp["name"], st.session_state.conv_id)
        did_act = handle_action_triggers_from_reply(raw, sp["name"], current_worldview(), st.session_state.conv_id)
        st.session_state.turns_since_last_act = 0 if did_act else st.session_state.turns_since_last_act + 1
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
    for k in ["awaiting_player_input", "player_suggested_line", "player_line_input",
              "player_memory_trigger_raw", "player_action_ideas"]:
        st.session_state.pop(k, None)
    st.session_state.turns_since_last_act = 999
    st.rerun()

# ---- 玩家输入区（主角）----
if st.session_state.get("awaiting_player_input"):
    st.markdown("### 轮到你发言（作为：**主角（你）**）")
    st.caption(f"提示：{ACT_USAGE_TIP}（你的动作会触发旁白与记忆）")
    st.session_state.player_line_input = st.text_area(
        "你的台词（可编辑）",
        value=st.session_state.get("player_line_input", st.session_state.get("player_suggested_line", "")),
        height=120,
        key="ta_player_line"
    )

    # —— 主角动作：模板 & 灵感 —— #
    cA, cB, cC, cD = st.columns(4)
    with cA:
        send_player = st.button("发送", key="btn_player_send")
    with cB:
        regen_sugg = st.button("换一个建议", key="btn_player_regen")
    with cC:
        insert_act_tpl = st.button("插入动作模板", key="btn_insert_act_tpl")
    with cD:
        get_act_ideas = st.button("给我动作灵感", key="btn_get_act_ideas")

    if insert_act_tpl:
        tpl = "\n[[ACT]] 我做了什么（例如：点燃火把，检查墙上刻痕） [[/ACT]]"
        st.session_state.player_line_input = (st.session_state.get("player_line_input","") or "") + tpl
        st.rerun()

    if get_act_ideas:
        st.session_state.player_action_ideas = gen_player_action_ideas(
            current_worldview(), store.get_role_recentbuffer(PROTAGONIST_NAME), topic_hint
        )
        st.rerun()

    ideas = st.session_state.get("player_action_ideas", [])
    if ideas:
        st.markdown("**动作灵感：**")
        for idx, idea in enumerate(ideas, 1):
            col1, col2 = st.columns([0.75, 0.25])
            with col1:
                st.write(f"- {idea}")
            with col2:
                if st.button(f"用这个 #{idx}", key=f"use_idea_{idx}"):
                    to_add = f"\n[[ACT]] {idea} [[/ACT]]"
                    st.session_state.player_line_input = (st.session_state.get("player_line_input","") or "") + to_add
                    st.rerun()

    cE, cF = st.columns(2)
    with cE:
        skip_player = st.button("换个说话者", key="btn_player_skip")
    with cF:
        cancel_ideas = st.button("清空动作灵感", key="btn_clear_ideas")

    if cancel_ideas:
        st.session_state.pop("player_action_ideas", None)
        st.rerun()

    if regen_sugg:
        mem_lines_me = build_memory_blocks_for_speaker(
            PROTAGONIST_NAME, st.session_state.chat_history, all_names_in_scene()
        )
        new_sugg_raw = suggest_for_protagonist(
            protagonist_persona=protagonist_persona,
            worldview=current_worldview(),
            rb_history=store.get_role_recentbuffer(PROTAGONIST_NAME),
            topic_hint=topic_hint,
            mem_lines=mem_lines_me
        )
        clean_sugg = strip_action_blocks(strip_memory_blocks(new_sugg_raw))
        st.session_state.player_suggested_line = sanitize_line(clean_sugg, PROTAGONIST_NAME, all_names_in_scene())
        st.session_state.player_line_input = st.session_state.player_suggested_line
        st.session_state.player_memory_trigger_raw = new_sugg_raw
        st.rerun()

    if skip_player:
        st.session_state.awaiting_player_input = False
        for k in ["player_suggested_line", "player_line_input", "player_memory_trigger_raw", "player_action_ideas"]:
            st.session_state.pop(k, None)
        one_step_generate(skip_protagonist=True)
        st.rerun()

    if send_player:
        typed_raw = st.session_state.get("player_line_input", "")
        cleaned = strip_action_blocks(strip_memory_blocks(typed_raw))
        line = sanitize_line(cleaned, PROTAGONIST_NAME, all_names_in_scene()).strip()
        trigger_blob = (st.session_state.get("player_memory_trigger_raw", "") or "") + "\n" + (typed_raw or "")
        if not line:
            audience = list(dict.fromkeys(st.session_state.locked_participants + [PROTAGONIST_NAME]))
            handle_memory_triggers_from_reply(trigger_blob, audience, PROTAGONIST_NAME, st.session_state.conv_id)
            did_act = handle_action_triggers_from_reply(trigger_blob, PROTAGONIST_NAME, current_worldview(), st.session_state.conv_id)
            st.session_state.turns_since_last_act = 0 if did_act else st.session_state.turns_since_last_act + 1
        else:
            persist_everything(PROTAGONIST_NAME, line, raw_reply_for_trigger=trigger_blob)
        st.session_state.awaiting_player_input = False
        for k in ["player_suggested_line", "player_line_input", "player_memory_trigger_raw", "player_action_ideas"]:
            st.session_state.pop(k, None)
        st.rerun()

# ---- 维护工具 ----
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
# streamlit run d:/NewProjects/LowCostChattingBot/npc-graph-chat/4.0_multi_character_chatbot_ChatTogether_WithRag_WithAction.py
