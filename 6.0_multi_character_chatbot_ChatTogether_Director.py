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
DEFAULT_PROTAGONIST_NAME = os.getenv("DEFAULT_PROTAGONIST_NAME", "健身教练")
PROTAGONIST_DEFAULT_DESIRE = 1
NARRATOR_NAME = "旁白"
DIRECTOR_NAME = "导演"

# 检索参数
RETR_TOPK = 5
RETR_WEIGHTS = [0.5, 0.3, 0.2]  # 最近 -> 更早

# 剧情推进/动作偏置参数
ACTION_BIAS_TURNS = 10          # 最近连续多少轮没有动作后，强烈要求本轮给动作
ACTION_BIAS_FORCE_TEXT = "【剧情推进】最近几轮缺少真实动作，请给出明确行动并务必追加 [[ACT]] 动作简述 [[/ACT]]。"
ACTION_BIAS_SOFT_TEXT  = "若你亲自执行动作，请在末行追加 [[ACT]] 动作简述 [[/ACT]]；若只是提问、和别人聊天，或等别人行动，则不要追加。"

# —— 导演干预节奏（UI 可调）默认范围 ——
DEFAULT_DIR_CADENCE_MIN = 5
DEFAULT_DIR_CADENCE_MAX = 15

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

# >>> NEW: RAG debug helpers
def _rag_enabled() -> bool:
    try:
        return bool(st.session_state.get("debug_rag", True))
    except Exception:
        return True

def rag_dbg(msg: str):
    if _rag_enabled():
        try:
            print(msg)
        except Exception:
            pass

def protagonist_name() -> str:
    try:
        pn = st.session_state.get("protagonist_name")
        return pn or DEFAULT_PROTAGONIST_NAME
    except Exception:
        return DEFAULT_PROTAGONIST_NAME

def display_name_ui(name: str) -> str:
    try:
        return f"{name}（你）" if name == protagonist_name() else name
    except Exception:
        return name

def make_conv_id(participants: List[str], title: str) -> str:
    names = sorted(set(participants + [protagonist_name()]))
    title_norm = re.sub(r"\s+", " ", (title or "").strip()) or "untitled"
    return f"conv::{title_norm}::{'|'.join(names)}"

# >>> NEW: 旁白“你”→明确化，防止 NPC 误当自己
def resolve_narrator_you(text: str) -> str:
    if not text:
        return ""
    t = text
    # 先长词，后短词，避免覆盖
    t = t.replace("你们的", "众人的")
    t = t.replace("你们", "众人")
    pn = protagonist_name()
    t = t.replace("你的", f"{pn}的")
    t = t.replace("你", pn)
    return t

def format_query_text(turn: Dict, speaker_pool: List[str]) -> str:
    role = turn.get("role", "")
    content = turn.get("content", "")
    # >>> NEW: 构造检索文本时对旁白做“你”解析
    if role == NARRATOR_NAME:
        content = resolve_narrator_you(content)
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
# ==== CHANGED: 增加规范化与宽松匹配，覆盖【】/大小写/空格/[[ACT:]] 等变体 ====

# 兼容开关标签的多种写法
_ACT_OPEN_VARIANTS = r'(?:\[\[|【)\s*(?:ACT|ACTION)\s*(?:\]\]|】)'
_ACT_CLOSE_VARIANTS = r'(?:\[\[\/\s*(?:ACT|ACTION)\s*\]\]|【\/\s*(?:ACT|ACTION)\s*】)'

# 大小写不敏感、跨行
_ACT_BLOCK_CI = re.compile(
    rf'{_ACT_OPEN_VARIANTS}\s*(.+?)\s*{_ACT_CLOSE_VARIANTS}',
    flags=re.IGNORECASE | re.DOTALL | re.UNICODE
)

def _normalize_action_markup(text: str) -> str:
    """把全角【】替换为 [[ ]]，并把标签统一成 [[ACT]] / [[/ACT]]；兼容 [[ACT:]] 写法"""
    if not text:
        return ""
    t = text.replace("【", "[[").replace("】", "]]")
    # 修复误写：[[ACT 内容]] / [[ACTION 内容]] → [[ACT]] 内容
    t = re.sub(r"\[\[\s*(act|action)\s+([^\]]+?)\s*\]\]", r"[[ACT]] \2", t, flags=re.IGNORECASE)
    # 开标签 -> [[ACT]]
    t = re.sub(r'\[\[\s*(act|action)\s*\]\]', '[[ACT]]', t, flags=re.IGNORECASE)
    # 关标签 -> [[/ACT]]
    t = re.sub(r'\[\[\s*/\s*(act|action)\s*\]\]', '[[/ACT]]', t, flags=re.IGNORECASE)
    # 兼容 [[ACT:]] 误写
    t = re.sub(r'\[\[\s*ACT\s*:\s*\]\]', '[[ACT]]', t, flags=re.IGNORECASE)
    return t

def parse_action_triggers(text: str) -> List[str]:
    if not text:
        return []
    try:
        t = _normalize_action_markup(text)
        acts = [m.group(1).strip() for m in _ACT_BLOCK_CI.finditer(t)]
        return acts
    except Exception:
        return []

def strip_action_blocks(text: str) -> str:
    if not text:
        return ""
    try:
        t = _normalize_action_markup(text)
        t = _ACT_BLOCK_CI.sub(" ", t)  # 清掉整段 [[ACT]]...[[/ACT]]
        return re.sub(r"\s+\n|\n\s+", "\n", t).strip()
    except Exception:
        return text

# ====== 关系触发块（仅在说话者发言时可用）======
# 兼容多种标签：[[REL]] / [[REL-UPDATE]] / [[RELATIONSHIP]] 以及全角【】
_REL_OPEN_VARIANTS = r'(?:\[\[|【)\s*(?:REL|REL-UPDATE|RELATIONSHIP)\s*(?:\]\]|】)'
_REL_CLOSE_VARIANTS = r'(?:\[\[\/\s*(?:REL|REL-UPDATE|RELATIONSHIP)\s*\]\]|【\/\s*(?:REL|REL-UPDATE|RELATIONSHIP)\s*】)'
_REL_BLOCK_CI = re.compile(
    rf'{_REL_OPEN_VARIANTS}\s*(.+?)\s*{_REL_CLOSE_VARIANTS}',
    flags=re.IGNORECASE | re.DOTALL | re.UNICODE
)

def _normalize_rel_markup(text: str) -> str:
    if not text:
        return ""
    t = text.replace("【", "[[").replace("】", "]]")
    # 统一开闭标签到 [[REL]] / [[/REL]]
    t = re.sub(r"\[\[\s*(rel-update|relationship)\s*\]\]", "[[REL]]", t, flags=re.IGNORECASE)
    t = re.sub(r"\[\[\s*rel\s*\]\]", "[[REL]]", t, flags=re.IGNORECASE)
    # 兼容误写：[[REL: ...]] → [[REL]]
    t = re.sub(r"\[\[\s*rel\s*:\s*[^\]]*\]\]", "[[REL]]", t, flags=re.IGNORECASE)
    t = re.sub(r"\[\[\s*/\s*(rel|rel-update|relationship)\s*\]\]", "[[/REL]]", t, flags=re.IGNORECASE)
    return t

def strip_relationship_blocks(text: str) -> str:
    if not text:
        return ""
    try:
        t = _normalize_rel_markup(text)
        # 先清理规范的 [[REL]]...[[/REL]] 区块
        t = _REL_BLOCK_CI.sub(" ", t)
        # 再清理可能残留的裸标签或误写（如只有开/闭标签，或 [[REL:xxx]] 已在规范化变为 [[REL]]）
        t = re.sub(r"\[\[\s*/?\s*REL\s*\]\]", " ", t, flags=re.IGNORECASE)
        return re.sub(r"\s+\n|\n\s+", "\n", t).strip()
    except Exception:
        return text

def _safe_json_array(text: str):
    if not text:
        return []
    t = text.strip()
    try:
        data = json.loads(t)
        return data if isinstance(data, list) else []
    except Exception:
        pass
    try:
        m = re.search(r"\[.*\]", t, flags=re.S)
        if m:
            data = json.loads(m.group(0))
            return data if isinstance(data, list) else []
    except Exception:
        return []
    return []

def parse_relationship_updates(text: str) -> List[Dict]:
    """从 [[REL]]...[[/REL]] 中解析 JSON 数组，元素形如：
    {src, dst, add:{rel[],att[],notes[]}, remove:{rel[],att[],notes[]}}
    返回规范化后的列表（字段均为去重后的列表）。
    """
    if not text:
        return []
    try:
        t = _normalize_rel_markup(text)
        payloads = [m.group(1).strip() for m in _REL_BLOCK_CI.finditer(t)]
        updates = []
        for blob in payloads:
            arr = _safe_json_array(blob)
            for it in (arr or []):
                try:
                    src = str(it.get("src", "")).strip()
                    dst = str(it.get("dst", "")).strip()
                    if not src or not dst or src == dst:
                        continue
                    add = it.get("add", {}) or {}
                    rm = it.get("remove", {}) or {}
                    def _norm_list(x):
                        if not x:
                            return []
                        if not isinstance(x, list):
                            x = [x]
                        vals = [str(v).strip() for v in x if str(v).strip()]
                        out = []
                        seen = set()
                        for v in vals:
                            if v not in seen:
                                seen.add(v)
                                out.append(v)
                        return out
                    upd = {
                        "src": src,
                        "dst": dst,
                        "add": {
                            "rel": _norm_list(add.get("rel")),
                            "att": _norm_list(add.get("att")),
                            "notes": _norm_list(add.get("notes")),
                        },
                        "remove": {
                            "rel": _norm_list(rm.get("rel")),
                            "att": _norm_list(rm.get("att")),
                            "notes": _norm_list(rm.get("notes")),
                        }
                    }
                    updates.append(upd)
                except Exception:
                    continue
        return updates
    except Exception:
        return []

# >>> NEW: 问句/点名/沉默时长
_Q_MARKS = ("?", "？")

def is_question(text: str) -> bool:
    if not text:
        return False
    s = text.strip()
    return any(s.endswith(m) for m in _Q_MARKS)

def is_addressed_to_someone(text: str, names: List[str]) -> bool:
    if not text:
        return False
    return any(n and (n in text) for n in names)

def silence_ages(chat_history: List[Dict], names: List[str]) -> Dict[str, int]:
    last_pos = {n: None for n in names}
    for idx in range(len(chat_history) - 1, -1, -1):
        r = chat_history[idx].get("role")
        if r in last_pos and last_pos[r] is None:
            last_pos[r] = idx
        if all(v is not None for v in last_pos.values()):
            break
    ages = {}
    L = len(chat_history)
    for n in names:
        ages[n] = (L - 1 - last_pos[n]) if last_pos[n] is not None else (L + 1)
    return ages

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
        # 确保功能性角色（旁白/导演）打上 Functional 标签与标记
        try:
            self.ensure_functional_labels()
        except Exception:
            pass

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
            # Verify existence and ONLINE state using SHOW INDEXES
            try:
                with self.driver.session(database=NEO4J_DATABASE) as s2:
                    rec = s2.run(
                        "SHOW INDEXES YIELD name, type, state WHERE name = $n RETURN name, type, state",
                        n=self.MEM_INDEX
                    ).single()
                    if not rec:
                        # Fallback: try the procedure API available on some Neo4j versions
                        with self.driver.session(database=NEO4J_DATABASE) as s3:
                            s3.run("CALL db.index.vector.createNodeIndex($name, 'Memory', 'embedding', $dims, 'cosine')",
                                   name=self.MEM_INDEX, dims=int(dims))
                        rec = s2.run(
                            "SHOW INDEXES YIELD name, type, state WHERE name = $n RETURN name, type, state",
                            n=self.MEM_INDEX
                        ).single()
                    # If exists but wrong type, drop and recreate
                    if rec and str(rec["type"]).upper() != "VECTOR":
                        try:
                            s2.run(f"DROP INDEX {self.MEM_INDEX} IF EXISTS")
                        except Exception:
                            pass
                        with self.driver.session(database=NEO4J_DATABASE) as s3:
                            s3.run(f"""
                            CREATE VECTOR INDEX {self.MEM_INDEX}
                            FOR (m:Memory) ON (m.embedding)
                            OPTIONS {{
                              indexConfig: {{
                                `vector.dimensions`: {int(dims)},
                                `vector.similarity_function`: 'cosine'
                              }}
                            }}
                            """)
                        rec = s2.run(
                            "SHOW INDEXES YIELD name, type, state WHERE name = $n RETURN name, type, state",
                            n=self.MEM_INDEX
                        ).single()
                    self.mem_index_ready = bool(rec) and str(rec["type"]).upper() == "VECTOR" and str(rec["state"]).upper() == "ONLINE"
            except Exception:
                # If SHOW INDEXES unsupported in env, assume created; query path will re-check.
                self.mem_index_ready = True
            self.mem_index_dims = int(dims) if self.mem_index_ready else None
        except Exception:
            self.mem_index_ready = False
            self.mem_index_dims = None

    def _get_index_info(self) -> Dict:
        try:
            with self.driver.session(database=NEO4J_DATABASE) as s:
                rec = s.run(
                    "SHOW INDEXES YIELD name, type, state, indexProvider, labelsOrTypes, properties WHERE name = $n RETURN name, type, state, indexProvider, labelsOrTypes, properties",
                    n=self.MEM_INDEX
                ).single()
                return rec.data() if rec else {}
        except Exception:
            return {}

    # 角色 & RB
    def ensure_character(self, name: str, persona: str = "", worldview: str = "", desire: float = 0.3):
        with self.driver.session(database=NEO4J_DATABASE) as s:
            s.run("""
                MERGE (c:Character {name:$name})
                  ON CREATE SET c.created_at = datetime()
                SET c.persona=$persona, c.worldview=$worldview, c.desire=$desire, c.updated_at=datetime()
            """, name=name, persona=persona, worldview=worldview, desire=float(desire))
            # 若为功能性角色，则打上标签与标记
            if name in (NARRATOR_NAME, DIRECTOR_NAME):
                s.run(
                    """
                    MATCH (c:Character {name:$name})
                    SET c:Functional, c.functional = true, c.updated_at = datetime()
                    """,
                    name=name
                )
            self.ensure_role_recentbuffer(name)
            # 确保导演对新角色有 GUIDES 关系
            try:
                s.run(
                    """
                    MERGE (d:Character {name:$director})
                    MERGE (c:Character {name:$name})
                    MERGE (d)-[g:GUIDES]->(c)
                    ON CREATE SET g.scene_focus = coalesce($scene_focus, ""),
                                  g.state_intent = coalesce($state_intent, ""),
                                  g.created_at = datetime()
                    SET g.updated_at = datetime()
                    """,
                    director=DIRECTOR_NAME, name=name, scene_focus="", state_intent=""
                )
            except Exception:
                pass

    def get_characters(self):
        with self.driver.session(database=NEO4J_DATABASE) as s:
            res = s.run("""
                MATCH (c:Character)
                RETURN c.name AS name, c.persona AS persona, c.worldview AS worldview, coalesce(c.desire,0.3) AS desire, coalesce(c.functional,false) AS functional
                ORDER BY c.name
            """)
            return res.data()

    def ensure_functional_labels(self):
        """为旁白/导演节点设置 :Functional 标签与 functional=true 属性（若存在则更新）。"""
        with self.driver.session(database=NEO4J_DATABASE) as s:
            s.run(
                """
                MERGE (n:Character {name:$nar})
                SET n:Functional, n.functional = true, n.updated_at = datetime()
                """,
                nar=NARRATOR_NAME
            )
            s.run(
                """
                MERGE (d:Character {name:$dir})
                SET d:Functional, d.functional = true, d.updated_at = datetime()
                """,
                dir=DIRECTOR_NAME
            )

    def update_character_desire(self, name: str, desire: float):
        with self.driver.session(database=NEO4J_DATABASE) as s:
            s.run("MATCH (c:Character {name:$n}) SET c.desire=$d, c.updated_at=datetime()", n=name, d=float(desire))

    def _rb_id(self, name: str) -> str:
        # 让导演与旁白共享同一个 RecentBuffer（便于统一上下文/RAG）
        if name == DIRECTOR_NAME:
            return f"rb::{NARRATOR_NAME}"
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

    def get_director_profile(self) -> Dict:
        with self.driver.session(database=NEO4J_DATABASE) as s:
            try:
                rec = s.run(
                    "MATCH (c:Character {name:$n}) RETURN c.style AS style, c.method AS method",
                    n=DIRECTOR_NAME
                ).single()
                return rec.data() if rec else {"style": None, "method": None}
            except Exception:
                return {"style": None, "method": None}

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
        # 尝试懒加载：若还未标记索引可用，但库里已有 embedding，则推断维度并创建/标记索引
        if not self.mem_index_ready:
            try:
                with self.driver.session(database=NEO4J_DATABASE) as s:
                    rec = s.run(
                        """
                        MATCH (m:Memory)
                        WHERE m.embedding IS NOT NULL AND size(m.embedding) > 0
                        RETURN size(m.embedding) AS d
                        LIMIT 1
                        """
                    ).single()
                    if rec and rec["d"]:
                        self.ensure_mem_index(int(rec["d"]))
                        if self.mem_index_ready:
                            try:
                                st.info(f"[mem-index] 已自动启用向量索引 {self.MEM_INDEX} (dims={self.mem_index_dims})")
                            except Exception:
                                pass
                            try:
                                if _rag_enabled():
                                    print(f"[RAG] mem-index ready dims={self.mem_index_dims}")
                            except Exception:
                                pass
            except Exception:
                pass
        # If still not ready, attempt a final existence/type check and bail gracefully
        if not self.mem_index_ready:
            try:
                info = self._get_index_info()
                is_vector_online = bool(info) and str(info.get("type","" )).upper()=="VECTOR" and str(info.get("state","" )).upper()=="ONLINE"
                if not is_vector_online:
                    if _rag_enabled():
                        print(f"[RAG] vector index '{self.MEM_INDEX}' missing/wrong-type/offline: {info}; skip retrieval or use fallback")
                    return []
                else:
                    self.mem_index_ready = True
            except Exception:
                return []
        try:
            if _rag_enabled():
                print(f"[RAG] query speaker={speaker} mem_index_ready={self.mem_index_ready} dims={self.mem_index_dims} k={k} q_lens={[len(e) for e in (q_embs or [])]} weights={weights}")
        except Exception:
            pass
        if not q_embs:
            return []
        # Python fallback when vector index is not ready but we still want retrieval
        if not self.mem_index_ready:
            try:
                if _rag_enabled():
                    print("[RAG] fallback to Python cosine retrieval (no vector index)")
            except Exception:
                pass
            # Pull AWARE_OF memories embeddings for the speaker (cap to avoid huge transfers)
            PY_FALLBACK_MAX_MEMS = 1000
            with self.driver.session(database=NEO4J_DATABASE) as s:
                try:
                    rows = s.run(
                        """
                        MATCH (:Character {name:$speaker})-[:AWARE_OF]->(m:Memory)
                        WHERE m.embedding IS NOT NULL AND size(m.embedding) > 0
                        RETURN m.content AS content, m.embedding AS embedding
                        LIMIT $lim
                        """,
                        speaker=speaker, lim=int(PY_FALLBACK_MAX_MEMS)
                    ).data()
                except Exception:
                    rows = []
            if not rows:
                return []
            # Cosine helper
            def _cos(a: List[float], b: List[float]) -> float:
                try:
                    import math
                    if not a or not b:
                        return 0.0
                    n = min(len(a), len(b))
                    dot = 0.0; na = 0.0; nb = 0.0
                    for i in range(n):
                        va = float(a[i]); vb = float(b[i])
                        dot += va * vb; na += va * va; nb += vb * vb
                    if na <= 0 or nb <= 0:
                        return 0.0
                    return dot / (math.sqrt(na) * math.sqrt(nb))
                except Exception:
                    return 0.0
            # Weighted sum of cosine(q, mem)
            scored = []
            for r in rows:
                emb = r.get("embedding") or []
                total = 0.0
                for q, w in zip(q_embs, weights):
                    total += float(w) * _cos(q, emb)
                scored.append({"content": r.get("content",""), "total_score": total})
            scored.sort(key=lambda x: x["total_score"], reverse=True)
            topk = scored[: int(k)]
            try:
                if _rag_enabled():
                    print(f"[RAG] py-fallback rows={[(round(x.get('total_score',0.0),3), (x.get('content','')[:40])) for x in topk]}")
            except Exception:
                pass
            return topk
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
                rows = res.data()
                try:
                    if _rag_enabled():
                        print(f"[RAG] cypher_parts={len(parts)} rows={[(round(r.get('total_score',0.0),3), (r.get('content','')[:40])) for r in (rows or [])]}")
                except Exception:
                    pass
                return rows
            except Exception as e:
                try:
                    if _rag_enabled():
                        print(f"[RAG] query failed: {e}")
                except Exception:
                    pass
                return []

    # —— 关系/态度参考（使用 RELATES_TO，属性 att/rel/notes） ——
    def get_outgoing_relationships_for(self, speaker: str, others: List[str]):
        """返回列表 [{other, rel, att, notes}, ...]，仅 speaker→other。"""
        if not others:
            return []
        with self.driver.session(database=NEO4J_DATABASE) as s:
            try:
                res = s.run(
                    """
                    MATCH (a:Character {name:$a})
                    MATCH (b:Character)
                    WHERE b.name IN $others
                    MATCH (a)-[r:RELATES_TO]->(b)
                    RETURN b.name AS other,
                           r.rel   AS rel,
                           r.att   AS att,
                           r.notes AS notes
                    ORDER BY b.name
                    """,
                    a=speaker, others=list(others)
                )
                return res.data()
            except Exception:
                return []

    def get_relationship_matrix(self, names: List[str]):
        """返回列表 [{src, dst, rel, att, notes}]，仅返回存在的有向关系（a)-[RELATES_TO]->(b)。"""
        if not names:
            return []
        with self.driver.session(database=NEO4J_DATABASE) as s:
            try:
                res = s.run(
                    """
                    MATCH (a:Character) WHERE a.name IN $names
                    MATCH (b:Character) WHERE b.name IN $names AND a <> b
                    OPTIONAL MATCH (a)-[r:RELATES_TO]->(b)
                    WITH a,b,r
                    WHERE r IS NOT NULL
                    RETURN a.name AS src,
                           b.name AS dst,
                           r.rel   AS rel,
                           r.att   AS att,
                           r.notes AS notes
                    ORDER BY src, dst
                    """,
                    names=list(names)
                )
                return res.data()
            except Exception:
                return []

    # —— 导演 GUIDES 关系：读/写 ——
    def upsert_guides(self, target: str, scene_focus: Optional[str], state_intent: Optional[str]):
        with self.driver.session(database=NEO4J_DATABASE) as s:
            try:
                s.run(
                    """
                    MERGE (d:Character {name:$director})
                    MERGE (c:Character {name:$target})
                    MERGE (d)-[g:GUIDES]->(c)
                    ON CREATE SET g.scene_focus = coalesce($scene_focus, ""),
                                  g.state_intent = coalesce($state_intent, ""),
                                  g.created_at = datetime()
                    SET g.scene_focus = coalesce($scene_focus, g.scene_focus),
                        g.state_intent = coalesce($state_intent, g.state_intent),
                        g.updated_at = datetime()
                    """,
                    director=DIRECTOR_NAME, target=target, scene_focus=scene_focus, state_intent=state_intent
                )
            except Exception:
                pass

    def get_guides_for(self, targets: List[str]) -> List[Dict]:
        if not targets:
            return []
        with self.driver.session(database=NEO4J_DATABASE) as s:
            try:
                res = s.run(
                    """
                    MATCH (d:Character {name:$director})-[g:GUIDES]->(c:Character)
                    WHERE c.name IN $targets
                    RETURN c.name AS target, g.scene_focus AS scene_focus, g.state_intent AS state_intent
                    ORDER BY c.name
                    """,
                    director=DIRECTOR_NAME, targets=targets
                )
                return res.data()
            except Exception:
                return []

    # —— 关系写入/读取 ——
    def get_relationship_props(self, src: str, dst: str) -> Dict:
        with self.driver.session(database=NEO4J_DATABASE) as s:
            try:
                rec = s.run(
                    """
                    MATCH (:Character {name:$src})-[r:RELATES_TO]->(:Character {name:$dst})
                    RETURN r.rel AS rel, r.att AS att, r.notes AS notes
                    """,
                    src=src, dst=dst
                ).single()
                return rec.data() if rec else {"rel": None, "att": None, "notes": None}
            except Exception:
                return {"rel": None, "att": None, "notes": None}

    def upsert_relationship_lists(self, src: str, dst: str, rel_list: List[str], att_list: List[str], notes_list: List[str]):
        def _norm_list(x):
            if not x:
                return []
            if not isinstance(x, list):
                x = [x]
            vals = []
            for v in x:
                s = str(v).strip()
                if s:
                    vals.append(s)
            out = []
            seen = set()
            for v in vals:
                if v not in seen:
                    seen.add(v)
                    out.append(v)
            return out[:10]
        rel_list = _norm_list(rel_list)
        att_list = _norm_list(att_list)
        notes_list = _norm_list(notes_list)
        with self.driver.session(database=NEO4J_DATABASE) as s:
            s.run(
                """
                MERGE (a:Character {name:$src})
                MERGE (b:Character {name:$dst})
                MERGE (a)-[r:RELATES_TO]->(b)
                SET r.rel=$rel, r.att=$att, r.notes=$notes, r.updated_at=datetime()
                """,
                src=src, dst=dst, rel=rel_list, att=att_list, notes=notes_list
            )

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
        out = list(vec) if vec else []
        try:
            rag_dbg(f"[EMB] len={len(out)} text='{(text.strip()[:60] + ('...' if len(text.strip())>60 else ''))}'")
        except Exception:
            pass
        return out
    except Exception:
        try:
            rag_dbg(f"[EMB] failed for text='{(text or '')[:60]}'")
        except Exception:
            pass
        return []

# ===================== LLM 调试与封装 =====================
def _llm_dbg_enabled() -> bool:
    try:
        return bool(st.session_state.get("debug_llm", True))
    except Exception:
        return True

def llm_generate_text(contents: str, tag: str = "") -> str:
    try:
        if _llm_dbg_enabled():
            try:
                print(f"[LLM REQ]{(' ' + tag) if tag else ''}\n<<<\n{contents}\n>>>")
            except Exception:
                pass
        resp = client.models.generate_content(model=GEMINI_MODEL, contents=contents)
        txt = (getattr(resp, "text", "") or "").strip()
        if _llm_dbg_enabled():
            try:
                print(f"[LLM RESP]{(' ' + tag) if tag else ''}\n<<<\n{txt}\n>>>")
            except Exception:
                pass
        return txt
    except Exception as e:
        try:
            if _llm_dbg_enabled():
                print(f"[LLM ERR]{(' ' + tag) if tag else ''} {e}")
        except Exception:
            pass
        return ""

# ===================== 生成逻辑（加入 Memory 检索 + 触发） =====================
def build_memory_blocks_for_speaker(speaker_name: str, recent_turns: List[Dict], speaker_pool: List[str]) -> List[str]:
    """Use last 3 embeddings from RecentBuffer; fallback to compute if missing."""
    try:
        rb = store.get_role_recentbuffer(speaker_name) or []
        last3 = list(reversed(rb[-3:]))
        try:
            rag_dbg(f"[RAG] Q source speaker={speaker_name} last3_count={len(last3)}")
        except Exception:
            pass
        q_embs: List[List[float]] = []
        for t in last3:
            role = t.get("role") if isinstance(t, dict) else None
            content = (t.get("content") or "").strip() if isinstance(t, dict) else ""
            emb = t.get("embedding") if isinstance(t, dict) else None
            emb_from = "rb"
            if not emb:
                emb = embed_text(content) if isinstance(t, dict) else []
                emb_from = "calc"
            if isinstance(emb, list) and emb:
                q_embs.append(emb)
            try:
                rag_dbg(f"[RAG] q from={emb_from} role={role} emb_len={len(emb) if emb else 0} text='{content[:40]}'")
            except Exception:
                pass
        try:
            rag_dbg(f"[RAG] q_embs_lens={[len(e) for e in q_embs]}")
        except Exception:
            pass
        rows = store.query_memories_sum_scores(speaker_name, q_embs, RETR_WEIGHTS, RETR_TOPK)
        try:
            rag_dbg(f"[RAG] hits={[(round(r.get('total_score',0.0),3), (r.get('content','')[:40])) for r in (rows or [])]}")
        except Exception:
            pass
        return [r["content"] for r in rows]
    except Exception:
        return []

def build_memory_blocks_for_narrator(recent_turns: List[Dict]) -> List[str]:
    """Use last 3 embeddings for narrator queries and return matched memories aware to narrator."""
    try:
        rb = store.get_role_recentbuffer(NARRATOR_NAME) or []
        last3 = list(reversed(rb[-3:]))
        q_embs: List[List[float]] = []
        for t in last3:
            content = (t.get("content") or "").strip() if isinstance(t, dict) else ""
            emb = t.get("embedding") if isinstance(t, dict) else None
            if not emb and content:
                emb = embed_text(content)
            if isinstance(emb, list) and emb:
                q_embs.append(emb)
        if not q_embs and recent_turns:
            # fallback to last 3 turn contents
            for t in list(reversed(recent_turns[-3:])):
                txt = (t.get("content") or "").strip()
                if not txt:
                    continue
                emb = embed_text(txt)
                if emb:
                    q_embs.append(emb)
        rows = store.query_memories_sum_scores(NARRATOR_NAME, q_embs, RETR_WEIGHTS, RETR_TOPK)
        return [r.get("content","") for r in (rows or [])]
    except Exception:
        return []

def join_memory_lines(mem_lines: List[str], max_n: int = 3) -> str:
    if not mem_lines:
        return ""
    mem_lines = mem_lines[:max_n]
    return "\n".join(f"- {m}" for m in mem_lines)

def integrate_memory_into_prompt(base_prompt: str, mem_lines: List[str]) -> str:
    if not mem_lines:
        try:
            rag_dbg("[RAG] integrate: no mem_lines")
        except Exception:
            pass
        return base_prompt
    mem_block = join_memory_lines(mem_lines)
    prefix = "你已知的关键信息（可能有用）：\n" + f"{mem_block}\n\n"
    try:
        rag_dbg(f"[RAG] integrate mem_lines={mem_lines}")
    except Exception:
        pass
    return prefix + base_prompt

# —— 小工具：把关系属性（可能是列表）转成简洁文本 ——
def _rel_val_to_text(v) -> str:
    try:
        if v is None:
            return ""
        # 列表/元组：拼成“、”分隔
        if isinstance(v, (list, tuple)):
            parts = [str(x).strip() for x in v if x is not None and str(x).strip()]
            return "、".join(parts)
        # 其他类型：直接转字符串
        s = str(v).strip()
        return s
    except Exception:
        return ""

def apply_relationship_trigger_updates(updates: List[Dict], participants: List[str]):
    """将 [[REL]] 解析出的 updates 应用到 Neo4j；仅允许参与者之间的有向边。"""
    if not updates:
        return
    pset = set(participants or [])
    for up in updates:
        src = up.get("src"); dst = up.get("dst")
        if not src or not dst or src == dst:
            continue
        if src not in pset or dst not in pset:
            continue
        current = store.get_relationship_props(src, dst) or {}
        def _to_list(x):
            if not x:
                return []
            if not isinstance(x, list):
                x = [x]
            vals = []
            for v in x:
                s = str(v).strip()
                if s:
                    vals.append(s)
            # 去重保序
            out = []
            seen = set()
            for v in vals:
                if v not in seen:
                    seen.add(v)
                    out.append(v)
            return out
        rel_old = _to_list(current.get("rel"))
        att_old = _to_list(current.get("att"))
        notes_old = _to_list(current.get("notes"))

        add = up.get("add", {}) or {}
        rm  = up.get("remove", {}) or {}
        rel_add = _to_list(add.get("rel")); att_add = _to_list(add.get("att")); notes_add = _to_list(add.get("notes"))
        rel_rm  = set(_to_list(rm.get("rel"))); att_rm  = set(_to_list(rm.get("att"))); notes_rm  = set(_to_list(rm.get("notes")))

        def _merge_add(old, addv):
            seen = set(old)
            out = list(old)
            for v in addv:
                if v not in seen:
                    seen.add(v)
                    out.append(v)
            return out
        def _merge_rm(old, rset):
            return [v for v in old if v not in rset]

        rel_new = _merge_rm(_merge_add(rel_old, rel_add), rel_rm)[:10]
        att_new = _merge_rm(_merge_add(att_old, att_add), att_rm)[:10]
        notes_new = _merge_rm(_merge_add(notes_old, notes_add), notes_rm)[:10]

        store.upsert_relationship_lists(src, dst, rel_new, att_new, notes_new)

# —— 构造“关系/态度”提示块（发言者 → 他人，仅使用 RELATES_TO.att/rel/notes） ——
def build_relationship_block_for_prompt(speaker_name: str, speaker_pool: List[str]) -> str:
    try:
        others = [n for n in speaker_pool if n and n != speaker_name]
        if not others:
            return ""
        rows = store.get_outgoing_relationships_for(speaker_name, others)
        print(f"[REL DBG] speaker={speaker_name}, others={others}, rows={len(rows) if rows else 0}")
        if not rows:
            return ""
        lines = []
        for r in rows:
            other = r.get("other", "")
            rel = _rel_val_to_text(r.get("rel"))
            att = _rel_val_to_text(r.get("att"))
            notes = _rel_val_to_text(r.get("notes"))
            if not any([rel, att, notes]):
                continue
            seg = f"- 你→{other}：关系={rel or '（未知）'}，态度={att or '（未知）'}"
            if notes:
                seg += f"，备注={notes}"
            lines.append(seg)
        if not lines:
            return ""
        return "你对在场他人的关系与态度（仅你→他人，参考）：\n" + "\n".join(lines) + "\n\n"
    except Exception:
        return ""

# —— Narrator 使用：全体参与者之间的关系/态度矩阵 ——
def build_all_relationships_block_for_narrator(participants: List[str]) -> str:
    try:
        names = [n for n in participants if n]
        if not names:
            return ""
        rows = store.get_relationship_matrix(names)
        print(f"[REL-MATRIX DBG] participants={names}, rows={len(rows) if rows else 0}")
        if not rows:
            return ""
        lines = []
        for r in rows:
            src = r.get("src", "")
            dst = r.get("dst", "")
            rel = _rel_val_to_text(r.get("rel")) or "（未知）"
            att = _rel_val_to_text(r.get("att")) or "（未知）"
            notes = _rel_val_to_text(r.get("notes"))
            seg = f"- {src}→{dst}：关系={rel}，态度={att}"
            if notes:
                seg += f"，备注={notes}"
            lines.append(seg)
        if not lines:
            return ""
        return "在场人物之间的关系与态度（全矩阵，参考）：\n" + "\n".join(lines) + "\n\n"
    except Exception:
        return ""

def handle_memory_triggers_from_reply(reply_text: str, audience_names: List[str], speaker_name: str, conv_id: str):
    triggered = parse_memory_triggers(reply_text)
    if not triggered:
        return
    for raw_content in triggered:
        content = normalize_memory_content(raw_content, speaker_name)
        emb = embed_text(content)
        store.upsert_memory(content, emb)
        # 旁白与导演都应知晓新记忆
        aware_list = list(dict.fromkeys(audience_names + [NARRATOR_NAME, DIRECTOR_NAME]))
        store.link_awareness(aware_list, content)
        store.link_conversation_to_memory(conv_id, content)

# ===================== 旁白与动作旁白 =====================
def _ctx_block_from_rb(rb_history: List[Dict]) -> str:
    if not rb_history:
        return "（无历史；请以自然口吻开场，先抛出一个与话题提示相关的轻松问题。）"
    return "\n".join([f"{t.get('role','')}: {t.get('content','')}" for t in rb_history])

def gen_opening_narration(worldview: str, topic_hint: str, participants: List[str]) -> str:
    rel_block = build_all_relationships_block_for_narrator(participants)
    # 可选：旁白记忆注入
    mem_block = ""
    try:
        if st.session_state.get("enable_narrator_mem", True):
            mem_lines = build_memory_blocks_for_narrator(st.session_state.get("chat_history", []))
            mem_block = join_memory_lines(mem_lines)
    except Exception:
        pass
    # 导演风格与对在场角色的指导
    dir_style, dir_method = None, None
    guides_txt = ""
    try:
        dprof = store.get_director_profile() or {}
        dir_style = dprof.get("style") or ""
        dir_method = dprof.get("method") or ""
        g_rows = store.get_guides_for(list(dict.fromkeys(participants + [NARRATOR_NAME])))
        gmap = {r.get("target"): r for r in (g_rows or [])}
        lines = []
        for nm in participants:
            r = gmap.get(nm) or {}
            sf = r.get("scene_focus") or ""
            si = r.get("state_intent") or ""
            if sf or si:
                lines.append(f"- {nm}：场景关注={sf or '（无）'}；状态/意图={si or '（无）'}")
        if lines or dir_style or dir_method:
            head = f"导演风格：{dir_style or '（未设置）'}；导演方法：{dir_method or '（未设置）'}\n"
            guides_txt = head + ("当前指导：\n" + "\n".join(lines) + "\n" if lines else "")
    except Exception:
        pass
    print(f"[REL-MATRIX USED] empty={not bool(rel_block.strip())}")
    base = (
        f"世界观：{worldview or '（未提供）'}\n" +
        f"到场人物：{', '.join(participants)}\n" +
        f"{rel_block}" +
        ("导演指示：\n" + guides_txt + "\n" if guides_txt else "") +
        ("你已知的关键信息（旁白参考）：\n" + mem_block + "\n\n" if mem_block else "") +
        f"场景提示：{topic_hint or '（无）'}\n\n" +
        "如果认为在场人物之间的关系需要调整，可以先输出一个 [[REL]]...[[/REL]] 区块，" +
        "其中每项为 {src,dst,add:{rel[],att[],notes[]},remove:{rel[],att[],notes[]}}；然后给出开场旁白。\n" +
        "请写一段开场旁白（2-4句），第三人称、气氛清晰，给出一个短期目标或紧张因子；" +
        "仅当出现**新且长期重要**的信息，且不与“你已知的关键信息”重复时，可在末行输出：[[MEMORY]] 一句话 [[/MEMORY]]（可选且最多一条）。\n"
    )
    print(base)  # DEBUG
    try:
        return llm_generate_text(base, tag="opening_narration")
    except Exception as e:
        return f"(旁白生成出错：{e})"

def gen_action_narration(actor_name: str, action_text: str, worldview: str, rb_history: List[Dict], participants: Optional[List[str]] = None) -> str:
    ctx = _ctx_block_from_rb(rb_history)
    rel_block = build_all_relationships_block_for_narrator(participants or [])
    # 可选：旁白记忆注入
    mem_block = ""
    try:
        if st.session_state.get("enable_narrator_mem", True):
            mem_lines = build_memory_blocks_for_narrator(st.session_state.get("chat_history", []))
            mem_block = join_memory_lines(mem_lines)
    except Exception:
        pass
    # 导演风格与在场指导
    dir_style, dir_method = None, None
    guides_txt = ""
    try:
        dprof = store.get_director_profile() or {}
        dir_style = dprof.get("style") or ""
        dir_method = dprof.get("method") or ""
        ps = participants or []
        g_rows = store.get_guides_for(list(dict.fromkeys(ps + [NARRATOR_NAME])))
        gmap = {r.get("target"): r for r in (g_rows or [])}
        lines = []
        for nm in ps:
            r = gmap.get(nm) or {}
            sf = r.get("scene_focus") or ""
            si = r.get("state_intent") or ""
            if sf or si:
                lines.append(f"- {nm}：场景关注={sf or '（无）'}；状态/意图={si or '（无）'}")
        if lines or dir_style or dir_method:
            head = f"导演风格：{dir_style or '（未设置）'}；导演方法：{dir_method or '（未设置）'}\n"
            guides_txt = head + ("当前指导：\n" + "\n".join(lines) + "\n" if lines else "")
    except Exception:
        pass
    print(f"[REL-MATRIX USED in action] empty={not bool(rel_block.strip())}")
    prompt = (
        f"世界观：{worldview or '（未提供）'}\n" +
        f"{rel_block}" +
        ("导演指示：\n" + guides_txt + "\n" if guides_txt else "") +
        ("你已知的关键信息（旁白参考）：\n" + mem_block + "\n\n" if mem_block else "") +
        f"最近片段（含在场台词）：\n{ctx}\n\n" +
        f"现在，{actor_name} 执行动作：{action_text}\n" +
        "如需要，也可以先用 [[REL]]...[[/REL]] 区块调整在场任意人物间的关系，然后再写旁白。\n" +
        "请用第三人称写出该动作的过程与结果（1-3句），不要编造台词；" +
        "仅当出现**新且长期重要**的信息，且不与“你已知的关键信息”重复时，可在末行额外输出：[[MEMORY]] 一句话 [[/MEMORY]]（可选，最多一条）。\n"
    )
    print(prompt)  # DEBUG
    try:
        return llm_generate_text(prompt, tag=f"action_narration:{actor_name}")
    except Exception as e:
        return f"(动作旁白生成出错：{e})"

def handle_action_triggers_from_reply(reply_text: str, actor_name: str, worldview: str, conv_id: str) -> bool:
    """解析 [[ACT]]... 生成动作旁白；返回是否出现动作。"""
    # ==== CHANGED: 入口处先规范化，提升命中率 ====
    reply_text = _normalize_action_markup(reply_text or "")
    acts = parse_action_triggers(reply_text)
    if not acts:
        return False
    did_any = False
    for act in acts:
        rb_for_narr = store.get_role_recentbuffer(actor_name)
        participants = list(dict.fromkeys(st.session_state.locked_participants + [protagonist_name()]))
        narration_raw = gen_action_narration(actor_name, act, worldview, rb_for_narr, participants)
        # 解析并应用旁白里的关系更新（可选）
        rel_updates = parse_relationship_updates(narration_raw)
        participants_all = list(dict.fromkeys(st.session_state.locked_participants + [protagonist_name()]))
        if rel_updates:
            apply_relationship_trigger_updates(rel_updates, participants_all)
        nar_clean = strip_action_blocks(strip_memory_blocks(strip_relationship_blocks(narration_raw))).strip()
        if not nar_clean:
            continue
        nar_turn = {"role": NARRATOR_NAME, "content": nar_clean}
        st.session_state.chat_history.append(nar_turn)
        store.set_conversation_history(st.session_state.conv_id, st.session_state.chat_history)
        # >>> NEW: RecentBuffer 写入“解析后的旁白”（含 embedding）
        rb_nar_text = resolve_narrator_you(nar_clean)
        rb_nar_turn = {"role": NARRATOR_NAME, "content": rb_nar_text, "embedding": embed_text(rb_nar_text)}
        try:
            rag_dbg(f"[RB] append role={NARRATOR_NAME} emb_len={len(rb_nar_turn.get('embedding') or [])} text='{rb_nar_text[:40]}'")
        except Exception:
            pass
        roles_to_update = list(dict.fromkeys(st.session_state.locked_participants + [protagonist_name()]))
        store.append_turn_to_roles_recent(roles_to_update, rb_nar_turn, keep=RB_MAX_TURNS_PER_ROLE)
        audience = list(dict.fromkeys(st.session_state.locked_participants + [protagonist_name()]))
        handle_memory_triggers_from_reply(narration_raw, audience, NARRATOR_NAME, conv_id)
        did_any = True
    return did_any

# ===================== 推进提示（核心改动） =====================
def action_bias_text() -> str:
    turns = int(st.session_state.get("turns_since_last_act", 999))
    return ACTION_BIAS_FORCE_TEXT if turns >= ACTION_BIAS_TURNS else ACTION_BIAS_SOFT_TEXT

STORY_DRIVE_RULES = (
    "请满足下列其一（优先靠前项）：\n"
    "1) 推进冲突或目标：提出/回答一个推动剧情的**关键问题**，抛出新线索或障碍，但保留适度悬念；\n"
    "2) 明确下一步：提出一个**可执行的提议**（谁/何时/何地/做什么），或对他人提议给出清晰立场（同意/反对/附加条件）；\n"
    "3) 角色扮演：在符合人设与动机前提下表达态度与情绪（如幽默、讽刺、敷衍、试探、欺骗、质疑、拒绝、阴谋、敌对、讨价还价等），不要所有 NPC 都无条件支持主角；\n"
    "4) 当且仅当你亲自执行具体动作时，才在末行追加 [[ACT]] 动作简述 [[/ACT]]。\n"
    "写作风格约束：\n"
    "- 只输出你的台词；不要为他人代言/安排行动；不要写叙事旁白；\n"
    "- 台词要具体可感，优先“展示”而非“讲解”，避免空泛口号；可用语气词或细微动作词辅助情绪，但不描写环境；\n"
    "- 适度利用关系与记忆：可以点名他人、引用过往、提醒利益/风险/承诺，制造轻微张力（设置条件、交换信息、索要保证）；\n"
    "- 信息边界：不要知道你不应知道的内容；不要无依据泄露世界设定；\n"
    "- 若上一条台词点名你或向你提问，优先回应；否则可以主动推进；\n"
    "- 避免寒暄、复读与机械确认。"
)

ACT_USAGE_TIP = "动作标记写法：[[ACT]] 我做了什么（简短祈使/陈述） [[/ACT]]（也支持全角【ACT】写法）"

# ===================== 文本生成（加入 Memory 注入 & 动作偏置） =====================
def gen_with_role_recentbuffer(
    speaker_name: str,
    speaker_persona: str,
    worldview: str,
    rb_history: List[Dict],
    topic_hint: str = "",
    mem_lines: Optional[List[str]] = None
) -> str:
    # —— 参考线索（稍后放） ——
    ctx = _ctx_block_from_rb(rb_history)
    last_role = rb_history[-1]["role"] if rb_history else ""
    last_text = rb_history[-1]["content"] if rb_history else ""
    rel_block = build_relationship_block_for_prompt(speaker_name, all_names_in_scene())
    print(f"[REL OUT USED] speaker={speaker_name}, empty={not bool(rel_block.strip())}")

    # —— 导演指示（风格/方法 + 对该角色的 scene_focus/state_intent） ——
    dir_lines = []
    try:
        dprof = store.get_director_profile() or {}
        dstyle = dprof.get("style") or ""
        dmethod = dprof.get("method") or ""
        if dstyle or dmethod:
            dir_lines.append(f"导演风格：{dstyle or '（未设置）'}；导演方法：{dmethod or '（未设置）'}")
        grows = store.get_guides_for([speaker_name])
        if grows:
            r = grows[0]
            sf = r.get("scene_focus") or ""
            si = r.get("state_intent") or ""
            if sf or si:
                dir_lines.append(f"给你的当前指导：场景关注={sf or '（无）'}；状态/意图={si or '（无）'}")
    except Exception:
        pass
    director_block = ("\n".join(dir_lines) + "\n\n") if dir_lines else ""

    # —— 身份与任务（置顶） ——
    header = (
        f"你是：{speaker_name}\n"
        f"人设：{speaker_persona or '（未提供）'}\n\n"
        f"{director_block}"
        f"任务：\n"
        f"{STORY_DRIVE_RULES}\n"
        f"{action_bias_text()}\n"
        f"- 只输出你的台词，不要名字前缀，不要加引号；保持自然、自洽。\n"
        f"- 仅当你确实执行具体动作时，才在末行使用 [[ACT]]…[[/ACT]]；不要滥用特殊标记。\n"
        f"- 【记忆触发】当且仅当出现新且长期重要、且不与“已知关键信息”重复的内容时，可在末行输出：[[MEMORY]] 一句话 [[/MEMORY]]（最多一条）。\n"
        "- 在输出台词之前，如需调整你→他人的关系/态度，请先输出一个 [[REL]]...[[/REL]] 区块（src 必须等于你；不要修改他人→你的关系）。\n"
        "示例：\n[[REL]]\n[\n  {\"src\":\"%s\",\"dst\":\"某人\",\"add\":{\"rel\":[\"朋友\"]},\"remove\":{\"att\":[\"怀疑\"]}}\n]\n[[/REL]]\n" % (speaker_name,)
    )

    # —— 其他线索（放在任务之后） ——
    mem_block = join_memory_lines(mem_lines or [])
    clues_parts = []
    if mem_block:
        clues_parts.append("你已知的关键信息（可能有用）：\n" + mem_block)
    if rel_block.strip():
        clues_parts.append(rel_block.strip())
    clues_parts.append(f"世界观：{worldview or '（未提供）'}")
    clues_parts.append(f"话题提示：{topic_hint or '（无）'}")
    clues_parts.append(f"== 你参与的最近对话片段（含所有人台词） ==\n{ctx}\n最近一条：{last_role or '（无）'}：{last_text or '（无）'}")
    clues = "\n\n".join([p for p in clues_parts if p]) + "\n"

    prompt = header + "\n" + clues
    print(prompt)  # DEBUG
    try:
        return llm_generate_text(prompt, tag=f"npc_line:{speaker_name}")
    except Exception as e:
        return f"(模型调用出错：{e})"

def suggest_for_protagonist(
    protagonist_persona: str,
    worldview: str,
    rb_history: List[Dict],
    topic_hint: str = "",
    mem_lines: Optional[List[str]] = None
) -> str:
    # —— 参考线索（稍后放） ——
    ctx = _ctx_block_from_rb(rb_history)
    last_role = rb_history[-1]["role"] if rb_history else ""
    last_text = rb_history[-1]["content"] if rb_history else ""
    rel_block = build_relationship_block_for_prompt(protagonist_name(), all_names_in_scene())
    print(f"[REL OUT USED] speaker={protagonist_name()}, empty={not bool(rel_block.strip())}")

    # —— 导演指示（风格/方法 + 对主角的 scene_focus/state_intent） ——
    dir_lines = []
    try:
        dprof = store.get_director_profile() or {}
        dstyle = dprof.get("style") or ""
        dmethod = dprof.get("method") or ""
        if dstyle or dmethod:
            dir_lines.append(f"导演风格：{dstyle or '（未设置）'}；导演方法：{dmethod or '（未设置）'}")
        grows = store.get_guides_for([protagonist_name()])
        if grows:
            r = grows[0]
            sf = r.get("scene_focus") or ""
            si = r.get("state_intent") or ""
            if sf or si:
                dir_lines.append(f"给你的当前指导：场景关注={sf or '（无）'}；状态/意图={si or '（无）'}")
    except Exception:
        pass
    director_block = ("\n".join(dir_lines) + "\n\n") if dir_lines else ""

    # —— 身份与任务（置顶） ——
    header = (
    f"你是：{protagonist_name()}\n"
        f"人设：{protagonist_persona or '（未提供）'}\n\n"
        f"{director_block}"
        f"任务：\n"
        f"{STORY_DRIVE_RULES}\n"
        f"{action_bias_text()}\n"
        f"- 给出一条建议台词（第一人称“我”，1-2句；不要加引号/名字前缀）。\n"
        f"- 仅当你确实要执行动作时，才在末行追加 [[ACT]] 动作简述 [[/ACT]]。\n"
        "- 在输出建议之前，如需调整你→他人的关系/态度，请先输出一个 [[REL]]...[[/REL]] 区块。\n"
    "示例：\n[[REL]]\n[\n  {\"src\":\"%s\",\"dst\":\"某人\",\"add\":{\"rel\":[\"朋友\"]},\"remove\":{\"att\":[\"怀疑\"]}}\n]\n[[/REL]]\n" % (protagonist_name(),)
    )

    # —— 其他线索（放在任务之后） ——
    mem_block = join_memory_lines(mem_lines or [])
    clues_parts = []
    if mem_block:
        clues_parts.append("你已知的关键信息（可能有用）：\n" + mem_block)
    if rel_block.strip():
        clues_parts.append(rel_block.strip())
    clues_parts.append(f"世界观：{worldview or '（未提供）'}")
    clues_parts.append(f"话题提示：{topic_hint or '（无）'}")
    clues_parts.append(f"== 你参与的最近片段 ==\n{ctx}\n最近一条：{last_role or '（无）'}：{last_text or '（无）'}")
    clues = "\n\n".join([p for p in clues_parts if p]) + "\n"

    prompt = header + "\n" + clues
    print(prompt)  # DEBUG
    try:
        return llm_generate_text(prompt, tag="protagonist_suggest")
    except Exception as e:
        return f"(建议生成出错：{e})"

# ===================== 导演干预（定期为所有角色提供指导） =====================
def _first_k_dialogue(turns: List[Dict], k: int = 3) -> str:
    if not turns:
        return "（无）"
    head = turns[: max(0, int(k))]
    return "\n".join([f"{t.get('role','')}: {t.get('content','')}" for t in head])

def _recent_rb_snapshot(names: List[str], per_role: int = 2) -> str:
    lines = []
    for n in names:
        rb = store.get_role_recentbuffer(n) or []
        tail = rb[-max(0, int(per_role)):] if rb else []
        for t in tail:
            lines.append(f"{n}: {t.get('content','')}")
    return "\n".join(lines) if lines else "（无）"

def _uniq(seq: List[str]) -> List[str]:
    out = []
    seen = set()
    for x in seq or []:
        if x not in seen and x:
            seen.add(x); out.append(x)
    return out

from typing import Tuple

def _dir_cadence_range() -> Tuple[int, int]:
    try:
        mn = int(st.session_state.get("director_cadence_min", DEFAULT_DIR_CADENCE_MIN))
        mx = int(st.session_state.get("director_cadence_max", DEFAULT_DIR_CADENCE_MAX))
        if mn < 1:
            mn = 1
        if mx < mn:
            mx = mn
        return mn, mx
    except Exception:
        return DEFAULT_DIR_CADENCE_MIN, DEFAULT_DIR_CADENCE_MAX

def generate_director_guidance(participants: List[str], topic_hint: str = "", worldview: str = "", is_opening: bool = False) -> List[Dict]:
    names = _uniq(list(participants or []) + [NARRATOR_NAME])
    try:
        dprof = store.get_director_profile() or {}
        dstyle = dprof.get("style") or ""
        dmethod = dprof.get("method") or ""
    except Exception:
        dstyle = ""; dmethod = ""
    # 关系矩阵
    try:
        rel_rows = store.get_relationship_matrix(names) or []
    except Exception:
        rel_rows = []
    rel_lines = []
    for r in rel_rows:
        src = r.get("src") or ""; dst = r.get("dst") or ""
        rel = ",".join(r.get("rel") or [])
        att = ",".join(r.get("att") or [])
        notes = ",".join(r.get("notes") or [])
        rel_lines.append(f"{src} -> {dst} | rel=[{rel}] att=[{att}] notes=[{notes}]")
    rel_txt = "\n".join(rel_lines) or "（无）"
    # 对话首三段
    first3 = _first_k_dialogue(st.session_state.get("chat_history", []), 3)
    # 每个角色的记忆线索（RAG）与 RB 摘要
    mem_map = {}
    for n in names:
        try:
            mem_lines = build_memory_blocks_for_speaker(n, st.session_state.get("chat_history", []), names)
            mem_map[n] = "\n".join(mem_lines or [])
        except Exception:
            mem_map[n] = ""
    rb_snap = _recent_rb_snapshot(names, per_role=2)
    # 生成 JSON 指导
    mem_bundle = "\n".join([f"- {k}:\n{v}" for k, v in mem_map.items() if v])
    prompt = (
        "你是电影/小说的导演。你的目标是减少重复、增加戏剧性，并让故事朝新的方向发展。\n"
        f"导演风格：{dstyle or '（未设置）'}；导演方法：{dmethod or '（未设置）'}\n"
        f"在场角色（含旁白）：{', '.join(names)}\n\n"
        f"共同世界观：{(worldview or '（未提供）')}\n"
        f"场景/话题提示：{(topic_hint or '（无）')}\n"
        + ("（开场）请首先基于提示确立本场戏的焦点与各角色初始意图；\n" if is_opening else "")
        + "\n关系矩阵（有向，仅存在的边）：\n" + rel_txt + "\n\n"
        + "对话前3段（用于确立前提）：\n" + first3 + "\n\n"
        + "每个角色的关键信息（RAG）摘要：\n" + (mem_bundle or "（无）") + "\n\n"
        + "RecentBuffer 摘要（每人最近2条）：\n" + rb_snap + "\n\n"
        + "请针对上面每个角色（包含旁白）生成一条指导，严格输出 JSON 数组，不要任何额外文本。\n"
        + "JSON 结构：[{\"target\":\"角色名\", \"scene_focus\":\"外在状况/关注主题<=20字\", \"state_intent\":\"该角色当前状态与想做<=20字\"}, ...]。\n"
        + "要求：\n"
        + "- 指导要具体、能推动剧情朝新的方向发展，避免复读；\n"
        + "- 合理利用关系与记忆，不违背人设；\n"
        + "- 不要安排行动，只给指导；\n"
        + "- 确保覆盖输入列出的所有角色名（包含旁白）。\n"
    )
    try:
        txt = llm_generate_text(prompt, tag="director_guidance") or "[]"
    except Exception:
        txt = "[]"
    # 提取 JSON
    import json as _json
    cleaned = txt
    # 容错：截取第一个 '[' 到最后一个 ']'
    if "[" in cleaned and "]" in cleaned:
        try:
            i = cleaned.index("["); j = cleaned.rindex("]")
            cleaned = cleaned[i:j+1]
        except Exception:
            pass
    try:
        data = _json.loads(cleaned)
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return []

def maybe_director_intervene(participants: List[str], force: bool = False):
    if not st.session_state.get("director_enabled", True):
        return
    turns = int(st.session_state.get("turns_since_last_director", 0))
    due_in = int(st.session_state.get("director_due_in", 0))
    if not force:
        if due_in <= 0:
            mn, mx = _dir_cadence_range()
            st.session_state.director_due_in = random.randint(mn, mx)
            st.session_state.turns_since_last_director = 0
            return
        if turns + 1 < due_in:
            st.session_state.turns_since_last_director = turns + 1
            return
    # 触发一次导演指导
    targets = _uniq(list(participants or []) + [NARRATOR_NAME])
    rows = generate_director_guidance(targets, topic_hint=st.session_state.get("ti_topic", ""), worldview=current_worldview() if 'current_worldview' in globals() else "")
    updated = 0
    updated_lines = []
    for r in rows or []:
        tgt = (r.get("target") or "").strip()
        if tgt in targets:
            sf = (r.get("scene_focus") or "").strip()
            si = (r.get("state_intent") or "").strip()
            try:
                store.upsert_guides(tgt, sf, si)
                updated += 1
                if sf or si:
                    updated_lines.append(f"- {tgt}：场景关注={sf or '（无）'}；状态/意图={si or '（无）'}")
            except Exception:
                pass
    try:
        print(f"[DIRECTOR] updated guides for {updated}/{len(targets)} targets")
    except Exception:
        pass
    # 若有更新，添加一条简短“导演指示（更新）”到历史
    if updated_lines and st.session_state.get("conv_locked"):
        dprof = store.get_director_profile() or {}
        dstyle = dprof.get("style") or ""
        dmethod = dprof.get("method") or ""
        header = "导演指示（更新）\n"
        if dstyle or dmethod:
            header += f"导演风格：{dstyle or '（未设置）'}；导演方法：{dmethod or '（未设置）'}\n"
        dir_msg = header + "\n".join(updated_lines)
        turn = {"role": DIRECTOR_NAME, "content": dir_msg}
        st.session_state.chat_history.append(turn)
        store.set_conversation_history(st.session_state.conv_id, st.session_state.chat_history)
        rb_turn = {"role": DIRECTOR_NAME, "content": dir_msg, "embedding": embed_text(dir_msg)}
        roles_to_update = list(dict.fromkeys(st.session_state.locked_participants + [protagonist_name()]))
        store.append_turn_to_roles_recent(roles_to_update, rb_turn, keep=RB_MAX_TURNS_PER_ROLE)

    # 重置节奏（使用 UI 范围）
    st.session_state.turns_since_last_director = 0
    mn, mx = _dir_cadence_range()
    st.session_state.director_due_in = random.randint(mn, mx)

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
    print(prompt)  # DEBUG
    try:
        txt = llm_generate_text(prompt, tag="action_ideas")
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

# 不把功能性角色放入可选列表；允许选择任意非功能角色作为主角
_selectable_chars = [c for c in st.session_state.characters if not c.get("functional", False)]
all_char_names = [c["name"] for c in _selectable_chars]
if "protagonist_name" not in st.session_state:
    st.session_state.protagonist_name = (DEFAULT_PROTAGONIST_NAME if DEFAULT_PROTAGONIST_NAME in all_char_names else (all_char_names[0] if all_char_names else ""))

st.markdown("### 选择主角")
st.session_state.protagonist_name = st.selectbox(
    "选择作为主角的角色（仅用于前端交互，存储中与 NPC 无区别）",
    options=all_char_names,
    index=(all_char_names.index(st.session_state.protagonist_name) if st.session_state.protagonist_name in all_char_names else 0),
    key="sb_protagonist"
)

# NPC 列表中排除选定的主角；主角将自动加入会话（并且排除功能性角色）
npc_chars = [c for c in _selectable_chars if c["name"] != protagonist_name()]
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
    st.text(f"{display_name_ui(protagonist_name())} 的发言欲望固定为 {PROTAGONIST_DEFAULT_DESIRE}。")
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
    store.ensure_character(protagonist_name(), protagonist_persona, protagonist_worldview or "", PROTAGONIST_DEFAULT_DESIRE)
    for n in st.session_state.locked_participants:
        store.ensure_character(n)
    participants_payload = (
    [{"name": n, "role_tag": "npc"} for n in st.session_state.locked_participants] +
    [{"name": protagonist_name(), "role_tag": "protagonist"}]
    )
    store.ensure_conversation_with_participants(cid, conv_title, participants_payload)
    st.session_state.chat_history = store.get_conversation_history(cid) or []
    # 初始化导演节奏（使用 UI 范围）
    st.session_state.turns_since_last_director = 0
    _mn, _mx = _dir_cadence_range()
    st.session_state.director_due_in = random.randint(_mn, _mx)

    # —— 开场先运行“导演”：基于话题/世界观生成并落库指导，同时输出一条简短的导演指示 ——
    try:
        _targets = list(dict.fromkeys([protagonist_name()] + st.session_state.locked_participants + [NARRATOR_NAME]))
        dir_rows = generate_director_guidance(_targets, topic_hint=topic_hint, worldview=(protagonist_worldview or ""), is_opening=True)
        updated = 0
        lines = []
        for r in dir_rows or []:
            tgt = (r.get("target") or "").strip()
            if tgt in _targets:
                sf = (r.get("scene_focus") or "").strip()
                si = (r.get("state_intent") or "").strip()
                store.upsert_guides(tgt, sf, si)
                updated += 1
                if sf or si:
                    lines.append(f"- {tgt}：场景关注={sf or '（无）'}；状态/意图={si or '（无）'}")
        if lines:
            # 生成一条简短的“导演指示（开场）”消息
            dprof = store.get_director_profile() or {}
            dstyle = dprof.get("style") or ""
            dmethod = dprof.get("method") or ""
            header = "导演指示（开场）\n"
            if topic_hint or protagonist_worldview:
                header += f"场景/话题：{topic_hint or '（无）'}；世界观：{protagonist_worldview or '（未提供）'}\n"
            if dstyle or dmethod:
                header += f"导演风格：{dstyle or '（未设置）'}；导演方法：{dmethod or '（未设置）'}\n"
            dir_msg = header + "\n".join(lines)
            # 写入会话历史
            dir_turn = {"role": DIRECTOR_NAME, "content": dir_msg}
            st.session_state.chat_history.append(dir_turn)
            store.set_conversation_history(st.session_state.conv_id, st.session_state.chat_history)
            # RecentBuffer：导演与旁白共享 RB，且需要为所有参与者附着该条
            rb_turn = {"role": DIRECTOR_NAME, "content": dir_msg, "embedding": embed_text(dir_msg)}
            roles_to_update = list(dict.fromkeys(st.session_state.locked_participants + [protagonist_name()]))
            store.append_turn_to_roles_recent(roles_to_update, rb_turn, keep=RB_MAX_TURNS_PER_ROLE)
    except Exception as _e:
        try:
            print(f"[DIRECTOR opening] failed: {_e}")
        except Exception:
            pass

    # 开场旁白（随后）
    opening = gen_opening_narration(
        worldview=protagonist_worldview or "",
        topic_hint=topic_hint,
        participants=[protagonist_name()] + st.session_state.locked_participants
    )
    # 开场旁白可带 [[REL]] 区块，先应用再清理
    rel_updates = parse_relationship_updates(opening)
    if rel_updates:
        participants_all = [protagonist_name()] + st.session_state.locked_participants
        apply_relationship_trigger_updates(rel_updates, participants_all)
    opening_clean = strip_action_blocks(strip_memory_blocks(strip_relationship_blocks(opening)))
    if opening_clean:
        nar_turn = {"role": NARRATOR_NAME, "content": opening_clean}
        # 追加到会话历史
        st.session_state.chat_history.append(nar_turn)
        store.set_conversation_history(st.session_state.conv_id, st.session_state.chat_history)

        # RecentBuffer 写入“解析后的旁白”（把“你”替换为主角），并保存 embedding
        rb_text = resolve_narrator_you(opening_clean)
        rb_nar_turn = {"role": NARRATOR_NAME, "content": rb_text, "embedding": embed_text(rb_text)}
        try:
            rag_dbg(f"[RB] append role={NARRATOR_NAME} emb_len={len(rb_nar_turn.get('embedding') or [])} text='{rb_text[:40]}'")
        except Exception:
            pass
        roles_to_update = list(dict.fromkeys(st.session_state.locked_participants + [protagonist_name()]))
        store.append_turn_to_roles_recent(roles_to_update, rb_nar_turn, keep=RB_MAX_TURNS_PER_ROLE)

    # 处理可能的记忆触发
    audience = list(dict.fromkeys(st.session_state.locked_participants + [protagonist_name()]))
    handle_memory_triggers_from_reply(opening, audience, NARRATOR_NAME, st.session_state.conv_id)
    # 导演可能干预（根据轮数节奏）
    maybe_director_intervene([protagonist_name()] + st.session_state.locked_participants)

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
    role_disp = display_name_ui(t.get('role','角色'))
    st.chat_message("assistant").markdown(f"**{role_disp}**：{t.get('content','')}")

# ---- 新建角色 ----
with st.expander("＋ 新建角色"):
    with st.form("add_char_form"):
        name = st.text_input("角色名", key="ti_new_name")
        persona = st.text_area("人设", key="ta_new_persona")
        worldview = st.text_area("世界观", key="ta_new_world")
        desire = st.slider("初始发言欲望", 0.0, 1.0, value=0.3, step=0.05, key="sl_new_desire")
        submitted = st.form_submit_button("添加角色")
        if submitted and name and persona:
            if name in (NARRATOR_NAME, DIRECTOR_NAME):
                st.error("旁白/导演为功能性角色，不能手工创建或修改。")
            else:
                store.ensure_character(name, persona, worldview, desire)
                st.session_state.characters = store.get_characters()
                # 刷新后也会带出 functional 标记，用于 UI 过滤
                st.success(f"角色 {name} 已添加！")

# ---- 控制区 ----
st.markdown("### 控制")
cdbg1, cdbg2 = st.columns(2)
with cdbg1:
    st.session_state.debug_weights = st.checkbox("打印权重调试", value=st.session_state.get("debug_weights", True), key="cb_debug_weights")
with cdbg2:
    st.session_state.auto_protagonist_autosend = st.checkbox("允许主角自动发言（自动采用建议）", value=st.session_state.get("auto_protagonist_autosend", False), key="cb_auto_protagonist")
# —— RAG 调试开关 ——
st.session_state.debug_rag = st.checkbox("打印RAG检索调试", value=st.session_state.get("debug_rag", True), key="cb_debug_rag")
copt_llm = st.checkbox("打印LLM请求/响应", value=st.session_state.get("debug_llm", True), key="cb_debug_llm")
copt1, copt2, copt3, copt4, copt5, copt6 = st.columns(6)
with copt1:
    st.session_state.enable_narrator_mem = st.checkbox(
        "旁白记忆注入", value=st.session_state.get("enable_narrator_mem", True), key="cb_narrator_mem"
    )
with copt2:
    st.session_state.director_enabled = st.checkbox(
        "启用导演定期指导", value=st.session_state.get("director_enabled", True), key="cb_director_enable"
    )
with copt3:
    # 占位，让后面的 UI 对齐
    st.empty()
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

# —— 导演设置 ——
with st.expander("🎛️ 导演设置"):
    min_default = int(st.session_state.get("director_cadence_min", DEFAULT_DIR_CADENCE_MIN))
    max_default = int(st.session_state.get("director_cadence_max", DEFAULT_DIR_CADENCE_MAX))
    col_min, col_max, col_btn = st.columns([0.3, 0.3, 0.4])
    with col_min:
        st.session_state.director_cadence_min = st.number_input(
            "干预最少间隔（轮）", min_value=1, max_value=50, value=min_default, step=1, key="ni_dir_min"
        )
    with col_max:
        st.session_state.director_cadence_max = st.number_input(
            "干预最多间隔（轮）", min_value=1, max_value=50, value=max_default, step=1, key="ni_dir_max"
        )
    with col_btn:
        manual_intervene = st.button("立即介入一次（导演）", key="btn_dir_now")

if 'manual_intervene' in locals() and manual_intervene and st.session_state.get("conv_locked"):
    maybe_director_intervene([protagonist_name()] + st.session_state.locked_participants, force=True)
    st.rerun()

# —— 导演指导（scene_focus / state_intent）——
with st.expander("🎬 导演指导（场景关注 / 角色意图）"):
    try:
        scene_participants_ui = st.session_state.locked_participants if st.session_state.conv_locked else participants_selected
        targets = list(
            dict.fromkeys(
                [protagonist_name()] + scene_participants_ui + ([NARRATOR_NAME] if NARRATOR_NAME not in scene_participants_ui else [])
            )
        )
        g_rows = store.get_guides_for(targets)
        guide_map = {r["target"]: r for r in (g_rows or [])}
        for nm in targets:
            colA, colB = st.columns(2)
            with colA:
                sf = st.text_input(f"{nm} · scene_focus（外在/主题）", value=(guide_map.get(nm, {}).get("scene_focus") or ""), key=f"sf_{nm}")
            with colB:
                si = st.text_input(f"{nm} · state_intent（状态/想做）", value=(guide_map.get(nm, {}).get("state_intent") or ""), key=f"si_{nm}")
            if st.button(f"更新 {nm}", key=f"upd_{nm}"):
                store.upsert_guides(nm, sf, si)
                st.success(f"已更新 {nm} 的指导")
    except Exception as _e:
        st.caption("（指导读取失败，可稍后重试）")

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
    "name": protagonist_name(),
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
        roles_to_update = list(dict.fromkeys(st.session_state.locked_participants + [protagonist_name()]))
        # >>> NEW: RB 中存 embedding；旁白写入解析版
        if speaker == NARRATOR_NAME:
            rb_text = resolve_narrator_you(text.strip())
            rb_turn = {"role": speaker, "content": rb_text, "embedding": embed_text(rb_text)}
        else:
            rb_turn = {"role": speaker, "content": text.strip(), "embedding": embed_text(text.strip())}
        try:
            rag_dbg(f"[RB] append role={speaker} emb_len={len(rb_turn.get('embedding') or [])} text='{rb_turn.get('content','')[:40]}'")
        except Exception:
            pass
        store.append_turn_to_roles_recent(roles_to_update, rb_turn, keep=RB_MAX_TURNS_PER_ROLE)

    did_act = False
    if raw_reply_for_trigger:
        audience = list(dict.fromkeys(st.session_state.locked_participants + [protagonist_name()]))
        handle_memory_triggers_from_reply(raw_reply_for_trigger, audience, speaker, st.session_state.conv_id)
        did_act = handle_action_triggers_from_reply(raw_reply_for_trigger, speaker, current_worldview(), st.session_state.conv_id)

    # 更新计数
    st.session_state.turns_since_last_act = 0 if did_act else st.session_state.turns_since_last_act + 1
    # 回合后尝试导演干预（按节奏）
    if st.session_state.conv_locked:
        maybe_director_intervene([protagonist_name()] + st.session_state.locked_participants)

# ===================== 生成一步（含 Memory/Action 注入） =====================
def one_step_generate(skip_protagonist: bool = False, exclude_name: Optional[str] = None):
    if not scene_participants:
        st.warning("请至少选择一个 NPC。")
        return

    base_weights = {p["name"]: float(p["desire"]) for p in active_participants}

    # >>> NEW: 被点名者强力加权；未点名问句 → 给“沉默最久且非提问者”的角色 +0.5 临时加成
    tmp_weights = dict(base_weights)
    last_turn = st.session_state.chat_history[-1] if st.session_state.chat_history else None
    last_speaker = last_turn.get("role") if last_turn else None
    last_text = last_turn.get("content", "") if last_turn else ""
    if last_turn:
        others = [n for n in all_names_in_scene() if n != last_speaker]
        # 点名：上一条文本包含某参与者的准确名字 → 该参与者临时权重 +2
        addressed = [n for n in others if n and (n in last_text)]
        if addressed:
            for n in addressed:
                tmp_weights[n] = tmp_weights.get(n, 0.0) + 2.0
        # 否则：若为未点名的问句，则倾向沉默最久者回应（+0.5）
        elif is_question(last_text):
            ages = silence_ages(st.session_state.chat_history, all_names_in_scene())
            candidates = {n: a for n, a in ages.items() if n != last_speaker}
            if candidates:
                longest_silent = max(candidates.items(), key=lambda x: x[1])[0]
                tmp_weights[longest_silent] = min(1.0, tmp_weights.get(longest_silent, 0.0) + 0.5)

    # 公平性加权：长时间沉默者获得临时加成（与点名+2并存）
    ages_all = silence_ages(st.session_state.chat_history, all_names_in_scene())
    for n, age in ages_all.items():
        if n == last_speaker:
            continue
        bonus = 0.0
        if age >= 8:
            bonus = 0.5
        elif age >= 4:
            bonus = 0.25
        if bonus > 0:
            tmp_weights[n] = tmp_weights.get(n, 0.0) + bonus

    # —— 根据导演指导微调权重 ——
    try:
        targets = all_names_in_scene()
        g_rows = store.get_guides_for(targets)
        guide_map = {r.get("target"): r for r in (g_rows or [])}
        # 若指导中的 state_intent 有“想…/打算…/准备…/需要回应”等词，适度加权
        intent_keywords = ["想", "打算", "准备", "需要", "回应", "解释", "反驳", "推进", "开口", "说明"]
        for name in targets:
            gi = guide_map.get(name) or {}
            si = (gi.get("state_intent") or "").strip()
            if si and any(k in si for k in intent_keywords):
                tmp_weights[name] = tmp_weights.get(name, 0.0) + 0.35
        # 冲突链条：若上一条台词点名了某人，则该被点名者再+0.25，促成来回
        if last_turn and last_text:
            for name in targets:
                if name != last_speaker and name in last_text:
                    tmp_weights[name] = tmp_weights.get(name, 0.0) + 0.25
    except Exception:
        pass

    if skip_protagonist:
        tmp_weights[protagonist_name()] = 0.0
    if exclude_name:
        tmp_weights[exclude_name] = 0.0

    if st.session_state.get("debug_weights", True):
        try:
            print(f"[WEIGHT DBG] base={base_weights}")
            print(f"[WEIGHT DBG] after tmp={tmp_weights}, ages={ages_all}, skip_protagonist={skip_protagonist}, exclude={exclude_name}")
        except Exception:
            pass

    speaker_name = weighted_next_speaker(tmp_weights, exclude=last_speaker)
    if st.session_state.get("debug_weights", True):
        print(f"[WEIGHT DBG] chosen={speaker_name}")
    sp = next(p for p in active_participants if p["name"] == speaker_name)

    rb_history = store.get_role_recentbuffer(sp["name"])
    mem_lines = build_memory_blocks_for_speaker(speaker_name, st.session_state.chat_history, all_names_in_scene())
    try:
        rag_dbg(f"[RAG] injected for {speaker_name}: {mem_lines}")
    except Exception:
        pass

    if sp["name"] == protagonist_name():
        sugg_raw = suggest_for_protagonist(
            protagonist_persona=protagonist_persona,
            worldview=current_worldview(),
            rb_history=rb_history,
            topic_hint=topic_hint,
            mem_lines=mem_lines
        )
        clean_sugg = strip_action_blocks(strip_memory_blocks(strip_relationship_blocks(sugg_raw)))
        if st.session_state.get("auto_protagonist_autosend", False):
            # 自动采用建议并发送
            line = sanitize_line(clean_sugg, protagonist_name(), all_names_in_scene()).strip()
            if line:
                # 若建议中包含关系变更，则先应用（仅限主角→他人）
                rel_updates = parse_relationship_updates(sugg_raw)
                if rel_updates:
                    participants_all = list(dict.fromkeys(st.session_state.locked_participants + [protagonist_name()]))
                    rel_updates = [u for u in rel_updates if u.get("src") == protagonist_name() and u.get("dst") in participants_all]
                    if rel_updates:
                        apply_relationship_trigger_updates(rel_updates, participants_all)
                persist_everything(protagonist_name(), line, raw_reply_for_trigger=sugg_raw)
                return "npc"
            else:
                # 没有有效台词，则仅处理触发，不追加台词
                audience = list(dict.fromkeys(st.session_state.locked_participants + [protagonist_name()]))
                handle_memory_triggers_from_reply(sugg_raw, audience, protagonist_name(), st.session_state.conv_id)
                did_act = handle_action_triggers_from_reply(sugg_raw, protagonist_name(), current_worldview(), st.session_state.conv_id)
                st.session_state.turns_since_last_act = 0 if did_act else st.session_state.turns_since_last_act + 1
                return "npc"
        else:
            # 维持原交互：等待玩家确认发送
            st.session_state.awaiting_player_input = True
            st.session_state.player_suggested_line = sanitize_line(clean_sugg, protagonist_name(), all_names_in_scene())
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
    # 解析模型返回中的关系区块（可选）并暂存为本轮待应用的关系更新
    rel_updates = parse_relationship_updates(raw)
    # 清理 MEMORY/ACT/REL 区块得到干净台词
    cleaned = strip_action_blocks(strip_memory_blocks(strip_relationship_blocks(raw)))
    line = sanitize_line(cleaned, sp["name"], all_names_in_scene()).strip()

    if not line:
        audience = list(dict.fromkeys(st.session_state.locked_participants + [protagonist_name()]))
        handle_memory_triggers_from_reply(raw, audience, sp["name"], st.session_state.conv_id)
        did_act = handle_action_triggers_from_reply(raw, sp["name"], current_worldview(), st.session_state.conv_id)
        st.session_state.turns_since_last_act = 0 if did_act else st.session_state.turns_since_last_act + 1
        return "npc"

    # 应用关系更新（仅限参与者之间）；若有提供
    if rel_updates:
        participants_all = list(dict.fromkeys(st.session_state.locked_participants + [protagonist_name()]))
        # 只接受 src==说话者 的更新
        rel_updates = [u for u in rel_updates if u.get("src") == sp["name"] and u.get("dst") in participants_all]
        if rel_updates:
            apply_relationship_trigger_updates(rel_updates, participants_all)

    persist_everything(sp["name"], line, raw_reply_for_trigger=raw)
    return "npc"

# ===================== 触发按钮逻辑 =====================
if reroll_btn:
    one_step_generate(skip_protagonist=True, exclude_name=protagonist_name())
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
    st.markdown(f"### 轮到你发言（作为：**{display_name_ui(protagonist_name())}**）")
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
            current_worldview(), store.get_role_recentbuffer(protagonist_name()), topic_hint
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
            protagonist_name(), st.session_state.chat_history, all_names_in_scene()
        )
        new_sugg_raw = suggest_for_protagonist(
            protagonist_persona=protagonist_persona,
            worldview=current_worldview(),
            rb_history=store.get_role_recentbuffer(protagonist_name()),
            topic_hint=topic_hint,
            mem_lines=mem_lines_me
        )
        clean_sugg = strip_action_blocks(strip_memory_blocks(strip_relationship_blocks(new_sugg_raw)))
        st.session_state.player_suggested_line = sanitize_line(clean_sugg, protagonist_name(), all_names_in_scene())
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
        cleaned = strip_action_blocks(strip_memory_blocks(strip_relationship_blocks(typed_raw)))
        line = sanitize_line(cleaned, protagonist_name(), all_names_in_scene()).strip()
        trigger_blob = (st.session_state.get("player_memory_trigger_raw", "") or "") + "\n" + (typed_raw or "")
        if not line:
            audience = list(dict.fromkeys(st.session_state.locked_participants + [protagonist_name()]))
            handle_memory_triggers_from_reply(trigger_blob, audience, protagonist_name(), st.session_state.conv_id)
            did_act = handle_action_triggers_from_reply(trigger_blob, protagonist_name(), current_worldview(), st.session_state.conv_id)
            st.session_state.turns_since_last_act = 0 if did_act else st.session_state.turns_since_last_act + 1
        else:
            # 若玩家输入中含 [[REL]]，先应用主角→他人的关系更新
            rel_updates = parse_relationship_updates(typed_raw)
            if rel_updates:
                participants_all = list(dict.fromkeys(st.session_state.locked_participants + [protagonist_name()]))
                rel_updates = [u for u in rel_updates if u.get("src") == protagonist_name() and u.get("dst") in participants_all]
                if rel_updates:
                    apply_relationship_trigger_updates(rel_updates, participants_all)
            persist_everything(protagonist_name(), line, raw_reply_for_trigger=trigger_blob)
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
# streamlit run d:/NewProjects/LowCostChattingBot/npc-graph-chat/6.0_multi_character_chatbot_ChatTogether_Director.py
