# -*- coding: utf-8 -*-
import os
import re
import json
import random
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

# UI 与记忆窗口
CONV_UI_MAX_TURNS = 100             # 前端展示最多多少条
RB_MAX_TURNS_PER_ROLE = 30          # 每角色 RecentBuffer 的滚动窗口大小
PROTAGONIST_NAME = "主角（你）"
PROTAGONIST_DEFAULT_DESIRE = 0.5

# ===================== 小工具 =====================
def sanitize_line(raw: str, speaker_name: str, all_names):
    """去名字前缀/引号/长省略号，仅保留台词本身。"""
    if not raw:
        return "……"
    text = raw.strip()
    text = text.strip('“”"\'').strip()
    if all_names:
        name_pat = "|".join(re.escape(n) for n in sorted(set(all_names), key=len, reverse=True))
        for _ in range(2):
            text = re.sub(rf'^(?:{name_pat})\s*[:：]\s*', "", text)
    text = re.sub(r'[。！？!?…]{3,}', '…', text)
    return text or "……"

def clamp_tail(lst, n: int):
    return lst[-n:] if len(lst) > n else lst

def weighted_next_speaker(weights: dict, exclude: str | None = None) -> str:
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

def make_conv_id(participants: list[str], title: str) -> str:
    """统一生成 conv_id（包含主角，参与者去重排序，标题规范化）"""
    names = sorted(set(participants + [PROTAGONIST_NAME]))
    title_norm = re.sub(r"\s+", " ", (title or "").strip()) or "untitled"
    return f"conv::{title_norm}::{'|'.join(names)}"

# ===================== Neo4j 存储层（Conversation 单节点仅存 history_json） =====================
def _bolt_fallback(uri: str) -> str:
    return (uri.replace("neo4j+ssc://", "bolt+ssc://")
               .replace("neo4j+s://", "bolt+s://")
               .replace("neo4j://", "bolt://", 1))

class Neo4jStore:
    """
    模型：
      (:Character {name, persona, worldview, desire})
        -[:HAS_RECENT_BUFFER]-> (:RecentBuffer {id, lines_json})   # 每角色一个，存该角色参与片段（含所有人台词）的最近窗口

      (:Conversation {id, title, history_json, created_at, updated_at})
        # 单节点存整个会话：history_json = [{role, content}, ...]
      (:Character)-[:PARTICIPATED_IN {role:"npc"|"protagonist"}]->(:Conversation)
        # 标明谁参与了该会话
    """
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
            return [dict(r) for r in res]

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

    def get_role_recentbuffer(self, name: str) -> list[dict]:
        with self.driver.session(database=NEO4J_DATABASE) as s:
            rec = s.run("""
                MATCH (:Character {name:$name})-[:HAS_RECENT_BUFFER]->(rb:RecentBuffer {id:$rb})
                RETURN rb.lines_json AS j
            """, name=name, rb=self._rb_id(name)).single()
        if rec and rec["j"]:
            try: return json.loads(rec["j"])
            except Exception: return []
        return []

    def append_turn_to_roles_recent(self, role_names: list[str], turn: dict, keep: int = RB_MAX_TURNS_PER_ROLE):
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
                    try: lines = json.loads(rec["j"]) or []
                    except Exception: lines = []
                lines.append(turn)
                lines = clamp_tail(lines, keep)
                j = json.dumps(lines, ensure_ascii=False)
                s.run("""
                    MATCH (:Character {name:$name})-[:HAS_RECENT_BUFFER]->(rb:RecentBuffer {id:$rb})
                    SET rb.lines_json=$j, rb.updated=datetime()
                """, name=n, rb=self._rb_id(n), j=j)

    # ----- Conversation（单节点仅存 history_json） + 参与关系 -----
    def ensure_conversation_with_participants(self, conv_id: str, title: str, participants: list[dict]):
        """
        participants: [{name: '小明', role_tag: 'npc'|'protagonist'}, ...]
        建会话节点，并幂等建立参与关系 (:Character)-[:PARTICIPATED_IN {role:role_tag}]->(:Conversation)
        """
        with self.driver.session(database=NEO4J_DATABASE) as s:
            s.run("""
                MERGE (conv:Conversation {id:$cid})
                  ON CREATE SET conv.created_at = datetime()
                SET conv.title = coalesce($title, conv.title),
                    conv.updated_at = datetime()
            """, cid=conv_id, title=title or "")
            # 建参与边
            s.run("""
                UNWIND $ps AS p
                MERGE (c:Character {name:p.name})
                MERGE (conv:Conversation {id:$cid})
                MERGE (c)-[r:PARTICIPATED_IN]->(conv)
                SET r.role = p.role_tag
            """, cid=conv_id, ps=participants)

    def set_conversation_history(self, conv_id: str, turns: list[dict]):
        """把整段对话存回 Conversation.history_json（只有这一份）。"""
        j = json.dumps(turns, ensure_ascii=False)
        with self.driver.session(database=NEO4J_DATABASE) as s:
            s.run("""
                MERGE (conv:Conversation {id:$cid})
                SET conv.history_json = $j,
                    conv.updated_at = datetime()
            """, cid=conv_id, j=j)

    def get_conversation_history(self, conv_id: str) -> list[dict]:
        with self.driver.session(database=NEO4J_DATABASE) as s:
            rec = s.run("MATCH (conv:Conversation {id:$cid}) RETURN conv.history_json AS j", cid=conv_id).single()
        if rec and rec["j"]:
            try: return json.loads(rec["j"])
            except Exception: return []
        return []

# ===================== Gemini 调用 =====================
client = genai.Client(vertexai=True, project=GEMINI_PROJECT, location=GEMINI_LOCATION)

def gen_with_role_recentbuffer(
    speaker_name: str,
    speaker_persona: str,
    worldview: str,
    rb_history: list[dict],
    topic_hint: str = ""
) -> str:
    ctx = "\n".join([f"{t.get('role','')}: {t.get('content','')}" for t in rb_history])
    last_role = rb_history[-1]["role"] if rb_history else ""
    last_text = rb_history[-1]["content"] if rb_history else ""
    prompt = (
        f"世界观：{worldview or '（未提供）'}\n"
        f"你的身份：{speaker_name}（人设：{speaker_persona or '（未提供）'}）\n"
        f"话题提示：{topic_hint or '（无）'}\n\n"
        f"== 你参与的最近对话片段（含所有人台词） ==\n{ctx or '（暂无）'}\n"
        f"最近一条：{last_role or '（无）'}：{last_text or '（无）'}\n\n"
        f"只输出你的台词（1-2句），不要名字前缀，不要加引号；优先自然回应最近一条，也可按人设转移但要自洽。"
    )
    try:
        resp = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        return (getattr(resp, "text", "") or "").strip()
    except Exception as e:
        return f"(模型调用出错：{e})"

def suggest_for_protagonist(
    protagonist_persona: str,
    worldview: str,
    rb_history: list[dict],
    topic_hint: str = ""
) -> str:
    ctx = "\n".join([f"{t.get('role','')}: {t.get('content','')}" for t in rb_history])
    last_role = rb_history[-1]["role"] if rb_history else ""
    last_text = rb_history[-1]["content"] if rb_history else ""
    prompt = (
        f"世界观：{worldview or '（未提供）'}\n"
        f"玩家角色：主角（你）（人设：{protagonist_persona or '（未提供）'}）\n"
        f"话题提示：{topic_hint or '（无）'}\n\n"
        f"== 你参与的最近片段 ==\n{ctx or '（暂无）'}\n"
        f"最近一条：{last_role or '（无）'}：{last_text or '（无）'}\n\n"
        f"给出一条建议台词（第一人称“我”，1-2句；不要加引号/名字前缀）。"
    )
    try:
        resp = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        return (getattr(resp, "text", "") or "").strip()
    except Exception as e:
        return f"(建议生成出错：{e})"

# ===================== Streamlit UI =====================
st.set_page_config(page_title="多角色聊天 · Conversation 单节点 history_json + 参与关系", page_icon="🤖", layout="centered")
st.title("🤖 多角色多人聊天（首次发言才创建会话，单节点 history_json + PARTICIPATED_IN）")
st.caption(f"BUILD: 2025-08-17 · CONV_UI_MAX={CONV_UI_MAX_TURNS} · RB_MAX={RB_MAX_TURNS_PER_ROLE}")

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
# 显示滑条：未锁定用当前选择；锁定后用 locked_participants
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
    """
    仅在第一次真正写入发言时调用。
    作用：创建 Conversation 节点和 PARTICIPATED_IN 关系，并锁定 conv_id 与参与者列表。
    """
    if st.session_state.conv_locked:
        return
    if not participants_selected:
        st.warning("请先选择至少一个 NPC 才能开始对话。")
        return

    cid = make_conv_id(participants_selected, conv_title)
    st.session_state.conv_id = cid
    st.session_state.conv_locked = True
    st.session_state.locked_participants = list(participants_selected)

    # 确保主角 + 参与者都是 Character（以及各自 RecentBuffer）
    store.ensure_character(PROTAGONIST_NAME, protagonist_persona, protagonist_worldview or "", PROTAGONIST_DEFAULT_DESIRE)
    for n in st.session_state.locked_participants:
        store.ensure_character(n)

    # 真正创建会话节点 + 参与关系（一次性）
    participants_payload = (
        [{"name": n, "role_tag": "npc"} for n in st.session_state.locked_participants] +
        [{"name": PROTAGONIST_NAME, "role_tag": "protagonist"}]
    )
    store.ensure_conversation_with_participants(cid, conv_title, participants_payload)

    # 历史：新会话一般为空；若复用旧 id 会拉回历史
    st.session_state.chat_history = store.get_conversation_history(cid) or []

def reset_conversation():
    """另起新会话：解锁并清空本地状态，不碰各角色 RB"""
    for k in ["conv_locked", "conv_id", "history_loaded_for", "chat_history",
              "awaiting_player_input", "player_suggested_line", "player_line_input", "locked_participants"]:
        st.session_state.pop(k, None)
    st.session_state.conv_locked = False
    st.session_state.locked_participants = []

# 手动按钮（可选）
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

# ---- 读取会话历史（单节点 history_json；仅锁定后读取）----
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

def persist_everything(speaker: str, text: str):
    """把新 turn 写入：首次发言时创建会话；随后写 Conversation.history_json；并更新每位参与者的 RecentBuffer。"""
    # 1) 首次发言时创建会话节点 + 参与关系，并锁定 participants
    create_conv_if_needed()
    if not st.session_state.conv_locked:
        return  # 没选 NPC 等情况

    turn = {"role": speaker, "content": text}

    # 2) UI 内存
    st.session_state.chat_history.append(turn)

    # 3) Conversation（单节点，只存 history_json）
    store.set_conversation_history(st.session_state.conv_id, st.session_state.chat_history)

    # 4) 各自 RecentBuffer（只更新“锁定的参与者 + 主角”）
    roles_to_update = list(dict.fromkeys(st.session_state.locked_participants + [PROTAGONIST_NAME]))
    store.append_turn_to_roles_recent(roles_to_update, turn, keep=RB_MAX_TURNS_PER_ROLE)

# ===================== 生成逻辑 =====================
client = genai.Client(vertexai=True, project=GEMINI_PROJECT, location=GEMINI_LOCATION)

def one_step_generate(skip_protagonist: bool = False, exclude_name: str | None = None):
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

    # 读取“当前说话者自己”的 RecentBuffer 作为上文
    rb_history = store.get_role_recentbuffer(sp["name"])

    # 主角：提供可编辑建议
    if sp["name"] == PROTAGONIST_NAME:
        sugg_raw = suggest_for_protagonist(
            protagonist_persona=protagonist_persona,
            worldview=current_worldview(),
            rb_history=rb_history,
            topic_hint=topic_hint
        )
        st.session_state.awaiting_player_input = True
        st.session_state.player_suggested_line = sanitize_line(sugg_raw, PROTAGONIST_NAME, all_names_in_scene())
        st.session_state.player_line_input = st.session_state.player_suggested_line
        return "player"

    # NPC：用 TA 自己的 RB 生成
    raw = gen_with_role_recentbuffer(
        speaker_name=sp["name"],
        speaker_persona=sp.get("persona", ""),
        worldview=current_worldview(),
        rb_history=rb_history,
        topic_hint=topic_hint
    )
    line = sanitize_line(raw, sp["name"], all_names_in_scene())
    persist_everything(sp["name"], line)
    return "npc"

# ---- 触发 ----
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
    # 清空当前会话历史（同时覆盖 Conversation 单节点），不影响 RB
    st.session_state.chat_history = []
    if st.session_state.conv_locked and st.session_state.conv_id:
        store.set_conversation_history(st.session_state.conv_id, [])
    for k in ["awaiting_player_input", "player_suggested_line", "player_line_input"]:
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
        new_sugg = suggest_for_protagonist(
            protagonist_persona=protagonist_persona,
            worldview=current_worldview(),
            rb_history=store.get_role_recentbuffer(PROTAGONIST_NAME),
            topic_hint=topic_hint
        )
        st.session_state.player_suggested_line = sanitize_line(new_sugg, PROTAGONIST_NAME, all_names_in_scene())
        st.session_state.player_line_input = st.session_state.player_suggested_line
        st.rerun()

    if skip_player:
        st.session_state.awaiting_player_input = False
        st.session_state.pop("player_suggested_line", None)
        st.session_state.pop("player_line_input", None)
        one_step_generate(skip_protagonist=True)
        st.rerun()

    if send_player:
        line = sanitize_line(st.session_state.get("player_line_input", "").strip(), PROTAGONIST_NAME, all_names_in_scene())
        if not line:
            st.warning("台词不能为空。可以点“换一个建议”或自己输入。")
        else:
            persist_everything(PROTAGONIST_NAME, line)
            st.session_state.awaiting_player_input = False
            st.session_state.pop("player_suggested_line", None)
            st.session_state.pop("player_line_input", None)
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






#streamlit run d:/NewProjects/LowCostChattingBot/npc-graph-chat/3.0_multi_character_chatbot_ChatTogether_WithRag.py






#streamlit run d:/NewProjects/LowCostChattingBot/npc-graph-chat/multi_character_chatbot_ChatTogether.py