import os
import json
import streamlit as st
from neo4j import GraphDatabase
from google import genai
from dotenv import load_dotenv

# ========== 环境与配置 ==========
load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

GEMINI_PROJECT = os.getenv("GEMINI_PROJECT_ID")
GEMINI_LOCATION = os.getenv("GEMINI_LOCATION", "us-central1")
GEMINI_MODEL = os.getenv("GEMINI_MODEL_STD", "gemini-2.0-flash-lite")

# ========== Neo4j 工具 ==========
class Neo4jRecentBuffer:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        self._init_schema()

    def _init_schema(self):
        with self.driver.session() as session:
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Character) REQUIRE c.name IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (rb:RecentBuffer) REQUIRE rb.id IS UNIQUE")

    def get_characters(self):
        with self.driver.session() as session:
            result = session.run("""
                MATCH (c:Character)
                RETURN c.name AS name, c.persona AS persona, c.worldview AS worldview
                ORDER BY c.name
            """)
            return [dict(record) for record in result]

    def add_character(self, name, persona, worldview):
        """创建角色并确保唯一 RecentBuffer 节点与之相连"""
        rb_id = self._rb_id(name)
        with self.driver.session() as session:
            session.run(
                """
                MERGE (c:Character {name: $name})
                  ON CREATE SET c.created_at = datetime()
                SET  c.persona    = $persona,
                     c.worldview  = $worldview,
                     c.updated_at = datetime()
                MERGE (rb:RecentBuffer {id: $rb_id})
                  ON CREATE SET rb.history_json = "[]", rb.created_at = datetime()
                MERGE (c)-[:HAS_RECENT_BUFFER]->(rb)
                """,
                name=name, persona=persona, worldview=worldview, rb_id=rb_id
            )

    def _rb_id(self, char_name: str) -> str:
        return f"{char_name}_recent_buffer"

    def get_recent_buffer(self, char_name):
        """
        只读取与角色相连的唯一 RecentBuffer 节点。
        兼容：如果历史上存在 Memory 节点，则作为 fallback 读取其 history_json。
        """
        with self.driver.session() as session:
            rec = session.run(
                """
                MATCH (:Character {name: $name})-[:HAS_RECENT_BUFFER]->(rb:RecentBuffer {id: $rb_id})
                RETURN rb.history_json AS history_json
                """,
                name=char_name, rb_id=self._rb_id(char_name)
            ).single()

            if rec and rec["history_json"]:
                try:
                    return json.loads(rec["history_json"])
                except Exception:
                    return []

            # 兼容旧数据：如果没有 RecentBuffer，尝试读取 Memory（如果存在）
            fallback = session.run(
                """
                MATCH (:Character {name: $name})-[:HAS_MEMORY]->(m:Memory)
                RETURN m.history_json AS history_json
                ORDER BY m.updated DESC
                LIMIT 1
                """,
                name=char_name
            ).single()

        if fallback and fallback["history_json"]:
            try:
                return json.loads(fallback["history_json"])
            except Exception:
                return []
        return []

    def update_recent_buffer(self, char_name, history):
        """
        将 list[{"role": str, "content": str}, ...] 存为 JSON 字符串。
        只更新与角色相连的唯一 RecentBuffer 节点。
        """
        history_json = json.dumps(history, ensure_ascii=False)
        with self.driver.session() as session:
            session.run(
                """
                MERGE (c:Character {name: $name})
                MERGE (rb:RecentBuffer {id: $rb_id})
                  ON CREATE SET rb.history_json = "[]", rb.created_at = datetime()
                MERGE (c)-[:HAS_RECENT_BUFFER]->(rb)
                SET rb.history_json = $history_json,
                    rb.updated      = datetime(),
                    c.updated_at    = datetime()
                """,
                name=char_name,
                rb_id=self._rb_id(char_name),
                history_json=history_json
            )

# ========== Gemini 工具 ==========
client = genai.Client(vertexai=True, project=GEMINI_PROJECT, location=GEMINI_LOCATION)

def gemini_chat(role_prompt, worldview, memory, user_input):
    context = "\n".join([f"{turn.get('role','assistant')}: {turn.get('content','')}" for turn in memory])
    sys_prompt = (
        f"你是{role_prompt}。\n"
        f"世界观：{worldview}\n"
        f"以下是你和用户的最近对话：\n{context}\n"
        f"用户: {user_input}\n"
        f"请以角色身份简洁、自然地回复。"
    )
    try:
        response = client.models.generate_content(model=GEMINI_MODEL, contents=sys_prompt)
        if hasattr(response, "text") and response.text:
            return response.text.strip()
        return "..."
    except Exception as e:
        return f"(模型调用出错：{e})"

# ========== Streamlit UI ==========
st.set_page_config(page_title="多角色 Gemini 聊天机器人", page_icon="🤖", layout="centered")
st.title("🤖 多角色 Gemini 聊天机器人")

# 初始化Neo4j
memory_db = Neo4jRecentBuffer()

# 角色管理
if "characters" not in st.session_state:
    st.session_state.characters = memory_db.get_characters()
if "selected_char" not in st.session_state:
    st.session_state.selected_char = None

# 角色选择与添加
char_names = [c["name"] for c in st.session_state.characters]
selected = st.selectbox("选择角色", ["新建角色"] + char_names)

if selected == "新建角色":
    with st.form("add_char_form"):
        name = st.text_input("角色名")
        persona = st.text_area("人设（如：冷静的侦探、热情的厨师等）")
        worldview = st.text_area("世界观（如：赛博朋克城市、魔法世界等）")
        submitted = st.form_submit_button("添加角色")
        if submitted and name and persona:
            memory_db.add_character(name, persona, worldview)
            st.session_state.characters = memory_db.get_characters()
            st.session_state.selected_char = name
            st.success(f"角色 {name} 已添加！")
            st.rerun()
else:
    st.session_state.selected_char = selected

# 聊天界面
if st.session_state.selected_char:
    char = next(c for c in st.session_state.characters if c["name"] == st.session_state.selected_char)
    st.subheader(f"与 {char['name']} 聊天")
    st.markdown(f"**人设：** {char.get('persona','')}\n\n**世界观：** {char.get('worldview','')}")

    # 首次/切换角色时加载与之相连的唯一 RecentBuffer 节点
    if "chat_history" not in st.session_state or st.session_state.get("last_char") != char["name"]:
        st.session_state.chat_history = memory_db.get_recent_buffer(char["name"]) or []
        st.session_state.last_char = char["name"]

    # 展示历史
    for turn in st.session_state.chat_history:
        st.chat_message(turn.get("role", "assistant")).markdown(turn.get("content", ""))

    # 用户输入
    user_input = st.chat_input("请输入你的问题...")
    if user_input:
        # 追加一轮（仅保留最近6轮）
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.session_state.chat_history = st.session_state.chat_history[-6:]

        reply = gemini_chat(char.get("persona", ""), char.get("worldview", ""), st.session_state.chat_history, user_input)
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        st.session_state.chat_history = st.session_state.chat_history[-6:]

        # 写入该角色唯一的 RecentBuffer 节点
        memory_db.update_recent_buffer(char["name"], st.session_state.chat_history)
        st.rerun()

    if st.button("清空对话"):
        st.session_state.chat_history = []
        memory_db.update_recent_buffer(char["name"], [])
        st.rerun()
