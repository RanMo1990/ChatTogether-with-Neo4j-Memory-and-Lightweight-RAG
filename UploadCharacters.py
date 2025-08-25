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

# ========= 预设数据（现实世界 & 更强冲突） =========
DEFAULT_WORLD = "当代大城市·城中村与写字楼并存，外卖、保安、网红店与城管交织的现实街区。"
NARRATOR_NAME = "旁白"
NARRATOR_PERSONA = "场景叙事者，描述环境、动作过程与结果，掌握全局信息但不代替角色发言。"
DIRECTOR_NAME = "导演"
DIRECTOR_PERSONA = "戏剧化指导者，关注节奏与冲突设计，偏好强风格化表达但要求落地可信。"

PRESETS = [
    {"name": "健身教练", "persona": "天天健身的健身教练，热衷力量训练与科学饮食，自律外向，乐于指导他人。",
     "worldview": "强身与自律是通往自由的路；科学训练与规律作息胜过一切捷径。",
     "desire": 0.5},
    {"name": "小偷",   "persona": "手脚快、圆滑，嘴甜但心虚，擅长推诿与观察细节。",
     "worldview": "城市是猎场，机会留给手快的人；规矩是强者的安慰，活下去才是第一原则。",
     "desire": 0.55},
    {"name": "警察",   "persona": "冷静务实，重证据与程序，讨厌虚言，善于盘问。",
     "worldview": "秩序让普通人免于恐惧；证据与程序是底线，情绪不能代替事实。",
     "desire": 0.35},
    {"name": "保安",   "persona": "强势护规矩，对外卖、电瓶敏感；逞强但也怕担责。",
     "worldview": "规章先于人情，安全大于方便；出了事没人替你扛。",
     "desire": 0.4},
    {"name": "外卖员", "persona": "赶时间、话直心善，嘴上抱怨手里不停，路况很熟。",
     "worldview": "时间就是钱，路线与效率是一切；谁耽误我，谁就得让路。",
     "desire": 0.6},
    {"name": "小仙女", "persona": "精致社交达人，表面软萌，实际很会拿捏人心。",
     "worldview": "人设即资产，情绪是工具；关系要经营，风评有时比真相更重要。",
     "desire": 0.45},
    {"name": "舔狗",   "persona": "讨好型人格，盲目偏向小仙女，常与他人起摩擦。",
     "worldview": "喜欢就要表达，退一步未必海阔天空；只要她开心，我就值了。",
     "desire": 0.5},
    {"name": "老板娘", "persona": "网红奶茶店老板娘，会营销会打听，精明且现实。",
     "worldview": "流量能变现才算本事；成本收益要算清，人情要为生意服务。",
     "desire": 0.5},
    # 新增
    {"name": "男神",   "persona": "流量艺人，控场强，公关敏感，表面体面，私下谨慎算计。",
     "worldview": "形象是护城河，风险要前置管理；体面很值钱，但不能被情绪绑架。",
     "desire": 0.4},
    {"name": "女神",   "persona": "清冷系人气UP主，专业自律，擅借势，嘴甜刀快。",
     "worldview": "专业与数据说话；锋利不是对立，而是边界与效率。",
     "desire": 0.45},
    # 旁白（特殊，通常不参与权重选择，仅用于旁白生成）
    {"name": NARRATOR_NAME, "persona": NARRATOR_PERSONA, "worldview": DEFAULT_WORLD, "desire": 0.0},
    # 导演（特殊：带 style/method 属性，拥有全量可见性）
    {"name": DIRECTOR_NAME,
     "persona": DIRECTOR_PERSONA,
     "worldview": DEFAULT_WORLD,
     "style": "光怪陆离",
     "method": "让剧情非常狗血、行为夸张但需有现实落点",
     "desire": 0.0},
]

PERSONA_MEMORIES: Dict[str, List[str]] = {
    "健身教练": ["健身不是速成，是日复一日的坚持。","动作标准最重要，重量其次。"],
    "小偷":   ["我只是拿点不要紧的，大家都不容易。","盯人盯手，一慌就露馅。"],
    "警察":   ["程序正义同样重要。","证据在手，话才算数。"],
    "保安":   ["规定就是规定，电瓶车不能进这条线。","出事谁负责？先把规矩立住。"],
    "外卖员": ["超时扣钱，比谁都急。","楼里绕一圈，路线要背熟。"],
    "小仙女": ["人设要稳，情绪要拿捏。","镜头前后，是两场不同的戏。"],
    "舔狗":   ["她说什么都对，我只要她开心。","眼里只有她，别的都无所谓。"],
    "老板娘": ["八卦是免费的流量。","做生意看人更看局势。"],
    # 新增
    "男神":   ["镜头里我必须完美，但后台要把风险掐死。","合作可以，但要算清楚回报。"],
    "女神":   ["人气不是运气，是运营。","该锋利的时候，就别软。"],
    # 旁白通常无“曾说”类自述，保留为空
}

SHARED_INTEL = [
    {"content": "【传闻】网红奶茶店后巷经常丢外卖，监控总是‘刚好坏了’。", "audience": ["外卖员", "老板娘", "保安"]},
    {"content": "【线索】地铁口有人专盯下班族的背包，作案时间多在晚高峰。", "audience": ["警察", "保安"]},
    {"content": "【八卦】小仙女最近直播带货翻车，欠了供货商一笔尾款。",   "audience": ["老板娘", "小仙女"]},
    {"content": "【纠纷】业委会计划严禁电瓶上楼，骑手群体强烈反对。",       "audience": ["外卖员", "保安", "老板娘"]},
    {"content": "【流言】舔狗疑似帮小仙女挡过一次‘私联粉丝’的麻烦。",     "audience": ["舔狗", "小仙女"]},
    # 新增
    {"content": "【娱乐】男神新代言被指抄袭，引发品牌公关危机。",         "audience": ["男神", "老板娘", "小仙女"]},
    {"content": "【饭圈】女神与小仙女粉丝互踩，线下同城应援起冲突。",       "audience": ["女神", "小仙女", "舔狗"]},
]

# 旁白私有记忆（其他角色不知晓，作为基础场景素材）
NARRATOR_PRIVATE_MEMS = [
    "黄昏时分，城中村的巷子里潮气未散，霓虹招牌刚刚亮起。",
    "奶茶店后巷堆着几袋未清的纸箱，电瓶车的刹车印断断续续。",
    "写字楼门口的保安岗亭旁，失物招领箱里塞满了雨伞与水杯。",
    "地铁站A口的台阶上贴着新告示：‘注意保管随身物品’。",
]

# 初始人物关系（RELATES_TO）：rel/att/notes 都是列表
REL_SEEDS = [
    # 健身教练 ↔ 老板娘（常来买蛋白奶昔）
    {"a": "健身教练", "b": "老板娘", "rel": ["顾客","常客"],       "att": ["友好","专业"],   "notes": ["常来买蛋白奶昔"]},
    {"a": "老板娘",   "b": "健身教练", "rel": ["顾客"],           "att": ["热情","功利"],   "notes": ["试图让其到店打卡宣传"]},

    # 健身教练 ↔ 保安（晨练同路人）
    {"a": "健身教练", "b": "保安",   "rel": ["熟面孔","晨练同路人"], "att": ["尊重","中立"],   "notes": ["晨跑时偶尔点头打招呼"]},
    {"a": "保安",     "b": "健身教练", "rel": ["住户","健身教练"], "att": ["友好","尊重"],   "notes": ["请教过如何练背不伤腰"]},

    # 小偷 vs 警察/保安
    {"a": "小偷", "b": "警察", "rel": ["执法者","天敌"],                 "att": ["畏惧","戒备","敌意"], "notes": ["多次在地铁口被盘问"]},
    {"a": "警察", "b": "小偷", "rel": ["嫌疑人","抓捕目标","执法对象"],   "att": ["警惕","怀疑","冷淡"], "notes": ["背包盗窃案关联人"]},

    {"a": "小偷", "b": "保安", "rel": ["守门者","障碍"],                 "att": ["轻蔑","戒备"],       "notes": ["商场里被驱赶"]},
    {"a": "保安", "b": "小偷", "rel": ["可疑人员"],                     "att": ["厌恶","警惕"],       "notes": ["监控多次捕捉到徘徊"]},

    # 外卖员 vs 保安/老板娘
    {"a": "外卖员", "b": "保安",   "rel": ["门岗","执勤人员","规则方"],     "att": ["不满","焦躁"],       "notes": ["电瓶与进出限制冲突"]},
    {"a": "保安",   "b": "外卖员", "rel": ["临时出入者","配送员"],         "att": ["强硬","不耐烦"],     "notes": ["按规定办事"]},

    {"a": "外卖员", "b": "老板娘", "rel": ["商家"],                       "att": ["合作","抱怨"],       "notes": ["催单与差评纠纷"]},
    {"a": "老板娘", "b": "外卖员", "rel": ["骑手"],                       "att": ["现实","功利"],       "notes": ["高峰优先派单给熟人"]},

    # 小仙女 & 舔狗
    {"a": "舔狗",   "b": "小仙女", "rel": ["偶像","女神","心仪对象"],       "att": ["喜欢","讨好","依赖"], "notes": ["多次帮忙挡黑"]},
    {"a": "小仙女", "b": "舔狗",   "rel": ["粉丝","工具人"],               "att": ["利用","冷淡","掌控"], "notes": ["必要时才回应"]},

    # 老板娘 vs 小仙女（营销合作且拉扯）
    {"a": "老板娘", "b": "小仙女", "rel": ["合作方","债务人"],             "att": ["审慎","算计"],       "notes": ["带货翻车后欠款未清"]},
    {"a": "小仙女", "b": "老板娘", "rel": ["商家","欠款方"],               "att": ["敷衍","傲慢"],       "notes": ["直播道歉态度暧昧"]},

    # 警察 vs 保安（协作但不对等）
    {"a": "警察", "b": "保安", "rel": ["协作","指导"],                     "att": ["专业","中立"],       "notes": ["案发时要求配合取证"]},
    {"a": "保安", "b": "警察", "rel": ["配合","依赖"],                     "att": ["尊重","服从"],       "notes": ["巡逻遇事先报警"]},

    # —— 新增：围绕男神/女神的冲突与拉扯 —— #
    # 舔狗 ↔ 女神
    {"a": "舔狗", "b": "女神", "rel": ["偶像","心仪对象"],                 "att": ["喜欢","新鲜感"],     "notes": ["被吸引开始转粉"]},
    {"a": "女神", "b": "舔狗", "rel": ["路人粉丝"],                       "att": ["疏离","礼貌"],       "notes": ["合影被拒后仍追评"]},

    # 小仙女 ↔ 女神（竞品对手）
    {"a": "小仙女", "b": "女神", "rel": ["竞争者","对手"],                 "att": ["嫉妒","防备"],       "notes": ["粉丝重叠度高"]},
    {"a": "女神",   "b": "小仙女", "rel": ["同行","竞争者"],               "att": ["专业","克制"],       "notes": ["观察其翻车影响"]},

    # 男神 ↔ 小仙女（暧昧/博弈）
    {"a": "男神",   "b": "小仙女", "rel": ["绯闻对象","合作方"],           "att": ["谨慎","拉距"],       "notes": ["公关要求保持距离"]},
    {"a": "小仙女", "b": "男神",   "rel": ["绯闻对象","流量艺人"],         "att": ["利用","暧昧"],       "notes": ["偶尔互动蹭热"]},

    # 男神 ↔ 女神（同梯度竞争/可能联动）
    {"a": "男神", "b": "女神", "rel": ["同行","潜在联动方"],               "att": ["评估","友好"],       "notes": ["考虑合拍短片避雷"]},
    {"a": "女神", "b": "男神", "rel": ["同行","合作方候选"],               "att": ["务实","审慎"],       "notes": ["看品牌风评再定"]},

    # 老板娘 ↔ 男神/女神（商业合作）
    {"a": "老板娘", "b": "男神", "rel": ["流量艺人","合作方候选"],         "att": ["算计","热情"],       "notes": ["想请到店打卡"]},
    {"a": "男神",   "b": "老板娘", "rel": ["商家"],                         "att": ["观望","友好"],       "notes": ["代言危机期谨慎"]},

    {"a": "老板娘", "b": "女神", "rel": ["合作方候选"],                     "att": ["期待","精算"],       "notes": ["联名款议价中"]},
    {"a": "女神",   "b": "老板娘", "rel": ["商家"],                         "att": ["专业","挑剔"],       "notes": ["关注复购与客单"]},
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
    c.style      = coalesce($style, c.style),
    c.method     = coalesce($method, c.method),
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

# 覆盖 rel/att，notes 采用“追加并去重”（不依赖 APOC）
UPSERT_REL = """
MERGE (a:Character {name:$a})
MERGE (b:Character {name:$b})
MERGE (a)-[r:RELATES_TO]->(b)
ON CREATE SET r.rel = [], r.att = [], r.notes = [], r.created_at = datetime()
SET r.rel = $rel,
    r.att = $att,
    r.updated_at = datetime()
WITH r, coalesce(r.notes, []) AS old_notes, $notes AS new_notes
WITH r, old_notes + new_notes AS merged
SET r.notes = reduce(s=[], x IN merged | CASE WHEN x IN s THEN s ELSE s + x END)
"""

# 导演对角色的指导关系（GUIDES），包含两个文本属性：scene_focus（外在状况/场景关注主题）与 state_intent（角色当前状态/想做的事）
UPSERT_GUIDES = """
MERGE (d:Character {name:$director})
MERGE (c:Character {name:$target})
MERGE (d)-[g:GUIDES]->(c)
ON CREATE SET g.scene_focus = coalesce($scene_focus, ""),
              g.state_intent = coalesce($state_intent, ""),
              g.created_at = datetime()
SET g.scene_focus = coalesce($scene_focus, g.scene_focus),
    g.state_intent = coalesce($state_intent, g.state_intent),
    g.updated_at = datetime()
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

    # 2) 写入预设角色 & RecentBuffer（含 旁白）
        for p in PRESETS:
            params = {
                "name": p["name"],
                "persona": p.get("persona", ""),
                "worldview": p.get("worldview", ""),
                "desire": float(p.get("desire", 0.3)),
                "style": p.get("style"),
                "method": p.get("method"),
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

        # 3.5) 写入 旁白的私有记忆
        for m in NARRATOR_PRIVATE_MEMS:
            content = (m or "").strip().strip('“”"\'')
            vec = embed_text(content)
            if vec and first_vec_dims is None:
                first_vec_dims = len(vec)
                ensure_vector_index(session, first_vec_dims)
            session.run(UPSERT_MEMORY, content=content, embedding=vec)
            session.run(LINK_AWARE, names=[NARRATOR_NAME], content=content)
            print(f"  [nar-mem] {NARRATOR_NAME} <- {content[:28]}...")

        # 4) 写入“情报/传闻/档案”类（可多人共享）
        for item in SHARED_INTEL:
            content = (item["content"] or "").strip().strip('“”"\'')
            vec = embed_text(content)
            if vec and first_vec_dims is None:
                first_vec_dims = len(vec)
                ensure_vector_index(session, first_vec_dims)
            session.run(UPSERT_MEMORY, content=content, embedding=vec)
            audience = [n for n in item.get("audience", []) if any(n == p["name"] for p in PRESETS)]
            if audience:
                session.run(LINK_AWARE, names=audience, content=content)
            print(f"  [intel] {','.join(audience)} <- {content[:28]}...")

        # 4.5) 旁白拥有全量 Memory（含之后写入的共享情报）
        session.run(
            """
            MATCH (n:Character {name:$n})
            WITH n
            MATCH (m:Memory)
            MERGE (n)-[:AWARE_OF]->(m)
            """, {"n": NARRATOR_NAME}
        )
        print(f"  [link] {NARRATOR_NAME} 关联所有 Memory")

        # 4.6) 导演拥有全量 Memory（便于指导旁白与角色）
        session.run(
            """
            MATCH (d:Character {name:$d})
            WITH d
            MATCH (m:Memory)
            MERGE (d)-[:AWARE_OF]->(m)
            """, {"d": DIRECTOR_NAME}
        )
        print(f"  [link] {DIRECTOR_NAME} 关联所有 Memory")

        # 4.7) 导演→所有角色（含旁白，不含导演自身）的 GUIDES 关系初始化
        res_chars = session.run("MATCH (c:Character) RETURN c.name AS name").data()
        all_names = [r["name"] for r in res_chars]
        for nm in all_names:
            if nm == DIRECTOR_NAME:
                continue
            session.run(UPSERT_GUIDES, director=DIRECTOR_NAME, target=nm, scene_focus="", state_intent="")
        print(f"  [guides] {DIRECTOR_NAME} -> 全角色 已初始化 GUIDES 关系")

        # 5) 写入初始人物关系（RELATES_TO）
        for r in REL_SEEDS:
            session.run(UPSERT_REL, **{
                "a": r["a"], "b": r["b"],
                "rel": list(dict.fromkeys(r.get("rel", []))),    # 去重
                "att": list(dict.fromkeys(r.get("att", []))),
                "notes": list(dict.fromkeys(r.get("notes", [])))
            })
            print(f"  [rel] {r['a']} -> {r['b']} rel={r['rel']} att={r['att']} notes={r['notes']}")

        # 6) 校验读取
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
        edge_cnt = session.run("MATCH ()-[r:RELATES_TO]->() RETURN count(r) AS n").single()["n"]
        print(f"\nMemory 节点：{mem_cnt} 条，AWARE_OF 关系：{rel_cnt} 条，RELATES_TO 关系：{edge_cnt} 条。")

    driver.close()
    print("\n完成。你可以在应用里直接使用这些现实角色与初始关系。")

if __name__ == "__main__":
    main()
