"""
Neo4j聊天机器人主应用
基于Streamlit的Web界面
"""
import streamlit as st
import os
from config import config
from neo4j_manager import neo4j_manager

st.set_page_config(
    page_title="Neo4j聊天机器人",
    page_icon="🤖",
    layout="wide"
)

# 主标题
st.title("🤖 Neo4j智能聊天机器人")
st.markdown("---")

# 侧边栏 - 系统状态
with st.sidebar:
    st.header("🔧 系统状态")
    
    # Neo4j连接状态
    if neo4j_manager and neo4j_manager.driver:
        st.success("✅ Neo4j已连接")
        
        # 显示数据库信息
        try:
            result = neo4j_manager.run_query("CALL db.info()")
            if result:
                db_info = result[0]
                st.info(f"数据库: {db_info.get('name', 'default')}")
        except:
            pass
            
        # 显示节点统计
        try:
            result = neo4j_manager.run_query("MATCH (n) RETURN count(n) as node_count")
            if result:
                node_count = result[0]['node_count']
                st.metric("节点数量", node_count)
                
            result = neo4j_manager.run_query("MATCH ()-[r]->() RETURN count(r) as rel_count")
            if result:
                rel_count = result[0]['rel_count']
                st.metric("关系数量", rel_count)
        except Exception as e:
            st.warning(f"无法获取统计信息: {e}")
    else:
        st.error("❌ Neo4j未连接")
        st.markdown("请检查Neo4j服务是否运行")
    
    st.markdown("---")
    
    # 配置信息
    st.subheader("⚙️ 配置信息")
    st.text(f"URI: {config.NEO4J_URI}")
    st.text(f"用户: {config.NEO4J_USER}")
    st.text(f"嵌入模型: {config.SBERT_MODEL}")
    st.text(f"LLM: {config.LLM_PROVIDER}")

# 主界面 - 选项卡
tab1, tab2, tab3 = st.tabs(["💬 聊天", "🔍 数据查询", "📊 数据管理"])

with tab1:
    st.header("💬 智能对话")
    
    # 初始化会话状态
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # 显示聊天历史
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # 用户输入
    if prompt := st.chat_input("请输入您的问题..."):
        # 显示用户消息
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # 生成AI回复
        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                # 这里可以集成您的智能问答逻辑
                if "节点" in prompt or "数据" in prompt:
                    try:
                        result = neo4j_manager.run_query("MATCH (n) RETURN n LIMIT 5")
                        if result:
                            response = "数据库中的节点:\n"
                            for record in result:
                                node = record['n']
                                labels = list(node.labels)
                                properties = dict(node)
                                response += f"- {labels}: {properties}\n"
                        else:
                            response = "数据库中暂时没有节点数据。"
                    except Exception as e:
                        response = f"查询失败: {e}"
                elif "关系" in prompt:
                    try:
                        result = neo4j_manager.run_query("MATCH (a)-[r]->(b) RETURN a, r, b LIMIT 5")
                        if result:
                            response = "数据库中的关系:\n"
                            for record in result:
                                a = record['a']
                                r = record['r']
                                b = record['b']
                                response += f"- {dict(a)} -[{r.type}]-> {dict(b)}\n"
                        else:
                            response = "数据库中暂时没有关系数据。"
                    except Exception as e:
                        response = f"查询失败: {e}"
                else:
                    response = f"您说: {prompt}\n\n这是一个基础版本的聊天机器人。我可以帮您查询Neo4j数据库中的节点和关系信息。\n\n请尝试询问：\n- '有什么节点？'\n- '有什么关系？'\n- '数据库状态如何？'"
                
            st.markdown(response)
        
        # 保存AI回复
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # 清除对话
    if st.button("🗑️ 清除对话"):
        st.session_state.messages = []
        st.rerun()

with tab2:
    st.header("🔍 数据库查询")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📋 预设查询")
        
        if st.button("查看所有节点"):
            with st.spinner("查询中..."):
                try:
                    result = neo4j_manager.run_query("MATCH (n) RETURN n LIMIT 20")
                    if result:
                        st.success(f"找到 {len(result)} 个节点")
                        for i, record in enumerate(result):
                            node = record['n']
                            with st.expander(f"节点 {i+1}: {list(node.labels)}"):
                                st.json(dict(node))
                    else:
                        st.info("数据库中暂时没有节点")
                except Exception as e:
                    st.error(f"查询失败: {e}")
        
        if st.button("查看所有关系"):
            with st.spinner("查询中..."):
                try:
                    result = neo4j_manager.run_query("MATCH (a)-[r]->(b) RETURN a, r, b LIMIT 20")
                    if result:
                        st.success(f"找到 {len(result)} 个关系")
                        for i, record in enumerate(result):
                            a = record['a']
                            r = record['r']
                            b = record['b']
                            with st.expander(f"关系 {i+1}: {r.type}"):
                                st.write(f"起始节点: {dict(a)}")
                                st.write(f"关系类型: {r.type}")
                                st.write(f"目标节点: {dict(b)}")
                    else:
                        st.info("数据库中暂时没有关系")
                except Exception as e:
                    st.error(f"查询失败: {e}")
    
    with col2:
        st.subheader("🔧 自定义查询")
        
        cypher_query = st.text_area(
            "输入Cypher查询:",
            placeholder="例如: MATCH (n) RETURN n LIMIT 10",
            height=100
        )
        
        if st.button("执行查询"):
            if cypher_query.strip():
                with st.spinner("执行中..."):
                    try:
                        result = neo4j_manager.run_query(cypher_query)
                        if result:
                            st.success(f"查询成功，返回 {len(result)} 条记录")
                            
                            # 显示结果
                            for i, record in enumerate(result):
                                with st.expander(f"记录 {i+1}"):
                                    st.json(dict(record))
                        else:
                            st.info("查询没有返回结果")
                    except Exception as e:
                        st.error(f"查询执行失败: {e}")
            else:
                st.warning("请输入查询语句")

with tab3:
    st.header("📊 数据管理")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("➕ 添加测试数据")
        
        if st.button("创建示例人员数据"):
            with st.spinner("创建中..."):
                try:
                    # 创建示例人员节点
                    queries = [
                        "CREATE (p:Person {name: '张三', age: 30, job: '工程师', description: '资深软件工程师'})",
                        "CREATE (p:Person {name: '李四', age: 28, job: '设计师', description: 'UI/UX设计师'})",
                        "CREATE (p:Person {name: '王五', age: 35, job: '产品经理', description: '产品规划专家'})",
                        "CREATE (p:Person {name: '赵六', age: 32, job: '数据科学家', description: '数据分析专家'})"
                    ]
                    
                    for query in queries:
                        neo4j_manager.run_query(query)
                    
                    # 创建关系
                    relationship_queries = [
                        "MATCH (a:Person {name: '张三'}), (b:Person {name: '李四'}) CREATE (a)-[:WORKS_WITH]->(b)",
                        "MATCH (a:Person {name: '李四'}), (b:Person {name: '王五'}) CREATE (a)-[:REPORTS_TO]->(b)",
                        "MATCH (a:Person {name: '王五'}), (b:Person {name: '赵六'}) CREATE (a)-[:COLLABORATES_WITH]->(b)"
                    ]
                    
                    for query in relationship_queries:
                        neo4j_manager.run_query(query)
                    
                    st.success("✅ 示例数据创建成功！")
                    st.info("已创建4个人员节点和3个关系")
                    
                except Exception as e:
                    st.error(f"创建失败: {e}")
        
        if st.button("创建示例公司数据"):
            with st.spinner("创建中..."):
                try:
                    queries = [
                        "CREATE (c:Company {name: 'TechCorp', type: '科技公司', employees: 100})",
                        "CREATE (d:Department {name: '研发部', budget: 1000000})",
                        "CREATE (d:Department {name: '设计部', budget: 500000})",
                        "MATCH (c:Company {name: 'TechCorp'}), (d:Department {name: '研发部'}) CREATE (c)-[:HAS_DEPARTMENT]->(d)",
                        "MATCH (c:Company {name: 'TechCorp'}), (d:Department {name: '设计部'}) CREATE (c)-[:HAS_DEPARTMENT]->(d)"
                    ]
                    
                    for query in queries:
                        neo4j_manager.run_query(query)
                    
                    st.success("✅ 公司数据创建成功！")
                    
                except Exception as e:
                    st.error(f"创建失败: {e}")
    
    with col2:
        st.subheader("🗑️ 数据清理")
        
        st.warning("⚠️ 危险操作 - 请谨慎使用")
        
        if st.button("清除所有测试数据", type="secondary"):
            if st.checkbox("我确认要删除所有数据"):
                with st.spinner("清理中..."):
                    try:
                        neo4j_manager.run_query("MATCH (n) DETACH DELETE n")
                        st.success("✅ 所有数据已清除")
                    except Exception as e:
                        st.error(f"清理失败: {e}")

# 底部信息
st.markdown("---")
st.markdown("""
### 💡 使用提示:
- **聊天功能**: 可以询问数据库中的节点和关系信息
- **数据查询**: 使用预设查询或自定义Cypher语句
- **数据管理**: 添加测试数据或清理数据库
- **系统状态**: 在侧边栏查看连接状态和统计信息

### 🔧 功能说明:
- 支持Neo4j图数据库查询
- 基础的自然语言理解
- 可扩展的智能对话功能
- 直观的数据可视化
""")

if __name__ == "__main__":
    # 这里可以添加启动逻辑
    pass
