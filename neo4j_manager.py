"""
Neo4j数据库管理器
处理Neo4j连接和查询操作
"""
from neo4j import GraphDatabase
from config import config
import logging

class Neo4jManager:
    def __init__(self):
        self.driver = None
        self.connect()
    
    def connect(self):
        """连接Neo4j数据库"""
        try:
            self.driver = GraphDatabase.driver(
                config.NEO4J_URI,
                auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
            )
            # 测试连接
            self.driver.verify_connectivity()
            print("✅ Neo4j连接成功")
            return True
        except Exception as e:
            print(f"❌ Neo4j连接失败: {e}")
            self.driver = None
            return False
    
    def close(self):
        """关闭连接"""
        if self.driver:
            self.driver.close()
            print("Neo4j连接已关闭")
    
    def run_query(self, query, parameters=None):
        """执行Cypher查询"""
        if not self.driver:
            print("❌ 数据库未连接")
            return None
        
        try:
            with self.driver.session() as session:
                result = session.run(query, parameters or {})
                return [record for record in result]
        except Exception as e:
            print(f"❌ 查询执行失败: {e}")
            return None
    
    def get_all_nodes(self, limit=10):
        """获取所有节点（限制数量）"""
        query = "MATCH (n) RETURN n LIMIT $limit"
        return self.run_query(query, {"limit": limit})
    
    def search_nodes_by_name(self, name):
        """根据名称搜索节点"""
        query = """
        MATCH (n) 
        WHERE toLower(toString(n.name)) CONTAINS toLower($name)
        RETURN n
        LIMIT 20
        """
        return self.run_query(query, {"name": name})
    
    def get_node_relationships(self, node_name):
        """获取节点的关系"""
        query = """
        MATCH (n)-[r]-(m)
        WHERE toLower(toString(n.name)) = toLower($name)
        RETURN n, r, m
        LIMIT 20
        """
        return self.run_query(query, {"name": node_name})
    
    def get_database_stats(self):
        """获取数据库统计信息"""
        stats = {}
        
        # 节点数量
        try:
            result = self.run_query("MATCH (n) RETURN count(n) as node_count")
            if result:
                stats['node_count'] = result[0]['node_count']
        except:
            stats['node_count'] = 0
        
        # 关系数量
        try:
            result = self.run_query("MATCH ()-[r]->() RETURN count(r) as rel_count")
            if result:
                stats['relationship_count'] = result[0]['rel_count']
        except:
            stats['relationship_count'] = 0
        
        # 标签信息
        try:
            result = self.run_query("CALL db.labels()")
            if result:
                stats['labels'] = [record['label'] for record in result]
        except:
            stats['labels'] = []
        
        # 关系类型
        try:
            result = self.run_query("CALL db.relationshipTypes()")
            if result:
                stats['relationship_types'] = [record['relationshipType'] for record in result]
        except:
            stats['relationship_types'] = []
        
        return stats
    
    def create_sample_data(self):
        """创建示例数据"""
        print("🔄 创建示例数据...")
        
        try:
            # 清除现有测试数据
            self.run_query("MATCH (n:TestNode) DETACH DELETE n")
            
            # 创建人员节点
            people = [
                {"name": "张三", "age": 30, "job": "软件工程师", "description": "资深Python开发者，专注于AI应用"},
                {"name": "李四", "age": 28, "job": "数据科学家", "description": "机器学习专家，擅长数据分析"},
                {"name": "王五", "age": 35, "job": "产品经理", "description": "产品策划专家，负责AI产品设计"},
                {"name": "赵六", "age": 32, "job": "UI设计师", "description": "用户体验设计师，专注界面优化"}
            ]
            
            for person in people:
                query = """
                CREATE (p:TestNode:Person {
                    name: $name,
                    age: $age,
                    job: $job,
                    description: $description,
                    created_at: datetime()
                })
                """
                self.run_query(query, person)
            
            # 创建关系
            relationships = [
                ("张三", "李四", "COLLABORATES_WITH", {"project": "AI聊天机器人", "since": "2024-01-01"}),
                ("李四", "王五", "REPORTS_TO", {"team": "AI产品团队"}),
                ("王五", "赵六", "WORKS_WITH", {"project": "用户界面设计"}),
                ("张三", "赵六", "COLLABORATES_WITH", {"project": "前端开发"})
            ]
            
            for source, target, rel_type, props in relationships:
                query = f"""
                MATCH (a:TestNode {{name: $source}})
                MATCH (b:TestNode {{name: $target}})
                CREATE (a)-[r:{rel_type}]->(b)
                SET r += $props
                """
                self.run_query(query, {"source": source, "target": target, "props": props})
            
            print("✅ 示例数据创建成功")
            return True
            
        except Exception as e:
            print(f"❌ 示例数据创建失败: {e}")
            return False

# 创建全局实例
neo4j_manager = Neo4jManager()

# 如果连接成功，可以创建一些示例数据
if neo4j_manager.driver:
    try:
        # 检查是否已有数据
        result = neo4j_manager.run_query("MATCH (n) RETURN count(n) as count")
        if result and result[0]['count'] == 0:
            print("💡 数据库为空，是否需要创建示例数据？")
            # 这里可以选择是否自动创建示例数据
    except:
        pass
