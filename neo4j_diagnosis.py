"""
Neo4j连接诊断脚本
帮助诊断和解决Neo4j连接问题
"""
import os
from dotenv import load_dotenv

load_dotenv()

def test_neo4j_connection():
    """测试Neo4j连接的多种方式"""
    print("🔍 Neo4j连接诊断")
    print("=" * 40)
    
    # 获取配置
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER") 
    password = os.getenv("NEO4J_PASSWORD")
    
    print(f"URI: {uri}")
    print(f"用户: {user}")
    print(f"密码: {'已设置' if password else '未设置'}")
    print()
    
    # 尝试不同的连接方式
    connection_uris = [
        "bolt://localhost:7687",
        "bolt://127.0.0.1:7687", 
        "neo4j://localhost:7687",
        "neo4j://127.0.0.1:7687"
    ]
    
    try:
        from neo4j import GraphDatabase
        
        for test_uri in connection_uris:
            print(f"🔄 尝试连接: {test_uri}")
            try:
                driver = GraphDatabase.driver(test_uri, auth=(user, password))
                driver.verify_connectivity()
                print(f"✅ 连接成功！")
                
                # 测试简单查询
                with driver.session() as session:
                    result = session.run("RETURN 1 as test")
                    test_value = result.single()["test"]
                    print(f"✅ 查询测试成功: {test_value}")
                
                driver.close()
                
                # 如果成功，更新.env文件
                if test_uri != uri:
                    print(f"💡 建议更新.env文件中的URI为: {test_uri}")
                
                return True
                
            except Exception as e:
                print(f"❌ 连接失败: {e}")
                continue
        
        print("\n❌ 所有连接尝试都失败了")
        print("\n💡 可能的解决方案:")
        print("1. 启动Neo4j桌面版或社区版")
        print("2. 检查Neo4j服务是否运行在端口7687")
        print("3. 验证用户名和密码是否正确")
        print("4. 检查防火墙设置")
        
        return False
        
    except ImportError:
        print("❌ neo4j包未安装，请运行: pip install neo4j")
        return False

def check_neo4j_status():
    """检查Neo4j服务状态"""
    print("\n🔍 检查Neo4j服务状态")
    print("-" * 30)
    
    import subprocess
    import socket
    
    # 检查端口7687是否开放
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('127.0.0.1', 7687))
        sock.close()
        
        if result == 0:
            print("✅ 端口7687可访问")
        else:
            print("❌ 端口7687不可访问")
            print("   Neo4j服务可能未启动")
    except Exception as e:
        print(f"❌ 端口检查失败: {e}")
    
    # 尝试检查Neo4j进程
    try:
        if os.name == 'nt':  # Windows
            result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq java.exe'], 
                                  capture_output=True, text=True)
            if 'java.exe' in result.stdout:
                print("✅ 检测到Java进程（可能是Neo4j）")
            else:
                print("❌ 未检测到Java进程")
        else:  # Unix/Linux
            result = subprocess.run(['pgrep', '-f', 'neo4j'], 
                                  capture_output=True, text=True)
            if result.stdout.strip():
                print("✅ 检测到Neo4j进程")
            else:
                print("❌ 未检测到Neo4j进程")
    except Exception as e:
        print(f"⚠️ 进程检查失败: {e}")

def install_guide():
    """Neo4j安装指南"""
    print("\n📚 Neo4j安装指南")
    print("-" * 30)
    print("如果Neo4j未安装，请选择以下方式之一:")
    print()
    print("1️⃣ Neo4j桌面版（推荐）:")
    print("   - 下载: https://neo4j.com/download/")
    print("   - 创建新数据库")
    print("   - 设置密码为: OhMyDear")
    print("   - 启动数据库")
    print()
    print("2️⃣ Neo4j社区版（命令行）:")
    print("   - 下载并解压Neo4j社区版")
    print("   - 运行: bin/neo4j console")
    print("   - 首次访问 http://localhost:7474")
    print("   - 设置密码")
    print()
    print("3️⃣ Docker方式:")
    print("   docker run -p 7474:7474 -p 7687:7687 \\")
    print("   -e NEO4J_AUTH=neo4j/OhMyDear neo4j:latest")

if __name__ == "__main__":
    # 运行连接测试
    success = test_neo4j_connection()
    
    if not success:
        check_neo4j_status()
        install_guide()
