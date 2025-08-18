import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Neo4j配置
    NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
    NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
    
    # 嵌入模型配置
    EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "sbert")
    SBERT_MODEL = os.getenv("SBERT_MODEL", "all-MiniLM-L6-v2")

    # LLM配置
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")
    GEMINI_PROJECT_ID = os.getenv("GEMINI_PROJECT_ID")
    GEMINI_LOCATION = os.getenv("GEMINI_LOCATION", "us-central1")
    GEMINI_MODEL_MID = os.getenv("GEMINI_MODEL_MID", "gemini-2.5-flash")
    GEMINI_MODEL_STD = os.getenv("GEMINI_MODEL_STD", "gemini-2.0-flash-lite")
    GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    
    # 检索阈值
    RETRIEVE_HIGH = float(os.getenv("RETRIEVE_HIGH", "0.82"))
    RETRIEVE_MID = float(os.getenv("RETRIEVE_MID", "0.62"))
    
    # 内存设置
    WRITE_MEMORY = os.getenv("WRITE_MEMORY", "true").lower() == "true"
    
    # 范围锁定检索设置
    SCOPE_LOCKED = os.getenv("SCOPE_LOCKED", "true").lower() == "true"
    HIGH_THRESHOLD = float(os.getenv("HIGH_THRESHOLD", "0.82"))
    MID_THRESHOLD = float(os.getenv("MID_THRESHOLD", "0.62"))
    AMBIG_DELTA = float(os.getenv("AMBIG_DELTA", "0.04"))
    WEIGHT_RECENT_VEC = float(os.getenv("WEIGHT_RECENT_VEC", "0.04"))
    PENALTY_REPEAT = float(os.getenv("PENALTY_REPEAT", "0.10"))
    RECENT_MAX_LEN = int(os.getenv("RECENT_MAX_LEN", "6"))
    RECENT_PREFIX_ITEM_MAX_CHARS = int(os.getenv("RECENT_PREFIX_ITEM_MAX_CHARS", "120"))

config = Config()

def verify_config():
    """验证配置是否正确"""
    print("🔧 配置验证:")
    print(f"Neo4j URI: {config.NEO4J_URI}")
    print(f"Neo4j User: {config.NEO4J_USER}")
    print(f"Neo4j Password: {'✅ 已设置' if config.NEO4J_PASSWORD else '❌ 未设置'}")
    print(f"Gemini Project ID: {config.GEMINI_PROJECT_ID}")
    print(f"Google Credentials: {'✅ 已设置' if config.GOOGLE_APPLICATION_CREDENTIALS else '❌ 未设置'}")
    print(f"嵌入模型: {config.SBERT_MODEL}")
    print()

if __name__ == "__main__":
    verify_config()
