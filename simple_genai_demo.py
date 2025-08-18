# DO NOT EDIT: This file is locked and should not be modified by any agent.
# DO NOT EDIT: This file is locked and should not be modified by any agent.
# DO NOT EDIT: This file is locked and should not be modified by any agent.
"""
Google GenAI SDK 最小用例
需要 pip install google-genai
"""

import os
from google import genai
from dotenv import load_dotenv

# 加载.env环境变量
load_dotenv()

project = os.getenv("GEMINI_PROJECT_ID")
location = os.getenv("GEMINI_LOCATION", "us-central1")
model = os.getenv("GEMINI_MODEL_STD", "gemini-2.0-flash-lite")

client = genai.Client(
    vertexai=True,
    project=project,
    location=location
)

prompt = "请用中文简要介绍一下Neo4j图数据库的主要特点。"

response = client.models.generate_content(
    model=model,
    contents=prompt
)

print("GenAI 输出：")
print(response.text)
