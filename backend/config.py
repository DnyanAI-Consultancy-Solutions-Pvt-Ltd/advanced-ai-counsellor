import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
EDUCATION_API_KEY = os.getenv("EDUCATION_API_KEY")

# External Service URLs
COLLEGE_DATA_API_URL = os.getenv("COLLEGE_DATA_API_URL", "https://api.educationdata.org/v1")
SCHOLARSHIP_REGISTRY_URL = os.getenv("SCHOLARSHIP_REGISTRY_URL", "https://api.scholarships.gov/v1")

# Storage & Vector DB Settings
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Server settings
FASTAPI_HOST = os.getenv("FASTAPI_HOST", "0.0.0.0")
FASTAPI_PORT = int(os.getenv("FASTAPI_PORT", 8000))