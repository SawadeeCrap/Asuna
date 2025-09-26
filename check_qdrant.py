import os
from qdrant_client import QdrantClient

QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")

qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# Получаем список коллекций
collections = qdrant.get_collections()
print("Коллекции в Qdrant:", collections)
