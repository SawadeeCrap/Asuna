import os
import logging
import uuid
from threading import Thread

from flask import Flask, request, abort
import telebot
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from openai import OpenAI

# ---------------- ЛОГИ ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------- ТОКЕНЫ ----------------
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
RENDER_URL = os.getenv("RENDER_URL", "https://example.onrender.com")

if not TELEGRAM_TOKEN or not OPENAI_API_KEY or not QDRANT_API_KEY or not QDRANT_URL:
    raise ValueError("Проверьте все секреты: TELEGRAM, OPENAI, QDRANT")

# ---------------- ИНИЦИАЛИЗАЦИЯ ----------------
app = Flask(__name__)
bot = telebot.TeleBot(TELEGRAM_TOKEN)

# Qdrant клиент
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# OpenAI клиент
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Создание коллекции
try:
    qdrant.recreate_collection(
        collection_name="knowledge_base",
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
    )
    logger.info("Qdrant collection recreated")
except Exception:
    logger.info("Collection already exists или игнорируем ошибку")

# ---------------- ФУНКЦИИ ----------------
def embed_text(text: str) -> list:
    """Получение эмбеддинга через OpenAI 1.0 API"""
    if not text.strip():
        return [0.0]*1536
    try:
        resp = openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        embedding = resp.data[0].embedding
        logger.info(f"Эмбеддинг получен, длина: {len(embedding)}")
        return embedding
    except Exception as e:
        logger.error(f"Ошибка эмбеддинга OpenAI: {e}")
        return [0.0]*1536

def add_doc(doc: str):
    """Добавление документа в Qdrant"""
    vector = embed_text(doc)
    qdrant.upsert(
        collection_name="knowledge_base",
        points=[PointStruct(id=str(uuid.uuid4()), vector=vector, payload={"text": doc})]
    )
    logger.info(f"Документ добавлен в базу: {doc}")

def search_docs(query: str, top_k=3):
    """Поиск похожих документов"""
    vector = embed_text(query)
    results = qdrant.search(collection_name="knowledge_base", query_vector=vector, limit=top_k)
    found = [r.payload["text"] for r in results]
    logger.info(f"Поиск по базе для запроса '{query}' вернул: {found}")
    return found

# ---------------- ОБРАБОТКА СООБЩЕНИЙ ----------------
def handle_message(message):
    user_id = message.from_user.id
    chat_id = message.chat.id
    user_text = message.text
    logger.info(f"Сообщение от {user_id}: {user_text}")

    # Сохраняем сообщение в базе
    add_doc(user_text)

    # Поиск контекста
    retrieved = search_docs(user_text)
    context = "\n".join(retrieved) if retrieved else "Нет найденной информации."

    # Формируем запрос к LLM
    messages = [
        {"role": "system", "content": "Ты Asuna Cat — веселая кошка, которая отвечает смешно и по делу."},
        {"role": "system", "content": f"В базе знаний нашлось:\n{context}"},
        {"role": "user", "content": user_text},
    ]

    try:
        resp = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=800,
            temperature=0.7,
        )
        ai_response = resp.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Ошибка запроса к OpenAI Chat: {e}")
        ai_response = "Упс, не удалось получить ответ 😅"

    bot.send_message(chat_id, ai_response, parse_mode="Markdown", disable_web_page_preview=True)

# ---------------- КОМАНДЫ TELEGRAM ----------------
@bot.message_handler(commands=["start"])
def start_message(message):
    bot.send_message(message.chat.id, "SKYNET BOT ACTIVATE! 😘")

@bot.message_handler(commands=["help"])
def help_message(message):
    bot.send_message(message.chat.id, "/start — перезапуск\n/help — справка")

# ---------------- ОБРАБОТКА ВСЕХ СООБЩЕНИЙ ----------------
@bot.message_handler(func=lambda m: True)
def all_messages(message):
    Thread(target=handle_message, args=(message,)).start()

# ---------------- FLASK WEBHOOK ----------------
@app.route(f"/{TELEGRAM_TOKEN}", methods=["POST"])
def webhook():
    if request.headers.get("content-type") == "application/json":
        json_string = request.get_data().decode("utf-8")
        update = telebot.types.Update.de_json(json_string)
        bot.process_new_updates([update])
        return ""
    else:
        abort(403)

# ---------------- MAIN ----------------
if __name__ == "__main__":
    webhook_url = f"{RENDER_URL}/{TELEGRAM_TOKEN}"
    bot.remove_webhook()
    bot.set_webhook(url=webhook_url)
    logger.info(f"Webhook set: {webhook_url}")
    port = int(os.getenv("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
