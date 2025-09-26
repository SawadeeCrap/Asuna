import os
import logging
import uuid
import requests
from flask import Flask, request, abort
import telebot
from dotenv import load_dotenv
from threading import Thread
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

# ---------------- ЛОГИ ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------- ТОКЕНЫ ----------------
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
RENDER_URL = os.getenv("RENDER_URL", "https://example.onrender.com")

if not TELEGRAM_TOKEN:
    raise ValueError("TELEGRAM_TOKEN not set")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY not set")
if not QDRANT_API_KEY or not QDRANT_URL:
    raise ValueError("QDRANT credentials not set")

# ---------------- ИНИЦИАЛИЗАЦИЯ ----------------
app = Flask(__name__)
bot = telebot.TeleBot(TELEGRAM_TOKEN)

# Qdrant клиент
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# Создание коллекции (один раз)
try:
    qdrant.recreate_collection(
        collection_name="knowledge_base",
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
    )
    logger.info("Qdrant collection recreated")
except Exception:
    logger.info("Collection exists or ignored")

# ---------------- ФУНКЦИИ ----------------
def embed_text(text: str) -> list:
    """Берём эмбеддинг через OpenRouter с защитой от пустого ответа"""
    try:
        resp = requests.post(
            "https://openrouter.ai/api/v1/embeddings",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            },
            json={"model": "openai/text-embedding-3-small", "input": text},
            timeout=15
        )
        resp.raise_for_status()
        data = resp.json()
        embedding = data["data"][0]["embedding"]
        logger.info(f"Эмбеддинг получен для текста: {text[:30]}...")
        return embedding
    except Exception as e:
        logger.error(f"Ошибка эмбеддинга OpenRouter: {e} | Response: {getattr(resp, 'text', 'нет ответа')}")
        return [0.0] * 1536  # безопасная заглушка

def add_doc(doc: str):
    """Добавить документ в Qdrant"""
    vector = embed_text(doc)
    qdrant.upsert(
        collection_name="knowledge_base",
        points=[
            PointStruct(id=str(uuid.uuid4()), vector=vector, payload={"text": doc})
        ]
    )
    logger.info(f"Документ добавлен в базу: {doc[:30]}...")

def search_docs(query: str, top_k=3):
    """Поиск похожих документов в Qdrant"""
    vector = embed_text(query)
    results = qdrant.query_points(
        collection_name="knowledge_base",
        vector=vector,
        limit=top_k
    )
    texts = [r.payload["text"] for r in results if r.payload and "text" in r.payload]
    logger.info(f"Поиск по базе для запроса '{query[:30]}...' вернул: {texts}")
    return texts

def handle_message(message):
    user_id = message.from_user.id
    chat_id = message.chat.id
    user_text = message.text
    logger.info(f"Сообщение от {user_id}: {user_text}")

    # Поиск в базе
    retrieved = search_docs(user_text)
    context = "\n".join(retrieved) if retrieved else "Нет найденной информации."

    # Сообщения для LLM
    messages = [
        {"role": "system", "content": "Ты Asuna Cat — веселая кошка, которая отвечает с юмором."},
        {"role": "system", "content": f"В базе знаний нашлось:\n{context}"},
        {"role": "user", "content": user_text},
    ]

    try:
        # Запрос в OpenRouter Chat
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "x-ai/grok-4-fast:free",
                "messages": messages,
                "max_tokens": 800,
                "temperature": 0.7,
            },
            timeout=20
        )
        response.raise_for_status()
        ai_response = response.json()["choices"][0]["message"]["content"].strip()
        bot.send_message(chat_id, ai_response, parse_mode="Markdown", disable_web_page_preview=True)
    except Exception as e:
        logger.error(f"Ошибка запроса к OpenRouter Chat: {e}")
        bot.send_message(chat_id, f"Ошибка API: {str(e)} 😅")

# ---------------- TELEGRAM ХЕНДЛЕРЫ ----------------
@bot.message_handler(commands=["start"])
def start_message(message):
    bot.send_message(message.chat.id, "SKYNET BOT ACTIVATE! 😘")

@bot.message_handler(commands=["help"])
def help_message(message):
    bot.send_message(message.chat.id, "/start — перезапуск.\n/add текст — добавить документ\n/search текст — поиск в базе")

@bot.message_handler(commands=["add"])
def add_document(message):
    text = message.text.replace("/add", "").strip()
    if text:
        add_doc(text)
        bot.send_message(message.chat.id, "Документ добавлен в базу ✅")
    else:
        bot.send_message(message.chat.id, "Напиши текст после /add")

@bot.message_handler(commands=["search"])
def search_document(message):
    query = message.text.replace("/search", "").strip()
    if not query:
        bot.send_message(message.chat.id, "Напиши текст после /search")
        return
    retrieved = search_docs(query)
    if retrieved:
        result_text = "\n\n".join(f"{i+1}. {doc}" for i, doc in enumerate(retrieved))
        bot.send_message(message.chat.id, f"Найденные документы:\n{result_text}")
    else:
        bot.send_message(message.chat.id, "По вашему запросу ничего не найдено 😅")

# Любые сообщения обрабатываем через LLM + поиск
@bot.message_handler(func=lambda message: True)
def echo_all(message):
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
