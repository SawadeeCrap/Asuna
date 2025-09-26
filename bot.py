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
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# Создание коллекции (если её ещё нет)
try:
    qdrant.recreate_collection(
        collection_name="knowledge_base",
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
    )
    logger.info("Qdrant collection recreated")
except Exception:
    logger.info("Collection already exists or ignored")

# ---------------- ПАМЯТЬ ЧАТА ----------------
chat_history = {}  # {user_id: [{"role": "user"/"assistant", "content": "..."}]}

# ---------------- ФУНКЦИИ ----------------
def embed_text(text: str) -> list:
    """Берём эмбеддинг через OpenRouter"""
    try:
        resp = requests.post(
            "https://openrouter.ai/api/v1/embeddings",
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"},
            json={"model": "openai/text-embedding-3-small", "input": text},
            timeout=15
        )
        resp.raise_for_status()
        data = resp.json()
        return data["data"][0]["embedding"]
    except Exception as e:
        logger.error(f"Ошибка эмбеддинга OpenRouter: {e} | Response: {getattr(resp, 'text', 'нет ответа')}")
        return [0.0] * 1536  # безопасная заглушка

def add_doc(doc: str):
    """Добавить документ в Qdrant"""
    vector = embed_text(doc)
    qdrant.upsert(
        collection_name="knowledge_base",
        points=[PointStruct(id=str(uuid.uuid4()), vector=vector, payload={"text": doc})]
    )
    logger.info(f"Документ добавлен в базу: {doc}")

def search_docs(query: str, top_k=3):
    """Поиск похожих документов в Qdrant"""
    vector = embed_text(query)
    results = qdrant.search(collection_name="knowledge_base", query_vector=vector, limit=top_k)
    found = [r.payload["text"] for r in results]
    logger.info(f"Поиск по базе для запроса '{query}' вернул: {found}")
    return found

def handle_message(message):
    user_id = message.from_user.id
    chat_id = message.chat.id
    user_text = message.text

    logger.info(f"Сообщение от {user_id}: {user_text}")

    # --- Сохраняем сообщение в историю ---
    if user_id not in chat_history:
        chat_history[user_id] = []
    chat_history[user_id].append({"role": "user", "content": user_text})

    # --- Сохраняем в Qdrant ---
    add_doc(user_text)

    # --- Поиск по базе ---
    retrieved = search_docs(user_text)
    context = "\n".join(retrieved) if retrieved else "Нет найденной информации."

    # --- Формируем контекст для LLM ---
    messages = [
        {"role": "system", "content": "Ты Asuna Cat — веселая кошка, которая рассказывает смешные ответы."},
        {"role": "system", "content": f"В базе знаний нашлось:\n{context}"},
    ] + chat_history[user_id]  # добавляем историю пользователя

    # --- Запрос к OpenRouter Chat ---
    try:
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
    except Exception as e:
        logger.error(f"Ошибка запроса к OpenRouter Chat: {e}")
        ai_response = "Упс, не удалось получить ответ 😅"

    # --- Сохраняем ответ в историю ---
    chat_history[user_id].append({"role": "assistant", "content": ai_response})

    bot.send_message(chat_id, ai_response, parse_mode="Markdown", disable_web_page_preview=True)

# ---------------- TELEGRAM ХЕНДЛЕРЫ ----------------
@bot.message_handler(commands=["start"])
def start_message(message):
    bot.send_message(message.chat.id, "Привет! Я запоминаю всё, что ты пишешь 😼")

@bot.message_handler(func=lambda message: True)
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
