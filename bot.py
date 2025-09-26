import os
import logging
from flask import Flask, request
import telebot
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from qdrant_client.models import PointStruct, VectorParams

# --- Логи ---
logging.basicConfig(level=logging.INFO)

# --- Настройки ---
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "knowledge_base"

# --- Инициализация клиентов ---
bot = telebot.TeleBot(TELEGRAM_TOKEN)
app = Flask(__name__)
openai_client = OpenAI(api_key=OPENAI_API_KEY)
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# --- Функции ---
def embed_text(text: str):
    try:
        resp = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        vector = resp.data[0].embedding
        logging.info(f"Эмбеддинг текста '{text}' создан")
        return vector
    except Exception as e:
        logging.error(f"Ошибка эмбеддинга OpenAI: {e}")
        return None

def add_doc_to_qdrant(text: str):
    vector = embed_text(text)
    if not vector:
        return
    point = PointStruct(id=None, vector=vector, payload={"text": text})
    qdrant.upsert(collection_name=COLLECTION_NAME, points=[point])
    logging.info(f"Документ добавлен в базу: {text}")

def search_docs(query: str, top_k=5):
    vector = embed_text(query)
    if not vector:
        return []
    try:
        results = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query_vector=vector,
            limit=top_k,
            with_payload=True
        )
        texts = [p.payload["text"] for p in results.result]
        logging.info(f"Поиск по базе для запроса '{query}' вернул: {texts}")
        return texts
    except Exception as e:
        logging.error(f"Ошибка поиска в Qdrant: {e}")
        return []

def ask_openai_chat(prompt: str, context: list):
    try:
        messages = [{"role": "system", "content": "Ты помощник, который отвечает по базе данных."}]
        if context:
            context_text = "\n".join(context)
            messages.append({"role": "system", "content": f"Контекст из базы: {context_text}"})
        messages.append({"role": "user", "content": prompt})

        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        return resp.choices[0].message.content
    except Exception as e:
        logging.error(f"Ошибка запроса к OpenAI Chat: {e}")
        return "Извини, не могу ответить сейчас."

# --- Обработка сообщений ---
@bot.message_handler(func=lambda m: True)
def handle_message(message):
    user_text = message.text
    logging.info(f"Сообщение от {message.from_user.id}: {user_text}")

    if user_text.lower().startswith("запомни "):
        text_to_remember = user_text[7:]
        add_doc_to_qdrant(text_to_remember)
        bot.send_message(message.chat.id, f"Запомнил: {text_to_remember}")
        return

    # Поиск по базе
    retrieved = search_docs(user_text)
    reply = ask_openai_chat(user_text, context=retrieved)
    bot.send_message(message.chat.id, reply)

# --- Flask webhook ---
@app.route(f"/{TELEGRAM_TOKEN}", methods=["POST"])
def webhook():
    update = request.get_json()
    bot.process_new_updates([telebot.types.Update.de_json(update)])
    return "", 200

if __name__ == "__main__":
    bot.remove_webhook()  # удаляем старый webhook
    logging.info("Webhook удалён (если был). Flask сервер запущен.")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
