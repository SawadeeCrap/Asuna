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

if not TELEGRAM_TOKEN or not OPENROUTER_API_KEY or not QDRANT_API_KEY or not QDRANT_URL:
    raise ValueError("Не установлены все ключи/URL!")

# ---------------- ИНИЦИАЛИЗАЦИЯ ----------------
app = Flask(__name__)
bot = telebot.TeleBot(TELEGRAM_TOKEN)
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# ---------------- КОЛЛЕКЦИЯ ----------------
collection_name = "knowledge_base"
try:
    qdrant.get_collection(collection_name=collection_name)
    logger.info(f"Коллекция '{collection_name}' уже существует")
except Exception:
    qdrant.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
    )
    logger.info(f"Коллекция '{collection_name}' создана")

# ---------------- ПАМЯТЬ ЧАТА ----------------
chat_history = {}  # chat_id -> [{"role":"user"/"assistant", "content": "..."}]
MAX_HISTORY = 10  # количество последних сообщений для контекста

# ---------------- ФУНКЦИИ ----------------
def embed_text(text: str) -> list:
    """Получаем эмбеддинг через OpenRouter с полной отладкой"""
    if not text.strip():
        logger.warning("Пустой текст для эмбеддинга")
        return [0.0]*1536

    try:
        resp = requests.post(
            "https://openrouter.ai/api/v1/embeddings",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "openai/text-embedding-3-small",
                "input": text
            },
            timeout=30
        )
        logger.info(f"Raw response from OpenRouter: {resp.text}")  # <-- печатаем что пришло
        resp.raise_for_status()
        data = resp.json()
        embedding = data["data"][0]["embedding"]
        logger.info(f"Эмбеддинг получен, длина: {len(embedding)}")
        return embedding
    except requests.RequestException as e:
        logger.error(f"Ошибка запроса к OpenRouter: {e}")
    except ValueError as e:
        logger.error(f"Ошибка разбора JSON: {e} | Response: {getattr(resp, 'text', 'нет ответа')}")
    except KeyError as e:
        logger.error(f"Ошибка формата ответа: {e} | Response: {resp.text}")
    
    return [0.0]*1536

def add_doc(doc: str):
    vector = embed_text(doc)
    qdrant.upsert(
        collection_name=collection_name,
        points=[PointStruct(id=str(uuid.uuid4()), vector=vector, payload={"text": doc})]
    )
    logger.info(f"Документ добавлен в базу: {doc}")

def search_docs(query: str, top_k=3):
    vector = embed_text(query)
    results = qdrant.search(collection_name=collection_name, query_vector=vector, limit=top_k)
    found = [r.payload["text"] for r in results]
    logger.info(f"Поиск по базе для запроса '{query}' вернул: {found}")
    return found

def handle_message(message):
    chat_id = message.chat.id
    user_text = message.text.strip()
    if not user_text:
        return

    logger.info(f"Сообщение от {chat_id}: {user_text}")

    # Авто-добавление текста в базу
    if not user_text.startswith("/"):
        add_doc(user_text)

    # Поиск в базе
    retrieved = search_docs(user_text)
    retrieved_context = "\n".join(retrieved) if retrieved else "Нет найденной информации."

    # История диалога
    history = chat_history.get(chat_id, [])
    history_messages = history[-MAX_HISTORY:]

    # Формируем полный контекст LLM
    messages = [
        {"role": "system", "content": "Ты Asuna Cat — веселая кошка, которая рассказывает анекдоты."},
        {"role": "system", "content": f"В базе знаний нашлось:\n{retrieved_context}"},
    ] + history_messages + [{"role": "user", "content": user_text}]

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "openai/gpt-3.5-turbo",
                "messages": messages,
                "max_tokens": 800,
                "temperature": 0.7
            },
            timeout=20
        )
        response.raise_for_status()
        ai_response = response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.error(f"Ошибка запроса к OpenRouter: {e}")
        ai_response = "Упс, не удалось получить ответ 😅"

    # Отправка ответа
    bot.send_message(chat_id, ai_response, parse_mode="Markdown", disable_web_page_preview=True)

    # Обновляем историю чата
    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": ai_response})
    chat_history[chat_id] = history[-MAX_HISTORY:]

# ---------------- КОМАНДЫ ----------------
@bot.message_handler(commands=["start"])
def start_message(message):
    bot.send_message(message.chat.id, "SKYNET BOT ACTIVATE! 😘")

@bot.message_handler(commands=["help"])
def help_message(message):
    bot.send_message(message.chat.id, "/start — перезапуск\n/add текст — добавить документ\n/search текст — поиск в базе")

@bot.message_handler(commands=["add"])
def add_document(message):
    text = message.text.replace("/add", "").strip()
    if text:
        add_doc(text)
        bot.send_message(message.chat.id, "Документ добавлен ✅")
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

# ---------------- ОБРАБОТКА ВСЕХ СООБЩЕНИЙ ----------------
@bot.message_handler(func=lambda message: True)
def echo_all(message):
    Thread(target=handle_message, args=(message,)).start()

# ---------------- FLASK WEBHOOK ----------------
@app.route(f"/{TELEGRAM_TOKEN}", methods=["POST"])
def webhook():
    if request.headers.get("content-type") == "application/json":
        update = telebot.types.Update.de_json(request.get_data().decode("utf-8"))
        bot.process_new_updates([update])
        return ""
    else:
        abort(403)

# ---------------- MAIN ----------------
if __name__ == "__main__":
    webhook_url = f"{RENDER_URL}/{TELEGRAM_TOKEN}"
    bot.remove_webhook()
    bot.set_webhook(url=webhook_url)
    logger.info(f"Webhook установлен: {webhook_url}")
    port = int(os.getenv("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
