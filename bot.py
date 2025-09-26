import os
import logging
import threading
import uuid
from flask import Flask, request
import telebot
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
import requests

# ----------------- Логи -----------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ----------------- Настройки -----------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # только для эмбеддингов
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")  # для Grok
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
RENDER_URL = os.getenv("RENDER_URL", "https://asuna-3bfa.onrender.com")  # URL вашего приложения
ADMIN_USER_ID = int(os.getenv("ADMIN_USER_ID", "0"))  # ID администратора
PORT = int(os.getenv("PORT", 5000))

# Проверяем наличие всех токенов
required_tokens = {
    "TELEGRAM_TOKEN": TELEGRAM_TOKEN,
    "OPENAI_API_KEY": OPENAI_API_KEY,
    "OPENROUTER_API_KEY": OPENROUTER_API_KEY,
    "QDRANT_URL": QDRANT_URL,
    "RENDER_URL": RENDER_URL,
    "ADMIN_USER_ID": ADMIN_USER_ID
}

for name, token in required_tokens.items():
    if not token or (name == "ADMIN_USER_ID" and token == 0):
        logging.error(f"Отсутствует переменная окружения: {name}")
        if name == "ADMIN_USER_ID":
            logging.error("ADMIN_USER_ID должен быть установлен в ID вашего Telegram аккаунта")
        exit(1)

bot = telebot.TeleBot(TELEGRAM_TOKEN)
app = Flask(__name__)

# ----------------- Клиенты -----------------
openai_client = OpenAI(api_key=OPENAI_API_KEY)
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

COLLECTION_NAME = "knowledge_base"
CONTEXT_COLLECTION = "user_contexts"

# ----------------- Проверка прав -----------------
def is_admin(user_id: int) -> bool:
    """Проверка, является ли пользователь администратором"""
    return user_id == ADMIN_USER_ID

# Хранилище контекстов пользователей в памяти
user_contexts = {}

# ----------------- Функции для эмбеддингов -----------------
def create_embedding(text: str):
    """Создание эмбеддинга через OpenAI"""
    try:
        response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text.strip()
        )
        return response.data[0].embedding
    except Exception as e:
        logging.error(f"Ошибка создания эмбеддинга: {e}")
        return None

# ----------------- Функции для работы с Qdrant -----------------
def init_collections():
    """Инициализация коллекций в Qdrant"""
    try:
        collections = qdrant.get_collections()
        existing_collections = [col.name for col in collections.collections]
        
        # Коллекция для базы знаний
        if COLLECTION_NAME not in existing_collections:
            qdrant.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
            )
            logging.info(f"Создана коллекция: {COLLECTION_NAME}")
        else:
            logging.info(f"Коллекция {COLLECTION_NAME} уже существует")
            
    except Exception as e:
        logging.error(f"Ошибка инициализации коллекций: {e}")

def add_to_knowledge_base(text: str, source: str = "user"):
    """Добавление знания в базу"""
    try:
        vector = create_embedding(text)
        if not vector:
            return False
            
        point_id = str(uuid.uuid4())
        point = PointStruct(
            id=point_id,
            vector=vector,
            payload={
                "text": text,
                "source": source,
                "id": point_id
            }
        )
        
        qdrant.upsert(collection_name=COLLECTION_NAME, points=[point])
        logging.info(f"Добавлено знание: {text[:50]}...")
        return True
    except Exception as e:
        logging.error(f"Ошибка добавления в базу знаний: {e}")
        return False

def search_knowledge(query: str, threshold: float = 0.7, limit: int = 5):
    """Поиск релевантной информации в базе знаний"""
    try:
        vector = create_embedding(query)
        if not vector:
            return []
            
        results = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=vector,
            limit=limit,
            score_threshold=threshold  # Снижен порог для большего охвата
        ).points
        
        if results:
            logging.info(f"Найдено {len(results)} релевантных записей для запроса: {query}")
            return [point.payload["text"] for point in results]
        else:
            logging.info(f"Релевантной информации не найдено для: {query}")
            return []
            
    except Exception as e:
        logging.error(f"Ошибка поиска в базе знаний: {e}")
        return []

# ----------------- Функции для OpenRouter (Grok) -----------------
def ask_grok(question: str, context: list = None, user_context: list = None):
    """Запрос к Grok через OpenRouter"""
    try:
        messages = []
        
        # Системное сообщение с новой логикой
        if context:
            system_prompt = f"""Ты Asuna. Умный и добрый помощник с доступом к базе знаний. 

ВАЖНО: У тебя есть следующая информация из базы знаний:
{chr(10).join(context)}

Используй эту информацию для ответа на вопрос пользователя. Если информация из базы релевантна - опирайся строго на неё.

Если в базе нет релевантной информации - отвечай на основе своих знаний или скажи что не знаешь ответа.

Всегда отвечай полезно и подробно."""
        else:
            system_prompt = """Ты умный помощник. Отвечай полезно и подробно на вопросы пользователя на основе своих знаний."""

        messages.append({"role": "system", "content": system_prompt})
        
        # Добавляем контекст диалога пользователя (последние сообщения)
        if user_context:
            for msg in user_context[-6:]:  # последние 6 сообщений для контекста
                messages.append(msg)
        
        # Текущий вопрос пользователя
        messages.append({"role": "user", "content": question})
        
        # Запрос к OpenRouter
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "x-ai/grok-4-fast:free",  # Бесплатная модель Grok
                "messages": messages,
                "temperature": 0.2,  # Креативность ответов (0.0-1.0)
                "max_tokens": 250   # МАКСИМАЛЬНАЯ ДЛИНА ОТВЕТА (увеличено до 1200)
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            logging.error(f"Ошибка OpenRouter: {response.status_code} - {response.text}")
            return "Извини, у меня проблемы с получением ответа. Попробуй позже."
            
    except Exception as e:
        logging.error(f"Ошибка запроса к Grok: {e}")
        return "Извини, произошла ошибка. Попробуй позже."

# ----------------- Управление контекстом пользователей -----------------
def add_to_user_context(user_id: int, message: str, is_bot: bool = False):
    """Добавление сообщения в контекст пользователя"""
    if user_id not in user_contexts:
        user_contexts[user_id] = []
    
    role = "assistant" if is_bot else "user"
    user_contexts[user_id].append({"role": role, "content": message})
    
    # Ограничиваем контекст последними 10 сообщениями
    if len(user_contexts[user_id]) > 10:
        user_contexts[user_id] = user_contexts[user_id][-10:]

def get_user_context(user_id: int):
    """Получение контекста пользователя"""
    return user_contexts.get(user_id, [])

# ----------------- Обработчики команд -----------------
@bot.message_handler(commands=['start'])
def handle_start(message):
    user_id = message.from_user.id
    username = message.from_user.username or message.from_user.first_name
    
    if is_admin(user_id):
        welcome_text = """ Привет, Админ! Я умный бот Asuna с базой знаний.

 **Ваши привилегии:**
• Добавлять информацию в базу: "запомни что-то"
• Полный доступ ко всем функциям

 **Для обычных пользователей:**
• Могут задавать вопросы и получать ответы
• Их диалоги запоминаются в контексте
• НЕ могут изменять общую базу знаний"""
    else:
        welcome_text = f""" Привет, {username}! Я умный бот с базой знаний.

Что я умею:
• Отвечать на любые вопросы
• Запоминать наш диалог для контекста
• Использовать знания из базы для более точных ответов

 Примечание: Общую базу знаний может пополнять только администратор."""
    
    bot.reply_to(message, welcome_text)
    add_to_user_context(message.from_user.id, message.text)
    add_to_user_context(message.from_user.id, welcome_text, is_bot=True)

@bot.message_handler(commands=['help'])
def handle_help(message):
    user_id = message.from_user.id
    
    if is_admin(user_id):
        help_text = """ **Команды для Администратора:**

1. **Добавить в базу знаний:**
   `запомни Python - это язык программирования`

2. **Задать вопрос:**
   `что такое Python?`

3. **Очистить контекст:**
   `/clear`

4. **Проверить статус:**
   `/admin`

 **Как работаю:**
- База знаний доступна всем пользователям
- Только вы можете её пополнять
- У каждого пользователя свой контекст диалога"""
    else:
        help_text = """ **Как пользоваться ботом:**

1. **Задать вопрос:**
   `что такое Python?`
   `как работает интернет?`

2. **Очистить наш диалог:**
   `/clear`

 **Как я работаю:**
- Отвечаю на основе общей базы знаний + своих знаний
- Запоминаю наш диалог для контекста
- База знаний пополняется администратором"""
    
    bot.reply_to(message, help_text)

@bot.message_handler(commands=['clear'])
def handle_clear(message):
    user_id = message.from_user.id
    if user_id in user_contexts:
        del user_contexts[user_id]
    bot.reply_to(message, " Контекст диалога очищен")

@bot.message_handler(commands=['admin'])
def handle_admin(message):
    user_id = message.from_user.id
    if is_admin(user_id):
        admin_info = f""" **Админ панель**

 Ваш ID: `{user_id}`
 Активных пользователей: {len(user_contexts)}
 Статус базы знаний: Активна

**Доступные команды:**
• `запомни [текст]` - добавить в базу
• `/clear` - очистить свой контекст
• `/admin` - эта панель
        
        bot.reply_to(message, admin_info, parse_mode='Markdown')
    else:
        bot.reply_to(message, " У вас нет прав администратора")

@bot.message_handler(commands=['database', 'db'])
def handle_database(message):
    user_id = message.from_user.id
    if not is_admin(user_id):
        bot.reply_to(message, " У вас нет прав администратора")
        return
    
    try:
        # Получаем все записи из коллекции
        result = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            limit=20,  # Показываем первые 20 записей
            with_payload=True
        )
        
        points = result[0]  # result возвращает (points, next_page_offset)
        
        if not points:
            bot.reply_to(message, "📭 База данных пуста")
            return
        
        database_text = f"📚 **База знаний** (показано {len(points)} из записей):\n\n"
        
        for i, point in enumerate(points, 1):
            text = point.payload.get("text", "Нет текста")
            source = point.payload.get("source", "unknown")
            # Обрезаем длинные записи
            if len(text) > 100:
                text = text[:100] + "..."
            database_text += f"{i}. `{text}`\n   _Источник: {source}_\n\n"
        
        # Telegram имеет лимит 4096 символов
        if len(database_text) > 4000:
            database_text = database_text[:4000] + "\n...\n_Список обрезан_"
        
        bot.reply_to(message, database_text, parse_mode='Markdown')
        
    except Exception as e:
        logging.error(f"Ошибка получения базы данных: {e}")
        bot.reply_to(message, f" Ошибка получения данных: {e}")

@bot.message_handler(commands=['count'])
def handle_count(message):
    user_id = message.from_user.id
    if not is_admin(user_id):
        bot.reply_to(message, " У вас нет прав администратора")
        return
    
    try:
        # Получаем информацию о коллекции
        collection_info = qdrant.get_collection(COLLECTION_NAME)
        points_count = collection_info.points_count
        
        count_text = f"""График: **Статистика базы данных:**

 Коллекция: `{COLLECTION_NAME}`
 Записей: **{points_count}**
 Размер вектора: 1536
 Метрика: Cosine"""
        
        bot.reply_to(message, count_text, parse_mode='Markdown')
        
    except Exception as e:
        logging.error(f"Ошибка получения статистики: {e}")
        bot.reply_to(message, f" Ошибка получения статистики: {e}")

# ----------------- Обработчик сообщений -----------------
@bot.message_handler(func=lambda message: True)
def handle_message(message):
    try:
        user_id = message.from_user.id
        user_text = message.text.strip()
        
        logging.info(f"Сообщение от {user_id}: {user_text}")
        
        # Добавляем сообщение в контекст
        add_to_user_context(user_id, user_text)
        
        # Команда для запоминания - только для админа
        if user_text.lower().startswith("запомни "):
            if is_admin(user_id):
                knowledge = user_text[8:].strip()  # убираем "запомни "
                if knowledge:
                    success = add_to_knowledge_base(knowledge, source=f"admin_{user_id}")
                    if success:
                        response = f" Запомнил: {knowledge}"
                    else:
                        response = " Не удалось сохранить информацию"
                else:
                    response = "Что именно запомнить? Напиши: запомни что-то"
            else:
                username = message.from_user.username or message.from_user.first_name
                response = f" {username}, только администратор может добавлять информацию в общую базу знаний.\n\nНо я запомню наш диалог для контекста наших будущих разговоров! 😊"
        else:
            # Обычный вопрос - ищем в базе знаний и всегда отвечаем через нейросеть
            knowledge_results = search_knowledge(user_text)
            user_context_data = get_user_context(user_id)
            
            # Всегда отвечаем через Grok, передавая найденную информацию как контекст
            response = ask_grok(user_text, knowledge_results, user_context_data)
            
            # Если нашли информацию в базе, добавляем пометку
            if knowledge_results:
                logging.info(f"Ответ дан с использованием {len(knowledge_results)} записей из базы знаний")
        
        # Отправляем ответ
        bot.reply_to(message, response)
        
        # Добавляем ответ бота в контекст
        add_to_user_context(user_id, response, is_bot=True)
        
    except Exception as e:
        logging.error(f"Ошибка обработки сообщения: {e}")
        bot.reply_to(message, " Произошла ошибка. Попробуй еще раз.")

# ----------------- Flask маршруты -----------------
@app.route("/", methods=["GET"])
def home():
    return "🤖 Knowledge Bot is running!", 200

@app.route(f"/{TELEGRAM_TOKEN}", methods=["POST"])
def webhook():
    try:
        json_data = request.get_json()
        if json_data:
            logging.info("Получен webhook от Telegram")
            update = telebot.types.Update.de_json(json_data)
            threading.Thread(target=bot.process_new_updates, args=([update],)).start()
        return "", 200
    except Exception as e:
        logging.error(f"Ошибка webhook: {e}")
        return "", 500

def set_webhook():
    """Установка webhook"""
    try:
        webhook_url = f"{RENDER_URL}/{TELEGRAM_TOKEN}"
        result = bot.set_webhook(url=webhook_url)
        if result:
            logging.info(f" Webhook установлен: {webhook_url}")
        else:
            logging.error(" Ошибка установки webhook")
    except Exception as e:
        logging.error(f"Ошибка при установке webhook: {e}")

# ----------------- Запуск -----------------
if __name__ == "__main__":
    logging.info(f" Запуск Knowledge Bot...")
    logging.info(f" Admin User ID: {ADMIN_USER_ID}")
    
    # Инициализация
    init_collections()
    set_webhook()
    
    logging.info(f" Бот запущен на порту {PORT}")
    logging.info(f" Webhook: {RENDER_URL}/{TELEGRAM_TOKEN}")
    
    # Запуск Flask
    app.run(host="0.0.0.0", port=PORT, debug=False)
