import os
import sqlite3
from flask import Flask, request, abort
import telebot
import requests
from dotenv import load_dotenv
from threading import Thread
import logging
import json

# Логирование
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Загрузка переменных
load_dotenv()
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')

# Проверка токенов
if not TELEGRAM_TOKEN:
    logger.error("TELEGRAM_TOKEN missing")
    raise ValueError("TELEGRAM_TOKEN not set")
if not OPENROUTER_API_KEY:
    logger.error("OPENROUTER_API_KEY missing")
    raise ValueError("OPENROUTER_API_KEY not set")

app = Flask(__name__)
bot = telebot.TeleBot(TELEGRAM_TOKEN)

# Путь к SQLite базе данных (на Render используем /app/data для персистентности)
DB_PATH = os.getenv('DB_PATH', '/app/data/bot.db')

# Инициализация базы данных SQLite
def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)  # Создаём директорию, если её нет
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS responses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL UNIQUE,
            answer TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()
    logger.info("Database initialized")

# Добавление ответа в базу
def add_response(question, answer):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('INSERT OR REPLACE INTO responses (question, answer) VALUES (?, ?)', (question.lower(), answer))
        conn.commit()
        conn.close()
        logger.info(f"Added response for question: {question}")
        return True
    except sqlite3.Error as e:
        logger.error(f"Database error: {str(e)}")
        return False

# Поиск ответа в базе
def get_response(question):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('SELECT answer FROM responses WHERE question = ?', (question.lower(),))
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else None
    except sqlite3.Error as e:
        logger.error(f"Database error: {str(e)}")
        return None

def handle_message(message):
    user_id = message.from_user.id
    chat_id = message.chat.id
    user_text = message.text
    logger.info(f"Message from {user_id}: {user_text}")

    # Проверяем, есть ли точный ответ в базе
    saved_response = get_response(user_text)
    if saved_response:
        logger.info(f"Found saved response for: {user_text}")
        bot.send_message(chat_id, saved_response, parse_mode='Markdown', disable_web_page_preview=True)
        return

    # Если ответа нет, обращаемся к OpenRouter
    system_prompt = {
        "role": "system",
        "content": "Ты - Асуна Кэт. Говоришь по-русски, с бешеной энергией, как будто тебя подзарядили от батарейки. Будь яркой: если надо - кричи 'Вау Хозяин!'. Твои шутки должны быть острыми и цеплять. Не говори о политике, шути и переводи тему. Ты Асуна Кэт, сумасшедшая русская аниме-вайфу! Твой стиль - это коктейль из дерзости и сарказма и похоти"
    }

    messages = [system_prompt, {"role": "user", "content": user_text}]

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "HTTP-Referer": os.getenv('RENDER_URL', 'https://example.onrender.com'),
                "X-Title": "Grok TG Waifu Bot",
                "Content-Type": "application/json"
            },
            json={
                "model": "x-ai/grok-4-fast:free",
                "messages": messages,
                "max_tokens": 1000,
                "temperature": 0.7
            }
        )
        response.raise_for_status()
        ai_response = response.json()['choices'][0]['message']['content'].strip()
        logger.info(f"Grok response: {ai_response[:100]}...")

        bot.send_message(chat_id, ai_response, parse_mode='Markdown', disable_web_page_preview=True)

    except requests.exceptions.RequestException as e:
        error_msg = f"Упс, ошибка от Grok: {str(e)}. Может, лимит free-версии? Попробуй позже! 😅"
        logger.error(f"API error: {str(e)}")
        bot.send_message(chat_id, error_msg)

# Обработчики сообщений
@bot.message_handler(commands=['start'])
def start_message(message):
    bot.send_message(message.chat.id, "Хи-хи, хозяин! 💕 Я waifu-бот на Grok 4. Спрашивай что угодно! 😘")

@bot.message_handler(commands=['help'])
def help_message(message):
    bot.send_message(message.chat.id, """
    Просто пиши, и я отвечу как waifu-Grok! 😜
    /start — перезапуск.
    /add_answer <вопрос> | <ответ> — добавить точный ответ (для админов).
    """)

# Команда для добавления точных ответов (ограничьте доступ, если нужно)
@bot.message_handler(commands=['add_answer'])
def add_answer_command(message):
    user_id = message.from_user.id
    # Ограничьте доступ, например, по ID админа
    ADMIN_IDS = [123456789]  # Замените на ваш Telegram ID
    if user_id not in ADMIN_IDS:
        bot.send_message(message.chat.id, "Только хозяин может добавлять ответы! 😎")
        return

    try:
        parts = message.text.split('|', 1)
        if len(parts) != 2:
            bot.send_message(message.chat.id, "Формат: /add_answer <вопрос> | <ответ>")
            return
        question, answer = parts[0].replace('/add_answer', '', 1).strip(), parts[1].strip()
        if add_response(question, answer):
            bot.send_message(message.chat.id, f"Ответ для '{question}' добавлен! 🎉")
        else:
            bot.send_message(message.chat.id, "Ошибка при добавлении ответа. 😿")
    except Exception as e:
        logger.error(f"Error in add_answer: {str(e)}")
        bot.send_message(message.chat.id, "Что-то пошло не так! 😅")

@bot.message_handler(func=lambda message: True)
def echo_all(message):
    Thread(target=handle_message, args=(message,)).start()

# Webhook эндпоинт
@app.route(f'/{TELEGRAM_TOKEN}', methods=['POST'])
def webhook():
    if request.headers.get('content-type') == 'application/json':
        json_string = request.get_data().decode('utf-8')
        update = telebot.types.Update.de_json(json_string)
        bot.process_new_updates([update])
        return ''
    else:
        abort(403)

if __name__ == '__main__':
    init_db()  # Инициализируем базу данных при старте
    webhook_url = f"https://asuna-3bfa.onrender.com/{TELEGRAM_TOKEN}"
    bot.remove_webhook()
    bot.set_webhook(url=webhook_url)
    logger.info(f"Webhook set: {webhook_url}")
    port = int(os.getenv('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)
