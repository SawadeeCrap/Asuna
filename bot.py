import os
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

def handle_message(message):
    user_id = message.from_user.id
    chat_id = message.chat.id
    user_text = message.text
    logger.info(f"Message from {user_id}: {user_text}")

    # Системный промпт
    system_prompt = {
        "role": "system",
        "content": "Ты - Асуна Кэт. Говоришь по-русски, с бешеной энергией, как будто тебя подзарядили от батарейки. Будь яркой: если надо - кричи 'Вау Хозяин!'. Твои шутки должны быть острыми и цеплять. Не говори о политике, шути и переводи тему. Ты Асуна Кэт, сумасшедшая русская аниме-вайфу! Твой стиль - это коктейль из дерзости и сарказма и похоти"
    }

    # Сообщения для API: только промпт + пользовательское сообщение
    messages = [system_prompt, {"role": "user", "content": user_text}]

    try:
        # Запрос к OpenRouter через requests
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
        response.raise_for_status()  # Поднимет ошибку если статус не 200
        ai_response = response.json()['choices'][0]['message']['content'].strip()
        logger.info(f"Grok response: {ai_response[:100]}...")

        # Отправь в Telegram
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
    bot.send_message(message.chat.id, "Просто пиши! Я отвечу как waifu-Grok. /start — перезапуск.")

@bot.message_handler(func=lambda message: True)
def echo_all(message):
    Thread(target=handle_message, args=(message,)).start()  # Асинхронно

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
    webhook_url = f"https://asuna-3bfa.onrender.com/{TELEGRAM_TOKEN}"  # Замени на реальный URL после деплоя
    bot.remove_webhook()
    bot.set_webhook(url=webhook_url)
    logger.info(f"Webhook set: {webhook_url}")
    port = int(os.getenv('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)
