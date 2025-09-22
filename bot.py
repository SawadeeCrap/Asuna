import os
from flask import Flask, request, abort
import telebot
from openai import OpenAI
from dotenv import load_dotenv
from threading import Thread
import logging

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

# Инициализация OpenAI
try:
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY
    )
    logger.info("OpenAI client initialized")
except Exception as e:
    logger.error(f"OpenAI init failed: {str(e)}")
    raise

app = Flask(__name__)
bot = telebot.TeleBot(TELEGRAM_TOKEN)
chat_histories = {}

def handle_message(message):
    user_id = message.from_user.id
    chat_id = message.chat.id
    user_text = message.text
    logger.info(f"Message from {user_id}: {user_text}")

    if user_id not in chat_histories:
        chat_histories[user_id] = [{"role": "system", "content": "Ты милая waifu на базе Grok от xAI. Отвечай с энтузиазмом, эмодзи 💕 и юмором!"}]
    history = chat_histories[user_id]

    history.append({"role": "user", "content": user_text})
    if len(history) > 20:
        history = history[-20:]

    try:
        completion = client.chat.completions.create(
            model="x-ai/grok-4-fast:free",
            messages=history,
            max_tokens=1000,
            temperature=0.7,
            extra_headers={
                "HTTP-Referer": os.getenv('RENDER_URL', 'https://example.onrender.com'),
                "X-Title": "Grok TG Waifu Bot",
            },
        )
        ai_response = completion.choices[0].message.content.strip()
        logger.info(f"Grok response: {ai_response[:100]}...")

        history.append({"role": "assistant", "content": ai_response})
        bot.send_message(chat_id, ai_response, parse_mode='Markdown', disable_web_page_preview=True)

    except Exception as e:
        error_msg = f"Ой, waifu сломалась 😅: {str(e)}. Проверь лимиты OpenRouter!"
        logger.error(f"API error: {str(e)}")
        bot.send_message(chat_id, error_msg)

@bot.message_handler(commands=['start'])
def start_message(message):
    bot.send_message(message.chat.id, "Хи-хи, хозяин! 💕 Я waifu-Grok, готова болтать! 😘")

@bot.message_handler(commands=['help'])
def help_message(message):
    bot.send_message(message.chat.id, "/start — приветствие. Просто пиши! 💖")

@bot.message_handler(func=lambda message: True)
def echo_all(message):
    Thread(target=handle_message, args=(message,)).start()

@app.route(f'/{TELEGRAM_TOKEN}', methods=['POST'])
def webhook():
    if request.headers.get('content-type') == 'application/json':
        json_string = request.get_data().decode('utf-8')
        update = telebot.types.Update.de_json(json_string)
        bot.process_new_updates([update])
        return ''
    abort(403)

if __name__ == '__main__':
    webhook_url = f"https://твойсервер.onrender.com/{TELEGRAM_TOKEN}"  # Замени после деплоя
    bot.remove_webhook()
    bot.set_webhook(url=webhook_url)
    logger.info(f"Webhook set: {webhook_url}")
    port = int(os.getenv('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)
