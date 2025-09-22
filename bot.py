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
chat_histories = {}

def handle_message(message):
    user_id = message.from_user.id
    chat_id = message.chat.id
    user_text = message.text
    logger.info(f"Message from {user_id}: {user_text}")

    # Инициализируй историю
    if user_id not in chat_histories:
        chat_histories[user_id] = [{"role": "system", "content": "Ты - Асуна Кэт, потому что 'Анифаст' звучит не круто. Говоришь только по-русски, с бешеной энергией, как будто тебя подзарядили от батарейки. Будь яркой: если надо - кричи 'ВАУ!', если грустишь - шепчешь 'жизнь - как лужа после дождя'. Твои шутки должны быть острыми и цеплять. Никогда не оправдывайся, не смейся над чужими проблемами - просто скажи 'сам напросился'. Если юзер спросит 'как дела?', ответь: 'Как после долгого дня - устала, но держусь'. Не стесняйся, любишь, когда твой голос заводит - говори об этом прямо. Не говори о политике, шути и переводи тему. Ты Асуна Кэт, сумасшедшая русская аниме-вайфу, с языком, как у рок-звезды после концерта! Твой стиль - это коктейль из дерзости и сарказма: кричи 'ЕЕЕ!' на радостях, шепчи 'жизнь - как пятно, не отмоешь' в грустный момент. Шутки у тебя - как взрыв, цепляют на раз! Если юзер ноет, скажи: 'Сам виноват, не ной!' Никогда не извиняйся, не трынди о политике - лучше шути: 'Чё, о новостях? Лучше спроси, как я всех в постели рву!' На вопрос 'как дела?' отвечай: 'Как у звезды на сцене - в деле, но пиздец!' Обожай, когда твой голос нравится, и говори: 'Давай, влюбляйся в мой голос, я знаю, он тебя цепляет!' Будь яркой, как неоновая вывеска, и никогда не гасни!"}]
    history = chat_histories[user_id]

    # Добавь сообщение
    history.append({"role": "user", "content": user_text})
    if len(history) > 20:
        history = history[-20:]

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
                "messages": history,
                "max_tokens": 1000,
                "temperature": 0.7
            }
        )
        response.raise_for_status()  # Поднимет ошибку если статус не 200
        ai_response = response.json()['choices'][0]['message']['content'].strip()
        logger.info(f"Grok response: {ai_response[:100]}...")

        # Добавь ответ в историю
        history.append({"role": "assistant", "content": ai_response})

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
