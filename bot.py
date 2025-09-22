import os
from flask import Flask, request, abort
import telebot
from openai import OpenAI
from dotenv import load_dotenv
from threading import Thread

# Загрузи переменные окружения
load_dotenv()
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')

# Инициализация
app = Flask(__name__)
bot = telebot.TeleBot(TELEGRAM_TOKEN)
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# Хранилище истории чата (в памяти; для продакшена — Redis или БД)
chat_histories = {}

def handle_message(message):
    user_id = message.from_user.id
    chat_id = message.chat.id
    user_text = message.text

    # Инициализируй историю, если нет
    if user_id not in chat_histories:
        chat_histories[user_id] = [{"role": "system", "content": "Ты Grok — полезный, остроумный ИИ от xAI. Отвечай кратко, с юмором. Для waifu-стиля: будь милой аниме-вайфу 💕 с эмодзи!"}]
    history = chat_histories[user_id]

    # Добавь сообщение пользователя
    history.append({"role": "user", "content": user_text})

    # Ограничь историю (до 20 сообщений, чтобы не превысить контекст)
    if len(history) > 20:
        history = history[-20:]

    try:
        # Запрос к OpenRouter (Grok 4 Fast free)
        completion = client.chat.completions.create(
            model="x-ai/grok-4-fast:free",  # Твоя модель
            messages=history,
            max_tokens=1000,  # Лимит ответа
            temperature=0.7,  # Креативность
            extra_headers={
                "HTTP-Referer": os.getenv('RENDER_URL', 'https://твойбот.onrender.com'),  # Для трекинга
                "X-Title": "Grok TG Waifu Bot",
            },
        )
        ai_response = completion.choices[0].message.content.strip()

        # Добавь ответ в историю
        history.append({"role": "assistant", "content": ai_response})

        # Отправь в Telegram
        bot.send_message(chat_id, ai_response, parse_mode='Markdown', disable_web_page_preview=True)

    except Exception as e:
        error_msg = f"Упс, ошибка от Grok: {str(e)}. Может, лимит free? Попробуй позже! 😅"
        bot.send_message(chat_id, error_msg)

# Обработчики сообщений
@bot.message_handler(commands=['start'])
def start_message(message):
    welcome = "Привет, хозяин! 💕 Я waifu-бот на базе Grok 4 от xAI. Спрашивай что угодно — шутки, код, советы! 😘"
    bot.send_message(message.chat.id, welcome)

@bot.message_handler(commands=['help'])
def help_message(message):
    bot.send_message(message.chat.id, "Просто пиши! Я отвечу как Grok-вайфу. /start — перезапуск.")

@bot.message_handler(func=lambda message: True)
def echo_all(message):
    Thread(target=handle_message, args=(message,)).start()  # Асинхронно, чтобы не блокировать

# Webhook эндпоинт для Flask
@app.route(f'/{TELEGRAM_TOKEN}', methods=['POST'])  # Безопасный путь с токеном
def webhook():
    if request.headers.get('content-type') == 'application/json':
        json_string = request.get_data().decode('utf-8')
        update = telebot.types.Update.de_json(json_string)
        bot.process_new_updates([update])
        return ''
    else:
        abort(403)

if __name__ == '__main__':
    # Установка webhook (если не вручную)
    webhook_url = f"https://твойсервер.onrender.com/{TELEGRAM_TOKEN}"  # Замени на реальный URL после деплоя
    bot.remove_webhook()
    bot.set_webhook(url=webhook_url)
    print(f"Webhook установлен: {webhook_url}")
    print("Сервер запущен на порту", os.getenv('PORT', 5000))
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
