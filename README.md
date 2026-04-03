# Instagram Video Transcriber — Telegram Bot

Telegram-бот, который транскрибирует видео из публичных Instagram-профилей с помощью OpenAI Whisper.

## Как работает

1. Пользователь отправляет ссылку на Instagram-профиль
2. Бот парсит профиль и находит последние видео
3. Скачивает каждое видео и извлекает аудио через FFmpeg
4. Отправляет аудио в OpenAI Whisper API
5. Возвращает текстовую транскрипцию в Telegram

## Требования

- Python 3.12+
- FFmpeg
- Telegram Bot Token ([@BotFather](https://t.me/BotFather))
- OpenAI API Key ([platform.openai.com](https://platform.openai.com/api-keys))

## Локальный запуск

```bash
# Клонировать репозиторий
git clone https://github.com/YOUR_USERNAME/insta-transcriber.git
cd insta-transcriber

# Установить зависимости
pip install -r requirements.txt

# Настроить переменные окружения
cp .env.example .env
# Заполнить BOT_TOKEN и OPENAI_API_KEY в .env

# Запустить
export $(cat .env | xargs)
python main.py
```

## Docker

```bash
docker build -t insta-transcriber .
docker run --env-file .env insta-transcriber
```

## Railway

1. Форкните репозиторий
2. Создайте проект на [Railway](https://railway.app)
3. Подключите репозиторий
4. Добавьте переменные `BOT_TOKEN` и `OPENAI_API_KEY` в настройках
5. Деплой произойдёт автоматически

## GitHub Actions

Бот может работать как GitHub Actions workflow:

1. Перейдите в Settings → Secrets and variables → Actions
2. Добавьте секреты `BOT_TOKEN` и `OPENAI_API_KEY`
3. Запустите workflow вручную или через push в `main`

> ⚠️ GitHub Actions имеет лимит 6 часов на задачу.

## Стек

- **aiogram 3** — Telegram Bot Framework (async)
- **instaloader** — парсинг Instagram
- **openai** — Whisper API (speech-to-text)
- **aiohttp** — скачивание видео
- **FFmpeg** — извлечение аудио
