# Telegram Invoice Bot

## 📋 Можливості:
- Витягує дані з Google Sheets за номером інвойсу
- Змінює статус оплати
- Завантажує файли з Google Drive
- Обробляє запити через ChatGPT
- Логує запити в Google Sheet

## 🔧 Запуск
1. Створи `.env` на основі `.env.example`
2. Поклади `credentials.json` у корінь
3. Запусти: `python bot.py`

Або:

```bash
docker build -t invoice-bot .
docker run -d --env-file .env -v $(pwd):/app invoice-bot
```
