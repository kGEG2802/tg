import os
from telegram.ext import Updater, MessageHandler, Filters, CommandHandler
from dotenv import load_dotenv
from chatgpt_parser import parse_user_intent
from google_sheet import handle_invoice_request, update_invoice_status, list_by_status
from google_drive import send_invoice_file
from logger import log_query

load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
ALLOWED_USER_ID = int(os.getenv("ALLOWED_USER_ID"))

def start(update, context):
    user_id = update.effective_user.id
    if user_id != ALLOWED_USER_ID:
        update.message.reply_text("⛔ Доступ заборонено.")
        return
    update.message.reply_text("👋 Вітаю! Я бот для роботи з інвойсами.")

def handle_message(update, context):
    user_id = update.effective_user.id
    if user_id != ALLOWED_USER_ID:
        update.message.reply_text("⛔ Доступ заборонено.")
        return

    message = update.message.text
    intent = parse_user_intent(message)
    response = "Невідомий запит."

    if intent["action"] == "invoice":
        response = handle_invoice_request(intent["invoice"])
    elif intent["action"] == "pay":
        response = update_invoice_status(intent["invoice"])
    elif intent["action"] == "status":
        response = list_by_status(intent["status"])
    elif intent["action"] == "file":
        send_invoice_file(update, intent["invoice"])
        log_query(user_id, message)
        return

    update.message.reply_text(response)
    log_query(user_id, message)

def main():
    updater = Updater(TELEGRAM_TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_message))

    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    main()
