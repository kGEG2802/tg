import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

def parse_user_intent(text):
    prompt = f"Розпізнай дію користувача: '{text}'. Можливі дії: invoice <номер>, pay <номер>, file <номер>, status unpaid/paid. Поверни об'єкт JSON з action: one of [invoice, pay, file, status] та відповідним invoice або status."

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        result = response['choices'][0]['message']['content']
        return eval(result)  # Очікуємо, що GPT поверне словник
    except Exception as e:
        return {"action": "unknown"}
