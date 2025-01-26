
import os
import torch
from transformers import pipeline
from dotenv import load_dotenv
import telebot

BOT_TOKEN = "7645332606:AAFA8qHezhAP17UeWOh9AFi1efmIbMhGkks"

# Configurar el bot de Telegram
bot = telebot.TeleBot(BOT_TOKEN)

# Configurar el modelo TinyLlama
pipe = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    torch_dtype=torch.bfloat16,
    #device_map="auto"  # Usa GPU si está disponible
)

def generate_response(user_message: str) -> str:
    
    prompt = (
        f"You are a friendly and concise chatbot. Respond only to the user's message.\n"
        f"User: {user_message}\n"
        "Chatbot:"
    )
    # Generate answer
    outputs = pipe(
        prompt,
        max_new_tokens=250,  # Answer longitude
        do_sample=True,
        temperature=0.5,  
        top_k=30,
        top_p=0.85,
    )
    # Obtain answer
    raw_response = outputs[0]["generated_text"]
    
    if "Chatbot:" in raw_response:
        response = raw_response.split("Chatbot:")[-1].strip()
    else:
        response = raw_response.strip()
    
    
    return response


# Message handler
@bot.message_handler(commands="start")
def send_welcome(message):
    bot.reply_to(message, "Hello I am a Telegram bot. What do you want me to talk about?")

@bot.message_handler(func=lambda message: True)
def handle_message(message):
    user_message = message.text
    response = generate_response(user_message)
    bot.reply_to(message, response)

# Ejecutar el bot
if __name__ == "__main__":
    print("Bot is running. Press Ctrl+C to stop.")
    bot.infinity_polling()


