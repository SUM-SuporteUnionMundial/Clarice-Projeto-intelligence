import openai

# Configure sua chave de API do OpenAI
openai.api_key = "sua_chave_aqui"

# Dados de exemplo
user_message = "Olá, como você está?"

# Gere uma resposta usando DialoGPT
response = openai.Completion.create(
  engine="davinci",
  prompt=user_message,
  max_tokens=50
)

bot_reply = response.choices[0].text.strip()
print("Resposta do Bot:", bot_reply)
