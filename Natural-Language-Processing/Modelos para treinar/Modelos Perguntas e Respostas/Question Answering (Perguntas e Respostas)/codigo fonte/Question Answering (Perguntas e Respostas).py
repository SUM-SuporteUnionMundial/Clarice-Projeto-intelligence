import openai

# Configure sua chave de API do OpenAI
openai.api_key = "sua_chave_aqui"

# Dados de exemplo
context = "Exemplo de contexto..."
question = "Qual é a resposta para a pergunta?"

# Faça a previsão usando GPT-3
response = openai.Completion.create(
  engine="text-davinci-003",
  prompt=f"Contexto: {context}\nPergunta: {question}\nResposta:",
  temperature=0.6,
  max_tokens=50
)

print("Resposta:", response.choices[0].text.strip())
