import openai

# Configure sua chave de API do OpenAI
openai.api_key = "sua_chave_aqui"

# Prompt para geração de texto
prompt = "Escreva um parágrafo sobre o clima de hoje:"

# Gere um parágrafo usando GPT-3
response = openai.Completion.create(
  engine="text-davinci-003",
  prompt=prompt,
  max_tokens=100
)

generated_text = response.choices[0].text.strip()
print("Texto Gerado:", generated_text)
