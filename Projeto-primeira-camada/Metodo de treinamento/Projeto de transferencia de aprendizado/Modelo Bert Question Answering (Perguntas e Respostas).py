from transformers import BertTokenizer, BertForQuestionAnswering
import torch
from torch.utils.data import DataLoader, Dataset
import time

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        context = self.data[idx]["context"]
        question = self.data[idx]["question"]
        answer = self.data[idx]["answers"]["text"][0]
        
        inputs = self.tokenizer.encode_plus(
            question, context, add_special_tokens=True, padding="max_length", truncation=True, max_length=512
        )
        
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        # Encontra o índice do token '[SEP]' no input_ids
        sep_index = input_ids.index(self.tokenizer.sep_token_id)
        
        # Codifica a resposta para obter os índices dos tokens
        answer_ids = self.tokenizer.encode(answer, add_special_tokens=False)
        
        # Encontra o índice da primeira ocorrência da resposta nos input_ids após o token '[SEP]'
        subsequence = input_ids[sep_index+1:]
        if answer_ids[0] in subsequence:
            start_idx = subsequence.index(answer_ids[0]) + sep_index + 1
            end_idx = start_idx + len(answer_ids) - 1
        else:
            start_idx = end_idx = 0  # Ou outro valor de sua escolha se a resposta não for encontrada

        
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "start_positions": torch.tensor(start_idx),
            "end_positions": torch.tensor(end_idx)
        }

# Suponha que dataset seja sua variável contendo os dados
train_dataset = CustomDataset(dataset["train"])
valid_dataset = CustomDataset(dataset["validation"])

# Crie dataloaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False)

# Carregar o modelo BERT pré-treinado
model = BertForQuestionAnswering.from_pretrained("bert-base-uncased")

# Configurar otimizador e função de perda
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
loss_fn = torch.nn.CrossEntropyLoss()

# Escolha de treinar um novo modelo ou carregar um existente
user_choice = input("Deseja treinar um novo modelo (Digite 'novo') ou carregar um modelo existente (Digite 'existente')? ")

if user_choice.lower() == "novo":
    num_epochs = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    start_time = time.time()
    total_batches = len(train_loader) * num_epochs
    interval = 300  # Intervalo em segundos (5 minutos)

    for epoch in range(num_epochs):
        total_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            start_positions = batch["start_positions"].to(device)
            end_positions = batch["end_positions"].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Monitoramento a cada 5 minutos
            if time.time() - start_time >= interval:
                elapsed_time = time.time() - start_time
                completed_batches = epoch * len(train_loader) + batch_idx + 1
                completion_percentage = (completed_batches / total_batches) * 100
                print(f"Tempo decorrido: {elapsed_time:.2f} segundos")
                print(f"Porcentagem de conclusão: {completion_percentage:.2f}%")
                print(f"Perda média até agora: {total_loss / completed_batches:.4f}")
                print("----------------------------------------")
                start_time = time.time()

        print(f"Época {epoch+1}/{num_epochs}, Perda: {total_loss/len(train_loader)}")

    print("Treinamento concluído.")
    
else:
    model_path = input("Digite o caminho para o modelo existente: ")
    model = BertForQuestionAnswering.from_pretrained(model_path)
    print("Modelo carregado.")

# Avaliação do desempenho final
correct_predictions = 0
total_predictions = 0

model.eval()
with torch.no_grad():
    for batch in valid_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        start_positions = batch["start_positions"].to(device)
        end_positions = batch["end_positions"].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
        loss = outputs.loss

        start_logits = outputs.start_logits
        end_logits = outputs.end_logits
        predicted_start = torch.argmax(start_logits, dim=1)
        predicted_end = torch.argmax(end_logits, dim=1)

        correct_predictions += torch.sum(predicted_start == start_positions).item()
        correct_predictions += torch.sum(predicted_end == end_positions).item()
        total_predictions += len(start_positions) + len(end_positions)

accuracy = correct_predictions / total_predictions
print(f"Acurácia final: {accuracy*100:.2f}%")
