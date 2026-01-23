import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import os

# Importa seus módulos da pasta src
from src.dataset import MVTecDataset
from src.model import Autoencoder

def train():
    # --- CONFIGURAÇÕES (HIPERPARÂMETROS) ---
    DATA_PATH = './data/mvtec_anomaly_detection'
    CATEGORY = 'screw'
    EPOCHS = 100          # Quantas vezes ele vai ver todas as imagens
    BATCH_SIZE = 32       # Quantas imagens processa por vez
    LR = 1e-3             # Learning Rate (velocidade de aprendizado)
    
    # Configura para usar GPU se tiver (NVIDIA), senão vai de CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚙️ Usando dispositivo: {device}")

    # --- PREPARAÇÃO DOS DADOS ---
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    # Carrega APENAS o treino (imagens boas)
    train_dataset = MVTecDataset(DATA_PATH, CATEGORY, transform=transform, mode='train')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # --- MODELO, LOSS E OTIMIZADOR ---
    model = Autoencoder().to(device)
    criterion = nn.MSELoss() # Erro Quadrático Médio (mede a diferença pixel a pixel)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print(f"🚀 Iniciando treinamento com {len(train_dataset)} imagens de parafusos normais...")

    # --- LOOP DE TREINAMENTO ---
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for images, labels in train_loader:
            # Move imagens para GPU/CPU
            images = images.to(device)
            
            # Zerar gradientes
            optimizer.zero_grad()
            
            # Forward (O modelo tenta reconstruir a imagem)
            outputs = model(images)
            
            # Calcula o erro (Entrada vs Saída)
            loss = criterion(outputs, images)
            
            # Backward (Aprende com o erro)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Log a cada 10 épocas para não poluir o terminal
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss/len(train_loader):.6f}")

    # --- SALVAR O MODELO ---
    # Cria pasta weights se não existir
    os.makedirs("weights", exist_ok=True)
    save_path = f"weights/autoencoder_{CATEGORY}.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\n✅ Treinamento concluído! Modelo salvo em: {save_path}")

if __name__ == "__main__":
    train()