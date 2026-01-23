import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import random

# Importa seus módulos
from src.dataset import MVTecDataset
from src.model import Autoencoder

def plot_anomaly(original, reconstructed, loss_value):
    # Calcula o mapa de erro (Diferença absoluta)
    error_map = torch.abs(original - reconstructed)
    
    # Tira a média dos canais RGB para virar uma escala de cinza (intensidade do erro)
    error_map = error_map.mean(dim=0).detach().cpu().numpy()
    
    # Prepara imagens para plotar (Canais: C,H,W -> H,W,C)
    org_img = original.permute(1, 2, 0).detach().cpu().numpy()
    rec_img = reconstructed.permute(1, 2, 0).detach().cpu().numpy()
    
    # --- PLOTAGEM ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # 1. Original
    axes[0].imshow(org_img)
    axes[0].set_title("Original (Com Defeito)")
    axes[0].axis('off')
    
    # 2. Reconstrução
    axes[1].imshow(rec_img)
    axes[1].set_title("O que o Modelo 'Imaginou'")
    axes[1].axis('off')
    
    # 3. Heatmap (A Diferença)
    # cmap='jet' faz o azul ser erro baixo e vermelho erro alto
    im = axes[2].imshow(error_map, cmap='jet', vmin=0, vmax=0.05) 
    axes[2].set_title(f"Mapa de Calor (MSE: {loss_value:.5f})")
    axes[2].axis('off')
    
    # Adiciona barra de cor para referência
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.show()

def run_inference():
    # --- CONFIGURAÇÕES ---
    DATA_PATH = './data/mvtec_anomaly_detection'
    CATEGORY = 'screw'
    MODEL_PATH = f'weights/autoencoder_{CATEGORY}.pth'
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- CARREGAR DADOS ---
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    
    # Note o mode='test' para pegar os defeitos
    test_dataset = MVTecDataset(DATA_PATH, CATEGORY, transform=transform, mode='test')
    
    # --- CARREGAR MODELO ---
    model = Autoencoder().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval() # Modo de avaliação (congela dropout, batchnorm, etc)
    
    print("🔎 Buscando uma anomalia para analisar...")

    # Vamos procurar aleatoriamente até achar uma imagem com rótulo 1 (defeito)
    while True:
        idx = random.randint(0, len(test_dataset)-1)
        image, label = test_dataset[idx]
        
        if label == 1: # Achamos um defeito!
            break
    
    # Prepara a imagem para o modelo (adiciona dimensão do batch: [1, 3, 128, 128])
    image = image.unsqueeze(0).to(device)
    
    # Inferência
    with torch.no_grad():
        reconstructed = model(image)
        criterion = nn.MSELoss()
        loss = criterion(reconstructed, image)
    
    print(f"✅ Anomalia encontrada! Erro de reconstrução: {loss.item():.5f}")
    
    # Plota o resultado (removendo dimensão do batch)
    plot_anomaly(image[0], reconstructed[0], loss.item())

if __name__ == "__main__":
    run_inference()