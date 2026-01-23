import matplotlib.pyplot as plt
from torchvision import transforms
from src.dataset import MVTecDataset # Importa sua classe criada

# --- CONFIGURAÇÕES ---
# Garanta que este caminho aponta para onde você extraiu a pasta 'screw'
DATA_PATH = './data/mvtec_anomaly_detection' 
CATEGORY = 'screw'

# --- TESTE ---
def run_test():
    print(f"🔍 Verificando pasta: {DATA_PATH}/{CATEGORY}")
    
    # Simula as transformações que o modelo vai usar
    data_transforms = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    try:
        # Tenta carregar Treino e Teste
        train_ds = MVTecDataset(DATA_PATH, CATEGORY, data_transforms, mode='train')
        test_ds = MVTecDataset(DATA_PATH, CATEGORY, data_transforms, mode='test')

        print("\n✅ SUCESSO! Dataset encontrado.")
        print(f"   -> Imagens de Treino (Só boas): {len(train_ds)}")
        print(f"   -> Imagens de Teste (Boas + Defeitos): {len(test_ds)}")

        # Pega a primeira imagem para ver se abre
        img, label = train_ds[0]
        print(f"\n📸 Teste de imagem:")
        print(f"   -> Shape: {img.shape} (Deve ser 3x128x128)")
        print(f"   -> Label: {label} (Deve ser 0 para treino)")
        
        # Mostra a imagem (se estiver no VS Code ou Notebook)
        plt.imshow(img.permute(1, 2, 0))
        plt.title("Se você está vendo um parafuso, funcionou!")
        plt.show()

    except Exception as e:
        print("\n❌ ERRO: Algo deu errado.")
        print(e)

if __name__ == "__main__":
    run_test()