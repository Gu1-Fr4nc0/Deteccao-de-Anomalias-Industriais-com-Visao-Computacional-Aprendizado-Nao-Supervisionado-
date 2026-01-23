import os
from glob import glob
from torch.utils.data import Dataset
from PIL import Image
import torch

class MVTecDataset(Dataset):
    def __init__(self, root_dir, category, transform=None, mode='train'):
        """
        Args:
            root_dir (string): Diretório raiz onde extraiu os dados (ex: ./data/mvtec_anomaly_detection)
            category (string): A categoria (ex: 'screw')
            transform (callable, optional): Transformações do PyTorch (Resize, ToTensor...)
            mode (string): 'train' (só boas) ou 'test' (boas + defeitos)
        """
        self.transform = transform
        self.mode = mode
        
        # Define o caminho base
        # Se mode='train', buscamos em: root/category/train/good/*.png
        # Se mode='test', buscamos em: root/category/test/**/*/*.png (pega todas as subpastas de defeitos)
        if mode == 'train':
            # No treino, SÓ queremos o que é 'good'
            self.image_paths = glob(os.path.join(root_dir, category, 'train', 'good', '*.png'))
        else:
            # No teste, queremos TUDO (inclusive os defeitos para validar)
            # O uso de recursive=True com ** garante que pegue todas as subpastas (scratch, manipulated, etc)
            search_path = os.path.join(root_dir, category, 'test', '**', '*.png')
            self.image_paths = glob(search_path, recursive=True)
            
        # Verificação de segurança (Engenharia defensiva)
        if len(self.image_paths) == 0:
            raise RuntimeError(f"Nenhuma imagem encontrada em: {os.path.join(root_dir, category, mode)} \nVerifique se o caminho está correto!")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # Carrega imagem e garante RGB (algumas pngs podem ter canal alpha)
        image = Image.open(img_path).convert('RGB')
        
        # Lógica de Rótulo (Label):
        # 0 = Normal (Good)
        # 1 = Anomalia (Qualquer coisa que não esteja na pasta 'good')
        # Obs no Windows: as vezes o path vem com \\, então normalizamos para checar
        is_good = 'good' in img_path.replace("\\", "/")
        label = 0 if is_good else 1
        
        if self.transform:
            image = self.transform(image)
            
        return image, label