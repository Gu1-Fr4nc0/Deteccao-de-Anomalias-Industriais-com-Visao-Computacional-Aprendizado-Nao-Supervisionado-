# Detecção de Anomalias Industriais com Visão Computacional (Aprendizado Não Supervisionado)

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/Computer%20Vision-Industrial%20Inspection-green.svg" alt="CV">
</div>

---

## 🎯 Objetivo do Projeto

Este projeto implementa um sistema de **Detecção de Anomalias Visuais para Controle de Qualidade Industrial**, utilizando técnicas de **Aprendizado Não Supervisionado** aplicadas à Visão Computacional.

O objetivo central é responder à seguinte pergunta:

> **É possível detectar defeitos de fabricação treinando um modelo apenas com imagens de peças sem defeito?**

Esse cenário reflete uma limitação real da indústria, onde imagens defeituosas são raras, caras ou inexistentes durante a fase de treinamento.

---

## 🧠 Abordagem Utilizada

A solução proposta utiliza um **Autoencoder Convolucional**, treinado exclusivamente com imagens de peças consideradas normais (*good samples*).

O modelo aprende uma representação latente da normalidade. Durante a inferência, defeitos são identificados por meio do **erro de reconstrução**, uma vez que padrões anômalos não são bem reconstruídos pelo modelo.

A anomalia é quantificada e localizada utilizando o mapa de erro absoluto entre a imagem original e sua reconstrução.

---

## 📊 Resultados (Prova de Conceito)

O modelo foi treinado utilizando a categoria **`screw` (parafusos)** do dataset padrão da indústria **MVTec Anomaly Detection**.

A figura abaixo apresenta um exemplo de inferência em uma peça defeituosa, evidenciando uma ranhura no corpo do parafuso.

> **Interpretação do heatmap:** > Áreas em tons mais quentes indicam regiões onde o modelo apresentou maior erro de reconstrução, sugerindo a presença de anomalias visuais.

<div align="center">
  <img src="https://github.com/user-attachments/assets/2cad4a36-505e-476a-8a6f-1e452458536d" alt="Resultado da Detecção de Anomalia" width="800">
</div>

---

## 📈 Avaliação Qualitativa

- **Robustez:** Imagens normais apresentam baixo erro de reconstrução.
- **Localização:** Defeitos estruturais são destacados de forma consistente nos mapas de erro.
- **Sensibilidade:** O método se mostrou capaz de identificar defeitos locais, mesmo sem ter visto exemplos defeituosos no treinamento.

> 💡 Este projeto tem caráter de **prova de conceito**, priorizando a interpretabilidade e a validação da hipótese de *Unsupervised Learning*.

---

## 🛠️ Arquitetura e Tecnologias

- **Modelo:** Autoencoder Convolucional (CNN)
- **Framework:** PyTorch
- **Dataset:** [MVTec Anomaly Detection](https://www.mvtec.com/company/research/datasets/mvtec-ad) — Categoria `screw`
- **Função de perda:** Mean Squared Error (MSE)
- **Pré-processamento:** Redimensionamento para 128×128 e normalização
- **Aceleração:** GPU via CUDA (quando disponível)

---

## 📂 Estrutura do Repositório

```bash
├── data/                  # Dataset MVTec AD (Gitignored)
├── src/
│   ├── model.py           # Arquitetura do Autoencoder
│   ├── dataset.py         # Dataset e DataLoader customizados
│   └── utils.py           # Funções auxiliares (visualização e métricas)
├── weights/               # Pesos do modelo treinado (.pth)
├── train.py               # Script de treinamento
├── inference.py           # Inferência e geração de heatmaps
└── requirements.txt       # Dependências
```
## 🚀 Como Executar
1. Instalação
Clone este repositório e instale as dependências:

```bash

# Clone o repositório
git clone [https://github.com/Gu1-Fr4nc0/anomalia-mvtec.git](https://github.com/Gu1-Fr4nc0/anomalia-mvtec.git)
cd anomalia-mvtec

# Instale as dependências
pip install -r requirements.txt
```
2. Preparação do Dataset
Baixe a categoria screw do dataset MVTec Anomaly Detection e organize a pasta data/ da seguinte forma:

```Plaintext
data/
└── mvtec_anomaly_detection/
    └── screw/
        ├── train/
        └── test/
```
3. Treinamento (Opcional)
Para treinar o modelo do zero:

```bash

python train.py
O modelo treinado será salvo na pasta weights/ após 100 épocas.
```
4. Inferência
Para testar a detecção em uma imagem de teste aleatória:


```bash

python inference.py

```
O script seleciona uma amostra defeituosa e exibe o mapa de anomalia correspondente.

## ⚠️ Limitações Conhecidas
Definição de Limiar: O método é sensível à escolha do threshold para classificar o que é defeito ou ruído.

Defeitos Sutis: Autoencoders muito potentes podem acabar "reconstruindo" defeitos pequenos, mascarando a anomalia.

Performance: Não houve otimização específica para latência de tempo real ou dispositivos de borda (Edge Devices) nesta etapa.

## 🔮 Próximos Passos
[ ] Comparação com métodos baseados em embeddings (k-NN, Mahalanobis).

[ ] Avaliação quantitativa com métricas de ROC-AUC.

[ ] Testes de robustez com outras categorias do MVTec AD.

[ ] Deploy do modelo via API (FastAPI/Flask).

<div align="center">

Desenvolvido por Guilherme Pança Franco Projeto aplicado em Engenharia de Computação e Visão Computacional

</div>
