# Detenção de Fake News com Redes Neuronais de Grafos (GNNs)

Este repositório contém o código-fonte desenvolvido para o nosso Trabalho de Conclusão de Curso (TCC). O objetivo principal do projeto é aplicar técnicas de *Deep Learning* em grafos para identificar a propagação de desinformação (*Fake News*) em redes sociais, baseando-se nas metodologias do curso CS224W da Universidade de Stanford.

## 📖 Sobre o Projeto

A deteção de *fake news* é frequentemente abordada apenas pela análise de texto. Este projeto eleva a análise ao incorporar a **topologia da rede de propagação**. 
O modelo utiliza o *benchmark* **UPFD (User Preference-aware Fake News Detection)** com a base de dados PolitiFact para analisar árvores de propagação de notícias, classificando o grafo inteiro como "Real" ou "Falso".

O histórico do projeto também inclui pipelines de extração de dados da rede social Bluesky para construção de grafos de interação (através da API `atproto` e `networkx`).

## 🧠 Arquitetura do Modelo

A arquitetura central é uma **Graph Convolutional Network (GCN)** construída com `PyTorch Geometric`.
A rede é composta por:
- Três camadas de convolução em grafos (`GCNConv`) para capturar as relações estruturais até 3 saltos de distância.
- Uma camada de agregação (*Global Mean Pooling*) que transforma os *embeddings* dos nós individuais num único vetor latente que representa a árvore de propagação da notícia.
- Uma camada linear totalmente conectada para a classificação binária final.

## 🛠️ Tecnologias e Dependências

O projeto é construído sobre um ecossistema robusto de análise de dados e *Machine Learning*, garantindo a reprodutibilidade e legibilidade do código através de ferramentas como `black` e `flake8`.

**Principais Bibliotecas:**
- **Machine Learning & GNNs:** `torch`, `torch_geometric`
- **Processamento de Texto (Features):** `sentence-transformers`
- **Análise de Redes:** `networkx`, `python-louvain`
- **Manipulação de Dados:** `pandas`, `pyarrow`
- **Visualização:** `matplotlib`, `seaborn`, `pyvis`

## 🚀 Como Executar

### 1. Instalação do Ambiente

Recomenda-se a utilização de um ambiente virtual (ex: `venv` ou `conda`). Para instalar as dependências necessárias, execute:

```bash
pip install -r requirements.txt