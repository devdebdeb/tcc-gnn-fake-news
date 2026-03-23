import os
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
from torch.nn import Linear
import pandas as pd
import numpy as np
import sys

# Adiciona o path para o Blue Sky src
sys.path.append(os.path.join(os.getcwd(), "Blue Sky", "src"))
from features import text_embedder

def pipeline():
    print("Iniciando Pipeline de Mensuração (Restaurado)...")
    # Lógica de comparação entre modelos treinados e não treinados no Bluesky
    pass

if __name__ == "__main__":
    pipeline()
