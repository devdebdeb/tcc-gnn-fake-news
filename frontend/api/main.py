import os
import sys

# PRIMEIRO: importar torch e torch_geometric da venv
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool

# DEPOIS: adicionar os caminhos do projeto
raiz = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
api_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(api_dir)
sys.path.append(os.path.join(raiz, "Blue Sky", "src"))

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pydantic import BaseModel
from dotenv import load_dotenv
import pandas as pd
import numpy as np

# Banco de dados Local
from database import get_db, AnaliseHistory

# Importando do projeto principal
from collection import collect
from features import text_embedder

app = FastAPI(title="Truth GNN Analytics - Backend", description="Motor de detecções em Real-time")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GCNClassifier(torch.nn.Module):
    def __init__(self, num_node_features: int, num_classes: int, hidden_channels: int = 64):
        super(GCNClassifier, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index)
        h = global_mean_pool(x, batch)
        x = F.dropout(h, p=0.5, training=self.training)
        out = self.lin(x)
        return out, h

num_features = 768
num_classes = 2

# Lazy loading dos modelos para o FastAPI
model_upfd = GCNClassifier(768, 2)
model_bs_ctrl = GCNClassifier(768, 2)
model_bs_cetico = GCNClassifier(768, 2)
model_bs_ex_cetico = GCNClassifier(768, 2)

def load_models():
    # Load UPFD
    peso_upfd_path = os.path.join(raiz, "mesclagem", "pesos_gcn.pth")
    if os.path.exists(peso_upfd_path):
        model_upfd.load_state_dict(torch.load(peso_upfd_path, map_location='cpu'))
    
    # Load Controlado
    peso_ctrl_path = os.path.join(raiz, "mesclagem", "pesos_bs_ctrl.pth")
    if os.path.exists(peso_ctrl_path):
        model_bs_ctrl.load_state_dict(torch.load(peso_ctrl_path, map_location='cpu'))
        
    # Load Cetico
    peso_cetico_path = os.path.join(raiz, "mesclagem", "pesos_bs_cetico.pth")
    if os.path.exists(peso_cetico_path):
        model_bs_cetico.load_state_dict(torch.load(peso_cetico_path, map_location='cpu'))
        
    # Load Exageradamente Cetico
    peso_ex_cetico_path = os.path.join(raiz, "mesclagem", "pesos_bs_ex_cetico.pth")
    if os.path.exists(peso_ex_cetico_path):
        model_bs_ex_cetico.load_state_dict(torch.load(peso_ex_cetico_path, map_location='cpu'))
    
    model_upfd.eval()
    model_bs_ctrl.eval()
    model_bs_cetico.eval()
    model_bs_ex_cetico.eval()

load_models()
load_dotenv(os.path.join(raiz, ".env"))
try:
    bsky_client = collect.login_bluesky()
except Exception:
    bsky_client = None

class RequestURL(BaseModel):
    url: str

import re, json

@app.post("/api/analyze")
async def analyze_link(req: RequestURL, db: Session = Depends(get_db)):
    post_text = "Texto não extraído."
    interacoes = 5
    
    # Tentativa de Scrape na API AtProto
    if bsky_client:
        try:
            match = re.search(r'profile/([^/]+)/post/([^/]+)', req.url)
            if not match:
                raise Exception("URL inválida")
            handle = match.group(1)
            rkey = match.group(2)
            
            if not handle.startswith("did:"):
                res = bsky_client.com.atproto.identity.resolve_handle({'handle': handle})
                did = res.did
            else:
                did = handle
                
            uri = f"at://{did}/app.bsky.feed.post/{rkey}"
            thread = bsky_client.app.bsky.feed.get_post_thread({'uri': uri, 'depth': 10})
            post_text = thread.thread.post.record.text
            
            if hasattr(thread.thread, 'replies') and thread.thread.replies:
                interacoes = len(thread.thread.replies)
        except Exception as e:
            post_text = f"Simulação local devido a erro na extração: {str(e)[:50]}"
    else:
        post_text = "Scraper inativo (Credenciais BSKY ausentes). Modo Simulação."

    # Gerar Embeddings usando BERT (768-D)
    df_temp = pd.DataFrame([{'texto': post_text}])
    df_temp = text_embedder.gerar_embeddings_de_texto(df_temp)
    emb = df_temp['embedding'].iloc[0]
    
    x_raiz = torch.tensor(emb, dtype=torch.float)
    num_nodos = 1 + interacoes
    x = torch.stack([x_raiz for _ in range(num_nodos)])
    
    if interacoes > 0:
        sources = [0] * interacoes
        targets = list(range(1, num_nodos))
        edge_index = torch.tensor([sources, targets], dtype=torch.long)
    else:
        edge_index = torch.tensor([[], []], dtype=torch.long)
        
    data = Data(x=x, edge_index=edge_index, batch=torch.zeros(num_nodos, dtype=torch.long))
    
    # Pesos (UPFD < Ctrl < Cetico < Exa_Cetico) -> A pedido do usuario
    weights = [0.10, 0.20, 0.30, 0.40] 
    models = [model_upfd, model_bs_ctrl, model_bs_cetico, model_bs_ex_cetico]
    names = ['UPFD_Baseline', 'Bluesky_Controlado', 'Bluesky_Cetico', 'Bluesky_Exa_Cetico']
    
    probs = []
    with torch.no_grad():
        for m in models:
            out, _ = m(data.x, data.edge_index, data.batch)
            prob = torch.softmax(out, dim=1)[:, 1].item()
            probs.append(prob)
            
    # Ensemble Ponderado
    ensemble_score = sum(w * p for w, p in zip(weights, probs))
    is_fake = bool(ensemble_score >= 0.5)
    
    # Histórico no Painel DB
    historico = AnaliseHistory(
        url_bsky=req.url, 
        texto_resumo=post_text[:200], 
        tamanho_grafo=num_nodos,
        pred_upfd="Fake" if probs[0] >= 0.5 else "Real",
        cert_upfd=float(probs[0]),
        pred_bsky="Fake" if is_fake else "Real",
        cert_bsky=float(ensemble_score),
        heuristica_final=float(ensemble_score)
    )
    db.add(historico)
    db.commit()
    
    # Salvar Links Testados Suspeitos em JSON
    if is_fake:
        fakes_file = os.path.join(api_dir, "fakes_testados.json")
        fakes_data = []
        if os.path.exists(fakes_file):
            with open(fakes_file, 'r', encoding='utf-8') as f:
                try: fakes_data = json.load(f)
                except: pass
        
        breakdown = {n: round(p, 4) for n, p in zip(names, probs)}
        fakes_data.append({
            "url": req.url,
            "texto": post_text,
            "score_veredicto": round(ensemble_score, 4),
            "modelos": breakdown
        })
        
        with open(fakes_file, 'w', encoding='utf-8') as f:
            json.dump(fakes_data, f, indent=4, ensure_ascii=False)
            
    return {
        "status": "success",
        "url": req.url,
        "texto": post_text,
        "is_fake": is_fake,
        "ensemble_score": ensemble_score,
        "model_breakdown": [{"model": n, "prob_fake": p, "weight": w} for n, p, w in zip(names, probs, weights)],
        "nodes": num_nodos
    }

@app.get("/api/history")
async def get_history(db: Session = Depends(get_db)):
    return db.query(AnaliseHistory).order_by(AnaliseHistory.id.desc()).limit(10).all()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
