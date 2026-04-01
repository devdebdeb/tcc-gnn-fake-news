# CLAUDE.md — Truth GNN Analytics: Fake News Detection via GNN no Bluesky

## O que é esse projeto

TCC de graduação que detecta fake news na rede social Bluesky usando **Graph Neural Networks (GNNs)**. A inovação é combinar **embeddings BERT do texto** com a **topologia do grafo de propagação** (quem compartilha o quê) para classificar posts como reais ou falsos. Há um ensemble ponderado de 4 modelos GCN com pesos diferentes de ceticismo.

---

## Mapa do Projeto

```
tcc-gnn-fake-news/
├── frontend/
│   ├── api/                    ← FastAPI (porta 8000) — inferência e histórico
│   │   ├── main.py            ← API principal: /api/analyze e /api/history
│   │   ├── database.py        ← SQLite via SQLAlchemy (tabela historico_analises)
│   │   └── fakes_testados.json← Posts fake detectados
│   └── web/                    ← Next.js 14 (porta 3000) — UI estilo Apple/glassmorphism
│       └── src/app/page.tsx   ← Interface principal (input URL, resultados, histórico)
│
├── Blue Sky/src/
│   ├── collection/collect.py  ← Login AT Protocol + busca de posts por termo
│   └── features/text_embedder.py ← Gera embeddings 768-D com paraphrase-multilingual-mpnet-base-v2
│
├── mesclagem/                  ← Scripts de treinamento + 4 modelos .pth
│   ├── pesos_gcn.pth          ← Modelo UPFD baseline (Twitter/PolitiFact)
│   ├── pesos_bs_ctrl.pth      ← Bluesky controlado
│   ├── pesos_bs_cetico.pth    ← Bluesky cético (6x penalidade falso positivo)
│   └── pesos_bs_ex_cetico.pth ← Bluesky extremamente cético
│
├── UPFD-GCN/                   ← Pipeline completo de treino + visualização t-SNE
├── UPFD/                       ← Ferramentas de visualização de grafos (Pyvis)
├── politifact/                 ← Dataset PolitiFact em formato PyTorch Geometric
│   └── processed/bert/        ← train.pt, val.pt, test.pt (embeddings BERT pré-calculados)
│
├── 14669616.zip               ← Mega Dataset Bluesky 21GB (NÃO incluir no Docker!)
├── requirements.txt           ← ~105 deps Python (torch 2.9, torch-geometric 2.7, etc.)
├── iniciar_tcc.bat            ← Launcher Windows (abre dois terminais)
└── .env                       ← BSKY_HANDLE e BSKY_PASSWORD (não commitar!)
```

---

## Arquitetura do Sistema

```
URL do Bluesky
     ↓
FastAPI /api/analyze
     ↓
[1] Extrai texto via AT Protocol (atproto lib)
     ↓
[2] BERT embedding 768-D (paraphrase-multilingual-mpnet-base-v2)
     ↓
[3] Constrói grafo estrela: nó raiz = post, filhos = interações
     ↓
[4] Inferência nos 4 modelos GCN (3 camadas GCNConv + global_mean_pool + Linear)
     ↓
[5] Ensemble ponderado: UPFD(0.10) + Ctrl(0.20) + Cético(0.30) + ExCético(0.40)
     ↓
[6] Threshold 0.5 → is_fake: bool
     ↓
SQLite (histórico) + JSON (fakes detectados) + Response ao frontend
```

## Arquitetura do Modelo GCN

```python
GCNClassifier(num_features=768, num_classes=2, hidden_channels=64):
  GCNConv(768 → 64) + ReLU
  GCNConv(64 → 64) + ReLU
  GCNConv(64 → 64)
  global_mean_pool → representação nível grafo
  Dropout(0.5) + Linear(64 → 2)
  → softmax → prob_fake
```

---

## Os 3 Tasks Pendentes

### 1. Background Tasks (FastAPI)
**Problema:** `/api/analyze` roda BERT + GCN de forma síncrona, travando a UI.

**Abordagem:** FastAPI `BackgroundTasks` (sem Redis, sem Celery — mais simples pro TCC).
- POST `/api/analyze` → retorna `task_id` imediatamente com `status: "processing"`
- Background processa BERT + GCN + salva no DB com `task_id`
- GET `/api/result/{task_id}` → retorna resultado ou `status: "processing"`
- Frontend faz polling a cada 1s até completar
- Adicionar campo `task_id` e `status` na tabela SQLite

### 2. Reduzir 14669616.zip (21GB)
**Conteúdo do zip:**
- `user_posts.tar.gz` — 18.5GB (vilão principal)
- `interactions.csv.gz` — 993MB
- `graphs.tar.gz` — 851MB
- `followers.csv.gz` — 468MB
- `feed_posts_likes.tar.gz` — 34MB
- `feed_posts.tar.gz` — 15.6MB
- `feed_bookmarks.csv` — 527KB
- `scripts.tar.gz` — 16KB

**Abordagem:** Script Python que:
1. Extrai os arquivos seletivamente
2. Amostra N% dos dados (ex: 10-20% do `user_posts` e `interactions`)
3. Converte CSV → Parquet (compressão snappy, ~10x menor)
4. Cria `dataset_mini.zip` < 2GB

### 3. Dockerização
**Estrutura:**
- `Dockerfile.api` — Python 3.11-slim, instala requirements, copia código
- `Dockerfile.web` — Node 20-alpine, build do Next.js
- `docker-compose.yml` — orquestra api + web, volume para modelos .pth
- `.dockerignore` — exclui 14669616.zip, node_modules, __pycache__, .venv
- Ajustar `database.py` para usar path relativo ao container

**IMPORTANTE:** O Mega Dataset (14669616.zip) NUNCA vai no Docker. Fica como volume externo opcional.

---

## Como Rodar Localmente (hoje)

```bash
# Backend
cd frontend/api
../../.venv/Scripts/python.exe -m uvicorn main:app --reload
# ou simplesmente:
iniciar_tcc.bat
```

## Dependências Críticas

- `torch==2.9.0` + `torch-geometric==2.7.0` — pesados, mas necessários
- `sentence-transformers==5.1.2` — carrega modelo BERT no startup (~400MB)
- `atproto==0.0.63` — cliente AT Protocol para Bluesky
- `fastapi` + `uvicorn` + `sqlalchemy` — stack do backend
- `next@14.2.3` + `react@18` — stack do frontend

## Notas de Implementação

- O `text_embedder` carrega `SentenceTransformer` uma vez e fica em memória (correto, não recarregar)
- Os 4 modelos `.pth` ficam em `mesclagem/` e são carregados no startup do FastAPI
- O `database.py` usa caminho absoluto hardcoded (PRECISA mudar pro Docker)
- O `.env` com credenciais Bluesky não está no git (correto)
- `fakes_testados.json` é append-only, pode crescer indefinidamente (considerar limpeza)
