# 🎓 Planejamento Estratégico do TCC: Detecção de Fake News via GNN no Bluesky

Este documento serve como o roteiro definitivo para a conclusão do Trabalho de Conclusão de Curso (TCC). Ele abrange desde a evolução técnica do sistema até a redação acadêmica final.

---

## 🛠️ 1. Evolução Técnica e Engenharia de Dados
### 💨 Otimização e Performance
- [ ] `[Engenharia/Arquitetura]` **Data Lake Local:** Migrar o processamento do Mega Dataset (21GB) para um formato colunar (`Parquet`) e um banco de dados indexado (`SQLite?`), eliminando o uso de memória excessivo de CSVs.
- [ ] `[Engenharia/Arquitetura]` **Processamento Assíncrono:** Implementar `Celery` ou `FastAPI Background Tasks` para que a coleta e análise de grafos grandes não trave a interface do usuário.
- [ ] `[Engenharia/Arquitetura]` **Dockerização:** Criar um `Dockerfile` e `docker-compose.yaml` para que o orientador/avaliador possa rodar o projeto completo com um único comando.

### 🧠 Refino da Inteligência Artificial
- [ ] `[Engenharia/Arquitetura]` **Arquiteturas Avançadas:**
    - Testar **GAT (Graph Attention Networks)**: Permite que a IA dê pesos diferentes para seguidores mais "influentes" na propagação da notícia.
    - Testar **GraphSAGE**: Melhor para lidar com grafos que a IA nunca viu antes (indutivo).
- [ ] `[Engenharia/Arquitetura]` **Explainable AI (XAI):** Implementar o `GNNExplainer` para mostrar ao usuário *quais conexões* no grafo foram decisivas para classificar a notícia como fake.
- [ ] `[Frontend]` **Visualização de Grafos:** Integrar `react-force-graph` no portal web para visualização interativa das bolhas de propagação.

---

## 📊 2. Rigor Científico e Experimentação
- [ ] `[Análise de Dados]` **O Problema do "Cold Start":** Analisar como a falta de histórico no Bluesky (rede nova) afeta a precisão comparado ao Twitter (base UPFD).
- [ ] `[Análise de Dados]` **Avaliação Estatística:** 
    - Realizar testes de significância (t-test) entre os modelos.
    - Criar matrizes de confusão detalhadas por domínio (Política vs Saúde vs Tecnologia).
- [ ] `[Análise de Dados]` **Ablation Study:** Testar o modelo *apenas* com texto, *apenas* com grafo, e com ambos. Provar que o Grafo realmente ajuda na acurácia.

---

## 📝 3. Guia de Redação do Relatório (ABNT/Manual da Instituição)
### `[Escrita]` Capítulo 1: Introdução
- **Contextualização:** A era da pós-verdade e a saturação de análise de texto simples.
- **Objetivos:** Detectar fake news através do comportamento de rede (quem compartilha).

### `[Escrita]` Capítulo 2: Fundamentação Teórica
- **NLP e Embeddings:** Transformação de texto em vetores (BERT/Sentence-Transformers).
- **Teoria de Grafos:** O que são nodos (usuários) e arestas (interações) no contexto social.
- **Deep Learning on Graphs:** Explicação intuitiva da convolução em grafos.

### `[Escrita]` Capítulo 3: Metodologia (O seu diferencial)
- **O Dataset Bluesky:** Como foi feita a coleta via AT Protocol.
- **Arquitetura do Sistema:** Desenhar um diagrama do fluxo de dados (Frontend <-> FastAPI <-> GCN).

### `[Escrita]` Capítulo 4: Resultados e Discussão
- **Duelo de Modelos:** Comparar o modelo treinado em dados antigos (Twitter) com o modelo nativo (Bluesky).
- **Limitações:** Discutir a dificuldade de rotulagem manual.

---

## ⚖️ 4. Ética e Responsabilidade
- [ ] `[Escrita / Análise de Dados]` **LGPD e Privacidade:** Documentar que o sistema utiliza apenas dados públicos (posts e likes) e que não armazena informações sensíveis de perfis privados.
- [ ] `[Análise de Dados]` **Viés Algorítmico:** Discutir se a IA tende a classificar certas bolhas ideológicas como fake com mais frequência.

---

## 📂 5. Preparação para a Defesa
- [ ] `[Escrita]` **Apresentação Visual:** Criar slides focados em visualização de grafos (os professores amam ver os desenhos das redes).
- [ ] `[Frontend]` **Demo ao Vivo:** Garantir que o portal web esteja hospedado (ex: Vercel/Render) ou fácil de rodar localmente.

---
> **Dica Extra:** Mantenha os logs de treinamento. Se a IA atingir 90% de acurácia, tire um print do terminal. Isso é prova de resultado!
