import pandas as pd
import networkx as nx

def criar_grafo_bipartido(caminho_real_news, caminho_fake_news, caminho_edges_txt):
    """
    Cria um grafo bipartido direcionado (DiGraph) do Fakenewsnet.
    
    O grafo conecta Nós de 'Usuário' a Nós de 'Notícia' com base 
    no arquivo de arestas.
    
    Argumentos:
        caminho_real_news (str): Caminho para o CSV de notícias reais.
        caminho_fake_news (str): Caminho para o CSV de notícias falsas.
        caminho_edges_txt (str): Caminho para o .txt com as conexões (ex: BuzzFeedNewsUser.txt).
        
    Retorna:
        nx.DiGraph: O grafo bipartido pronto para análise.
    """
    
    print("Iniciando a criação do grafo...")
    
    # 1. Carregar e rotular os dados das notícias
    df_real = pd.read_csv(caminho_real_news)
    df_fake = pd.read_csv(caminho_fake_news)
    
    # Adicionamos o "ground truth" (nosso label)
    df_real['label'] = 'real'
    df_fake['label'] = 'fake'
    
    df_noticias = pd.concat([df_real, df_fake])
    
    # 2. Carregar as arestas (usuário -> notícia)
    # O arquivo .txt parece não ter cabeçalho, vamos nomear as colunas
    df_edges = pd.read_csv(caminho_edges_txt, sep='\t', header=None, names=['user_id', 'news_id'])
    
    # 3. Inicializar o Grafo Direcionado (como no CS224W)
    G = nx.DiGraph()
    
    # 4. Adicionar Nós de Notícia com atributos (CS224W - Node, célula 9)
    print(f"Adicionando {len(df_noticias)} nós de notícia...")
    for _, row in df_noticias.iterrows():
        # Usamos o 'id' da notícia como identificador do nó
        G.add_node(
            row['id'], 
            type='noticia',         # Atributo para o tipo (crucial para GNN e Pyvis)
            label=row['label'],     # Atributo para o label (fake/real)
            title=row['title']      # Atributo para o título (útil no Pyvis)
        )
        
    # 5. Adicionar Nós de Usuário com atributos
    usuarios_unicos = df_edges['user_id'].unique()
    print(f"Adicionando {len(usuarios_unicos)} nós de usuário...")
    for user_id in usuarios_unicos:
        G.add_node(
            user_id,
            type='usuario'          # Atributo para o tipo
        )
        
    # 6. Adicionar Arestas Direcionadas (Usuário -> Notícia)
    print(f"Adicionando {len(df_edges)} arestas de propagação...")
    for _, row in df_edges.iterrows():
        # Garantir que ambos os nós existem no grafo antes de ligar
        if G.has_node(row['user_id']) and G.has_node(row['news_id']):
            G.add_edge(row['user_id'], row['news_id'])
            
    print("--- Grafo criado com sucesso! ---")
    print(f"Total de Nós: {G.number_of_nodes()}")
    print(f"Total de Arestas: {G.number_of_edges()}")
    
    return G

# --- Exemplo de como usar (ajuste os caminhos) ---
# (Estou usando os nomes do BuzzFeed como exemplo)

PATH_REAL = "FakeNewsNet\FNN Dataset\PolitiFact_real_news_content.csv"
PATH_FAKE = "FakeNewsNet\FNN Dataset\PolitiFact_fake_news_content.csv"
PATH_EDGES = "FakeNewsNet\FNN Dataset\PolitiFactNewsUser.txt"

G = criar_grafo_bipartido(PATH_REAL, PATH_FAKE, PATH_EDGES)