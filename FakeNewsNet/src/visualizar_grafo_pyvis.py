from pyvis.network import Network

def visualizar_grafo_pyvis(G, nome_arquivo="grafo_fakenewsnet.html"):
    """
    Cria uma visualização interativa do grafo com Pyvis.
    
    Colore os nós baseado nos atributos 'type' e 'label'.
    
    Argumentos:
        G (nx.DiGraph): O grafo NetworkX criado na Etapa 1.
        nome_arquivo (str): Nome do arquivo HTML de saída.
    """
    
    print(f"Iniciando visualização com Pyvis... (Salvando em {nome_arquivo})")
    
    net = Network(height="800px", width="100%", directed=True, notebook=True, cdn_resources='in_line')
    
    # 1. Definir cores baseado nos atributos do nó
    cores = {
        'usuario': '#4B0082',  # Índigo/Azul escuro
        'noticia_real': '#008000', # Verde
        'noticia_fake': '#FF0000'  # Vermelho
    }
    
    # 2. Iterar sobre os nós do NetworkX e adicioná-los ao Pyvis
    for node, attrs in G.nodes(data=True):
        if attrs['type'] == 'usuario':
            cor = cores['usuario']
            titulo = f"Usuário: {node}"
            tamanho = 10 # Usuários menores
        
        elif attrs['type'] == 'noticia':
            titulo = f"Notícia ({attrs['label']}): {attrs.get('title', 'Sem título')}"
            tamanho = 25 # Notícias maiores
            if attrs['label'] == 'real':
                cor = cores['noticia_real']
            else:
                cor = cores['noticia_fake']
        
        net.add_node(node, label=str(node), color=cor, size=tamanho, title=titulo)
        
    # 3. Adicionar as arestas
    net.add_edges(G.edges())
    
    # 4. Configurar física (como você fez no Puxarmsgs)
    net.show_buttons(filter_=['physics'])
    
    # 5. Salvar o grafo
    net.save_graph(nome_arquivo)
    print(f"Visualização salva! Abra o arquivo '{nome_arquivo}' no seu navegador.")
    
    # Se estiver no Jupyter, pode exibir direto:
    # return net.show(nome_arquivo)


# --- Exemplo de como usar ---
# (Assumindo que você já rodou a Etapa 1 e tem o objeto 'G')

visualizar_grafo_pyvis(G, nome_arquivo="buzzfeed_grafo.html")