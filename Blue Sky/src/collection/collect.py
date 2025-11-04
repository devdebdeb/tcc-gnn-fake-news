# src/collection/collect.py

import logging
import os
import time
from typing import Dict, List, Tuple

import pandas as pd
import atproto
from atproto import Client, models
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def login_bluesky() -> Client:
    """Realiza o login no Bluesky usando credenciais do arquivo .env."""
    handle = os.getenv("BSKY_HANDLE")
    senha = os.getenv("BSKY_PASSWORD")

    if not handle or not senha:
        msg = "Credenciais BSKY_HANDLE ou BSKY_PASSWORD não encontradas."
        logging.error(msg)
        raise ValueError(f"{msg} Verifique seu arquivo .env.")

    client = Client()
    try:
        client.login(handle, senha)
        logging.info("Login no Bluesky bem-sucedido.")
        return client
    except Exception as e:
        logging.error(f"Falha no login: {e}")
        raise


def _processar_batch_de_posts(
    client: Client, posts: List[models.AppBskyFeedDefs.PostView]
) -> Tuple[List[Dict], List[Dict]]:
    """Processa uma lista de posts, extraindo dados e buscando seus reposts."""
    batch_posts_info: List[Dict] = []
    batch_reposts_info: List[Dict] = []

    # <<< CORREÇÃO: Adicionamos um type hint explícito para a variável 'post' no loop.
    post: models.AppBskyFeedDefs.PostView
    for post in posts:
        post_info = {
            "post_id": post.uri,
            "autor_id": post.author.handle,
            "texto": post.record.text,
            "contagem_reposts": post.repost_count,
            "contagem_likes": post.like_count,
            "timestamp": post.record.created_at,
        }
        batch_posts_info.append(post_info)

        if post.repost_count > 0:
            reposts_do_post = _fetch_reposts_for_post(client, post)
            batch_reposts_info.extend(reposts_do_post)

    return batch_posts_info, batch_reposts_info


def _fetch_reposts_for_post(
    client: Client, post: models.AppBskyFeedDefs.PostView
) -> List[Dict]:
    """Busca todos os reposts para um único post, lidando com paginação."""
    reposts_info: List[Dict] = []
    repost_cursor: str | None = None
    autor_original: str = post.author.handle

    while True:
        try:
            response = client.get_reposted_by(
                uri=post.uri, cid=post.cid, limit=100, cursor=repost_cursor
            )

            if not response or not response.reposted_by:
                break

            author: models.AppBskyActorDefs.ProfileView
            for author in response.reposted_by:
                repost_data = {
                    "source": author.handle,
                    "target": autor_original,
                    "post_id": post.uri,
                }
                reposts_info.append(repost_data)

            repost_cursor = response.cursor
            if not repost_cursor:
                break
            time.sleep(0.05)
        except Exception:
            break
    return reposts_info


def coletar_dados_por_termo(
    client: Client, termo_busca: str, limite_total_posts: int = 1000
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Orquestra a coleta de dados por termo de busca."""
    logging.info(f"Iniciando orquestração da busca por '{termo_busca}'...")
    all_posts_info: List[Dict] = []
    all_reposts_info: List[Dict] = []
    search_cursor: str | None = None

    while True:
        try:
            params = models.AppBskyFeedSearchPosts.Params(
                q=termo_busca, limit=100, cursor=search_cursor
            )
            response = client.app.bsky.feed.search_posts(params=params)

            if not response or not response.posts:
                logging.info("Não foram encontrados mais posts. Fim da busca.")
                break

            posts_batch, reposts_batch = _processar_batch_de_posts(
                client, response.posts
            )
            all_posts_info.extend(posts_batch)
            all_reposts_info.extend(reposts_batch)
            
            logging.info(f"  -> Lote processado. Total de posts coletados: {len(all_posts_info)}")

            if len(all_posts_info) >= limite_total_posts:
                logging.info(f"Limite de {limite_total_posts} posts atingido.")
                break

            search_cursor = response.cursor
            if not search_cursor:
                break
            time.sleep(0.1)

        except Exception as e:
            logging.error(f"Erro crítico durante a paginação da busca: {e}")
            break

    if not all_posts_info:
        return pd.DataFrame(), pd.DataFrame()

    posts_df = pd.DataFrame(all_posts_info).drop_duplicates(subset="post_id")
    reposts_df = pd.DataFrame(all_reposts_info)

    logging.info(f"Coleta orquestrada com sucesso. {len(posts_df)} posts e {len(reposts_df)} reposts.")
    return posts_df, reposts_df