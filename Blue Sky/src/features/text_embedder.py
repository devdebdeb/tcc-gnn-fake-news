# src/features/text_embedder.py

import logging
import pandas as pd
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def gerar_embeddings_de_texto(
    posts_df: pd.DataFrame, model_name: str = "distiluse-base-multilingual-cased-v1"
) -> pd.DataFrame:
    logging.info(f"Carregando o modelo de linguagem: '{model_name}'...")
    try:
        model = SentenceTransformer(model_name)
    except Exception as e:
        logging.error(f"Não foi possível carregar o modelo SentenceTransformer: {e}")
        raise

    if "texto" not in posts_df.columns:
        logging.error("O DataFrame de posts não contém a coluna 'texto'.")
        raise ValueError("Coluna 'texto' em falta no DataFrame.")

    textos_para_converter = posts_df["texto"].astype(str).fillna("").tolist()

    logging.info(f"Gerando embeddings para {len(textos_para_converter)} textos...")
    embeddings = model.encode(textos_para_converter, show_progress_bar=True)

    df_com_embeddings = posts_df.copy()
    df_com_embeddings["embedding"] = list(embeddings)

    logging.info("Geração de embeddings concluída com sucesso.")
    return df_com_embeddings