import streamlit as st
from neo4j import GraphDatabase
import time
from llm import embeddings

# --- 1. Configurações ---
# As credenciais são lidas do Streamlit Secrets
NEO4J_URI = st.secrets["NEO4J_URI"]
NEO4J_USER = st.secrets["NEO4J_USERNAME"]
NEO4J_PASSWORD = st.secrets["NEO4J_PASSWORD"]
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

BATCH_SIZE = 50


driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def fetch_movies_to_embed(tx, limit):
    """Busca um lote de filmes que ainda não têm embedding."""
    query = """
    MATCH (m:Movie)
    WHERE m.plot IS NOT NULL AND m.plotEmbedding IS NULL
    RETURN elementId(m) AS id, m.plot AS plot
    LIMIT $limit
    """
    result = tx.run(query, limit=limit)
    return [{"id": record["id"], "plot": record["plot"]} for record in result]


def update_movie_embeddings(tx, batch_data):
    """Atualiza os nós do Neo4j com os novos embeddings."""
    query = """
    UNWIND $batch AS row
    MATCH (m:Movie) WHERE elementId(m) = row.id
    SET m.plotEmbedding = row.embedding
    """
    tx.run(query, batch=batch_data)


# --- 3. Loop de Processamento em Lote ---
print("Iniciando o processo de backfill de embeddings...")
with driver.session() as session:
    processed_count = 0
    while True:
        # Busca um lote de filmes
        movies = session.execute_read(fetch_movies_to_embed, BATCH_SIZE)

        # Se não houver mais filmes, encerre o loop
        if not movies:
            print("\n✅ Processamento de backfill concluído!")
            print(f"Total de filmes processados: {processed_count}")
            break

        current_batch_size = len(movies)
        processed_count += current_batch_size
        print(f"Processando um lote de {current_batch_size} filmes... (Total: {processed_count})")

        # Extrai apenas os textos dos plots para enviar ao modelo
        plots = [movie['plot'] for movie in movies]

        # Gera os embeddings para o lote de plots
        plot_embeddings = embeddings.embed_documents(plots)

        # Prepara os dados para a atualização no Neo4j
        batch_to_update = []
        for i, movie in enumerate(movies):
            batch_to_update.append({
                "id": movie['id'],
                "embedding": plot_embeddings[i]
            })

        # Executa a escrita no banco de dados
        session.execute_write(update_movie_embeddings, batch_to_update)

        # Pequena pausa para não sobrecarregar as APIs
        time.sleep(1)

driver.close()