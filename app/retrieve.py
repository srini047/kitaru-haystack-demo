from kitaru import flow, checkpoint

from haystack.dataclasses import Document
from haystack.dataclasses.sparse_embedding import SparseEmbedding
from haystack.components.embedders import (
    SentenceTransformersTextEmbedder,
    SentenceTransformersSparseTextEmbedder,
)
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack_integrations.components.retrievers.qdrant import (
    QdrantHybridRetriever,
)

from constants import (
    EMBEDDING_MODEL,
    SPARSE_EMBEDDING_MODEL,
    EMBEDDING_DIM,
    QDRANT_INDEX,
)

dense_embedder = SentenceTransformersTextEmbedder(model=EMBEDDING_MODEL)
dense_embedder.warm_up()

sparse_embedder = SentenceTransformersSparseTextEmbedder(model=SPARSE_EMBEDDING_MODEL)
sparse_embedder.warm_up()

document_store = QdrantDocumentStore(
    host="localhost",
    port=6333,
    index=QDRANT_INDEX,
    embedding_dim=EMBEDDING_DIM,
    recreate_index=False,
    use_sparse_embeddings=True,
    return_embedding=True,
)

retriever = QdrantHybridRetriever(document_store=document_store)


@checkpoint
def embed_query(query: str) -> list[float]:
    """
    Embeds the query using the dense embedding model

    :param query: The input query string to be embedded
    :return: A list of floats representing the dense embedding of the query
    """
    result = dense_embedder.run(text=query)

    return result["embedding"]


@checkpoint
def sparse_embed_query(query: str) -> SparseEmbedding:
    """
    Embeds the query using the sparse embedding model

    :param query: The input query string to be embedded
    :return: A SparseEmbedding object representing the sparse embedding of the query
    """
    result = sparse_embedder.run(text=query)

    return result["sparse_embedding"]


@checkpoint(cache=False)
def retrieve_documents(
    embedding: list[float],
    sparse_embedding: SparseEmbedding,
    top_k: int,
) -> list[Document]:
    """
    Retrieves documents from the Qdrant document store based on the provided embeddings

    :param embedding: The dense embedding of the query
    :param sparse_embedding: The sparse embedding of the query
    :param top_k: The number of top documents to retrieve

    :return: A list of Document objects representing the retrieved documents
    """
    result = retriever.run(
        query_embedding=embedding,
        query_sparse_embedding=sparse_embedding,
        top_k=top_k,
    )

    return result["documents"]


@flow
def retrieval_flow(
    query: str,
    top_k: int = 3,
) -> list[Document]:
    """
    Retrieves documents from the Qdrant document store based on the provided query

    :param query: The input query string for which relevant documents are to be retrieved
    :param top_k: The number of top documents to retrieve (default is 3)

    :return: A list of Document objects representing the retrieved documents
    """

    dense_embedding = embed_query(query=query)
    sparse_embedding = sparse_embed_query(query=query)

    return retrieve_documents(
        embedding=dense_embedding,
        sparse_embedding=sparse_embedding,
        top_k=top_k,
    )
