from pathlib import Path

from kitaru import checkpoint, flow
from haystack import Pipeline
from haystack.dataclasses import Document
from haystack.components.converters import CSVToDocument
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.preprocessors import (
    DocumentCleaner,
    DocumentSplitter,
)
from haystack.components.writers import DocumentWriter
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack.document_stores.types import DuplicatePolicy

from src.utils.constants import (
    QDRANT_INDEX,
    QDRANT_HOST,
    QDRANT_PORT,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
)


# Haystack Pipeline Checkpoint
@checkpoint
def process_input_pipeline(csv_file_path: Path) -> list[Document]:
    if not csv_file_path.exists():
        raise FileNotFoundError(f"{csv_file_path} does not exist.")

    pipeline = Pipeline()
    pipeline.add_component("converter", CSVToDocument())
    pipeline.add_component("cleaner", DocumentCleaner())
    pipeline.add_component(
        "splitter", DocumentSplitter(split_by="line", split_length=1)
    )

    pipeline.connect("converter.documents", "cleaner.documents")
    pipeline.connect("cleaner.documents", "splitter.documents")

    result = pipeline.run({"converter": {"sources": [csv_file_path]}})

    return result["splitter"]["documents"]


# Haystack Embedding Checkpoint
@checkpoint
def embed_documents(documents: list[Document]) -> list[Document]:
    embedder = SentenceTransformersDocumentEmbedder(model=EMBEDDING_MODEL)
    embedder.warm_up()

    result = embedder.run(documents=documents)

    return result["documents"]


# Haystack Document Write Checkpoint
@checkpoint(cache=False)
def write_documents(documents: list[Document]) -> int:
    document_store = QdrantDocumentStore(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
        index=QDRANT_INDEX,
        recreate_index=False,
        embedding_dim=EMBEDDING_DIM,
        use_sparse_embeddings=True,
    )
    writer = DocumentWriter(document_store=document_store, policy=DuplicatePolicy.SKIP)
    result = writer.run(documents=documents)

    return result["documents_written"]


# Flow
@flow
def document_ingestion_flow(csv_file_path: Path | None) -> int:
    """
    Flow to ingest documents from a CSV file.

    Args:
        csv_file_path (Path | None): The path to the CSV file containing the documents.

    Returns:
        int: The number of documents written to the document store.
    """
    if csv_file_path is None:
        raise RuntimeError("No file path provided.")

    documents = process_input_pipeline(csv_file_path)
    embedded_documents = embed_documents(documents)

    return write_documents(embedded_documents)
