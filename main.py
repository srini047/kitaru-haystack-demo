from kitaru import flow, checkpoint
import kitaru

from haystack import Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.dataclasses import ChatMessage
from haystack_integrations.components.generators.cohere import (
    CohereChatGenerator,
)
from haystack.utils import Secret


# Sample Retriever Checkpoint
@checkpoint
def retrieve(query: str):
    return [
        {
            "content": """
Haystack is an open-source framework for building search systems, question answering, and retrieval-augmented generation applications.
It provides pipelines and components for document retrieval, semantic search, and chat-based interfaces.
Haystack supports connectors to document stores, retrievers like BM25 and dense embeddings, and integrates with large language models.
"""
        },
        {
            "content": """
Key Haystack features include:
- Modular pipelines
- Prompt builders for chat-style generation
- Integration with LLM providers such as Cohere, OpenAI, and Hugging Face
- Support for document stores, retrievers, and generators
"""
        },
    ]


# Haystack Pipeline Checkpoint
@checkpoint
def run_rag_pipeline(query: str, documents: list):
    pipeline = Pipeline()

    prompt_builder = ChatPromptBuilder(
        template=[
            ChatMessage.from_system("Answer only using the provided context."),
            ChatMessage.from_user(
                """
Context:
{% for doc in documents %}
{{ doc.content }}
{% endfor %}

Question:
{{ query }}
"""
            ),
        ]
    )

    generator = CohereChatGenerator(
        model="command-a-plus-05-2026",
        api_key=Secret.from_env_var("COHERE_API_KEY"),
    )

    pipeline.add_component("prompt_builder", prompt_builder)
    pipeline.add_component("generator", generator)

    pipeline.connect(
        "prompt_builder.prompt",
        "generator.messages",
    )

    result = pipeline.run(
        {
            "prompt_builder": {
                "query": query,
                "documents": documents,
            }
        }
    )

    return result["generator"]["replies"][0].text


# Flow
@flow
def rag_flow(query: str):
    documents = retrieve(query)

    answer = run_rag_pipeline(
        query=query,
        documents=documents,
    )

    return answer


# Driver code
if __name__ == "__main__":
    client = kitaru.KitaruClient()

    execution = rag_flow.run(query="What are the payment retry rules?")

    print("Execution ID:", execution.exec_id)
    print("Status:", execution.status)

    # Replay from pipeline checkpoint
    replay = client.executions.replay(
        execution.exec_id,
        from_="run_rag_pipeline",
    )

    # # Override the query input for the replay
    # execution_id = "a64825b4-0687-4003-9f6b-6cbfc393075e"
    # replay = client.executions.replay(
    #     execution_id,
    #     from_="retrieve",
    #     query="What are the refund rules?",
    # )

    print("Replay ID:", replay.exec_id)
