from typing import Any

from kitaru import checkpoint, flow

from haystack import Pipeline
from haystack.dataclasses import ChatMessage, Document
from haystack.utils import Secret
from haystack.components.builders import ChatPromptBuilder
from haystack_integrations.components.generators.cohere import CohereChatGenerator


# Generation Checkpoint
@checkpoint
def run_generation_pipeline(query: str, documents: list[Document]) -> str:
    """
    Generates a response to `query`, grounded in `documents`.

    :param query: The user's question.
    :param documents: A list of Document objects to ground the answer in.
    :return: A string containing the generated response.
    """
    pipeline = Pipeline()

    prompt_builder = ChatPromptBuilder(
        template=[
            ChatMessage.from_system("Answer only using the provided context."),
            ChatMessage.from_user("""
Context:
{% for doc in documents %}
{{ doc.content }}
{% endfor %}

Question:
{{ query }}
"""),
        ]
    )

    generator = CohereChatGenerator(
        model="command-a-plus-05-2026",
        api_key=Secret.from_env_var("COHERE_API_KEY"),
    )

    pipeline.add_component("prompt_builder", prompt_builder)
    pipeline.add_component("generator", generator)

    pipeline.connect("prompt_builder.prompt", "generator.messages")

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
def generate_flow(query: str, documents: list[Document]) -> dict[str, Any]:
    """
    Generates a response to `query`, grounded in `documents`.

    :param query: The user's question.
    :param documents: A list of Document objects to ground the answer in.
    :return: A dictionary containing the generated response.
    """
    if not documents:
        raise RuntimeError("No documents provided for generation.")

    response = run_generation_pipeline(query=query, documents=documents)

    return {"response": response}
