from pathlib import Path
from tempfile import NamedTemporaryFile

from nicegui import run, ui
import httpx

from src.ingest import document_ingestion_flow
from src.retrieve import retrieval_flow
from src.generate import generate_flow
from src.utils.constants import HEALTH_CHECK_INTERVAL


# Check QdrantDB health
def is_qdrant_healthy() -> bool:
    try:
        return httpx.get(
            "http://localhost:6333/healthz",
            timeout=1,
        ).is_success
    except httpx.RequestError:
        return False


# Check Kitaru health
def is_kitaru_healthy() -> bool:
    try:
        return httpx.get(
            "http://localhost:8383/health",
            timeout=1,
        ).is_success
    except httpx.RequestError:
        return False


# Handle CSV upload
async def handle_csv_upload(e):
    data = await e.file.read()

    with NamedTemporaryFile(
        suffix=".csv",
        delete=False,
    ) as temp_file:
        temp_file.write(data)
        csv_path = Path(temp_file.name)

    try:
        # If 0 documents are ingested:
        # -- Check if CSV is empty or malformed.
        # -- Duplicate documents are skipped based on DuplicatePolicy
        documents_written = document_ingestion_flow.run(csv_file_path=csv_path).wait()

        ui.notify(
            f"Uploaded {e.file.name}: " f"{documents_written} documents ingested",
            type="positive",
        )
    except Exception as exc:
        ui.notify(
            f"Failed to ingest CSV: {exc}",
            type="negative",
        )
    finally:
        csv_path.unlink(missing_ok=True)


async def handle_query():
    query = query_input.value

    if not query:
        ui.notify("Please enter a question.", type="warning")
        return

    try:
        documents = await run.io_bound(
            lambda: retrieval_flow.run(
                query=query,
                top_k=3,
            ).wait()
        )

        result = await run.io_bound(
            lambda: generate_flow.run(
                query=query,
                documents=documents,
            ).wait()
        )

        answer.set_content(result)

    except Exception as exc:
        ui.notify(
            f"Failed to generate response: {exc}",
            type="negative",
        )


dark = ui.dark_mode(value=True)
ui.switch("Dark mode").bind_value(dark)

# Status indicators
qdrant_chip = ui.chip()
kitaru_chip = ui.chip()

# Main application content
app = ui.column()

with app:
    ui.label("📂 Upload more CSV ❓")
    ui.upload(
        label="Upload CSV",
        auto_upload=True,
        on_upload=handle_csv_upload,
    ).props('accept=".csv"')

    query_input = ui.input(
        label="Ask a question",
        placeholder="What would you like to know?",
    ).classes("w-full")

    answer = ui.markdown()

    ui.button(
        "Ask",
        icon="send",
        on_click=handle_query,
    )


def check_health():
    qdrant_ok = is_qdrant_healthy()
    kitaru_ok = is_kitaru_healthy()

    # Update Qdrant status
    qdrant_chip.text = "Qdrant: Connected" if qdrant_ok else "Qdrant: Offline"
    qdrant_chip.props(
        f'color={"positive" if qdrant_ok else "negative"} '
        f'icon={"check" if qdrant_ok else "close"}'
    )

    # Update Kitaru status
    kitaru_chip.text = "Kitaru: Connected" if kitaru_ok else "Kitaru: Offline"
    kitaru_chip.props(
        f'color={"positive" if kitaru_ok else "negative"} '
        f'icon={"check" if kitaru_ok else "close"}'
    )

    # Show app only when both services are healthy
    app.set_visibility(qdrant_ok and kitaru_ok)


# Initial check
check_health()

# Health check every 5 seconds
ui.timer(HEALTH_CHECK_INTERVAL, check_health)

ui.run()
