import os

from kitaru import KitaruClient

EXEC_ID = "" or os.getenv("EXEC_ID")  # Replace with your execution ID


client = KitaruClient()

replay = client.executions.replay(exec_id=EXEC_ID, from_="run_generation_pipeline")
print("Replay ID:", replay.exec_id)
