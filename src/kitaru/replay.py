from src.generate import generate_flow

EXEC_ID = "" # Replace with your execution ID

replay = generate_flow.replay(exec_id=EXEC_ID, from_="run_generation_pipeline")
