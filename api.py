from fastapi import FastAPI

app = FastAPI()


@app.post("/generate")
def generate_data(payload: dict):
  # Your processing logic here
  return {"result": "success"}