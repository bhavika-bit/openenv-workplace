from fastapi import FastAPI
from inference import run
app = FastAPI()
@app.get("/")
def root():
    return {"message": "API running"}    
@app.post("/reset")
def reset():
    return {"status": "reset successful"}
@app.post("/infer")
def infer():
    try:
        result = run()
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}
