from fastapi import FastAPI
import subprocess

app = FastAPI()

@app.get("/")
def run_env():
    try:
        result = subprocess.check_output(["python", "inference.py"], stderr=subprocess.STDOUT)
        return {"output": result.decode()}
    except subprocess.CalledProcessError as e:
        return {"error": e.output.decode()}