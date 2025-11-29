from fastapi import FastAPI,BackgroundTasks
import os
from dotenv import load_dotenv
import json
load_dotenv()

app = FastAPI()
SECRATE = os.getenv("secrate","test")

@app.post("/solve")
def solve(task):
    print(task)    

@app.post("/test")
def test():
    pass