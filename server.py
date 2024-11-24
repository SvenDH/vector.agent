from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from graph import KnowledgeGraph

app = FastAPI()
graph = KnowledgeGraph()

@app.get("/")
def read_root():
    return HTMLResponse(content="<h1>Knowledge Graph API</h1>")
