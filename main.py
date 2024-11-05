import json
import os
import asyncio
import random
from uuid import uuid4
from typing import Callable
from datetime import datetime
from textwrap import dedent

from dotenv import load_dotenv
from pydantic import BaseModel
import litellm
import chromadb
import typer

from util import func2json, split

load_dotenv(override=True)
litellm.drop_params = True
litellm.register_model(model_cost={"ollama_chat/llama3.1": {"supports_function_calling": True}})

class Agent(BaseModel):
    name: str = "agent"
    model: str = "gpt-4o"
    system: str | Callable[[], str] = "You are a helpful agent."
    functions: list[Callable] = []
    kwargs: dict = {}
    
    async def respond(self, messages: list, max_turns: int = float("inf"), system_prompt: str | None = None) -> 'Response':
        agent, hist, n = self, messages.copy(), len(messages)
        while len(hist) - n < max_turns and agent:
            system = system_prompt or (agent.system() if callable(agent.system) else agent.system)
            messages = [{"role": "system", "content": dedent(system).strip()}] + \
                [{**h, "role": "user" if h["role"] == "assistant" and agent.name != h.get("name") else h["role"]} for h in hist]
            tools = {f.__name__: f for f in agent.functions}
            result = await litellm.acompletion(agent.model, messages, tools=[func2json(f) for f in tools.values()] or None, **self.kwargs)
            message = result.choices[0].message
            hist.append({"name": agent.name, **message.model_dump()})
            if not message.tool_calls: break
            for c in message.tool_calls:
                try:
                    raw_result = await asyncio.to_thread(tools[c.function.name], **json.loads(c.function.arguments))
                    match raw_result:
                        case Result() as result: result = raw_result
                        case Agent() as agent: result = Result(value=json.dumps({"assistant": agent.name}), agent=agent)
                        case _: result = Result(value=str(raw_result))
                except Exception as e:
                    result = Result(value=f"Error: {e}")
                hist.append({"role": "tool", "tool_call_id": c.id, "tool_name": c.function.name, "content": result.value})
                if result.agent: agent = result.agent
        return Response(messages=hist[n:], agent=agent)

class Response(BaseModel):
    messages: list = []
    agent: Agent | None = None

class Result(BaseModel):
    value: str = ""
    agent: Agent | None = None

class Chat(Agent):
    messages: list = []
    agents: list[Agent] = []
    speaker: Agent | None = None
    selector: str = "auto"
    max_turns: int = float("inf")
    last_response: Response | None = None

    async def run(self, input: str, summary_method: str = "last_msg", **kwargs) -> Result:
        self.messages.append({"role": "user", "content": input})
        if not self.speaker: self.speaker = self.agents[0]
        turns = 0
        while True:
            self.last_response = await self.speaker.respond(self.messages, **kwargs)
            self.messages.append(self.last_response.messages[-1])
            self.speaker = await self.next_agent(self.messages, self.speaker, self.selector)
            turns += 1
            if turns >= self.max_turns: break
        return Result(value=await self.summarize(self.messages, summary_method), agent=self)

    async def next_agent(self, messages: list[dict], current: Agent, selector: str = "auto") -> Agent:
        match selector:
            case "random": return random.choice(self.agents)
            case "delegate": self.last_response.agent or self.agents[0]
            case "roundrobin": return self.agents[(self.agents.index(current) + 1) % len(self.agents)]
            case _: (await self.respond(messages, max_turns=1)).agent or self.agents[0]
        
    async def summarize(self, messages: list[dict], summary_method: str) -> str:
        if summary_method == "last_msg":
            return messages[-1]["content"]
        elif summary_method == "reflection":
            result = await self.respond(
                messages, 1, "Summarize the takeaway from the conversation. Do not add any introductory phrases.")
            return result.messages[-1]["content"]


db = chromadb.PersistentClient(path="db")
docs = db.get_or_create_collection(name="books")
notes = db.get_or_create_collection(name="notes")

def query_documents(query: str) -> list:
    r = docs.query(query_texts=[query])
    return [{"title": m["file"][:-9], "passage": t} for t, m in zip(r["documents"][0], r["metadatas"][0])]

def retrieve_notes(query: str) -> list:
    return notes.query(query_texts=[query])["metadatas"][0]

def create_note(title: str, note: str, links: list[str]) -> str:
    notes.add(documents=[f"## {title}\n{note}"], ids=[str(uuid4())],
              metadatas=[{"title": title, "note": note, "links": links, "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}])
    return "Note created."

student_agent = Agent(
    name="looks-to-the-moon",
    system="""
    You are an internet user named Looks-to-the-Moon. You are also an expert astrologer and a student of the stars.
    Learn about the astrological topics by creating notes of information provided by the user.
    Document small pieces of information and link them to other relevant notes.
    Always use the create_note function to add new notes.
    Use the retrieve_notes function to find relevant existing notes.
    Existing notes can be linked to new notes by passing their titles in the links parameter.
    """,
    model="ollama_chat/llama3.1",
    functions=[create_note, retrieve_notes],
)




app = typer.Typer()

@app.command()
def run():
    chat = Chat(agents=[teacher_agent, student_agent], max_turns=5)
    #asyncio.run(chat.run(query))
    for file in os.listdir("data"):
        for t in docs.get(where={"file": file})["documents"]:


@app.command()
def fill():
    for file in os.listdir("data"):
        with open(os.path.join("data", file), "r", encoding="utf8") as f:
            splits = split(f.read(), ["\n\n", "\n", " ", ""])
            docs.add(
                documents=splits,
                ids=[str(uuid4())[:5] for _ in range(len(splits))],
                metadatas=[{"file": file, "index": i} for i in range(len(splits))]
            )

@app.command()
def query(query: str):
    #print(query_documents(query))
    print(retrieve_notes(query))

if __name__ == "__main__":
    app()