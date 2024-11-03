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

from util import function_to_json, split

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
            tools = [function_to_json(f) for f in agent.functions] 
            result = await litellm.acompletion(agent.model, messages, tools=tools or None, **self.kwargs)
            message = result.choices[0].message
            hist.append({"name": agent.name, **message.model_dump()})
            if not message.tool_calls:
                break
            function_map = {f.__name__: f for f in agent.functions}
            for c in message.tool_calls:
                try:
                    func, args = function_map[c.function.name], json.loads(c.function.arguments)
                    raw_result = await asyncio.to_thread(func, **args)
                    match raw_result:
                        case Result() as result: result = raw_result
                        case Agent() as agent: result = Result(value=json.dumps({"assistant": agent.name}), agent=agent)
                        case _: result = Result(value=str(raw_result))
                except Exception as e:
                    result = Result(value=f"Error: {e}")
                hist.append({"role": "tool", "tool_call_id": c.id, "tool_name": func.__name__, "content": result.value})
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

    async def run(self, input: str, summary_method: str = "last_msg", **kwargs) -> Response:
        self.messages.append({"role": "user", "content": input})
        if not self.speaker: self.speaker = self.agents[0]
        turns = 0
        while True:
            response = await self.speaker.respond(self.messages, **kwargs)
            self.messages.append(response.messages[-1])
            self.speaker = await self.next_agent(self.messages, self.speaker)
            turns += 1
            if turns >= self.max_turns: break
        summary = await self.summarize(self.messages, summary_method)
        print(summary)

    async def next_agent(self, messages: list[dict], current: Agent) -> Agent:
        if self.selector == "random":
            return random.choice(self.agents)
        elif self.selector == "delegate":
            return self.last_response.agent or self.agents[0]
        elif self.selector == "roundrobin":
            return self.agents[(self.agents.index(current) + 1) % len(self.agents)]
        return (await self.respond(messages, max_turns=1)).agent or self.agents[0]
        
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

def transfer_back_to_triage():
    """Call this function if a user is asking about a topic that is not handled by the current agent."""
    return triage_agent

def transfer_to_researcher(): return research_agent

def transfer_to_writer(): return writer_agent

research_agent = Agent(
    name="researcher",
    system="""
    Search the library for relevant passages. Always try a diversity of queries to find the most relevant information.
    """,
    functions=[query_documents, transfer_back_to_triage, transfer_to_writer],
)
writer_agent = Agent(
    name="writer-agent",
    system="""
    Document pieces of information that we found and link them to other relevant notes.
    Always use the create_note function to add new notes.
    Use the retrieve_notes function to find relevant existing notes.
    Existing notes can be linked to new notes by passing their titles in the links parameter.
    """,
    functions=[create_note, retrieve_notes, transfer_back_to_triage],
)
triage_agent = Agent(
    name="triage-agent",
    system="""
    Determine which agent is best suited to handle the user's request, and transfer the conversation to that agent.
    Wait for the agent to finish processing the request before making another transfer.
    Transfer to the Research Agent if the user is asking for information.
    Transfer to the Writer Agent if the user is asking for a note to be created.
    """,
    functions=[transfer_to_writer, transfer_to_researcher],
)

teacher_agent = Agent(
    name="teacher",
    system="""
    Teach the user about the requested topics by creating a layout of a lesson and search the library for relevant passages of each topic.
    Always try a diversity of queries to find the most relevant information.
    """,
    model="ollama_chat/llama3.1",
    functions=[query_documents],
)
student_agent = Agent(
    name="student",
    system="""
    Learn about the requested topics by creating notes of information provided by the user/teacher.
    When something is unclear, ask the teacher for more information, else ask to continue the lesson.
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