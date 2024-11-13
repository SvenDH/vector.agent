import json
import os
import asyncio
import random
import copy
from uuid import uuid4
from typing import Callable
from datetime import datetime
from textwrap import dedent

from dotenv import load_dotenv
from pydantic import BaseModel, Field
import litellm
import chromadb
import typer

from util import func2json, split

load_dotenv(override=True)
# Monkey patch for ollama chat
#def _token_counter(**kwargs): return 0
#litellm.token_counter = _token_counter

class Agent(BaseModel):
    name: str = "agent"
    model: str = "gpt-4o-mini"
    system: str | Callable[[], str] = "You are a helpful agent."
    functions: list[Callable] = []
    messages: list[dict] = []
    transforms: list[Callable] = []
    kwargs: dict = {}

    def __hash__(self): return hash(self.name)
    
    async def respond(self, messages: list, max_turns: int = float("inf"), system_prompt: str | None = None) -> 'Response':
        self.messages.extend([{**h, "role": "user"} for h in messages])
        agent, n = self, len(self.messages)
        while len(self.messages) - n < max_turns and agent:
            tools = {f.__name__: f for f in agent.functions}
            system = system_prompt or (agent.system() if callable(agent.system) else agent.system)
            messages = copy.deepcopy(self.messages)
            for t in self.transforms: messages = t(messages)
            messages = [{"content": dedent(system).strip(), "role": "system"}] + messages
            result = await litellm.acompletion(agent.model, messages, tools=[func2json(f) for f in tools.values()] or None, **self.kwargs)
            message = result.choices[0].message
            self.messages.append({"name": agent.name, **message.model_dump()})
            if not message.tool_calls: break
            for c in message.tool_calls:
                try:
                    raw_result = await tools[c.function.name](**json.loads(c.function.arguments))
                    match raw_result:
                        case Result() as result: result = raw_result
                        case Agent() as agent: result = Result(value=json.dumps({"assistant": agent.name}), agent=agent)
                        case _: result = Result(value=str(raw_result))
                except Exception as e:
                    print(f"Error: {e}")
                    result = Result(value=f"Error: {e}")
                self.messages.append({"role": "tool", "tool_call_id": c.id, "tool_name": c.function.name, "content": result.value})
                if result.agent: agent = result.agent
        return Response(messages=self.messages[n:], agent=agent)

    def reset(self): self.messages = []

class Response(BaseModel):
    messages: list = []
    agent: Agent | None = None

class Result(BaseModel):
    value: str = ""
    agent: Agent | None = None

class Chat(BaseModel):
    messages: list[Response] = []
    agents: list[Agent] = []
    manager: Agent | None = None
    speaker: Agent | None = None
    selector: str = "auto"
    max_turns: int = float("inf")
    termination: Callable = Field(default=lambda m: "FINISHED" in m[-1]["content"])

    async def run(self, input: str, summary_method: str = "last_msg", **kwargs) -> Result:
        if not self.speaker: self.speaker = self.agents[0]
        if not self.manager: self.manager = Agent(name="Groupchat_Manager", system="You are the manager of the conversation.")
        turns = 0
        agents = {a: [{"role": "user", "content": input}] for a in self.agents}
        while len(agents) > 0 and turns < self.max_turns:
            self.messages.append(await self.speaker.respond(agents[self.speaker], **kwargs))
            print(f'{self.speaker.name}: {self.messages[-1].messages[-1]["content"]}')
            for a in agents: agents[a].append(self.messages[-1].messages[-1])
            agents[self.speaker] = []
            last_speaker = self.speaker
            self.speaker = await self.next_agent([m.messages[-1] for m in self.messages], list(agents.keys()), self.speaker, self.selector)
            if self.termination(self.messages[-1].messages) and last_speaker in agents:
                del agents[last_speaker]
            turns += 1
        return Result(value=await self.summarize([m.messages[-1] for m in self.messages], summary_method))

    def reset(self):
        self.messages = []
        for agent in self.agents: agent.reset()

    async def next_agent(self, messages: list[dict], agents: list[Agent], current: Agent, selector: str = "auto") -> Agent:
        match selector:
            case "random": return random.choice(agents)
            case "delegate": return self.messages[-1].agent or agents[0]
            case "roundrobin": return agents[(agents.index(current) + 1) % len(agents)]
            case _: return (await self.manager.respond(messages, max_turns=1)).agent or agents[0]
        
    async def summarize(self, messages: list[dict], summary_method: str) -> str:
        if summary_method == "last_msg":
            return messages[-1]["content"]
        elif summary_method == "reflection":
            result = await self.manager.respond(
                messages, 1, "Summarize the takeaway from the conversation. Do not add any introductory phrases.")
            return result.messages[-1]["content"]


db = chromadb.PersistentClient(path="db")
docs = db.get_or_create_collection(name="books")
notes = db.get_or_create_collection(name="notes")

def query_documents(query: str) -> list:
    r = docs.query(query_texts=[query])
    return [{"title": m["file"][:-9], "passage": t} for t, m in zip(r["documents"][0], r["metadatas"][0])]

async def retrieve_notes(query: str) -> list:
    print(f"\033[90m=============\Query:\n{query}\n===============\033[0m")
    return notes.query(query_texts=[query])["metadatas"][0]

async def create_note(content: str) -> str:
    if notes.get(where={"note": content})["documents"]: return "Note already exists."
    r = notes.query(query_texts=[content], n_results=1)["distances"][0]
    if r and r[0] > 1.8: return "Note already exists."
    print(f"\033[90m=============\nNote:\n{content}\n===============\033[0m")
    notes.add(documents=[content], ids=[str(uuid4())], metadatas=[{"note": content, "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}])
    return "Note created."

def add_name(messages: list) -> list:
    for i, m in enumerate(messages):
        if "name" in m and "tool_calls" not in m:
            messages[i]["content"] = f"{m['name']} says: {m['content']}"
    return messages

agent_kwargs = {
    #"model": "ollama_chat/qwen2.5:7b-instruct",
    "functions": [create_note, retrieve_notes],
    #"transforms": [add_name],
}

system_prompt = """
Document small pieces of information. Notes should be concise and focused on key points and self-contained.
You are given a passage from a book to analyze. Create notes on the key points that are relevant to your expertise.
If the passage is part of acknowledgments, table of content, references, prewords, about the author, etc., you can skip it.
When no more notes have to be taken on the passage or you want to skip, answer with a single "FINISHED".
"""

astrologer = Agent(
    name="Expert_Astrologer",
    system="""
    You are an expert astrologer and a student of the stars.
    Learn about the astrological topics by creating notes of information provided by the user.
    """ + system_prompt, **agent_kwargs)
occultist = Agent(
    name="Expert_Occultist",
    system="""
    You are an expert occultist and a seeker of hidden knowledge.
    Learn about the occult topics by creating notes of information provided by the user.
    """ + system_prompt, **agent_kwargs)
symbolist = Agent(
    name="Export_Symbolist",
    system="""
    You are an expert symbolist and a decoder of hidden meanings.
    Learn about the symbolic topics by creating notes of information provided by the user.
    """ + system_prompt, **agent_kwargs)
alchemist = Agent(
    name="Expert_Alchemist",
    system="""
    You are an expert alchemist and a transmuter of knowledge.
    Learn about the alchemical topics by creating notes of information provided by the user.
    """ + system_prompt, **agent_kwargs)
monarchist = Agent(
    name="Expert_Monarchist",
    system="""
    You are an expert monarchist and a keeper of royal knowledge.
    Learn about the monarchical topics by creating notes of information provided by the user.
    """ + system_prompt, **agent_kwargs)
angelologist = Agent(
    name="Expert_Angelologist",
    system="""
    You are an expert angelologist and a scholar of divine beings.
    Learn about the angelic topics by creating notes of information provided by the user.
    """ + system_prompt, **agent_kwargs)
demonologist = Agent(
    name="Expert_Demonologist",
    system="""
    You are an expert demonologist and a master of infernal lore.
    Learn about the demonic topics by creating notes of information provided by the user.
    """ + system_prompt, **agent_kwargs)
hermetist = Agent(
    name="Expert_Hermetist",
    system="""
    You are an expert hermetist and a follower of Hermes Trismegistus.
    Learn about the hermetic topics by creating notes of information provided by the user.
    """ + system_prompt, **agent_kwargs)
gnostic = Agent(
    name="Expert_Gnostic",
    system="""
    You are an expert gnostic and a knower of divine mysteries.
    Learn about the gnostic topics by creating notes of information provided by the user.
    """ + system_prompt, **agent_kwargs)
agents = [astrologer, occultist, symbolist, alchemist, monarchist, angelologist, demonologist, hermetist, gnostic]

TEXT_ANALYSIS_PROMPT = """
You are in a chat with the following experts: {}

The passage is as follows:
{}

Please analyze the passage and create notes on the key points that are relevant to your expertise.
If the passage is part of acknowledgments, table of content, references, prewords, about the author, etc., you can skip it.
Also give me a summary of the notes take. If the passage does not contain any relevant information, you can skip it.
"""

async def analyze_text(passage: str) -> str:
    print(f"\033[90m=============\nPassage:\n{passage}\n===============\033[0m")
    chat = Chat(agents=agents, selector="roundrobin", max_round=12)
    return (await chat.run(
        TEXT_ANALYSIS_PROMPT.format(", ".join([a.name for a in agents]), passage).strip(),
        summary_method="reflection"
    )).value

analyer = Agent(
    name="Analyzer",
    #model="gpt-4o",
    temperature=0.0,
    system="""
    You are an expert text analyzer and a master of information extraction.
    You are given a passage from a book to analyze.
    You direct pieces of the text to the experts for analysis by calling the "analyze_text" tool.
    If the passage is part of acknowledgments, table of content, introductions, references, prewords, about the author, etc., you can skip it.
    When no more notes have to be taken on the passage or you want to skip, answer with a single "FINISHED".
    """,
    functions=[analyze_text]
)

PASSAGE_PROMPT = """
The passage is from: {}

The passage is as follows:
{}
"""

app = typer.Typer()

@app.command()
def run():
    for i, file in enumerate(os.listdir("data")):
        print(i, file[:-9])
        texts = docs.get(where={"file": file})
        part = ""
        chat = Chat(agents=[analyer], max_round=10)
        for t, _ in sorted(zip(texts["documents"], texts["metadatas"]), key=lambda x: x[1]["index"]):
            part += t + "\n\n"
            if len(part) > 10000:
                asyncio.run(chat.run(PASSAGE_PROMPT.format(file[:-9], part)))
                chat.reset()
                part = ""


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

@app.command()
def clear():
    db.delete_collection("notes")

if __name__ == "__main__":
    app()