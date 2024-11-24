import os
import asyncio
from uuid import uuid4
from datetime import datetime

from dotenv import load_dotenv
import chromadb
import typer
import litellm
import instructor

from agent import Agent, Chat
from graph import KnowledgeGraph
from util import split

load_dotenv(override=True)
# Monkey patch for ollama chat
#def _token_counter(**kwargs): return 0
#litellm.token_counter = _token_counter


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


import openai
llm = instructor.from_openai(openai.AsyncOpenAI(), mode=instructor.Mode.JSON_SCHEMA)
llm = instructor.from_litellm(litellm.acompletion, mode=instructor.Mode.JSON_SCHEMA, base_url=os.getenv("OPENAI_BASE_URL"))

mem = KnowledgeGraph(llm, db) #, model="ollama_chat/llama3.1")
mem.custom_prompt = "If the passage is part of acknowledgments, table of content, references, prewords, about the author, etc., you can skip it."

chat = Chat(agents=agents, selector="roundrobin", max_round=12)

app = typer.Typer()

@app.command()
def run():
    seen = set([(m["file"], m.get("index", 0)) for m in asyncio.run(mem.get_all(limit=1000000))])
    for i, file in enumerate(os.listdir("data")):
        texts = docs.get(where={"file": file})
        last = ""
        for t, m in sorted(zip(texts["documents"], texts["metadatas"]), key=lambda x: x[1]["index"]):
            if (file, m["index"]) in seen: continue
            print(t)
            #asyncio.run(chat.run(PASSAGE_PROMPT.format(file[:-9], part)))
            #chat.reset()
            while True:
                try:
                    print(asyncio.run(mem.add(last + " " + t, filters={"index": m["index"], "file": file})))
                    break
                except Exception as e:
                    print(e)
            last = t

@app.command()
def show():
    import networkx as nx
    import matplotlib.pyplot as plt

    G = asyncio.run(mem.get_graph())
    
    degree_cent = nx.degree_centrality(G)
    most_connected = sorted(degree_cent.items(), key=lambda x: x[1], reverse=True)
    top_node_ids = [n for n, _ in most_connected[:25]]
    top_neighbor_ids = set([n for n, _ in most_connected[:120]])
    
    neighbors = set()
    for node in top_node_ids:
        neighbors.update(set(G.neighbors(node)) & top_neighbor_ids)
    
    subgraph = G.subgraph(set(top_node_ids) | neighbors)

    pos = nx.spring_layout(subgraph, k=1, iterations=50)
    nx.draw_networkx_edges(subgraph, pos, alpha=0.2)
    nx.draw_networkx_nodes(subgraph, pos,
                          node_color='lightgray')
    nx.draw_networkx_nodes(subgraph, pos,
                          nodelist=top_node_ids,
                          node_color='gray')
    
    labels = {}
    for node in subgraph.nodes():
        if node in top_node_ids:
            labels[node] = f"{node}\n(deg: {degree_cent[node]})"
        else:
            labels[node] = str(node)
    
    nx.draw_networkx_labels(subgraph, pos, font_size=6)

    plt.axis('off')
    plt.tight_layout()
    plt.show()

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
    print("\n".join([f"{r['source'].replace('_', ' ')} {r['relation'].replace('_', ' ').upper()} {r['destination'].replace('_', ' ')}" for r in asyncio.run(mem.search(query))]))

@app.command()
def stats():
    import networkx as nx
    G = asyncio.run(mem.get_graph())

    print("Entities:", len(G.nodes))
    print("Relations:", len(G.edges))
    print("Components:", [len(c) for c in sorted(nx.weakly_connected_components(G), key=len, reverse=True)])

    types = [n.get("type", "entity") for n in asyncio.run(mem.get_all_nodes())]
    print("Types:", sorted(set(types), key=types.count, reverse=True))

    relations = [r["relation"] for r in asyncio.run(mem.get_all())]
    print("Relations:", sorted(set(relations), key=lambda x: relations.count(x), reverse=True))

@app.command()
def cleanup():
    from collections import defaultdict
    import networkx as nx

    print("REMOVE NODES WITH NONE DOCUMENT")
    nodes = mem.nodes.get()
    ids_to_delete = [id for id, n in zip(nodes["ids"], nodes["documents"]) if n is None]
    if ids_to_delete: mem.nodes.delete(ids=ids_to_delete)
    print(f"Deleted {len(ids_to_delete)} nodes with None document.")

    print("REMOVE DANGLING NODES")
    relations = mem.relations.get(include=["metadatas"])["metadatas"]
    rel_nodes = set([r["source"] for r in relations] + [r["destination"] for r in relations])
    ids_to_delete = []
    nodes = mem.nodes.get(include=["metadatas"])
    for id, n in zip(nodes["ids"], nodes["metadatas"]):
        if n["name"] not in rel_nodes:
            print(f"Removing dangling node: {n['name']}")
            ids_to_delete.append(id)
    if ids_to_delete: mem.nodes.delete(ids=ids_to_delete)
    print(f"Deleted {len(ids_to_delete)} dangling nodes.")
    
    nodes = set([n["name"] for n in mem.nodes.get(include=["metadatas"])["metadatas"]])
    ids_to_delete = []
    rels = mem.relations.get(include=["metadatas"])
    for id, r in zip(rels["ids"], rels["metadatas"]):
        if r["source"] not in nodes or r["destination"] not in nodes:
            print(f"Removing dangling relation: {id}")
            ids_to_delete.append(id)
    if ids_to_delete: mem.relations.delete(ids=ids_to_delete)
    print(f"Deleted {len(ids_to_delete)} dangling relations.")

    print("REMOVE SELF CONNECTED RELATIONS")
    ids_to_delete = []
    rels = mem.relations.get(include=["metadatas"])
    for id, r in zip(rels["ids"], rels["metadatas"]):
        if r["source"] == r["destination"]:
            print(f"Removing self connected relation: {id}")
            ids_to_delete.append(id)
    if ids_to_delete: mem.relations.delete(ids=ids_to_delete)
    print(f"Deleted {len(ids_to_delete)} self connected relations.")

    print("REMOVE DUPLICATE RELATIONS")
    edge_counts = defaultdict(list)
    rels = mem.relations.get(include=["metadatas"])
    for id, r in zip(rels["ids"], rels["metadatas"]):
        edge_counts[(r["source"], r["destination"])] += [id]
    ids_to_delete = [r for rel in edge_counts.values() for r in rel[1:]]
    if ids_to_delete: mem.relations.delete(ids=ids_to_delete)
    print(f"Deleted {len(ids_to_delete)} duplicate relations.")

    print("REMOVE LONG NODES")
    relations = asyncio.run(mem.get_all(limit=100000))
    nodes = set([r["source"] for r in relations] + [r["destination"] for r in relations])
    for n in nodes:
        if len(n) > 35:
            asyncio.run(mem.remove_node(n))
        else:
            new = n.strip("_").replace("'", "").replace('"', "").replace(",", "")\
                .replace("'", "").replace(";", "").replace("\n", "")\
                    .removeprefix("the_").removeprefix("a_")
            if new != n:
                print(n, "->", new)
                asyncio.run(mem.merge_nodes(new, n))

    print("MERGE SIMILAR NODES")
    chunk_size = 100
    limit = 100
    threshold = 0.09
    while True:
        pairs = set()
        nodes = mem.nodes.get(include=["documents"])["documents"]
        for i in range(0, len(nodes), chunk_size):
            chunk_pairs = mem.nodes.query(query_texts=nodes[i:i + chunk_size], n_results=2, include=["documents", "distances"])
            pairs.update([(*sorted([p[0], p[1]]), d[1]) for p, d in zip(chunk_pairs["documents"], chunk_pairs["distances"]) if len(d) > 1 and p[1] is not None and d[1] < threshold])
            if len(pairs) >= limit: break
        pairs = list(sorted(pairs, key=lambda x: x[2], reverse=True))
        if not pairs: break
        print(pairs)
        for pair in pairs:
            pair = list(pair)
            if (pair[0] != pair[1]+"s" and len(pair[0]) > len(pair[1])):
                temp = pair[0]
                pair[0] = pair[1]
                pair[1] = temp

            print(pair[1], "->", pair[0])
            asyncio.run(mem.merge_nodes(pair[0], pair[1]))
    
    print("REMOVE SINGLETONS")
    G = asyncio.run(mem.get_graph())
    for comp in nx.weakly_connected_components(G):
        if len(comp) <= 2:
            for node in comp:
                print(node)
                asyncio.run(mem.remove_node(node))

@app.command()
def delete_lose_components():
    import networkx as nx
    G = asyncio.run(mem.get_graph())
    for c in list(sorted(nx.weakly_connected_components(G), key=len, reverse=True))[1:2]:
        for node in c:
            print(node)
            #asyncio.run(mem.remove_node(node))

@app.command()
def remove(entity: str):
    asyncio.run(mem.remove_node(entity))

@app.command()
def merge(entity: str, into: str):
    asyncio.run(mem.merge_nodes(into, entity))

@app.command()
def clear():
    asyncio.run(mem.delete_all())

@app.command()
def dump():
    import json
    print(json.dumps(asyncio.run(mem.dump())))

if __name__ == "__main__":
    app()