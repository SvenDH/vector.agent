import json

import chromadb
from pydantic import BaseModel, Field

EXTRACT_ENTITIES_PROMPT = """
You are an advanced algorithm designed to extract structured information from text to construct knowledge graphs. Your goal is to capture comprehensive information while maintaining accuracy. Follow these key principles:

1. Extract only explicitly stated information from the text.
2. Identify nodes (entities/concepts), their types, and relationships.
3. Use "USER_ID" as the source node for any self-references (I, me, my, etc.) in user messages.
CUSTOM_PROMPT

Nodes and Types:
- Aim for simplicity and clarity in node representation.
- Use basic, general types for node labels (e.g. "person" instead of "mathematician").

Relationships:
- Use consistent, general, and timeless relationship types.
- Example: Prefer "PROFESSOR" over "BECAME_PROFESSOR".

Entity Consistency:
- Use the most complete identifier for entities mentioned multiple times.
- Example: Always use "John Doe" instead of variations like "Joe" or pronouns.

Strive for a coherent, easily understandable knowledge graph by maintaining consistency in entity references and relationship types.

Adhere strictly to these guidelines to ensure high-quality knowledge graph extraction."""

UPDATE_GRAPH_PROMPT = """
You are an AI expert specializing in graph memory management and optimization. Your task is to analyze existing graph memories alongside new information, and update the relationships in the memory list to ensure the most accurate, current, and coherent representation of knowledge.

Input:
1. Existing Graph Memories: A list of current graph memories, each containing source, target, and relationship information.
2. New Graph Memory: Fresh information to be integrated into the existing graph structure.

Guidelines:
1. Identification: Use the source and target as primary identifiers when matching existing memories with new information.
2. Conflict Resolution:
   - If new information contradicts an existing memory:
     a) For matching source and target but differing content, update the relationship of the existing memory.
     b) If the new memory provides more recent or accurate information, update the existing memory accordingly.
3. Comprehensive Review: Thoroughly examine each existing graph memory against the new information, updating relationships as necessary. Multiple updates may be required.
4. Consistency: Maintain a uniform and clear style across all memories. Each entry should be concise yet comprehensive.
5. Semantic Coherence: Ensure that updates maintain or improve the overall semantic structure of the graph.
6. Temporal Awareness: If timestamps are available, consider the recency of information when making updates.
7. Relationship Refinement: Look for opportunities to refine relationship descriptions for greater precision or clarity.
8. Redundancy Elimination: Identify and merge any redundant or highly similar relationships that may result from the update.

Task Details:
- Existing Graph Memories:
{existing_memories}

- New Graph Memory: {memory}

Output:
Provide a list of update instructions, each specifying the source, target, and the new relationship to be set. Only include memories that require updates.
"""

class Search(BaseModel):
    entities: list[str] = Field(..., description="List of entities to search for.")
    relations: list[str] = Field(..., description="List of relations to search for.")

class Relation(BaseModel):
    source_node: str = Field(..., description="The identifier of the source node in the relationship.")
    source_type: str = Field(..., description="The type or category of the source node.")
    relationship: str = Field(..., description="The type of relationship between the source and target nodes.")
    target_node: str = Field(..., description="The identifier of the target node in the relationship.")
    target_type: str = Field(..., description="The type or category of the target node.")

class Entities(BaseModel):
    entities: list[Relation] = Field(..., description="Add new entities and relationships to the graph based on the provided query.")

class Update(BaseModel):
    source_node: str = Field(..., description="The identifier of the source node in the relationship to be updated.")
    target_node: str = Field(..., description="The identifier of the target node in the relationship to be updated.")
    relationship: str = Field(..., description="The new or updated relationship between the source and target nodes.")

class GraphMutations(BaseModel):
    additions: list[Relation] = Field(..., description="List of new relationships to be added to the graph. Add a new graph memories to the knowledge graph. Each item creates a new relationship between two nodes, potentially creating new nodes if they don't exist.")
    updates: list[Update] = Field(..., description="List of relationships to be updated in the graph. Each item updates the relationship key of an existing graph memory based on new information. Updates should only be performed if the new information is more recent, more accurate, or provides additional context compared to the existing information. The source and destination nodes of the relationship must remain the same as in the existing graph memory; only the relationship itself can be updated.")

def normalize(text: str) -> str:
    return text.strip(" _").lower().replace(" ", "_").replace("'", "").replace('"', "").replace(",", "").replace("'", "").replace(";", "").replace("\n", "").removeprefix("the_").removeprefix("a_")

class KnowledgeGraph:
    def __init__(self, llm, db: chromadb.ClientAPI, model: str = "gpt-4o-mini"):
        self.nodes = db.get_or_create_collection(name="nodes", metadata={"hnsw:space": "cosine"})
        self.relations = db.get_or_create_collection(name="relations", metadata={"hnsw:space": "cosine"})
        self.model = model
        self.user_id = "USER"
        self.custom_prompt = None
        self.client = llm
    
    async def add(self, data: str, filters: dict = {}):
        new_memory: Entities = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": EXTRACT_ENTITIES_PROMPT.replace("USER_ID", self.user_id).replace("CUSTOM_PROMPT", f"4. {self.custom_prompt}") if self.custom_prompt else ""},
                {"role": "user", "content": data},
            ],
            response_model=Entities,
            max_retries=2,
        )
        if not new_memory.entities:
            return []
        
        search_output = "\n".join([
            json.dumps({"source": entity["source"], "relation": entity["relation"], "destination": entity["destination"]})
            for entity in await self.search(data)])
        
        response: GraphMutations = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": UPDATE_GRAPH_PROMPT.format(existing_memories=search_output, memory=new_memory)}],
            response_model=GraphMutations,
            max_retries=2,
        )
        to_be_added = response.additions + response.updates
        to_delete = [{"$and": [
                {"source": normalize(item.source_node)},
                {"destination": normalize(item.target_node)}]}
                for item in response.updates]
        node_ids, node_metadatas, node_documents = [], [], []
        rel_ids, rel_metadatas, rel_documents = [], [], []
        returned_entities = []
        for item in to_be_added:
            source = normalize(item.source_node)
            relation = normalize(item.relationship)
            target = normalize(item.target_node)
            rel_id = f"{source}_{relation}_{target}"
            returned_entities.append({"source": source, "relationship": relation, "target": target})
            for node in [source, target]:
                if node not in node_ids:
                    node_ids.append(node)
                    node_documents.append(node)
                    metadata = {"name": node, **filters}
                    if node == target and isinstance(item, Relation):
                        metadata["type"] = normalize(item.target_type)
                    elif node == source and isinstance(item, Relation):
                        metadata["type"] = normalize(item.source_type)
                    node_metadatas.append(metadata)
            if rel_id not in rel_ids:
                rel_ids.append(rel_id)
                rel_metadatas.append({"source": source, "destination": target, "relation": relation, **filters})
                rel_documents.append(f"{item.source_node} {item.relationship.lower().replace('_', ' ')} {item.target_node}")
        
        if to_delete: self.relations.delete(where={"$or": to_delete} if len(to_delete) > 1 else to_delete[0])
        if node_ids: self.nodes.upsert(node_ids, None, node_metadatas, node_documents)
        if rel_ids: self.relations.upsert(rel_ids, None, rel_metadatas, rel_documents)
        return returned_entities

    async def search(self, query: str, filters: dict = {}, limit: int = 100, threshold: float = 0.7):
        response: Search = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": f"You are a smart assistant who understands the entities, their types, and relations in a given text. If user message contains self reference such as 'I', 'me', 'my' etc. then use {self.user_id} as the source node. Extract the entities. ***DO NOT*** answer the question itself if the given text is a question.",
                },
                {"role": "user", "content": query},
            ],
            response_model=Search,
            max_retries=2,
        )
        nodes = [normalize(node) for node in set(response.entities)]
        if not nodes:
            return []
        
        nodes = self.nodes.query(query_texts=nodes, where=filters, n_results=1)
        print(nodes)
        nodes = list(set([n[0]["name"] for d, n in zip(nodes["distances"], nodes["metadatas"]) if d[0] < threshold]))
        if not nodes:
            return []
        
        results = self.relations.query(
            query_texts=[query],
            where={"$or": [{"source": {"$in": nodes}}, {"destination": {"$in": nodes}}]},
            n_results=limit
        )
        return [r for m in results["metadatas"] for r in m]

    async def delete_all(self, filters=None):
        self.nodes.delete(where=filters)
        self.relations.delete(where=filters)
    
    async def get_node(self, node: str):
        return {
            **self.nodes.get(where={"name": node})["metadatas"],
            "to": self.relations.get(where={"source": node})["metadatas"],
            "from": self.relations.get(where={"destination": node})["metadatas"],
            "close": self.relations.query(query_texts=[node], n_results=10)["metadatas"][1:],
        }

    async def get_all(self, filters=None, limit=None):
        return self.relations.get(where=filters, limit=limit)["metadatas"]

    async def get_all_nodes(self, filters=None, limit=None):
        return self.nodes.get(where=filters, limit=limit)["metadatas"]
    
    async def get_relation(self, relation: str, offset=None, limit=None):
        return self.relations.get(where={"relation": relation}, offset=offset, limit=limit)["metadatas"]
    
    async def get_type(self, type: str, offset=None, limit=None):
        return self.nodes.get(where={"type": type}, offset=offset, limit=limit)["metadatas"]

    async def dump(self):
        return {
            "nodes": self.nodes.get(include=["metadatas"])["metadatas"],
            "relations": self.relations.get(include=["metadatas"])["metadatas"],
        }
    
    async def load(self, data):
        node_ids = [node["name"] for node in data["nodes"]]
        node_documents = [node["name"].replace("_", " ") for node in data["nodes"]]
        rel_ids = [f"{rel['source']}_{rel['relation']}_{rel['destination']}" for rel in data["relations"]]
        rel_documents = [
            f"{rel['source'].replace('_', ' ')} {rel['relation'].replace('_', ' ')} {rel['destination'].replace('_', ' ')}"
            for rel in data["relations"]]
        if node_ids: self.nodes.upsert(node_ids, None, data["node"], node_documents)
        if rel_ids: self.relations.upsert(rel_ids, None, data["relations"], rel_documents)

    async def merge_nodes(self, node1: str, node2: str):
        self.nodes.delete(where={"name": node2})
        rel = self.relations.get(where={"$or": [{"source": node2}, {"destination": node2}]}, include=["metadatas"])
        if not rel["metadatas"]:
            print(f"No relations found for node {node2}.")
            return
        for r in rel["metadatas"]:
            if r["source"] == node2:
                r["source"] = node1
            if r["destination"] == node2:
                r["destination"] = node1
        ids = [f"{r['source']}_{r['relation']}_{r['destination']}" for r in rel["metadatas"]]
        documents = [f"{r['source'].replace('_', ' ')} {r['relation'].replace('_', ' ')} {r['destination'].replace('_', ' ')}" for r in rel["metadatas"]]
        self.relations.upsert(ids, None, rel["metadatas"], documents)
        self.relations.delete(where={"$or": [{"source": node2}, {"destination": node2}]})

    async def remove_node(self, node: str):
        self.relations.delete(where={"$or": [{"source": node}, {"destination": node}]})
        self.nodes.delete(where={"name": node})
        self.nodes.delete(ids=[node])

    async def get_graph(self):
        import networkx as nx

        relations = await self.get_all()
        nodes = set([r["source"] for r in relations] + [r["destination"] for r in relations])
        G = nx.MultiDiGraph()
        for n in nodes:
            G.add_node(n)
        for r in relations:
            G.add_edge(r["source"], r["destination"], relation=r["relation"])
        return G