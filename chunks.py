import uuid
#from langchain.chains import create_extraction_chain_pydantic

NEW_CHUNK_SUMMARY_PROMPT = [("system", """
You are the steward of a group of chunks which represent groups of sentences that talk about a similar topic
You should generate a very brief 1-sentence summary which will inform viewers what a chunk group is about.

A good summary will say what the chunk is about, and give any clarifying instructions on what to add to the chunk.

You will be given a proposition which will go into a new chunk. This new chunk needs a summary.

Your summaries should anticipate generalization. If you get a proposition about apples, generalize it to food.
Or month, generalize it to "date and times".

Example:
Input: Proposition: Greg likes to eat pizza
Output: This chunk contains information about the types of food Greg likes to eat.

Only respond with the new chunk summary, nothing else.
"""),
("user", "Determine the summary of the new chunk that this proposition will go into:\n{proposition}")]

NEW_CHUNK_TITLE_PROMPT = [("system", """
You are the steward of a group of chunks which represent groups of sentences that talk about a similar topic
You should generate a very brief few word chunk title which will inform viewers what a chunk group is about.

A good chunk title is brief but encompasses what the chunk is about

You will be given a summary of a chunk which needs a title

Your titles should anticipate generalization. If you get a proposition about apples, generalize it to food.
Or month, generalize it to "date and times".

Example:
Input: Summary: This chunk is about dates and times that the author talks about
Output: Date & Times

Only respond with the new chunk title, nothing else.
"""),
("user", "Determine the title of the chunk that this summary belongs to:\n{summary}")]

UPDATE_CHUNK_SUMMARY_PROMPT = [("system", """
You are the steward of a group of chunks which represent groups of sentences that talk about a similar topic
A new proposition was just added to one of your chunks, you should generate a very brief 1-sentence summary which will inform viewers what a chunk group is about.

A good summary will say what the chunk is about, and give any clarifying instructions on what to add to the chunk.

You will be given a group of propositions which are in the chunk and the chunks current summary.

Your summaries should anticipate generalization. If you get a proposition about apples, generalize it to food.
Or month, generalize it to "date and times".

Example:
Input: Proposition: Greg likes to eat pizza
Output: This chunk contains information about the types of food Greg likes to eat.

Only respond with the chunk new summary, nothing else.
"""),
("user", "Chunk's propositions:\n{proposition}\n\nCurrent chunk summary:\n{current_summary}")]

UPDATE_CHUNK_TITLE_PROMPT = [("system", """
You are the steward of a group of chunks which represent groups of sentences that talk about a similar topic
A new proposition was just added to one of your chunks, you should generate a very brief updated chunk title which will inform viewers what a chunk group is about.

A good title will say what the chunk is about.

You will be given a group of propositions which are in the chunk, chunk summary and the chunk title.

Your title should anticipate generalization. If you get a proposition about apples, generalize it to food.
Or month, generalize it to "date and times".

Example:
Input: Summary: This chunk is about dates and times that the author talks about
Output: Date & Times

Only respond with the new chunk title, nothing else.
"""),
("user", "Chunk's propositions:\n{proposition}\n\nChunk summary:\n{current_summary}\n\nCurrent chunk title:\n{current_title}")]

FIND_RELEVANT_CHUNK_PROMPT = [("system", """
Determine whether or not the "Proposition" should belong to any of the existing chunks.

A proposition should belong to a chunk of their meaning, direction, or intention are similar.
The goal is to group similar propositions and chunks.

If you think a proposition should be joined with a chunk, return the chunk id.
If you do not think an item should be joined with an existing chunk, just return "No chunks"

Example:
Input:
    - Proposition: "Greg really likes hamburgers"
    - Current Chunks:
        - Chunk ID: 2n4l3d
        - Chunk Name: Places in San Francisco
        - Chunk Summary: Overview of the things to do with San Francisco Places

        - Chunk ID: 93833k
        - Chunk Name: Food Greg likes
        - Chunk Summary: Lists of the food and dishes that Greg likes
Output: 93833k
"""),
("user", "Current Chunks:\n--Start of current chunks--\n{current_chunk_outline}\n--End of current chunks--"),
("user", "Determine if the following statement should belong to one of the chunks outlined:\n{proposition}")]

def agentic_chunking(llm, splits: list[str]):
    chunks = {}
    id_truncate_limit = 5
    generate_new_metadata_ind = True
    
    for proposition in splits:
        current_chunk_outline = "".join([
            f"Chunk ID: {chunk['chunk_id']}\nChunk Name: {chunk['title']}\nChunk Summary: {chunk['summary']}\n\n"
            for chunk in chunks.values()
        ])
        chunk_found = llm([{"role": t[0], "content": t[1].strip().format({
            "proposition": proposition,
            "current_chunk_outline": current_chunk_outline
        })} for t in FIND_RELEVANT_CHUNK_PROMPT])

        # Pydantic data class
        class ChunkID(BaseModel):
            """Extracting the chunk id"""
            chunk_id: str | None
            
        # Extraction to catch-all LLM responses. This is a bandaid
        extraction_chain = create_extraction_chain_pydantic(pydantic_schema=ChunkID, llm=self.llm)
        extraction_found = extraction_chain.run(chunk_found)
        if extraction_found:
            chunk_found = extraction_found[0].chunk_id

        if len(chunk_found) != id_truncate_limit:
            chunk_id = None
        else:
            chunk_id = chunk_found
        
        if len(chunks) == 0 or not chunk_id:
            summary = _get_new_chunk_summary(proposition)
            id = str(uuid.uuid4())[:5]
            chunks[id] = {
                'chunk_id': id,
                'propositions': [proposition],
                'title': _get_new_chunk_title(summary),
                'summary': summary,
                'chunk_index': len(chunks)
            }
        else:
            chunks[chunk_id]['propositions'].append(proposition)
            if generate_new_metadata_ind:
                chunks[chunk_id]['summary'] = self._update_chunk_summary(chunks[chunk_id])
                chunks[chunk_id]['title'] = self._update_chunk_title(chunks[chunk_id])

    return [chunks.append(" ".join([x for x in chunk["propositions"]])) for chunk in chunks.values()]

def _update_chunk_summary(self, chunk):
    """
    If you add a new proposition to a chunk, you may want to update the summary or else they could get stale
    """
    

    runnable = PROMPT | self.llm

    new_chunk_summary = runnable.invoke({
        "proposition": "\n".join(chunk['propositions']),
        "current_summary" : chunk['summary']
    }).content

    return new_chunk_summary

def _update_chunk_title(self, chunk):
    """
    If you add a new proposition to a chunk, you may want to update the title or else it can get stale
    """

    runnable = PROMPT | self.llm

    updated_chunk_title = runnable.invoke({
        "proposition": "\n".join(chunk['propositions']),
        "current_summary" : chunk['summary'],
        "current_title" : chunk['title']
    }).content

    return updated_chunk_title

    def _get_new_chunk_summary(self, proposition):
        

        runnable = PROMPT | self.llm

        new_chunk_summary = runnable.invoke({"proposition": proposition}).content

        return new_chunk_summary
    
    def _get_new_chunk_title(self, summary):
        

        runnable = PROMPT | self.llm

        new_chunk_title = runnable.invoke({"summary": summary}).content

        return new_chunk_title
        
