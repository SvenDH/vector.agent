import json
import random
import copy
from typing import Callable
from textwrap import dedent

from pydantic import BaseModel, Field
import litellm

from util import func2json

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
            agent.messages.append({"name": agent.name, **message.model_dump()})
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
                agent.messages.append({"role": "tool", "tool_call_id": c.id, "tool_name": c.function.name, "content": result.value})
                if result.agent: agent = result.agent
        return Response(messages=agent.messages[n:], agent=agent)

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