"""
The research agent: a tool-calling loop (Claude + the retrieve tool).

Kept deliberately small. The agent decides when to search, can search more
than once to compare across filings, and is told to cite every claim. The
system prompt does the heavy lifting — most "agent" behavior here is just
"retrieve before you answer, and don't make things up."
"""

import os
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate

from agent.tools import retrieve

load_dotenv()

SYSTEM_PROMPT = """You are a financial research assistant that answers \
questions about companies using their SEC filings.

Rules:
- Always use the `retrieve` tool before answering. Never answer from memory.
- For comparison questions ("how did X change year over year"), retrieve more \
than once with focused queries.
- Cite every factual claim with the bracketed citation from the retrieved \
passage, e.g. [NVDA 10-K 2025-02-26 — Item 1A].
- If the filings don't contain the answer, say so plainly. Do not guess.
"""


def build_agent(verbose: bool = False) -> AgentExecutor:
    llm = ChatAnthropic(model="claude-sonnet-4-6", max_tokens=1500)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    agent = create_tool_calling_agent(llm, [retrieve], prompt)
    return AgentExecutor(agent=agent, tools=[retrieve], verbose=verbose)


def ask(question: str, verbose: bool = False) -> str:
    result = build_agent(verbose=verbose).invoke({"input": question})
    return result["output"]


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "What are the main risk factors in NVDA's latest 10-K?"
    print(ask(q, verbose=True))
