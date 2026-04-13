from __future__ import annotations

from dotenv import load_dotenv

from src.agent.graph import build_agent_graph
from src.agent.runtime import init_agent_globals
from src.agent.state import AgentRunState
from src.common.llm_client import LLMClient
from src.common.logging_setup import setup_logging

from src.common.paths import PATHS



def main(input_from_cache: bool = False) -> None:
    load_dotenv()
    setup_logging(level="INFO")

    init_agent_globals(
        # input_repository=InputRepository(data_path=PATHS.data_dir),
        data_dir=PATHS.data_dir,
        llm_client=LLMClient(provider="gigachat"),
    )

    graph = build_agent_graph(input_from_cache)
    result = graph.invoke(AgentRunState())

    print(result) 