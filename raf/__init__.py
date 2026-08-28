"""Recoverability-Aware Forgetting (RA-FM).

A memory-management framework for long-horizon LLM agents that decides what to
delete based on whether the deleted information stays *reconstructable* from the
memories that remain, not merely on individual utility.

Public entry points:
    raf.config.RafConfig          - experiment configuration
    raf.memory.store.MemoryStore  - the agent's episodic memory
    raf.memory.forgetter.RaFmForgetter - the RA-FM eviction policy
    raf.agent.react_agent.ReactAgent   - ReAct agent for AppWorld
    raf.appworld_runner           - run a dataset x budget x method grid
"""

__version__ = "0.1.0"
