"""Candidate generation module (JOB-06).

Shared pre-recall stage for both DQN and LLM pipelines.
Provides ``get_candidates(worker_id, timestamp, K)`` as the single entry point.
"""

from src.candidates.generator import get_candidates

__all__ = ["get_candidates"]
