"""Data module: paper loading + section-aware chunking."""

from .chunker import Chunk, chunk_paper
from .paper_loader import Paper, iter_papers, load_paper

__all__ = ["Chunk", "Paper", "chunk_paper", "iter_papers", "load_paper"]
