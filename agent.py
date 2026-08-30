"""Convenience submission entry point.

The official local evaluator imports starter.agent.Agent. This module exports
the same class for packaging systems that expect a root entry file.
"""

from starter.agent import Agent

__all__ = ["Agent"]
