# Agent Package

This package contains the core scaffolding for the AI Agent layer. The agent is responsible for coordinating tool executions and answering user queries based strictly on tool evidence.

## Overview
It provides compatibility with the existing RAG API response schema, meaning the agent layer can be seamlessly dropped into the existing application without breaking Streamlit UI parsing.

## Files
- `prompts.py`: Defines the system instructions and behavioral guidelines for the AI agent (e.g. failing safely, returning correct schema).
- `types.py`: Defines types (like `AgentResponse` and `ToolTraceItem`) to ensure data integrity during tool execution and serialization.
- `__init__.py`: Provides convenient exports for core items.
