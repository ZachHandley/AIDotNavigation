# AIDotNavigation

A demonstration of AI-powered navigation through complex nested data structures using dot notation and LlamaIndex function calling.

## Overview

AIDotNavigation showcases how AI agents can intelligently navigate, sort, and extract information from complex nested data structures that would be too large to fit in a normal context window. This example repository demonstrates an approach to data exploration that uses dot notation traversal (like `data.users[0].profile.address`) to explore potentially large objects via LlamaIndex and Function Calling.

## Key Features

- **Intelligent Path Discovery**: AI agent can autonomously explore complex data structures
- **Advanced Sorting**: Sort arrays and collections by various properties
- **Data Extraction**: Extract targeted information from deeply nested structures
- **Goal-Oriented Analysis**: Perform complex analyses based on specific goals
- **Adaptable Architecture**: Works with any function-calling LLM that LlamaIndex supports

## Components

The project consists of two main components:

1. **PathNavigator**: A utility class for navigating and sorting complex nested data structures using dot notation and array indexing.
2. **ObjectAgent**: An AI-powered workflow that uses function calling to intelligently explore data structures.

## Requirements

- Python 3.11+
- OpenAI API key (or another function-calling model)
- Dependencies listed in pyproject.toml (managed with [uv](https://github.com/astral-sh/uv))

## Setup

1. Clone the repository
2. Create a virtual environment using uv or your preferred tool
3. Install dependencies with uv:

   ```bash
   uv pip install -e .
   ```

4. Create a `.env` file with your OpenAI API key:

   ```bash
   OPENAI_API_KEY=your_api_key_here
   ```

## Running the Demo

You can run the test script using:

```bash
uv run test
```

The test script supports multiple options:

```bash
# Single analysis with default data
uv run test --type single --prompt "Find the top 3 customers by total order value"

# Multiple analyses with custom goals
uv run test --type multiple --goals "Find top performers, Analyze trends, Identify opportunities"

# Using custom JSON data
uv run test --json-input data.json --prompt "Find anomalies"

# Interactive mode (recommended for first-time users)
uv run test
```

## Adapting to Other Models

This code can be adapted to any function-calling supported model that LlamaIndex supports by changing the LLM initialization in the `ObjectAgent` class:

```python
# Change this line in ObjectAgent
self.llm = OpenAI(model="gpt-4o", temperature=0)  # Current default

# Also change in config.py
AI_MODEL = "gpt-4o"
OPENAI_API_KEY = "your_api_key_here" # change this to ANTHROPIC_API_KEY if using anthropic, for instance

# Example using a different model
self.llm = OpenAI(model="gpt-3.5-turbo", temperature=0)

# Example using a different provider (requires appropriate LlamaIndex integration)
from llama_index.llms.anthropic import Anthropic
self.llm = Anthropic(model="claude-3-opus-20240229", temperature=0)
```

## Use Cases

This library is useful for:

1. Exploring large JSON API responses
2. Analyzing complex nested data structures
3. Performing targeted data extraction without loading the entire dataset into context
4. Creating AI agents that can intelligently explore databases or object hierarchies

## Project Structure

- `app/path_navigator.py`: Core utility for path-based data traversal
- `app/object_agent.py`: LlamaIndex Workflow implementation of the AI agent
- `app/config.py`: Configuration and settings management
- `scripts/test.py`: Test script with example data and usage patterns

## License

MIT

## Acknowledgements

This project showcases the capabilities of LlamaIndex and demonstrates how to build agentic AI systems that can work with data structures too large for traditional approaches.
