# MindMapper

Python simulation where LLM-backed agents interact, maintain thought trees, and track sentiment over time. Each agent has a goal, a tree of thoughts, and uses keyword extraction (YAKE) and sentiment analysis (TextBlob) to update its internal state. Agents generate responses via the Anthropic API and look up entities through the Google Knowledge Graph API.

## Status

Proof of concept. Runs but requires manual API key insertion in source code. No tests, no CLI arguments, no config file.

## What it does

1. Creates N agents, each with a goal string and a thought tree (rooted tree of `ThoughtNode` objects).
2. On each simulation tick, agents interact with each other:
   - Generate a response using Anthropic Claude.
   - Extract keywords from the response (YAKE).
   - Analyze sentiment (TextBlob).
   - Update emotional state, interpersonal state, and knowledge graph.
   - Expand the thought tree with new child nodes.
3. The thought tree can be explored via depth-first, breadth-first, or heuristic search.
4. Simulation runs for a configurable number of iterations.

## Components

| File | Purpose |
|---|---|
| `main.py` | Entry point, creates agents and runs the simulation loop |
| `agents.py` | `Agent` class with LLM calls, sentiment, keyword extraction, thought tree |
| `simulation.py` | `Simulation` orchestrator |
| `thought_tree_explorer.py` | DFS, BFS, and heuristic traversal of thought trees |
| `theory_of_mind.py` | Stub for theory-of-mind reasoning between agents |
| `decision_model.py` | Decision logic |
| `evaluation_metric.py` | Metrics for evaluating agent understanding |
| `interactive_interface.py` | Console-based interaction (if used) |
| `nlp_messaging.py` | NLP utilities for agent messaging |
| `task_scenario.py` | Task/scenario definitions |

## Requirements

| Dependency | Used for |
|---|---|
| anthropic | LLM calls (Claude) |
| textblob | Sentiment analysis |
| yake | Keyword extraction |
| requests | Google Knowledge Graph API |
| google-api-python-client | Google API client |
| colorama | Console output formatting |
| numpy, pandas | Data handling |
| tensorflow, torch | Listed in requirements but not clearly used in main flow |

Python 3.8+.

## Setup

```bash
git clone https://github.com/jmanhype/MindMapper.git
cd MindMapper
pip install -r requirements.txt
```

Edit `agents.py` and replace the empty strings with your API keys:

```python
self.client = anthropic.Client(api_key="YOUR_ANTHROPIC_KEY")
self.google_api_key = "YOUR_GOOGLE_API_KEY"
```

Run:

```bash
python main.py
```

## Limitations

- API keys are hardcoded in source files, not loaded from environment variables or a config file.
- `requirements.txt` includes tensorflow and torch, but neither appears to be used in the core simulation loop. They add ~2 GB of downloads.
- No tests.
- No command-line arguments; iteration count and agent goals are set by editing `main.py`.
- The `theory_of_mind.py` module is a stub; agents do not actually model each other's beliefs.
- Thought tree exploration strategies exist but are not used to drive agent decisions in the main loop.
- The Google Knowledge Graph API integration is defined but it is unclear when it fires during simulation.
- No persistence; simulation state is lost on exit.

## License

Apache 2.0
