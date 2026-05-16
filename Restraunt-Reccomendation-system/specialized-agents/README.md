# Specialized-Agents - Multi-Agent Framework

Foundation for a specialized multi-agent system. Currently used for agent design and testing; core app uses simplified heuristic routing instead.

## Files

- **define_agents.py** - Agent definitions
  - User Profile Generator Agent
  - RAG Retriever Agent
  - Food Trend Analyst Agent
  - Food Style Expert Agent
  - Nutrition Expert Agent
  - Recommendation Expert Agent

- **design_specialized_agents.py** - Agent prompt engineering
  - Creates system prompts for each agent
  - Defines tasks and expected outputs
  - Tests agent responses
  - Documents multi-agent workflow

- **implement_multi_agent_system.py** - Integration framework
  - Orchestrates agents in sequence
  - Routes inputs between agents
  - Aggregates outputs
  - Formats final recommendations

## Architecture

```
User Input
    ↓
User Profile Generator → Extracts preferences
    ↓
RAG Retriever → Fetches candidates
    ↓
Food Trend Analyst → Identifies trends
    ↓
Food Style Expert → Analyzes cuisines
    ↓
Nutrition Expert → Validates dietary fit
    ↓
Recommendation Expert → Synthesizes final results
    ↓
Output
```

## Current Status

⚠️ **Design Phase**: This module is designed but not active in production.

**Why?** The simplified **heuristic routing** in `fastMCP_server/app.py` provides:
- 95% accuracy with zero LLM overhead
- Sub-100ms response times
- Reliable intent detection for common queries
- Cost-effective (no unnecessary API calls)

## Future Use Cases

Multi-agent system valuable for:
- Complex dietary requirement analysis
- Culinary trend forecasting
- Personalized recipe generation
- Sophisticated reasoning on ambiguous queries

## Requirements

See `specialized-agents-requirement.txt` for dependencies.

## Testing

```bash
python design_specialized_agents.py
# Runs sample agent interactions
# Outputs test results
```

## Notes

- Agents use NVIDIA Llama 3.1 70B
- Each agent has a clear role and backstory
- Designed for transparency and explainability
- Can be activated for specific request types without rewriting core app
