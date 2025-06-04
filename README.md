# Agents101: Into the world of LLM Agents

Welcome to **Agents101** - my hands-on journey through the fascinating world of AI Agents. This repository is largely based on the [HuggingFace AI Agents Course](https://huggingface.co/learn/agents-course/en/unit0/introduction) (highly recommended) and contains practical implementations of agents from scratch to multiple frameworks including SmolAgents, LlamaIndex, and LangGraph.

A PDF with notes is also attached in the repo here: [Course Notes](./AI_Agents_CourseNotes.pdf)

## Table of Contents

- [Agents101: Into the world of LLM Agents](#agents101-into-the-world-of-llm-agents)
  - [Table of Contents](#table-of-contents)
  - [What are AI (LLM) Agents?](#what-are-ai-llm-agents)
  - [Course Overview](#course-overview)
  - [Learning Path](#learning-path)
  - [Projects \& Tutorials](#projects--tutorials)
    - [1. Foundation: Building Your First Agent](#1-foundation-building-your-first-agent)
    - [2. Function Calling \& Tool Use](#2-function-calling--tool-use)
    - [3. Code Agents with SmolAgents](#3-code-agents-with-smolagents)
    - [4. Multi-Agent Systems (SmolAgents)](#4-multi-agent-systems-smolagents)
    - [5. Other Agent Frameworks](#5-other-agent-frameworks)
      - [LlamaIndex Agents](#llamaindex-agents)
      - [LangGraph Agents](#langgraph-agents)
  - [Key Concepts Covered](#key-concepts-covered)
    - [🧠 Agent Fundamentals](#-agent-fundamentals)
    - [🛠️ Tool Use \& Function Calling](#️-tool-use--function-calling)
    - [💻 Code Execution](#-code-execution)
    - [🌐 Web Integration](#-web-integration)
    - [🤝 Multi-Agent Coordination](#-multi-agent-coordination)
    - [🔍 Retrieval-Augmented Generation (RAG)](#-retrieval-augmented-generation-rag)
  - [Prerequisites](#prerequisites)
  - [Getting Started](#getting-started)
    - [Practical Tips Learnt](#practical-tips-learnt)
  - [Next Steps](#next-steps)
    - [Expanding Your Agent Skills](#expanding-your-agent-skills)

## What are AI (LLM) Agents?

AI Agents are autonomous systems that can perceive their environment, make decisions, and take actions to achieve specific goals. Unlike traditional chatbots that just respond to queries, agents can; **plan** multi-step approaches to complex problems, **use tools** to interact with external systems (web search, APIs, databases), **execute code** to perform calculations and data analysis and much more!

For me, the real motivation behind exploring AI agents is understanding the parallels between computational agents and embodied agents in the physical world. As someone currently exploring robotics, I can’t help but notice how many AI agent concepts are already deeply rooted in robotics research.

This raises some interesting questions:

- 👉 How can learnings from sensing and interacting with the physical world inform the design of computational LLM-based agents?
- 👉 Conversely, how can the capabilities of LLM-based agents accelerate progress in robotics?

Some of the biggest breakthroughs might come from bridging these two paradigms.

## Course Overview

This course goes from zero to advanced AI agent development through practical, hands-on projects. Based on the HuggingFace AI Agents Course, it covers everything from building primitive agents from scratch to implementing sophisticated multi-agent systems with modern frameworks.

## Learning Path

1. **Foundations** - Understanding the basics of AI agents and building one from scratch
2. **Tool Integration** - Learning how agents use external tools and function calling
3. **Code Agents** - Building agents that can write and execute code
4. **Multi-Agent Systems** - Creating teams of specialized agents
5. **Advanced Frameworks** - Exploring production-ready agent frameworks

## Projects & Tutorials

### 1. Foundation: Building Your First Agent

**Location**: [`./primitive/`](./primitive/)

**Content**: The fundamental concepts of AI agents by building one from scratch.

**Key Files**:

- [`PrimitiveAgent.ipynb`](./primitive/PrimitiveAgent.ipynb) - Complete walkthrough of building an agent from scratch

### 2. Function Calling & Tool Use

**Location**: [`./fine-tune-function-calling/`](./fine-tune-function-calling/)

**Content**: How to fine-tune models for better function calling and tool use.

**Key Files**:

- [`Fine_tune_a_model_for_Function_Calling.ipynb`](./fine-tune-function-calling/Fine_tune_a_model_for_Function_Calling.ipynb) - Complete fine-tuning pipeline

Fine-tuning enables models to better understand when and how to use tools. This is crucial for creating reliable agents that can interact with external systems.

### 3. Code Agents with SmolAgents

**Location**: [`./smolagents-demos/`](./smolagents-demos/)

**Content**: Building production-ready agents using the SmolagAgents framework.

**Key Files**:

- [`code_agents_with_smolagents.ipynb`](./smolagents-demos/code_agents_with_smolagents.ipynb) - Interactive tutorial
- [`alfred.py`](./smolagents-demos/alfred.py) - CLI agent example
- [`seeing-alfred.py`](./smolagents-demos/seeing-alfred.py) - Vision-enabled agent
- [`retrieval_agents.ipynb`](./smolagents-demos/retrieval_agents.ipynb) - RAG-powered agents

**Mini Tutorial**:

```python
from smolagents import CodeAgent, DuckDuckGoSearchTool, HfApiModel

# Create an agent with web search capabilities
agent = CodeAgent(
    tools=[DuckDuckGoSearchTool()], 
    model=HfApiModel("Qwen/Qwen2.5-Coder-32B-Instruct")
)

# Run the agent
result = agent.run("Find the latest news about AI agents")
```

**Agent Types Demonstrated**:

- **Code Agents** - Agents that can write and execute Python code
- **Vision Agents** - Agents that can analyze images and visual content
- **Retrieval Agents** - Agents with access to knowledge bases (RAG)
- **Web Agents** - Agents that can browse and search the internet

### 4. Multi-Agent Systems (SmolAgents)

**Location**: [`./smolagents-demos/multiagent_notebook.ipynb`](./smolagents-demos/multiagent_notebook.ipynb)

**Content**: Creating teams of specialized agents that work together.

**Mini Tutorial**:

```python
# Create specialized agents
web_agent = CodeAgent(
    model=model,
    tools=[GoogleSearchTool(), VisitWebpageTool()],
    name="web_agent",
    description="Browses the web to find information"
)

manager_agent = CodeAgent(
    model=stronger_model,
    tools=[calculation_tools],
    managed_agents=[web_agent],  
    planning_interval=5,
    verbosity_level=2
)
```

**Benefits of Multi-Agent Systems**:

- **Specialization** - Each agent focuses on what it does best
- **Memory Separation** - Reduces context length and improves performance
- **Scalability** - Easy to add new specialized agents
- **Cost Efficiency** - Use smaller models for specialized tasks

### 5. Other Agent Frameworks

#### LlamaIndex Agents

**Location**: [`./llamaindex-demos/`](./llamaindex-demos/)

LlamaIndex excels at building RAG-powered agents and complex workflows. It provides powerful abstractions for document processing, vector stores, and agent coordination.

**Key Files**:

- [`agents.ipynb`](./llamaindex-demos/agents.ipynb) - Basic agent creation and tool use
- [`workflows.ipynb`](./llamaindex-demos/workflows.ipynb) - Advanced workflow patterns
- [`lindex-agents.py`](./llamaindex-demos/lindex-agents.py) - An example of an AccountantAgent
- [`components.ipynb`](./llamaindex-demos/components.ipynb) - RAG system components
- [`tools.ipynb`](./llamaindex-demos/tools.ipynb) - Custom tool creation

#### LangGraph Agents

**Location**: [`./langraph-demos/`](./langraph-demos/)

LangGraph provides state-based workflow orchestration with advanced control flow, conditional routing, and visual graph representations.

**Key Files**:

- [`agent.ipynb`](./langraph-demos/agent.ipynb) - ReAct agent with vision capabilities
- [`mail_sorting.ipynb`](./langraph-demos/mail_sorting.ipynb) - Email classification workflow example
- [`explainer.py`](./langraph-demos/explainer.py) - An agent that analyzes, translates and summarises visual content from one or multiple files.

## Key Concepts Covered

### 🧠 Agent Fundamentals

- **ReAct Pattern** - Reasoning and Acting in language models
- **System Prompts** - How to effectively prompt agents
- **Tool Integration** - Connecting agents to external systems
- **Memory Management** - Handling conversation history and context

### 🛠️ Tool Use & Function Calling

- **Function Definitions** - Describing tools to language models
- **Parameter Extraction** - How models parse function arguments
- **Error Handling** - Robust tool execution
- **Custom Tools** - Building domain-specific capabilities

### 💻 Code Execution

- **Code Generation** - How agents write Python code
- **Sandboxed Execution** - Safe code execution environments
- **State Management** - Maintaining variables across executions
- **Debugging** - How agents handle and fix code errors

### 🌐 Web Integration

- **Search Capabilities** - Using search engines as tools
- **Web Browsing** - Visiting and extracting content from websites
- **API Integration** - Connecting to web services
- **Data Extraction** - Processing web content

### 🤝 Multi-Agent Coordination

- **Agent Hierarchies** - Manager and worker agent patterns
- **Communication Protocols** - How agents share information
- **Task Delegation** - Distributing work among specialists
- **Planning & Coordination** - High-level task orchestration

### 🔍 Retrieval-Augmented Generation (RAG)

- **Vector Databases** - Storing and searching embeddings
- **Document Processing** - Preparing knowledge bases
- **Query Enhancement** - Improving search relevance
- **Context Integration** - Combining retrieved info with generation

## Prerequisites

- **Python 3.8+** with basic programming knowledge
- **API Keys** for various services (HuggingFace, OpenAI, etc.)
- **Basic ML Understanding** - Familiarity with LLMs is helpful but not required

## Getting Started

1. **Clone the repository**:

```bash
git clone https://github.com/ufakz/agents101
cd agents101
```

2. **Set up environment**:

```bash
# Create .env file with your API keys
echo "HF_TOKEN=your_huggingface_token" > .env
echo "OPENAI_API_KEY=your_openai_key" >> .env
```

3. **Start with the basics**:
   - Begin with [`./primitive/PrimitiveAgent.ipynb`](./primitive/PrimitiveAgent.ipynb)
   - Follow the learning path sequentially (or not)
   - Experiment with the provided examples and extend with yours

4. **Run CLI agents or any of the notebooks**:

```bash
# Simple task agent
python smolagents-demos/alfred.py "Plan a birthday party"

# Vision agent
python smolagents-demos/seeing-alfred.py path/to/image.jpg
```

### Practical Tips Learnt

**Agent Design**:

- Always keep system prompts clear and specific
- Provide examples of good tool usage to the agent
- Design tools with clear input/output specifications

**Multi-Agent Systems**:

- Assign clear roles and responsibilities to each agent
- Use a coordinator/manager pattern for complex tasks
- Monitor inter-agent communication for debugging

**Performance Optimization**:

- Cache frequently used results
- Use appropriate model sizes for each task (some small models outperform larger ones on specific tasks)
- Implement timeouts for long-running operations

## Next Steps

### Expanding Your Agent Skills

**Immediate Next Steps you can explore**:

1. **Experiment with Examples** - Run the provided notebooks and scripts
2. **Modify and Extend** - Add your own tools and capabilities
3. **Build Your Own Agent** - Create an agent for your specific use case

**Advanced Projects to Try (Some Examples)**:

- **Personal Assistant Agent** - Integrate with your calendar, email, and task management
- **Research Agent** - Build an agent that can conduct comprehensive research on any topic
- **Code Review Agent** - Create an agent that can review and suggest improvements to code
- **Data Analysis Agent** - Build an agent that can analyze datasets and generate insights
- .... and many more

**Learn More about AI Agents**:

- **HuggingFace Agent Documentation** - [smolagents docs](https://huggingface.co/docs/smolagents)
- **LlamaIndex Agent Guide** - [LlamaIndex workflows](https://docs.llamaindex.ai/en/stable/module_guides/workflow/)
- **LangGraph Tutorials** - [LangGraph documentation](https://langchain-ai.github.io/langgraph/)

---

**Happy Building! 🤖✨**
