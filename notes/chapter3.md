# Chapter 3: Agent Loop Design - The OODA Framework

## 🎯 The Problem: Agents Need Structure

You've learned to build graphs and manage state. But **how do you design an agent that thinks, acts, and learns effectively?**

Without a structured loop, agents become:
- **Reactive** - only responding, never planning
- **Rigid** - can't adapt to new situations
- **Inefficient** - wasting tokens on unnecessary steps

Enter the **OODA Loop** - a battle-tested framework from military strategy, now applied to AI agents.

---

## 🔧 Important: Switching to Ollama (Local LLMs)

From this chapter onwards, we'll use **Ollama** for all examples. This gives you:
- ✅ **Free** - No API costs
- ✅ **Private** - Runs locally
- ✅ **Fast iteration** - No rate limits
- ✅ **Production-ready** - Same patterns work with any LLM

### Setup Ollama

```bash
# 1. Install Ollama (if not already installed)
# Visit: https://ollama.ai/download

# 2. Pull llama3.2 model (3B version for speed)
ollama pull llama3.2

# 3. Verify it's running
ollama run llama3.2 "Hello"

# 4. Install LangChain Ollama integration
pip install langchain-ollama
# or with uv
uv pip install langchain-ollama
```

### Basic Usage

```python
from langchain_ollama import ChatOllama

# Initialize Ollama LLM
llm = ChatOllama(
    model="llama3.2",
    temperature=0.7,
    # Runs on http://localhost:11434 by default
)

# Test it
response = llm.invoke("What is 2+2?")
print(response.content)
```

**Note:** All examples from now on use `ChatOllama` instead of `ChatOpenAI`.

---

## 🔄 The OODA Loop Explained

**OODA = Observe → Orient → Decide → Act**

Originally developed by military strategist John Boyd, it's perfect for agent design:

```
┌─────────────────────────────────────────────────┐
│                  OODA LOOP                       │
│                                                  │
│  ┌──────────┐      ┌──────────┐                │
│  │ OBSERVE  │─────>│  ORIENT  │                │
│  └──────────┘      └──────────┘                │
│       ↑                  │                       │
│       │                  ↓                       │
│       │            ┌──────────┐                 │
│  ┌────┴────┐      │  DECIDE  │                 │
│  │   ACT   │<─────┴──────────┘                 │
│  └─────────┘                                    │
│       │                                          │
│       └──────> (back to OBSERVE)                │
└─────────────────────────────────────────────────┘
```

### The Four Phases

1. **OBSERVE** - Gather information (inputs, environment, state)
2. **ORIENT** - Analyze and understand context
3. **DECIDE** - Choose the best action
4. **ACT** - Execute the decision

---

## 📚 Building Each Phase in LangGraph

### Phase 1: OBSERVE - Perception Layer

**Goal:** Gather all relevant information before acting.

```python
from typing import TypedDict, Annotated, Literal
from operator import add
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# State definition for OODA agent
class OODAState(TypedDict):
    # Input
    user_input: str
    
    # Observation phase
    observations: Annotated[list[str], add]
    context: dict
    
    # Orient phase
    analysis: str
    understanding: str
    
    # Decide phase
    decision: str
    action_plan: str
    
    # Act phase
    action_result: str
    
    # Control
    current_phase: Literal["observe", "orient", "decide", "act", "complete"]
    iteration: Annotated[int, add]
    max_iterations: int

# Initialize LLM
llm = ChatOllama(model="llama3.2", temperature=0.7)

def observe(state: OODAState) -> dict:
    """
    OBSERVE: Gather information about the task
    
    In a real agent, this might:
    - Check available tools
    - Review conversation history
    - Assess current knowledge
    - Identify information gaps
    """
    
    observations = []
    
    # Observation 1: Analyze the user's request
    user_query = state["user_input"]
    observations.append(f"User request: {user_query}")
    
    # Observation 2: Check what we know
    prompt = f"""Analyze this request and identify:
1. What information is needed to answer it
2. What tools or resources might be required
3. Any potential challenges

Request: {user_query}

Provide a brief analysis."""

    analysis = llm.invoke([SystemMessage(content=prompt)])
    observations.append(f"Initial analysis: {analysis.content}")
    
    # Observation 3: Environmental context (simplified)
    observations.append(f"Iteration: {state['iteration']}")
    observations.append(f"Max iterations: {state['max_iterations']}")
    
    return {
        "observations": observations,
        "current_phase": "orient",
        "context": {
            "query_length": len(user_query),
            "iteration": state["iteration"]
        }
    }
```

### Phase 2: ORIENT - Analysis and Understanding

**Goal:** Make sense of observations, understand the situation.

```python
def orient(state: OODAState) -> dict:
    """
    ORIENT: Analyze observations and build understanding
    
    This is where the agent:
    - Synthesizes observations
    - Builds mental model
    - Identifies patterns
    - Frames the problem
    """
    
    # Gather all observations
    all_observations = "\n".join(state["observations"])
    
    prompt = f"""You are an AI agent analyzing a situation.

Observations:
{all_observations}

Based on these observations:
1. What is the core problem or question?
2. What approach should be taken?
3. What are the key considerations?

Provide a clear analysis."""

    understanding = llm.invoke([SystemMessage(content=prompt)])
    
    # Extract key insights
    analysis_prompt = f"""Summarize this understanding in one sentence:

{understanding.content}"""
    
    summary = llm.invoke([SystemMessage(content=analysis_prompt)])
    
    return {
        "analysis": summary.content,
        "understanding": understanding.content,
        "current_phase": "decide"
    }
```

### Phase 3: DECIDE - Plan the Action

**Goal:** Choose the best course of action.

```python
def decide(state: OODAState) -> dict:
    """
    DECIDE: Choose the best action based on understanding
    
    The agent:
    - Evaluates options
    - Selects strategy
    - Plans execution
    - Anticipates outcomes
    """
    
    prompt = f"""Based on this understanding:

{state['understanding']}

Decide on the best action to take. Your options:
1. Provide a direct answer (if you have enough information)
2. Search for more information (if needed)
3. Break down into sub-problems (if complex)
4. Ask for clarification (if ambiguous)

Choose ONE action and explain why."""

    decision = llm.invoke([SystemMessage(content=prompt)])
    
    # Create action plan
    plan_prompt = f"""Given this decision:

{decision.content}

Create a brief action plan (2-3 steps) for execution."""

    plan = llm.invoke([SystemMessage(content=plan_prompt)])
    
    return {
        "decision": decision.content,
        "action_plan": plan.content,
        "current_phase": "act"
    }
```

### Phase 4: ACT - Execute the Decision

**Goal:** Carry out the plan and observe results.

```python
def act(state: OODAState) -> dict:
    """
    ACT: Execute the decided action
    
    This is where:
    - Tools are called
    - Responses are generated
    - Changes are made
    - Results are captured
    """
    
    # For this example, we'll generate a response
    # In a real agent, this might call tools, APIs, etc.
    
    prompt = f"""Execute this action plan:

{state['action_plan']}

For the original request: {state['user_input']}

Provide your response or action result."""

    result = llm.invoke([SystemMessage(content=prompt)])
    
    return {
        "action_result": result.content,
        "current_phase": "complete",
        "iteration": 1  # Increment iteration
    }
```

### Routing Logic

```python
def route_next_phase(state: OODAState) -> str:
    """Route to next phase in OODA loop"""
    
    current = state["current_phase"]
    
    # Special handling for complete phase to enable looping
    if current == "complete":
        # Check if we should loop for improvement
        if state["iteration"] < state["max_iterations"]:
            # Check if response seems incomplete (optional improvement logic)
            if len(state.get("action_result", "")) < 50:
                return "observe"  # Loop back for refinement
        return "end"  # Done - exit the loop
    
    # For all other phases, simply go where the node indicated
    return current
```

---

## 🏗️ Complete OODA Agent Implementation

```python
from typing import TypedDict, Annotated, Literal
from operator import add
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

# Full state definition
class OODAState(TypedDict):
    user_input: str
    observations: Annotated[list[str], add]
    context: dict
    analysis: str
    understanding: str
    decision: str
    action_plan: str
    action_result: str
    current_phase: Literal["observe", "orient", "decide", "act", "complete"]
    iteration: Annotated[int, add]
    max_iterations: int

# Initialize LLM with better settings for local models
llm = ChatOllama(
    model="llama3.2", 
    temperature=0.7,
    num_ctx=2048  # Context window size
)

def observe(state: OODAState) -> dict:
    """OBSERVE: Gather and analyze initial information"""
    observations = []
    user_query = state["user_input"]
    observations.append(f"User request: {user_query}")
    
    # More explicit prompting for local models
    system_prompt = """You are an expert analyst. Your job is to carefully observe and analyze requests.

TASK: Analyze the user's request and identify:
1. What is the main question or goal?
2. What type of information is needed to answer?
3. Are there any ambiguities or missing details?
4. What challenges might arise in answering this?

Keep your analysis clear and structured."""
    
    user_prompt = f"""USER REQUEST: "{user_query}"

Now analyze this request following the 4 points above. Be specific and concise."""
    
    analysis = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])
    
    observations.append(f"Initial analysis: {analysis.content}")
    
    return {
        "observations": observations,
        "context": {
            "query_length": len(user_query),
            "query_type": "question" if "?" in user_query else "statement"
        },
        "current_phase": "orient"  # ← Node sets next phase
    }

def orient(state: OODAState) -> dict:
    """ORIENT: Synthesize observations into understanding"""
    all_obs = "\n".join(state["observations"])
    
    system_prompt = """You are an expert at synthesizing information and forming clear understanding.

TASK: Based on the observations provided, create a clear understanding by:
1. Identifying the CORE question or need
2. Determining the BEST approach to address it
3. Noting any IMPORTANT context or constraints

Format your response as:
CORE QUESTION: [state the main question]
APPROACH: [describe the best way to handle this]
CONTEXT: [any important notes]"""
    
    user_prompt = f"""OBSERVATIONS:
{all_obs}

Now provide your understanding in the format specified above."""
    
    understanding = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])
    
    # Extract a brief analysis for state
    analysis_brief = understanding.content.split('\n')[0][:150]
    
    return {
        "analysis": analysis_brief,
        "understanding": understanding.content,
        "current_phase": "decide"  # ← Node sets next phase
    }

def decide(state: OODAState) -> dict:
    """DECIDE: Make a clear decision on what action to take"""
    
    system_prompt = """You are a decision-maker. Your job is to decide the best action based on your understanding.

TASK: Decide what action to take. Choose ONE of these options:
1. ANSWER_DIRECTLY - You have enough information to answer the question directly
2. NEED_MORE_INFO - You need to ask clarifying questions
3. SEARCH_REQUIRED - You need to search for external information
4. COMPLEX_REASONING - The question requires step-by-step reasoning

Format your response as:
DECISION: [one of the 4 options above]
REASONING: [brief explanation why]
NEXT_STEP: [what specifically to do]"""
    
    user_prompt = f"""CURRENT UNDERSTANDING:
{state['understanding']}

ORIGINAL QUESTION: {state['user_input']}

Now make your decision using the format above."""
    
    decision = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])
    
    decision_content = decision.content
    
    # Extract decision type for action plan
    decision_type = "ANSWER_DIRECTLY"  # default
    if "NEED_MORE_INFO" in decision_content:
        decision_type = "NEED_MORE_INFO"
    elif "SEARCH_REQUIRED" in decision_content:
        decision_type = "SEARCH_REQUIRED"
    elif "COMPLEX_REASONING" in decision_content:
        decision_type = "COMPLEX_REASONING"
    
    return {
        "decision": decision_content,
        "action_plan": f"Execute {decision_type} strategy",
        "current_phase": "act"  # ← Node sets next phase
    }

def act(state: OODAState) -> dict:
    """ACT: Execute the decision and provide final response"""
    
    system_prompt = """You are a helpful AI assistant. Your job is to provide clear, accurate, and helpful responses.

TASK: Based on the action plan and decision made, provide a complete response to the user's original question.

Guidelines:
- Be clear and direct
- Use examples when helpful
- Structure your response well
- If you're uncertain, say so
- Keep it concise but complete"""
    
    user_prompt = f"""ACTION PLAN: {state['action_plan']}

DECISION DETAILS:
{state['decision']}

ORIGINAL QUESTION: {state['user_input']}

Now provide your complete response to the user."""
    
    result = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])
    
    return {
        "action_result": result.content,
        "iteration": 1,
        "current_phase": "complete"  # ← Node sets next phase
    }

def route_phase(state: OODAState) -> str:
    """Route based on the phase that nodes have already determined"""
    current = state["current_phase"]
    
    # Special handling for complete phase to enable looping
    if current == "complete":
        # Check if we should loop for improvement
        if state["iteration"] < state["max_iterations"]:
            # Check if response seems incomplete (optional improvement logic)
            if len(state.get("action_result", "")) < 50:
                return "observe"  # Loop back for refinement
        return "end"  # Done - exit the loop
    
    # For all other phases, simply go where the node indicated
    return current

# Build the graph
workflow = StateGraph(OODAState)

# Add all phases as nodes
workflow.add_node("observe", observe)
workflow.add_node("orient", orient)
workflow.add_node("decide", decide)
workflow.add_node("act", act)

# Set entry point
workflow.set_entry_point("observe")

# Add conditional routing from each phase
# All phases use the same routing logic
workflow.add_conditional_edges(
    "observe",
    route_phase,
    {
        "observe": "observe",
        "orient": "orient",
        "decide": "decide",
        "act": "act",
        "end": END
    }
)

workflow.add_conditional_edges(
    "orient",
    route_phase,
    {
        "observe": "observe",
        "orient": "orient",
        "decide": "decide",
        "act": "act",
        "end": END
    }
)

workflow.add_conditional_edges(
    "decide",
    route_phase,
    {
        "observe": "observe",
        "orient": "orient",
        "decide": "decide",
        "act": "act",
        "end": END
    }
)

workflow.add_conditional_edges(
    "act",
    route_phase,
    {
        "observe": "observe",
        "orient": "orient",
        "decide": "decide",
        "act": "act",
        "end": END
    }
)

# Compile
app = workflow.compile()

# Test the OODA agent
def run_ooda_agent(question: str, max_iter: int = 1):
    """Run OODA agent with a question"""
    
    initial_state = {
        "user_input": question,
        "observations": [],
        "context": {},
        "analysis": "",
        "understanding": "",
        "decision": "",
        "action_plan": "",
        "action_result": "",
        "current_phase": "observe",  # Start at observe
        "iteration": 0,
        "max_iterations": max_iter
    }
    
    print(f"\n{'='*60}")
    print(f"QUESTION: {question}")
    print(f"{'='*60}\n")
    
    # Stream through phases
    for step in app.stream(initial_state):
        for node_name, node_output in step.items():
            print(f"\n📍 NODE: {node_name.upper()}")
            print(f"{'─'*60}")
            
            # Show what phase this node set for next
            next_phase = node_output.get('current_phase', 'unknown')
            print(f"Next phase will be: {next_phase}")
            
            if node_name == "observe":
                obs_list = node_output.get('observations', [])
                print(f"Observations collected: {len(obs_list)}")
                if obs_list:
                    print(f"Latest: {obs_list[-1][:200]}...")
                    
            elif node_name == "orient":
                analysis = node_output.get('analysis', '')
                print(f"Analysis: {analysis}")
                understanding = node_output.get('understanding', '')
                print(f"\nFull understanding:\n{understanding[:300]}...")
                
            elif node_name == "decide":
                decision = node_output.get('decision', '')
                print(f"Decision made:\n{decision[:300]}...")
                
            elif node_name == "act":
                result = node_output.get('action_result', '')
                print(f"\n{'='*60}")
                print(f"✅ FINAL RESPONSE:")
                print(f"{'='*60}")
                print(f"\n{result}\n")
                print(f"{'='*60}\n")
    
    # Return final state
    final_state = app.invoke(initial_state)
    return final_state

# Example usage
if __name__ == "__main__":
    # Test 1: Simple question
    print("\n🧪 TEST 1: Simple Question")
    result1 = run_ooda_agent(
        "What are the main benefits of using local LLMs like Ollama?"
    )
    
    # Test 2: Complex question
    print("\n🧪 TEST 2: Complex Question")
    result2 = run_ooda_agent(
        "Explain how the OODA loop applies to AI agent design"
    )
    
    # Test 3: Technical question
    print("\n🧪 TEST 3: Technical Question")
    result3 = run_ooda_agent(
        "How do I optimize memory usage when running 7B models on an RTX 4070?"
    )
```

---

## 🔄 Advanced Pattern: Reflection in the Loop

Let's enhance our OODA agent with **reflection** - the agent critiques its own output.

```python
from typing import TypedDict, Annotated, Literal
from operator import add
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage

class ReflectiveOODAState(TypedDict):
    user_input: str
    observations: Annotated[list[str], add]
    understanding: str
    decision: str
    action_result: str
    reflection: str  # New: self-critique
    quality_score: int  # New: 1-10 rating
    current_phase: Literal["observe", "orient", "decide", "act", "reflect", "complete"]
    iteration: Annotated[int, add]
    max_iterations: int

llm = ChatOllama(model="llama3.2", temperature=0.7)

# Reuse observe, orient, decide from before
def observe(state: ReflectiveOODAState) -> dict:
    observations = [f"User request: {state['user_input']}"]
    
    # Include previous reflection if exists
    if state.get("reflection"):
        observations.append(f"Previous reflection: {state['reflection']}")
    
    return {
        "observations": observations,
        "current_phase": "orient"
    }

def orient(state: ReflectiveOODAState) -> dict:
    all_obs = "\n".join(state["observations"])
    
    prompt = f"""Observations:
{all_obs}

Summarize the core question and approach."""
    
    understanding = llm.invoke([SystemMessage(content=prompt)])
    
    return {
        "understanding": understanding.content,
        "current_phase": "decide"
    }

def decide(state: ReflectiveOODAState) -> dict:
    prompt = f"""Understanding: {state['understanding']}

What action should I take?"""
    
    decision = llm.invoke([SystemMessage(content=prompt)])
    
    return {
        "decision": decision.content,
        "current_phase": "act"
    }

def act(state: ReflectiveOODAState) -> dict:
    prompt = f"""Action: {state['decision']}
Question: {state['user_input']}

Provide your response."""
    
    result = llm.invoke([SystemMessage(content=prompt)])
    
    return {
        "action_result": result.content,
        "current_phase": "reflect"  # Go to reflection instead of complete
    }

def reflect(state: ReflectiveOODAState) -> dict:
    """
    REFLECT: Critique the action result
    
    This adds a feedback loop to OODA:
    Observe → Orient → Decide → Act → REFLECT → (back to Observe if needed)
    """
    
    prompt = f"""Evaluate this response:

Original question: {state['user_input']}
Response: {state['action_result']}

Rate quality (1-10) and provide critique:
- Is it accurate?
- Is it complete?
- Is it clear?
- What could be improved?

Format: SCORE: X/10
CRITIQUE: [your critique]"""
    
    reflection = llm.invoke([SystemMessage(content=prompt)])
    reflection_text = reflection.content
    
    # Extract score (simple parsing)
    score = 7  # Default
    if "SCORE:" in reflection_text:
        try:
            score_part = reflection_text.split("SCORE:")[1].split("/")[0].strip()
            score = int(score_part)
        except:
            pass
    
    return {
        "reflection": reflection_text,
        "quality_score": score,
        "current_phase": "complete",
        "iteration": 1
    }

def route_reflective(state: ReflectiveOODAState) -> str:
    """Route with reflection consideration"""
    phase = state["current_phase"]
    
    if phase == "observe":
        return "orient"
    elif phase == "orient":
        return "decide"
    elif phase == "decide":
        return "act"
    elif phase == "act":
        return "reflect"
    elif phase == "complete":
        # Check if we should improve
        if state["iteration"] >= state["max_iterations"]:
            return "end"
        
        # If quality is low, loop back
        if state.get("quality_score", 10) < 7:
            return "observe"  # Try again with reflection as context
        
        return "end"
    
    return "end"

# Build reflective graph
workflow = StateGraph(ReflectiveOODAState)

workflow.add_node("observe", observe)
workflow.add_node("orient", orient)
workflow.add_node("decide", decide)
workflow.add_node("act", act)
workflow.add_node("reflect", reflect)

workflow.set_entry_point("observe")

for phase in ["observe", "orient", "decide", "act", "complete"]:
    workflow.add_conditional_edges(
        phase,
        route_reflective,
        {
            "observe": "observe",
            "orient": "orient",
            "decide": "decide",
            "act": "act",
            "reflect": "reflect",
            "end": END
        }
    )

# Also add routing from reflect
workflow.add_conditional_edges(
    "reflect",
    route_reflective,
    {
        "observe": "observe",
        "end": END
    }
)

app_reflective = workflow.compile()

# Test reflective agent
def run_reflective_agent(question: str):
    initial_state = {
        "user_input": question,
        "observations": [],
        "understanding": "",
        "decision": "",
        "action_result": "",
        "reflection": "",
        "quality_score": 0,
        "current_phase": "observe",
        "iteration": 0,
        "max_iterations": 2
    }
    
    print(f"\n{'='*60}")
    print(f"QUESTION: {question}")
    print(f"{'='*60}\n")
    
    for step in app_reflective.stream(initial_state):
        for node_name, node_output in step.items():
            phase = node_output.get("current_phase", "unknown")
            print(f"📍 {phase.upper()}")
            
            if phase == "reflect":
                score = node_output.get("quality_score", 0)
                print(f"   Quality Score: {score}/10")
                print(f"   Reflection: {node_output.get('reflection', '')[:100]}...")
            elif phase == "complete":
                print(f"\n✅ FINAL ANSWER:")
                print(f"{node_output.get('action_result', '')}\n")
            
            print()
    
    return app_reflective.invoke(initial_state)

# Test
if __name__ == "__main__":
    result = run_reflective_agent(
        "What is the difference between Ollama and OpenAI?"
    )
    print(f"\nFinal iterations: {result['iteration']}")
    print(f"Final quality score: {result['quality_score']}/10")
```

---

## 🎯 Practical Patterns: When to Use OODA

### Pattern 1: Research Agent

```python
# OBSERVE: Identify search queries needed
# ORIENT: Analyze search results
# DECIDE: Determine if more searches needed
# ACT: Synthesize final answer
# REFLECT: Check answer quality
```

### Pattern 2: Code Generation Agent

```python
# OBSERVE: Understand requirements
# ORIENT: Plan code structure
# DECIDE: Choose implementation approach
# ACT: Generate code
# REFLECT: Test code, check for errors
```

### Pattern 3: Customer Support Agent

```python
# OBSERVE: Gather user issue details
# ORIENT: Classify issue type
# DECIDE: Choose resolution path
# ACT: Provide solution
# REFLECT: Verify user satisfaction
```

---

## 🏭 Production Considerations

### 1. **Timeout Control**

```python
import time

class TimedOODAState(TypedDict):
    # ... other fields
    start_time: float
    timeout_seconds: int

def act_with_timeout(state: TimedOODAState) -> dict:
    elapsed = time.time() - state["start_time"]
    
    if elapsed > state["timeout_seconds"]:
        return {
            "action_result": "Timeout - returning best effort response",
            "current_phase": "complete"
        }
    
    # Normal processing
    result = llm.invoke(...)
    return {"action_result": result.content}
```

### 2. **Cost Tracking**

```python
class CostTrackingState(TypedDict):
    # ... other fields
    llm_calls: Annotated[int, add]
    total_tokens: Annotated[int, add]

def observe_with_tracking(state: CostTrackingState) -> dict:
    response = llm.invoke(...)
    
    # Track usage (if available from llm response)
    tokens = getattr(response, 'usage', {}).get('total_tokens', 0)
    
    return {
        "observations": [response.content],
        "llm_calls": 1,
        "total_tokens": tokens
    }
```

### 3. **Graceful Degradation**

```python
def act_with_fallback(state: OODAState) -> dict:
    try:
        # Try primary action
        result = llm.invoke(...)
        return {"action_result": result.content}
    
    except Exception as e:
        # Fallback to simpler response
        return {
            "action_result": f"Unable to complete full analysis. Error: {str(e)}",
            "current_phase": "complete"
        }
```

---

## ⚠️ Common Mistakes

### Mistake 1: Skipping Phases

```python
# ❌ Jumping directly to ACT without Orient/Decide
def bad_agent(state):
    # No analysis, just act
    return llm.invoke(state["input"])

# ✅ Go through all phases
# Each phase adds value and reduces errors
```

### Mistake 2: No Exit Condition

```python
# ❌ Infinite loop
def route(state):
    if state["quality_score"] < 10:  # Will never reach 10!
        return "observe"
    return "end"

# ✅ Add max iterations
def route(state):
    if state["iteration"] >= state["max_iterations"]:
        return "end"
    if state["quality_score"] < 7:
        return "observe"
    return "end"
```

### Mistake 3: Heavy Processing in Every Observation

```python
# ❌ Expensive operations every loop
def observe(state):
    # Don't re-fetch everything every time
    all_data = fetch_entire_database()  # Slow!
    
# ✅ Cache and incrementally update
def observe(state):
    if not state.get("context"):
        state["context"] = fetch_initial_data()
    # Just update what changed
```

---

## 🧠 Key Concepts to Remember

1. **OODA = Observe → Orient → Decide → Act** (+ optional Reflect)
2. **Each phase has a specific purpose** - don't skip
3. **Reflection adds a quality control loop**
4. **Always have max iteration limits**
5. **Use Ollama for free, local LLM testing**
6. **State should accumulate insights across iterations**
7. **Route based on quality and iteration count**

---

## 🚀 What's Next?

In **Chapter 4**, we'll explore:
- **Chain-of-Thought (CoT)** reasoning
- Zero-shot vs Few-shot CoT
- Self-Consistency with multiple reasoning paths
- Implementing CoT in LangGraph with Ollama
- When to use CoT vs other reasoning paradigms

---

## ✅ Chapter 3 Complete!

**You now understand:**
- ✅ The OODA Loop framework (Observe-Orient-Decide-Act)
- ✅ How to structure agent perception and action
- ✅ Adding reflection for self-improvement
- ✅ Using Ollama (llama3.2) instead of OpenAI
- ✅ Building iterative agent loops
- ✅ Production patterns (timeouts, cost tracking, fallbacks)
- ✅ Common pitfalls and how to avoid them

**Ready for Chapter 4?** Just say "Continue to Chapter 4" or ask any questions about Chapter 3!