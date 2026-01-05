# Chapter 11: Agent Design Patterns

## 🎯 The Problem: Complex Tasks Need Specialized Agents

**Single-agent limitations:**
- ❌ Can't handle diverse task types well
- ❌ No specialization for different domains
- ❌ Hard to scale complexity
- ❌ Difficult to maintain and debug
- ❌ Poor separation of concerns

**Multi-agent systems** solve this by dividing work among specialized agents, each optimized for specific tasks.

---

## 🔀 Part 1: Router Agent Pattern
### What Is the Router Pattern?

The **Router Pattern** is a classification-based approach where a central router agent analyzes incoming tasks and routes them to specialized agents based on task characteristics.

```
                    ┌──────────┐
         Task ─────►│  ROUTER  │
                    │(Classify)│
                    └────┬─────┘
                         │
         ┌───────────────┼───────────────┐
         ↓               ↓               ↓
    ┌────────┐      ┌────────┐     ┌────────┐
    │Specialist│    │Specialist│   │Specialist│
    │   A     │    │   B     │   │   C     │
    └────────┘      └────────┘     └────────┘
```

### Core Concept

Think of a router like a **triage nurse** in an emergency room:
- Patients arrive with different issues
- Nurse quickly assesses symptoms
- Routes to appropriate specialist (cardiology, orthopedics, etc.)
- Doesn't treat the patient, just directs them

### Key Characteristics

**1. Single Entry Point:**
- All tasks enter through the router
- No direct access to specialists
- Router is the gatekeeper

**2. Classification Logic:**
- Router examines task characteristics
- Applies rules or uses LLM to classify
- Makes routing decision

**3. Specialized Agents:**
- Each specialist handles specific task types
- Optimized for their domain
- Don't need to know about other specialists

**4. One-Way Flow:**
- Router → Specialist → End
- Usually no loops back to router
- Linear execution after routing

### Router Classification Strategies

**1. Rule-Based Routing:**
```
IF task contains "code" OR "programming" → Code Specialist
IF task contains "data" OR "analyze" → Data Specialist
IF task contains "write" OR "content" → Writing Specialist
ELSE → General Agent
```

**Advantages:**
- Fast and predictable
- Easy to debug
- No LLM cost

**Disadvantages:**
- Brittle (misses edge cases)
- Hard to update (requires code changes)
- Limited to predefined patterns

**2. LLM-Based Routing:**
```
Router LLM: "Given this task, which specialist is most appropriate?"
→ Uses semantic understanding
→ Adapts to nuanced requests
→ More flexible
```

**Advantages:**
- Handles nuance and ambiguity
- Adapts to new task types
- More intelligent classification

**Disadvantages:**
- Adds LLM call overhead
- Less predictable
- Costs money per routing decision

**3. Hybrid Routing:**
```
1. Try rule-based first (fast path)
2. If uncertain, use LLM (fallback)
3. Cache LLM decisions for similar tasks
```

Best of both worlds.

### When to Use Router Pattern

✅ **Use Router When:**
- Tasks clearly fall into distinct categories
- You have multiple specialized agents
- Each specialist can handle their tasks independently
- Classification is relatively straightforward
- You want simple, understandable flow

❌ **Don't Use Router When:**
- Tasks require multiple specialists working together
- Task classification is highly complex
- You need iterative refinement
- Specialists need to collaborate

### Router Pattern Variations

**1. Simple Router:**
- One router, multiple specialists
- Single routing decision
- No further coordination

**2. Multi-Level Router:**
```
Main Router → Category (Tech/Business/Creative)
              ↓
Sub-Router → Specific Specialist
```

**3. Confidence-Based Router:**
```
High confidence → Route directly
Medium confidence → Route with warning flag
Low confidence → Route to generalist or ask for clarification
```

### Real-World Analogies

**1. Call Center:**
- Automated menu system (router)
- "Press 1 for sales, 2 for support..."
- Routes to appropriate department (specialist)

**2. Mail Sorting:**
- Postal service sorts mail by zip code (router)
- Delivers to appropriate postal carrier (specialist)
- Each carrier handles their route

**3. Restaurant Host:**
- Host greets customers (router)
- Seats them in appropriate section (specialist)
- Server handles that table (execution)

### When to Use: Task Classification and Routing

**Router Agent** analyzes tasks and routes them to appropriate specialist agents.

```python
from typing import TypedDict, Annotated, Sequence, Literal
from operator import add
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== ROUTER AGENT PATTERN ====================

class RouterState(TypedDict):
    """State for router pattern"""
    messages: Annotated[Sequence[BaseMessage], add]
    task_type: Literal["code", "data", "text", "math"]
    specialist_response: str

# Initialize LLM
llm = ChatOllama(model="llama3.2", temperature=0.3)

# Router prompt
router_prompt = ChatPromptTemplate.from_messages([
    ("human", """Analyze this task and classify it:

Task: {task}

Categories:
- CODE: Programming, debugging, code generation
- DATA: Data analysis, statistics, visualization
- TEXT: Writing, summarization, translation
- MATH: Mathematical calculations, proofs

Respond with ONLY one word: CODE, DATA, TEXT, or MATH

Classification:""")
])

router_chain = router_prompt | llm

# Node: Router
def router_node(state: RouterState) -> dict:
    """Route task to appropriate specialist"""
    
    try:
        task = state["messages"][-1].content
        
        logger.info(f"Routing task: {task[:50]}...")
        
        response = router_chain.invoke({"task": task})
        classification = response.content.strip().upper()
        
        # Map to task types
        task_map = {
            "CODE": "code",
            "DATA": "data",
            "TEXT": "text",
            "MATH": "math"
        }
        
        task_type = task_map.get(classification, "text")  # Default to text
        
        logger.info(f"Routed to: {task_type}")
        
        return {"task_type": task_type}
    
    except Exception as e:
        logger.error(f"Router error: {e}")
        return {"task_type": "text"}

# Specialist agents
def code_specialist(state: RouterState) -> dict:
    """Specialized for coding tasks"""
    
    task = state["messages"][-1].content
    
    code_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert programmer. Provide code solutions with explanations."),
        ("human", "{task}")
    ])
    
    chain = code_prompt | llm
    response = chain.invoke({"task": task})
    
    return {
        "specialist_response": response.content,
        "messages": [AIMessage(content=response.content)]
    }

def data_specialist(state: RouterState) -> dict:
    """Specialized for data analysis"""
    
    task = state["messages"][-1].content
    
    data_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a data scientist. Provide analytical insights and statistical solutions."),
        ("human", "{task}")
    ])
    
    chain = data_prompt | llm
    response = chain.invoke({"task": task})
    
    return {
        "specialist_response": response.content,
        "messages": [AIMessage(content=response.content)]
    }

def text_specialist(state: RouterState) -> dict:
    """Specialized for text tasks"""
    
    task = state["messages"][-1].content
    
    text_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a writing expert. Provide clear, well-structured text."),
        ("human", "{task}")
    ])
    
    chain = text_prompt | llm
    response = chain.invoke({"task": task})
    
    return {
        "specialist_response": response.content,
        "messages": [AIMessage(content=response.content)]
    }

def math_specialist(state: RouterState) -> dict:
    """Specialized for math tasks"""
    
    task = state["messages"][-1].content
    
    math_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a mathematician. Provide step-by-step solutions with clear explanations."),
        ("human", "{task}")
    ])
    
    chain = math_prompt | llm
    response = chain.invoke({"task": task})
    
    return {
        "specialist_response": response.content,
        "messages": [AIMessage(content=response.content)]
    }

# Router function
def route_to_specialist(state: RouterState) -> str:
    """Route based on task type"""
    task_type = state["task_type"]
    logger.info(f"Routing to {task_type} specialist")
    return task_type

# Build router graph
router_workflow = StateGraph(RouterState)

# Add nodes
router_workflow.add_node("router", router_node)
router_workflow.add_node("code", code_specialist)
router_workflow.add_node("data", data_specialist)
router_workflow.add_node("text", text_specialist)
router_workflow.add_node("math", math_specialist)

# Set entry
router_workflow.set_entry_point("router")

# Conditional routing
router_workflow.add_conditional_edges(
    "router",
    route_to_specialist,
    {
        "code": "code",
        "data": "data",
        "text": "text",
        "math": "math"
    }
)

# All specialists go to END
router_workflow.add_edge("code", END)
router_workflow.add_edge("data", END)
router_workflow.add_edge("text", END)
router_workflow.add_edge("math", END)

# Compile
router_agent = router_workflow.compile()

# Test
def test_router_agent(task: str):
    """Test router agent"""
    
    result = router_agent.invoke({
        "messages": [HumanMessage(content=task)],
        "task_type": "text",
        "specialist_response": ""
    })
    
    print(f"\n{'='*60}")
    print(f"ROUTER AGENT")
    print(f"{'='*60}")
    print(f"Task: {task}")
    print(f"Routed to: {result['task_type']}")
    print(f"Response: {result['specialist_response'][:200]}...")

if __name__ == "__main__":
    test_router_agent("Write a Python function to calculate fibonacci numbers")
    test_router_agent("Analyze this dataset and find correlations")
    test_router_agent("Write a blog post about AI")
    test_router_agent("Solve the equation x^2 + 5x + 6 = 0")
```

---

## 👔 Part 2: Supervisor Agent Pattern
### What Is the Supervisor Pattern?

The **Supervisor Pattern** uses a manager agent that orchestrates multiple worker agents, assigns tasks, monitors progress, and aggregates results. Unlike a router, the supervisor maintains ongoing control throughout execution.

```
                ┌──────────────┐
                │  SUPERVISOR  │
                │  - Plans     │
                │  - Assigns   │
                │  - Monitors  │
                │  - Aggregates│
                └──────┬───────┘
                       │
          ┌────────────┼────────────┐
          ↓            ↓            ↓
     ┌────────┐   ┌────────┐   ┌────────┐
     │Worker A│   │Worker B│   │Worker C│
     │(Exec)  │   │(Exec)  │   │(Exec)  │
     └────────┘   └────────┘   └────────┘
```

### Core Concept

Think of a supervisor like a **project manager**:
- Breaks down project into tasks
- Assigns tasks to team members
- Monitors progress
- Adjusts assignments if needed
- Combines results into final deliverable

### Key Characteristics

**1. Active Management:**
- Supervisor doesn't just route once
- Maintains control throughout
- Can make dynamic decisions
- Tracks worker status

**2. Task Decomposition:**
- Supervisor breaks complex tasks into subtasks
- Creates execution plan
- Determines dependencies

**3. Work Assignment:**
- Assigns tasks to appropriate workers
- Considers worker capabilities
- May load balance

**4. Result Aggregation:**
- Collects outputs from workers
- Combines/synthesizes results
- Produces final output

**5. Monitoring:**
- Can check worker progress
- May reassign tasks
- Handles worker failures

### Supervisor Responsibilities

**Phase 1 - Planning:**
```
Complex Task → Supervisor analyzes → Creates subtask plan
Example:
  "Write report" → [Research, Draft, Review, Format]
```

**Phase 2 - Assignment:**
```
Supervisor evaluates:
- Worker capabilities
- Worker availability
- Task requirements
→ Assigns task to best worker
```

**Phase 3 - Execution:**
```
Supervisor → Worker 1 (research)
          → Worker 2 (draft) [waits for Worker 1]
          → Worker 3 (review) [waits for Worker 2]
```

**Phase 4 - Aggregation:**
```
Supervisor collects all results
Combines into coherent output
May do final processing
```

### Supervisor Strategies

**1. Sequential Execution:**
```
Worker A completes → Supervisor → Worker B starts
→ Simple but slow
```

**2. Parallel Execution:**
```
Supervisor → Worker A, B, C start simultaneously
→ Fast but requires independent tasks
```

**3. Pipeline Execution:**
```
Worker A → produces → Worker B → produces → Worker C
Like assembly line
```

**4. Adaptive Execution:**
```
Supervisor checks Worker A results
Decides whether to:
- Proceed to Worker B
- Retry with Worker A
- Change approach entirely
```

### Router vs Supervisor: Key Differences

| Aspect | Router | Supervisor |
|--------|--------|------------|
| **Control** | Routes once, hands off | Maintains control throughout |
| **Planning** | No planning | Creates execution plan |
| **Monitoring** | No monitoring | Tracks progress |
| **Aggregation** | No aggregation | Combines results |
| **Complexity** | Simple, single decision | Complex, ongoing management |
| **Use Case** | Classification → Execution | Orchestration of workflow |

### When to Use Supervisor Pattern

✅ **Use Supervisor When:**
- Complex task needs breakdown
- Multiple workers must collaborate
- Results need aggregation/synthesis
- Execution order matters
- You need progress monitoring
- Tasks have dependencies

❌ **Don't Use Supervisor When:**
- Simple, single-step tasks
- No need for coordination
- Workers are independent
- Router pattern is sufficient

### Real-World Analogies

**1. Film Director:**
- Script (complex task)
- Director (supervisor) coordinates actors, camera crew, sound
- Each specialist does their part
- Director ensures everything comes together

**2. Orchestra Conductor:**
- Musical score (plan)
- Conductor (supervisor) coordinates musicians
- Each section plays their part
- Conductor ensures harmony

**3. Construction Foreman:**
- Building plans (task breakdown)
- Foreman (supervisor) coordinates trades (workers)
- Electricians, plumbers, carpenters
- Foreman ensures quality and timing

### When to Use: Orchestrating Multiple Agents

**Supervisor Agent** manages worker agents, assigns tasks, and aggregates results.

```python
from typing import TypedDict, Annotated, Sequence, List, Literal
from operator import add
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== SUPERVISOR PATTERN ====================

class SupervisorState(TypedDict):
    """State for supervisor pattern"""
    messages: Annotated[Sequence[BaseMessage], add]
    task_plan: List[dict]  # List of subtasks
    current_subtask_idx: int
    worker_results: Annotated[List[dict], add]
    final_result: str
    next_worker: Literal["researcher", "writer", "reviewer", "supervisor", "finish"]

# Initialize
llm = ChatOllama(model="llama3.2", temperature=0.3)

# Supervisor prompts
planning_prompt = ChatPromptTemplate.from_messages([
    ("human", """You are a supervisor managing a team of workers.

Workers available:
- RESEARCHER: Gathers information and facts
- WRITER: Creates content based on research
- REVIEWER: Reviews and improves content

Task: {task}

Create a plan by breaking this into subtasks for each worker.

Respond in JSON format:
{{
  "subtasks": [
    {{"worker": "RESEARCHER", "task": "gather info on X"}},
    {{"worker": "WRITER", "task": "write draft"}},
    {{"worker": "REVIEWER", "task": "review and improve"}}
  ]
}}

Plan:""")
])

supervisor_router_prompt = ChatPromptTemplate.from_messages([
    ("human", """Based on the current progress, decide the next step:

Original task: {original_task}
Completed subtasks: {completed}
Current results: {results}

Options:
- RESEARCHER: Need more information
- WRITER: Ready to write
- REVIEWER: Content ready for review
- FINISH: All done

Respond with ONLY one word: RESEARCHER, WRITER, REVIEWER, or FINISH

Next:""")
])

planning_chain = planning_prompt | llm
supervisor_router_chain = supervisor_router_prompt | llm

# Node: Supervisor - Planning
def supervisor_plan(state: SupervisorState) -> dict:
    """Supervisor creates task plan"""
    
    try:
        task = state["messages"][-1].content
        
        logger.info("Supervisor creating plan...")
        
        response = planning_chain.invoke({"task": task})
        response_text = response.content.strip()
        
        # Parse JSON
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        import json
        try:
            plan_data = json.loads(response_text)
            subtasks = plan_data.get("subtasks", [])
        except:
            # Fallback plan
            subtasks = [
                {"worker": "RESEARCHER", "task": "Research the topic"},
                {"worker": "WRITER", "task": "Write content"},
                {"worker": "REVIEWER", "task": "Review content"}
            ]
        
        logger.info(f"Created plan with {len(subtasks)} subtasks")
        
        return {
            "task_plan": subtasks,
            "current_subtask_idx": 0
        }
    
    except Exception as e:
        logger.error(f"Planning error: {e}")
        return {
            "task_plan": [],
            "current_subtask_idx": 0
        }

# Node: Supervisor - Routing
def supervisor_route(state: SupervisorState) -> dict:
    """Supervisor decides next worker"""
    
    try:
        original_task = state["messages"][0].content
        
        # Format completed work
        completed = []
        for i, result in enumerate(state["worker_results"]):
            completed.append(f"{i+1}. {result['worker']}: {result['task'][:50]}...")
        
        completed_text = "\n".join(completed) if completed else "None yet"
        
        # Get latest results
        results_text = "None yet"
        if state["worker_results"]:
            latest = state["worker_results"][-1]
            results_text = latest.get("result", "")[:200]
        
        response = supervisor_router_chain.invoke({
            "original_task": original_task,
            "completed": completed_text,
            "results": results_text
        })
        
        next_worker = response.content.strip().upper()
        
        worker_map = {
            "RESEARCHER": "researcher",
            "WRITER": "writer",
            "REVIEWER": "reviewer",
            "FINISH": "finish"
        }
        
        next_node = worker_map.get(next_worker, "finish")
        
        logger.info(f"Supervisor routing to: {next_node}")
        
        return {"next_worker": next_node}
    
    except Exception as e:
        logger.error(f"Routing error: {e}")
        return {"next_worker": "finish"}

# Worker nodes
def researcher_node(state: SupervisorState) -> dict:
    """Researcher worker"""
    
    task = state["messages"][-1].content
    
    # Check if there's a specific subtask
    if state["task_plan"] and state["current_subtask_idx"] < len(state["task_plan"]):
        subtask = state["task_plan"][state["current_subtask_idx"]]
        if subtask["worker"].upper() == "RESEARCHER":
            task = subtask["task"]
    
    logger.info(f"Researcher working on: {task[:50]}...")
    
    researcher_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a researcher. Gather facts and information."),
        ("human", "{task}")
    ])
    
    chain = researcher_prompt | llm
    response = chain.invoke({"task": task})
    
    result = {
        "worker": "researcher",
        "task": task,
        "result": response.content
    }
    
    return {
        "worker_results": [result],
        "current_subtask_idx": state["current_subtask_idx"] + 1
    }

def writer_node(state: SupervisorState) -> dict:
    """Writer worker"""
    
    task = state["messages"][-1].content
    
    # Get context from previous workers
    context = ""
    for result in state["worker_results"]:
        context += f"\n{result['worker']}: {result['result']}\n"
    
    logger.info(f"Writer working with context from {len(state['worker_results'])} previous workers")
    
    writer_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a writer. Create clear, engaging content based on research."),
        ("human", """Task: {task}

Context from previous work:
{context}

Write content:""")
    ])
    
    chain = writer_prompt | llm
    response = chain.invoke({"task": task, "context": context})
    
    result = {
        "worker": "writer",
        "task": task,
        "result": response.content
    }
    
    return {
        "worker_results": [result],
        "current_subtask_idx": state["current_subtask_idx"] + 1
    }

def reviewer_node(state: SupervisorState) -> dict:
    """Reviewer worker"""
    
    # Get content to review
    content_to_review = ""
    for result in state["worker_results"]:
        if result["worker"] == "writer":
            content_to_review = result["result"]
            break
    
    if not content_to_review:
        content_to_review = "No content to review yet"
    
    logger.info("Reviewer reviewing content...")
    
    reviewer_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a reviewer. Provide constructive feedback and improvements."),
        ("human", """Review this content and provide an improved version:

{content}

Improved version:""")
    ])
    
    chain = reviewer_prompt | llm
    response = chain.invoke({"content": content_to_review})
    
    result = {
        "worker": "reviewer",
        "task": "Review and improve content",
        "result": response.content
    }
    
    return {
        "worker_results": [result],
        "current_subtask_idx": state["current_subtask_idx"] + 1
    }

# Node: Finalize
def finalize_node(state: SupervisorState) -> dict:
    """Finalize and aggregate results"""
    
    logger.info("Finalizing results...")
    
    # Get the latest/best result
    final = "No results produced"
    
    # Prioritize reviewer, then writer, then researcher
    for result in reversed(state["worker_results"]):
        if result["worker"] == "reviewer":
            final = result["result"]
            break
    
    if final == "No results produced":
        for result in reversed(state["worker_results"]):
            if result["worker"] == "writer":
                final = result["result"]
                break
    
    if final == "No results produced" and state["worker_results"]:
        final = state["worker_results"][-1]["result"]
    
    return {
        "final_result": final,
        "messages": [AIMessage(content=final)]
    }

# Router
def route_supervisor(state: SupervisorState) -> str:
    """Route based on supervisor decision"""
    return state["next_worker"]

# Build supervisor graph
supervisor_workflow = StateGraph(SupervisorState)

# Add nodes
supervisor_workflow.add_node("plan", supervisor_plan)
supervisor_workflow.add_node("supervisor", supervisor_route)
supervisor_workflow.add_node("researcher", researcher_node)
supervisor_workflow.add_node("writer", writer_node)
supervisor_workflow.add_node("reviewer", reviewer_node)
supervisor_workflow.add_node("finalize", finalize_node)

# Define flow
supervisor_workflow.set_entry_point("plan")
supervisor_workflow.add_edge("plan", "supervisor")

supervisor_workflow.add_conditional_edges(
    "supervisor",
    route_supervisor,
    {
        "researcher": "researcher",
        "writer": "writer",
        "reviewer": "reviewer",
        "finish": "finalize"
    }
)

# Workers route back to supervisor
supervisor_workflow.add_edge("researcher", "supervisor")
supervisor_workflow.add_edge("writer", "supervisor")
supervisor_workflow.add_edge("reviewer", "supervisor")

supervisor_workflow.add_edge("finalize", END)

# Compile
supervisor_agent = supervisor_workflow.compile()

# Test
def test_supervisor_agent(task: str):
    """Test supervisor agent"""
    
    result = supervisor_agent.invoke({
        "messages": [HumanMessage(content=task)],
        "task_plan": [],
        "current_subtask_idx": 0,
        "worker_results": [],
        "final_result": "",
        "next_worker": "researcher"
    })
    
    print(f"\n{'='*60}")
    print(f"SUPERVISOR AGENT")
    print(f"{'='*60}")
    print(f"Task: {task}")
    print(f"Workers used: {len(result['worker_results'])}")
    for i, worker in enumerate(result['worker_results'], 1):
        print(f"  {i}. {worker['worker']}: {worker['task'][:50]}...")
    print(f"\nFinal Result:\n{result['final_result'][:300]}...")

if __name__ == "__main__":
    test_supervisor_agent("Write a short article about the benefits of AI in healthcare")
```

---

## 🏗️ Part 3: Hierarchical Agent Pattern
### What Is the Hierarchical Pattern?

The **Hierarchical Pattern** creates a **tree structure** of managers and workers, with multiple levels of supervision. Think organizational chart: executives, middle managers, workers.

```
            ┌─────────────────┐
            │ EXECUTIVE AGENT │ (Top Level)
            └────────┬────────┘
                     │
        ┌────────────┼────────────┐
        ↓            ↓            ↓
   ┌─────────┐  ┌─────────┐  ┌─────────┐
   │Manager A│  │Manager B│  │Manager C│ (Middle)
   └────┬────┘  └────┬────┘  └────┬────┘
        │            │            │
    ┌───┴───┐    ┌───┴───┐    ┌───┴───┐
    ↓       ↓    ↓       ↓    ↓       ↓
  [W1]   [W2]  [W3]   [W4]  [W5]   [W6] (Workers)
```

### Core Concept

Think of **corporate structure**:
- CEO (executive agent) sets high-level strategy
- VPs (middle managers) handle divisions
- Team leads (sub-managers) coordinate teams
- Individual contributors (workers) execute tasks

### Key Characteristics

**1. Multi-Level Structure:**
- Top level: Strategic planning
- Middle levels: Tactical coordination
- Bottom level: Task execution

**2. Delegation Cascade:**
- Each level delegates to level below
- Authority flows downward
- Results flow upward

**3. Scope Abstraction:**
- Top level: High-level goals
- Middle level: Phase/component management
- Bottom level: Specific task execution

**4. Hierarchical Communication:**
- Primarily vertical (up/down)
- Limited horizontal (peer-to-peer)
- Chain of command

### Why Use Hierarchy?

**Problem it solves:**
```
Without Hierarchy:
CEO → directly manages 50 employees
→ Overwhelming, impossible to scale

With Hierarchy:
CEO → 5 VPs → each manages 10 employees
→ Manageable spans of control
```

**Span of Control:**
- Each manager typically oversees 3-7 direct reports
- Allows for effective management
- Scales to large organizations

### Hierarchical Levels

**Level 1 - Executive (Strategic):**
- Understands overall objective
- Breaks into major phases/components
- Assigns to middle managers
- Monitors high-level progress

**Level 2 - Middle Management (Tactical):**
- Takes a phase/component
- Breaks into specific tasks
- Assigns to workers
- Coordinates within their scope
- Reports progress upward

**Level 3 - Workers (Operational):**
- Executes specific tasks
- No further delegation
- Reports results upward

### Example Breakdown

**Task:** "Build a customer feedback system"

**Executive Agent thinks:**
```
Phase 1: Requirements & Design
Phase 2: Implementation
Phase 3: Testing & Deployment
```

**Manager A (Requirements) thinks:**
```
Task 1: Research existing solutions
Task 2: Gather stakeholder input
Task 3: Create specification document
```

**Worker A1 executes:**
```
"Research existing solutions" → Produces research report
```

### Hierarchical vs Flat Supervisor

| Aspect | Flat Supervisor | Hierarchical |
|--------|----------------|--------------|
| **Levels** | 2 (supervisor + workers) | 3+ levels |
| **Scalability** | Limited (supervisor bottleneck) | High (distributed management) |
| **Complexity** | Lower | Higher |
| **Task Scope** | Medium tasks | Very large, complex projects |
| **Supervision** | Centralized | Distributed across levels |

### When to Use Hierarchical Pattern

✅ **Use Hierarchical When:**
- Very large, complex projects
- Clear phases or components
- Need distributed management
- Many workers (>10)
- Task naturally decomposes into levels
- Different expertise needed at each level

❌ **Don't Use Hierarchical When:**
- Simple or medium complexity tasks
- Flat structure is sufficient
- Communication overhead outweighs benefits
- You need agility and quick decisions

### Hierarchical Challenges

**1. Communication Overhead:**
- Information passes through multiple levels
- Can slow decision-making
- Risk of miscommunication

**2. Rigidity:**
- Hard to adapt quickly
- Bureaucratic
- Changes require multi-level coordination

**3. Complexity:**
- More moving parts
- Harder to debug
- More points of failure

**4. Context Loss:**
- Executive doesn't see details
- Workers don't see big picture
- Middle managers must translate both ways

### Real-World Analogies

**1. Military Command:**
- General → Colonels → Captains → Soldiers
- Clear chain of command
- Scales to armies

**2. Corporate Structure:**
- CEO → VPs → Directors → Managers → ICs
- Divides responsibility
- Handles large organizations

**3. Government:**
- Federal → State → County → City
- Each level has its domain
- Coordinates within hierarchy


### When to Use: Complex Multi-Level Tasks

**Hierarchical Agents** create a tree structure with managers and sub-managers.

```python
from typing import TypedDict, Annotated, Sequence, List, Dict, Literal
from operator import add
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== HIERARCHICAL PATTERN ====================

class HierarchicalState(TypedDict):
    """State for hierarchical agents"""
    messages: Annotated[Sequence[BaseMessage], add]
    high_level_plan: List[str]
    current_phase: int
    phase_results: Dict[str, str]
    sub_team_results: Annotated[List[dict], add]
    final_output: str

# Initialize
llm = ChatOllama(model="llama3.2", temperature=0.3)

# High-level manager
def executive_manager(state: HierarchicalState) -> dict:
    """Top-level manager creates high-level plan"""
    
    try:
        task = state["messages"][-1].content
        
        logger.info("Executive Manager planning...")
        
        exec_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an executive manager. Break down complex projects into phases."),
            ("human", """Project: {task}

Create a 3-phase plan:
1. Phase 1: [description]
2. Phase 2: [description]
3. Phase 3: [description]

Plan:""")
        ])
        
        chain = exec_prompt | llm
        response = chain.invoke({"task": task})
        
        # Parse phases
        phases = []
        for line in response.content.split('\n'):
            if line.strip().startswith(('1.', '2.', '3.', 'Phase')):
                phases.append(line.strip())
        
        if not phases:
            phases = ["Research phase", "Development phase", "Review phase"]
        
        logger.info(f"Executive created {len(phases)}-phase plan")
        
        return {
            "high_level_plan": phases,
            "current_phase": 0
        }
    
    except Exception as e:
        logger.error(f"Executive planning error: {e}")
        return {
            "high_level_plan": ["Phase 1", "Phase 2", "Phase 3"],
            "current_phase": 0
        }

# Middle managers for each phase
def research_manager(state: HierarchicalState) -> dict:
    """Manages research phase"""
    
    logger.info("Research Manager executing phase 1...")
    
    task = state["messages"][-1].content
    
    research_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a research manager. Coordinate research activities."),
        ("human", """Project: {task}
Phase: Research and Information Gathering

Coordinate research and provide findings:""")
    ])
    
    chain = research_prompt | llm
    response = chain.invoke({"task": task})
    
    result = {
        "phase": "research",
        "manager": "research_manager",
        "output": response.content
    }
    
    return {
        "phase_results": {"research": response.content},
        "sub_team_results": [result],
        "current_phase": 1
    }

def development_manager(state: HierarchicalState) -> dict:
    """Manages development phase"""
    
    logger.info("Development Manager executing phase 2...")
    
    task = state["messages"][-1].content
    research_context = state["phase_results"].get("research", "No research available")
    
    dev_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a development manager. Create solutions based on research."),
        ("human", """Project: {task}
Phase: Development

Research findings:
{research}

Develop solution:""")
    ])
    
    chain = dev_prompt | llm
    response = chain.invoke({"task": task, "research": research_context})
    
    result = {
        "phase": "development",
        "manager": "development_manager",
        "output": response.content
    }
    
    return {
        "phase_results": {"development": response.content},
        "sub_team_results": [result],
        "current_phase": 2
    }

def quality_manager(state: HierarchicalState) -> dict:
    """Manages quality/review phase"""
    
    logger.info("Quality Manager executing phase 3...")
    
    dev_output = state["phase_results"].get("development", "No development output")
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a quality manager. Review and ensure excellence."),
        ("human", """Review this development output:

{output}

Provide quality assessment and final version:""")
    ])
    
    chain = qa_prompt | llm
    response = chain.invoke({"output": dev_output})
    
    result = {
        "phase": "quality",
        "manager": "quality_manager",
        "output": response.content
    }
    
    return {
        "phase_results": {"quality": response.content},
        "sub_team_results": [result],
        "current_phase": 3
    }

# Integration node
def integrate_results(state: HierarchicalState) -> dict:
    """Integrate all phase results"""
    
    logger.info("Integrating results from all phases...")
    
    # Combine results
    integration = []
    for phase in ["research", "development", "quality"]:
        if phase in state["phase_results"]:
            integration.append(f"=== {phase.upper()} ===\n{state['phase_results'][phase]}")
    
    final_output = "\n\n".join(integration)
    
    return {
        "final_output": final_output,
        "messages": [AIMessage(content=final_output)]
    }

# Router
def route_phase(state: HierarchicalState) -> str:
    """Route to appropriate phase manager"""
    
    phase = state["current_phase"]
    
    if phase == 0:
        return "research"
    elif phase == 1:
        return "development"
    elif phase == 2:
        return "quality"
    else:
        return "integrate"

# Build hierarchical graph
hierarchical_workflow = StateGraph(HierarchicalState)

# Add nodes
hierarchical_workflow.add_node("executive", executive_manager)
hierarchical_workflow.add_node("research", research_manager)
hierarchical_workflow.add_node("development", development_manager)
hierarchical_workflow.add_node("quality", quality_manager)
hierarchical_workflow.add_node("integrate", integrate_results)

# Define flow
hierarchical_workflow.set_entry_point("executive")

hierarchical_workflow.add_conditional_edges(
    "executive",
    route_phase,
    {
        "research": "research",
        "development": "development",
        "quality": "quality",
        "integrate": "integrate"
    }
)

# Create phase transitions
def route_after_phase(state: HierarchicalState) -> str:
    """Route after completing a phase"""
    phase = state["current_phase"]
    
    if phase == 1:
        return "development"
    elif phase == 2:
        return "quality"
    else:
        return "integrate"

hierarchical_workflow.add_conditional_edges(
    "research",
    route_after_phase,
    {
        "development": "development",
        "quality": "quality",
        "integrate": "integrate"
    }
)

hierarchical_workflow.add_conditional_edges(
    "development",
    route_after_phase,
    {
        "development": "development",
        "quality": "quality",
        "integrate": "integrate"
    }
)

hierarchical_workflow.add_conditional_edges(
    "quality",
    route_after_phase,
    {
        "development": "development",
        "quality": "quality",
        "integrate": "integrate"
    }
)

hierarchical_workflow.add_edge("integrate", END)

# Compile
hierarchical_agent = hierarchical_workflow.compile()

# Test
def test_hierarchical_agent(task: str):
    """Test hierarchical agent"""
    
    result = hierarchical_agent.invoke({
        "messages": [HumanMessage(content=task)],
        "high_level_plan": [],
        "current_phase": 0,
        "phase_results": {},
        "sub_team_results": [],
        "final_output": ""
    })
    
    print(f"\n{'='*60}")
    print(f"HIERARCHICAL AGENT")
    print(f"{'='*60}")
    print(f"Task: {task}")
    print(f"Phases completed: {len(result['phase_results'])}")
    print(f"Sub-teams: {len(result['sub_team_results'])}")
    print(f"\nFinal Output:\n{result['final_output'][:400]}...")

if __name__ == "__main__":
    test_hierarchical_agent("Design and implement a customer feedback system")
```

---

## 🎯 Part 4: Specialized Sub-Agent Pattern
### What Is the Specialized Sub-Agent Pattern?

The **Specialized Sub-Agent Pattern** uses multiple **expert agents**, each with deep knowledge in a narrow domain, working collaboratively on a task that requires multiple perspectives.

```
                ┌──────────────┐
                │ COORDINATOR  │
                │  (Decides    │
                │   who needed)│
                └──────┬───────┘
                       │
       ┌───────────────┼───────────────┐
       ↓               ↓               ↓
  ┌─────────┐    ┌─────────┐    ┌─────────┐
  │Security │    │  UX     │    │Performance│
  │Expert   │    │Expert   │    │Expert     │
  │         │    │         │    │           │
  └─────────┘    └─────────┘    └─────────┘
       ↓               ↓               ↓
  [Security       [UX              [Performance
   Analysis]       Review]          Optimization]
       │               │               │
       └───────────────┴───────────────┘
                       ↓
                ┌──────────────┐
                │  SYNTHESIS   │
                │  (Combines   │
                │   expertise) │
                └──────────────┘
```

### Core Concept

Think of **consulting a team of specialists**:
- Designing a building? Need architect, structural engineer, electrical engineer
- Each brings deep expertise in their domain
- Collaboration produces better results than generalist

### Key Characteristics

**1. Domain Expertise:**
- Each agent is highly specialized
- Deep knowledge in narrow area
- Optimized for specific type of analysis

**2. Collaborative Process:**
- Multiple agents contribute
- Each provides their perspective
- Results are synthesized

**3. Selective Invocation:**
- Not all experts needed for every task
- Coordinator determines which to call
- Flexible participation

**4. Synthesis Required:**
- Individual expert opinions must be combined
- May have conflicting recommendations
- Need to resolve and integrate

### Expert Agent Characteristics

**What makes an agent "specialized"?**

**1. Specialized System Prompt:**
```
Generic Agent: "You are a helpful assistant"
Security Expert: "You are a cybersecurity expert. Focus on threats, 
                  vulnerabilities, encryption, and secure practices."
```

**2. Specialized Knowledge:**
- May have domain-specific training
- Access to specialized tools
- Domain-specific evaluation criteria

**3. Specialized Output:**
- Security expert rates security risk
- UX expert rates usability
- Performance expert rates efficiency

### Coordination Strategies

**1. Sequential Consultation:**
```
Coordinator → Security Expert (analyzes)
           → UX Expert (uses security results)
           → Performance Expert (uses both)
→ Each builds on previous
```

**2. Parallel Consultation:**
```
Coordinator → All experts analyze simultaneously
           → Gather all opinions
           → Synthesize
→ Faster, independent perspectives
```

**3. Iterative Consultation:**
```
Round 1: All experts provide initial opinion
Round 2: Experts respond to each other
Round 3: Refine based on discussion
→ More collaborative, converges to consensus
```

### Synthesis Approaches

**Problem:** Multiple expert opinions, potentially conflicting

**Approach 1 - Weighted Average:**
```
If numeric scores:
  Security: 7/10
  UX: 9/10
  Performance: 5/10
  
  Overall = (7 + 9 + 5) / 3 = 7/10
```

**Approach 2 - Veto System:**
```
Any expert can veto (block) if critical issue
Example: Security expert vetoes due to vulnerability
→ Must address before proceeding
```

**Approach 3 - Priority Ranking:**
```
For this project, priorities are:
  1. Security (critical)
  2. UX (important)
  3. Performance (nice-to-have)
  
→ Weight security expert opinion heavily
```

**Approach 4 - Consensus Building:**
```
Use synthesis agent to:
- Identify common ground
- Resolve conflicts
- Create unified recommendation
```

### Specialized vs Generic Agents

| Aspect | Generic Agent | Specialized Agents |
|--------|---------------|-------------------|
| **Knowledge** | Broad, shallow | Narrow, deep |
| **Use Case** | General tasks | Domain-specific |
| **Quality** | Good enough | Expert-level |
| **Collaboration** | Not needed | Essential |
| **Cost** | Low (one agent) | Higher (multiple) |

### When to Use Specialized Sub-Agents

✅ **Use Specialized Sub-Agents When:**
- Task requires multiple domains of expertise
- Each domain is complex enough to need specialist
- Quality and thoroughness are critical
- Different perspectives add value
- Cost of multiple agents is justified

❌ **Don't Use When:**
- Task is straightforward
- Single domain is sufficient
- Speed is more important than thoroughness
- Generic agent is good enough

### Real-World Analogies

**1. Medical Diagnosis:**
- Patient with complex symptoms
- Consult: Cardiologist, Neurologist, Endocrinologist
- Each provides expert opinion
- Primary doctor synthesizes

**2. Home Inspection:**
- Buying a house
- Structural inspector
- Electrical inspector
- Plumbing inspector
- Combine reports for full picture

**3. Code Review:**
- Pull request review
- Security reviewer checks for vulnerabilities
- Performance reviewer checks for inefficiencies
- UX reviewer checks for usability
- All feedback combined


### When to Use: Domain Expertise Required

**Specialized Sub-Agents** are experts in narrow domains that collaborate.

```python
from typing import TypedDict, Annotated, Sequence, List, Dict
from operator import add
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== SPECIALIZED SUB-AGENTS ====================

class SpecializedState(TypedDict):
    """State for specialized agents"""
    messages: Annotated[Sequence[BaseMessage], add]
    required_experts: List[str]
    expert_opinions: Dict[str, str]
    consensus: str

# Initialize
llm = ChatOllama(model="llama3.2", temperature=0.5)

# Coordinator
def expert_coordinator(state: SpecializedState) -> dict:
    """Coordinate which experts are needed"""
    
    try:
        task = state["messages"][-1].content
        
        logger.info("Coordinator analyzing task...")
        
        coord_prompt = ChatPromptTemplate.from_messages([
            ("human", """Analyze this task and determine which experts are needed:

Task: {task}

Available experts:
- SECURITY: Cybersecurity, encryption, vulnerabilities
- PERFORMANCE: Optimization, scalability, efficiency
- UX: User experience, interface design, usability
- DATA: Data structures, algorithms, storage

List 1-3 experts needed (comma-separated):

Experts:""")
        ])
        
        chain = coord_prompt | llm
        response = chain.invoke({"task": task})
        
        # Parse experts
        experts_text = response.content.strip()
        experts = []
        
        for expert in ["SECURITY", "PERFORMANCE", "UX", "DATA"]:
            if expert in experts_text.upper():
                experts.append(expert.lower())
        
        if not experts:
            experts = ["security", "performance"]  # Default
        
        logger.info(f"Calling experts: {experts}")
        
        return {"required_experts": experts}
    
    except Exception as e:
        logger.error(f"Coordination error: {e}")
        return {"required_experts": ["security"]}

# Specialized expert agents
def security_expert(state: SpecializedState) -> dict:
    """Security specialist"""
    
    if "security" not in state["required_experts"]:
        return {}
    
    logger.info("Security Expert analyzing...")
    
    task = state["messages"][-1].content
    
    sec_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a cybersecurity expert. Focus on security implications, vulnerabilities, and protection."),
        ("human", "{task}")
    ])
    
    chain = sec_prompt | llm
    response = chain.invoke({"task": task})
    
    return {
        "expert_opinions": {"security": response.content}
    }

def performance_expert(state: SpecializedState) -> dict:
    """Performance specialist"""
    
    if "performance" not in state["required_experts"]:
        return {}
    
    logger.info("Performance Expert analyzing...")
    
    task = state["messages"][-1].content
    
    perf_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a performance optimization expert. Focus on efficiency, scalability, and speed."),
        ("human", "{task}")
    ])
    
    chain = perf_prompt | llm
    response = chain.invoke({"task": task})
    
    return {
        "expert_opinions": {"performance": response.content}
    }

def ux_expert(state: SpecializedState) -> dict:
    """UX specialist"""
    
    if "ux" not in state["required_experts"]:
        return {}
    
    logger.info("UX Expert analyzing...")
    
    task = state["messages"][-1].content
    
    ux_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a UX expert. Focus on user experience, usability, and interface design."),
        ("human", "{task}")
    ])
    
    chain = ux_prompt | llm
    response = chain.invoke({"task": task})
    
    return {
        "expert_opinions": {"ux": response.content}
    }

def data_expert(state: SpecializedState) -> dict:
    """Data specialist"""
    
    if "data" not in state["required_experts"]:
        return {}
    
    logger.info("Data Expert analyzing...")
    
    task = state["messages"][-1].content
    
    data_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a data expert. Focus on data structures, algorithms, and storage solutions."),
        ("human", "{task}")
    ])
    
    chain = data_prompt | llm
    response = chain.invoke({"task": task})
    
    return {
        "expert_opinions": {"data": response.content}
    }

# Synthesis
def synthesize_opinions(state: SpecializedState) -> dict:
    """Synthesize expert opinions into consensus"""
    
    logger.info("Synthesizing expert opinions...")
    
    # Format opinions
    opinions_text = []
    for expert, opinion in state["expert_opinions"].items():
        opinions_text.append(f"=== {expert.upper()} EXPERT ===\n{opinion}")
    
    combined = "\n\n".join(opinions_text)
    
    synth_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a synthesis expert. Combine different expert perspectives into a coherent recommendation."),
        ("human", """Expert opinions:

{opinions}

Synthesize into a comprehensive recommendation:""")
    ])
    
    chain = synth_prompt | llm
    response = chain.invoke({"opinions": combined})
    
    return {
        "consensus": response.content,
        "messages": [AIMessage(content=response.content)]
    }

# Build specialized agent graph
specialized_workflow = StateGraph(SpecializedState)

# Add nodes
specialized_workflow.add_node("coordinator", expert_coordinator)
specialized_workflow.add_node("security", security_expert)
specialized_workflow.add_node("performance", performance_expert)
specialized_workflow.add_node("ux", ux_expert)
specialized_workflow.add_node("data", data_expert)
specialized_workflow.add_node("synthesize", synthesize_opinions)

# Flow
specialized_workflow.set_entry_point("coordinator")

# All experts run in parallel after coordinator
specialized_workflow.add_edge("coordinator", "security")
specialized_workflow.add_edge("coordinator", "performance")
specialized_workflow.add_edge("coordinator", "ux")
specialized_workflow.add_edge("coordinator", "data")

# All experts feed into synthesis
specialized_workflow.add_edge("security", "synthesize")
specialized_workflow.add_edge("performance", "synthesize")
specialized_workflow.add_edge("ux", "synthesize")
specialized_workflow.add_edge("data", "synthesize")

specialized_workflow.add_edge("synthesize", END)

# Compile
specialized_agent = specialized_workflow.compile()

# Test
def test_specialized_agent(task: str):
    """Test specialized agent"""
    
    result = specialized_agent.invoke({
        "messages": [HumanMessage(content=task)],
        "required_experts": [],
        "expert_opinions": {},
        "consensus": ""
    })
    
    print(f"\n{'='*60}")
    print(f"SPECIALIZED AGENT TEAM")
    print(f"{'='*60}")
    print(f"Task: {task}")
    print(f"Experts consulted: {list(result['expert_opinions'].keys())}")
    print(f"\nConsensus:\n{result['consensus'][:400]}...")

if __name__ == "__main__":
    test_specialized_agent("Design a secure user authentication system")
    test_specialized_agent("Optimize database queries for a high-traffic application")
```

---

## 🔄 Part 5: Error Recovery Pattern
### What Is the Error Recovery Pattern?

The **Error Recovery Pattern** designs agents to gracefully handle failures through retry strategies, fallback mechanisms, and adaptive error handling.

```
┌─────────────┐
│ Primary     │
│ Agent       │
│ (Attempt 1) │
└──────┬──────┘
       ↓
    Success? ──Yes──→ Done
       │
       No
       ↓
┌─────────────┐
│ Error       │
│ Analysis    │
└──────┬──────┘
       │
   ┌───┴────┐
   ↓        ↓
Retry?   Fallback?
   │        │
   ↓        ↓
Retry    Use Alternative
Agent    Agent
```

### Core Concept

**Failures are normal** in production systems:
- APIs go down
- Models hallucinate
- Rate limits hit
- Network issues

**Error recovery** means systems continue working despite failures.

### Types of Failures

**1. Transient Failures (Temporary):**
- Network timeout
- Rate limit hit
- Service temporarily unavailable
- **Solution:** Retry (will likely succeed)

**2. Permanent Failures (Won't fix themselves):**
- Invalid API key
- Service doesn't exist
- Malformed request
- **Solution:** Fallback (retry won't help)

**3. Partial Failures:**
- Some data retrieved, some failed
- Mixed results
- **Solution:** Use what works, retry what failed

### Error Recovery Strategies

### Strategy 1: Retry with Backoff

**Basic Retry:**
```
Attempt 1: Fails
Attempt 2: Fails
Attempt 3: Fails
→ All attempts immediate
→ May overwhelm failing service
```

**Exponential Backoff:**
```
Attempt 1: Fails → Wait 1 second
Attempt 2: Fails → Wait 2 seconds
Attempt 3: Fails → Wait 4 seconds
Attempt 4: Fails → Wait 8 seconds
→ Gives service time to recover
→ Avoids thundering herd
```

**Why exponential?**
- If many clients retry simultaneously, all fail again
- Exponential backoff spreads out retry attempts
- Gives overwhelmed service breathing room

### Strategy 2: Fallback Mechanisms

**Fallback Chain:**
```
Primary: Call external API
  ↓ (fails)
Fallback 1: Use cached result
  ↓ (cache miss)
Fallback 2: Use simpler heuristic
  ↓ (still fails)
Fallback 3: Return graceful error message
```

**Degraded Service:**
Instead of total failure, provide reduced functionality:
```
Full Service: Real-time API + ML enhancement
Degraded: Cached data + rule-based logic
Emergency: Static default response
```

### Strategy 3: Circuit Breaker

**Concept:** Stop calling failing service to prevent cascade failures

```
States:
CLOSED: Normal operation, calls go through
OPEN: Service is failing, don't even try (fail fast)
HALF-OPEN: Testing if service recovered

Transitions:
CLOSED --[Too many failures]→ OPEN
OPEN --[Timeout expires]→ HALF-OPEN
HALF-OPEN --[Success]→ CLOSED
HALF-OPEN --[Failure]→ OPEN
```

**Why circuit breaker?**
- Prevents wasting resources on doomed requests
- Allows failing service to recover
- Faster failure detection
- Cascade failure prevention

### Strategy 4: Timeout Management

**Problem:** Request hangs forever

**Solution:** Set timeouts at multiple levels
```
Request timeout: 5 seconds (single request)
Operation timeout: 30 seconds (entire operation)
Overall timeout: 2 minutes (user-facing)
```

**Timeout strategy:**
- Start with short timeout
- If fails, retry with longer timeout
- Progressive timeout extension

### Strategy 5: Graceful Degradation

**Principle:** Partial functionality is better than total failure

**Example: Search system**
```
Perfect: ML-powered semantic search
Degraded: Keyword-based search
Emergency: Show popular/recent results
```

**Example: Recommendation system**
```
Perfect: Personalized ML recommendations
Degraded: Category-based recommendations
Emergency: Top-rated items
```

### Error Classification

**Classify errors to choose strategy:**

**Retriable Errors:**
- Timeout
- Rate limit (429)
- Server error (500, 502, 503)
- **Action:** Retry with backoff

**Non-Retriable Errors:**
- Bad request (400)
- Unauthorized (401, 403)
- Not found (404)
- **Action:** Fallback or fail

**Conditional Errors:**
- May work with different parameters
- May work at different time
- **Action:** Modify and retry

### Retry Budget

**Problem:** Unlimited retries can make things worse

**Solution: Retry Budget**
```
Max attempts per operation: 3
Max total retries per minute: 100
If budget exceeded: Fail fast, use fallback
```

**Benefits:**
- Prevents retry storms
- Limits resource consumption
- Forces using fallbacks

### Error Recovery Best Practices

**1. Fail Fast When Appropriate:**
```
Don't retry permanent errors
Fail quickly so user gets feedback
Don't waste time on doomed attempts
```

**2. Provide Context:**
```
Not: "Error occurred"
Better: "External API timeout after 3 retries, using cached data"
```

**3. Log Everything:**
```
What failed?
Why did it fail?
What recovery was attempted?
Did recovery succeed?
```

**4. Monitor Patterns:**
```
Track retry rates
Track fallback usage
Detect degradation trends
Alert when thresholds crossed
```

**5. Test Failure Modes:**
```
Unit tests: Mock failures
Integration tests: Inject failures
Chaos engineering: Random failures in production
```

### When to Use Error Recovery

✅ **Always use error recovery in production**

The question is not "if" but "how much":

**Light error handling:**
- Simple retry (2-3 attempts)
- Basic timeout
- Log errors

**Medium error handling:**
- Exponential backoff
- One fallback strategy
- Circuit breaker

**Heavy error handling:**
- Multiple fallback strategies
- Circuit breakers
- Retry budgets
- Graceful degradation
- Comprehensive monitoring

### Real-World Analogies

**1. Calling Someone:**
- Call once: Busy (transient)
- Call again in 5 min: Busy (retry with backoff)
- Call a third time: Busy (circuit breaker opens)
- Leave voicemail instead (fallback)

**2. ATM Machine:**
- Try PIN: Wrong
- Try again: Wrong
- Try third time: Wrong
- Card gets locked (circuit breaker)
- Call bank (fallback)

**3. Restaurant:**
- First choice restaurant: Full (failure)
- Second choice: Also full (retry failed)
- Third choice: Open! (retry succeeded)
- If all full: Order delivery (fallback)

### When to Use: Fault-Tolerant Agent Systems

**Error Recovery Agents** handle failures gracefully and implement retry/fallback strategies.

```python
from typing import TypedDict, Annotated, Sequence, List, Literal
from operator import add
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.tools import tool
import logging
import time
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== ERROR RECOVERY PATTERN ====================

class ErrorRecoveryState(TypedDict):
    """State with error tracking"""
    messages: Annotated[Sequence[BaseMessage], add]
    task: str
    attempts: Annotated[int, add]
    max_attempts: int
    errors: Annotated[List[dict], add]
    fallback_used: bool
    recovery_strategy: Literal["retry", "fallback", "simplify", "abort"]
    result: str
    status: Literal["pending", "success", "failed"]

# Initialize
llm = ChatOllama(model="llama3.2", temperature=0.3)

# Simulated unreliable tool
@tool
def unreliable_api_call(query: str) -> str:
    """
    Simulated API that fails 60% of the time.
    """
    if random.random() < 0.6:
        raise Exception("API temporarily unavailable")
    
    return f"API result for: {query}"

# Node: Primary agent (can fail)
def primary_agent(state: ErrorRecoveryState) -> dict:
    """Primary agent that might fail"""
    
    try:
        logger.info(f"Primary agent attempt {state['attempts'] + 1}/{state['max_attempts']}")
        
        task = state["task"]
        
        # Simulate calling unreliable API
        result = unreliable_api_call.invoke(task)
        
        logger.info("Primary agent succeeded!")
        
        return {
            "result": result,
            "status": "success",
            "attempts": 1,
            "messages": [AIMessage(content=f"Success: {result}")]
        }
    
    except Exception as e:
        logger.error(f"Primary agent failed: {e}")
        
        error_record = {
            "attempt": state["attempts"] + 1,
            "error": str(e),
            "timestamp": time.time()
        }
        
        return {
            "attempts": 1,
            "errors": [error_record],
            "status": "pending"
        }

# Node: Error analyzer
def analyze_error(state: ErrorRecoveryState) -> dict:
    """Analyze error and determine recovery strategy"""
    
    try:
        if not state["errors"]:
            return {"recovery_strategy": "retry"}
        
        latest_error = state["errors"][-1]
        attempts = state["attempts"]
        max_attempts = state["max_attempts"]
        
        logger.info(f"Analyzing error after attempt {attempts}/{max_attempts}")
        
        # Strategy selection
        if attempts < max_attempts // 2:
            strategy = "retry"  # Early attempts - just retry
        elif attempts < max_attempts:
            strategy = "simplify"  # Middle attempts - simplify task
        else:
            strategy = "fallback"  # Final attempts - use fallback
        
        logger.info(f"Selected strategy: {strategy}")
        
        return {"recovery_strategy": strategy}
    
    except Exception as e:
        logger.error(f"Error analysis failed: {e}")
        return {"recovery_strategy": "abort"}

# Node: Retry with backoff
def retry_with_backoff(state: ErrorRecoveryState) -> dict:
    """Retry with exponential backoff"""
    
    try:
        attempt = state["attempts"]
        
        # Exponential backoff
        wait_time = min(2 ** attempt, 8)  # Max 8 seconds
        logger.info(f"Waiting {wait_time}s before retry...")
        time.sleep(wait_time)
        
        # Retry primary agent
        task = state["task"]
        result = unreliable_api_call.invoke(task)
        
        logger.info("Retry succeeded!")
        
        return {
            "result": result,
            "status": "success",
            "attempts": 1,
            "messages": [AIMessage(content=f"Success on retry: {result}")]
        }
    
    except Exception as e:
        logger.error(f"Retry failed: {e}")
        
        error_record = {
            "attempt": state["attempts"] + 1,
            "error": str(e),
            "timestamp": time.time(),
            "strategy": "retry"
        }
        
        return {
            "attempts": 1,
            "errors": [error_record],
            "status": "pending"
        }

# Node: Simplify task
def simplify_task(state: ErrorRecoveryState) -> dict:
    """Simplify the task and retry"""
    
    try:
        task = state["task"]
        
        logger.info("Simplifying task...")
        
        simplify_prompt = ChatPromptTemplate.from_messages([
            ("human", """This task is failing. Simplify it to a more basic version:

Original task: {task}

Provide a simpler version that's more likely to succeed:""")
        ])
        
        chain = simplify_prompt | llm
        response = chain.invoke({"task": task})
        
        simplified_task = response.content.strip()
        
        logger.info(f"Simplified task: {simplified_task[:50]}...")
        
        # Try with simplified task
        result = unreliable_api_call.invoke(simplified_task)
        
        logger.info("Simplified task succeeded!")
        
        return {
            "result": f"Simplified approach: {result}",
            "status": "success",
            "attempts": 1,
            "messages": [AIMessage(content=f"Success with simplified approach: {result}")]
        }
    
    except Exception as e:
        logger.error(f"Simplified task failed: {e}")
        
        error_record = {
            "attempt": state["attempts"] + 1,
            "error": str(e),
            "timestamp": time.time(),
            "strategy": "simplify"
        }
        
        return {
            "attempts": 1,
            "errors": [error_record],
            "status": "pending"
        }

# Node: Fallback agent
def fallback_agent(state: ErrorRecoveryState) -> dict:
    """Fallback agent using alternative approach"""
    
    try:
        task = state["task"]
        
        logger.info("Using fallback agent...")
        
        # Alternative approach - use LLM directly without unreliable API
        fallback_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a fallback agent. Provide a reasonable response without external APIs."),
            ("human", "{task}")
        ])
        
        chain = fallback_prompt | llm
        response = chain.invoke({"task": task})
        
        result = f"Fallback result: {response.content}"
        
        logger.info("Fallback succeeded!")
        
        return {
            "result": result,
            "status": "success",
            "fallback_used": True,
            "attempts": 1,
            "messages": [AIMessage(content=result)]
        }
    
    except Exception as e:
        logger.error(f"Fallback failed: {e}")
        
        return {
            "status": "failed",
            "result": f"All recovery attempts failed. Last error: {str(e)}",
            "messages": [AIMessage(content=f"Failed after all attempts: {str(e)}")]
        }

# Node: Abort
def abort_gracefully(state: ErrorRecoveryState) -> dict:
    """Abort with informative message"""
    
    logger.info("Aborting after exhausting all strategies")
    
    error_summary = "\n".join([
        f"Attempt {err['attempt']}: {err['error']}"
        for err in state["errors"][-3:]  # Last 3 errors
    ])
    
    message = f"Task failed after {state['attempts']} attempts.\n\nRecent errors:\n{error_summary}"
    
    return {
        "status": "failed",
        "result": message,
        "messages": [AIMessage(content=message)]
    }

# Router: Decide recovery action
def route_recovery(state: ErrorRecoveryState) -> str:
    """Route based on recovery strategy"""
    
    if state["status"] == "success":
        return "end"
    
    if state["attempts"] >= state["max_attempts"]:
        logger.info("Max attempts reached - using fallback")
        return "fallback"
    
    strategy = state.get("recovery_strategy", "retry")
    
    return strategy

def route_after_recovery(state: ErrorRecoveryState) -> str:
    """Route after recovery attempt"""
    
    if state["status"] == "success":
        return "end"
    
    if state["attempts"] >= state["max_attempts"]:
        return "fallback"
    
    return "analyze"

# Build error recovery graph
error_recovery_workflow = StateGraph(ErrorRecoveryState)

# Add nodes
error_recovery_workflow.add_node("primary", primary_agent)
error_recovery_workflow.add_node("analyze", analyze_error)
error_recovery_workflow.add_node("retry", retry_with_backoff)
error_recovery_workflow.add_node("simplify", simplify_task)
error_recovery_workflow.add_node("fallback", fallback_agent)
error_recovery_workflow.add_node("abort", abort_gracefully)

# Define flow
error_recovery_workflow.set_entry_point("primary")

error_recovery_workflow.add_conditional_edges(
    "primary",
    route_recovery,
    {
        "end": END,
        "analyze": "analyze",
        "fallback": "fallback"
    }
)

error_recovery_workflow.add_conditional_edges(
    "analyze",
    route_recovery,
    {
        "retry": "retry",
        "simplify": "simplify",
        "fallback": "fallback",
        "abort": "abort"
    }
)

error_recovery_workflow.add_conditional_edges(
    "retry",
    route_after_recovery,
    {
        "end": END,
        "analyze": "analyze",
        "fallback": "fallback"
    }
)

error_recovery_workflow.add_conditional_edges(
    "simplify",
    route_after_recovery,
    {
        "end": END,
        "analyze": "analyze",
        "fallback": "fallback"
    }
)

error_recovery_workflow.add_edge("fallback", END)
error_recovery_workflow.add_edge("abort", END)

# Compile
error_recovery_agent = error_recovery_workflow.compile()

# Test
def test_error_recovery(task: str, max_attempts: int = 5):
    """Test error recovery agent"""
    
    result = error_recovery_agent.invoke({
        "messages": [HumanMessage(content=task)],
        "task": task,
        "attempts": 0,
        "max_attempts": max_attempts,
        "errors": [],
        "fallback_used": False,
        "recovery_strategy": "retry",
        "result": "",
        "status": "pending"
    })
    
    print(f"\n{'='*60}")
    print(f"ERROR RECOVERY AGENT")
    print(f"{'='*60}")
    print(f"Task: {task}")
    print(f"Status: {result['status']}")
    print(f"Attempts: {result['attempts']}")
    print(f"Fallback used: {result['fallback_used']}")
    print(f"Errors encountered: {len(result['errors'])}")
    print(f"\nResult: {result['result']}")

if __name__ == "__main__":
    test_error_recovery("Get data from external API")
```

---

## 🎼 Part 6: Production Orchestration Pattern
### What Is Production Orchestration?

**Production Orchestration** is a comprehensive pattern that combines all previous patterns with production-grade features: monitoring, quality gates, error recovery, and metrics tracking.

```
┌────────────────────────────────────────────┐
│         PRODUCTION ORCHESTRATOR            │
│                                            │
│  ┌──────────┐  ┌──────────┐  ┌─────────┐ │
│  │ Task     │→ │ Planning │→ │ Execution│ │
│  │ Analysis │  │          │  │          │ │
│  └──────────┘  └──────────┘  └─────────┘ │
│                                            │
│  ┌──────────┐  ┌──────────┐  ┌─────────┐ │
│  │ Quality  │  │ Error    │  │ Metrics │ │
│  │ Gates    │  │ Recovery │  │ Tracking│ │
│  └──────────┘  └──────────┘  └─────────┘ │
└────────────────────────────────────────────┘
```

### Core Concept

Production orchestration is like **running an actual business operation**:
- Not just getting the job done
- But getting it done **reliably, efficiently, measurably**
- With monitoring, quality control, and accountability

### Key Components

### 1. Task Analysis & Classification

**Why:** Different tasks need different approaches

**What it does:**
- Analyzes incoming task
- Classifies complexity (simple/complex/urgent)
- Determines priority (low/medium/high)
- Estimates resources needed
- Decides execution strategy

**Example:**
```
Task: "Generate monthly report"
Analysis:
  - Complexity: Medium (multiple data sources)
  - Priority: High (deadline today)
  - Strategy: Parallel execution with quality checks
  - Estimated time: 5 minutes
```

### 2. Execution Planning

**Why:** Complex tasks need structured approach

**What it does:**
- Breaks task into steps
- Determines agent assignments
- Identifies dependencies
- Creates execution timeline
- Allocates resources

**Example Plan:**
```
Step 1: Data Collection (Agent A, B in parallel) - 2 min
Step 2: Data Analysis (Agent C, needs Step 1) - 2 min
Step 3: Report Writing (Agent D, needs Step 2) - 1 min
Step 4: Quality Review (Agent E, needs Step 3) - 1 min
```

### 3. Quality Gates

**Why:** Ensure output quality before proceeding

**What it does:**
- Checks output at key points
- Validates against criteria
- Blocks progression if quality low
- May trigger rework

**Quality Gate Example:**
```
After Data Analysis:
  ✓ Data completeness > 95%
  ✓ No outliers beyond 3 std dev
  ✓ Format matches schema
  
  If any check fails:
    → Retry data collection
    or → Flag for human review
```

**Types of Gates:**
- **Input gate:** Validate task is feasible
- **Intermediate gates:** Check progress quality
- **Output gate:** Validate final result
- **Compliance gates:** Ensure regulations met

### 4. Error Recovery Integration

**Why:** Production systems must handle failures

**What it does:**
- Detects errors at any step
- Classifies error type
- Applies appropriate recovery
- Tracks recovery attempts
- Escalates if needed

**Recovery Levels:**
```
Level 1: Retry (same agent, same approach)
Level 2: Alternative approach (same agent, different method)
Level 3: Fallback agent (different agent)
Level 4: Human escalation
```

### 5. Metrics & Monitoring

**Why:** Can't improve what you don't measure

**Tracks:**
- Execution time per step
- Agent utilization
- Success/failure rates
- Quality scores
- Cost (LLM tokens)
- Retry counts
- Bottlenecks

**Metrics Example:**
```
Task ID: task-12345
Duration: 4.2 minutes
Agents used: [A, C, D]
Steps completed: 4/4
Quality score: 8.5/10
Retries: 1 (Step 2)
Cost: $0.15
```

### 6. Adaptive Execution

**Why:** One size doesn't fit all

**What it does:**
- Monitors execution progress
- Adjusts strategy if needed
- Reallocates resources
- Changes priorities dynamically

**Example:**
```
Initial plan: Sequential execution (safe)
Progress update: Step 1 faster than expected
Adaptation: Switch to parallel for Step 2 & 3 (faster)
```

### Production vs Simple System

| Aspect | Simple System | Production System |
|--------|--------------|-------------------|
| **Planning** | None or basic | Comprehensive analysis |
| **Execution** | Fire and forget | Monitored continuously |
| **Quality** | Hope for best | Multiple quality gates |
| **Errors** | May crash | Graceful recovery |
| **Metrics** | None | Everything tracked |
| **Adaptation** | Fixed flow | Dynamic adjustment |
| **Visibility** | Black box | Full observability |

### Production Orchestration Phases

**Phase 1: Pre-Execution**
```
1. Task arrives
2. Validate feasibility
3. Analyze and classify
4. Create execution plan
5. Allocate resources
6. Initialize metrics
```

**Phase 2: Execution**
```
For each step:
  1. Assign to agent
  2. Execute with timeout
  3. Monitor progress
  4. Check quality gate
  5. Record metrics
  6. Handle errors if any
```

**Phase 3: Post-Execution**
```
1. Aggregate results
2. Final quality check
3. Generate report
4. Update metrics
5. Archive for audit
```

### Decision Trees in Production

**Task Priority Decision:**
```
Is deadline < 1 hour? 
  Yes → Priority: URGENT
        → Strategy: Fast path, fewer quality checks
  No → Is task complexity HIGH?
        Yes → Priority: HIGH
              → Strategy: Thorough, all quality gates
        No → Priority: NORMAL
             → Strategy: Standard flow
```

**Agent Selection Decision:**
```
Is specialist available?
  Yes → Use specialist (better quality)
  No → Is task urgent?
        Yes → Use generalist (available)
        No → Wait for specialist (better quality)
```

### Production Concerns

**1. Observability:**
```
Can I see what's happening?
Can I trace execution?
Can I debug issues?
→ Comprehensive logging and metrics
```

**2. Reliability:**
```
Does it work consistently?
Can it handle failures?
Does it recover automatically?
→ Error recovery and fallbacks
```

**3. Performance:**
```
Is it fast enough?
Where are bottlenecks?
Can it scale?
→ Performance monitoring and optimization
```

**4. Cost:**
```
How much does it cost?
Where is money spent?
Can we optimize?
→ Token tracking and budgets
```

**5. Quality:**
```
Is output good enough?
How do we measure quality?
Can we maintain standards?
→ Quality gates and scoring
```

### When to Use Production Orchestration

✅ **Use Production Orchestration When:**
- Running in actual production (real users)
- Reliability is critical
- Need to track performance
- Must handle failures gracefully
- Quality standards must be met
- Costs need to be controlled
- Compliance required

❌ **Overkill When:**
- Prototyping/experimenting
- Simple scripts
- One-off tasks
- Personal projects
- Learning/teaching

### Real-World Analogies

**1. Factory Production Line:**
- Not just "make product"
- Quality control stations
- Performance monitoring
- Defect handling
- Metrics on efficiency
- Continuous improvement

**2. Hospital Surgery:**
- Not just "do operation"
- Pre-op checklist
- Vital signs monitoring
- Quality protocols
- Error prevention
- Post-op verification
- Comprehensive documentation

**3. Air Traffic Control:**
- Not just "land planes"
- Continuous monitoring
- Safety checks at every stage
- Error recovery procedures
- Metrics and logging
- Coordinated execution

### Complete Enterprise Agent System

```python
from typing import TypedDict, Annotated, Sequence, List, Dict, Literal, Optional
from operator import add
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from dataclasses import dataclass, field
from datetime import datetime
import logging
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== PRODUCTION ORCHESTRATION ====================

@dataclass
class TaskMetrics:
    """Track task execution metrics"""
    task_id: str
    task_type: str
    start_time: float
    end_time: Optional[float] = None
    agents_used: List[str] = field(default_factory=list)
    retry_count: int = 0
    success: bool = False
    error: Optional[str] = None
    
    @property
    def duration(self) -> float:
        """Calculate task duration"""
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time

class ProductionState(TypedDict):
    """Complete production orchestration state"""
    # Core
    messages: Annotated[Sequence[BaseMessage], add]
    task_id: str
    user_id: str
    
    # Task understanding
    task_type: Literal["simple", "complex", "urgent", "research"]
    priority: Literal["low", "medium", "high"]
    complexity_score: float
    
    # Orchestration
    execution_plan: List[dict]
    current_step: int
    agent_assignments: Dict[str, str]
    
    # Results
    intermediate_results: Dict[str, str]
    final_result: str
    
    # Quality & Control
    quality_score: float
    needs_human_review: bool
    
    # Error handling
    errors: Annotated[List[dict], add]
    retry_count: Annotated[int, add]
    max_retries: int
    
    # Metrics
    metrics: Optional[TaskMetrics]
    status: Literal["pending", "in_progress", "completed", "failed", "needs_review"]

# Initialize
llm = ChatOllama(model="llama3.2", temperature=0.3)

# ==================== TASK ANALYSIS ====================

task_analyzer_prompt = ChatPromptTemplate.from_messages([
    ("human", """Analyze this task comprehensively:

Task: {task}

Provide analysis in JSON format:
{{
  "type": "simple|complex|urgent|research",
  "priority": "low|medium|high",
  "complexity": 0.0-1.0,
  "estimated_steps": number,
  "requires_human_review": true|false,
  "key_requirements": ["req1", "req2"]
}}

Analysis:""")
])

task_analyzer_chain = task_analyzer_prompt | llm

def analyze_task_node(state: ProductionState) -> dict:
    """Comprehensive task analysis"""
    
    try:
        task = state["messages"][-1].content
        task_id = f"task-{int(time.time())}"
        
        logger.info(f"Analyzing task {task_id}...")
        
        response = task_analyzer_chain.invoke({"task": task})
        response_text = response.content.strip()
        
        # Parse JSON
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        import json
        try:
            analysis = json.loads(response_text)
        except:
            analysis = {
                "type": "simple",
                "priority": "medium",
                "complexity": 0.5,
                "estimated_steps": 3,
                "requires_human_review": False
            }
        
        # Create metrics
        metrics = TaskMetrics(
            task_id=task_id,
            task_type=analysis.get("type", "simple"),
            start_time=time.time()
        )
        
        logger.info(f"Task analysis: type={analysis['type']}, priority={analysis['priority']}, complexity={analysis['complexity']}")
        
        return {
            "task_id": task_id,
            "task_type": analysis.get("type", "simple"),
            "priority": analysis.get("priority", "medium"),
            "complexity_score": analysis.get("complexity", 0.5),
            "needs_human_review": analysis.get("requires_human_review", False),
            "metrics": metrics,
            "status": "in_progress"
        }
    
    except Exception as e:
        logger.error(f"Task analysis error: {e}")
        return {
            "task_id": "error",
            "task_type": "simple",
            "priority": "medium",
            "complexity_score": 0.5,
            "needs_human_review": False,
            "status": "in_progress"
        }

# ==================== EXECUTION PLANNING ====================

planner_prompt = ChatPromptTemplate.from_messages([
    ("human", """Create an execution plan for this task:

Task: {task}
Type: {task_type}
Complexity: {complexity}

Break down into steps with agent assignments.

Respond in JSON:
{{
  "steps": [
    {{"step": 1, "agent": "research|code|data|text", "description": "..."}},
    {{"step": 2, "agent": "...", "description": "..."}}
  ]
}}

Plan:""")
])

planner_chain = planner_prompt | llm

def create_execution_plan(state: ProductionState) -> dict:
    """Create detailed execution plan"""
    
    try:
        task = state["messages"][-1].content
        
        logger.info("Creating execution plan...")
        
        response = planner_chain.invoke({
            "task": task,
            "task_type": state["task_type"],
            "complexity": state["complexity_score"]
        })
        
        response_text = response.content.strip()
        
        # Parse JSON
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        import json
        try:
            plan_data = json.loads(response_text)
            steps = plan_data.get("steps", [])
        except:
            steps = [
                {"step": 1, "agent": "text", "description": "Process task"}
            ]
        
        logger.info(f"Created plan with {len(steps)} steps")
        
        return {
            "execution_plan": steps,
            "current_step": 0
        }
    
    except Exception as e:
        logger.error(f"Planning error: {e}")
        return {
            "execution_plan": [{"step": 1, "agent": "text", "description": "Process task"}],
            "current_step": 0
        }

# ==================== AGENT EXECUTION ====================

def execute_agent_step(state: ProductionState) -> dict:
    """Execute current step with assigned agent"""
    
    try:
        if not state["execution_plan"]:
            return {"status": "completed"}
        
        step_idx = state["current_step"]
        
        if step_idx >= len(state["execution_plan"]):
            return {"status": "completed"}
        
        current_step = state["execution_plan"][step_idx]
        agent_type = current_step.get("agent", "text")
        description = current_step.get("description", "")
        
        logger.info(f"Executing step {step_idx + 1}/{len(state['execution_plan'])}: {agent_type}")
        
        # Get context from previous steps
        context = "\n".join([
            f"Step {k}: {v}"
            for k, v in state["intermediate_results"].items()
        ])
        
        # Execute with appropriate agent
        agent_prompt = ChatPromptTemplate.from_messages([
            ("system", f"You are a {agent_type} specialist agent."),
            ("human", """Task: {task}
Step description: {description}

Previous context:
{context}

Execute this step:""")
        ])
        
        chain = agent_prompt | llm
        response = chain.invoke({
            "task": state["messages"][-1].content,
            "description": description,
            "context": context if context else "No previous context"
        })
        
        result = response.content
        
        # Track metrics
        if state["metrics"]:
            state["metrics"].agents_used.append(agent_type)
        
        logger.info(f"Step {step_idx + 1} completed by {agent_type}")
        
        return {
            "intermediate_results": {f"step_{step_idx + 1}": result},
            "current_step": step_idx + 1
        }
    
    except Exception as e:
        logger.error(f"Agent execution error: {e}")
        
        error_record = {
            "step": state["current_step"],
            "error": str(e),
            "timestamp": time.time()
        }
        
        return {
            "errors": [error_record],
            "retry_count": 1
        }

# ==================== QUALITY ASSURANCE ====================

qa_prompt = ChatPromptTemplate.from_messages([
    ("human", """Evaluate the quality of these results:

Original task: {task}

Results:
{results}

Rate quality from 0-10 considering:
- Completeness
- Accuracy
- Clarity

Respond with ONLY a number 0-10.

Quality score:""")
])

qa_chain = qa_prompt | llm

def quality_assurance(state: ProductionState) -> dict:
    """Check result quality"""
    
    try:
        task = state["messages"][-1].content
        
        # Combine all results
        results = "\n\n".join([
            f"{k}: {v}"
            for k, v in state["intermediate_results"].items()
        ])
        
        logger.info("Performing quality assurance...")
        
        response = qa_chain.invoke({
            "task": task,
            "results": results
        })
        
        # Parse score
        score = 7.0
        try:
            import re
            numbers = re.findall(r'\d+', response.content)
            if numbers:
                score = min(float(numbers[0]), 10.0)
        except:
            pass
        
        logger.info(f"Quality score: {score}/10")
        
        # Update metrics
        if state["metrics"]:
            state["metrics"].success = score >= 7.0
        
        return {
            "quality_score": score
        }
    
    except Exception as e:
        logger.error(f"QA error: {e}")
        return {
            "quality_score": 5.0
        }

# ==================== FINALIZATION ====================

def finalize_results(state: ProductionState) -> dict:
    """Finalize and format results"""
    
    try:
        logger.info("Finalizing results...")
        
        # Combine results
        final_output = []
        
        for key in sorted(state["intermediate_results"].keys()):
            final_output.append(state["intermediate_results"][key])
        
        final_result = "\n\n".join(final_output)
        
        # Update metrics
        if state["metrics"]:
            state["metrics"].end_time = time.time()
            
            logger.info(f"Task completed in {state['metrics'].duration:.2f}s using {len(state['metrics'].agents_used)} agents")
        
        # Determine final status
        if state["quality_score"] < 6.0 or state["needs_human_review"]:
            status = "needs_review"
        elif state["errors"]:
            status = "completed"  # Completed with issues
        else:
            status = "completed"
        
        return {
            "final_result": final_result,
            "status": status,
            "messages": [AIMessage(content=final_result)]
        }
    
    except Exception as e:
        logger.error(f"Finalization error: {e}")
        return {
            "final_result": "Error finalizing results",
            "status": "failed",
            "messages": [AIMessage(content=f"Error: {str(e)}")]
        }

# ==================== ROUTING ====================

def should_continue_execution(state: ProductionState) -> str:
    """Decide if execution should continue"""
    
    # Check for errors
    if state["retry_count"] >= state["max_retries"]:
        logger.warning("Max retries exceeded")
        return "qa"
    
    # Check if plan is complete
    if state["current_step"] >= len(state["execution_plan"]):
        return "qa"
    
    return "execute"

def should_retry_or_finish(state: ProductionState) -> str:
    """Decide if retry needed or finish"""
    
    if state["errors"] and state["retry_count"] < state["max_retries"]:
        logger.info("Retrying failed step...")
        return "execute"
    
    return "finalize"

# ==================== BUILD PRODUCTION GRAPH ====================

production_workflow = StateGraph(ProductionState)

# Add nodes
production_workflow.add_node("analyze", analyze_task_node)
production_workflow.add_node("plan", create_execution_plan)
production_workflow.add_node("execute", execute_agent_step)
production_workflow.add_node("qa", quality_assurance)
production_workflow.add_node("finalize", finalize_results)

# Define flow
production_workflow.set_entry_point("analyze")
production_workflow.add_edge("analyze", "plan")
production_workflow.add_edge("plan", "execute")

production_workflow.add_conditional_edges(
    "execute",
    should_continue_execution,
    {
        "execute": "execute",  # Loop
        "qa": "qa"
    }
)

production_workflow.add_conditional_edges(
    "qa",
    should_retry_or_finish,
    {
        "execute": "execute",
        "finalize": "finalize"
    }
)

production_workflow.add_edge("finalize", END)

# Compile with checkpointing
checkpointer = SqliteSaver.from_conn_string("./production_orchestration.db")
production_agent = production_workflow.compile(checkpointer=checkpointer)

# ==================== PRODUCTION API ====================

def execute_production_task(
    task: str,
    user_id: str = "default",
    max_retries: int = 2
) -> dict:
    """
    Execute task with production orchestration.
    
    Args:
        task: Task description
        user_id: User identifier
        max_retries: Maximum retry attempts
    
    Returns:
        dict with result and comprehensive metadata
    """
    
    config = {
        "configurable": {
            "thread_id": f"prod-{user_id}-{int(time.time())}"
        }
    }
    
    initial_state = {
        "messages": [HumanMessage(content=task)],
        "task_id": "",
        "user_id": user_id,
        "task_type": "simple",
        "priority": "medium",
        "complexity_score": 0.5,
        "execution_plan": [],
        "current_step": 0,
        "agent_assignments": {},
        "intermediate_results": {},
        "final_result": "",
        "quality_score": 0.0,
        "needs_human_review": False,
        "errors": [],
        "retry_count": 0,
        "max_retries": max_retries,
        "metrics": None,
        "status": "pending"
    }
    
    try:
        result = production_agent.invoke(initial_state, config=config)
        
        metrics = result.get("metrics")
        
        return {
            "success": True,
            "task_id": result["task_id"],
            "result": result["final_result"],
            "status": result["status"],
            "metadata": {
                "task_type": result["task_type"],
                "priority": result["priority"],
                "complexity": result["complexity_score"],
                "quality_score": result["quality_score"],
                "steps_executed": len(result["execution_plan"]),
                "agents_used": metrics.agents_used if metrics else [],
                "duration": metrics.duration if metrics else 0.0,
                "retries": result["retry_count"],
                "errors": len(result["errors"]),
                "needs_review": result["needs_human_review"]
            }
        }
    
    except Exception as e:
        logger.error(f"Production execution failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }

# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("PRODUCTION ORCHESTRATION DEMO")
    print("="*60)
    
    test_tasks = [
        "Analyze customer feedback and create improvement recommendations",
        "Research competitors and write a market analysis report",
        "Design a data pipeline for processing user events"
    ]
    
    for i, task in enumerate(test_tasks, 1):
        print(f"\n{'='*60}")
        print(f"TASK {i}: {task}")
        print(f"{'='*60}")
        
        result = execute_production_task(task, user_id=f"demo-{i}")
        
        if result["success"]:
            meta = result["metadata"]
            
            print(f"\n✅ COMPLETED")
            print(f"Task ID: {result['task_id']}")
            print(f"Status: {result['status']}")
            print(f"\n📊 METADATA:")
            print(f"  Type: {meta['task_type']}")
            print(f"  Priority: {meta['priority']}")
            print(f"  Complexity: {meta['complexity']:.2f}")
            print(f"  Quality: {meta['quality_score']:.1f}/10")
            print(f"  Steps: {meta['steps_executed']}")
            print(f"  Agents: {', '.join(meta['agents_used'])}")
            print(f"  Duration: {meta['duration']:.2f}s")
            print(f"  Retries: {meta['retries']}")
            print(f"  Needs Review: {meta['needs_review']}")
            
            print(f"\n📝 RESULT:")
            print(result['result'][:300] + "...")
        else:
            print(f"\n❌ FAILED: {result['error']}")
```

---
## 🎯 Pattern Selection Framework

### Quick Decision Tree

```
Start: What's your task?

Is it clearly categorizable? (yes/no)
  Yes → Use ROUTER PATTERN
  
Does it need multiple sequential steps?
  Yes → Does each step require different expertise?
    Yes → Use SUPERVISOR PATTERN
    No → Use simple workflow

Is it very complex/large?
  Yes → Use HIERARCHICAL PATTERN
  
Do you need multiple expert perspectives?
  Yes → Use SPECIALIZED SUB-AGENTS
  
Is reliability critical?
  Yes → Add ERROR RECOVERY
  
Is this going to production?
  Yes → Use PRODUCTION ORCHESTRATION
```

### Pattern Combination

**Patterns are NOT mutually exclusive!**

**Common Combinations:**

**1. Router + Specialized Sub-Agents**
```
Router classifies task type
→ Routes to team of specialists
→ Specialists collaborate
```

**2. Supervisor + Error Recovery**
```
Supervisor coordinates workers
→ Each worker has error recovery
→ Supervisor handles worker failures
```

**3. Hierarchical + Specialized**
```
Top level: Strategic planning
Middle level: Specialized managers (Tech, Business, Creative)
Bottom level: Specialized workers
```

**4. Production Orchestration (combines all)**
```
Router for classification
+ Supervisor for coordination
+ Error Recovery for reliability
+ Metrics for monitoring
= Production-ready system
```
---

## 🧠 Mental Models

### Mental Model 1: Restaurant Kitchen

**Router = Host Station**
- Classifies party size and preferences
- Directs to appropriate section

**Supervisor = Head Chef**
- Coordinates all line cooks
- Ensures dishes come out together

**Hierarchical = Restaurant Chain**
- Corporate (executive)
- Regional managers (middle)
- Individual kitchens (workers)

**Specialized = Stations**
- Grill station
- Pasta station
- Dessert station

**Error Recovery = Backup Plans**
- Out of ingredient? Substitute
- Order delayed? Free appetizer
- Dish rejected? Remake

### Mental Model 2: Software Development

**Router = Issue Triage**
- Bug? → QA team
- Feature? → Dev team
- Docs? → Technical writing

**Supervisor = Tech Lead**
- Breaks down features
- Assigns to developers
- Reviews and integrates

**Hierarchical = CTO → VPs → Directors → Managers → ICs**

**Specialized = Domain Experts**
- Frontend specialist
- Backend specialist
- Database specialist

**Error Recovery = CI/CD**
- Test fails? Retry
- Deploy fails? Rollback
- All fails? Alert on-call

---

## 📋 Part 7: Best Practices Summary

### 1. Pattern Selection Guide

```python
# ✅ GOOD: Choose the right pattern for the problem

def select_agent_pattern(task_characteristics: dict) -> str:
    """
    Select appropriate agent pattern based on task.
    
    Task characteristics:
    - complexity: simple, medium, high
    - domain_diversity: single, multiple
    - parallelizable: bool
    - needs_specialization: bool
    - error_prone: bool
    """
    
    if task_characteristics["error_prone"]:
        return "error_recovery"
    
    elif task_characteristics["needs_specialization"]:
        return "specialized_sub_agents"
    
    elif task_characteristics["complexity"] == "high":
        if task_characteristics["domain_diversity"] == "multiple":
            return "hierarchical"
        else:
            return "supervisor"
    
    elif task_characteristics["domain_diversity"] == "multiple":
        return "router"
    
    else:
        return "single_agent"

# ❌ BAD: Using complex pattern for simple task
# Don't use hierarchical pattern for "What's 2+2?"
```

### 2. Agent Communication

```python
# ✅ GOOD: Clear agent interfaces
class AgentInterface:
    """Standard interface for all agents"""
    
    def execute(self, task: str, context: dict) -> dict:
        """
        Execute task with context.
        
        Returns:
            {
                "success": bool,
                "result": str,
                "metadata": dict
            }
        """
        pass
    
    def validate_input(self, task: str) -> bool:
        """Validate if agent can handle task"""
        pass

# ❌ BAD: Inconsistent agent interfaces
# Agent A returns string, Agent B returns dict, Agent C raises exception
```

### 3. State Management

```python
# ✅ GOOD: Comprehensive state tracking
class MultiAgentState(TypedDict):
    # Core
    messages: Annotated[Sequence[BaseMessage], add]
    
    # Agent tracking
    active_agent: str
    agent_history: Annotated[List[str], add]
    
    # Results
    agent_results: Dict[str, Any]
    
    # Control
    status: Literal["pending", "in_progress", "completed", "failed"]
    
    # Metrics
    start_time: float
    agent_timings: Dict[str, float]

# ❌ BAD: Minimal state
class BadState(TypedDict):
    messages: list  # Missing everything else
```

### 4. Error Handling

```python
# ✅ GOOD: Comprehensive error handling
def execute_with_recovery(agent_fn, state, max_retries=3):
    """Execute agent with error recovery"""
    
    for attempt in range(max_retries):
        try:
            result = agent_fn(state)
            return result
        
        except TemporaryError as e:
            logger.warning(f"Temporary error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            else:
                # Try fallback
                return fallback_agent(state)
        
        except PermanentError as e:
            logger.error(f"Permanent error: {e}")
            return error_response(e)

# ❌ BAD: No error handling
def bad_execute(agent_fn, state):
    return agent_fn(state)  # What if it fails?
```

### 5. Monitoring and Observability

```python
# ✅ GOOD: Track agent performance
@dataclass
class AgentMetrics:
    """Track individual agent metrics"""
    agent_name: str
    executions: int = 0
    successes: int = 0
    failures: int = 0
    avg_duration: float = 0.0
    avg_quality: float = 0.0
    
    @property
    def success_rate(self) -> float:
        return self.successes / self.executions if self.executions > 0 else 0.0
    
    def log_execution(self, duration: float, success: bool, quality: float):
        """Log agent execution"""
        self.executions += 1
        
        if success:
            self.successes += 1
        else:
            self.failures += 1
        
        # Update averages
        self.avg_duration = (
            (self.avg_duration * (self.executions - 1) + duration) /
            self.executions
        )
        
        self.avg_quality = (
            (self.avg_quality * (self.executions - 1) + quality) /
            self.executions
        )

# ❌ BAD: No metrics
# Agent executes but we don't know if it's working well
```

---

## 🧠 Key Concepts to Remember

1. **Router Pattern** = classify and route to specialists
2. **Supervisor Pattern** = manage multiple workers
3. **Hierarchical Pattern** = multi-level management
4. **Specialized Sub-Agents** = domain experts collaborate
5. **Error Recovery** = graceful failure handling
6. **Production Orchestration** = complete enterprise system
7. **Choose pattern based on task** characteristics
8. **Clear agent interfaces** for communication
9. **Comprehensive state management** across agents
10. **Error recovery** is essential for production
11. **Monitor agent performance** continuously
12. **Quality gates** at key checkpoints

---

## 🎯 Pattern Selection Checklist

Use this to choose the right pattern:

| Characteristic | Recommended Pattern |
|---------------|-------------------|
| Task needs classification | **Router** |
| Multiple sequential tasks | **Supervisor** |
| Complex, multi-level project | **Hierarchical** |
| Needs domain experts | **Specialized Sub-Agents** |
| Unreliable dependencies | **Error Recovery** |
| Production deployment | **Orchestration** |
| Simple, single-domain | **Single Agent** |

---
**very important architectural question**

> **Chapter 11 patterns can be implemented in TWO valid ways**
> 1️⃣ **Single-agent, multi-role graphs**
> 2️⃣ **True multi-agent systems using subgraphs (agents-as-nodes)**
>
> **Both are correct — but they solve different problems.**
> We intentionally start with **(1)** in Chapter 11, and **(2)** becomes mandatory in **Multi-Agent chapters (12+)**.

Let’s unpack this carefully.

---

## 1️⃣ Why Chapter 11 LOOKS Like “One Agent”

You’re absolutely right:
In Chapter 11, patterns like:

* Router
* Supervisor
* Critic–Generator
* Plan-and-Execute

were shown as:

* One `StateGraph`
* Multiple nodes
* Conditional edges
* Different *roles* implemented as functions

### That is **intentional**

Because **Chapter 11 is about architectural patterns**, not deployment topology.

> At this stage, we are answering:
> **“How should control flow be structured?”**
> not
> **“How many agents should exist?”**

---

## 2️⃣ Key Concept You Need to Internalize

### 🔑 **Agent ≠ Graph ≠ Node**

These are different abstraction levels.

| Concept | What it really is                      |
| ------- | -------------------------------------- |
| Node    | A step / role / responsibility         |
| Graph   | A control system                       |
| Agent   | A system with autonomy + memory + loop |

👉 In Chapter 11, we are building **one agent with multiple internal roles**.

This is **not wrong** — in fact, it’s often **preferred**.

---

## 3️⃣ Single-Agent, Multi-Role Graphs (Chapter 11)

### What this means

You have:

* One agent
* One state
* One execution loop
* Multiple **roles** implemented as nodes

Example:

```
Single Agent
 ├─ Router node
 ├─ Worker node
 ├─ Critic node
 └─ Aggregator node
```

### Why this is useful

✅ Simpler
✅ Easier to debug
✅ Cheaper
✅ Deterministic
✅ Shared state is easy

### When this is the RIGHT choice

Use **single-agent patterns** when:

* Tasks are tightly coupled
* Shared context is critical
* You want strong control
* You don’t need autonomy between roles

> **Most production “agents” should start this way.**

---

## 4️⃣ So When Do We Need Subgraphs / Agents-as-Nodes?

This is the **next level**, and yes — **we WILL cover it**, explicitly.

### This belongs to **Multi-Agent Systems (Chapters 12–15)**

Now let’s see the difference.

---

## 5️⃣ True Multi-Agent Systems (Subgraphs)

### What changes fundamentally

Instead of:

```
One Graph
 ├─ Node A
 ├─ Node B
 └─ Node C
```

You now have:

```
Top-Level Graph
 ├─ Agent A (subgraph)
 ├─ Agent B (subgraph)
 └─ Agent C (subgraph)
```

Each sub-agent has:

* Its own state
* Its own memory
* Its own loop
* Its own termination
* Possibly its own tools

This is **not just an implementation detail** — it’s a **different system**.

---

## 6️⃣ Why We Don’t Jump to Subgraphs Immediately

### Because most people misuse multi-agent systems.

Common beginner mistake:

> “Let’s create multiple agents because it sounds powerful”

Reality:

* More agents = more coordination problems
* More latency
* More cost
* Harder debugging
* Harder evaluation

### Architectural rule (important)

> **If roles share the same state and goals → use ONE agent**
> **If roles have independent goals or autonomy → use MULTIPLE agents**

---

## 7️⃣ Concrete Comparison (Very Important)

### Router Pattern — Two Ways

#### ✅ Single-Agent Router (Chapter 11)

```
StateGraph
 ├─ router_node()
 ├─ math_handler()
 └─ qa_handler()
```

* One agent
* Deterministic routing
* Shared context

Best for:

* Tool routing
* Task classification

---

#### 🤖 Multi-Agent Router (Later)

```
Coordinator Agent
 ├─ Math Agent
 └─ QA Agent
```

* Independent reasoning
* Independent memory
* More expensive

Best for:

* Different expertise
* Long-running autonomy

---

## 8️⃣ Supervisor Pattern — Two Ways

### Single-Agent Supervisor

* Supervisor plans
* Workers are nodes
* Shared state
* Graph-controlled

This is **Plan-and-Execute**.

### Multi-Agent Supervisor

* Supervisor is an agent
* Workers are agents
* Parallel execution
* Message passing

This is **Manager–Worker multi-agent**.

> Same pattern.
> Different *deployment topology*.

---

## 9️⃣ Why Chapter 11 Was Done This Way (Pedagogical Reason)

We intentionally did Chapter 11 with:

* One graph
* One agent
* Multiple roles

Because:

1. It isolates **control-flow patterns**
2. It avoids premature complexity
3. It teaches **thinking in graphs**
4. It prevents “multi-agent hype misuse”

Only after this foundation does multi-agent make sense.

---

## 🔟 Where Subgraphs Fit Exactly

### We will explicitly cover:

* Using a **compiled graph as a node**
* Passing state between graphs
* Message-passing vs shared state
* Failure isolation
* Parallel agent execution

📍 **This starts in Chapter 12**
📍 Deep dive in Chapters 13–15

---

## 1️⃣1️⃣ Interview-Level Answer (Use This)

If asked:

> “Are router/supervisor patterns single-agent or multi-agent?”

Answer:

> “They’re architectural patterns, not agent-count prescriptions.
> You can implement them inside a single agent using graph nodes when roles are tightly coupled and share state.
> When roles need autonomy, independent memory, or parallel execution, the same patterns naturally lift into multi-agent systems using subgraphs or agent-as-node designs.”

That answer is **very strong**.

---

## ✅ Final Takeaway (Memorize This)

> **Chapter 11 teaches *how to structure intelligence***
> **Chapter 12+ teaches *how to distribute intelligence***

You’re asking the *right* questions — this is exactly the mindset needed for real agent systems.

---


## 🎯 Three Approaches to "Multi-Agent" Systems

### Approach 1: Single Graph with Nodes

```python
# ❌ This is NOT truly multi-agent
# ✅ This IS a single agent with multiple steps

workflow = StateGraph(State)
workflow.add_node("researcher", researcher_function)  # Just a function
workflow.add_node("writer", writer_function)          # Just a function
workflow.add_node("reviewer", reviewer_function)      # Just a function
# All nodes share same state, same graph, same execution context
```

**When to use:**
- Simple workflows with clear steps
- All "agents" need same state
- Tight coupling is acceptable
- You want simplicity

---

### Approach 2: Subgraphs (Each Agent is a Graph)

```python
# ✅ TRUE multi-agent with subgraphs
# Each agent is an independent compiled graph

# Agent 1: Research Agent (independent graph)
research_agent = research_workflow.compile()

# Agent 2: Writer Agent (independent graph)  
writer_agent = writer_workflow.compile()

# Parent orchestrator uses agents as nodes
main_workflow.add_node("research", research_agent)  # Graph as node!
main_workflow.add_node("write", writer_agent)       # Graph as node!
```

**When to use:**
- Each agent has complex internal logic
- Agents can be developed/tested independently
- Different state schemas for each agent
- Reusable agent modules

---

### Approach 3: Distributed Agents (Separate Processes)

```python
# ✅ TRUE distributed multi-agent
# Agents run in separate processes/services
# Communicate via message passing, APIs, or queues

# Agent 1 running on server A
research_service = FastAPI_Service(research_agent)

# Agent 2 running on server B
writer_service = FastAPI_Service(writer_agent)

# Coordinator invokes via HTTP/RPC
coordinator.invoke_agent("research_service", task)
```

**When to use:**
- Agents need to scale independently
- Different deployment requirements
- True isolation needed
- Enterprise microservices architecture

---

## 📝 Let me show you the CORRECT implementations

### Example 1: Router Pattern with Subgraphs (The Right Way)

```python
from typing import TypedDict, Annotated, Sequence, Literal
from operator import add
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== APPROACH 2: SUBGRAPHS (PROPER MULTI-AGENT) ====================

# State for individual agents
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add]
    task: str
    result: str

# State for router
class RouterState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add]
    task_type: Literal["code", "data", "text", "math"]
    agent_result: str

llm = ChatOllama(model="llama3.2", temperature=0.3)

# ==================== DEFINE INDIVIDUAL AGENT GRAPHS ====================

# CODE AGENT (independent graph)
def code_agent_node(state: AgentState) -> dict:
    """Code agent implementation"""
    task = state["task"]
    
    code_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert programmer."),
        ("human", "{task}")
    ])
    
    chain = code_prompt | llm
    response = chain.invoke({"task": task})
    
    return {
        "result": response.content,
        "messages": [AIMessage(content=response.content)]
    }

# Build code agent graph
code_workflow = StateGraph(AgentState)
code_workflow.add_node("process", code_agent_node)
code_workflow.set_entry_point("process")
code_workflow.add_edge("process", END)

# Compile code agent (this is a standalone agent!)
code_agent = code_workflow.compile()

# DATA AGENT (independent graph)
def data_agent_node(state: AgentState) -> dict:
    """Data agent implementation"""
    task = state["task"]
    
    data_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a data scientist."),
        ("human", "{task}")
    ])
    
    chain = data_prompt | llm
    response = chain.invoke({"task": task})
    
    return {
        "result": response.content,
        "messages": [AIMessage(content=response.content)]
    }

data_workflow = StateGraph(AgentState)
data_workflow.add_node("process", data_agent_node)
data_workflow.set_entry_point("process")
data_workflow.add_edge("process", END)

# Compile data agent
data_agent = data_workflow.compile()

# TEXT AGENT (independent graph)
def text_agent_node(state: AgentState) -> dict:
    """Text agent implementation"""
    task = state["task"]
    
    text_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a writing expert."),
        ("human", "{task}")
    ])
    
    chain = text_prompt | llm
    response = chain.invoke({"task": task})
    
    return {
        "result": response.content,
        "messages": [AIMessage(content=response.content)]
    }

text_workflow = StateGraph(AgentState)
text_workflow.add_node("process", text_agent_node)
text_workflow.set_entry_point("process")
text_workflow.add_edge("process", END)

# Compile text agent
text_agent = text_workflow.compile()

# ==================== ROUTER GRAPH (USES AGENTS AS NODES) ====================

# Router classifier
router_prompt = ChatPromptTemplate.from_messages([
    ("human", """Classify this task:

Task: {task}

Respond with ONLY: CODE, DATA, or TEXT

Classification:""")
])

router_chain = router_prompt | llm

def router_node(state: RouterState) -> dict:
    """Route to appropriate agent"""
    task = state["messages"][-1].content
    
    logger.info(f"Routing task: {task[:50]}...")
    
    response = router_chain.invoke({"task": task})
    classification = response.content.strip().upper()
    
    task_map = {
        "CODE": "code",
        "DATA": "data",
        "TEXT": "text"
    }
    
    task_type = task_map.get(classification, "text")
    
    logger.info(f"Routed to: {task_type}")
    
    return {"task_type": task_type}

# Wrapper nodes that invoke the compiled agents
def invoke_code_agent(state: RouterState) -> dict:
    """Invoke code agent (separate graph)"""
    logger.info("Invoking CODE AGENT (subgraph)")
    
    task = state["messages"][-1].content
    
    # Invoke the compiled code agent
    result = code_agent.invoke({
        "messages": [],
        "task": task,
        "result": ""
    })
    
    return {
        "agent_result": result["result"],
        "messages": [AIMessage(content=f"[Code Agent] {result['result']}")]
    }

def invoke_data_agent(state: RouterState) -> dict:
    """Invoke data agent (separate graph)"""
    logger.info("Invoking DATA AGENT (subgraph)")
    
    task = state["messages"][-1].content
    
    # Invoke the compiled data agent
    result = data_agent.invoke({
        "messages": [],
        "task": task,
        "result": ""
    })
    
    return {
        "agent_result": result["result"],
        "messages": [AIMessage(content=f"[Data Agent] {result['result']}")]
    }

def invoke_text_agent(state: RouterState) -> dict:
    """Invoke text agent (separate graph)"""
    logger.info("Invoking TEXT AGENT (subgraph)")
    
    task = state["messages"][-1].content
    
    # Invoke the compiled text agent
    result = text_agent.invoke({
        "messages": [],
        "task": task,
        "result": ""
    })
    
    return {
        "agent_result": result["result"],
        "messages": [AIMessage(content=f"[Text Agent] {result['result']}")]
    }

# Router function
def route_to_agent(state: RouterState) -> str:
    """Route based on classification"""
    return state["task_type"]

# Build router graph
router_workflow = StateGraph(RouterState)

router_workflow.add_node("router", router_node)
router_workflow.add_node("code", invoke_code_agent)
router_workflow.add_node("data", invoke_data_agent)
router_workflow.add_node("text", invoke_text_agent)

router_workflow.set_entry_point("router")

router_workflow.add_conditional_edges(
    "router",
    route_to_agent,
    {
        "code": "code",
        "data": "data",
        "text": "text"
    }
)

router_workflow.add_edge("code", END)
router_workflow.add_edge("data", END)
router_workflow.add_edge("text", END)

# Compile router
multi_agent_router = router_workflow.compile()

# Test
def test_subgraph_router(task: str):
    """Test router with subgraphs"""
    
    result = multi_agent_router.invoke({
        "messages": [HumanMessage(content=task)],
        "task_type": "text",
        "agent_result": ""
    })
    
    print(f"\n{'='*60}")
    print(f"SUBGRAPH ROUTER (TRUE MULTI-AGENT)")
    print(f"{'='*60}")
    print(f"Task: {task}")
    print(f"Routed to: {result['task_type']} agent")
    print(f"Result: {result['agent_result'][:200]}...")

if __name__ == "__main__":
    test_subgraph_router("Write a Python function to sort a list")
    test_subgraph_router("Analyze this dataset for trends")
    test_subgraph_router("Write a blog post about AI")
```

---

### Example 2: Supervisor with Subgraphs

```python
from typing import TypedDict, Annotated, Sequence, List
from operator import add
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== SUPERVISOR WITH TRUE WORKER AGENTS ====================

# Worker agent state
class WorkerState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add]
    task: str
    context: str
    result: str

# Supervisor state
class SupervisorState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add]
    plan: List[str]
    current_step: int
    worker_results: Annotated[List[dict], add]
    final_result: str

llm = ChatOllama(model="llama3.2", temperature=0.3)

# ==================== DEFINE WORKER AGENT GRAPHS ====================

# RESEARCHER AGENT
def researcher_process(state: WorkerState) -> dict:
    """Research worker"""
    logger.info("RESEARCHER AGENT processing...")
    
    from langchain_core.prompts import ChatPromptTemplate
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a researcher. Gather information and facts."),
        ("human", """Task: {task}
Context: {context}

Research:""")
    ])
    
    chain = prompt | llm
    response = chain.invoke({
        "task": state["task"],
        "context": state.get("context", "No context")
    })
    
    return {"result": response.content}

researcher_workflow = StateGraph(WorkerState)
researcher_workflow.add_node("research", researcher_process)
researcher_workflow.set_entry_point("research")
researcher_workflow.add_edge("research", END)

researcher_agent = researcher_workflow.compile()
logger.info("✅ Researcher agent compiled")

# WRITER AGENT
def writer_process(state: WorkerState) -> dict:
    """Writer worker"""
    logger.info("WRITER AGENT processing...")
    
    from langchain_core.prompts import ChatPromptTemplate
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a writer. Create clear, engaging content."),
        ("human", """Task: {task}
Context: {context}

Write:""")
    ])
    
    chain = prompt | llm
    response = chain.invoke({
        "task": state["task"],
        "context": state.get("context", "No context")
    })
    
    return {"result": response.content}

writer_workflow = StateGraph(WorkerState)
writer_workflow.add_node("write", writer_process)
writer_workflow.set_entry_point("write")
writer_workflow.add_edge("write", END)

writer_agent = writer_workflow.compile()
logger.info("✅ Writer agent compiled")

# REVIEWER AGENT
def reviewer_process(state: WorkerState) -> dict:
    """Reviewer worker"""
    logger.info("REVIEWER AGENT processing...")
    
    from langchain_core.prompts import ChatPromptTemplate
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a reviewer. Provide constructive feedback."),
        ("human", """Task: {task}
Content to review: {context}

Review:""")
    ])
    
    chain = prompt | llm
    response = chain.invoke({
        "task": state["task"],
        "context": state.get("context", "No context")
    })
    
    return {"result": response.content}

reviewer_workflow = StateGraph(WorkerState)
reviewer_workflow.add_node("review", reviewer_process)
reviewer_workflow.set_entry_point("review")
reviewer_workflow.add_edge("review", END)

reviewer_agent = reviewer_workflow.compile()
logger.info("✅ Reviewer agent compiled")

# ==================== SUPERVISOR GRAPH ====================

def supervisor_plan(state: SupervisorState) -> dict:
    """Supervisor creates plan"""
    logger.info("SUPERVISOR creating plan...")
    
    plan = [
        "Research the topic",
        "Write content based on research",
        "Review and improve content"
    ]
    
    return {
        "plan": plan,
        "current_step": 0
    }

def invoke_researcher(state: SupervisorState) -> dict:
    """Supervisor invokes researcher agent"""
    logger.info("SUPERVISOR → calling RESEARCHER AGENT")
    
    task = state["messages"][-1].content
    
    # Get context from previous work
    context = "\n".join([
        f"{r['worker']}: {r['result'][:100]}..."
        for r in state["worker_results"]
    ]) if state["worker_results"] else "No previous context"
    
    # Invoke researcher agent (separate graph!)
    result = researcher_agent.invoke({
        "messages": [],
        "task": task,
        "context": context,
        "result": ""
    })
    
    worker_result = {
        "worker": "researcher",
        "result": result["result"]
    }
    
    return {
        "worker_results": [worker_result],
        "current_step": 1
    }

def invoke_writer(state: SupervisorState) -> dict:
    """Supervisor invokes writer agent"""
    logger.info("SUPERVISOR → calling WRITER AGENT")
    
    task = state["messages"][-1].content
    
    # Get research context
    context = ""
    for r in state["worker_results"]:
        if r["worker"] == "researcher":
            context = r["result"]
            break
    
    # Invoke writer agent (separate graph!)
    result = writer_agent.invoke({
        "messages": [],
        "task": task,
        "context": context,
        "result": ""
    })
    
    worker_result = {
        "worker": "writer",
        "result": result["result"]
    }
    
    return {
        "worker_results": [worker_result],
        "current_step": 2
    }

def invoke_reviewer(state: SupervisorState) -> dict:
    """Supervisor invokes reviewer agent"""
    logger.info("SUPERVISOR → calling REVIEWER AGENT")
    
    # Get writer's content
    context = ""
    for r in state["worker_results"]:
        if r["worker"] == "writer":
            context = r["result"]
            break
    
    # Invoke reviewer agent (separate graph!)
    result = reviewer_agent.invoke({
        "messages": [],
        "task": "Review and improve this content",
        "context": context,
        "result": ""
    })
    
    worker_result = {
        "worker": "reviewer",
        "result": result["result"]
    }
    
    return {
        "worker_results": [worker_result],
        "current_step": 3
    }

def finalize(state: SupervisorState) -> dict:
    """Finalize results"""
    logger.info("SUPERVISOR finalizing...")
    
    # Get final result (from reviewer)
    final = ""
    for r in reversed(state["worker_results"]):
        if r["worker"] == "reviewer":
            final = r["result"]
            break
    
    return {
        "final_result": final,
        "messages": [AIMessage(content=final)]
    }

# Router
def supervisor_route(state: SupervisorState) -> str:
    """Route to next worker"""
    step = state["current_step"]
    
    if step == 0:
        return "researcher"
    elif step == 1:
        return "writer"
    elif step == 2:
        return "reviewer"
    else:
        return "finalize"

# Build supervisor graph
supervisor_workflow = StateGraph(SupervisorState)

supervisor_workflow.add_node("plan", supervisor_plan)
supervisor_workflow.add_node("researcher", invoke_researcher)
supervisor_workflow.add_node("writer", invoke_writer)
supervisor_workflow.add_node("reviewer", invoke_reviewer)
supervisor_workflow.add_node("finalize", finalize)

supervisor_workflow.set_entry_point("plan")

supervisor_workflow.add_conditional_edges(
    "plan",
    supervisor_route,
    {
        "researcher": "researcher",
        "writer": "writer",
        "reviewer": "reviewer",
        "finalize": "finalize"
    }
)

supervisor_workflow.add_conditional_edges(
    "researcher",
    supervisor_route,
    {
        "researcher": "researcher",
        "writer": "writer",
        "reviewer": "reviewer",
        "finalize": "finalize"
    }
)

supervisor_workflow.add_conditional_edges(
    "writer",
    supervisor_route,
    {
        "researcher": "researcher",
        "writer": "writer",
        "reviewer": "reviewer",
        "finalize": "finalize"
    }
)

supervisor_workflow.add_conditional_edges(
    "reviewer",
    supervisor_route,
    {
        "researcher": "researcher",
        "writer": "writer",
        "reviewer": "reviewer",
        "finalize": "finalize"
    }
)

supervisor_workflow.add_edge("finalize", END)

# Compile supervisor
multi_agent_supervisor = supervisor_workflow.compile()

# Test
def test_subgraph_supervisor(task: str):
    """Test supervisor with worker agents as subgraphs"""
    
    result = multi_agent_supervisor.invoke({
        "messages": [HumanMessage(content=task)],
        "plan": [],
        "current_step": 0,
        "worker_results": [],
        "final_result": ""
    })
    
    print(f"\n{'='*60}")
    print(f"SUPERVISOR WITH WORKER AGENTS (SUBGRAPHS)")
    print(f"{'='*60}")
    print(f"Task: {task}")
    print(f"Workers called: {len(result['worker_results'])}")
    for r in result['worker_results']:
        print(f"  - {r['worker']}")
    print(f"\nFinal Result:\n{result['final_result'][:300]}...")

if __name__ == "__main__":
    test_subgraph_supervisor("Write a short article about machine learning")
```

---

## 🎯 When to Use Each Approach

| Scenario | Use Approach |
|----------|--------------|
| Simple sequential workflow | ❶ Single graph with nodes |
| Each "agent" is complex internally | ❷ Subgraphs |
| Want to reuse agents in different systems | ❷ Subgraphs |
| Need independent testing of agents | ❷ Subgraphs |
| Agents scale independently | ❸ Distributed |
| Enterprise microservices | ❸ Distributed |
| Different teams build agents | ❸ Distributed |

---


## 🚀 What's Next?

In **Chapter 12**, we'll explore:
- **Multi-Agent Communication**
- Message passing protocols
- Shared memory patterns
- Agent coordination
- Consensus mechanisms
- Conflict resolution
- Production multi-agent systems

---

## ✅ Chapter 11 Complete!

**You now understand:**
- ✅ Router agent pattern for task classification
- ✅ Supervisor pattern for worker management
- ✅ Hierarchical pattern for complex projects
- ✅ Specialized sub-agent collaboration
- ✅ Error recovery with retry/fallback strategies
- ✅ Complete production orchestration
- ✅ Pattern selection guidelines
- ✅ Agent communication interfaces
- ✅ State management across agents
- ✅ Monitoring and metrics
- ✅ Best practices for production

**Ready for Chapter 12?** Just say "Continue to Chapter 12" or ask any questions!