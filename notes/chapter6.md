# Chapter 6: Reflection & Self-Improvement (Reflexion)

## 🎯 The Problem: Agents Need to Learn from Mistakes

Even with great reasoning (CoT, ReAct, etc.), agents still:
- Make errors in their first attempt
- Miss edge cases
- Generate incomplete solutions
- Can't improve without feedback

**Reflection** and **Reflexion** solve this by adding self-critique and learning loops.

---

## 🔍 Core Concepts

### Reflection vs Reflexion

| Concept | What It Does | Scope |
|---------|--------------|-------|
| **Reflection** | Self-critique within a single task | Single conversation |
| **Reflexion** | Learn from feedback across multiple tasks | Multiple conversations |

```
Reflection (single task):
Generate → Critique → Improve → Done

Reflexion (across tasks):
Task 1: Generate → Fail → Store lesson
Task 2: Generate (with lesson) → Fail → Store lesson  
Task 3: Generate (with lessons) → Success!
```

---

## 🎓 Part 1: Reflection Agent (Educational)

### Simple Concept Demo

```python
from typing import TypedDict, Annotated
from operator import add
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# State
class ReflectionState(TypedDict):
    task: str
    solution: str
    critique: str
    revision: str
    iteration: Annotated[int, add]
    max_iterations: int

llm = ChatOllama(model="llama3.2", temperature=0.7)

# Prompts
generate_prompt = ChatPromptTemplate.from_messages([
    ("human", """Task: {task}

Previous critique (if any): {critique}

Generate a solution. If there was a critique, improve based on it.

Solution:""")
])

critique_prompt = ChatPromptTemplate.from_messages([
    ("human", """Task: {task}

Current solution:
{solution}

Critique this solution. What could be improved? Be specific and constructive.

Critique:""")
])

# Chains
generate_chain = generate_prompt | llm
critique_chain = critique_prompt | llm

def generate(state: ReflectionState):
    """Generate or revise solution"""
    response = generate_chain.invoke({
        "task": state["task"],
        "critique": state.get("critique", "None yet")
    })
    
    return {
        "solution": response.content,
        "iteration": 1
    }

def reflect(state: ReflectionState):
    """Critique the solution"""
    response = critique_chain.invoke({
        "task": state["task"],
        "solution": state["solution"]
    })
    
    return {"critique": response.content}

def should_continue(state: ReflectionState) -> str:
    if state["iteration"] >= state["max_iterations"]:
        return "end"
    
    # Simple heuristic: if critique mentions "good" or "excellent", stop
    if state.get("critique"):
        critique_lower = state["critique"].lower()
        if "good" in critique_lower or "excellent" in critique_lower:
            return "end"
    
    return "revise"

# Build graph
workflow = StateGraph(ReflectionState)
workflow.add_node("generate", generate)
workflow.add_node("reflect", reflect)

workflow.set_entry_point("generate")
workflow.add_edge("generate", "reflect")
workflow.add_conditional_edges(
    "reflect",
    should_continue,
    {
        "revise": "generate",
        "end": END
    }
)

app = workflow.compile()

# Test
result = app.invoke({
    "task": "Write a haiku about programming",
    "solution": "",
    "critique": "",
    "revision": "",
    "iteration": 0,
    "max_iterations": 2
})

print(f"Final solution:\n{result['solution']}")
```

---

## 🏭 Part 2: Reflection Agent (PRODUCTION)

### Using LangGraph's Built-in Patterns

LangGraph doesn't have a prebuilt `create_reflection_agent`, but we can use **production patterns**:

```python
from typing import TypedDict, Annotated, Literal, Sequence
from operator import add
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Production state with proper message handling
class ProductionReflectionState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add]
    current_solution: str
    reflection_count: Annotated[int, add]
    max_reflections: int
    quality_threshold: float
    current_quality: float
    status: Literal["generating", "reflecting", "complete", "error"]

# Initialize LLM
llm = ChatOllama(model="llama3.2", temperature=0.7)

# System prompts as constants (best practice)
GENERATOR_SYSTEM = """You are an expert problem solver. Generate high-quality solutions.
If you receive feedback, incorporate it to improve your solution."""

REFLECTOR_SYSTEM = """You are a critical reviewer. Analyze solutions objectively.

Provide:
1. Quality score (0-10)
2. Specific strengths
3. Specific weaknesses  
4. Actionable improvement suggestions

Format:
SCORE: X/10
STRENGTHS: ...
WEAKNESSES: ...
SUGGESTIONS: ..."""

# Prompts with proper templates
generate_prompt = ChatPromptTemplate.from_messages([
    ("system", GENERATOR_SYSTEM),
    ("placeholder", "{messages}"),
])

reflect_prompt = ChatPromptTemplate.from_messages([
    ("system", REFLECTOR_SYSTEM),
    ("human", """Original task: {task}

Current solution:
{solution}

Provide your analysis:""")
])

# Chains
generate_chain = generate_prompt | llm
reflect_chain = reflect_prompt | llm

# Nodes with error handling
def generate_solution(state: ProductionReflectionState) -> dict:
    """Generate or improve solution based on messages"""
    try:
        logger.info(f"Generating solution (attempt {state['reflection_count'] + 1})")
        
        response = generate_chain.invoke({"messages": state["messages"]})
        
        return {
            "messages": [AIMessage(content=response.content)],
            "current_solution": response.content,
            "status": "reflecting",
            "reflection_count": 1
        }
    
    except Exception as e:
        logger.error(f"Generation error: {e}")
        return {
            "status": "error",
            "messages": [AIMessage(content=f"Error: {str(e)}")]
        }

def reflect_on_solution(state: ProductionReflectionState) -> dict:
    """Provide critique and quality assessment"""
    try:
        logger.info("Reflecting on solution")
        
        # Extract original task from first human message
        original_task = ""
        for msg in state["messages"]:
            if isinstance(msg, HumanMessage):
                original_task = msg.content
                break
        
        response = reflect_chain.invoke({
            "task": original_task,
            "solution": state["current_solution"]
        })
        
        reflection = response.content
        
        # Parse quality score
        quality = 5.0  # Default
        if "SCORE:" in reflection:
            try:
                score_line = [line for line in reflection.split('\n') if 'SCORE:' in line][0]
                score_str = score_line.split('SCORE:')[1].split('/')[0].strip()
                quality = float(score_str)
            except:
                logger.warning("Could not parse quality score")
        
        return {
            "messages": [SystemMessage(content=f"Reflection: {reflection}")],
            "current_quality": quality,
            "status": "generating"  # Will decide next in router
        }
    
    except Exception as e:
        logger.error(f"Reflection error: {e}")
        return {
            "status": "error",
            "messages": [SystemMessage(content=f"Reflection error: {str(e)}")]
        }

# Router with quality threshold
def should_continue_reflecting(state: ProductionReflectionState) -> str:
    """Decide whether to continue reflection loop"""
    
    # Error state
    if state["status"] == "error":
        logger.error("Stopping due to error")
        return "end"
    
    # Max iterations reached
    if state["reflection_count"] >= state["max_reflections"]:
        logger.info(f"Max reflections ({state['max_reflections']}) reached")
        return "end"
    
    # Quality threshold met
    if state["current_quality"] >= state["quality_threshold"]:
        logger.info(f"Quality threshold met: {state['current_quality']:.1f}/10")
        return "end"
    
    # Continue improving
    logger.info(f"Quality {state['current_quality']:.1f}/10 below threshold {state['quality_threshold']}, continuing")
    return "generate"

# Build production graph
workflow = StateGraph(ProductionReflectionState)

workflow.add_node("generate", generate_solution)
workflow.add_node("reflect", reflect_on_solution)

workflow.set_entry_point("generate")
workflow.add_edge("generate", "reflect")
workflow.add_conditional_edges(
    "reflect",
    should_continue_reflecting,
    {
        "generate": "generate",
        "end": END
    }
)

# Compile with checkpointing for production
checkpointer = MemorySaver()
production_reflection_agent = workflow.compile(checkpointer=checkpointer)

# Production API
def reflect_and_improve(
    task: str,
    max_reflections: int = 3,
    quality_threshold: float = 8.0,
    session_id: str = "default"
) -> dict:
    """
    Production-ready reflection agent.
    
    Args:
        task: The problem to solve
        max_reflections: Maximum number of improvement iterations
        quality_threshold: Minimum quality score to accept (0-10)
        session_id: Session identifier for checkpointing
    
    Returns:
        dict with solution, quality, and metadata
    """
    
    config = {
        "configurable": {
            "thread_id": f"reflection-{session_id}"
        }
    }
    
    initial_state = {
        "messages": [HumanMessage(content=task)],
        "current_solution": "",
        "reflection_count": 0,
        "max_reflections": max_reflections,
        "quality_threshold": quality_threshold,
        "current_quality": 0.0,
        "status": "generating"
    }
    
    try:
        result = production_reflection_agent.invoke(initial_state, config=config)
        
        # Extract reflections
        reflections = [
            msg.content for msg in result["messages"] 
            if isinstance(msg, SystemMessage) and "Reflection:" in msg.content
        ]
        
        return {
            "success": True,
            "solution": result["current_solution"],
            "quality_score": result["current_quality"],
            "iterations": result["reflection_count"],
            "reflections": reflections,
            "status": result["status"]
        }
    
    except Exception as e:
        logger.error(f"Reflection agent failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }

# Example usage
if __name__ == "__main__":
    result = reflect_and_improve(
        task="Write a Python function to calculate fibonacci numbers efficiently",
        max_reflections=3,
        quality_threshold=8.0
    )
    
    if result["success"]:
        print(f"\n{'='*60}")
        print(f"FINAL SOLUTION (Quality: {result['quality_score']}/10)")
        print(f"{'='*60}")
        print(result["solution"])
        print(f"\nIterations: {result['iterations']}")
    else:
        print(f"Error: {result['error']}")
```

---

## 🧠 Part 3: Reflexion (Learning Across Tasks)

### Concept: Memory-Augmented Reflection

**Reflexion** stores lessons learned from failures and applies them to future tasks.

```
Task 1: Write SQL query → Fail (forgot JOIN) → Store: "Remember to use JOINs"
Task 2: Write SQL query → Success (uses stored lesson)
Task 3: Different task → Still has access to all lessons
```

### Production Implementation with Persistent Memory

```python
from typing import TypedDict, Annotated, Sequence, List
from operator import add
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# State for Reflexion
class ReflexionState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add]
    task: str
    solution: str
    feedback: str  # External or self-generated
    lessons_learned: Annotated[List[str], add]
    attempt_count: Annotated[int, add]
    max_attempts: int
    success: bool

# Initialize
llm = ChatOllama(model="llama3.2", temperature=0.7)

# Prompts
REFLEXION_SYSTEM = """You are a learning agent. You improve by learning from past experiences.

Available lessons from past tasks:
{lessons}

Use these lessons to inform your solution."""

generate_with_memory_prompt = ChatPromptTemplate.from_messages([
    ("system", REFLEXION_SYSTEM),
    ("human", "{task}")
])

analyze_failure_prompt = ChatPromptTemplate.from_messages([
    ("human", """Task: {task}

Your solution:
{solution}

Feedback/Error:
{feedback}

Extract a concrete lesson from this failure. What should you remember for future tasks?

Format as: "Lesson: [specific actionable insight]"

Your analysis:""")
])

# Chains
generate_chain = generate_with_memory_prompt | llm
analyze_chain = analyze_failure_prompt | llm

# Simulated environment for testing (replace with real evaluator)
def evaluate_solution(task: str, solution: str) -> tuple[bool, str]:
    """
    Simulate evaluation of solution.
    In production, this could be:
    - Unit tests for code
    - Human feedback
    - Automated grading
    - Tool execution results
    """
    
    # Simple heuristic for demo
    if "function" in task.lower() and "def " not in solution.lower():
        return False, "Solution doesn't define a function"
    
    if "efficient" in task.lower() and "O(n)" not in solution.lower() and "memoization" not in solution.lower():
        return False, "Solution doesn't mention efficiency considerations"
    
    if len(solution.strip()) < 50:
        return False, "Solution is too brief"
    
    return True, "Solution looks good"

# Nodes
def generate_solution(state: ReflexionState) -> dict:
    """Generate solution using past lessons"""
    try:
        logger.info(f"Generating solution (attempt {state['attempt_count'] + 1})")
        
        # Format lessons for prompt
        lessons_text = "\n".join([
            f"- {lesson}" for lesson in state["lessons_learned"]
        ]) if state["lessons_learned"] else "No lessons yet."
        
        response = generate_chain.invoke({
            "lessons": lessons_text,
            "task": state["task"]
        })
        
        solution = response.content
        
        # Evaluate solution
        success, feedback = evaluate_solution(state["task"], solution)
        
        return {
            "messages": [AIMessage(content=solution)],
            "solution": solution,
            "feedback": feedback,
            "success": success,
            "attempt_count": 1
        }
    
    except Exception as e:
        logger.error(f"Generation error: {e}")
        return {
            "feedback": str(e),
            "success": False,
            "attempt_count": 1
        }

def learn_from_failure(state: ReflexionState) -> dict:
    """Extract lesson from failure"""
    
    if state["success"]:
        # No learning needed on success
        return {}
    
    try:
        logger.info("Analyzing failure to extract lesson")
        
        response = analyze_chain.invoke({
            "task": state["task"],
            "solution": state["solution"],
            "feedback": state["feedback"]
        })
        
        analysis = response.content
        
        # Extract lesson
        lesson = "General improvement needed"
        if "Lesson:" in analysis:
            lesson = analysis.split("Lesson:")[1].strip().split('\n')[0]
        
        logger.info(f"Learned: {lesson}")
        
        return {
            "lessons_learned": [lesson],
            "messages": [SystemMessage(content=f"Learned: {lesson}")]
        }
    
    except Exception as e:
        logger.error(f"Learning error: {e}")
        return {}

# Router
def should_retry(state: ReflexionState) -> str:
    """Decide whether to try again"""
    
    if state["success"]:
        logger.info("Solution successful!")
        return "end"
    
    if state["attempt_count"] >= state["max_attempts"]:
        logger.warning(f"Max attempts ({state['max_attempts']}) reached")
        return "end"
    
    logger.info("Retrying with new lesson")
    return "retry"

# Build graph
workflow = StateGraph(ReflexionState)

workflow.add_node("generate", generate_solution)
workflow.add_node("learn", learn_from_failure)

workflow.set_entry_point("generate")
workflow.add_conditional_edges(
    "generate",
    should_retry,
    {
        "retry": "learn",
        "end": END
    }
)
workflow.add_edge("learn", "generate")

# Compile with SQLite checkpointer for persistence
checkpointer = SqliteSaver.from_conn_string("./reflexion_memory.db")
reflexion_agent = workflow.compile(checkpointer=checkpointer)

# Production API
def solve_with_reflexion(
    task: str,
    user_id: str = "default",
    max_attempts: int = 3
) -> dict:
    """
    Solve task using reflexion (learns from past tasks in this session).
    
    Args:
        task: The problem to solve
        user_id: User identifier (lessons are per-user)
        max_attempts: Maximum retry attempts
    
    Returns:
        dict with solution and learning history
    """
    
    config = {
        "configurable": {
            "thread_id": f"reflexion-{user_id}"
        }
    }
    
    initial_state = {
        "messages": [HumanMessage(content=task)],
        "task": task,
        "solution": "",
        "feedback": "",
        "lessons_learned": [],
        "attempt_count": 0,
        "max_attempts": max_attempts,
        "success": False
    }
    
    try:
        result = reflexion_agent.invoke(initial_state, config=config)
        
        return {
            "success": result["success"],
            "solution": result["solution"],
            "feedback": result["feedback"],
            "attempts": result["attempt_count"],
            "lessons_learned": result["lessons_learned"],
            "total_lessons": len(result["lessons_learned"])
        }
    
    except Exception as e:
        logger.error(f"Reflexion failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }

# Example: Multi-task learning
if __name__ == "__main__":
    user_id = "developer-123"
    
    tasks = [
        "Write a Python function to reverse a string efficiently",
        "Write a Python function to find duplicates in a list efficiently",
        "Write a Python function to merge two sorted lists efficiently"
    ]
    
    print("\n" + "="*60)
    print("REFLEXION DEMO: Learning Across Tasks")
    print("="*60)
    
    for i, task in enumerate(tasks, 1):
        print(f"\n--- TASK {i} ---")
        print(f"Task: {task}")
        
        result = solve_with_reflexion(task, user_id=user_id)
        
        if result["success"]:
            print(f"✅ Success on attempt {result['attempts']}")
            print(f"Solution: {result['solution'][:100]}...")
        else:
            print(f"❌ Failed after {result['attempts']} attempts")
            print(f"Feedback: {result['feedback']}")
        
        print(f"Lessons learned this task: {len(result['lessons_learned'])}")
        print(f"Total lessons accumulated: {result['total_lessons']}")
    
    # Show all lessons learned
    print(f"\n{'='*60}")
    print("ALL LESSONS LEARNED:")
    print(f"{'='*60}")
    
    # Get final state to see all lessons
    final_result = solve_with_reflexion(
        "Test task to retrieve lessons",
        user_id=user_id,
        max_attempts=1
    )
    
    for i, lesson in enumerate(final_result["lessons_learned"], 1):
        print(f"{i}. {lesson}")
```

---

## 🎯 Part 4: Advanced Pattern - Tool-Using Reflection

### Production: Reflection + ReAct with bind_tools

```python
from typing import TypedDict, Annotated, Sequence
from operator import add
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate

# Define tools
@tool
def execute_python(code: str) -> str:
    """Execute Python code safely and return the result."""
    try:
        # In production, use a sandbox
        local_vars = {}
        exec(code, {"__builtins__": {}}, local_vars)
        return f"Executed successfully. Variables: {local_vars}"
    except Exception as e:
        return f"Execution error: {str(e)}"

@tool
def validate_code(code: str) -> str:
    """Validate Python code for syntax errors."""
    try:
        compile(code, '<string>', 'exec')
        return "Code is syntactically valid"
    except SyntaxError as e:
        return f"Syntax error: {str(e)}"

tools = [execute_python, validate_code]

# State
class ToolReflectionState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add]
    reflection_count: Annotated[int, add]
    max_reflections: int

# Initialize
llm = ChatOllama(model="llama3.2", temperature=0.7)
llm_with_tools = llm.bind_tools(tools)

# Nodes
def agent_with_reflection(state: ToolReflectionState):
    """Agent that can use tools and self-reflect"""
    
    # Add reflection prompt if we've done iterations
    if state["reflection_count"] > 0:
        reflection_msg = SystemMessage(
            content="Review your previous attempts. Use tools to validate your solution."
        )
        messages = list(state["messages"]) + [reflection_msg]
    else:
        messages = state["messages"]
    
    response = llm_with_tools.invoke(messages)
    
    return {
        "messages": [response],
        "reflection_count": 1
    }

tool_node = ToolNode(tools)

def should_continue(state: ToolReflectionState):
    """Route to tools or reflect"""
    last_message = state["messages"][-1]
    
    # Check for tool calls
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    
    # Check reflection limit
    if state["reflection_count"] >= state["max_reflections"]:
        return "end"
    
    # Could add logic here to decide if solution needs reflection
    return "end"

# Build graph
workflow = StateGraph(ToolReflectionState)
workflow.add_node("agent", agent_with_reflection)
workflow.add_node("tools", tool_node)

workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "end": END
    }
)
workflow.add_edge("tools", "agent")  # After tools, back to agent

tool_reflection_agent = workflow.compile()

# Usage
result = tool_reflection_agent.invoke({
    "messages": [HumanMessage(content="Write and test a function to calculate factorial")],
    "reflection_count": 0,
    "max_reflections": 3
})

print("\n=== Tool-Using Reflection Trace ===")
for msg in result["messages"]:
    print(f"{type(msg).__name__}: {getattr(msg, 'content', 'tool call')[:100]}")
```

---

## 📊 Comparison: Reflection Patterns

| Pattern | Use Case | Memory | Tool Use | Complexity |
|---------|----------|--------|----------|------------|
| **Simple Reflection** | Single-task improvement | No | No | Low |
| **Reflexion** | Learning across tasks | Yes (persistent) | No | Medium |
| **Tool Reflection** | Code/technical tasks | Optional | Yes | Medium |
| **Multi-Agent Reflection** | Complex problems | Yes | Yes | High |

---

## 🏭 Production Best Practices Summary

### 1. Always Use Proper Checkpointing

```python
# ❌ BAD: No persistence
app = workflow.compile()

# ✅ GOOD: SQLite for production
from langgraph.checkpoint.sqlite import SqliteSaver
checkpointer = SqliteSaver.from_conn_string("./agent_memory.db")
app = workflow.compile(checkpointer=checkpointer)
```

### 2. Use Structured State

```python
# ❌ BAD: Untyped dictionary
state = {"stuff": [], "things": ""}

# ✅ GOOD: TypedDict with annotations
class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add]
    quality_score: float
    iteration: Annotated[int, add]
```

### 3. Proper Prompt Templates

```python
# ❌ BAD: F-strings
prompt = f"Task: {task}\nDo something"

# ✅ GOOD: ChatPromptTemplate
from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert."),
    ("human", "{task}")
])
```

### 4. bind_tools for Tool Use

```python
# ❌ BAD: Manual parsing
if "search(" in llm_output:
    ...

# ✅ GOOD: bind_tools + ToolNode
llm_with_tools = llm.bind_tools(tools)
tool_node = ToolNode(tools)
```

### 5. Error Handling

```python
# ✅ ALWAYS wrap nodes in try/except
def my_node(state):
    try:
        result = risky_operation()
        return {"result": result}
    except Exception as e:
        logger.error(f"Node failed: {e}")
        return {"error": str(e), "status": "error"}
```

### 6. Logging

```python
import logging
logger = logging.getLogger(__name__)

def my_node(state):
    logger.info(f"Processing iteration {state['iteration']}")
    # ... node logic
```

---

## 🧠 Key Concepts to Remember

1. **Reflection = self-critique within a task**
2. **Reflexion = learning across tasks with memory**
3. **Use checkpointing for persistent memory**
4. **Always use bind_tools, not manual parsing**
5. **Quality thresholds prevent infinite loops**
6. **Proper TypedDict state is essential**
7. **Error handling and logging are non-negotiable**

---

## 🚀 What's Next?

In **Chapter 7**, we'll dive deep into:
- **Tool Use & Function Calling** (production patterns)
- Tool definition best practices
- Parallel tool execution
- Custom tool creation
- Error handling for tools
- Tool use with streaming

---

## ✅ Chapter 6 Complete!

**You now understand:**
- ✅ Reflection vs Reflexion (single task vs learning)
- ✅ Educational implementations (concept understanding)
- ✅ **Production implementations** (LangGraph best practices)
- ✅ Memory-augmented reflection with SQLite
- ✅ Tool-using reflection agents
- ✅ Quality thresholds and termination
- ✅ bind_tools and ToolNode patterns

**Ready for Chapter 7?** Just say "Continue to Chapter 7" or ask any questions!