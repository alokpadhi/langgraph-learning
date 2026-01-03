# Chapter 5: Reasoning Paradigms - Part 2 (Tree of Thoughts, ReAct, Plan-and-Execute)

## 🎯 The Problem: Some Tasks Need Exploration, Not Just Linear Reasoning

Chain-of-Thought works great for linear problems, but what about:
- **Creative tasks** where you need to explore multiple ideas
- **Complex problems** requiring backtracking when a path fails
- **Action-heavy tasks** where reasoning must alternate with tool use
- **Multi-step projects** needing upfront planning

Enter: **Tree of Thoughts**, **ReAct**, and **Plan-and-Execute**.

---

## 🌳 Tree of Thoughts (ToT) - Branching Exploration

### The Core Idea

Instead of one linear reasoning path, explore **multiple branches** like a tree:

```
                    Problem
                       |
        ┌──────────────┼──────────────┐
        ↓              ↓              ↓
    Approach 1     Approach 2     Approach 3
        |              |              |
    ┌───┴───┐      ┌───┴───┐      ┌───┴───┐
    ↓       ↓      ↓       ↓      ↓       ↓
  Step1a  Step1b Step2a  Step2b Step3a  Step3b
    |       ✗      |       ✗      |       |
    ↓              ↓              ↓       ↓
  Result1      Result2        Result3a Result3b
    ↓              ↓              ↓       ↓
Evaluate       Evaluate       Best!   Good
```

**Key differences from CoT:**
- CoT: Single path → Answer
- ToT: Multiple paths → Evaluate → Best path → Answer

---

## 💻 Implementing Tree of Thoughts

### Level 1: Simple ToT (3 Approaches, Pick Best)

```python
from typing import TypedDict, Annotated, Literal
from operator import add
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# State definition
class ToTState(TypedDict):
    problem: str
    approaches: Annotated[list[dict], add]  # [{idea: str, reasoning: str, score: int}]
    num_approaches: int
    approaches_generated: Annotated[int, add]
    best_approach: dict
    final_solution: str

# Initialize
llm = ChatOllama(model="llama3.2", temperature=0.8)  # Higher temp for diversity

# Prompt: Generate different approaches
generate_approach_prompt = ChatPromptTemplate.from_messages([
    ("human", """Problem: {problem}

Generate a unique approach to solve this. Be creative and different from conventional methods.

Approach {approach_num}:""")
])

# Prompt: Develop the approach
develop_approach_prompt = ChatPromptTemplate.from_messages([
    ("human", """Problem: {problem}

Approach: {approach}

Develop this approach step-by-step. Show your reasoning.

Development:""")
])

# Prompt: Evaluate approach
evaluate_approach_prompt = ChatPromptTemplate.from_messages([
    ("human", """Problem: {problem}

Approach and reasoning:
{development}

Rate this approach from 1-10 based on:
- Feasibility (can it actually work?)
- Completeness (does it solve the full problem?)
- Clarity (is it easy to understand?)

Respond with ONLY a number from 1-10.

Rating:""")
])

# Chains
generate_chain = generate_approach_prompt | llm
develop_chain = develop_approach_prompt | llm
evaluate_chain = evaluate_approach_prompt | llm

# Node: Generate one approach
def generate_approach(state: ToTState) -> dict:
    """Generate a unique approach to the problem"""
    
    approach_num = state["approaches_generated"] + 1
    
    # Generate approach
    approach_response = generate_chain.invoke({
        "problem": state["problem"],
        "approach_num": approach_num
    })
    
    approach_idea = approach_response.content.strip()
    
    # Develop the approach
    development_response = develop_chain.invoke({
        "problem": state["problem"],
        "approach": approach_idea
    })
    
    development = development_response.content.strip()
    
    # Evaluate the approach
    eval_response = evaluate_chain.invoke({
        "problem": state["problem"],
        "development": development
    })
    
    # Parse score (defensive)
    score = 5  # Default
    try:
        score_text = eval_response.content.strip()
        # Extract first number found
        import re
        numbers = re.findall(r'\d+', score_text)
        if numbers:
            score = min(int(numbers[0]), 10)  # Cap at 10
    except:
        pass
    
    return {
        "approaches": [{
            "idea": approach_idea,
            "reasoning": development,
            "score": score
        }],
        "approaches_generated": 1
    }

# Node: Select best approach
def select_best(state: ToTState) -> dict:
    """Select the highest-scoring approach"""
    
    if not state["approaches"]:
        return {
            "best_approach": {},
            "final_solution": "No approaches generated"
        }
    
    # Find highest score
    best = max(state["approaches"], key=lambda x: x["score"])
    
    return {
        "best_approach": best,
        "final_solution": best["reasoning"]
    }

# Routing
def should_generate_more(state: ToTState) -> str:
    """Decide if we need more approaches"""
    
    if state["approaches_generated"] >= state["num_approaches"]:
        return "select"
    
    return "generate"

# Build graph
workflow = StateGraph(ToTState)

workflow.add_node("generate", generate_approach)
workflow.add_node("select", select_best)

workflow.set_entry_point("generate")

workflow.add_conditional_edges(
    "generate",
    should_generate_more,
    {
        "generate": "generate",
        "select": "select"
    }
)

workflow.add_edge("select", END)

# Compile
tot_agent = workflow.compile()

# Test function
def solve_with_tot(problem: str, num_approaches: int = 3):
    """Solve using Tree of Thoughts"""
    
    initial_state = {
        "problem": problem,
        "approaches": [],
        "num_approaches": num_approaches,
        "approaches_generated": 0,
        "best_approach": {},
        "final_solution": ""
    }
    
    print(f"\n{'='*60}")
    print(f"PROBLEM: {problem}")
    print(f"{'='*60}\n")
    print(f"Generating {num_approaches} different approaches...\n")
    
    result = tot_agent.invoke(initial_state)
    
    # Display all approaches
    print("ALL APPROACHES:")
    for i, approach in enumerate(result["approaches"], 1):
        print(f"\n--- Approach {i} (Score: {approach['score']}/10) ---")
        print(f"Idea: {approach['idea'][:100]}...")
        print(f"Reasoning: {approach['reasoning'][:150]}...")
    
    print(f"\n{'='*60}")
    print(f"✅ BEST APPROACH (Score: {result['best_approach']['score']}/10)")
    print(f"{'='*60}")
    print(f"\n{result['final_solution']}\n")
    
    return result

# Example usage
if __name__ == "__main__":
    solve_with_tot(
        "How can a small business with a limited budget effectively compete with larger companies?",
        num_approaches=3
    )
```

---

## 🎬 ReAct: Reasoning + Acting

### The Core Idea

**ReAct = Reason about what to do, then Act (use tools), then repeat**

Traditional CoT: Think → Think → Think → Answer
**ReAct: Think → Act → Think → Act → Think → Answer**

```
User: "What's the weather in Paris and convert the temperature to Celsius?"

ReAct Loop:
1. Thought: I need to search for Paris weather
2. Action: search_weather("Paris")
3. Observation: "Paris is 68°F"
4. Thought: Now I need to convert 68°F to Celsius
5. Action: calculate("(68-32)*5/9")
6. Observation: "20°C"
7. Thought: I have both pieces of information
8. Answer: "Paris weather is 68°F (20°C)"
```

---

## 💻 Implementing ReAct with Tool Use

```python
from typing import TypedDict, Annotated, Literal
from operator import add
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
import re

# Define some simple tools
@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    # Simulated search results
    mock_results = {
        "eiffel tower height": "The Eiffel Tower is 330 meters (1,083 feet) tall",
        "python creator": "Python was created by Guido van Rossum in 1991",
        "capital of japan": "Tokyo is the capital of Japan",
    }
    
    query_lower = query.lower()
    for key, value in mock_results.items():
        if key in query_lower:
            return value
    
    return f"Search results for '{query}': No specific information found."

@tool
def calculate(expression: str) -> str:
    """Perform mathematical calculations."""
    try:
        # Safe eval for simple math
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Calculation error: {str(e)}"

# Available tools
tools = [search_web, calculate]
tool_map = {tool.name: tool for tool in tools}

# State definition
class ReActState(TypedDict):
    question: str
    thoughts: Annotated[list[str], add]
    actions: Annotated[list[dict], add]  # {action: str, tool: str, input: str}
    observations: Annotated[list[str], add]
    final_answer: str
    step_count: Annotated[int, add]
    max_steps: int
    current_phase: Literal["think", "act", "answer"]

# Initialize
llm = ChatOllama(model="llama3.2", temperature=0.3)

# Prompt: Thinking phase
think_prompt = ChatPromptTemplate.from_messages([
    ("human", """You are solving this question: {question}

Previous thoughts and observations:
{history}

Available tools:
- search_web: Search for information online
- calculate: Perform math calculations

Think about what to do next. You can either:
1. Use a tool (specify which tool and what input)
2. Provide the final answer (if you have enough information)

Respond in this format:
Thought: [your reasoning]
Action: [tool_name(input)] OR Final Answer: [your answer]

Your response:""")
])

# Chains
think_chain = think_prompt | llm

# Node: Think
def think(state: ReActState) -> dict:
    """Reasoning phase - decide next action"""
    
    # Build history
    history_parts = []
    for i in range(len(state["thoughts"])):
        history_parts.append(f"Thought {i+1}: {state['thoughts'][i]}")
        if i < len(state["observations"]):
            history_parts.append(f"Observation {i+1}: {state['observations'][i]}")
    
    history = "\n".join(history_parts) if history_parts else "No previous steps."
    
    # Get next thought
    response = think_chain.invoke({
        "question": state["question"],
        "history": history
    })
    
    response_text = response.content.strip()
    
    # Parse response
    thought = ""
    action_line = ""
    
    lines = response_text.split('\n')
    for line in lines:
        if line.startswith("Thought:"):
            thought = line.replace("Thought:", "").strip()
        elif line.startswith("Action:"):
            action_line = line.replace("Action:", "").strip()
        elif line.startswith("Final Answer:"):
            # Ready to answer
            final_answer = line.replace("Final Answer:", "").strip()
            return {
                "thoughts": [thought] if thought else [],
                "final_answer": final_answer,
                "current_phase": "answer"
            }
    
    # Parse action (e.g., "search_web(eiffel tower)")
    if action_line:
        match = re.match(r'(\w+)\((.*)\)', action_line)
        if match:
            tool_name = match.group(1)
            tool_input = match.group(2).strip().strip('"\'')
            
            return {
                "thoughts": [thought] if thought else [],
                "actions": [{
                    "action": action_line,
                    "tool": tool_name,
                    "input": tool_input
                }],
                "current_phase": "act",
                "step_count": 1
            }
    
    # Fallback: if no clear action, mark as thought only
    return {
        "thoughts": [thought if thought else response_text],
        "current_phase": "think",
        "step_count": 1
    }

# Node: Act
def act(state: ReActState) -> dict:
    """Execute the chosen action (use tool)"""
    
    if not state["actions"]:
        return {
            "observations": ["No action to execute"],
            "current_phase": "think"
        }
    
    last_action = state["actions"][-1]
    tool_name = last_action["tool"]
    tool_input = last_action["input"]
    
    # Execute tool
    if tool_name in tool_map:
        try:
            result = tool_map[tool_name].invoke(tool_input)
            observation = str(result)
        except Exception as e:
            observation = f"Tool error: {str(e)}"
    else:
        observation = f"Unknown tool: {tool_name}"
    
    return {
        "observations": [observation],
        "current_phase": "think"  # Back to thinking
    }

# Routing
def route_react(state: ReActState) -> str:
    """Route based on current phase"""
    
    # Check step limit
    if state["step_count"] >= state["max_steps"]:
        return "end"
    
    phase = state["current_phase"]
    
    if phase == "think":
        # Check if we decided to act
        if state["actions"] and len(state["actions"]) > len(state["observations"]):
            return "act"
        return "think"  # Keep thinking
    
    elif phase == "act":
        return "think"  # After action, go back to thinking
    
    elif phase == "answer":
        return "end"
    
    return "end"

# Build graph
workflow = StateGraph(ReActState)

workflow.add_node("think", think)
workflow.add_node("act", act)

workflow.set_entry_point("think")

workflow.add_conditional_edges(
    "think",
    route_react,
    {
        "think": "think",
        "act": "act",
        "end": END
    }
)

workflow.add_conditional_edges(
    "act",
    route_react,
    {
        "think": "think",
        "end": END
    }
)

# Compile
react_agent = workflow.compile()

# Test function
def solve_with_react(question: str, max_steps: int = 5):
    """Solve using ReAct pattern"""
    
    initial_state = {
        "question": question,
        "thoughts": [],
        "actions": [],
        "observations": [],
        "final_answer": "",
        "step_count": 0,
        "max_steps": max_steps,
        "current_phase": "think"
    }
    
    print(f"\n{'='*60}")
    print(f"QUESTION: {question}")
    print(f"{'='*60}\n")
    
    result = react_agent.invoke(initial_state)
    
    # Display trace
    print("REACT TRACE:")
    for i in range(len(result["thoughts"])):
        print(f"\n💭 Thought {i+1}: {result['thoughts'][i]}")
        
        if i < len(result["actions"]):
            action = result["actions"][i]
            print(f"🔧 Action {i+1}: {action['action']}")
        
        if i < len(result["observations"]):
            print(f"👁️ Observation {i+1}: {result['observations'][i]}")
    
    print(f"\n{'='*60}")
    print(f"✅ FINAL ANSWER:")
    print(f"{result['final_answer']}")
    print(f"{'='*60}\n")
    
    return result

# Example usage
if __name__ == "__main__":
    # Test 1: Requires search
    solve_with_react(
        "How tall is the Eiffel Tower and what is half of that height in meters?"
    )
    
    # Test 2: Requires calculation
    solve_with_react(
        "What is 15% of 240, and then add 30 to that result?"
    )
```

---

## 📋 Plan-and-Execute Pattern

### The Core Idea

**Instead of acting immediately, create a complete plan first, then execute it.**

```
Traditional:
Think → Act → Think → Act → ...

Plan-and-Execute:
1. Create full plan
2. Execute step 1
3. Execute step 2
4. Execute step 3
5. Review results
```

**Benefits:**
- Better for complex, multi-step tasks
- Can parallelize execution
- Easier to review before execution
- More structured and predictable

---

## 💻 Implementing Plan-and-Execute

```python
from typing import TypedDict, Annotated, Literal
from operator import add
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# State definition
class PlanExecuteState(TypedDict):
    task: str
    plan: list[str]  # List of steps
    plan_created: bool
    current_step_idx: int
    step_results: Annotated[list[dict], add]  # [{step: str, result: str}]
    final_summary: str
    status: Literal["planning", "executing", "reviewing", "complete"]

# Initialize
llm = ChatOllama(model="llama3.2", temperature=0.3)

# Prompt: Create plan
plan_prompt = ChatPromptTemplate.from_messages([
    ("human", """Task: {task}

Create a step-by-step plan to accomplish this task. Each step should be clear and actionable.

Provide your plan as a numbered list:
1. [First step]
2. [Second step]
...

Plan:""")
])

# Prompt: Execute step
execute_prompt = ChatPromptTemplate.from_messages([
    ("human", """Overall task: {task}

Full plan:
{full_plan}

Current step: {current_step}

Execute this step and provide the result.

Result:""")
])

# Prompt: Review
review_prompt = ChatPromptTemplate.from_messages([
    ("human", """Task: {task}

Plan executed:
{plan_with_results}

Provide a brief summary of what was accomplished.

Summary:""")
])

# Chains
plan_chain = plan_prompt | llm
execute_chain = execute_prompt | llm
review_chain = review_prompt | llm

# Node: Create plan
def create_plan(state: PlanExecuteState) -> dict:
    """Create execution plan"""
    
    response = plan_chain.invoke({"task": state["task"]})
    
    # Parse plan (extract numbered items)
    plan_text = response.content
    steps = []
    
    for line in plan_text.split('\n'):
        line = line.strip()
        # Match "1. Step", "1) Step", or "Step 1: ..."
        if line and (line[0].isdigit() or line.startswith('Step')):
            # Clean up the step
            step = re.sub(r'^\d+[\.\)]\s*', '', line)
            step = re.sub(r'^Step\s+\d+:\s*', '', step)
            if step:
                steps.append(step)
    
    if not steps:
        # Fallback: treat whole response as one step
        steps = [plan_text.strip()]
    
    return {
        "plan": steps,
        "plan_created": True,
        "status": "executing"
    }

# Node: Execute step
def execute_step(state: PlanExecuteState) -> dict:
    """Execute current step in plan"""
    
    if state["current_step_idx"] >= len(state["plan"]):
        return {"status": "reviewing"}
    
    current_step = state["plan"][state["current_step_idx"]]
    
    # Format full plan for context
    full_plan = "\n".join([
        f"{i+1}. {step}" 
        for i, step in enumerate(state["plan"])
    ])
    
    # Execute step
    response = execute_chain.invoke({
        "task": state["task"],
        "full_plan": full_plan,
        "current_step": current_step
    })
    
    result = response.content.strip()
    
    # Record result
    step_result = {
        "step": current_step,
        "result": result
    }
    
    # Move to next step
    next_idx = state["current_step_idx"] + 1
    next_status = "executing" if next_idx < len(state["plan"]) else "reviewing"
    
    return {
        "step_results": [step_result],
        "current_step_idx": next_idx,
        "status": next_status
    }

# Node: Review results
def review_results(state: PlanExecuteState) -> dict:
    """Review all execution results"""
    
    # Format plan with results
    plan_with_results = []
    for i, step_result in enumerate(state["step_results"], 1):
        plan_with_results.append(
            f"Step {i}: {step_result['step']}\nResult: {step_result['result']}"
        )
    
    combined = "\n\n".join(plan_with_results)
    
    # Generate summary
    response = review_chain.invoke({
        "task": state["task"],
        "plan_with_results": combined
    })
    
    return {
        "final_summary": response.content.strip(),
        "status": "complete"
    }

# Routing
def route_plan_execute(state: PlanExecuteState) -> str:
    """Route based on status"""
    
    status = state["status"]
    
    if status == "planning":
        return "plan"
    elif status == "executing":
        return "execute"
    elif status == "reviewing":
        return "review"
    elif status == "complete":
        return "end"
    
    return "end"

# Build graph
workflow = StateGraph(PlanExecuteState)

workflow.add_node("plan", create_plan)
workflow.add_node("execute", execute_step)
workflow.add_node("review", review_results)

workflow.set_entry_point("plan")

# Add routing from each node
workflow.add_conditional_edges(
    "plan",
    route_plan_execute,
    {
        "execute": "execute",
        "end": END
    }
)

workflow.add_conditional_edges(
    "execute",
    route_plan_execute,
    {
        "execute": "execute",
        "review": "review",
        "end": END
    }
)

workflow.add_conditional_edges(
    "review",
    route_plan_execute,
    {
        "end": END
    }
)

# Compile
plan_execute_agent = workflow.compile()

# Test function
def solve_with_plan_execute(task: str):
    """Solve using Plan-and-Execute pattern"""
    
    initial_state = {
        "task": task,
        "plan": [],
        "plan_created": False,
        "current_step_idx": 0,
        "step_results": [],
        "final_summary": "",
        "status": "planning"
    }
    
    print(f"\n{'='*60}")
    print(f"TASK: {task}")
    print(f"{'='*60}\n")
    
    result = plan_execute_agent.invoke(initial_state)
    
    # Display plan
    print("📋 PLAN:")
    for i, step in enumerate(result["plan"], 1):
        print(f"  {i}. {step}")
    
    # Display execution
    print(f"\n⚙️ EXECUTION:")
    for i, step_result in enumerate(result["step_results"], 1):
        print(f"\n  Step {i}: {step_result['step']}")
        print(f"  Result: {step_result['result'][:100]}...")
    
    # Display summary
    print(f"\n{'='*60}")
    print(f"✅ SUMMARY:")
    print(f"{result['final_summary']}")
    print(f"{'='*60}\n")
    
    return result

# Example usage
if __name__ == "__main__":
    solve_with_plan_execute(
        "Research the history of artificial intelligence and create a timeline of major milestones"
    )
```

---

## 🎯 When to Use Which Paradigm

| Paradigm | Best For | Avoid For | Example Use Case |
|----------|----------|-----------|------------------|
| **Chain-of-Thought** | Math, logic, linear reasoning | Creative tasks, tool use | "Calculate compound interest" |
| **Tree of Thoughts** | Creative problems, multiple valid solutions | Simple questions, time-critical | "Design a marketing campaign" |
| **ReAct** | Questions needing external info, tool use | Pure reasoning tasks | "What's the weather + convert temp" |
| **Plan-and-Execute** | Complex multi-step projects | Simple queries, real-time | "Organize a conference" |

---

## 🔄 Combining Paradigms

Sometimes the best approach is to **combine** multiple paradigms:

```python
from typing import TypedDict, Annotated, Literal
from operator import add
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# Hybrid state: Plan → ReAct execution → ToT for complex steps
class HybridState(TypedDict):
    task: str
    plan: list[str]
    current_step: int
    step_type: Literal["simple", "complex"]  # Simple=ReAct, Complex=ToT
    step_results: Annotated[list[str], add]
    final_result: str

llm = ChatOllama(model="llama3.2", temperature=0.5)

# Classifier: Determine if step is simple or complex
classify_prompt = ChatPromptTemplate.from_messages([
    ("human", """Step: {step}

Is this step simple (needs one action) or complex (needs multiple approaches)?

Respond with ONLY: SIMPLE or COMPLEX

Classification:""")
])

classify_chain = classify_prompt | llm

def classify_step(state: HybridState) -> dict:
    """Classify current step complexity"""
    
    current_step = state["plan"][state["current_step"]]
    
    response = classify_chain.invoke({"step": current_step})
    
    step_type = "simple"
    if "complex" in response.content.lower():
        step_type = "complex"
    
    return {"step_type": step_type}

def route_by_complexity(state: HybridState) -> str:
    """Route to appropriate execution method"""
    
    if state["current_step"] >= len(state["plan"]):
        return "end"
    
    if state["step_type"] == "simple":
        return "react_execute"
    else:
        return "tot_execute"

# This is a conceptual example - in practice, you'd integrate
# the full ReAct and ToT implementations from earlier

print("Hybrid agent concept: Use Plan-and-Execute framework,")
print("but route each step to either ReAct or ToT based on complexity.")
```

---

## 📊 Comparison Summary

### Quick Decision Matrix

```python
def choose_paradigm(task_description: str) -> str:
    """Helper to choose the right paradigm"""
    
    task_lower = task_description.lower()
    
    # Keywords that suggest different paradigms
    if any(word in task_lower for word in ["calculate", "solve", "prove"]):
        return "Chain-of-Thought"
    
    if any(word in task_lower for word in ["creative", "design", "brainstorm", "ideas"]):
        return "Tree of Thoughts"
    
    if any(word in task_lower for word in ["search", "find", "weather", "current"]):
        return "ReAct"
    
    if any(word in task_lower for word in ["plan", "organize", "project", "multi-step"]):
        return "Plan-and-Execute"
    
    # Default to CoT for general questions
    return "Chain-of-Thought"

# Test it
test_tasks = [
    "What is the square root of 144?",
    "Come up with creative names for a coffee shop",
    "What's the current price of Bitcoin?",
    "Plan a wedding for 200 guests"
]

for task in test_tasks:
    paradigm = choose_paradigm(task)
    print(f"Task: {task}")
    print(f"→ Use: {paradigm}\n")
```

---

## 🏭 Production Pattern: Adaptive Reasoning

```python
from typing import TypedDict, Annotated, Literal
from operator import add
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# Adaptive state that can switch strategies
class AdaptiveState(TypedDict):
    query: str
    chosen_paradigm: Literal["cot", "tot", "react", "plan_execute"]
    result: str
    confidence: float

llm = ChatOllama(model="llama3.2", temperature=0.3)

# Classifier prompt
classifier_prompt = ChatPromptTemplate.from_messages([
    ("human", """Analyze this query and choose the best reasoning approach:

Query: {query}

Options:
- COT: For math, logic, step-by-step problems
- TOT: For creative tasks needing multiple approaches  
- REACT: For questions needing external information/tools
- PLAN_EXECUTE: For complex multi-step projects

Respond with ONLY one of: COT, TOT, REACT, PLAN_EXECUTE

Best approach:""")
])

classifier_chain = classifier_prompt | llm

def classify_query(state: AdaptiveState) -> dict:
    """Automatically choose the best paradigm"""
    
    response = classifier_chain.invoke({"query": state["query"]})
    
    paradigm_text = response.content.strip().upper()
    
    # Map to our paradigm names
    paradigm_map = {
        "COT": "cot",
        "TOT": "tot",
        "REACT": "react",
        "PLAN_EXECUTE": "plan_execute"
    }
    
    chosen = "cot"  # Default
    for key, value in paradigm_map.items():
        if key in paradigm_text:
            chosen = value
            break
    
    return {"chosen_paradigm": chosen}

def route_to_paradigm(state: AdaptiveState) -> str:
    """Route to the chosen paradigm"""
    return state["chosen_paradigm"]

# Simplified execution nodes (use full implementations from earlier)
def execute_cot(state: AdaptiveState) -> dict:
    return {"result": f"[CoT result for: {state['query']}]", "confidence": 0.9}

def execute_tot(state: AdaptiveState) -> dict:
    return {"result": f"[ToT result for: {state['query']}]", "confidence": 0.85}

def execute_react(state: AdaptiveState) -> dict:
    return {"result": f"[ReAct result for: {state['query']}]", "confidence": 0.95}

def execute_plan_execute(state: AdaptiveState) -> dict:
    return {"result": f"[Plan-Execute result for: {state['query']}]", "confidence": 0.88}

# Build adaptive graph
workflow = StateGraph(AdaptiveState)

workflow.add_node("classify", classify_query)
workflow.add_node("cot", execute_cot)
workflow.add_node("tot", execute_tot)
workflow.add_node("react", execute_react)
workflow.add_node("plan_execute", execute_plan_execute)

workflow.set_entry_point("classify")

workflow.add_conditional_edges(
    "classify",
    route_to_paradigm,
    {
        "cot": "cot",
        "tot": "tot",
        "react": "react",
        "plan_execute": "plan_execute"
    }
)

# All paths lead to END
for node in ["cot", "tot", "react", "plan_execute"]:
    workflow.add_edge(node, END)

adaptive_agent = workflow.compile()

# Test
def solve_adaptively(query: str):
    """Let the agent choose the best approach"""
    
    result = adaptive_agent.invoke({
        "query": query,
        "chosen_paradigm": "cot",
        "result": "",
        "confidence": 0.0
    })
    
    print(f"\nQuery: {query}")
    print(f"Chosen paradigm: {result['chosen_paradigm'].upper()}")
    print(f"Result: {result['result']}")
    print(f"Confidence: {result['confidence']:.1%}\n")

if __name__ == "__main__":
    queries = [
        "What is 25% of 160?",
        "Design a logo for a tech startup",
        "What's the population of Tokyo?",
        "Plan a 2-week European vacation"
    ]
    
    for q in queries:
        solve_adaptively(q)
```

---

## ⚠️ Common Mistakes

### Mistake 1: Using Wrong Paradigm

```python
# ❌ BAD: Using ToT for simple math
solve_with_tot("What is 2 + 2?")  # Overkill!

# ✅ GOOD: Use CoT for simple problems
solve_with_cot("What is 2 + 2?")
```

### Mistake 2: No Max Steps in Iterative Patterns

```python
# ❌ BAD: ReAct could loop forever
def route(state):
    if state["answer_found"]:  # Might never be true!
        return "end"
    return "think"

# ✅ GOOD: Always have a max limit
def route(state):
    if state["step_count"] >= MAX_STEPS:
        return "end"
    if state["answer_found"]:
        return "end"
    return "think"
```

### Mistake 3: Not Validating Tool Outputs

```python
# ❌ BAD: Blindly trust tool results
def act(state):
    result = tool.invoke(input)
    return {"observation": result}  # What if it failed?

# ✅ GOOD: Validate and handle errors
def act(state):
    try:
        result = tool.invoke(input)
        if not result or len(str(result)) < 5:
            result = "Tool returned insufficient information"
    except Exception as e:
        result = f"Tool error: {str(e)}"
    
    return {"observation": result}
```

---

## 🧠 Key Concepts to Remember

1. **ToT = Multiple approaches** → evaluate → pick best
2. **ReAct = Think → Act → Observe** loop with tools
3. **Plan-and-Execute = Plan first**, then execute all steps
4. **Choose paradigm based on task type** (not randomly)
5. **Combine paradigms** for complex tasks
6. **Always have exit conditions** (max steps, completion checks)
7. **Use proper prompt templates**, not f-strings

---

## 🚀 What's Next?

In **Chapter 6**, we'll explore:
- **Reflection Agent**: Self-critique and refinement
- **Reflexion**: Learning from feedback over time
- **Memory-augmented reflection**
- Building agents that improve themselves

---

## ✅ Chapter 5 Complete!

**You now understand:**
- ✅ Tree of Thoughts for creative exploration
- ✅ ReAct for reasoning + tool use
- ✅ Plan-and-Execute for complex projects
- ✅ When to use which paradigm
- ✅ Combining multiple paradigms
- ✅ Adaptive reasoning (auto-selecting strategy)

**Ready for Chapter 6?** Just say "Continue to Chapter 6" or ask any questions about Chapter 5!