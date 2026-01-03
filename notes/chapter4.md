# Chapter 4: Reasoning Paradigms - Part 1 (Chain-of-Thought & Self-Consistency)

## 🎯 The Problem: LLMs Need Structure to Reason

Without guidance, LLMs often:
- Jump to conclusions without showing work
- Make logical errors in multi-step problems
- Produce inconsistent results
- Struggle with complex reasoning

**Chain-of-Thought (CoT)** and **Self-Consistency** solve this by making reasoning explicit and verifiable.

---

## 🔧 Critical Update: Proper Prompting for Ollama

Before we dive in, let's address the two issues you've identified:

### Issue 1: ❌ Don't Use F-Strings for Prompts

```python
# ❌ BAD: F-strings are fragile and hard to manage
prompt = f"Answer this question: {question}"

# ✅ GOOD: Use LangChain prompt templates
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("human", "Answer this question: {question}")
])
```

### Issue 2: ❌ Llama3.2 is Stricter Than GPT-4

**OpenAI models** are forgiving:
```python
# Works fine with GPT-4
llm.invoke([SystemMessage(content="Complex nested instruction...")])
```

**Llama3.2 requires**:
- Clear, direct instructions
- Proper message role usage
- More explicit output formatting
- Simpler, focused prompts
- HumanMessage for instructions (SystemMessage can be finicky)

---

## 📋 Prompt Template Best Practices for Ollama

### Setup: Using Proper Templates

```python
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_ollama import ChatOllama

# Initialize Ollama
llm = ChatOllama(
    model="llama3.2",
    temperature=0.7,
)

# ✅ Method 1: ChatPromptTemplate (for chat models)
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{user_input}")
])

# ✅ Method 2: PromptTemplate (for simple text)
simple_prompt = PromptTemplate(
    template="Answer this question clearly: {question}\n\nAnswer:",
    input_variables=["question"]
)

# ✅ Method 3: Multi-turn with placeholders
multi_turn = ChatPromptTemplate.from_messages([
    ("system", "You are an expert in {domain}."),
    ("human", "{question}"),
    ("ai", "{previous_response}"),
    ("human", "{followup}")
])
```

### Llama3.2-Specific Prompting Tips

```python
from langchain_core.prompts import ChatPromptTemplate

# ✅ GOOD: Clear, direct, explicit formatting
good_prompt = ChatPromptTemplate.from_messages([
    ("human", """Question: {question}

Please provide a step-by-step answer. Format your response as:

Step 1: [first step]
Step 2: [second step]
...
Final Answer: [your answer]

Begin:""")
])

# ❌ BAD: Vague, complex nested instructions
bad_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an advanced reasoning system..."),
    ("human", "Using your knowledge, please analyze..."),
    ("system", "Remember to consider..."),  # Too many system messages
    ("human", "{question}")
])

# ✅ GOOD: Single human message with all context
better_prompt = ChatPromptTemplate.from_messages([
    ("human", """You are a math tutor helping a student.

Student's question: {question}

Provide a clear explanation with steps. Use simple language.

Your response:""")
])
```

---

## 🧠 Chain-of-Thought (CoT) Explained

**Core Idea:** Make the LLM "show its work" by reasoning step-by-step.

### Without CoT (Direct Answer)

```
Question: "If a train travels 60 mph for 2.5 hours, how far does it go?"
Answer: "150 miles"  ← No reasoning shown, could be wrong
```

### With CoT (Step-by-Step)

```
Question: "If a train travels 60 mph for 2.5 hours, how far does it go?"

Let me solve this step by step:
Step 1: Identify the formula - Distance = Speed × Time
Step 2: Plug in values - Distance = 60 mph × 2.5 hours
Step 3: Calculate - Distance = 150 miles

Final Answer: 150 miles
```

**Why it works:**
1. Forces explicit reasoning
2. Makes errors visible
3. Allows verification of logic
4. Improves accuracy on complex tasks

---

## 💻 Implementing CoT with Proper Templates

### Level 1: Zero-Shot CoT (Minimal Example)

```python
from typing import TypedDict
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# Initialize
llm = ChatOllama(model="llama3.2", temperature=0.3)

# ✅ Proper prompt template for Llama3.2
zero_shot_cot_prompt = ChatPromptTemplate.from_messages([
    ("human", """{question}

Think through this step-by-step:""")
])

# Create chain
chain = zero_shot_cot_prompt | llm

# Test
question = "A bakery makes 12 dozen cookies. They sell 3/4 of them. How many cookies remain?"

response = chain.invoke({"question": question})
print(response.content)
```

**Expected Output:**
```
Let me break this down:

Step 1: Calculate total cookies
- 12 dozen = 12 × 12 = 144 cookies

Step 2: Calculate cookies sold
- 3/4 of 144 = (3/4) × 144 = 108 cookies

Step 3: Calculate remaining
- Remaining = 144 - 108 = 36 cookies

Answer: 36 cookies remain
```

---

### Level 2: Few-Shot CoT (With Examples)

```python
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

# Define examples
examples = [
    {
        "question": "If 5 apples cost $10, how much do 8 apples cost?",
        "reasoning": """Let me solve step-by-step:
Step 1: Find cost per apple = $10 / 5 = $2 per apple
Step 2: Calculate for 8 apples = 8 × $2 = $16
Answer: $16"""
    },
    {
        "question": "A car travels 240 miles in 4 hours. What's its speed?",
        "reasoning": """Let me solve step-by-step:
Step 1: Use formula Speed = Distance / Time
Step 2: Calculate = 240 miles / 4 hours = 60 mph
Answer: 60 mph"""
    }
]

# Create example template
example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{question}"),
    ("ai", "{reasoning}")
])

# Create few-shot template
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

# Final prompt with examples + question
final_prompt = ChatPromptTemplate.from_messages([
    ("human", "Solve these problems step-by-step:"),
    few_shot_prompt,
    ("human", "{question}")
])

# Create chain
chain = final_prompt | llm

# Test
response = chain.invoke({
    "question": "A recipe needs 3 eggs for 12 muffins. How many eggs for 20 muffins?"
})

print(response.content)
```

---

### Level 3: CoT in LangGraph Agent

```python
from typing import TypedDict, Annotated
from operator import add
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# State definition
class CoTState(TypedDict):
    question: str
    reasoning_steps: Annotated[list[str], add]
    final_answer: str
    step_count: Annotated[int, add]

# Initialize
llm = ChatOllama(model="llama3.2", temperature=0.3)

# ✅ Proper prompt templates
generate_step_prompt = ChatPromptTemplate.from_messages([
    ("human", """Question: {question}

Previous steps:
{previous_steps}

What is the next reasoning step? Provide ONLY the next step, nothing else.

Next step:""")
])

verify_step_prompt = ChatPromptTemplate.from_messages([
    ("human", """Question: {question}
Reasoning step: {current_step}

Is this step logically correct? Reply with:
VALID - if correct
INVALID - if incorrect

Response:""")
])

finalize_prompt = ChatPromptTemplate.from_messages([
    ("human", """Question: {question}

Reasoning steps:
{all_steps}

Based on these steps, what is the final answer? Be concise.

Final Answer:""")
])

# Create chains
generate_chain = generate_step_prompt | llm
verify_chain = verify_step_prompt | llm
finalize_chain = finalize_prompt | llm

# Node implementations
def generate_reasoning_step(state: CoTState) -> dict:
    """Generate next reasoning step"""
    
    # Format previous steps
    if state["reasoning_steps"]:
        prev_steps = "\n".join([
            f"Step {i+1}: {step}" 
            for i, step in enumerate(state["reasoning_steps"])
        ])
    else:
        prev_steps = "None yet - this is the first step."
    
    # Generate next step
    response = generate_chain.invoke({
        "question": state["question"],
        "previous_steps": prev_steps
    })
    
    next_step = response.content.strip()
    
    return {
        "reasoning_steps": [next_step],
        "step_count": 1
    }

def verify_reasoning_step(state: CoTState) -> dict:
    """Verify the last reasoning step"""
    
    current_step = state["reasoning_steps"][-1]
    
    response = verify_chain.invoke({
        "question": state["question"],
        "current_step": current_step
    })
    
    # Simple validation (in production, be more robust)
    is_valid = "valid" in response.content.lower()
    
    if not is_valid:
        # Remove invalid step
        return {
            "reasoning_steps": [],  # Clear last step
        }
    
    return {}  # No changes needed

def finalize_answer(state: CoTState) -> dict:
    """Generate final answer from reasoning steps"""
    
    all_steps = "\n".join([
        f"Step {i+1}: {step}" 
        for i, step in enumerate(state["reasoning_steps"])
    ])
    
    response = finalize_chain.invoke({
        "question": state["question"],
        "all_steps": all_steps
    })
    
    return {
        "final_answer": response.content.strip()
    }

# Routing logic
def should_continue_reasoning(state: CoTState) -> str:
    """Decide if we need more reasoning steps"""
    
    # Stop conditions
    if state["step_count"] >= 5:  # Max 5 steps
        return "finalize"
    
    # Check if we have enough steps (simple heuristic)
    if state["step_count"] >= 3:
        last_step = state["reasoning_steps"][-1].lower()
        # If last step mentions "answer" or "result", we're done
        if "answer" in last_step or "result" in last_step or "=" in last_step:
            return "finalize"
    
    return "continue"

# Build graph
workflow = StateGraph(CoTState)

workflow.add_node("generate", generate_reasoning_step)
workflow.add_node("verify", verify_reasoning_step)
workflow.add_node("finalize", finalize_answer)

workflow.set_entry_point("generate")

workflow.add_edge("generate", "verify")

workflow.add_conditional_edges(
    "verify",
    should_continue_reasoning,
    {
        "continue": "generate",  # Generate more steps
        "finalize": "finalize"    # Done reasoning
    }
)

workflow.add_edge("finalize", END)

# Compile
cot_agent = workflow.compile()

# Test function
def solve_with_cot(question: str):
    """Solve a problem using CoT reasoning"""
    
    initial_state = {
        "question": question,
        "reasoning_steps": [],
        "final_answer": "",
        "step_count": 0
    }
    
    print(f"\n{'='*60}")
    print(f"QUESTION: {question}")
    print(f"{'='*60}\n")
    
    result = cot_agent.invoke(initial_state)
    
    print("REASONING STEPS:")
    for i, step in enumerate(result["reasoning_steps"], 1):
        print(f"  Step {i}: {step}")
    
    print(f"\n✅ FINAL ANSWER:")
    print(f"  {result['final_answer']}\n")
    
    return result

# Example usage
if __name__ == "__main__":
    # Test 1: Math problem
    solve_with_cot(
        "A store has 48 books. They sell 1/3 of them in the morning and 1/4 of the remainder in the afternoon. How many books are left?"
    )
    
    # Test 2: Logic problem
    solve_with_cot(
        "If all roses are flowers and some flowers are red, can we conclude that some roses are red?"
    )
```

---

## 🎲 Self-Consistency: Multiple Reasoning Paths

**Problem with single CoT:** One reasoning path might be wrong.

**Solution:** Generate multiple reasoning paths and take the majority vote.

### Concept

```
Question: "What is 15% of 80?"

Path 1: 15% = 0.15, so 0.15 × 80 = 12 ✓
Path 2: 15/100 × 80 = 1200/100 = 12 ✓
Path 3: 10% is 8, 5% is 4, so 15% = 8 + 4 = 12 ✓

Majority Answer: 12 (all agree)
```

### Implementation with LangGraph

```python
from typing import TypedDict, Annotated, Literal
from operator import add
from collections import Counter
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# State definition
class SelfConsistencyState(TypedDict):
    question: str
    reasoning_paths: Annotated[list[dict], add]  # List of {reasoning: str, answer: str}
    num_paths: int
    paths_generated: Annotated[int, add]
    final_answer: str
    confidence: float

# Initialize
llm = ChatOllama(model="llama3.2", temperature=0.8)  # Higher temp for diversity

# Prompt template
reasoning_prompt = ChatPromptTemplate.from_messages([
    ("human", """Question: {question}

Solve this step-by-step. At the end, state your final answer clearly as "Final Answer: [your answer]"

Your reasoning:""")
])

# Create chain
reasoning_chain = reasoning_prompt | llm

# Node: Generate one reasoning path
def generate_path(state: SelfConsistencyState) -> dict:
    """Generate a single reasoning path"""
    
    response = reasoning_chain.invoke({
        "question": state["question"]
    })
    
    reasoning = response.content
    
    # Extract final answer (simple parsing)
    answer = "unknown"
    if "final answer:" in reasoning.lower():
        parts = reasoning.lower().split("final answer:")
        if len(parts) > 1:
            # Get text after "final answer:" and clean it
            answer = parts[1].strip().split('\n')[0].strip()
    
    return {
        "reasoning_paths": [{
            "reasoning": reasoning,
            "answer": answer
        }],
        "paths_generated": 1
    }

# Node: Aggregate answers
def aggregate_answers(state: SelfConsistencyState) -> dict:
    """Find consensus answer from multiple paths"""
    
    # Extract all answers
    answers = [path["answer"] for path in state["reasoning_paths"]]
    
    # Count occurrences
    answer_counts = Counter(answers)
    
    # Get most common answer
    if answer_counts:
        most_common = answer_counts.most_common(1)[0]
        final_answer = most_common[0]
        votes = most_common[1]
        confidence = votes / len(answers)
    else:
        final_answer = "Unable to determine"
        confidence = 0.0
    
    return {
        "final_answer": final_answer,
        "confidence": confidence
    }

# Routing
def should_generate_more(state: SelfConsistencyState) -> str:
    """Decide if we need more reasoning paths"""
    
    if state["paths_generated"] >= state["num_paths"]:
        return "aggregate"
    
    return "generate"

# Build graph
workflow = StateGraph(SelfConsistencyState)

workflow.add_node("generate", generate_path)
workflow.add_node("aggregate", aggregate_answers)

workflow.set_entry_point("generate")

workflow.add_conditional_edges(
    "generate",
    should_generate_more,
    {
        "generate": "generate",  # Generate another path
        "aggregate": "aggregate"  # Done, aggregate results
    }
)

workflow.add_edge("aggregate", END)

# Compile
self_consistency_agent = workflow.compile()

# Test function
def solve_with_self_consistency(question: str, num_paths: int = 3):
    """Solve with multiple reasoning paths"""
    
    initial_state = {
        "question": question,
        "reasoning_paths": [],
        "num_paths": num_paths,
        "paths_generated": 0,
        "final_answer": "",
        "confidence": 0.0
    }
    
    print(f"\n{'='*60}")
    print(f"QUESTION: {question}")
    print(f"{'='*60}\n")
    print(f"Generating {num_paths} reasoning paths...\n")
    
    result = self_consistency_agent.invoke(initial_state)
    
    # Display all paths
    for i, path in enumerate(result["reasoning_paths"], 1):
        print(f"--- PATH {i} ---")
        print(path["reasoning"][:200] + "...")  # Truncate for brevity
        print(f"Answer: {path['answer']}\n")
    
    print(f"{'='*60}")
    print(f"✅ CONSENSUS ANSWER: {result['final_answer']}")
    print(f"   Confidence: {result['confidence']:.1%}")
    print(f"{'='*60}\n")
    
    return result

# Example usage
if __name__ == "__main__":
    solve_with_self_consistency(
        "A farmer has 17 sheep. All but 9 die. How many sheep are left?",
        num_paths=5
    )
```

---

## 🎯 When to Use CoT vs Self-Consistency

| Scenario | Use CoT | Use Self-Consistency |
|----------|---------|---------------------|
| **Math problems** | ✅ Single path often enough | ✅ For critical calculations |
| **Logic puzzles** | ✅ Good for showing work | ✅ Better - catches reasoning errors |
| **Factual Q&A** | ❌ Not needed | ❌ Overkill |
| **Creative writing** | ❌ Not appropriate | ❌ Not appropriate |
| **Code debugging** | ✅ Good for analysis | ✅ Multiple approaches help |
| **Medical diagnosis** | ✅ Must show reasoning | ✅✅ Critical - use multiple paths |
| **Real-time apps** | ✅ Faster | ❌ Too slow (multiple LLM calls) |
| **Batch processing** | ✅ Efficient | ✅ More accurate |

---

## 📊 Comparing Approaches with Code

```python
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
import time

llm = ChatOllama(model="llama3.2", temperature=0.3)

test_question = "A box contains 60 apples. 1/5 are rotten and removed. Then 1/3 of the remaining apples are sold. How many apples are left?"

# Method 1: Direct answer (no CoT)
direct_prompt = ChatPromptTemplate.from_messages([
    ("human", "{question}\n\nAnswer:")
])

# Method 2: Zero-shot CoT
cot_prompt = ChatPromptTemplate.from_messages([
    ("human", "{question}\n\nThink step-by-step:\n")
])

# Method 3: Explicit step format
structured_cot_prompt = ChatPromptTemplate.from_messages([
    ("human", """{question}

Solve this using these steps:
Step 1: [Identify what we know]
Step 2: [Calculate first operation]
Step 3: [Calculate second operation]
Final Answer: [result]

Begin:""")
])

def test_method(prompt, question, method_name):
    print(f"\n{'='*60}")
    print(f"METHOD: {method_name}")
    print(f"{'='*60}")
    
    start = time.time()
    chain = prompt | llm
    response = chain.invoke({"question": question})
    elapsed = time.time() - start
    
    print(response.content)
    print(f"\n⏱️ Time: {elapsed:.2f}s\n")
    
    return response.content

# Run comparisons
if __name__ == "__main__":
    print(f"\nTEST QUESTION: {test_question}\n")
    
    result1 = test_method(direct_prompt, test_question, "Direct Answer (No CoT)")
    result2 = test_method(cot_prompt, test_question, "Zero-Shot CoT")
    result3 = test_method(structured_cot_prompt, test_question, "Structured CoT")
```

---

## 🏭 Production-Ready CoT Pattern

```python
from typing import TypedDict, Annotated, Literal
from operator import add
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Production state
class ProductionCoTState(TypedDict):
    question: str
    reasoning_steps: Annotated[list[str], add]
    final_answer: str
    step_count: Annotated[int, add]
    max_steps: int
    status: Literal["reasoning", "complete", "error"]
    error_message: str | None
    llm_calls: Annotated[int, add]

# Initialize
llm = ChatOllama(model="llama3.2", temperature=0.3)
checkpointer = SqliteSaver.from_conn_string("./cot_checkpoints.db")

# Prompts
step_prompt = ChatPromptTemplate.from_messages([
    ("human", """Question: {question}

Previous steps:
{previous_steps}

What is the next logical step? Provide only one step.

Next step:""")
])

answer_prompt = ChatPromptTemplate.from_messages([
    ("human", """Question: {question}

Reasoning:
{reasoning}

Based on this reasoning, provide the final answer concisely.

Final Answer:""")
])

# Chains
step_chain = step_prompt | llm
answer_chain = answer_prompt | llm

def generate_step_safe(state: ProductionCoTState) -> dict:
    """Generate step with error handling"""
    try:
        logger.info(f"Generating step {state['step_count'] + 1}")
        
        prev_steps = "\n".join([
            f"{i+1}. {step}" 
            for i, step in enumerate(state["reasoning_steps"])
        ]) if state["reasoning_steps"] else "None yet."
        
        response = step_chain.invoke({
            "question": state["question"],
            "previous_steps": prev_steps
        })
        
        step = response.content.strip()
        
        if not step or len(step) < 5:
            raise ValueError("Generated step is too short or empty")
        
        return {
            "reasoning_steps": [step],
            "step_count": 1,
            "llm_calls": 1,
            "status": "reasoning"
        }
    
    except Exception as e:
        logger.error(f"Error generating step: {e}")
        return {
            "status": "error",
            "error_message": str(e)
        }

def finalize_safe(state: ProductionCoTState) -> dict:
    """Finalize answer with error handling"""
    try:
        logger.info("Finalizing answer")
        
        reasoning = "\n".join([
            f"Step {i+1}: {step}" 
            for i, step in enumerate(state["reasoning_steps"])
        ])
        
        response = answer_chain.invoke({
            "question": state["question"],
            "reasoning": reasoning
        })
        
        return {
            "final_answer": response.content.strip(),
            "llm_calls": 1,
            "status": "complete"
        }
    
    except Exception as e:
        logger.error(f"Error finalizing: {e}")
        return {
            "status": "error",
            "error_message": str(e)
        }

def route_cot(state: ProductionCoTState) -> str:
    """Route based on status and progress"""
    
    if state["status"] == "error":
        return "error"
    
    if state["step_count"] >= state["max_steps"]:
        return "finalize"
    
    # Check if reasoning looks complete
    if state["reasoning_steps"]:
        last_step = state["reasoning_steps"][-1].lower()
        completion_indicators = ["therefore", "final", "result", "answer is"]
        if any(indicator in last_step for indicator in completion_indicators):
            return "finalize"
    
    return "continue"

# Build graph
workflow = StateGraph(ProductionCoTState)

workflow.add_node("generate", generate_step_safe)
workflow.add_node("finalize", finalize_safe)

workflow.set_entry_point("generate")

workflow.add_conditional_edges(
    "generate",
    route_cot,
    {
        "continue": "generate",
        "finalize": "finalize",
        "error": END
    }
)

workflow.add_edge("finalize", END)

# Compile with checkpointing
production_cot = workflow.compile(checkpointer=checkpointer)

# API-like interface
def solve_with_production_cot(
    question: str,
    max_steps: int = 5,
    session_id: str = "default"
):
    """Production-ready CoT solver"""
    
    config = {
        "configurable": {
            "thread_id": f"cot-{session_id}"
        }
    }
    
    initial_state = {
        "question": question,
        "reasoning_steps": [],
        "final_answer": "",
        "step_count": 0,
        "max_steps": max_steps,
        "status": "reasoning",
        "error_message": None,
        "llm_calls": 0
    }
    
    try:
        result = production_cot.invoke(initial_state, config=config)
        
        if result["status"] == "error":
            logger.error(f"CoT failed: {result['error_message']}")
            return {
                "success": False,
                "error": result["error_message"]
            }
        
        logger.info(f"CoT completed in {result['llm_calls']} LLM calls")
        
        return {
            "success": True,
            "question": question,
            "reasoning_steps": result["reasoning_steps"],
            "final_answer": result["final_answer"],
            "llm_calls": result["llm_calls"]
        }
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return {
            "success": False,
            "error": str(e)
        }

# Example usage
if __name__ == "__main__":
    result = solve_with_production_cot(
        "A cyclist rides at 15 mph for 2 hours, then at 20 mph for 1.5 hours. What is the total distance?"
    )
    
    if result["success"]:
        print(f"\n✅ Answer: {result['final_answer']}")
        print(f"📊 Stats: {result['llm_calls']} LLM calls")
    else:
        print(f"\n❌ Error: {result['error']}")
```

---

## ⚠️ Common Mistakes

### Mistake 1: Using F-Strings

```python
# ❌ BAD
prompt = f"Solve this: {question}"
llm.invoke(prompt)

# ✅ GOOD
from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages([
    ("human", "Solve this: {question}")
])
chain = prompt | llm
chain.invoke({"question": question})
```

### Mistake 2: Complex System Messages with Llama3.2

```python
# ❌ BAD: Too many system messages
ChatPromptTemplate.from_messages([
    ("system", "You are an expert..."),
    ("system", "Remember to..."),
    ("human", "{question}")
])

# ✅ GOOD: Single human message with all context
ChatPromptTemplate.from_messages([
    ("human", """You are a helpful assistant.

Question: {question}

Provide a clear answer:""")
])
```

### Mistake 3: Not Handling Empty Responses

```python
# ❌ BAD: Assuming response always has content
response = llm.invoke(prompt)
answer = response.content.split("Answer:")[1]  # Can crash!

# ✅ GOOD: Defensive parsing
response = llm.invoke(prompt)
if response.content and "Answer:" in response.content:
    answer = response.content.split("Answer:")[1].strip()
else:
    answer = "Unable to generate answer"
```

---

## 🧠 Key Concepts to Remember

1. **Use ChatPromptTemplate, not f-strings** for all prompts
2. **Llama3.2 needs clearer, simpler prompts** than GPT-4
3. **CoT = explicit step-by-step reasoning**
4. **Self-Consistency = multiple paths + majority vote**
5. **Use HumanMessage for instructions** with Llama3.2
6. **Always handle empty/invalid responses** gracefully
7. **Structured output format helps** (Step 1, Step 2, etc.)

---

## 🚀 What's Next?

In **Chapter 5**, we'll explore:
- **Tree of Thoughts (ToT)** - branching exploration
- **ReAct** - Reasoning + Acting paradigm
- **Plan-and-Execute** patterns
- When to use each reasoning strategy
- Combining multiple paradigms

---

## ✅ Chapter 4 Complete!

**You now understand:**
- ✅ Proper prompt templates (no f-strings!)
- ✅ Llama3.2-specific prompting strategies
- ✅ Chain-of-Thought reasoning (zero-shot and few-shot)
- ✅ Self-Consistency with multiple paths
- ✅ Production patterns with error handling
- ✅ When to use CoT vs Self-Consistency

**Ready for Chapter 5?** Just say "Continue to Chapter 5" or ask any questions about Chapter 4!