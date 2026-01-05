# Agent System Architecture Design Guide

## 🎯 The Problem: Too Many Options!

You're absolutely right - we've covered SO many concepts that it's confusing to know what to actually use. Let me give you a **clear, systematic approach** to designing agent systems.

---

# Part 1: The 6-Step Design Process

## Step 1: Define Your Problem (Requirements Analysis)

**Start here ALWAYS. Answer these questions:**

### 1.1 What is the core task?
```
Examples:
- "Answer customer questions from our docs"
- "Analyze financial reports and generate insights"
- "Write and execute code to solve data problems"
- "Process insurance claims with human approval"
```

### 1.2 What are the inputs and outputs?
```
Input: What does the user provide?
Output: What does the system produce?

Example:
Input: Customer question + conversation history
Output: Answer with citations from docs
```

### 1.3 What's the complexity level?

**Simple (Single Agent):**
- One clear task
- Linear flow
- No branching logic
- Example: "Summarize this document"

**Medium (Structured Multi-Agent):**
- Multiple distinct steps
- Some branching
- 3-7 steps
- Example: "Research topic, write report, review it"

**Complex (Dynamic Multi-Agent):**
- Many interdependent tasks
- Lots of branching
- >7 steps, maybe loops
- Example: "Full software development pipeline"

### 1.4 What are the constraints?

```
Time: Real-time (<1s)? Batch (hours)? Async (minutes)?
Cost: Budget per request? Total budget?
Accuracy: Mission-critical? Best-effort?
Scale: 10 users? 10,000 users?
Data: Sensitive? Public? Size?
```

---

## Step 2: Choose Your Reasoning Paradigm (Chapter 7-8)

**Use this decision tree:**

```
┌─ Does the task need step-by-step thinking?
│  YES → Use Chain-of-Thought (CoT)
│         Examples: Math, logic, analysis
│         Pattern: Prompt with "Let's think step by step"
│
├─ Does the task need exploring multiple paths?
│  YES → Use Tree-of-Thoughts (ToT)
│         Examples: Creative writing, strategic planning
│         Pattern: Generate options → Evaluate → Select best
│
├─ Does the task need external actions/tools?
│  YES → Use ReAct
│         Examples: Web search, API calls, data retrieval
│         Pattern: Thought → Action → Observation → Loop
│
├─ Does output need iterative improvement?
│  YES → Use Reflection
│         Examples: Code generation, content creation
│         Pattern: Generate → Critique → Refine → Loop
│
└─ Is it a simple, direct task?
   YES → Use Direct Prompting
          Examples: Classification, simple Q&A
          Pattern: Single LLM call
```

**DECISION MATRIX:**

| Your Need | Use This | When NOT to Use |
|-----------|----------|-----------------|
| **Breaking down complex problems** | Chain-of-Thought | Task is simple/direct |
| **Exploring multiple solutions** | Tree-of-Thoughts | Only one right answer |
| **Needing to use tools/search** | ReAct | No external data needed |
| **Improving output quality** | Reflection | First draft is good enough |
| **Simple input→output** | Direct Prompting | Complex reasoning needed |

**EXAMPLE MAPPING:**

```
Task: "Analyze competitor websites and create strategy"

Breaking it down:
1. Need to search web → ReAct (for web search)
2. Need structured thinking → CoT (for analysis)
3. Need quality refinement → Reflection (for strategy)

Architecture: ReAct agent → CoT analysis → Reflection loop
```

---

## Step 3: Choose Your Architectural Pattern (Chapter 11)

**Use this decision tree:**

```
┌─ How many distinct tasks are there?
│
├─ ONE task
│  └─→ SINGLE AGENT
│      Pattern: Simple workflow with nodes
│      Example: "Summarize document"
│
├─ 2-3 tasks, sequential
│  └─→ ROUTER PATTERN
│      Pattern: Classify → Route to specialist → Done
│      Example: "Customer support routing"
│
├─ 3-7 tasks, need coordination
│  └─→ SUPERVISOR PATTERN
│      Pattern: Manager coordinates workers
│      Example: "Research assistant with specialists"
│
├─ >7 tasks, hierarchical
│  └─→ HIERARCHICAL PATTERN
│      Pattern: Multi-level management
│      Example: "Enterprise workflow automation"
│
└─ Modular components to reuse
   └─→ SUBGRAPH PATTERN
       Pattern: Composable workflows
       Example: "Shared validation/processing modules"
```

**PATTERN SELECTION MATRIX:**

| Pattern | Use When | Example | Chapter |
|---------|----------|---------|---------|
| **Single Agent** | 1 task, simple flow | Summarization | Ch 3-6 |
| **Router** | Classify then route | Customer support | Ch 11 |
| **Supervisor** | Coordinate specialists | Research team | Ch 11 |
| **Hierarchical** | Complex org structure | Enterprise automation | Ch 11 |
| **Subgraphs** | Reusable components | Validation modules | Ch 16 |
| **Pipeline** | Sequential processing | ETL pipeline | Ch 11 |

---

## Step 4: Choose Communication/Coordination (Chapter 12)

**Only needed if you chose multi-agent in Step 3.**

```
┌─ How should agents communicate?
│
├─ One agent immediately calls another?
│  └─→ DIRECT INVOCATION
│      When: Simple, synchronous
│      Example: Agent A finishes → calls Agent B
│
├─ Agents in same process, need shared data?
│  └─→ SHARED STATE
│      When: All agents need access to same data
│      Example: All agents update shared document
│
├─ Agents independent, different processes?
│  └─→ MESSAGE PASSING
│      When: Distributed, async, decoupled
│      Example: Microservices architecture
│
└─ Agents should operate independently?
   └─→ ISOLATED STATE
       When: Privacy, testing, independence
       Example: Each agent has own workspace
```

**DECISION MATRIX:**

| Communication | Use When | Avoid When |
|---------------|----------|------------|
| **Direct Invocation** | Simple, same process, sequential | Distributed, need async |
| **Shared State** | Need common data access | Want isolation, testing |
| **Message Passing** | Distributed, async, decoupled | Overhead not justified |
| **Isolated State** | Independence, privacy, testing | Need shared access |

---

## Step 5: Choose Cooperation/Competition Pattern (Ch 13-15)

**Only if agents need to work together or compete.**

```
┌─ What relationship do agents have?
│
├─ Working TOGETHER toward shared goal?
│  │
│  ├─ Need multiple perspectives?
│  │  └─→ TEAM COLLABORATION (Ch 13)
│  │      Example: Different experts review document
│  │
│  ├─ One manages others?
│  │  └─→ SUPERVISOR-WORKER (Ch 13)
│  │      Example: Manager assigns tasks to workers
│  │
│  ├─ Need agreement on decision?
│  │  └─→ CONSENSUS BUILDING (Ch 13)
│  │      Example: Committee votes on action
│  │
│  └─ Need to explore both sides?
│     └─→ DEBATE/ROLEPLAY (Ch 14)
│         Example: Pro/con analysis
│
└─ Working AGAINST each other?
   │
   ├─ Testing system robustness?
   │  └─→ RED TEAM / BLUE TEAM (Ch 15)
   │      Example: Security testing
   │
   ├─ Competing for resources?
   │  └─→ AUCTION/BIDDING (Ch 15)
   │      Example: Task allocation
   │
   └─ Strategic competition?
      └─→ GAME THEORY (Ch 15)
          Example: Nash equilibrium optimization
```

---

## Step 6: Add Cross-Cutting Concerns

**These apply to ANY architecture:**

### 6.1 Do you need RAG? (Chapter 9)
```
YES if:
- Need to reference specific documents
- Information changes frequently
- Can't fit all data in context
- Need citations/sources

NO if:
- LLM knowledge is sufficient
- Static information
- Small dataset
```

### 6.2 Do you need Human-in-Loop? (Chapter 17)
```
YES if:
- High-stakes decisions
- Regulatory compliance
- Quality control required
- Building trust with users

NO if:
- Fully autonomous OK
- Low-stakes operations
- High-frequency tasks
```

### 6.3 Do you need Memory? (Chapter 10)
```
YES if:
- Multi-turn conversations
- Personalization needed
- Learning from interactions
- Long-running sessions

NO if:
- Stateless operations
- Each request independent
- Privacy concerns
```

### 6.4 Production Features? (Chapter 18)
```
ALWAYS include:
- Error handling (retry, circuit breaker)
- Logging and monitoring
- Caching (for LLM responses)
- Testing strategy

INCLUDE if appropriate:
- Authentication/authorization
- Rate limiting
- Async processing
- Horizontal scaling
```

---

# Part 2: Complete Decision Framework

## The Master Decision Tree

```
START: I have an idea for an agent system
    |
    ↓
[1] DEFINE PROBLEM
    - What's the task?
    - What's the complexity? (Simple/Medium/Complex)
    - What are constraints? (Time/Cost/Accuracy/Scale)
    |
    ↓
[2] CHOOSE REASONING (if task needs reasoning)
    - Step-by-step logic? → CoT
    - Multiple paths? → ToT  
    - Need tools? → ReAct
    - Iterate to improve? → Reflection
    |
    ↓
[3] CHOOSE ARCHITECTURE
    - Single task? → Single Agent
    - 2-3 tasks? → Router
    - 3-7 tasks, need coordination? → Supervisor
    - >7 tasks? → Hierarchical
    - Reusable modules? → Subgraphs
    |
    ↓
[4] ADD COMMUNICATION (if multi-agent)
    - Same process, sequential? → Direct Invocation
    - Same process, shared data? → Shared State
    - Distributed, async? → Message Passing
    - Independent? → Isolated State
    |
    ↓
[5] ADD COOPERATION (if agents collaborate/compete)
    - Multiple perspectives? → Team Collaboration
    - One manages? → Supervisor-Worker
    - Need agreement? → Consensus
    - Test both sides? → Debate
    - Security testing? → Red/Blue Team
    - Resource competition? → Auction
    |
    ↓
[6] ADD COMPONENTS
    - Need documents? → Add RAG
    - Need user approval? → Add HITL
    - Need conversation history? → Add Memory
    - Going to production? → Add Production features
    |
    ↓
[7] VALIDATE DESIGN
    - Can it handle edge cases?
    - Can it scale?
    - Can it fail gracefully?
    - Can humans understand/debug it?
    |
    ↓
END: Architecture Complete → Implement
```

---

# Part 3: Architecture Templates

## Template 1: Simple Q&A System

**When:** Answer questions from documents

**Architecture:**
```
┌────────────────────────────────────────┐
│  USER QUESTION                          │
└──────────────┬─────────────────────────┘
               ↓
        ┌──────────────┐
        │  RAG System  │
        │ - Retrieve   │
        │ - Generate   │
        └──────┬───────┘
               ↓
        ┌──────────────┐
        │   ANSWER     │
        └──────────────┘
```

**Components:**
- Reasoning: Direct (no CoT needed)
- Architecture: Single Agent
- RAG: Yes
- Memory: Optional (for conversation)
- HITL: No

**Code Pattern:**
```python
def qa_agent(state):
    # Retrieve relevant docs
    docs = retriever.get_relevant_documents(state["question"])
    
    # Generate answer
    answer = llm.invoke({
        "question": state["question"],
        "context": docs
    })
    
    return {"answer": answer, "sources": docs}
```

---

## Template 2: Research Assistant

**When:** Multi-step research with tool use

**Architecture:**
```
┌────────────────────────────────────────┐
│  RESEARCH QUERY                         │
└──────────────┬─────────────────────────┘
               ↓
        ┌──────────────┐
        │   PLANNER    │ (CoT: Break down)
        └──────┬───────┘
               ↓
        ┌──────────────┐
        │ ReAct Agent  │ (Search, gather info)
        │ - Thought    │
        │ - Action     │
        │ - Observation│
        └──────┬───────┘
               ↓
        ┌──────────────┐
        │  SYNTHESIZER │ (Combine findings)
        └──────┬───────┘
               ↓
        ┌──────────────┐
        │  REVIEWER    │ (Reflection: Improve)
        └──────┬───────┘
               ↓
        ┌──────────────┐
        │ FINAL REPORT │
        └──────────────┘
```

**Components:**
- Reasoning: CoT (planning) + ReAct (tool use) + Reflection (quality)
- Architecture: Single Agent with multiple nodes
- RAG: Optional
- Memory: Yes (track research progress)
- HITL: Optional (review before final)

---

## Template 3: Customer Support System

**When:** Route to specialists, need human approval

**Architecture:**
```
┌────────────────────────────────────────┐
│  CUSTOMER REQUEST                       │
└──────────────┬─────────────────────────┘
               ↓
        ┌──────────────┐
        │  CLASSIFIER  │ (Route by category)
        └──────┬───────┘
               ↓
       ┌───────┴────────┐
       ↓                ↓
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│  Technical  │  │   Billing   │  │   General   │
│  Specialist │  │  Specialist │  │  Specialist │
└──────┬──────┘  └──────┬──────┘  └──────┬──────┘
       └────────────────┴────────────────┘
                       ↓
                ┌──────────────┐
                │   APPROVAL   │ (HITL: High stakes)
                │  (if needed) │
                └──────┬───────┘
                       ↓
                ┌──────────────┐
                │   RESPONSE   │
                └──────────────┘
```

**Components:**
- Reasoning: Direct (classification) + CoT (for complex issues)
- Architecture: Router Pattern
- RAG: Yes (knowledge base)
- Memory: Yes (conversation history)
- HITL: Yes (for refunds, account changes)

---

## Template 4: Content Creation Pipeline

**When:** Multiple specialists collaborate

**Architecture:**
```
┌────────────────────────────────────────┐
│  CONTENT REQUEST                        │
└──────────────┬─────────────────────────┘
               ↓
        ┌──────────────┐
        │  SUPERVISOR  │ (Coordinates team)
        └──────┬───────┘
               ↓
       ┌───────┴────────┬────────────┐
       ↓                ↓            ↓
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│ RESEARCHER  │  │   WRITER    │  │   EDITOR    │
│ (ReAct)     │  │ (Reflection)│  │  (Review)   │
└──────┬──────┘  └──────┬──────┘  └──────┬──────┘
       └────────────────┴────────────────┘
                       ↓
                ┌──────────────┐
                │   APPROVAL   │ (HITL: Before publish)
                └──────┬───────┘
                       ↓
                ┌──────────────┐
                │   PUBLISH    │
                └──────────────┘
```

**Components:**
- Reasoning: ReAct (research) + Reflection (writing) + CoT (editing)
- Architecture: Supervisor Pattern
- Communication: Shared State (collaborative document)
- Cooperation: Team Collaboration
- HITL: Yes (final approval)

---

## Template 5: Autonomous Agent with Safety

**When:** Agent takes actions autonomously but needs oversight

**Architecture:**
```
┌────────────────────────────────────────┐
│  USER GOAL                              │
└──────────────┬─────────────────────────┘
               ↓
        ┌──────────────┐
        │   PLANNER    │ (ToT: Explore options)
        └──────┬───────┘
               ↓
        ┌──────────────┐
        │ ReAct Agent  │ (Execute plan)
        │  with tools  │
        └──────┬───────┘
               ↓
        ┌──────────────┐
        │ SAFETY CHECK │ (Validate before action)
        └──────┬───────┘
               ↓
     [Critical?] ──YES──> ┌──────────────┐
          ↓                │    HITL      │ (Human approval)
          NO               │  APPROVAL    │
          ↓                └──────┬───────┘
          ↓                       ↓
          └───────────────────────┘
                       ↓
                ┌──────────────┐
                │   EXECUTE    │
                └──────┬───────┘
                       ↓
                ┌──────────────┐
                │  REFLECT &   │ (Learn from outcome)
                │   IMPROVE    │
                └──────────────┘
```

**Components:**
- Reasoning: ToT (planning) + ReAct (execution) + Reflection (learning)
- Architecture: Single Agent with decision points
- HITL: Yes (for critical actions)
- Memory: Yes (learn from experience)

---

# Part 4: The Complete Design Checklist

## Pre-Design Questionnaire

**Print this out for EVERY new project:**

```
PROJECT: _________________________________
DATE: ___________________________________

SECTION 1: PROBLEM DEFINITION
□ What is the main task/goal?
  _____________________________________________

□ What are the inputs?
  _____________________________________________

□ What are the outputs?
  _____________________________________________

□ Complexity level:
  □ Simple (1 task, linear)
  □ Medium (3-7 steps, some branching)
  □ Complex (>7 steps, lots of branching)

□ Constraints:
  Time: _________ Cost: _________ Accuracy: _________
  Scale: _________ Data sensitivity: _________

SECTION 2: REASONING PARADIGM
□ Does task need step-by-step thinking?
  □ Yes → Use Chain-of-Thought
  □ No

□ Does task need exploring multiple paths?
  □ Yes → Use Tree-of-Thoughts
  □ No

□ Does task need external tools/actions?
  □ Yes → Use ReAct
  □ No

□ Does output need iterative improvement?
  □ Yes → Use Reflection
  □ No

Selected: _____________________________________

SECTION 3: ARCHITECTURAL PATTERN
□ How many distinct tasks? _____

□ Selected pattern:
  □ Single Agent (1 task)
  □ Router (2-3 tasks, classify and route)
  □ Supervisor (3-7 tasks, coordination)
  □ Hierarchical (>7 tasks, complex)
  □ Subgraphs (reusable modules)

SECTION 4: MULTI-AGENT (if applicable)
□ Communication pattern:
  □ Direct Invocation (simple, sequential)
  □ Shared State (common data access)
  □ Message Passing (distributed, async)
  □ Isolated State (independence)

□ Cooperation pattern:
  □ Team Collaboration (multiple perspectives)
  □ Supervisor-Worker (coordination)
  □ Consensus Building (agreement)
  □ Debate/Roleplay (explore both sides)
  □ Red/Blue Team (testing)
  □ None (agents don't interact)

SECTION 5: ADDITIONAL COMPONENTS
□ RAG (Retrieval Augmented Generation)?
  □ Yes - Need to reference documents
  □ No

□ Human-in-the-Loop?
  □ Yes - For: ___________________________
  □ No

□ Memory/Conversation History?
  □ Yes - Type: ___________________________
  □ No

□ Production Requirements:
  □ Error handling (retry, circuit breaker)
  □ Monitoring and logging
  □ Caching
  □ Authentication/authorization
  □ Rate limiting
  □ Async processing
  □ Horizontal scaling

SECTION 6: VALIDATION
□ Can the design handle edge cases?
  □ Yes - Examples tested: _______________
  □ No - Need to refine

□ Can it scale to required load?
  □ Yes - Load tested: ___________________
  □ No - Need to optimize

□ Can it fail gracefully?
  □ Yes - Error handling in place
  □ No - Need to add

□ Is it debuggable?
  □ Yes - Logging/tracing implemented
  □ No - Need to add

SECTION 7: FINAL ARCHITECTURE
Draw your architecture here:




Components used:
1. _________________________________________
2. _________________________________________
3. _________________________________________
4. _________________________________________
5. _________________________________________
```

---

# Part 5: Real-World Examples

## Example 1: Code Generation Assistant

**Problem:** "Help users write code by understanding requirements and generating tested code"

**Design Process:**

1. **Problem Definition:**
   - Task: Generate code from natural language
   - Complexity: Medium (multiple steps)
   - Constraints: Must be correct, runnable

2. **Reasoning:** 
   - Reflection (iterate to improve code)
   - CoT (think through logic)

3. **Architecture:**
   - Single Agent with reflection loop

4. **Components:**
   - No RAG (LLM has coding knowledge)
   - Optional HITL (user reviews before running)
   - No memory (stateless)

**Final Architecture:**
```python
def code_agent(state):
    # Generate initial code (with CoT)
    code = generate_with_cot(state["requirements"])
    
    # Reflection loop
    for i in range(3):
        # Review code
        review = critique_code(code)
        
        if review["quality"] > 0.9:
            break
        
        # Improve based on critique
        code = improve_code(code, review)
    
    return {"code": code, "review": review}
```

---

## Example 2: Financial Report Analyzer

**Problem:** "Analyze quarterly reports and generate investment insights"

**Design Process:**

1. **Problem Definition:**
   - Task: Multi-step analysis → insights
   - Complexity: Medium
   - Constraints: High accuracy required

2. **Reasoning:**
   - ReAct (search for company data)
   - CoT (structured analysis)
   - Reflection (validate insights)

3. **Architecture:**
   - Supervisor with specialist workers

4. **Components:**
   - RAG: Yes (historical reports)
   - HITL: Yes (human approves before sharing)
   - Memory: Optional (track companies analyzed)

**Final Architecture:**
```
Supervisor
  ├─> Data Collector (ReAct: gather reports)
  ├─> Financial Analyst (CoT: analyze numbers)
  ├─> Market Researcher (ReAct: search news)
  ├─> Insight Generator (Reflection: synthesize)
  └─> Human Approval (HITL)
```

---

## Example 3: Customer Support Chatbot

**Problem:** "Answer customer questions, escalate complex issues"

**Design Process:**

1. **Problem Definition:**
   - Task: Route → Answer OR Escalate
   - Complexity: Simple-Medium
   - Constraints: Real-time (<2s)

2. **Reasoning:**
   - Direct (classification)
   - CoT (for complex explanations)

3. **Architecture:**
   - Router Pattern

4. **Components:**
   - RAG: Yes (knowledge base)
   - HITL: Yes (escalation to human)
   - Memory: Yes (conversation history)

**Final Architecture:**
```python
def support_bot(state):
    # Classify intent
    intent = classify(state["message"])
    
    if intent == "simple_faq":
        return rag_answer(state)
    
    elif intent == "complex_technical":
        return cot_explain(state)
    
    elif intent == "account_issue":
        return escalate_to_human(state)
```

---

# Part 6: Common Mistakes to Avoid

## ❌ Over-Engineering

**Mistake:**
```
Problem: "Summarize a document"
Solution: Hierarchical multi-agent system with debate
```

**Correct:**
```
Problem: "Summarize a document"
Solution: Single agent, direct prompting
```

**Rule:** Start simple, add complexity only when needed.

---

## ❌ Using Wrong Reasoning Paradigm

**Mistake:**
```
Problem: "Search web and answer question"
Solution: Chain-of-Thought only
Issue: Can't actually search web!
```

**Correct:**
```
Problem: "Search web and answer question"
Solution: ReAct (thought + action + observation)
```

**Rule:** Match paradigm to task requirements.

---

## ❌ Ignoring Production Requirements

**Mistake:**
```
Design: Perfect architecture
Reality: No error handling, no monitoring, fails in production
```

**Correct:**
```
Design: Good architecture + error handling + monitoring + caching
Reality: Works reliably in production
```

**Rule:** Production concerns are part of the design, not an afterthought.

---

## ❌ Not Considering Scale

**Mistake:**
```
Design: Everything in memory, stateful
Reality: Can't scale horizontally
```

**Correct:**
```
Design: Externalized state, stateless workers
Reality: Scales easily
```

**Rule:** Think about scale from day 1, even if starting small.

---

# Part 7: Quick Reference Card

**Print this and keep it next to you:**

```
╔════════════════════════════════════════════════╗
║   AGENT SYSTEM DESIGN QUICK REFERENCE          ║
╚════════════════════════════════════════════════╝

STEP 1: DEFINE PROBLEM
→ What's the task?
→ Simple/Medium/Complex?
→ Constraints?

STEP 2: REASONING (Ch 7-8)
→ Step-by-step? → CoT
→ Multiple paths? → ToT
→ Need tools? → ReAct
→ Need improvement? → Reflection

STEP 3: ARCHITECTURE (Ch 11)
→ 1 task → Single Agent
→ 2-3 tasks → Router
→ 3-7 tasks → Supervisor
→ >7 tasks → Hierarchical

STEP 4: COMMUNICATION (Ch 12) [if multi-agent]
→ Sequential, same process → Direct
→ Shared data → Shared State
→ Distributed → Message Passing

STEP 5: COOPERATION (Ch 13-15) [if applicable]
→ Multiple perspectives → Team
→ Coordination → Supervisor-Worker
→ Agreement → Consensus
→ Both sides → Debate

STEP 6: ADD COMPONENTS
→ Documents? → RAG (Ch 9)
→ Conversations? → Memory (Ch 10)
→ High stakes? → HITL (Ch 17)
→ Production? → Error handling, monitoring (Ch 18)

STEP 7: VALIDATE
→ Edge cases?
→ Scalable?
→ Fail gracefully?
→ Debuggable?

═══════════════════════════════════════════════════
REMEMBER: Start simple, iterate!
═══════════════════════════════════════════════════
```

---

# Summary: Your Go-To Process

**Every time you start a new project:**

1. **Fill out the design checklist** (Part 4)
2. **Follow the 6-step process** (Part 1)
3. **Check if your design matches a template** (Part 3)
4. **Validate against common mistakes** (Part 6)
5. **Keep the quick reference card handy** (Part 7)

**The golden rule:** 

> "Use the simplest architecture that solves the problem. Add complexity only when simpler solutions fail."

Start with a single agent. Add multi-agent only if needed. Add complex reasoning only if needed. Add HITL only if needed. Build up, don't build down.

---

Does this give you the clear reference you need? Would you like me to walk through designing a specific system you have in mind using this framework?