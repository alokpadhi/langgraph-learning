# Chapter 13: Cooperative Multi-Agent Patterns

## 🎯 Introduction: What Are Cooperative Multi-Agent Systems?

**Cooperative multi-agent systems** are where multiple agents work **together toward a shared goal**. Unlike competitive systems (where agents compete), cooperative agents:
- Share information
- Coordinate their actions
- Combine their strengths
- Help each other succeed

Think of a **sports team** - everyone has different roles, but they're all trying to win the same game.

---

## 🤝 Part 1: Team Collaboration Patterns

### Theory: Team Collaboration

#### What Is Team Collaboration?

**Team collaboration** is when multiple agents work on the same task simultaneously or sequentially, each contributing their unique capabilities, with shared visibility into the work.

#### Key Characteristics

**1. Shared Goal:**
- All agents working toward same objective
- Success is collective, not individual

**2. Visibility:**
- Agents can see what others are doing
- Transparency in contributions

**3. Complementary Skills:**
- Each agent brings different capabilities
- Together they're more powerful than individually

**4. Communication:**
- Agents share progress and findings
- Coordinate to avoid duplication

#### Team Collaboration Models

**Model 1: Parallel Collaboration**
```
Task → Agent A ┐
Task → Agent B ├→ Combine → Result
Task → Agent C ┘

All work simultaneously on different aspects
```

**Model 2: Sequential Collaboration**
```
Task → Agent A → Agent B → Agent C → Result

Each builds on previous agent's work
```

**Model 3: Iterative Collaboration**
```
Task → Agent A → Agent B → Review
         ↑                    │
         └────────────────────┘
         
Agents refine each other's work in rounds
```

#### When to Use Team Collaboration

✅ **Use Team Collaboration When:**
- Task benefits from multiple perspectives
- Different skills/expertise needed
- Parallel work can speed completion
- Quality improves with peer review

❌ **Don't Use When:**
- Single skill is sufficient
- Task is too simple
- Coordination overhead outweighs benefits

---

### Implementation: Team Collaboration System

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

# ==================== TEAM COLLABORATION PATTERN ====================

# Individual team member state
class TeamMemberState(TypedDict):
    """State for individual team member"""
    messages: Annotated[Sequence[BaseMessage], add]
    task: str
    team_context: str  # What other members have done
    contribution: str

# Team coordinator state
class TeamState(TypedDict):
    """State for team collaboration"""
    messages: Annotated[Sequence[BaseMessage], add]
    task: str
    team_members: List[str]
    contributions: Dict[str, str]
    collaboration_mode: str  # "parallel" or "sequential"
    final_result: str

llm = ChatOllama(model="llama3.2", temperature=0.3)

# ==================== TEAM MEMBER AGENTS ====================

# Research Team Member
def researcher_contribute(state: TeamMemberState) -> dict:
    """Researcher contributes findings"""
    logger.info("🔬 Researcher: Contributing")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a researcher on a collaborative team. Focus on gathering facts and data."),
        ("human", """Task: {task}

Team context (what others have done):
{context}

Provide your research contribution:""")
    ])
    
    chain = prompt | llm
    response = chain.invoke({
        "task": state["task"],
        "context": state.get("team_context", "You're the first contributor")
    })
    
    return {
        "contribution": response.content,
        "messages": [AIMessage(content=f"[Researcher] {response.content}")]
    }

researcher_workflow = StateGraph(TeamMemberState)
researcher_workflow.add_node("contribute", researcher_contribute)
researcher_workflow.set_entry_point("contribute")
researcher_workflow.add_edge("contribute", END)
researcher_agent = researcher_workflow.compile()

# Analyst Team Member
def analyst_contribute(state: TeamMemberState) -> dict:
    """Analyst contributes insights"""
    logger.info("📊 Analyst: Contributing")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an analyst on a collaborative team. Focus on interpreting data and finding patterns."),
        ("human", """Task: {task}

Team context (what others have done):
{context}

Provide your analytical contribution:""")
    ])
    
    chain = prompt | llm
    response = chain.invoke({
        "task": state["task"],
        "context": state.get("team_context", "You're the first contributor")
    })
    
    return {
        "contribution": response.content,
        "messages": [AIMessage(content=f"[Analyst] {response.content}")]
    }

analyst_workflow = StateGraph(TeamMemberState)
analyst_workflow.add_node("contribute", analyst_contribute)
analyst_workflow.set_entry_point("contribute")
analyst_workflow.add_edge("contribute", END)
analyst_agent = analyst_workflow.compile()

# Writer Team Member
def writer_contribute(state: TeamMemberState) -> dict:
    """Writer contributes narrative"""
    logger.info("✍️ Writer: Contributing")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a writer on a collaborative team. Focus on clear communication and storytelling."),
        ("human", """Task: {task}

Team context (what others have done):
{context}

Provide your writing contribution:""")
    ])
    
    chain = prompt | llm
    response = chain.invoke({
        "task": state["task"],
        "context": state.get("team_context", "You're the first contributor")
    })
    
    return {
        "contribution": response.content,
        "messages": [AIMessage(content=f"[Writer] {response.content}")]
    }

writer_workflow = StateGraph(TeamMemberState)
writer_workflow.add_node("contribute", writer_contribute)
writer_workflow.set_entry_point("contribute")
writer_workflow.add_edge("contribute", END)
writer_agent = writer_workflow.compile()

# ==================== TEAM COORDINATOR ====================

def invoke_team_parallel(state: TeamState) -> dict:
    """Invoke all team members in parallel"""
    logger.info("🚀 Team Coordinator: Parallel collaboration mode")
    
    task = state["task"]
    contributions = {}
    
    # All team members work simultaneously with no prior context
    members = {
        "researcher": researcher_agent,
        "analyst": analyst_agent,
        "writer": writer_agent
    }
    
    for member_name, member_agent in members.items():
        logger.info(f"   Invoking {member_name}...")
        
        result = member_agent.invoke({
            "messages": [],
            "task": task,
            "team_context": "Working in parallel with team",
            "contribution": ""
        })
        
        contributions[member_name] = result["contribution"]
    
    logger.info(f"✅ Collected {len(contributions)} contributions")
    
    return {"contributions": contributions}

def invoke_team_sequential(state: TeamState) -> dict:
    """Invoke team members sequentially"""
    logger.info("🔄 Team Coordinator: Sequential collaboration mode")
    
    task = state["task"]
    contributions = {}
    team_context = "You're the first contributor"
    
    # Sequential: Each member sees previous contributions
    sequence = [
        ("researcher", researcher_agent),
        ("analyst", analyst_agent),
        ("writer", writer_agent)
    ]
    
    for member_name, member_agent in sequence:
        logger.info(f"   Invoking {member_name}...")
        
        result = member_agent.invoke({
            "messages": [],
            "task": task,
            "team_context": team_context,
            "contribution": ""
        })
        
        contributions[member_name] = result["contribution"]
        
        # Update context for next member
        team_context = "\n\n".join([
            f"{name}: {contrib[:150]}..."
            for name, contrib in contributions.items()
        ])
    
    logger.info(f"✅ Sequential collaboration complete")
    
    return {"contributions": contributions}

def synthesize_team_work(state: TeamState) -> dict:
    """Synthesize all team contributions"""
    logger.info("🔗 Team Coordinator: Synthesizing contributions")
    
    contributions = state["contributions"]
    
    # Format contributions
    contrib_text = "\n\n".join([
        f"=== {member.upper()} ===\n{contrib}"
        for member, contrib in contributions.items()
    ])
    
    # Synthesize
    synthesis_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are synthesizing a team's collaborative work. Combine their contributions into a cohesive result."),
        ("human", """Original Task: {task}

Team Contributions:
{contributions}

Synthesize into a unified result:""")
    ])
    
    chain = synthesis_prompt | llm
    response = chain.invoke({
        "task": state["task"],
        "contributions": contrib_text
    })
    
    final = f"""TEAM COLLABORATION RESULT
Mode: {state['collaboration_mode']}
Team: {', '.join(state['contributions'].keys())}

{response.content}"""
    
    return {
        "final_result": final,
        "messages": [AIMessage(content=final)]
    }

# Build team coordination graph
def should_use_parallel(state: TeamState) -> str:
    """Decide collaboration mode"""
    return state["collaboration_mode"]

team_workflow = StateGraph(TeamState)
team_workflow.add_node("parallel", invoke_team_parallel)
team_workflow.add_node("sequential", invoke_team_sequential)
team_workflow.add_node("synthesize", synthesize_team_work)

team_workflow.set_entry_point("parallel")  # Default entry

# Route based on mode
team_workflow.add_conditional_edges(
    "parallel",
    should_use_parallel,
    {
        "parallel": "synthesize",
        "sequential": "synthesize"  # Won't reach here from parallel entry
    }
)

# Add alternative entry point for sequential
team_workflow_sequential = StateGraph(TeamState)
team_workflow_sequential.add_node("parallel", invoke_team_parallel)
team_workflow_sequential.add_node("sequential", invoke_team_sequential)
team_workflow_sequential.add_node("synthesize", synthesize_team_work)
team_workflow_sequential.set_entry_point("sequential")
team_workflow_sequential.add_edge("sequential", "synthesize")
team_workflow_sequential.add_edge("synthesize", END)

team_workflow.add_edge("synthesize", END)

# Compile both versions
team_parallel_system = team_workflow.compile()
team_sequential_system = team_workflow_sequential.compile()

# ==================== API ====================

def collaborate_on_task(
    task: str,
    mode: str = "parallel"
) -> dict:
    """
    Team collaboration on task.
    
    Args:
        task: Task description
        mode: "parallel" or "sequential"
    """
    
    if mode == "parallel":
        system = team_parallel_system
    else:
        system = team_sequential_system
    
    result = system.invoke({
        "messages": [HumanMessage(content=task)],
        "task": task,
        "team_members": ["researcher", "analyst", "writer"],
        "contributions": {},
        "collaboration_mode": mode,
        "final_result": ""
    })
    
    return {
        "success": True,
        "mode": mode,
        "contributors": list(result["contributions"].keys()),
        "result": result["final_result"]
    }

# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("TEAM COLLABORATION DEMO")
    print("="*60)
    
    task = "Analyze the impact of remote work on employee productivity"
    
    # Test parallel collaboration
    print(f"\n{'='*60}")
    print(f"PARALLEL COLLABORATION")
    print(f"{'='*60}")
    result_parallel = collaborate_on_task(task, mode="parallel")
    print(f"Contributors: {result_parallel['contributors']}")
    print(f"\nResult:\n{result_parallel['result'][:400]}...")
    
    # Test sequential collaboration
    print(f"\n{'='*60}")
    print(f"SEQUENTIAL COLLABORATION")
    print(f"{'='*60}")
    result_sequential = collaborate_on_task(task, mode="sequential")
    print(f"Contributors: {result_sequential['contributors']}")
    print(f"\nResult:\n{result_sequential['result'][:400]}...")
```

---

## 👔 Part 2: Supervisor-Worker Architectures

### Theory: Supervisor-Worker Pattern

#### What Is Supervisor-Worker Architecture?

A **supervisor-worker architecture** has a central supervisor that manages multiple workers:
- **Supervisor**: Plans, assigns, monitors, aggregates
- **Workers**: Execute specific tasks as assigned

This is the most common multi-agent pattern in production.

#### Supervisor Responsibilities

**1. Task Decomposition:**
```
Complex Task → [Subtask 1, Subtask 2, Subtask 3]
```

**2. Worker Assignment:**
```
Match subtask requirements to worker capabilities
Consider: expertise, availability, workload
```

**3. Progress Monitoring:**
```
Track: Which tasks assigned, which completed, any blockers
```

**4. Result Aggregation:**
```
Collect all worker outputs
Combine into coherent result
```

**5. Quality Control:**
```
Verify worker outputs meet standards
Request rework if needed
```

#### Worker Characteristics

**Workers should be:**
- **Specialized**: Each handles specific task types
- **Autonomous**: Can complete tasks independently
- **Stateless**: Don't maintain state between tasks (supervisor does)
- **Reportable**: Communicate status back to supervisor

#### Supervisor-Worker Communication Patterns

**Pattern 1: Push Assignment**
```
Supervisor → pushes task → Worker
Worker → completes → reports back
```

**Pattern 2: Pull from Queue**
```
Supervisor → adds tasks to queue
Workers → pull from queue when ready
```

**Pattern 3: Bidding System**
```
Supervisor → broadcasts task
Workers → bid based on capability
Supervisor → assigns to best bidder
```

#### Trade-offs

**Advantages:**
- Clear authority and responsibility
- Centralized decision-making
- Easy to understand and debug
- Good for hierarchical organizations

**Disadvantages:**
- Supervisor can become bottleneck
- Single point of failure
- Less flexibility than peer systems
- Supervisor must know all worker capabilities

---

### Implementation: Supervisor-Worker System

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

# ==================== SUPERVISOR-WORKER PATTERN ====================

# Worker state
class WorkerState(TypedDict):
    """Individual worker state"""
    messages: Annotated[Sequence[BaseMessage], add]
    task: str
    worker_type: str
    result: str
    status: Literal["pending", "working", "completed", "failed"]

# Supervisor state
class SupervisorState(TypedDict):
    """Supervisor orchestration state"""
    messages: Annotated[Sequence[BaseMessage], add]
    original_task: str
    work_plan: List[Dict]  # List of {task, assigned_to, status}
    current_step: int
    worker_results: Dict[str, str]
    supervisor_notes: Annotated[List[str], add]
    final_output: str

llm = ChatOllama(model="llama3.2", temperature=0.3)

# ==================== WORKER AGENTS ====================

# Data Worker
def data_worker_execute(state: WorkerState) -> dict:
    """Data worker executes task"""
    logger.info("📊 Data Worker: Executing")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a data worker. You handle data collection, cleaning, and analysis."),
        ("human", "{task}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"task": state["task"]})
    
    return {
        "result": response.content,
        "status": "completed"
    }

data_worker_workflow = StateGraph(WorkerState)
data_worker_workflow.add_node("execute", data_worker_execute)
data_worker_workflow.set_entry_point("execute")
data_worker_workflow.add_edge("execute", END)
data_worker = data_worker_workflow.compile()

# Compute Worker
def compute_worker_execute(state: WorkerState) -> dict:
    """Compute worker executes task"""
    logger.info("⚙️ Compute Worker: Executing")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a compute worker. You handle calculations, transformations, and processing."),
        ("human", "{task}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"task": state["task"]})
    
    return {
        "result": response.content,
        "status": "completed"
    }

compute_worker_workflow = StateGraph(WorkerState)
compute_worker_workflow.add_node("execute", compute_worker_execute)
compute_worker_workflow.set_entry_point("execute")
compute_worker_workflow.add_edge("execute", END)
compute_worker = compute_worker_workflow.compile()

# Report Worker
def report_worker_execute(state: WorkerState) -> dict:
    """Report worker executes task"""
    logger.info("📝 Report Worker: Executing")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a report worker. You create clear, structured reports and documentation."),
        ("human", "{task}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"task": state["task"]})
    
    return {
        "result": response.content,
        "status": "completed"
    }

report_worker_workflow = StateGraph(WorkerState)
report_worker_workflow.add_node("execute", report_worker_execute)
report_worker_workflow.set_entry_point("execute")
report_worker_workflow.add_edge("execute", END)
report_worker = report_worker_workflow.compile()

# ==================== SUPERVISOR ====================

def supervisor_plan(state: SupervisorState) -> dict:
    """Supervisor creates work plan"""
    logger.info("👔 Supervisor: Creating work plan")
    
    task = state["original_task"]
    
    planning_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a supervisor managing workers: data_worker, compute_worker, report_worker.
Create a work plan breaking the task into steps for these workers."""),
        ("human", """Task: {task}

Create a work plan. Respond in JSON:
{{
  "steps": [
    {{"task": "...", "assigned_to": "data_worker|compute_worker|report_worker"}},
    {{"task": "...", "assigned_to": "..."}}
  ]
}}

Work plan:""")
    ])
    
    chain = planning_prompt | llm
    response = chain.invoke({"task": task})
    
    # Parse plan
    response_text = response.content.strip()
    if "```json" in response_text:
        response_text = response_text.split("```json")[1].split("```")[0].strip()
    elif "```" in response_text:
        response_text = response_text.split("```")[1].split("```")[0].strip()
    
    import json
    try:
        plan_data = json.loads(response_text)
        work_plan = plan_data.get("steps", [])
    except:
        # Fallback plan
        work_plan = [
            {"task": "Gather required data", "assigned_to": "data_worker"},
            {"task": "Process and analyze", "assigned_to": "compute_worker"},
            {"task": "Create final report", "assigned_to": "report_worker"}
        ]
    
    # Add status to each step
    for step in work_plan:
        step["status"] = "pending"
    
    logger.info(f"👔 Supervisor: Created plan with {len(work_plan)} steps")
    
    return {
        "work_plan": work_plan,
        "current_step": 0,
        "supervisor_notes": [f"Plan created with {len(work_plan)} steps"]
    }

def supervisor_assign_and_monitor(state: SupervisorState) -> dict:
    """Supervisor assigns current task and monitors"""
    
    work_plan = state["work_plan"]
    current_step = state["current_step"]
    
    if current_step >= len(work_plan):
        logger.info("👔 Supervisor: All tasks completed")
        return {}
    
    current_task = work_plan[current_step]
    worker_type = current_task["assigned_to"]
    task_description = current_task["task"]
    
    logger.info(f"👔 Supervisor: Assigning step {current_step + 1}/{len(work_plan)} to {worker_type}")
    
    # Select worker
    workers = {
        "data_worker": data_worker,
        "compute_worker": compute_worker,
        "report_worker": report_worker
    }
    
    worker = workers.get(worker_type, data_worker)
    
    # Context from previous work
    context = ""
    if state["worker_results"]:
        context = "Previous results:\n" + "\n".join([
            f"{k}: {v[:100]}..."
            for k, v in state["worker_results"].items()
        ])
    
    full_task = f"{task_description}\n\n{context}" if context else task_description
    
    # Invoke worker
    result = worker.invoke({
        "messages": [],
        "task": full_task,
        "worker_type": worker_type,
        "result": "",
        "status": "pending"
    })
    
    # Update results
    worker_results = state["worker_results"].copy()
    worker_results[f"step_{current_step + 1}_{worker_type}"] = result["result"]
    
    # Update plan status
    work_plan[current_step]["status"] = "completed"
    
    logger.info(f"✅ Supervisor: Step {current_step + 1} completed")
    
    return {
        "worker_results": worker_results,
        "current_step": current_step + 1,
        "work_plan": work_plan,
        "supervisor_notes": [f"Step {current_step + 1} completed by {worker_type}"]
    }

def supervisor_aggregate(state: SupervisorState) -> dict:
    """Supervisor aggregates all results"""
    logger.info("👔 Supervisor: Aggregating results")
    
    # Compile work summary
    work_summary = "\n\n".join([
        f"Step {i+1} ({step['assigned_to']}): {step['task']}\nStatus: {step['status']}"
        for i, step in enumerate(state["work_plan"])
    ])
    
    # Compile results
    results_summary = "\n\n".join([
        f"{worker}: {result[:200]}..."
        for worker, result in state["worker_results"].items()
    ])
    
    aggregation_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a supervisor aggregating work from your team."),
        ("human", """Original Task: {task}

Work Plan Executed:
{work_plan}

Worker Results:
{results}

Aggregate into a final comprehensive result:""")
    ])
    
    chain = aggregation_prompt | llm
    response = chain.invoke({
        "task": state["original_task"],
        "work_plan": work_summary,
        "results": results_summary
    })
    
    final = f"""SUPERVISOR-WORKER COLLABORATION

Original Task: {state['original_task']}
Steps Completed: {len(state['work_plan'])}
Workers Involved: {len(set(step['assigned_to'] for step in state['work_plan']))}

Final Result:
{response.content}"""
    
    return {
        "final_output": final,
        "messages": [AIMessage(content=final)]
    }

def should_continue_work(state: SupervisorState) -> str:
    """Check if more work needed"""
    if state["current_step"] < len(state["work_plan"]):
        return "assign"
    return "aggregate"

# Build supervisor workflow
supervisor_workflow = StateGraph(SupervisorState)
supervisor_workflow.add_node("plan", supervisor_plan)
supervisor_workflow.add_node("assign", supervisor_assign_and_monitor)
supervisor_workflow.add_node("aggregate", supervisor_aggregate)

supervisor_workflow.set_entry_point("plan")
supervisor_workflow.add_edge("plan", "assign")

supervisor_workflow.add_conditional_edges(
    "assign",
    should_continue_work,
    {
        "assign": "assign",  # Loop
        "aggregate": "aggregate"
    }
)

supervisor_workflow.add_edge("aggregate", END)

supervisor_worker_system = supervisor_workflow.compile()

# ==================== API ====================

def execute_with_supervisor(task: str) -> dict:
    """Execute task using supervisor-worker pattern"""
    
    result = supervisor_worker_system.invoke({
        "messages": [HumanMessage(content=task)],
        "original_task": task,
        "work_plan": [],
        "current_step": 0,
        "worker_results": {},
        "supervisor_notes": [],
        "final_output": ""
    })
    
    return {
        "success": True,
        "steps_completed": len(result["work_plan"]),
        "workers_used": list(set(step["assigned_to"] for step in result["work_plan"])),
        "result": result["final_output"]
    }

# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("SUPERVISOR-WORKER DEMO")
    print("="*60)
    
    task = "Analyze customer churn data and create a report with recommendations"
    
    result = execute_with_supervisor(task)
    
    print(f"Task: {task}")
    print(f"Steps completed: {result['steps_completed']}")
    print(f"Workers used: {result['workers_used']}")
    print(f"\nResult:\n{result['result'][:500]}...")
```

---

## 🗳️ Part 3: Consensus Building

### Theory: Consensus Building

#### What Is Consensus Building?

**Consensus building** is when multiple agents with potentially different opinions work together to reach agreement on a decision or course of action.

#### Why Consensus Matters

In multi-agent systems:
- Agents may have different perspectives
- May evaluate options differently
- May prefer different approaches

**Consensus ensures:**
- All voices are heard
- Decision has broad support
- Higher quality decisions (wisdom of crowds)

#### Consensus Mechanisms

**1. Majority Vote:**
```
Agent A: Option 1
Agent B: Option 1
Agent C: Option 2
→ Option 1 wins (2 out of 3)
```

**2. Unanimous Agreement:**
```
All agents must agree
If any dissents, continue discussion
→ Strong agreement but may be slow
```

**3. Weighted Voting:**
```
Expert Agent A (weight 3): Option 1
Agent B (weight 1): Option 2
Agent C (weight 1): Option 1
→ Option 1 wins (4 vs 1)
```

**4. Iterative Refinement:**
```
Round 1: Agents propose options
Round 2: Agents discuss and critique
Round 3: Revised proposals
Round 4: Final vote
```

**5. Compromise:**
```
Agent A wants full feature set
Agent B wants minimal MVP
Consensus: Phase 1 MVP, Phase 2 features
```

#### Consensus Building Process

**Phase 1 - Information Gathering:**
- Each agent analyzes the problem
- Gathers relevant data
- Forms initial opinion

**Phase 2 - Proposal:**
- Agents present their recommendations
- Explain reasoning
- Provide evidence

**Phase 3 - Discussion:**
- Agents respond to each other
- Challenge assumptions
- Find common ground

**Phase 4 - Voting/Decision:**
- Apply consensus mechanism
- Make final decision
- Document rationale

---

### Implementation: Consensus Building System

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

# ==================== CONSENSUS BUILDING PATTERN ====================

# Agent vote state
class AgentVoteState(TypedDict):
    """Individual agent's vote"""
    messages: Annotated[Sequence[BaseMessage], add]
    decision_context: str
    agent_role: str
    vote: str
    reasoning: str
    confidence: float

# Consensus state
class ConsensusState(TypedDict):
    """State for consensus building"""
    messages: Annotated[Sequence[BaseMessage], add]
    decision_question: str
    options: List[str]
    agent_votes: Dict[str, Dict]  # {agent_name: {vote, reasoning, confidence}}
    discussion_rounds: Annotated[int, add]
    max_discussion_rounds: int
    consensus_reached: bool
    final_decision: str
    consensus_method: str  # "majority", "unanimous", "weighted"

llm = ChatOllama(model="llama3.2", temperature=0.5)

# ==================== VOTING AGENTS ====================

# Conservative Agent
def conservative_agent_vote(state: AgentVoteState) -> dict:
    """Conservative agent provides vote"""
    logger.info("🛡️ Conservative Agent: Voting")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a conservative decision-maker. Prioritize risk mitigation, safety, and proven approaches."),
        ("human", """Decision needed: {context}

Consider the decision carefully. What is your vote and reasoning?
Format: VOTE: [your choice]
REASONING: [your reasoning]
CONFIDENCE: [0.0-1.0]

Your response:""")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"context": state["decision_context"]})
    
    # Parse response
    content = response.content
    vote = "undecided"
    reasoning = content
    confidence = 0.7
    
    if "VOTE:" in content:
        vote = content.split("VOTE:")[1].split("REASONING:")[0].strip()
    if "REASONING:" in content:
        reasoning = content.split("REASONING:")[1].split("CONFIDENCE:")[0].strip() if "CONFIDENCE:" in content else content.split("REASONING:")[1].strip()
    if "CONFIDENCE:" in content:
        try:
            conf_str = content.split("CONFIDENCE:")[1].strip().split()[0]
            confidence = float(conf_str)
        except:
            pass
    
    return {
        "vote": vote,
        "reasoning": reasoning,
        "confidence": confidence
    }

conservative_workflow = StateGraph(AgentVoteState)
conservative_workflow.add_node("vote", conservative_agent_vote)
conservative_workflow.set_entry_point("vote")
conservative_workflow.add_edge("vote", END)
conservative_agent = conservative_workflow.compile()

# Progressive Agent
def progressive_agent_vote(state: AgentVoteState) -> dict:
    """Progressive agent provides vote"""
    logger.info("🚀 Progressive Agent: Voting")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a progressive decision-maker. Prioritize innovation, growth, and bold moves."),
        ("human", """Decision needed: {context}

Consider the decision carefully. What is your vote and reasoning?
Format: VOTE: [your choice]
REASONING: [your reasoning]
CONFIDENCE: [0.0-1.0]

Your response:""")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"context": state["decision_context"]})
    
    # Parse response
    content = response.content
    vote = "undecided"
    reasoning = content
    confidence = 0.7
    
    if "VOTE:" in content:
        vote = content.split("VOTE:")[1].split("REASONING:")[0].strip()
    if "REASONING:" in content:
        reasoning = content.split("REASONING:")[1].split("CONFIDENCE:")[0].strip() if "CONFIDENCE:" in content else content.split("REASONING:")[1].strip()
    if "CONFIDENCE:" in content:
        try:
            conf_str = content.split("CONFIDENCE:")[1].strip().split()[0]
            confidence = float(conf_str)
        except:
            pass
    
    return {
        "vote": vote,
        "reasoning": reasoning,
        "confidence": confidence
    }

progressive_workflow = StateGraph(AgentVoteState)
progressive_workflow.add_node("vote", progressive_agent_vote)
progressive_workflow.set_entry_point("vote")
progressive_workflow.add_edge("vote", END)
progressive_agent = progressive_workflow.compile()

# Pragmatic Agent
def pragmatic_agent_vote(state: AgentVoteState) -> dict:
    """Pragmatic agent provides vote"""
    logger.info("⚖️ Pragmatic Agent: Voting")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a pragmatic decision-maker. Balance risks and opportunities, focus on practical outcomes."),
        ("human", """Decision needed: {context}

Consider the decision carefully. What is your vote and reasoning?
Format: VOTE: [your choice]
REASONING: [your reasoning]
CONFIDENCE: [0.0-1.0]

Your response:""")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"context": state["decision_context"]})
    
    # Parse response
    content = response.content
    vote = "undecided"
    reasoning = content
    confidence = 0.7
    
    if "VOTE:" in content:
        vote = content.split("VOTE:")[1].split("REASONING:")[0].strip()
    if "REASONING:" in content:
        reasoning = content.split("REASONING:")[1].split("CONFIDENCE:")[0].strip() if "CONFIDENCE:" in content else content.split("REASONING:")[1].strip()
    if "CONFIDENCE:" in content:
        try:
            conf_str = content.split("CONFIDENCE:")[1].strip().split()[0]
            confidence = float(conf_str)
        except:
            pass
    
    return {
        "vote": vote,
        "reasoning": reasoning,
        "confidence": confidence
    }

pragmatic_workflow = StateGraph(AgentVoteState)
pragmatic_workflow.add_node("vote", pragmatic_agent_vote)
pragmatic_workflow.set_entry_point("vote")
pragmatic_workflow.add_edge("vote", END)
pragmatic_agent = pragmatic_workflow.compile()

# ==================== CONSENSUS COORDINATOR ====================

def collect_votes(state: ConsensusState) -> dict:
    """Collect votes from all agents"""
    logger.info("🗳️ Consensus: Collecting votes")
    
    decision_context = state["decision_question"]
    
    agents = {
        "conservative": conservative_agent,
        "progressive": progressive_agent,
        "pragmatic": pragmatic_agent
    }
    
    agent_votes = {}
    
    for agent_name, agent in agents.items():
        logger.info(f"   Asking {agent_name}...")
        
        result = agent.invoke({
            "messages": [],
            "decision_context": decision_context,
            "agent_role": agent_name,
            "vote": "",
            "reasoning": "",
            "confidence": 0.0
        })
        
        agent_votes[agent_name] = {
            "vote": result["vote"],
            "reasoning": result["reasoning"],
            "confidence": result["confidence"]
        }
    
    logger.info(f"✅ Collected {len(agent_votes)} votes")
    
    return {
        "agent_votes": agent_votes,
        "discussion_rounds": 1
    }

def evaluate_consensus(state: ConsensusState) -> dict:
    """Evaluate if consensus is reached"""
    logger.info("📊 Consensus: Evaluating agreement")
    
    agent_votes = state["agent_votes"]
    method = state["consensus_method"]
    
    votes = [v["vote"] for v in agent_votes.values()]
    
    if method == "unanimous":
        # All must agree
        unique_votes = set(votes)
        if len(unique_votes) == 1:
            consensus = True
            decision = votes[0]
        else:
            consensus = False
            decision = ""
    
    elif method == "majority":
        # Majority wins
        from collections import Counter
        vote_counts = Counter(votes)
        decision, count = vote_counts.most_common(1)[0]
        consensus = count > len(votes) / 2
    
    else:  # weighted
        # Weight by confidence
        weighted_votes = {}
        for agent, vote_data in agent_votes.items():
            vote = vote_data["vote"]
            confidence = vote_data["confidence"]
            weighted_votes[vote] = weighted_votes.get(vote, 0) + confidence
        
        decision = max(weighted_votes.items(), key=lambda x: x[1])[0]
        max_weight = weighted_votes[decision]
        total_weight = sum(weighted_votes.values())
        consensus = max_weight > total_weight / 2
    
    logger.info(f"Consensus reached: {consensus}, Decision: {decision}")
    
    return {
        "consensus_reached": consensus,
        "final_decision": decision if consensus else ""
    }

def format_results(state: ConsensusState) -> dict:
    """Format consensus results"""
    logger.info("📋 Consensus: Formatting results")
    
    # Format votes
    votes_summary = "\n".join([
        f"{agent.upper()}:\n  Vote: {data['vote']}\n  Reasoning: {data['reasoning'][:100]}...\n  Confidence: {data['confidence']}"
        for agent, data in state["agent_votes"].items()
    ])
    
    result = f"""CONSENSUS BUILDING RESULT

Decision Question: {state['decision_question']}
Method: {state['consensus_method']}
Rounds: {state['discussion_rounds']}

Votes:
{votes_summary}

Consensus Reached: {state['consensus_reached']}
Final Decision: {state['final_decision']}"""
    
    return {
        "messages": [AIMessage(content=result)]
    }

def should_continue_discussion(state: ConsensusState) -> str:
    """Decide if discussion should continue"""
    
    if state["consensus_reached"]:
        return "format"
    
    if state["discussion_rounds"] >= state["max_discussion_rounds"]:
        logger.info("Max discussion rounds reached")
        return "format"
    
    # In real system, would facilitate discussion and re-vote
    # For demo, we proceed to formatting
    return "format"

# Build consensus workflow
consensus_workflow = StateGraph(ConsensusState)
consensus_workflow.add_node("collect", collect_votes)
consensus_workflow.add_node("evaluate", evaluate_consensus)
consensus_workflow.add_node("format", format_results)

consensus_workflow.set_entry_point("collect")
consensus_workflow.add_edge("collect", "evaluate")

consensus_workflow.add_conditional_edges(
    "evaluate",
    should_continue_discussion,
    {
        "format": "format",
        "collect": "collect"  # Would re-vote after discussion
    }
)

consensus_workflow.add_edge("format", END)

consensus_system = consensus_workflow.compile()

# ==================== API ====================

def build_consensus(
    decision_question: str,
    method: str = "majority"
) -> dict:
    """
    Build consensus on a decision.
    
    Args:
        decision_question: The decision to be made
        method: "majority", "unanimous", or "weighted"
    """
    
    result = consensus_system.invoke({
        "messages": [HumanMessage(content=decision_question)],
        "decision_question": decision_question,
        "options": [],
        "agent_votes": {},
        "discussion_rounds": 0,
        "max_discussion_rounds": 3,
        "consensus_reached": False,
        "final_decision": "",
        "consensus_method": method
    })
    
    return {
        "success": True,
        "method": method,
        "consensus_reached": result["consensus_reached"],
        "decision": result["final_decision"],
        "votes": result["agent_votes"],
        "result": result["messages"][-1].content
    }

# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("CONSENSUS BUILDING DEMO")
    print("="*60)
    
    decision = "Should we launch the new product now or wait for more testing?"
    
    for method in ["majority", "weighted"]:
        print(f"\n{'='*60}")
        print(f"METHOD: {method.upper()}")
        print(f"{'='*60}")
        
        result = build_consensus(decision, method=method)
        
        print(f"Consensus reached: {result['consensus_reached']}")
        print(f"Decision: {result['decision']}")
        print(f"\n{result['result']}")
```

---

## 📦 Part 4: Work Distribution Strategies

### Theory: Work Distribution

#### What Is Work Distribution?

**Work distribution** is the process of dividing and assigning work among multiple agents to maximize efficiency, balance load, and complete tasks faster.

Think of it like **distributing packages among delivery drivers** - you want each driver to have a reasonable load, minimize travel time, and complete all deliveries efficiently.

#### Why Work Distribution Matters

**Problems without proper distribution:**
- Some agents overwhelmed, others idle
- Inefficient use of resources
- Bottlenecks slow everything down
- Unfair workload distribution

**Benefits of good distribution:**
- Faster completion (parallel work)
- Better resource utilization
- Balanced workload
- Scalability

#### Work Distribution Strategies

### Strategy 1: Round-Robin Distribution

**How it works:**
```
Task 1 → Agent A
Task 2 → Agent B
Task 3 → Agent C
Task 4 → Agent A (cycle back)
Task 5 → Agent B
...
```

**Characteristics:**
- Simple and predictable
- Fair distribution (equal task count)
- No consideration of task complexity
- No consideration of agent capabilities

**When to use:**
- Tasks are roughly equal in complexity
- All agents have same capabilities
- Simplicity is priority
- Predictability matters

**When NOT to use:**
- Tasks vary widely in size/complexity
- Agents have different capabilities
- Need optimization for speed

---

### Strategy 2: Load Balancing

**How it works:**
```
Check each agent's current load
Assign new task to least-loaded agent
Balance based on: task count, processing time, or resources
```

**Characteristics:**
- Considers current workload
- Dynamically adjusts
- More complex than round-robin
- Better resource utilization

**Load metrics:**
- **Task count**: Number of tasks assigned
- **Processing time**: Estimated/actual time to complete
- **Resource usage**: CPU, memory, tokens used

**Example:**
```
Agent A: 3 tasks (2 minutes remaining)
Agent B: 1 task (5 minutes remaining)
Agent C: 2 tasks (1 minute remaining)

New task (estimated 3 minutes):
→ Assign to Agent C (will finish current work soonest)
```

---

### Strategy 3: Capability-Based Distribution

**How it works:**
```
Analyze task requirements
Match to agent capabilities
Assign to best-fit agent
```

**Characteristics:**
- Considers agent specialization
- Higher quality results
- May create uneven distribution
- Requires capability metadata

**Example:**
```
Task: "Analyze financial data"
Agents:
  - Agent A: Financial expert (BEST FIT)
  - Agent B: Technical expert
  - Agent C: Writing expert
→ Assign to Agent A even if busier
```

---

### Strategy 4: Priority-Based Distribution

**How it works:**
```
Tasks have priority levels (high, medium, low)
High-priority tasks get best agents immediately
Low-priority tasks queued until capacity available
```

**Characteristics:**
- Ensures critical work done first
- May delay low-priority work indefinitely
- Requires priority classification
- Good for production systems

**Example:**
```
Queue:
  - Task A (HIGH): Process customer refund
  - Task B (LOW): Generate analytics report
  - Task C (HIGH): Security incident response

Assign HIGH priority tasks first, even if means delaying LOW
```

---

### Strategy 5: Bidding/Auction System

**How it works:**
```
1. Broadcast task to all agents
2. Agents "bid" based on:
   - Their current load
   - Their capability match
   - Their estimated completion time
3. Best bid wins the task
```

**Characteristics:**
- Decentralized decision-making
- Agents self-organize
- More complex implementation
- Good for heterogeneous agents

**Example:**
```
Task: "Translate document to Spanish"

Bids:
  - Agent A: "I can do it in 5 min, confidence 0.9" (knows Spanish well)
  - Agent B: "I can do it in 15 min, confidence 0.6" (basic Spanish)
  - Agent C: "I can do it in 10 min, confidence 0.8"

Winner: Agent A (best combo of time and confidence)
```

---

### Strategy 6: Pipeline Distribution

**How it works:**
```
Tasks flow through stages
Each agent handles one stage
Like assembly line
```

**Characteristics:**
- Agents specialize in one stage
- Continuous flow
- Good for standardized processes
- Sequential dependencies

**Example:**
```
Stage 1 (Agent A): Data collection
    ↓
Stage 2 (Agent B): Data processing
    ↓
Stage 3 (Agent C): Report generation

Each agent continuously processes their stage
```

---

### Comparison of Strategies

| Strategy | Complexity | Fairness | Efficiency | Use Case |
|----------|-----------|----------|------------|----------|
| Round-Robin | Low | High | Medium | Equal tasks, equal agents |
| Load Balancing | Medium | High | High | Variable task times |
| Capability-Based | Medium | Low | High (quality) | Specialized agents |
| Priority-Based | Medium | Low | High (critical work) | Production systems |
| Bidding | High | Medium | High | Heterogeneous agents |
| Pipeline | Low | Medium | High (throughput) | Standardized process |

---

### Implementation: Work Distribution System

```python
from typing import TypedDict, Annotated, Sequence, List, Dict, Literal
from operator import add
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from dataclasses import dataclass
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== WORK DISTRIBUTION PATTERNS ====================

@dataclass
class WorkItem:
    """Represents a unit of work"""
    id: str
    description: str
    priority: int  # 1=low, 5=high
    estimated_time: float  # minutes
    required_capability: str

@dataclass
class AgentProfile:
    """Agent capabilities and current load"""
    name: str
    capabilities: List[str]
    current_load: int  # number of tasks
    total_time_assigned: float  # minutes

# Distribution state
class DistributionState(TypedDict):
    """State for work distribution"""
    messages: Annotated[Sequence[BaseMessage], add]
    work_items: List[WorkItem]
    agent_profiles: Dict[str, AgentProfile]
    assignments: Dict[str, List[str]]  # agent_name -> [task_ids]
    distribution_strategy: Literal["round_robin", "load_balance", "capability", "priority"]
    distribution_log: Annotated[List[str], add]

# ==================== DISTRIBUTION STRATEGIES ====================

def distribute_round_robin(state: DistributionState) -> dict:
    """Distribute work in round-robin fashion"""
    logger.info("🔄 Distribution: Round-robin strategy")
    
    work_items = state["work_items"]
    agent_profiles = state["agent_profiles"]
    agents = list(agent_profiles.keys())
    
    assignments = {agent: [] for agent in agents}
    distribution_log = []
    
    for i, item in enumerate(work_items):
        agent = agents[i % len(agents)]
        assignments[agent].append(item.id)
        
        # Update agent load
        agent_profiles[agent].current_load += 1
        agent_profiles[agent].total_time_assigned += item.estimated_time
        
        distribution_log.append(f"Task {item.id} → {agent} (round-robin)")
        logger.info(f"  Assigned {item.id} to {agent}")
    
    return {
        "assignments": assignments,
        "agent_profiles": agent_profiles,
        "distribution_log": distribution_log
    }

def distribute_load_balance(state: DistributionState) -> dict:
    """Distribute work based on current load"""
    logger.info("⚖️ Distribution: Load balancing strategy")
    
    work_items = state["work_items"]
    agent_profiles = state["agent_profiles"]
    
    assignments = {agent: [] for agent in agent_profiles.keys()}
    distribution_log = []
    
    for item in work_items:
        # Find least-loaded agent
        least_loaded = min(
            agent_profiles.items(),
            key=lambda x: x[1].total_time_assigned
        )
        
        agent_name = least_loaded[0]
        assignments[agent_name].append(item.id)
        
        # Update agent load
        agent_profiles[agent_name].current_load += 1
        agent_profiles[agent_name].total_time_assigned += item.estimated_time
        
        distribution_log.append(
            f"Task {item.id} → {agent_name} (load: {agent_profiles[agent_name].total_time_assigned:.1f}min)"
        )
        logger.info(f"  Assigned {item.id} to {agent_name} (least loaded)")
    
    return {
        "assignments": assignments,
        "agent_profiles": agent_profiles,
        "distribution_log": distribution_log
    }

def distribute_capability_based(state: DistributionState) -> dict:
    """Distribute work based on agent capabilities"""
    logger.info("🎯 Distribution: Capability-based strategy")
    
    work_items = state["work_items"]
    agent_profiles = state["agent_profiles"]
    
    assignments = {agent: [] for agent in agent_profiles.keys()}
    distribution_log = []
    
    for item in work_items:
        # Find agents with required capability
        capable_agents = [
            (name, profile) for name, profile in agent_profiles.items()
            if item.required_capability in profile.capabilities
        ]
        
        if capable_agents:
            # Among capable agents, choose least loaded
            agent_name = min(capable_agents, key=lambda x: x[1].total_time_assigned)[0]
        else:
            # Fallback: assign to least loaded agent
            agent_name = min(agent_profiles.items(), key=lambda x: x[1].total_time_assigned)[0]
            distribution_log.append(f"⚠️ No agent with '{item.required_capability}', using fallback")
        
        assignments[agent_name].append(item.id)
        
        # Update agent load
        agent_profiles[agent_name].current_load += 1
        agent_profiles[agent_name].total_time_assigned += item.estimated_time
        
        distribution_log.append(
            f"Task {item.id} (needs: {item.required_capability}) → {agent_name}"
        )
        logger.info(f"  Assigned {item.id} to {agent_name} (capability match)")
    
    return {
        "assignments": assignments,
        "agent_profiles": agent_profiles,
        "distribution_log": distribution_log
    }

def distribute_priority_based(state: DistributionState) -> dict:
    """Distribute work prioritizing high-priority tasks"""
    logger.info("⭐ Distribution: Priority-based strategy")
    
    work_items = sorted(state["work_items"], key=lambda x: x.priority, reverse=True)
    agent_profiles = state["agent_profiles"]
    
    assignments = {agent: [] for agent in agent_profiles.keys()}
    distribution_log = []
    
    for item in work_items:
        # High priority gets best (least loaded) agent
        least_loaded = min(
            agent_profiles.items(),
            key=lambda x: x[1].total_time_assigned
        )
        
        agent_name = least_loaded[0]
        assignments[agent_name].append(item.id)
        
        # Update agent load
        agent_profiles[agent_name].current_load += 1
        agent_profiles[agent_name].total_time_assigned += item.estimated_time
        
        distribution_log.append(
            f"Task {item.id} (priority: {item.priority}) → {agent_name}"
        )
        logger.info(f"  Assigned {item.id} (priority {item.priority}) to {agent_name}")
    
    return {
        "assignments": assignments,
        "agent_profiles": agent_profiles,
        "distribution_log": distribution_log
    }

def format_distribution_results(state: DistributionState) -> dict:
    """Format distribution results"""
    
    assignments = state["assignments"]
    agent_profiles = state["agent_profiles"]
    
    # Summary by agent
    summary = []
    for agent_name, task_ids in assignments.items():
        profile = agent_profiles[agent_name]
        summary.append(
            f"{agent_name}: {len(task_ids)} tasks, {profile.total_time_assigned:.1f} min total"
        )
    
    # Distribution log
    log_text = "\n".join(state["distribution_log"])
    
    result = f"""WORK DISTRIBUTION RESULT

Strategy: {state['distribution_strategy']}
Total Tasks: {len(state['work_items'])}
Total Agents: {len(agent_profiles)}

Assignments:
{chr(10).join(summary)}

Distribution Log:
{log_text}"""
    
    return {
        "messages": [AIMessage(content=result)]
    }

def route_distribution_strategy(state: DistributionState) -> str:
    """Route to appropriate distribution strategy"""
    return state["distribution_strategy"]

# Build distribution workflow
distribution_workflow = StateGraph(DistributionState)

distribution_workflow.add_node("round_robin", distribute_round_robin)
distribution_workflow.add_node("load_balance", distribute_load_balance)
distribution_workflow.add_node("capability", distribute_capability_based)
distribution_workflow.add_node("priority", distribute_priority_based)
distribution_workflow.add_node("format", format_distribution_results)

distribution_workflow.set_entry_point("round_robin")  # Will be overridden by conditional

distribution_workflow.add_conditional_edges(
    "round_robin",
    route_distribution_strategy,
    {
        "round_robin": "format",
        "load_balance": "format",
        "capability": "format",
        "priority": "format"
    }
)

# Need to add routing from all strategies
for strategy in ["load_balance", "capability", "priority"]:
    distribution_workflow.add_edge(strategy, "format")

distribution_workflow.add_edge("format", END)

# Compile (we'll need to set proper entry based on strategy)
distribution_system = distribution_workflow.compile()

# ==================== API ====================

def distribute_work(
    work_items: List[Dict],
    agents: List[Dict],
    strategy: str = "load_balance"
) -> dict:
    """
    Distribute work among agents.
    
    Args:
        work_items: List of tasks with {id, description, priority, estimated_time, required_capability}
        agents: List of agents with {name, capabilities}
        strategy: "round_robin", "load_balance", "capability", or "priority"
    """
    
    # Convert to proper objects
    work_objects = [
        WorkItem(
            id=item["id"],
            description=item.get("description", ""),
            priority=item.get("priority", 3),
            estimated_time=item.get("estimated_time", 5.0),
            required_capability=item.get("required_capability", "general")
        )
        for item in work_items
    ]
    
    agent_profiles = {
        agent["name"]: AgentProfile(
            name=agent["name"],
            capabilities=agent.get("capabilities", ["general"]),
            current_load=0,
            total_time_assigned=0.0
        )
        for agent in agents
    }
    
    # Create initial state with strategy-specific entry point
    initial_state = {
        "messages": [HumanMessage(content=f"Distribute {len(work_items)} tasks")],
        "work_items": work_objects,
        "agent_profiles": agent_profiles,
        "assignments": {},
        "distribution_strategy": strategy,
        "distribution_log": []
    }
    
    # Execute appropriate strategy directly
    if strategy == "round_robin":
        state = distribute_round_robin(initial_state)
    elif strategy == "load_balance":
        state = distribute_load_balance(initial_state)
    elif strategy == "capability":
        state = distribute_capability_based(initial_state)
    elif strategy == "priority":
        state = distribute_priority_based(initial_state)
    else:
        state = distribute_load_balance(initial_state)
    
    # Merge with initial state
    result_state = {**initial_state, **state}
    
    # Format results
    final = format_distribution_results(result_state)
    
    return {
        "success": True,
        "strategy": strategy,
        "assignments": state["assignments"],
        "result": final["messages"][0].content
    }

# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("WORK DISTRIBUTION STRATEGIES DEMO")
    print("="*60)
    
    # Define work items
    tasks = [
        {"id": "T1", "description": "Analyze data", "priority": 5, "estimated_time": 10, "required_capability": "data"},
        {"id": "T2", "description": "Write report", "priority": 3, "estimated_time": 15, "required_capability": "writing"},
        {"id": "T3", "description": "Code review", "priority": 4, "estimated_time": 5, "required_capability": "coding"},
        {"id": "T4", "description": "Analyze trends", "priority": 5, "estimated_time": 8, "required_capability": "data"},
        {"id": "T5", "description": "Create docs", "priority": 2, "estimated_time": 12, "required_capability": "writing"},
        {"id": "T6", "description": "Fix bug", "priority": 5, "estimated_time": 6, "required_capability": "coding"},
    ]
    
    # Define agents
    agents = [
        {"name": "Agent_A", "capabilities": ["data", "general"]},
        {"name": "Agent_B", "capabilities": ["writing", "general"]},
        {"name": "Agent_C", "capabilities": ["coding", "general"]},
    ]
    
    # Test different strategies
    for strategy in ["round_robin", "load_balance", "capability", "priority"]:
        print(f"\n{'='*60}")
        print(f"STRATEGY: {strategy.upper()}")
        print(f"{'='*60}")
        
        result = distribute_work(tasks, agents, strategy=strategy)
        print(result["result"])
```

---

## 🤝 Part 5: Handoff and Delegation

### Theory: Handoff and Delegation

#### What Is Handoff?

**Handoff** is when one agent transfers a task to another agent, typically because:
- Task is outside first agent's expertise
- First agent is overloaded
- Task requires different permissions/access
- Workflow requires sequential processing

Think of **medical referrals** - your primary doctor refers you to a specialist.

#### What Is Delegation?

**Delegation** is when an agent assigns a subtask to another agent while maintaining overall responsibility for the task.

Think of **project management** - PM delegates specific tasks but owns the project outcome.

#### Handoff vs Delegation

| Aspect | Handoff | Delegation |
|--------|---------|------------|
| **Ownership** | Transfers to new agent | Remains with original agent |
| **Responsibility** | New agent responsible | Original agent responsible |
| **Monitoring** | Minimal | Continuous |
| **Scope** | Usually entire task | Subtask of larger work |
| **Return** | Usually doesn't return | Returns result to delegator |

#### When to Use Each

**Use Handoff When:**
- Task clearly belongs to another domain
- Original agent can't help further
- Clean transfer of ownership needed
- No need for original agent to monitor

**Use Delegation When:**
- Need specialist help but maintain control
- Subtask is part of larger workflow
- Original agent coordinates multiple subtasks
- Need to aggregate results

#### Handoff Patterns

**Pattern 1: Simple Handoff**
```
Agent A: "I can't handle this, passing to Agent B"
Agent B: Takes over completely
Agent A: No longer involved
```

**Pattern 2: Handoff with Context**
```
Agent A: "Passing to Agent B, here's what I've done so far"
Agent B: Receives task + context
Agent B: Continues from where Agent A left off
```

**Pattern 3: Handoff Chain**
```
Agent A → hands off to → Agent B → hands off to → Agent C
Like relay race
```

#### Delegation Patterns

**Pattern 1: Sequential Delegation**
```
Manager: Delegates Task 1 to Worker A
Manager: Waits for completion
Manager: Delegates Task 2 to Worker B (using Task 1 results)
Manager: Aggregates results
```

**Pattern 2: Parallel Delegation**
```
Manager: Delegates Task 1 to Worker A
Manager: Delegates Task 2 to Worker B (simultaneously)
Manager: Delegates Task 3 to Worker C (simultaneously)
Manager: Waits for all
Manager: Aggregates results
```

**Pattern 3: Hierarchical Delegation**
```
Executive: Delegates to Middle Manager 1, Middle Manager 2
Middle Manager 1: Delegates to Workers A, B
Middle Manager 2: Delegates to Workers C, D
Results flow back up
```

#### Critical Elements

**For Successful Handoff:**
1. **Clear handoff criteria** - When to handoff?
2. **Context transfer** - What information to pass?
3. **Acknowledgment** - Confirm new agent accepts
4. **Clean break** - Original agent fully disengages

**For Successful Delegation:**
1. **Clear task definition** - What exactly to do?
2. **Authority level** - What decisions can delegatee make?
3. **Reporting structure** - How to communicate progress?
4. **Integration plan** - How results will be used?

---

### Implementation: Handoff and Delegation System

```python
from typing import TypedDict, Annotated, Sequence, List, Dict, Optional
from operator import add
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== HANDOFF AND DELEGATION PATTERNS ====================

# Agent execution state
class AgentExecutionState(TypedDict):
    """State for individual agent execution"""
    messages: Annotated[Sequence[BaseMessage], add]
    task: str
    context: str
    agent_name: str
    result: str
    should_handoff: bool
    handoff_to: Optional[str]
    handoff_reason: str

# Orchestration state
class HandoffDelegationState(TypedDict):
    """State for handoff/delegation orchestration"""
    messages: Annotated[Sequence[BaseMessage], add]
    original_task: str
    current_owner: str
    handoff_chain: Annotated[List[str], add]
    context_accumulated: str
    delegated_tasks: Dict[str, str]  # subtask_id -> result
    final_result: str
    mode: str  # "handoff" or "delegation"

llm = ChatOllama(model="llama3.2", temperature=0.3)

# ==================== SPECIALIST AGENTS ====================

# Frontend Agent
def frontend_agent_execute(state: AgentExecutionState) -> dict:
    """Frontend specialist"""
    logger.info("🎨 Frontend Agent: Executing")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a frontend specialist. You handle UI/UX tasks."),
        ("human", """Task: {task}

Context from previous agents:
{context}

Execute your part. If this requires backend work, say 'HANDOFF: backend'

Your response:""")
    ])
    
    chain = prompt | llm
    response = chain.invoke({
        "task": state["task"],
        "context": state.get("context", "No prior context")
    })
    
    content = response.content
    
    # Check for handoff
    should_handoff = "HANDOFF:" in content
    handoff_to = ""
    handoff_reason = ""
    
    if should_handoff:
        parts = content.split("HANDOFF:")
        result = parts[0].strip()
        handoff_info = parts[1].strip()
        handoff_to = handoff_info.split()[0] if handoff_info.split() else ""
        handoff_reason = " ".join(handoff_info.split()[1:]) if len(handoff_info.split()) > 1 else "Requires other expertise"
    else:
        result = content
    
    return {
        "result": result,
        "should_handoff": should_handoff,
        "handoff_to": handoff_to,
        "handoff_reason": handoff_reason
    }

frontend_workflow = StateGraph(AgentExecutionState)
frontend_workflow.add_node("execute", frontend_agent_execute)
frontend_workflow.set_entry_point("execute")
frontend_workflow.add_edge("execute", END)
frontend_agent = frontend_workflow.compile()

# Backend Agent
def backend_agent_execute(state: AgentExecutionState) -> dict:
    """Backend specialist"""
    logger.info("⚙️ Backend Agent: Executing")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a backend specialist. You handle server-side logic and databases."),
        ("human", """Task: {task}

Context from previous agents:
{context}

Execute your part. If this requires database work, say 'HANDOFF: database'

Your response:""")
    ])
    
    chain = prompt | llm
    response = chain.invoke({
        "task": state["task"],
        "context": state.get("context", "No prior context")
    })
    
    content = response.content
    
    should_handoff = "HANDOFF:" in content
    handoff_to = ""
    handoff_reason = ""
    
    if should_handoff:
        parts = content.split("HANDOFF:")
        result = parts[0].strip()
        handoff_info = parts[1].strip()
        handoff_to = handoff_info.split()[0] if handoff_info.split() else ""
        handoff_reason = " ".join(handoff_info.split()[1:]) if len(handoff_info.split()) > 1 else "Requires other expertise"
    else:
        result = content
    
    return {
        "result": result,
        "should_handoff": should_handoff,
        "handoff_to": handoff_to,
        "handoff_reason": handoff_reason
    }

backend_workflow = StateGraph(AgentExecutionState)
backend_workflow.add_node("execute", backend_agent_execute)
backend_workflow.set_entry_point("execute")
backend_workflow.add_edge("execute", END)
backend_agent = backend_workflow.compile()

# Database Agent
def database_agent_execute(state: AgentExecutionState) -> dict:
    """Database specialist"""
    logger.info("💾 Database Agent: Executing")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a database specialist. You handle data modeling and queries."),
        ("human", """Task: {task}

Context from previous agents:
{context}

Execute your part.

Your response:""")
    ])
    
    chain = prompt | llm
    response = chain.invoke({
        "task": state["task"],
        "context": state.get("context", "No prior context")
    })
    
    return {
        "result": response.content,
        "should_handoff": False,
        "handoff_to": "",
        "handoff_reason": ""
    }

database_workflow = StateGraph(AgentExecutionState)
database_workflow.add_node("execute", database_agent_execute)
database_workflow.set_entry_point("execute")
database_workflow.add_edge("execute", END)
database_agent = database_workflow.compile()

# ==================== HANDOFF ORCHESTRATOR ====================

def execute_with_handoff(state: HandoffDelegationState) -> dict:
    """Execute task with handoff chain"""
    logger.info(f"🔄 Handoff: Current owner is {state['current_owner']}")
    
    agents = {
        "frontend": frontend_agent,
        "backend": backend_agent,
        "database": database_agent
    }
    
    current_agent = agents.get(state["current_owner"], frontend_agent)
    
    # Execute with current agent
    result = current_agent.invoke({
        "messages": [],
        "task": state["original_task"],
        "context": state["context_accumulated"],
        "agent_name": state["current_owner"],
        "result": "",
        "should_handoff": False,
        "handoff_to": "",
        "handoff_reason": ""
    })
    
    # Update context
    new_context = state["context_accumulated"]
    new_context += f"\n\n[{state['current_owner']}]: {result['result']}"
    
    # Check for handoff
    if result["should_handoff"] and result["handoff_to"]:
        logger.info(f"🔀 Handoff: {state['current_owner']} → {result['handoff_to']}")
        logger.info(f"   Reason: {result['handoff_reason']}")
        
        return {
            "current_owner": result["handoff_to"],
            "handoff_chain": [f"{state['current_owner']} → {result['handoff_to']}"],
            "context_accumulated": new_context
        }
    else:
        # No more handoffs, we're done
        logger.info(f"✅ Handoff chain complete")
        
        return {
            "final_result": result["result"],
            "context_accumulated": new_context
        }

def should_continue_handoff(state: HandoffDelegationState) -> str:
    """Check if handoff should continue"""
    if state.get("final_result"):
        return "format"
    
    # Limit handoff chain to prevent infinite loops
    if len(state["handoff_chain"]) >= 5:
        logger.warning("Max handoff chain reached")
        return "format"
    
    return "execute"

def format_handoff_results(state: HandoffDelegationState) -> dict:
    """Format handoff results"""
    
    chain_text = " → ".join(state["handoff_chain"]) if state["handoff_chain"] else "No handoffs"
    
    result = f"""HANDOFF CHAIN RESULT

Original Task: {state['original_task']}
Handoff Chain: {chain_text}
Final Owner: {state['current_owner']}

Context History:
{state['context_accumulated']}

Final Result:
{state.get('final_result', 'No final result')}"""
    
    return {
        "messages": [AIMessage(content=result)]
    }

# Build handoff workflow
handoff_workflow = StateGraph(HandoffDelegationState)
handoff_workflow.add_node("execute", execute_with_handoff)
handoff_workflow.add_node("format", format_handoff_results)

handoff_workflow.set_entry_point("execute")

handoff_workflow.add_conditional_edges(
    "execute",
    should_continue_handoff,
    {
        "execute": "execute",  # Loop for next handoff
        "format": "format"
    }
)

handoff_workflow.add_edge("format", END)

handoff_system = handoff_workflow.compile()

# ==================== DELEGATION ORCHESTRATOR ====================

def manager_delegate(state: HandoffDelegationState) -> dict:
    """Manager delegates subtasks"""
    logger.info("👔 Manager: Delegating subtasks")
    
    task = state["original_task"]
    
    # Break into subtasks
    subtasks = {
        "frontend": f"Design UI for: {task}",
        "backend": f"Implement logic for: {task}",
        "database": f"Design data model for: {task}"
    }
    
    agents = {
        "frontend": frontend_agent,
        "backend": backend_agent,
        "database": database_agent
    }
    
    results = {}
    
    # Delegate to all agents in parallel
    for subtask_id, subtask_desc in subtasks.items():
        logger.info(f"   Delegating to {subtask_id}...")
        
        agent = agents[subtask_id]
        result = agent.invoke({
            "messages": [],
            "task": subtask_desc,
            "context": "",
            "agent_name": subtask_id,
            "result": "",
            "should_handoff": False,
            "handoff_to": "",
            "handoff_reason": ""
        })
        
        results[subtask_id] = result["result"]
    
    logger.info(f"✅ Manager: All {len(results)} subtasks completed")
    
    return {
        "delegated_tasks": results
    }

def manager_aggregate(state: HandoffDelegationState) -> dict:
    """Manager aggregates delegated results"""
    logger.info("👔 Manager: Aggregating results")
    
    results_text = "\n\n".join([
        f"{agent.upper()}:\n{result}"
        for agent, result in state["delegated_tasks"].items()
    ])
    
    aggregation_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a manager synthesizing your team's work."),
        ("human", """Original Task: {task}

Team Results:
{results}

Synthesize into a cohesive solution:""")
    ])
    
    chain = aggregation_prompt | llm
    response = chain.invoke({
        "task": state["original_task"],
        "results": results_text
    })
    
    return {
        "final_result": response.content
    }

def format_delegation_results(state: HandoffDelegationState) -> dict:
    """Format delegation results"""
    
    result = f"""DELEGATION RESULT

Original Task: {state['original_task']}
Subtasks Delegated: {len(state['delegated_tasks'])}

Subtask Results:
{chr(10).join([f"- {agent}: {result[:100]}..." for agent, result in state['delegated_tasks'].items()])}

Final Integrated Solution:
{state['final_result']}"""
    
    return {
        "messages": [AIMessage(content=result)]
    }

# Build delegation workflow
delegation_workflow = StateGraph(HandoffDelegationState)
delegation_workflow.add_node("delegate", manager_delegate)
delegation_workflow.add_node("aggregate", manager_aggregate)
delegation_workflow.add_node("format", format_delegation_results)

delegation_workflow.set_entry_point("delegate")
delegation_workflow.add_edge("delegate", "aggregate")
delegation_workflow.add_edge("aggregate", "format")
delegation_workflow.add_edge("format", END)

delegation_system = delegation_workflow.compile()

# ==================== API ====================

def execute_with_handoff_or_delegation(
    task: str,
    mode: str = "delegation",
    starting_agent: str = "frontend"
) -> dict:
    """
    Execute task with handoff or delegation.
    
    Args:
        task: Task description
        mode: "handoff" or "delegation"
        starting_agent: For handoff mode, which agent starts
    """
    
    if mode == "handoff":
        system = handoff_system
        
        result = system.invoke({
            "messages": [HumanMessage(content=task)],
            "original_task": task,
            "current_owner": starting_agent,
            "handoff_chain": [],
            "context_accumulated": "",
            "delegated_tasks": {},
            "final_result": "",
            "mode": mode
        })
        
        return {
            "success": True,
            "mode": mode,
            "handoff_chain": result["handoff_chain"],
            "result": result["messages"][-1].content
        }
    
    else:  # delegation
        system = delegation_system
        
        result = system.invoke({
            "messages": [HumanMessage(content=task)],
            "original_task": task,
            "current_owner": "manager",
            "handoff_chain": [],
            "context_accumulated": "",
            "delegated_tasks": {},
            "final_result": "",
            "mode": mode
        })
        
        return {
            "success": True,
            "mode": mode,
            "subtasks": len(result["delegated_tasks"]),
            "result": result["messages"][-1].content
        }

# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("HANDOFF AND DELEGATION DEMO")
    print("="*60)
    
    task = "Build a user authentication system"
    
    # Test handoff
    print(f"\n{'='*60}")
    print(f"MODE: HANDOFF")
    print(f"{'='*60}")
    result_handoff = execute_with_handoff_or_delegation(task, mode="handoff", starting_agent="frontend")
    print(f"Handoff chain: {result_handoff['handoff_chain']}")
    print(f"\n{result_handoff['result'][:500]}...")
    
    # Test delegation
    print(f"\n{'='*60}")
    print(f"MODE: DELEGATION")
    print(f"{'='*60}")
    result_delegation = execute_with_handoff_or_delegation(task, mode="delegation")
    print(f"Subtasks: {result_delegation['subtasks']}")
    print(f"\n{result_delegation['result'][:500]}...")
```

---

## 📋 Part 6: Best Practices and Summary

### Best Practices for Cooperative Multi-Agent Systems

#### 1. Communication Patterns

**✅ DO:**
- Make communication explicit and visible
- Log all agent interactions
- Include context when passing between agents
- Use structured message formats

**❌ DON'T:**
- Assume agents know implicit context
- Skip logging communication
- Pass incomplete information
- Use ambiguous messages

---

#### 2. Coordination

**✅ DO:**
- Have clear authority structure
- Define decision-making processes upfront
- Handle conflicts gracefully
- Monitor coordination overhead

**❌ DON'T:**
- Create circular dependencies
- Have conflicting authorities
- Ignore coordination costs
- Over-coordinate simple tasks

---

#### 3. Work Distribution

**✅ DO:**
- Match agent capabilities to tasks
- Balance workload
- Monitor for bottlenecks
- Adjust distribution dynamically

**❌ DON'T:**
- Ignore agent specialization
- Create imbalanced loads
- Use fixed distribution for dynamic workloads
- Forget to measure distribution effectiveness

---

#### 4. Handoff and Delegation

**✅ DO:**
- Transfer complete context
- Confirm handoff acceptance
- Define clear ownership
- Document delegation hierarchy

**❌ DON'T:**
- Handoff without context
- Lose track of ownership
- Create unclear responsibilities
- Delegate without monitoring

---

### Pattern Selection Guide

```
Question: What's your goal?

Need specialists to collaborate?
  └─> Parallel work? → Team Collaboration (Parallel)
  └─> Sequential refinement? → Team Collaboration (Sequential)

Need central control?
  └─> Simple routing? → Router Pattern
  └─> Active management? → Supervisor-Worker

Need agreement from multiple perspectives?
  └─> Consensus Building

Need to distribute many tasks?
  └─> Tasks equal? → Round-Robin Distribution
  └─> Tasks varied? → Load Balancing
  └─> Need expertise match? → Capability-Based

Task needs to move between specialists?
  └─> Transfer ownership? → Handoff
  └─> Keep ownership? → Delegation
```

---

### Key Principles

1. **Cooperation requires communication**
   - Agents must share information effectively
   - Communication overhead is real - account for it

2. **Clear roles and responsibilities**
   - Each agent should know its role
   - Avoid ambiguity in ownership

3. **Balance autonomy and coordination**
   - Too much autonomy → chaos
   - Too much coordination → bottleneck

4. **Monitor and adapt**
   - Track performance metrics
   - Adjust strategies based on data

5. **Design for failure**
   - Agents will fail
   - Have fallback mechanisms
   - Don't let one failure cascade

---

### Comparison of All Patterns

| Pattern | Best For | Coordination | Complexity | Scalability |
|---------|----------|--------------|------------|-------------|
| **Team Collaboration** | Multiple perspectives needed | Medium | Medium | Medium |
| **Supervisor-Worker** | Clear hierarchy, active management | High | Medium | Medium-High |
| **Consensus Building** | Democratic decisions | High | High | Low |
| **Work Distribution** | Many similar tasks | Low | Low-Medium | High |
| **Handoff** | Task crosses domains | Low | Low | High |
| **Delegation** | Complex hierarchical work | Medium | Medium | Medium |

---

## 🧠 Mental Models

### Mental Model 1: Orchestra vs Jazz Band

**Supervisor-Worker (Orchestra):**
- Conductor (supervisor) leads
- Musicians (workers) follow conductor
- Clear coordination
- Produces consistent results

**Team Collaboration (Jazz Band):**
- Musicians collaborate
- Each contributes improvisation
- More fluid communication
- Creative but requires trust

### Mental Model 2: Restaurant Kitchen

**Work Distribution:**
- Host assigns tables to servers (round-robin)
- Chef assigns dishes based on cook expertise (capability-based)
- Expediter balances orders (load balancing)

**Handoff:**
- Waiter takes order → hands off to kitchen
- Line cook → hands off to expo
- Expo → hands off back to waiter

**Delegation:**
- Head chef delegates to stations
- Maintains oversight
- Each station handles their part
- Chef aggregates into final service

---

## ✅ Chapter 13 Complete!

**You now understand:**
- ✅ Team collaboration (parallel & sequential)
- ✅ Supervisor-worker architectures (planning, assigning, monitoring)
- ✅ Consensus building (voting, discussion, agreement)
- ✅ Work distribution strategies (6 different approaches)
- ✅ Handoff (transfer ownership)
- ✅ Delegation (maintain ownership)
- ✅ When to use each pattern
- ✅ Best practices for cooperative systems

**Ready for Chapter 14?** 

**Chapter 14: Roleplay & Debate Agents** will cover:
- Role-based agent systems
- Debate frameworks (Pro-con analysis)
- Perspective-taking and simulation
- Iterative refinement through discussion
- Use cases: decision making, content creation

Just say "Continue to Chapter 14" when ready!