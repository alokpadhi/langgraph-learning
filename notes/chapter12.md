# Chapter 12: Multi-Agent Fundamentals

## 🎯 Why Multi-Agent Systems?

### The Evolution from Single to Multi-Agent

```
Single Agent              Multi-Step Workflow       True Multi-Agent
    │                           │                         │
    ▼                           ▼                         ▼
┌─────────┐            ┌──────────────┐           ┌──────────┐
│         │            │  Node 1      │           │ Agent 1  │
│  Agent  │────────→   │  Node 2      │           │ (Graph)  │
│         │            │  Node 3      │           │          │
└─────────┘            └──────────────┘           └────┬─────┘
                       (Same State)                    │
                                                       ↓ message
                                                  ┌────┴─────┐
                                                  │ Agent 2  │
                                                  │ (Graph)  │
                                                  └──────────┘
                                                  (Own State)
```

### When You NEED Multi-Agent Systems

✅ **Use Multi-Agent When:**
- Different agents require different state schemas
- Agents developed by different teams
- Need to test/deploy agents independently
- Agents operate at different scales
- Complex internal logic per agent
- Reusable agent components

❌ **Don't Use Multi-Agent When:**
- Simple sequential workflow
- All steps share same state
- No need for modularity
- Tight coupling is acceptable

---

## 📡 Part 1: Communication Protocols

### Three Main Communication Patterns

```python
from typing import TypedDict, Annotated, Sequence, List, Dict, Any
from operator import add
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== COMMUNICATION PATTERNS ====================

# Pattern 1: DIRECT INVOCATION
# Parent → Child (synchronous)

# Pattern 2: MESSAGE PASSING
# Agent A → Queue → Agent B (asynchronous)

# Pattern 3: SHARED STATE
# Agent A writes → Shared Memory ← Agent B reads

# ==================== PATTERN 1: DIRECT INVOCATION ====================

class AgentState(TypedDict):
    """Individual agent state"""
    messages: Annotated[Sequence[BaseMessage], add]
    input: str
    output: str

class OrchestratorState(TypedDict):
    """Orchestrator state"""
    messages: Annotated[Sequence[BaseMessage], add]
    agent_a_result: str
    agent_b_result: str

llm = ChatOllama(model="llama3.2", temperature=0.3)

# Agent A
def agent_a_process(state: AgentState) -> dict:
    """Agent A processes input"""
    logger.info("🔵 Agent A processing...")
    
    from langchain_core.prompts import ChatPromptTemplate
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are Agent A. Analyze the input."),
        ("human", "{input}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"input": state["input"]})
    
    return {"output": f"[Agent A] {response.content}"}

agent_a_workflow = StateGraph(AgentState)
agent_a_workflow.add_node("process", agent_a_process)
agent_a_workflow.set_entry_point("process")
agent_a_workflow.add_edge("process", END)

agent_a = agent_a_workflow.compile()
logger.info("✅ Agent A compiled")

# Agent B
def agent_b_process(state: AgentState) -> dict:
    """Agent B processes input"""
    logger.info("🟢 Agent B processing...")
    
    from langchain_core.prompts import ChatPromptTemplate
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are Agent B. Enhance the analysis."),
        ("human", "{input}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"input": state["input"]})
    
    return {"output": f"[Agent B] {response.content}"}

agent_b_workflow = StateGraph(AgentState)
agent_b_workflow.add_node("process", agent_b_process)
agent_b_workflow.set_entry_point("process")
agent_b_workflow.add_edge("process", END)

agent_b = agent_b_workflow.compile()
logger.info("✅ Agent B compiled")

# Orchestrator with direct invocation
def invoke_agent_a(state: OrchestratorState) -> dict:
    """Orchestrator invokes Agent A"""
    logger.info("📞 Orchestrator → Agent A (direct invocation)")
    
    task = state["messages"][-1].content
    
    # Direct invocation
    result = agent_a.invoke({
        "messages": [],
        "input": task,
        "output": ""
    })
    
    return {"agent_a_result": result["output"]}

def invoke_agent_b(state: OrchestratorState) -> dict:
    """Orchestrator invokes Agent B with Agent A's result"""
    logger.info("📞 Orchestrator → Agent B (direct invocation)")
    
    # Pass Agent A's result to Agent B
    input_for_b = f"Previous analysis: {state['agent_a_result']}"
    
    # Direct invocation
    result = agent_b.invoke({
        "messages": [],
        "input": input_for_b,
        "output": ""
    })
    
    return {"agent_b_result": result["output"]}

def combine_results(state: OrchestratorState) -> dict:
    """Combine results from both agents"""
    logger.info("🔄 Combining results")
    
    combined = f"{state['agent_a_result']}\n\n{state['agent_b_result']}"
    
    return {"messages": [AIMessage(content=combined)]}

# Build orchestrator
orchestrator_workflow = StateGraph(OrchestratorState)
orchestrator_workflow.add_node("agent_a", invoke_agent_a)
orchestrator_workflow.add_node("agent_b", invoke_agent_b)
orchestrator_workflow.add_node("combine", combine_results)

orchestrator_workflow.set_entry_point("agent_a")
orchestrator_workflow.add_edge("agent_a", "agent_b")
orchestrator_workflow.add_edge("agent_b", "combine")
orchestrator_workflow.add_edge("combine", END)

direct_invocation_system = orchestrator_workflow.compile()

# Test
def test_direct_invocation(task: str):
    """Test direct invocation pattern"""
    
    result = direct_invocation_system.invoke({
        "messages": [HumanMessage(content=task)],
        "agent_a_result": "",
        "agent_b_result": ""
    })
    
    print(f"\n{'='*60}")
    print(f"PATTERN 1: DIRECT INVOCATION")
    print(f"{'='*60}")
    print(f"Task: {task}")
    print(f"\nAgent A Result:\n{result['agent_a_result'][:150]}...")
    print(f"\nAgent B Result:\n{result['agent_b_result'][:150]}...")
    print(f"\nCombined:\n{result['messages'][-1].content[:200]}...")

if __name__ == "__main__":
    test_direct_invocation("Analyze the benefits of cloud computing")
```

---

## 📨 Part 2: Message Passing Pattern

### Asynchronous Communication via Message Queue

```python
from typing import TypedDict, Annotated, Sequence, List, Dict
from operator import add
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from dataclasses import dataclass, field
from collections import deque
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== MESSAGE PASSING PATTERN ====================

@dataclass
class Message:
    """Message passed between agents"""
    sender: str
    recipient: str
    content: str
    message_type: str = "request"
    timestamp: float = field(default_factory=lambda: __import__('time').time())

class MessageQueue:
    """Simple message queue for agent communication"""
    
    def __init__(self):
        self.queues: Dict[str, deque] = {}
    
    def send(self, message: Message):
        """Send message to recipient's queue"""
        if message.recipient not in self.queues:
            self.queues[message.recipient] = deque()
        
        self.queues[message.recipient].append(message)
        logger.info(f"📨 {message.sender} → {message.recipient}: {message.content[:50]}...")
    
    def receive(self, agent_name: str) -> Message | None:
        """Receive message from queue"""
        if agent_name not in self.queues or not self.queues[agent_name]:
            return None
        
        message = self.queues[agent_name].popleft()
        logger.info(f"📬 {agent_name} received message from {message.sender}")
        return message
    
    def has_messages(self, agent_name: str) -> bool:
        """Check if agent has pending messages"""
        return agent_name in self.queues and len(self.queues[agent_name]) > 0

# Global message queue
message_queue = MessageQueue()

# ==================== AGENTS WITH MESSAGE PASSING ====================

class MessageAgentState(TypedDict):
    """Agent state with message handling"""
    agent_name: str
    messages: Annotated[Sequence[BaseMessage], add]
    inbox: List[Message]
    outbox: Annotated[List[Message], add]
    processed_count: Annotated[int, add]

llm = ChatOllama(model="llama3.2", temperature=0.3)

# Processor Agent
def processor_agent_check_inbox(state: MessageAgentState) -> dict:
    """Check for incoming messages"""
    agent_name = state["agent_name"]
    
    inbox = []
    while message_queue.has_messages(agent_name):
        msg = message_queue.receive(agent_name)
        if msg:
            inbox.append(msg)
    
    logger.info(f"🔍 {agent_name}: Found {len(inbox)} messages in inbox")
    
    return {"inbox": inbox}

def processor_agent_process(state: MessageAgentState) -> dict:
    """Process messages from inbox"""
    agent_name = state["agent_name"]
    inbox = state["inbox"]
    
    if not inbox:
        logger.info(f"⏭️  {agent_name}: No messages to process")
        return {}
    
    logger.info(f"⚙️  {agent_name}: Processing {len(inbox)} messages")
    
    outbox = []
    
    for msg in inbox:
        logger.info(f"🔧 {agent_name}: Processing message from {msg.sender}")
        
        # Process the message
        from langchain_core.prompts import ChatPromptTemplate
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"You are {agent_name}. Process this message and create a response."),
            ("human", "{content}")
        ])
        
        chain = prompt | llm
        response = chain.invoke({"content": msg.content})
        
        # Create response message
        response_msg = Message(
            sender=agent_name,
            recipient=msg.sender,
            content=response.content,
            message_type="response"
        )
        
        outbox.append(response_msg)
    
    return {
        "outbox": outbox,
        "processed_count": len(inbox)
    }

def processor_agent_send(state: MessageAgentState) -> dict:
    """Send messages from outbox"""
    agent_name = state["agent_name"]
    outbox = state["outbox"]
    
    for msg in outbox:
        message_queue.send(msg)
    
    logger.info(f"📤 {agent_name}: Sent {len(outbox)} messages")
    
    return {}

# Build processor agent
processor_workflow = StateGraph(MessageAgentState)
processor_workflow.add_node("check_inbox", processor_agent_check_inbox)
processor_workflow.add_node("process", processor_agent_process)
processor_workflow.add_node("send", processor_agent_send)

processor_workflow.set_entry_point("check_inbox")
processor_workflow.add_edge("check_inbox", "process")
processor_workflow.add_edge("process", "send")
processor_workflow.add_edge("send", END)

processor_agent = processor_workflow.compile()

# ==================== COORDINATOR ====================

class CoordinatorState(TypedDict):
    """Coordinator state"""
    messages: Annotated[Sequence[BaseMessage], add]
    task: str
    agents: List[str]
    results: Dict[str, str]

def coordinator_distribute(state: CoordinatorState) -> dict:
    """Distribute task to agents"""
    task = state["task"]
    agents = state["agents"]
    
    logger.info(f"📢 Coordinator: Distributing task to {len(agents)} agents")
    
    # Send task to each agent
    for agent_name in agents:
        msg = Message(
            sender="coordinator",
            recipient=agent_name,
            content=task,
            message_type="task"
        )
        message_queue.send(msg)
    
    return {}

def coordinator_invoke_agents(state: CoordinatorState) -> dict:
    """Invoke agents to process their messages"""
    agents = state["agents"]
    
    logger.info(f"🚀 Coordinator: Invoking {len(agents)} agents")
    
    for agent_name in agents:
        # Invoke agent to process messages
        processor_agent.invoke({
            "agent_name": agent_name,
            "messages": [],
            "inbox": [],
            "outbox": [],
            "processed_count": 0
        })
    
    return {}

def coordinator_collect(state: CoordinatorState) -> dict:
    """Collect responses from agents"""
    logger.info("📥 Coordinator: Collecting responses")
    
    results = {}
    
    # Check coordinator's inbox for responses
    while message_queue.has_messages("coordinator"):
        msg = message_queue.receive("coordinator")
        if msg:
            results[msg.sender] = msg.content
    
    logger.info(f"✅ Coordinator: Collected {len(results)} responses")
    
    # Combine results
    combined = "\n\n".join([
        f"[{agent}]:\n{content[:150]}..."
        for agent, content in results.items()
    ])
    
    return {
        "results": results,
        "messages": [AIMessage(content=combined)]
    }

# Build coordinator
coordinator_workflow = StateGraph(CoordinatorState)
coordinator_workflow.add_node("distribute", coordinator_distribute)
coordinator_workflow.add_node("invoke", coordinator_invoke_agents)
coordinator_workflow.add_node("collect", coordinator_collect)

coordinator_workflow.set_entry_point("distribute")
coordinator_workflow.add_edge("distribute", "invoke")
coordinator_workflow.add_edge("invoke", "collect")
coordinator_workflow.add_edge("collect", END)

message_passing_system = coordinator_workflow.compile()

# Test
def test_message_passing(task: str):
    """Test message passing pattern"""
    
    result = message_passing_system.invoke({
        "messages": [HumanMessage(content=task)],
        "task": task,
        "agents": ["agent_1", "agent_2", "agent_3"],
        "results": {}
    })
    
    print(f"\n{'='*60}")
    print(f"PATTERN 2: MESSAGE PASSING")
    print(f"{'='*60}")
    print(f"Task: {task}")
    print(f"Agents involved: {len(result['results'])}")
    print(f"\nResults:\n{result['messages'][-1].content}")

if __name__ == "__main__":
    test_message_passing("Analyze the impact of AI on different industries")
```

---

## 🗂️ Part 3: State Sharing vs Isolated State

### Comparing Different State Management Approaches

```python
from typing import TypedDict, Annotated, Sequence, Dict, Any
from operator import add
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== STATE MANAGEMENT PATTERNS ====================

llm = ChatOllama(model="llama3.2", temperature=0.3)

# ==================== APPROACH 1: SHARED STATE ====================

class SharedState(TypedDict):
    """Single shared state across all agents"""
    messages: Annotated[Sequence[BaseMessage], add]
    task: str
    shared_data: Dict[str, Any]  # All agents read/write here
    agent_a_done: bool
    agent_b_done: bool

def shared_agent_a(state: SharedState) -> dict:
    """Agent A with shared state"""
    logger.info("🔵 Agent A (Shared State): Processing")
    
    from langchain_core.prompts import ChatPromptTemplate
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are Agent A. Analyze the task."),
        ("human", "{task}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"task": state["task"]})
    
    # Write to shared state
    shared_data = state.get("shared_data", {})
    shared_data["agent_a_analysis"] = response.content
    
    logger.info("📝 Agent A: Wrote to shared state")
    
    return {
        "shared_data": shared_data,
        "agent_a_done": True
    }

def shared_agent_b(state: SharedState) -> dict:
    """Agent B with shared state"""
    logger.info("🟢 Agent B (Shared State): Processing")
    
    # Read from shared state
    shared_data = state["shared_data"]
    agent_a_analysis = shared_data.get("agent_a_analysis", "No analysis from Agent A")
    
    logger.info("📖 Agent B: Read from shared state")
    
    from langchain_core.prompts import ChatPromptTemplate
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are Agent B. Build on Agent A's analysis."),
        ("human", """Agent A's analysis:
{analysis}

Enhance this analysis:""")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"analysis": agent_a_analysis})
    
    # Write to shared state
    shared_data["agent_b_enhancement"] = response.content
    
    logger.info("📝 Agent B: Wrote to shared state")
    
    return {
        "shared_data": shared_data,
        "agent_b_done": True
    }

def shared_finalize(state: SharedState) -> dict:
    """Finalize with shared state"""
    shared_data = state["shared_data"]
    
    combined = f"""Agent A Analysis:
{shared_data.get('agent_a_analysis', 'None')}

Agent B Enhancement:
{shared_data.get('agent_b_enhancement', 'None')}"""
    
    return {"messages": [AIMessage(content=combined)]}

# Build shared state system
shared_workflow = StateGraph(SharedState)
shared_workflow.add_node("agent_a", shared_agent_a)
shared_workflow.add_node("agent_b", shared_agent_b)
shared_workflow.add_node("finalize", shared_finalize)

shared_workflow.set_entry_point("agent_a")
shared_workflow.add_edge("agent_a", "agent_b")
shared_workflow.add_edge("agent_b", "finalize")
shared_workflow.add_edge("finalize", END)

shared_state_system = shared_workflow.compile()

# ==================== APPROACH 2: ISOLATED STATE ====================

class AgentOwnState(TypedDict):
    """Each agent has its own state"""
    messages: Annotated[Sequence[BaseMessage], add]
    input: str
    output: str
    internal_data: Dict[str, Any]

class IsolatedOrchestratorState(TypedDict):
    """Orchestrator manages agent interactions"""
    messages: Annotated[Sequence[BaseMessage], add]
    task: str
    agent_a_output: str
    agent_b_output: str

# Agent A with isolated state
def isolated_agent_a_process(state: AgentOwnState) -> dict:
    """Agent A with isolated state"""
    logger.info("🔵 Agent A (Isolated): Processing")
    
    from langchain_core.prompts import ChatPromptTemplate
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are Agent A. Analyze independently."),
        ("human", "{input}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"input": state["input"]})
    
    # Agent's own state
    internal = {"processed": True, "timestamp": "2024-01-01"}
    
    logger.info("🔒 Agent A: Using isolated state")
    
    return {
        "output": response.content,
        "internal_data": internal
    }

agent_a_isolated_workflow = StateGraph(AgentOwnState)
agent_a_isolated_workflow.add_node("process", isolated_agent_a_process)
agent_a_isolated_workflow.set_entry_point("process")
agent_a_isolated_workflow.add_edge("process", END)

agent_a_isolated = agent_a_isolated_workflow.compile()

# Agent B with isolated state
def isolated_agent_b_process(state: AgentOwnState) -> dict:
    """Agent B with isolated state"""
    logger.info("🟢 Agent B (Isolated): Processing")
    
    from langchain_core.prompts import ChatPromptTemplate
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are Agent B. Enhance independently."),
        ("human", "{input}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"input": state["input"]})
    
    # Agent's own state
    internal = {"enhanced": True, "quality_score": 9.0}
    
    logger.info("🔒 Agent B: Using isolated state")
    
    return {
        "output": response.content,
        "internal_data": internal
    }

agent_b_isolated_workflow = StateGraph(AgentOwnState)
agent_b_isolated_workflow.add_node("process", isolated_agent_b_process)
agent_b_isolated_workflow.set_entry_point("process")
agent_b_isolated_workflow.add_edge("process", END)

agent_b_isolated = agent_b_isolated_workflow.compile()

# Orchestrator coordinates isolated agents
def invoke_isolated_agent_a(state: IsolatedOrchestratorState) -> dict:
    """Invoke Agent A (isolated state)"""
    logger.info("📞 Orchestrator → Agent A (isolated)")
    
    result = agent_a_isolated.invoke({
        "messages": [],
        "input": state["task"],
        "output": "",
        "internal_data": {}
    })
    
    return {"agent_a_output": result["output"]}

def invoke_isolated_agent_b(state: IsolatedOrchestratorState) -> dict:
    """Invoke Agent B (isolated state)"""
    logger.info("📞 Orchestrator → Agent B (isolated)")
    
    # Pass Agent A's output as input
    result = agent_b_isolated.invoke({
        "messages": [],
        "input": f"Previous: {state['agent_a_output']}",
        "output": "",
        "internal_data": {}
    })
    
    return {"agent_b_output": result["output"]}

def isolated_finalize(state: IsolatedOrchestratorState) -> dict:
    """Finalize isolated results"""
    combined = f"""Agent A:
{state['agent_a_output']}

Agent B:
{state['agent_b_output']}"""
    
    return {"messages": [AIMessage(content=combined)]}

# Build isolated system
isolated_workflow = StateGraph(IsolatedOrchestratorState)
isolated_workflow.add_node("agent_a", invoke_isolated_agent_a)
isolated_workflow.add_node("agent_b", invoke_isolated_agent_b)
isolated_workflow.add_node("finalize", isolated_finalize)

isolated_workflow.set_entry_point("agent_a")
isolated_workflow.add_edge("agent_a", "agent_b")
isolated_workflow.add_edge("agent_b", "finalize")
isolated_workflow.add_edge("finalize", END)

isolated_state_system = isolated_workflow.compile()

# ==================== COMPARISON ====================

def compare_state_approaches(task: str):
    """Compare shared vs isolated state"""
    
    print(f"\n{'='*60}")
    print(f"STATE MANAGEMENT COMPARISON")
    print(f"{'='*60}")
    print(f"Task: {task}\n")
    
    # Test shared state
    print("--- APPROACH 1: SHARED STATE ---")
    result_shared = shared_state_system.invoke({
        "messages": [HumanMessage(content=task)],
        "task": task,
        "shared_data": {},
        "agent_a_done": False,
        "agent_b_done": False
    })
    print(f"Shared data keys: {list(result_shared['shared_data'].keys())}")
    print(f"Result: {result_shared['messages'][-1].content[:200]}...\n")
    
    # Test isolated state
    print("--- APPROACH 2: ISOLATED STATE ---")
    result_isolated = isolated_state_system.invoke({
        "messages": [HumanMessage(content=task)],
        "task": task,
        "agent_a_output": "",
        "agent_b_output": ""
    })
    print(f"Agent A output length: {len(result_isolated['agent_a_output'])}")
    print(f"Agent B output length: {len(result_isolated['agent_b_output'])}")
    print(f"Result: {result_isolated['messages'][-1].content[:200]}...")

if __name__ == "__main__":
    compare_state_approaches("Analyze cloud computing benefits")
```

---

## 🔄 Part 4: Coordination Mechanisms

### Different Ways Agents Coordinate

```python
from typing import TypedDict, Annotated, Sequence, List, Literal
from operator import add
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== COORDINATION MECHANISMS ====================

llm = ChatOllama(model="llama3.2", temperature=0.3)

# ==================== MECHANISM 1: CENTRALIZED (Supervisor) ====================

class CentralizedState(TypedDict):
    """State for centralized coordination"""
    messages: Annotated[Sequence[BaseMessage], add]
    task: str
    assigned_agent: str
    agent_results: Annotated[List[dict], add]
    supervisor_decision: str

# Worker agents (simple)
class WorkerState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add]
    task: str
    result: str

def worker_a_process(state: WorkerState) -> dict:
    """Worker A"""
    logger.info("👷 Worker A: Executing task")
    
    from langchain_core.prompts import ChatPromptTemplate
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are Worker A specializing in data analysis."),
        ("human", "{task}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"task": state["task"]})
    
    return {"result": response.content}

worker_a_workflow = StateGraph(WorkerState)
worker_a_workflow.add_node("work", worker_a_process)
worker_a_workflow.set_entry_point("work")
worker_a_workflow.add_edge("work", END)
worker_a = worker_a_workflow.compile()

def worker_b_process(state: WorkerState) -> dict:
    """Worker B"""
    logger.info("👷 Worker B: Executing task")
    
    from langchain_core.prompts import ChatPromptTemplate
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are Worker B specializing in visualization."),
        ("human", "{task}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"task": state["task"]})
    
    return {"result": response.content}

worker_b_workflow = StateGraph(WorkerState)
worker_b_workflow.add_node("work", worker_b_process)
worker_b_workflow.set_entry_point("work")
worker_b_workflow.add_edge("work", END)
worker_b = worker_b_workflow.compile()

# Supervisor
def supervisor_assign(state: CentralizedState) -> dict:
    """Supervisor assigns task to worker"""
    task = state["task"]
    
    logger.info("👔 Supervisor: Analyzing task and assigning")
    
    # Simple assignment logic
    if "analyze" in task.lower() or "data" in task.lower():
        assigned = "worker_a"
    else:
        assigned = "worker_b"
    
    logger.info(f"👔 Supervisor: Assigned to {assigned}")
    
    return {
        "assigned_agent": assigned,
        "supervisor_decision": f"Task assigned to {assigned}"
    }

def invoke_worker_a(state: CentralizedState) -> dict:
    """Supervisor invokes Worker A"""
    logger.info("👔 Supervisor → Worker A")
    
    result = worker_a.invoke({
        "messages": [],
        "task": state["task"],
        "result": ""
    })
    
    return {
        "agent_results": [{
            "agent": "worker_a",
            "result": result["result"]
        }]
    }

def invoke_worker_b(state: CentralizedState) -> dict:
    """Supervisor invokes Worker B"""
    logger.info("👔 Supervisor → Worker B")
    
    result = worker_b.invoke({
        "messages": [],
        "task": state["task"],
        "result": ""
    })
    
    return {
        "agent_results": [{
            "agent": "worker_b",
            "result": result["result"]
        }]
    }

def supervisor_finalize(state: CentralizedState) -> dict:
    """Supervisor finalizes"""
    result = state["agent_results"][-1]
    
    output = f"""Supervisor Decision: {state['supervisor_decision']}

Worker Result:
{result['result']}"""
    
    return {"messages": [AIMessage(content=output)]}

def route_to_worker(state: CentralizedState) -> str:
    """Route to assigned worker"""
    return state["assigned_agent"]

# Build centralized system
centralized_workflow = StateGraph(CentralizedState)
centralized_workflow.add_node("supervisor", supervisor_assign)
centralized_workflow.add_node("worker_a", invoke_worker_a)
centralized_workflow.add_node("worker_b", invoke_worker_b)
centralized_workflow.add_node("finalize", supervisor_finalize)

centralized_workflow.set_entry_point("supervisor")
centralized_workflow.add_conditional_edges(
    "supervisor",
    route_to_worker,
    {
        "worker_a": "worker_a",
        "worker_b": "worker_b"
    }
)
centralized_workflow.add_edge("worker_a", "finalize")
centralized_workflow.add_edge("worker_b", "finalize")
centralized_workflow.add_edge("finalize", END)

centralized_system = centralized_workflow.compile()

# ==================== MECHANISM 2: DECENTRALIZED (Peer-to-Peer) ====================

class DecentralizedState(TypedDict):
    """State for decentralized coordination"""
    messages: Annotated[Sequence[BaseMessage], add]
    task: str
    peer_votes: Dict[str, str]  # Each peer votes
    consensus: str

class PeerState(TypedDict):
    """Individual peer state"""
    messages: Annotated[Sequence[BaseMessage], add]
    task: str
    vote: str

def peer_1_vote(state: PeerState) -> dict:
    """Peer 1 votes on approach"""
    logger.info("🤝 Peer 1: Voting")
    
    from langchain_core.prompts import ChatPromptTemplate
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Vote on the best approach: 'statistical' or 'machine_learning'"),
        ("human", "{task}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"task": state["task"]})
    
    # Extract vote
    vote = "statistical" if "statistical" in response.content.lower() else "machine_learning"
    
    logger.info(f"🤝 Peer 1: Voted for {vote}")
    
    return {"vote": vote}

peer_1_workflow = StateGraph(PeerState)
peer_1_workflow.add_node("vote", peer_1_vote)
peer_1_workflow.set_entry_point("vote")
peer_1_workflow.add_edge("vote", END)
peer_1 = peer_1_workflow.compile()

def peer_2_vote(state: PeerState) -> dict:
    """Peer 2 votes on approach"""
    logger.info("🤝 Peer 2: Voting")
    
    from langchain_core.prompts import ChatPromptTemplate
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Vote on the best approach: 'statistical' or 'machine_learning'"),
        ("human", "{task}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"task": state["task"]})
    
    vote = "statistical" if "statistical" in response.content.lower() else "machine_learning"
    
    logger.info(f"🤝 Peer 2: Voted for {vote}")
    
    return {"vote": vote}

peer_2_workflow = StateGraph(PeerState)
peer_2_workflow.add_node("vote", peer_2_vote)
peer_2_workflow.set_entry_point("vote")
peer_2_workflow.add_edge("vote", END)
peer_2 = peer_2_workflow.compile()

# Consensus builder
def collect_votes(state: DecentralizedState) -> dict:
    """Collect votes from all peers"""
    task = state["task"]
    
    logger.info("🗳️  Collecting votes from peers")
    
    # Invoke peer 1
    result_1 = peer_1.invoke({
        "messages": [],
        "task": task,
        "vote": ""
    })
    
    # Invoke peer 2
    result_2 = peer_2.invoke({
        "messages": [],
        "task": task,
        "vote": ""
    })
    
    votes = {
        "peer_1": result_1["vote"],
        "peer_2": result_2["vote"]
    }
    
    logger.info(f"🗳️  Votes collected: {votes}")
    
    return {"peer_votes": votes}

def build_consensus(state: DecentralizedState) -> dict:
    """Build consensus from votes"""
    votes = state["peer_votes"]
    
    # Count votes
    vote_counts = {}
    for vote in votes.values():
        vote_counts[vote] = vote_counts.get(vote, 0) + 1
    
    # Find consensus (majority)
    consensus = max(vote_counts.items(), key=lambda x: x[1])[0]
    
    logger.info(f"✅ Consensus reached: {consensus}")
    
    output = f"""Peer Votes:
{votes}

Consensus: {consensus}"""
    
    return {
        "consensus": consensus,
        "messages": [AIMessage(content=output)]
    }

# Build decentralized system
decentralized_workflow = StateGraph(DecentralizedState)
decentralized_workflow.add_node("collect", collect_votes)
decentralized_workflow.add_node("consensus", build_consensus)

decentralized_workflow.set_entry_point("collect")
decentralized_workflow.add_edge("collect", "consensus")
decentralized_workflow.add_edge("consensus", END)

decentralized_system = decentralized_workflow.compile()

# ==================== COMPARISON ====================

def compare_coordination(task: str):
    """Compare coordination mechanisms"""
    
    print(f"\n{'='*60}")
    print(f"COORDINATION MECHANISMS COMPARISON")
    print(f"{'='*60}")
    print(f"Task: {task}\n")
    
    # Test centralized
    print("--- MECHANISM 1: CENTRALIZED (Supervisor) ---")
    result_central = centralized_system.invoke({
        "messages": [HumanMessage(content=task)],
        "task": task,
        "assigned_agent": "",
        "agent_results": [],
        "supervisor_decision": ""
    })
    print(f"Result: {result_central['messages'][-1].content[:200]}...\n")
    
    # Test decentralized
    print("--- MECHANISM 2: DECENTRALIZED (Peer-to-Peer) ---")
    result_decentral = decentralized_system.invoke({
        "messages": [HumanMessage(content=task)],
        "task": task,
        "peer_votes": {},
        "consensus": ""
    })
    print(f"Result: {result_decentral['messages'][-1].content}")

if __name__ == "__main__":
    compare_coordination("Analyze customer data patterns")
```

---

## 📊 Part 5: Comparison and Best Practices

### When to Use Each Pattern

```python
# ==================== DECISION MATRIX ====================

from dataclasses import dataclass

@dataclass
class SystemRequirements:
    """Requirements for choosing multi-agent pattern"""
    agents_need_independence: bool
    agents_share_state: bool
    realtime_coordination: bool
    agents_scale_independently: bool
    different_deployment_locations: bool

def recommend_pattern(reqs: SystemRequirements) -> str:
    """Recommend communication pattern based on requirements"""
    
    if reqs.different_deployment_locations:
        return "DISTRIBUTED (Message Passing + Service Mesh)"
    
    elif reqs.agents_scale_independently:
        return "MESSAGE PASSING with Queue"
    
    elif reqs.agents_need_independence and not reqs.agents_share_state:
        return "ISOLATED STATE with Direct Invocation"
    
    elif reqs.agents_share_state and reqs.realtime_coordination:
        return "SHARED STATE"
    
    else:
        return "DIRECT INVOCATION (Simplest)"

# Examples
examples = [
    SystemRequirements(
        agents_need_independence=True,
        agents_share_state=False,
        realtime_coordination=False,
        agents_scale_independently=False,
        different_deployment_locations=False
    ),
    SystemRequirements(
        agents_need_independence=True,
        agents_share_state=False,
        realtime_coordination=False,
        agents_scale_independently=True,
        different_deployment_locations=False
    ),
    SystemRequirements(
        agents_need_independence=False,
        agents_share_state=True,
        realtime_coordination=True,
        agents_scale_independently=False,
        different_deployment_locations=False
    ),
]

print("\n" + "="*60)
print("PATTERN RECOMMENDATION GUIDE")
print("="*60)

for i, req in enumerate(examples, 1):
    print(f"\nScenario {i}:")
    print(f"  Independence: {req.agents_need_independence}")
    print(f"  Shared State: {req.agents_share_state}")
    print(f"  Realtime: {req.realtime_coordination}")
    print(f"  Scale Independently: {req.agents_scale_independently}")
    print(f"  → Recommended: {recommend_pattern(req)}")
```

---

## 📋 Summary Table

| Pattern | Communication | State | Coordination | Best For |
|---------|--------------|-------|--------------|----------|
| **Direct Invocation** | Synchronous | Isolated | Centralized | Simple workflows |
| **Message Passing** | Asynchronous | Isolated | Decentralized | Scalable systems |
| **Shared State** | Direct access | Shared | Both | Tight coupling needed |
| **Subgraphs** | Parent→Child | Both | Hierarchical | Modular agents |

---

## 🧠 Key Concepts to Remember

1. **Direct Invocation** = Parent calls child synchronously
2. **Message Passing** = Async communication via queues
3. **Shared State** = All agents read/write same state
4. **Isolated State** = Each agent has own state
5. **Centralized Coordination** = Supervisor assigns tasks
6. **Decentralized Coordination** = Peers vote/consensus
7. **Subgraphs** = Each agent is a compiled graph
8. **Choose pattern based on requirements**, not preference

---

## 🚀 What's Next?

In **Chapter 13**, we'll explore:
- **Advanced Multi-Agent Patterns**
- Hierarchical multi-agent systems
- Agent collaboration protocols
- Conflict resolution mechanisms
- Production deployment strategies
- Monitoring multi-agent systems

---

## ✅ Chapter 12 Complete!

**You now understand:**
- ✅ Why multi-agent systems are needed
- ✅ Direct invocation pattern
- ✅ Message passing with queues
- ✅ Shared vs isolated state trade-offs
- ✅ Centralized vs decentralized coordination
- ✅ When to use each pattern
- ✅ How to implement true multi-agent with subgraphs
- ✅ Best practices for choosing patterns

**Ready for Chapter 13?** Just say "Continue to Chapter 13" or ask any questions!