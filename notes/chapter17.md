# Chapter 17: Human-in-the-Loop Systems

## 👥 Introduction: Why Human-in-the-Loop?

**Human-in-the-Loop (HITL)** systems combine AI automation with human judgment. Instead of fully autonomous agents, these systems:
- Pause for human review
- Request approval for critical actions
- Accept human corrections
- Learn from human feedback

Think of it like **autopilot with a pilot** - the system handles routine tasks, but humans maintain control and can intervene when needed.

---

## 🛑 Part 1: Interrupt Patterns and Breakpoints

### Theory: Interrupts in LangGraph

#### What Are Interrupts?

**Interrupts** are points where LangGraph **pauses execution** and waits for external input before continuing.

```
Normal Flow:
┌─────┐   ┌─────┐   ┌─────┐
│  A  │──→│  B  │──→│  C  │
└─────┘   └─────┘   └─────┘
(Runs straight through)

With Interrupt:
┌─────┐   ┌─────┐   ⏸️    ┌─────┐
│  A  │──→│  B  │──→ PAUSE ──→│  C  │
└─────┘   └─────┘        └─────┘
              ↓
        Wait for human
```

#### Why Use Interrupts?

**1. Critical Decisions:**
```
Before: AI makes all decisions autonomously
With Interrupts: Human approves critical actions
Example: "About to delete 1000 records - approve?"
```

**2. Quality Control:**
```
Before: AI output goes directly to production
With Interrupts: Human reviews before publishing
Example: Pause before sending email to 10,000 customers
```

**3. Error Recovery:**
```
Before: System fails and stops
With Interrupts: Human can correct and resume
Example: "API key invalid - provide correct key to continue"
```

**4. Complex Judgments:**
```
Before: AI guesses at ambiguous situations
With Interrupts: Human resolves ambiguity
Example: "Two candidate solutions - which should we use?"
```

#### Types of Interrupts

**1. Approval Interrupts**
```python
# Pause before critical action
interrupt_before=["delete_data"]
→ System pauses BEFORE delete_data node
→ Human approves/rejects
→ System continues or stops
```

**2. Review Interrupts**
```python
# Pause after node for review
interrupt_after=["generate_report"]
→ System pauses AFTER generate_report
→ Human reviews output
→ Can edit and resume
```

**3. Conditional Interrupts**
```python
# Pause based on runtime conditions
if state["confidence"] < 0.8:
    return Command(goto="human_review")
→ Only interrupt when confidence is low
```

**4. Manual Breakpoints**
```python
# Developer-set breakpoints for debugging
# Similar to debugger breakpoints
interrupt_before=["suspicious_node"]
```

#### LangGraph Interrupt Mechanism

**Setting Interrupts at Compile Time:**
```python
graph = workflow.compile(
    interrupt_before=["critical_node"],  # Pause before this node
    interrupt_after=["review_node"]       # Pause after this node
)
```

**What Happens at Interrupt:**
1. Graph pauses execution
2. Returns current state to caller
3. Waits for `invoke()` or `stream()` to be called again
4. Resumes from where it paused

**Resume Pattern:**
```python
# Initial execution - runs until interrupt
result = graph.invoke(initial_state)

# System paused at interrupt
# Human reviews: result["current_state"]

# Resume execution
final_result = graph.invoke(result)
# Continues from where it paused
```

#### Checkpointing and State Persistence

**Problem:** What if server restarts while waiting for human?

**Solution:** Checkpointing - persist state to storage.

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()

graph = workflow.compile(
    checkpointer=checkpointer,
    interrupt_before=["approval_needed"]
)

# First execution
config = {"configurable": {"thread_id": "user-123"}}
result = graph.invoke(initial_state, config=config)

# [Later, after server restart]
# Resume with same thread_id
final = graph.invoke(None, config=config)
# Resumes from checkpoint
```

**Without checkpointing:** State lost on restart
**With checkpointing:** State persisted, can resume anytime

#### When to Use Interrupts

✅ **Use Interrupts When:**
- Actions have significant consequences (delete, spend money, send messages)
- Output needs human review (legal, medical, financial content)
- Ambiguous situations require human judgment
- Building trust in AI system (show human is in control)
- Compliance requires human oversight

❌ **Don't Use When:**
- Decisions are low-stakes and reversible
- High-frequency operations (interrupts every second)
- Fully autonomous operation is required
- Human won't be available to respond

---

### Implementation: Interrupt Patterns

```python
from typing import TypedDict, Annotated, Sequence, List, Optional
from operator import add
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== INTERRUPT PATTERNS ====================

class InterruptState(TypedDict):
    """State for interrupt demonstrations"""
    messages: Annotated[Sequence[BaseMessage], add]
    task_description: str
    analysis_result: str
    action_to_take: str
    approval_status: str
    execution_log: Annotated[List[str], add]

# ==================== WORKFLOW NODES ====================

def analyze_task(state: InterruptState) -> dict:
    """Analyze the task"""
    logger.info("📊 Analyzing task...")
    
    task = state["task_description"]
    
    # Simulate analysis
    if "delete" in task.lower():
        analysis = "HIGH RISK: Destructive operation detected"
        action = "delete_records"
    elif "email" in task.lower():
        analysis = "MEDIUM RISK: Communication to external parties"
        action = "send_email"
    else:
        analysis = "LOW RISK: Safe operation"
        action = "safe_operation"
    
    return {
        "analysis_result": analysis,
        "action_to_take": action,
        "execution_log": ["Task analyzed"],
        "messages": [AIMessage(content=f"Analysis: {analysis}")]
    }

def request_approval(state: InterruptState) -> dict:
    """Request approval node (interrupts BEFORE execution)"""
    logger.info("🛑 Requesting approval...")
    
    return {
        "execution_log": ["Approval requested"],
        "messages": [AIMessage(content=f"Requesting approval for: {state['action_to_take']}")]
    }

def execute_action(state: InterruptState) -> dict:
    """Execute the approved action"""
    logger.info("⚡ Executing action...")
    
    if state.get("approval_status") != "approved":
        return {
            "execution_log": ["Action rejected by human"],
            "messages": [AIMessage(content="Action cancelled - no approval")]
        }
    
    action = state["action_to_take"]
    
    # Simulate execution
    result = f"Successfully executed: {action}"
    
    return {
        "execution_log": [f"Executed {action}"],
        "messages": [AIMessage(content=result)]
    }

def finalize_workflow(state: InterruptState) -> dict:
    """Finalize the workflow"""
    logger.info("✅ Finalizing...")
    
    log = " → ".join(state["execution_log"])
    
    summary = f"""WORKFLOW COMPLETE

Task: {state['task_description']}
Analysis: {state['analysis_result']}
Action: {state['action_to_take']}
Approval: {state.get('approval_status', 'pending')}

Execution Log: {log}"""
    
    return {
        "execution_log": ["Workflow finalized"],
        "messages": [AIMessage(content=summary)]
    }

# ==================== BUILD WORKFLOW WITH INTERRUPTS ====================

# Create checkpointer for state persistence
checkpointer = MemorySaver()

# Build workflow
interrupt_workflow = StateGraph(InterruptState)

interrupt_workflow.add_node("analyze", analyze_task)
interrupt_workflow.add_node("request_approval", request_approval)
interrupt_workflow.add_node("execute", execute_action)
interrupt_workflow.add_node("finalize", finalize_workflow)

interrupt_workflow.set_entry_point("analyze")
interrupt_workflow.add_edge("analyze", "request_approval")
interrupt_workflow.add_edge("request_approval", "execute")
interrupt_workflow.add_edge("execute", "finalize")
interrupt_workflow.add_edge("finalize", END)

# Compile with interrupt BEFORE execute node
interrupt_system = interrupt_workflow.compile(
    checkpointer=checkpointer,
    interrupt_before=["execute"]  # Pause before executing action
)

logger.info("✅ Interrupt system compiled with breakpoint before 'execute'")

# ==================== DEMONSTRATION ====================

def demonstrate_interrupt_flow():
    """Demonstrate interrupt and resume flow"""
    
    print("\n" + "="*60)
    print("INTERRUPT PATTERN DEMONSTRATION")
    print("="*60)
    
    # Initial state
    initial_state = {
        "messages": [HumanMessage(content="Start workflow")],
        "task_description": "Delete all old records from database",
        "analysis_result": "",
        "action_to_take": "",
        "approval_status": "",
        "execution_log": []
    }
    
    # Configuration with thread_id for checkpointing
    config = {"configurable": {"thread_id": "demo-123"}}
    
    print("\n--- PHASE 1: Initial Execution (until interrupt) ---")
    print("Invoking workflow...")
    
    # Execute until interrupt
    result = interrupt_system.invoke(initial_state, config=config)
    
    print(f"\n🛑 WORKFLOW PAUSED")
    print(f"Analysis: {result['analysis_result']}")
    print(f"Proposed Action: {result['action_to_take']}")
    print(f"Status: Waiting for human approval...")
    print(f"Execution Log: {' → '.join(result['execution_log'])}")
    
    # Simulate human review
    print("\n--- HUMAN REVIEW ---")
    print("Human reviewing proposed action...")
    print(f"  Task: {result['task_description']}")
    print(f"  Risk Level: {result['analysis_result']}")
    print(f"  Action: {result['action_to_take']}")
    
    # Human decision
    human_decision = input("\nApprove this action? (yes/no): ").strip().lower()
    
    if human_decision == "yes":
        result["approval_status"] = "approved"
        print("✅ Human approved action")
    else:
        result["approval_status"] = "rejected"
        print("❌ Human rejected action")
    
    print("\n--- PHASE 2: Resume Execution ---")
    print("Resuming workflow with human decision...")
    
    # Resume execution with updated state
    final_result = interrupt_system.invoke(result, config=config)
    
    print("\n" + "="*60)
    print("FINAL RESULT")
    print("="*60)
    print(final_result["messages"][-1].content)

# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    demonstrate_interrupt_flow()
```

---

## ✅ Part 2: Approval Workflows

### Theory: Approval Workflows

#### What Are Approval Workflows?

**Approval workflows** are HITL patterns where specific actions require explicit human approval before execution.

```
Standard Workflow:
Generate → Execute → Complete

Approval Workflow:
Generate → Request Approval → Wait → [Approved?] → Execute → Complete
                                         ↓
                                      [Rejected?] → Cancel
```

#### Common Approval Patterns

**Pattern 1: Single Approver**
```
Agent proposes action
→ Single human approves/rejects
→ Execute if approved
```

**Pattern 2: Multi-Level Approval**
```
Agent proposes action
→ Manager approves → VP approves → Execute
→ Any level can reject
```

**Pattern 3: Threshold-Based**
```
if cost < $100:
    Auto-approve
elif cost < $1000:
    Require manager approval
else:
    Require executive approval
```

**Pattern 4: Committee Approval**
```
Multiple reviewers vote
→ Require majority (or unanimous)
→ Execute if threshold met
```

#### Approval Workflow Components

**1. Proposal Generation:**
```python
def generate_proposal(state: State) -> dict:
    """Generate action proposal for approval"""
    
    proposal = {
        "action": "deploy_to_production",
        "details": "Deploy version 2.3.1",
        "risk_level": "medium",
        "impact": "10,000 users",
        "estimated_cost": "$50"
    }
    
    return {"proposal": proposal}
```

**2. Approval Request:**
```python
def request_approval(state: State) -> dict:
    """Format approval request"""
    
    proposal = state["proposal"]
    
    request = f"""
    APPROVAL REQUIRED
    
    Action: {proposal['action']}
    Details: {proposal['details']}
    Risk: {proposal['risk_level']}
    Impact: {proposal['impact']}
    
    Please approve or reject.
    """
    
    return {"approval_request": request}
```

**3. Human Decision:**
```python
# This happens OUTSIDE the graph
# Human reviews and provides decision

human_decision = {
    "approved": True,  # or False
    "approver": "manager@company.com",
    "comments": "Approved - looks good",
    "timestamp": "2026-01-05T10:30:00Z"
}
```

**4. Execution or Cancellation:**
```python
def execute_if_approved(state: State) -> dict:
    """Execute only if approved"""
    
    if not state.get("approval_decision", {}).get("approved"):
        return {
            "status": "cancelled",
            "reason": state["approval_decision"]["comments"]
        }
    
    # Execute the action
    result = execute_action(state["proposal"])
    
    return {
        "status": "executed",
        "result": result
    }
```

#### Approval Metadata

**What to track:**
- **Proposal details**: What is being approved
- **Approver identity**: Who approved
- **Timestamp**: When approved
- **Comments/reasoning**: Why approved/rejected
- **Approval level**: If multi-level
- **Audit trail**: Full history

```python
approval_record = {
    "proposal_id": "prop-123",
    "action": "delete_customer_data",
    "requested_by": "system",
    "requested_at": "2026-01-05T10:00:00Z",
    "approvers": [
        {
            "name": "manager@company.com",
            "decision": "approved",
            "timestamp": "2026-01-05T10:15:00Z",
            "comments": "Customer requested deletion"
        }
    ],
    "final_decision": "approved",
    "executed_at": "2026-01-05T10:16:00Z"
}
```

#### Timeout Handling

**Problem:** What if human never responds?

**Solutions:**

**1. Expiration:**
```python
if time.time() - request_time > timeout:
    return {"decision": "rejected", "reason": "timeout"}
```

**2. Escalation:**
```python
if time.time() - request_time > initial_timeout:
    notify_manager()
    if time.time() - request_time > escalation_timeout:
        auto_reject()
```

**3. Default Action:**
```python
# Some orgs default to approve, others to reject
if timeout_reached:
    return default_decision  # "approved" or "rejected"
```

---

### Implementation: Approval Workflows

```python
from typing import TypedDict, Annotated, Sequence, List, Dict, Optional
from operator import add
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== APPROVAL WORKFLOW ====================

class ApprovalState(TypedDict):
    """State for approval workflows"""
    messages: Annotated[Sequence[BaseMessage], add]
    request_description: str
    proposal: Dict
    approval_request: str
    approval_decision: Optional[Dict]
    execution_result: str
    audit_log: Annotated[List[Dict], add]

# ==================== WORKFLOW NODES ====================

def generate_proposal(state: ApprovalState) -> dict:
    """Generate proposal based on request"""
    logger.info("📝 Generating proposal...")
    
    request = state["request_description"]
    
    # Analyze request and create proposal
    if "deploy" in request.lower():
        proposal = {
            "action": "deploy_to_production",
            "target": "web-app-v2.3.1",
            "risk_level": "medium",
            "impact": "10,000 active users",
            "estimated_downtime": "5 minutes",
            "rollback_plan": "Automated rollback if errors > 1%"
        }
    elif "email" in request.lower():
        proposal = {
            "action": "send_marketing_email",
            "recipients": "50,000 subscribers",
            "risk_level": "low",
            "impact": "Marketing campaign",
            "cost": "$250",
            "schedule": "Tomorrow 9 AM"
        }
    elif "delete" in request.lower():
        proposal = {
            "action": "delete_customer_data",
            "records": "Customer ID: 12345",
            "risk_level": "high",
            "impact": "Permanent data loss",
            "compliance": "GDPR deletion request",
            "backup_status": "Backed up"
        }
    else:
        proposal = {
            "action": "process_request",
            "risk_level": "low",
            "impact": "Standard operation"
        }
    
    audit_entry = {
        "timestamp": datetime.now().isoformat(),
        "event": "proposal_generated",
        "details": proposal
    }
    
    return {
        "proposal": proposal,
        "audit_log": [audit_entry],
        "messages": [AIMessage(content=f"Proposal generated: {proposal['action']}")]
    }

def create_approval_request(state: ApprovalState) -> dict:
    """Create formatted approval request"""
    logger.info("📋 Creating approval request...")
    
    proposal = state["proposal"]
    
    # Format approval request
    request_lines = ["=" * 50]
    request_lines.append("APPROVAL REQUIRED")
    request_lines.append("=" * 50)
    request_lines.append("")
    
    for key, value in proposal.items():
        formatted_key = key.replace("_", " ").title()
        request_lines.append(f"{formatted_key}: {value}")
    
    request_lines.append("")
    request_lines.append("=" * 50)
    request_lines.append("Please review and approve/reject")
    request_lines.append("=" * 50)
    
    approval_request = "\n".join(request_lines)
    
    audit_entry = {
        "timestamp": datetime.now().isoformat(),
        "event": "approval_requested",
        "action": proposal["action"]
    }
    
    return {
        "approval_request": approval_request,
        "audit_log": [audit_entry],
        "messages": [AIMessage(content="Approval request created")]
    }

def wait_for_approval(state: ApprovalState) -> dict:
    """Node that triggers interrupt - waits for approval"""
    logger.info("⏸️  Waiting for human approval...")
    
    return {
        "messages": [AIMessage(content="Waiting for approval decision...")]
    }

def execute_with_approval(state: ApprovalState) -> dict:
    """Execute action if approved"""
    logger.info("⚡ Checking approval and executing...")
    
    decision = state.get("approval_decision")
    
    if not decision:
        result = "ERROR: No approval decision provided"
        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "event": "execution_failed",
            "reason": "no_decision"
        }
    elif decision.get("approved"):
        # Execute the action
        action = state["proposal"]["action"]
        result = f"✅ Successfully executed: {action}"
        
        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "event": "action_executed",
            "action": action,
            "approver": decision.get("approver", "unknown"),
            "comments": decision.get("comments", "")
        }
        
        logger.info(f"✅ Action executed: {action}")
    else:
        result = f"❌ Action cancelled: {decision.get('comments', 'No reason provided')}"
        
        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "event": "action_rejected",
            "action": state["proposal"]["action"],
            "approver": decision.get("approver", "unknown"),
            "comments": decision.get("comments", "")
        }
        
        logger.info(f"❌ Action rejected")
    
    return {
        "execution_result": result,
        "audit_log": [audit_entry],
        "messages": [AIMessage(content=result)]
    }

def generate_report(state: ApprovalState) -> dict:
    """Generate final report"""
    logger.info("📊 Generating report...")
    
    # Format audit log
    audit_lines = []
    for entry in state["audit_log"]:
        timestamp = entry["timestamp"].split("T")[1][:8]  # Extract time
        event = entry["event"].replace("_", " ").title()
        audit_lines.append(f"[{timestamp}] {event}")
        
        if "details" in entry:
            audit_lines.append(f"           Details: {entry['details'].get('action', 'N/A')}")
        if "approver" in entry:
            audit_lines.append(f"           Approver: {entry['approver']}")
        if "comments" in entry:
            audit_lines.append(f"           Comments: {entry['comments']}")
    
    audit_log_text = "\n".join(audit_lines)
    
    report = f"""
APPROVAL WORKFLOW REPORT
{'='*60}

Request: {state['request_description']}

Proposal:
{chr(10).join([f"  {k}: {v}" for k, v in state['proposal'].items()])}

Decision: {state.get('approval_decision', {}).get('approved', 'Pending')}

Result: {state.get('execution_result', 'Not executed')}

Audit Trail:
{audit_log_text}
"""
    
    return {
        "messages": [AIMessage(content=report)]
    }

# ==================== BUILD APPROVAL WORKFLOW ====================

checkpointer = MemorySaver()

approval_workflow = StateGraph(ApprovalState)

approval_workflow.add_node("generate_proposal", generate_proposal)
approval_workflow.add_node("create_request", create_approval_request)
approval_workflow.add_node("wait_approval", wait_for_approval)
approval_workflow.add_node("execute", execute_with_approval)
approval_workflow.add_node("report", generate_report)

approval_workflow.set_entry_point("generate_proposal")
approval_workflow.add_edge("generate_proposal", "create_request")
approval_workflow.add_edge("create_request", "wait_approval")
approval_workflow.add_edge("wait_approval", "execute")
approval_workflow.add_edge("execute", "report")
approval_workflow.add_edge("report", END)

# Compile with interrupt after wait_approval
approval_system = approval_workflow.compile(
    checkpointer=checkpointer,
    interrupt_after=["wait_approval"]
)

logger.info("✅ Approval workflow compiled")

# ==================== DEMONSTRATION ====================

def demonstrate_approval_workflow():
    """Demonstrate approval workflow"""
    
    print("\n" + "="*60)
    print("APPROVAL WORKFLOW DEMONSTRATION")
    print("="*60)
    
    # Test cases
    test_requests = [
        "Deploy new version to production",
        "Delete customer data per GDPR request"
    ]
    
    for i, request in enumerate(test_requests):
        print(f"\n{'='*60}")
        print(f"TEST CASE {i+1}")
        print(f"{'='*60}")
        
        initial_state = {
            "messages": [HumanMessage(content="Start approval workflow")],
            "request_description": request,
            "proposal": {},
            "approval_request": "",
            "approval_decision": None,
            "execution_result": "",
            "audit_log": []
        }
        
        config = {"configurable": {"thread_id": f"approval-{i}"}}
        
        print(f"\nRequest: {request}")
        print("\n--- Phase 1: Generate Proposal ---")
        
        # Execute until interrupt
        result = approval_system.invoke(initial_state, config=config)
        
        print("\n🛑 WORKFLOW PAUSED FOR APPROVAL")
        print(result["approval_request"])
        
        # Simulate human review
        print("\n--- Human Review ---")
        human_input = input("Approve? (yes/no): ").strip().lower()
        
        if human_input == "yes":
            decision = {
                "approved": True,
                "approver": "manager@company.com",
                "comments": "Approved - requirements met",
                "timestamp": datetime.now().isoformat()
            }
            print("✅ Approved by manager@company.com")
        else:
            decision = {
                "approved": False,
                "approver": "manager@company.com",
                "comments": "Rejected - requires more review",
                "timestamp": datetime.now().isoformat()
            }
            print("❌ Rejected by manager@company.com")
        
        # Update state with decision
        result["approval_decision"] = decision
        
        print("\n--- Phase 2: Execute with Decision ---")
        
        # Resume workflow
        final_result = approval_system.invoke(result, config=config)
        
        print(final_result["messages"][-1].content)

# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    demonstrate_approval_workflow()
```

---

## 💬 Part 3: Dynamic User Input

### Theory: Dynamic Input Collection

#### What Is Dynamic User Input?

**Dynamic user input** means collecting information from users **during workflow execution**, not just at the start.

```
Static Input (traditional):
User provides all info → Workflow runs → Result

Dynamic Input:
Workflow starts → Needs info → Ask user → Continue → Needs more → Ask again → Result
```

#### Why Dynamic Input?

**1. Unknown Requirements:**
```
Can't know upfront what questions to ask
Example: Troubleshooting - questions depend on previous answers
```

**2. Progressive Refinement:**
```
Start with basic info, ask for details as needed
Example: Product recommendation - ask preferences iteratively
```

**3. Conditional Information:**
```
Only ask for information when relevant
Example: If "international shipping" → Ask for customs info
```

**4. Interactive Workflows:**
```
User guides the workflow through choices
Example: Guided configuration wizard
```

#### Dynamic Input Patterns

**Pattern 1: Question Loop**
```
while not enough_info:
    question = generate_question(state)
    → Interrupt for user input
    answer = user_provides_answer()
    update_state(answer)
```

**Pattern 2: Conditional Prompts**
```
if state["option_A"]:
    → Ask question X
elif state["option_B"]:
    → Ask question Y
```

**Pattern 3: Clarification Requests**
```
if ambiguous(user_request):
    → Ask for clarification
    → Continue with clarified request
```

**Pattern 4: Multi-Step Forms**
```
Step 1 → Collect basic info → Interrupt
Step 2 → Collect details based on Step 1 → Interrupt
Step 3 → Confirm and proceed
```

#### Input Validation

**Why validate:**
- Prevent invalid data from breaking workflow
- Catch errors early
- Provide immediate feedback

**Validation strategies:**

```python
def validate_input(user_input: str, expected_type: str) -> bool:
    """Validate user input"""
    
    if expected_type == "email":
        return "@" in user_input and "." in user_input
    
    elif expected_type == "number":
        try:
            float(user_input)
            return True
        except:
            return False
    
    elif expected_type == "choice":
        return user_input in ["A", "B", "C"]
    
    return True

# In workflow
while True:
    user_input = get_user_input()
    if validate_input(user_input, "email"):
        break
    else:
        show_error("Invalid email format")
```

#### Input Context

**Provide context with each prompt:**

```python
input_prompt = {
    "question": "What is your budget?",
    "context": "We found 3 options in different price ranges",
    "options": ["$50-100", "$100-200", "$200+"],
    "why_asking": "This helps us narrow down recommendations",
    "default": "$100-200"
}
```

---

### Implementation: Dynamic User Input System

```python
from typing import TypedDict, Annotated, Sequence, List, Dict, Optional, Literal
from operator import add
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== DYNAMIC INPUT SYSTEM ====================

class DynamicInputState(TypedDict):
    """State for dynamic input collection"""
    messages: Annotated[Sequence[BaseMessage], add]
    user_goal: str
    collected_info: Dict[str, str]
    current_question: Optional[Dict]
    questions_asked: Annotated[List[str], add]
    enough_info: bool

# ==================== WORKFLOW NODES ====================

def initialize_collection(state: DynamicInputState) -> dict:
    """Initialize information collection"""
    logger.info("🎯 Initializing collection...")
    
    goal = state["user_goal"]
    
    return {
        "collected_info": {},
        "enough_info": False,
        "messages": [AIMessage(content=f"Starting to collect info for: {goal}")]
    }

def determine_next_question(state: DynamicInputState) -> dict:
    """Determine what question to ask next"""
    logger.info("🤔 Determining next question...")
    
    collected = state["collected_info"]
    goal = state["user_goal"]
    
    # Determine what we still need
    if "budget" not in collected:
        question = {
            "id": "budget",
            "text": "What is your budget range?",
            "type": "choice",
            "options": ["Under $100", "$100-$500", "$500-$1000", "Over $1000"],
            "context": "This helps us find options in your price range"
        }
    
    elif "priority" not in collected:
        question = {
            "id": "priority",
            "text": "What's most important to you?",
            "type": "choice",
            "options": ["Performance", "Price", "Reliability", "Features"],
            "context": "We'll prioritize recommendations based on this"
        }
    
    elif "timeframe" not in collected:
        question = {
            "id": "timeframe",
            "text": "When do you need this?",
            "type": "choice",
            "options": ["Immediately", "Within a week", "Within a month", "No rush"],
            "context": "This affects availability and shipping options"
        }
    
    else:
        # Have enough information
        return {
            "enough_info": True,
            "messages": [AIMessage(content="Collected sufficient information")]
        }
    
    return {
        "current_question": question,
        "questions_asked": [question["id"]],
        "messages": [AIMessage(content=f"Question: {question['text']}")]
    }

def ask_question(state: DynamicInputState) -> dict:
    """Present question to user (triggers interrupt)"""
    logger.info("❓ Asking user for input...")
    
    question = state["current_question"]
    
    # Format question with options
    question_text = f"\n{question['text']}\n"
    question_text += f"Context: {question['context']}\n"
    question_text += f"Options: {', '.join(question['options'])}\n"
    
    return {
        "messages": [AIMessage(content=question_text)]
    }

def process_answer(state: DynamicInputState) -> dict:
    """Process user's answer"""
    logger.info("✅ Processing user answer...")
    
    # The answer should be in the last message from user
    last_message = state["messages"][-1]
    
    if isinstance(last_message, HumanMessage):
        answer = last_message.content
        question_id = state["current_question"]["id"]
        
        # Store answer
        collected = state["collected_info"].copy()
        collected[question_id] = answer
        
        logger.info(f"Collected {question_id}: {answer}")
        
        return {
            "collected_info": collected,
            "messages": [AIMessage(content=f"Recorded: {question_id} = {answer}")]
        }
    
    return {}

def check_if_enough_info(state: DynamicInputState) -> Literal["ask_more", "finalize"]:
    """Check if we have enough information"""
    
    if state.get("enough_info", False):
        logger.info("✅ Have enough information")
        return "finalize"
    else:
        logger.info("🔄 Need more information")
        return "ask_more"

def generate_recommendation(state: DynamicInputState) -> dict:
    """Generate recommendation based on collected info"""
    logger.info("🎁 Generating recommendation...")
    
    collected = state["collected_info"]
    
    # Generate personalized recommendation
    recommendation = f"""
PERSONALIZED RECOMMENDATION
{'='*50}

Based on your preferences:
"""
    
    for key, value in collected.items():
        recommendation += f"\n  {key.title()}: {value}"
    
    recommendation += f"""

Recommendation: Premium Option XYZ
- Matches your {collected.get('priority', 'needs')} priority
- Within your {collected.get('budget', 'budget')} range
- Available {collected.get('timeframe', 'soon')}

Why this recommendation:
This option best aligns with all your stated preferences.
"""
    
    return {
        "messages": [AIMessage(content=recommendation)]
    }

# ==================== BUILD DYNAMIC INPUT WORKFLOW ====================

checkpointer = MemorySaver()

dynamic_input_workflow = StateGraph(DynamicInputState)

dynamic_input_workflow.add_node("initialize", initialize_collection)
dynamic_input_workflow.add_node("determine_question", determine_next_question)
dynamic_input_workflow.add_node("ask", ask_question)
dynamic_input_workflow.add_node("process", process_answer)
dynamic_input_workflow.add_node("recommend", generate_recommendation)

dynamic_input_workflow.set_entry_point("initialize")
dynamic_input_workflow.add_edge("initialize", "determine_question")
dynamic_input_workflow.add_edge("determine_question", "ask")
dynamic_input_workflow.add_edge("ask", "process")

# Conditional: ask more or finalize
dynamic_input_workflow.add_conditional_edges(
    "process",
    check_if_enough_info,
    {
        "ask_more": "determine_question",  # Loop back
        "finalize": "recommend"
    }
)

dynamic_input_workflow.add_edge("recommend", END)

# Compile with interrupt after ask node
dynamic_input_system = dynamic_input_workflow.compile(
    checkpointer=checkpointer,
    interrupt_after=["ask"]  # Pause after each question
)

logger.info("✅ Dynamic input system compiled")

# ==================== DEMONSTRATION ====================

def demonstrate_dynamic_input():
    """Demonstrate dynamic input collection"""
    
    print("\n" + "="*60)
    print("DYNAMIC USER INPUT DEMONSTRATION")
    print("="*60)
    print("\nThis workflow will ask you questions iteratively")
    print("based on your previous answers.")
    print("="*60)
    
    initial_state = {
        "messages": [HumanMessage(content="Start")],
        "user_goal": "Find a laptop",
        "collected_info": {},
        "current_question": None,
        "questions_asked": [],
        "enough_info": False
    }
    
    config = {"configurable": {"thread_id": "dynamic-input-demo"}}
    
    # Execute workflow
    state = initial_state
    
    while True:
        # Execute until next interrupt
        state = dynamic_input_system.invoke(state, config=config)
        
        # Check if workflow is complete
        if state.get("enough_info", False):
            print("\n" + "="*60)
            print("INFORMATION COLLECTION COMPLETE")
            print("="*60)
            print(state["messages"][-1].content)
            break
        
        # Display question
        question = state["current_question"]
        print(f"\n{'='*60}")
        print(f"QUESTION {len(state['questions_asked'])}")
        print(f"{'='*60}")
        print(f"\n{question['text']}")
        print(f"\nContext: {question['context']}")
        print(f"\nOptions:")
        for i, option in enumerate(question["options"], 1):
            print(f"  {i}. {option}")
        
        # Get user input
        while True:
            user_input = input("\nYour choice (enter number or text): ").strip()
            
            # Convert number to option
            if user_input.isdigit():
                idx = int(user_input) - 1
                if 0 <= idx < len(question["options"]):
                    user_input = question["options"][idx]
                    break
            elif user_input in question["options"]:
                break
            
            print("Invalid choice. Please try again.")
        
        print(f"✅ You selected: {user_input}")
        
        # Add user's answer to state
        state["messages"].append(HumanMessage(content=user_input))

# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    demonstrate_dynamic_input()
```

---

## ✏️ Part 4: Editing and Resuming Execution

### Theory: State Editing

#### What Is State Editing?

**State editing** allows humans to **modify the workflow state** while it's paused, then resume with the modified state.

```
Normal Resume:
Pause → Human reviews → Resume with same state

Resume with Editing:
Pause → Human reviews → Human edits state → Resume with modified state
```

**Example:**
```
Agent generates email draft
→ Pause for review
→ Human edits: "Change subject line, fix typo"
→ Resume: Agent sends edited version
```

#### Why Allow State Editing?

**1. Correction:**
```
AI makes a mistake
→ Human fixes it without restarting
→ Workflow continues
```

**2. Refinement:**
```
AI output is good but not perfect
→ Human improves it
→ Workflow proceeds with improved version
```

**3. Adaptation:**
```
Circumstances changed while paused
→ Human updates parameters
→ Workflow adapts to new conditions
```

**4. Learning:**
```
Human edits serve as training examples
→ System can learn from corrections
→ Improves over time
```

#### What Can Be Edited?

**Editable Elements:**
- **Generated content**: Text, code, configurations
- **Parameters**: Settings, thresholds, options
- **Decisions**: Change AI's chosen path
- **Data**: Correct/add information

**Non-Editable Elements:**
- **Execution history**: What already happened
- **Timestamps**: When things occurred
- **Node completions**: Which nodes ran
- **Audit logs**: Immutable record

#### State Editing Patterns

**Pattern 1: Direct Field Editing**
```python
# Workflow paused
current_state = get_current_state()

# Human edits specific fields
current_state["email_subject"] = "Updated Subject"
current_state["email_body"] = "Revised body text..."

# Resume with edited state
resume_workflow(current_state)
```

**Pattern 2: Delta Updates**
```python
# Specify only what changed
edits = {
    "email_subject": "New subject",
    "recipient_count": 500  # Changed from 1000
}

# Apply edits and resume
resume_with_edits(edits)
```

**Pattern 3: Validation Before Resume**
```python
# Human edits state
edited_state = apply_human_edits(current_state)

# Validate edited state
if validate_state(edited_state):
    resume_workflow(edited_state)
else:
    show_error("Invalid edits - please fix")
```

**Pattern 4: Editing with History**
```python
# Track edit history
edit_history = [
    {
        "timestamp": "2026-01-05T10:30:00Z",
        "editor": "user@company.com",
        "field": "email_subject",
        "old_value": "Original Subject",
        "new_value": "Updated Subject"
    }
]

# Resume with edited state and history
resume_workflow(edited_state, edit_history)
```

#### Resume Strategies

**Strategy 1: Continue Forward**
```
Normal: A → B → [PAUSE] → C → D
Most common resume strategy
```

**Strategy 2: Replay from Edit Point**
```
A → B → [PAUSE + EDIT] → Replay B with new state → C → D
Re-execute nodes affected by edit
```

**Strategy 3: Branch from Edit**
```
A → B → [PAUSE + EDIT] → B' (alternative path) → C' → D'
Create alternative execution path
```

**Strategy 4: Restart with Edits**
```
A → B → [PAUSE + EDIT] → Restart from A with edited initial state
Complete restart with modifications
```

#### Safety Considerations

**1. Validate Edits:**
```python
def validate_edits(original_state, edited_state):
    """Ensure edits are safe"""
    
    # Check required fields present
    if not all(k in edited_state for k in required_fields):
        return False, "Missing required fields"
    
    # Check data types
    if not isinstance(edited_state["count"], int):
        return False, "Count must be integer"
    
    # Check constraints
    if edited_state["count"] < 0:
        return False, "Count cannot be negative"
    
    return True, "Valid"
```

**2. Track Changes:**
```python
def track_changes(original, edited):
    """Record what changed"""
    changes = []
    
    for key in set(original.keys()) | set(edited.keys()):
        if original.get(key) != edited.get(key):
            changes.append({
                "field": key,
                "old": original.get(key),
                "new": edited.get(key)
            })
    
    return changes
```

**3. Audit Edits:**
```python
audit_log.append({
    "event": "state_edited",
    "timestamp": now(),
    "editor": current_user,
    "changes": track_changes(original, edited)
})
```

---

### Implementation: State Editing System

```python
from typing import TypedDict, Annotated, Sequence, List, Dict, Optional
from operator import add
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from datetime import datetime
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== STATE EDITING SYSTEM ====================

class EditableState(TypedDict):
    """State that can be edited by humans"""
    messages: Annotated[Sequence[BaseMessage], add]
    document_title: str
    document_body: str
    target_audience: str
    tone: str
    draft_version: int
    edit_history: Annotated[List[Dict], add]
    review_status: str

# ==================== WORKFLOW NODES ====================

def generate_draft(state: EditableState) -> dict:
    """Generate initial document draft"""
    logger.info("📝 Generating draft...")
    
    audience = state["target_audience"]
    tone = state["tone"]
    
    # Simulate document generation
    title = f"Report for {audience}"
    body = f"""This is a draft document written in a {tone} tone.

Section 1: Introduction
This document addresses the needs of {audience}.

Section 2: Main Content
[Content would be generated here based on requirements]

Section 3: Conclusion
Thank you for your attention."""
    
    edit_entry = {
        "timestamp": datetime.now().isoformat(),
        "event": "draft_generated",
        "version": 1,
        "generator": "AI"
    }
    
    return {
        "document_title": title,
        "document_body": body,
        "draft_version": 1,
        "edit_history": [edit_entry],
        "review_status": "pending_review",
        "messages": [AIMessage(content="Draft generated")]
    }

def prepare_for_review(state: EditableState) -> dict:
    """Prepare document for human review"""
    logger.info("👁️ Preparing for review...")
    
    return {
        "review_status": "in_review",
        "messages": [AIMessage(content="Document ready for review and editing")]
    }

def review_point(state: EditableState) -> dict:
    """Review point - triggers interrupt"""
    logger.info("⏸️ Paused for human review and editing...")
    
    return {
        "messages": [AIMessage(content="Waiting for human review...")]
    }

def validate_edits(state: EditableState) -> dict:
    """Validate any edits made by human"""
    logger.info("✅ Validating edits...")
    
    # Check if there are edits
    if state["draft_version"] > 1:
        logger.info(f"Edits detected - now version {state['draft_version']}")
        
        validation_msg = f"Edits validated - document is now version {state['draft_version']}"
    else:
        validation_msg = "No edits made - proceeding with original draft"
    
    return {
        "messages": [AIMessage(content=validation_msg)]
    }

def finalize_document(state: EditableState) -> dict:
    """Finalize the document"""
    logger.info("🎯 Finalizing document...")
    
    edit_summary = []
    for entry in state["edit_history"]:
        timestamp = entry["timestamp"].split("T")[1][:8]
        event = entry["event"].replace("_", " ").title()
        editor = entry.get("editor", entry.get("generator", "Unknown"))
        edit_summary.append(f"  [{timestamp}] {event} by {editor}")
    
    report = f"""
DOCUMENT FINALIZED
{'='*60}

Title: {state['document_title']}
Audience: {state['target_audience']}
Tone: {state['tone']}
Final Version: {state['draft_version']}

Edit History:
{chr(10).join(edit_summary)}

Document Body:
{'-'*60}
{state['document_body']}
{'-'*60}

Status: Finalized and ready for distribution
"""
    
    return {
        "review_status": "finalized",
        "messages": [AIMessage(content=report)]
    }

# ==================== BUILD EDITABLE WORKFLOW ====================

checkpointer = MemorySaver()

editable_workflow = StateGraph(EditableState)

editable_workflow.add_node("generate", generate_draft)
editable_workflow.add_node("prepare_review", prepare_for_review)
editable_workflow.add_node("review", review_point)
editable_workflow.add_node("validate", validate_edits)
editable_workflow.add_node("finalize", finalize_document)

editable_workflow.set_entry_point("generate")
editable_workflow.add_edge("generate", "prepare_review")
editable_workflow.add_edge("prepare_review", "review")
editable_workflow.add_edge("review", "validate")
editable_workflow.add_edge("validate", "finalize")
editable_workflow.add_edge("finalize", END)

# Compile with interrupt after review
editable_system = editable_workflow.compile(
    checkpointer=checkpointer,
    interrupt_after=["review"]
)

logger.info("✅ Editable workflow compiled")

# ==================== EDITING UTILITIES ====================

def display_document(state: EditableState):
    """Display document for review"""
    print("\n" + "="*60)
    print("DOCUMENT FOR REVIEW")
    print("="*60)
    print(f"\nTitle: {state['document_title']}")
    print(f"Audience: {state['target_audience']}")
    print(f"Tone: {state['tone']}")
    print(f"Version: {state['draft_version']}")
    print("\nBody:")
    print("-"*60)
    print(state['document_body'])
    print("-"*60)

def apply_human_edits(state: EditableState, edits: Dict) -> EditableState:
    """Apply human edits to state"""
    
    # Track what changed
    changes = []
    
    for field, new_value in edits.items():
        if field in state and state[field] != new_value:
            changes.append({
                "field": field,
                "old_value": state[field],
                "new_value": new_value
            })
            state[field] = new_value
    
    if changes:
        # Increment version
        state["draft_version"] += 1
        
        # Log the edit
        edit_entry = {
            "timestamp": datetime.now().isoformat(),
            "event": "human_edited",
            "version": state["draft_version"],
            "editor": "human_reviewer",
            "changes": changes
        }
        
        state["edit_history"].append(edit_entry)
        
        logger.info(f"Applied {len(changes)} edits, now version {state['draft_version']}")
    
    return state

# ==================== DEMONSTRATION ====================

def demonstrate_state_editing():
    """Demonstrate state editing and resume"""
    
    print("\n" + "="*60)
    print("STATE EDITING DEMONSTRATION")
    print("="*60)
    
    initial_state = {
        "messages": [HumanMessage(content="Start document generation")],
        "document_title": "",
        "document_body": "",
        "target_audience": "executives",
        "tone": "professional",
        "draft_version": 0,
        "edit_history": [],
        "review_status": "not_started"
    }
    
    config = {"configurable": {"thread_id": "editing-demo"}}
    
    print("\n--- Phase 1: Generate Draft ---")
    
    # Execute until interrupt
    state = editable_system.invoke(initial_state, config=config)
    
    print("\n🛑 WORKFLOW PAUSED FOR REVIEW")
    display_document(state)
    
    # Human review interface
    print("\n" + "="*60)
    print("REVIEW OPTIONS")
    print("="*60)
    print("1. Approve as-is (no edits)")
    print("2. Edit document")
    print("3. Cancel workflow")
    
    choice = input("\nYour choice (1-3): ").strip()
    
    if choice == "1":
        print("\n✅ Approved without edits")
        # No edits, just resume
    
    elif choice == "2":
        print("\n--- EDITING MODE ---")
        
        # Collect edits
        print("\nWhat would you like to edit?")
        print("Leave blank to keep current value")
        
        edits = {}
        
        new_title = input(f"\nTitle [{state['document_title']}]: ").strip()
        if new_title:
            edits["document_title"] = new_title
        
        new_tone = input(f"Tone [{state['tone']}]: ").strip()
        if new_tone:
            edits["tone"] = new_tone
        
        print("\nDocument body:")
        print("Enter your edited version (or press Enter to skip)")
        print("Type 'END' on a new line when done")
        
        body_lines = []
        while True:
            line = input()
            if line == "END":
                break
            body_lines.append(line)
        
        if body_lines:
            edits["document_body"] = "\n".join(body_lines)
        
        if edits:
            # Apply edits
            state = apply_human_edits(state, edits)
            
            print(f"\n✅ Applied {len(edits)} edits")
            print(f"Document is now version {state['draft_version']}")
            
            # Show edited version
            print("\n--- EDITED DOCUMENT ---")
            display_document(state)
        else:
            print("\n⚠️ No edits made")
    
    else:
        print("\n❌ Workflow cancelled")
        return
    
    # Resume workflow
    print("\n--- Phase 2: Resume Workflow ---")
    input("Press Enter to resume workflow...")
    
    final_state = editable_system.invoke(state, config=config)
    
    print("\n" + "="*60)
    print("FINAL RESULT")
    print("="*60)
    print(final_state["messages"][-1].content)

# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    demonstrate_state_editing()
```

---

## 📋 Part 5: Audit Trails and Compliance

### Theory: Audit Trails

#### What Is an Audit Trail?

An **audit trail** is a comprehensive, chronological record of all events and actions in a workflow, providing:
- **Who**: Which agent or human
- **What**: What action was taken
- **When**: Timestamp
- **Why**: Reason or context
- **Result**: Outcome of action

```
Audit Trail Example:
[2026-01-05 10:00:00] SYSTEM: Workflow initiated by user@company.com
[2026-01-05 10:00:01] AI_AGENT: Generated proposal (id: prop-123)
[2026-01-05 10:00:02] SYSTEM: Requested approval from manager@company.com
[2026-01-05 10:15:30] HUMAN: manager@company.com approved (reason: "Meets requirements")
[2026-01-05 10:15:31] AI_AGENT: Executed action (result: success)
[2026-01-05 10:15:32] SYSTEM: Workflow completed
```

#### Why Audit Trails?

**1. Compliance:**
```
Regulations require proof of:
- Who approved what
- When decisions were made
- What data was accessed
- How processes were followed

Examples: GDPR, SOC 2, HIPAA, SOX
```

**2. Accountability:**
```
Determine responsibility:
- Who made this decision?
- Why was this action taken?
- Was proper approval obtained?
```

**3. Debugging:**
```
Troubleshoot issues:
- What went wrong?
- When did it go wrong?
- What was the state at that time?
```

**4. Process Improvement:**
```
Analyze patterns:
- How long do approvals take?
- Where are bottlenecks?
- What gets rejected most?
```

**5. Legal Protection:**
```
Defend against claims:
- Prove compliance
- Show due diligence
- Document decisions
```

#### What to Log in Audit Trails

**Essential Elements:**

**1. Timestamps:**
```python
{
    "timestamp": "2026-01-05T10:15:30.123Z",
    "timezone": "UTC"
}
```

**2. Actor Information:**
```python
{
    "actor_type": "human" | "ai_agent" | "system",
    "actor_id": "user@company.com",
    "actor_name": "John Smith",
    "actor_role": "manager"
}
```

**3. Action Details:**
```python
{
    "action": "approve_deployment",
    "action_type": "approval",
    "resource": "production_deployment",
    "resource_id": "deploy-456"
}
```

**4. Context:**
```python
{
    "workflow_id": "wf-123",
    "node_name": "approval_gate",
    "state_snapshot": {...},
    "previous_action": "generate_proposal"
}
```

**5. Result:**
```python
{
    "result": "success" | "failure",
    "outcome": "Deployment approved",
    "error": null
}
```

**6. Metadata:**
```python
{
    "ip_address": "192.168.1.1",
    "user_agent": "Mozilla/5.0...",
    "session_id": "sess-789",
    "request_id": "req-321"
}
```

#### Audit Trail Patterns

**Pattern 1: Structured Logging**
```python
def log_audit_event(
    event_type: str,
    actor: str,
    action: str,
    resource: str,
    result: str,
    metadata: dict = None
):
    """Log structured audit event"""
    
    event = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "event_type": event_type,
        "actor": actor,
        "action": action,
        "resource": resource,
        "result": result,
        "metadata": metadata or {}
    }
    
    # Write to audit log
    audit_logger.info(json.dumps(event))
```

**Pattern 2: State Snapshots**
```python
def log_with_state_snapshot(event: str, state: dict):
    """Log event with full state snapshot"""
    
    audit_log.append({
        "timestamp": now(),
        "event": event,
        "state_snapshot": copy.deepcopy(state)  # Full state at this moment
    })
```

**Pattern 3: Delta Logging**
```python
def log_state_change(old_state: dict, new_state: dict):
    """Log only what changed"""
    
    changes = {}
    for key in set(old_state.keys()) | set(new_state.keys()):
        if old_state.get(key) != new_state.get(key):
            changes[key] = {
                "old": old_state.get(key),
                "new": new_state.get(key)
            }
    
    audit_log.append({
        "timestamp": now(),
        "event": "state_changed",
        "changes": changes
    })
```

**Pattern 4: Hierarchical Events**
```python
# Parent event
workflow_audit = {
    "id": "audit-123",
    "workflow_id": "wf-456",
    "start_time": "2026-01-05T10:00:00Z",
    "events": []
}

# Child events
workflow_audit["events"].append({
    "timestamp": "2026-01-05T10:00:01Z",
    "event": "node_started",
    "node": "analyzer"
})

workflow_audit["events"].append({
    "timestamp": "2026-01-05T10:00:05Z",
    "event": "node_completed",
    "node": "analyzer"
})
```

#### Audit Trail Storage

**Requirements:**
- **Immutable**: Cannot be modified after writing
- **Tamper-proof**: Detect unauthorized changes
- **Durable**: Persisted reliably
- **Searchable**: Query by various criteria
- **Secure**: Access controlled

**Storage Options:**

**1. Database (Append-only table):**
```sql
CREATE TABLE audit_log (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    event_type VARCHAR(100) NOT NULL,
    actor VARCHAR(255) NOT NULL,
    action VARCHAR(100) NOT NULL,
    resource VARCHAR(255),
    result VARCHAR(50),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Index for common queries
CREATE INDEX idx_audit_timestamp ON audit_log(timestamp);
CREATE INDEX idx_audit_actor ON audit_log(actor);
CREATE INDEX idx_audit_resource ON audit_log(resource);
```

**2. File-based (JSONL):**
```python
# Append-only JSON Lines format
with open("audit_log.jsonl", "a") as f:
    f.write(json.dumps(audit_event) + "\n")
```

**3. Dedicated Audit Services:**
- CloudWatch Logs (AWS)
- Cloud Logging (GCP)
- Azure Monitor
- Splunk
- Elasticsearch

**4. Blockchain (for high-integrity needs):**
```python
# Each log entry hashed and chained
previous_hash = hash(previous_entry)
current_entry["previous_hash"] = previous_hash
current_entry["hash"] = hash(current_entry)
```

#### Compliance Considerations

**GDPR (Data Privacy):**
- Log access to personal data
- Record consent and withdrawals
- Log data deletions
- Provide audit trail to data subjects

**SOC 2 (Security Controls):**
- Log access to systems
- Track configuration changes
- Record security events
- Demonstrate access controls

**HIPAA (Healthcare):**
- Log access to medical records
- Track who viewed patient data
- Record data modifications
- Audit trail retention (6 years)

**SOX (Financial):**
- Log financial transactions
- Track approval workflows
- Record system changes
- Demonstrate controls

#### Retention and Archival

**Retention Policies:**
```python
retention_policies = {
    "operational_logs": "30 days",
    "security_logs": "1 year",
    "compliance_logs": "7 years",
    "financial_logs": "10 years"
}
```

**Archival Strategy:**
```
Active logs (recent): Fast storage, readily available
Archived logs (old): Cheaper storage, slower access
Compressed/encrypted: For long-term storage
```

---

### Implementation: Audit Trail System

```python
from typing import TypedDict, Annotated, Sequence, List, Dict, Optional, Literal
from operator import add
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from datetime import datetime
from enum import Enum
import json
import hashlib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== AUDIT TRAIL SYSTEM ====================

class EventType(str, Enum):
    """Types of audit events"""
    WORKFLOW_STARTED = "workflow_started"
    NODE_STARTED = "node_started"
    NODE_COMPLETED = "node_completed"
    HUMAN_APPROVED = "human_approved"
    HUMAN_REJECTED = "human_rejected"
    STATE_EDITED = "state_edited"
    ERROR_OCCURRED = "error_occurred"
    WORKFLOW_COMPLETED = "workflow_completed"

class AuditEvent:
    """Structured audit event"""
    def __init__(
        self,
        event_type: EventType,
        actor: str,
        action: str,
        resource: Optional[str] = None,
        result: Optional[str] = None,
        metadata: Optional[Dict] = None
    ):
        self.timestamp = datetime.utcnow().isoformat() + "Z"
        self.event_type = event_type
        self.actor = actor
        self.action = action
        self.resource = resource
        self.result = result
        self.metadata = metadata or {}
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type.value,
            "actor": self.actor,
            "action": self.action,
            "resource": self.resource,
            "result": self.result,
            "metadata": self.metadata
        }
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)

class AuditLogger:
    """Audit trail logger"""
    def __init__(self):
        self.events: List[AuditEvent] = []
    
    def log(self, event: AuditEvent):
        """Log an audit event"""
        self.events.append(event)
        
        # Also log to Python logger
        logger.info(f"AUDIT: {event.event_type.value} by {event.actor} - {event.action}")
    
    def get_events(self) -> List[AuditEvent]:
        """Get all events"""
        return self.events
    
    def get_events_by_type(self, event_type: EventType) -> List[AuditEvent]:
        """Get events of specific type"""
        return [e for e in self.events if e.event_type == event_type]
    
    def get_events_by_actor(self, actor: str) -> List[AuditEvent]:
        """Get events by specific actor"""
        return [e for e in self.events if e.actor == actor]
    
    def export_json(self) -> str:
        """Export all events as JSON"""
        return json.dumps([e.to_dict() for e in self.events], indent=2)
    
    def export_jsonl(self) -> str:
        """Export as JSON Lines (one JSON object per line)"""
        return "\n".join([json.dumps(e.to_dict()) for e in self.events])
    
    def generate_summary(self) -> str:
        """Generate human-readable summary"""
        lines = ["AUDIT TRAIL SUMMARY", "=" * 60, ""]
        
        for event in self.events:
            timestamp = event.timestamp.split("T")[1][:8]  # Extract time
            lines.append(f"[{timestamp}] {event.event_type.value}")
            lines.append(f"           Actor: {event.actor}")
            lines.append(f"           Action: {event.action}")
            if event.result:
                lines.append(f"           Result: {event.result}")
            if event.metadata:
                lines.append(f"           Metadata: {json.dumps(event.metadata)}")
            lines.append("")
        
        return "\n".join(lines)

# Global audit logger
audit_logger = AuditLogger()

# ==================== AUDITED WORKFLOW STATE ====================

class AuditedState(TypedDict):
    """State with audit logging"""
    messages: Annotated[Sequence[BaseMessage], add]
    workflow_id: str
    task: str
    analysis: str
    decision: str
    approval_status: str

# ==================== AUDITED WORKFLOW NODES ====================

def start_workflow_node(state: AuditedState) -> dict:
    """Start workflow with audit"""
    
    # Log workflow start
    audit_logger.log(AuditEvent(
        event_type=EventType.WORKFLOW_STARTED,
        actor="system",
        action="initialize_workflow",
        resource=state["workflow_id"],
        result="success",
        metadata={"task": state["task"]}
    ))
    
    return {
        "messages": [AIMessage(content="Workflow started with audit logging")]
    }

def analyze_with_audit(state: AuditedState) -> dict:
    """Analyze with audit logging"""
    
    # Log node start
    audit_logger.log(AuditEvent(
        event_type=EventType.NODE_STARTED,
        actor="ai_agent",
        action="start_analysis",
        resource=state["workflow_id"]
    ))
    
    # Perform analysis
    analysis = f"Analyzed task: {state['task']}"
    
    # Log node completion
    audit_logger.log(AuditEvent(
        event_type=EventType.NODE_COMPLETED,
        actor="ai_agent",
        action="complete_analysis",
        resource=state["workflow_id"],
        result="success",
        metadata={"analysis_length": len(analysis)}
    ))
    
    return {
        "analysis": analysis,
        "messages": [AIMessage(content="Analysis complete")]
    }

def decision_with_audit(state: AuditedState) -> dict:
    """Make decision with audit"""
    
    audit_logger.log(AuditEvent(
        event_type=EventType.NODE_STARTED,
        actor="ai_agent",
        action="make_decision",
        resource=state["workflow_id"]
    ))
    
    decision = "Proceed with action"
    
    audit_logger.log(AuditEvent(
        event_type=EventType.NODE_COMPLETED,
        actor="ai_agent",
        action="decision_made",
        resource=state["workflow_id"],
        result="success",
        metadata={"decision": decision}
    ))
    
    return {
        "decision": decision,
        "messages": [AIMessage(content=f"Decision: {decision}")]
    }

def approval_with_audit(state: AuditedState) -> dict:
    """Request approval with audit"""
    
    audit_logger.log(AuditEvent(
        event_type=EventType.NODE_STARTED,
        actor="system",
        action="request_approval",
        resource=state["workflow_id"]
    ))
    
    return {
        "messages": [AIMessage(content="Approval requested")]
    }

def execute_with_audit(state: AuditedState) -> dict:
    """Execute with audit logging"""
    
    # Check approval
    if state["approval_status"] == "approved":
        audit_logger.log(AuditEvent(
            event_type=EventType.HUMAN_APPROVED,
            actor="human_reviewer",
            action="approve_execution",
            resource=state["workflow_id"],
            result="approved"
        ))
        
        # Execute
        audit_logger.log(AuditEvent(
            event_type=EventType.NODE_STARTED,
            actor="ai_agent",
            action="execute_action",
            resource=state["workflow_id"]
        ))
        
        result = "Action executed successfully"
        
        audit_logger.log(AuditEvent(
            event_type=EventType.NODE_COMPLETED,
            actor="ai_agent",
            action="action_executed",
            resource=state["workflow_id"],
            result="success"
        ))
    
    else:
        audit_logger.log(AuditEvent(
            event_type=EventType.HUMAN_REJECTED,
            actor="human_reviewer",
            action="reject_execution",
            resource=state["workflow_id"],
            result="rejected"
        ))
        
        result = "Action cancelled by reviewer"
    
    return {
        "messages": [AIMessage(content=result)]
    }

def complete_workflow_node(state: AuditedState) -> dict:
    """Complete workflow with audit"""
    
    audit_logger.log(AuditEvent(
        event_type=EventType.WORKFLOW_COMPLETED,
        actor="system",
        action="finalize_workflow",
        resource=state["workflow_id"],
        result="success"
    ))
    
    # Generate audit report
    audit_report = audit_logger.generate_summary()
    
    return {
        "messages": [AIMessage(content=f"Workflow complete\n\n{audit_report}")]
    }

# ==================== BUILD AUDITED WORKFLOW ====================

checkpointer = MemorySaver()

audited_workflow = StateGraph(AuditedState)

audited_workflow.add_node("start", start_workflow_node)
audited_workflow.add_node("analyze", analyze_with_audit)
audited_workflow.add_node("decide", decision_with_audit)
audited_workflow.add_node("approval", approval_with_audit)
audited_workflow.add_node("execute", execute_with_audit)
audited_workflow.add_node("complete", complete_workflow_node)

audited_workflow.set_entry_point("start")
audited_workflow.add_edge("start", "analyze")
audited_workflow.add_edge("analyze", "decide")
audited_workflow.add_edge("decide", "approval")
audited_workflow.add_edge("approval", "execute")
audited_workflow.add_edge("execute", "complete")
audited_workflow.add_edge("complete", END)

audited_system = audited_workflow.compile(
    checkpointer=checkpointer,
    interrupt_after=["approval"]
)

logger.info("✅ Audited workflow compiled")

# ==================== DEMONSTRATION ====================

def demonstrate_audit_trail():
    """Demonstrate audit trail logging"""
    
    print("\n" + "="*60)
    print("AUDIT TRAIL DEMONSTRATION")
    print("="*60)
    
    # Clear previous audit logs
    audit_logger.events.clear()
    
    initial_state = {
        "messages": [HumanMessage(content="Start")],
        "workflow_id": "wf-audit-demo-001",
        "task": "Deploy new feature to production",
        "analysis": "",
        "decision": "",
        "approval_status": ""
    }
    
    config = {"configurable": {"thread_id": "audit-demo"}}
    
    print("\n--- Phase 1: Execute until approval ---")
    
    # Execute until interrupt
    state = audited_system.invoke(initial_state, config=config)
    
    print("\n🛑 WORKFLOW PAUSED FOR APPROVAL")
    print(f"Task: {state['task']}")
    print(f"Decision: {state['decision']}")
    
    # Show current audit trail
    print("\n--- AUDIT TRAIL SO FAR ---")
    print(audit_logger.generate_summary())
    
    # Human approval
    print("\n--- HUMAN REVIEW ---")
    approval = input("Approve? (yes/no): ").strip().lower()
    
    if approval == "yes":
        state["approval_status"] = "approved"
        print("✅ Approved")
    else:
        state["approval_status"] = "rejected"
        print("❌ Rejected")
    
    print("\n--- Phase 2: Resume with approval decision ---")
    
    # Resume
    final_state = audited_system.invoke(state, config=config)
    
    print("\n" + "="*60)
    print("FINAL RESULT WITH COMPLETE AUDIT TRAIL")
    print("="*60)
    print(final_state["messages"][-1].content)
    
    # Export audit log
    print("\n" + "="*60)
    print("EXPORT OPTIONS")
    print("="*60)
    print("\n1. JSON Format")
    print(audit_logger.export_json()[:500] + "...")
    
    print("\n2. JSONL Format (for log streaming)")
    print(audit_logger.export_jsonl()[:500] + "...")
    
    # Query audit log
    print("\n" + "="*60)
    print("AUDIT LOG QUERIES")
    print("="*60)
    
    human_events = audit_logger.get_events_by_actor("human_reviewer")
    print(f"\nHuman actions: {len(human_events)}")
    for event in human_events:
        print(f"  - {event.action}: {event.result}")
    
    ai_events = audit_logger.get_events_by_actor("ai_agent")
    print(f"\nAI agent actions: {len(ai_events)}")
    for event in ai_events:
        print(f"  - {event.action}")

# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    demonstrate_audit_trail()
```

---

## 📚 Best Practices Summary

### Interrupts and Breakpoints

**✅ DO:**
- Set interrupts before critical/irreversible actions
- Use checkpointing for long-running workflows
- Provide clear context when pausing
- Test interrupt and resume logic thoroughly

**❌ DON'T:**
- Interrupt too frequently (interruption fatigue)
- Forget to handle timeout scenarios
- Lose state between interrupts
- Make interrupts confusing for users

---

### Approval Workflows

**✅ DO:**
- Clearly describe what's being approved
- Track approval metadata (who, when, why)
- Handle rejection gracefully
- Implement timeout policies
- Support multi-level approvals when needed

**❌ DON'T:**
- Approve without showing full details
- Forget to audit approvals
- Make approval process too complex
- Skip validation of approvals

---

### Dynamic User Input

**✅ DO:**
- Validate input immediately
- Provide clear context for each question
- Support going back/editing previous answers
- Show progress through multi-step forms
- Make questions conditional based on previous answers

**❌ DON'T:**
- Ask for information you don't need
- Make questions ambiguous
- Accept invalid input without feedback
- Ask too many questions at once

---

### State Editing

**✅ DO:**
- Validate edited state before resuming
- Track all edits in audit log
- Show diff between original and edited
- Allow undo of edits
- Test resume with edited state

**❌ DON'T:**
- Allow edits that break workflow
- Lose track of what was edited
- Forget to version edited state
- Make editing process confusing

---

### Audit Trails

**✅ DO:**
- Log all significant events
- Include timestamps (UTC)
- Record actor (human/AI/system)
- Store immutably
- Make logs searchable
- Retain per compliance requirements
- Export in standard formats

**❌ DON'T:**
- Log sensitive data (passwords, keys)
- Make logs modifiable
- Forget to handle log rotation
- Skip audit logs for "minor" actions
- Make logs unreadable

---

## ✅ Chapter 17 Complete!

**You now understand:**
- ✅ Interrupt patterns and breakpoints (pause execution)
- ✅ Approval workflows (human approval gates)
- ✅ Dynamic user input (collect info during execution)
- ✅ State editing and resuming (modify and continue)
- ✅ Audit trails and compliance (comprehensive logging)
- ✅ Checkpointing for state persistence
- ✅ When to use each HITL pattern
- ✅ Best practices for human-in-the-loop systems

**Key Takeaways:**
- HITL combines AI automation with human judgment
- Interrupts enable critical decision points
- Approval workflows ensure oversight
- Dynamic input adapts to user needs
- State editing provides flexibility
- Audit trails ensure compliance and accountability
- Checkpointing enables resilience

---

**Ready for Chapter 18?**

**Chapter 18: Production Engineering for LangGraph Agents** will cover:
- Error handling and recovery strategies
- Performance optimization and caching
- Monitoring and observability
- Deployment patterns
- Testing strategies
- Scalability considerations

Just say "Continue to Chapter 18" when ready!