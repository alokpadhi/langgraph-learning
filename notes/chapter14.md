# Chapter 14: Roleplay & Debate Agents

## 🎭 Introduction: Roleplay and Debate in Multi-Agent Systems

**Roleplay and debate agents** use different perspectives and opposing viewpoints to:
- Explore problems from multiple angles
- Generate more comprehensive solutions
- Identify weaknesses in arguments
- Simulate real-world scenarios
- Make better decisions through discourse

Think of it like a **courtroom** - prosecutor and defense attorney argue opposing sides, helping the judge reach a better verdict than either side alone could provide.

---

## 🎭 Part 1: Role-Based Agent Systems

### Theory: Role-Based Agents

#### What Are Role-Based Agents?

**Role-based agents** are agents designed to embody specific roles, personas, or perspectives. Each agent stays "in character" and provides input consistent with their assigned role.

```
Traditional Agent: "I am an AI assistant"
Role-Based Agent: "I am a financial analyst with 20 years of Wall Street experience"
```

#### Why Use Roles?

**1. Perspective Diversity:**
- Different roles bring different viewpoints
- Reduces groupthink
- Uncovers blind spots

**2. Domain Expertise:**
- Role = specialist knowledge
- More authentic responses
- Better quality analysis

**3. Simulation:**
- Model real-world interactions
- Test scenarios
- Train for situations

**4. Creative Exploration:**
- Explore "what if" scenarios
- Role-play user personas
- Design thinking exercises

#### Types of Roles

**1. Professional Roles:**
```
- CFO: Financial perspective
- CTO: Technical perspective
- CMO: Marketing perspective
- CEO: Strategic perspective
```

**2. Stakeholder Roles:**
```
- Customer: User experience perspective
- Investor: Financial return perspective
- Employee: Work environment perspective
- Regulator: Compliance perspective
```

**3. Personality Roles:**
```
- Optimist: Sees opportunities
- Pessimist: Sees risks
- Realist: Sees practicalities
- Innovator: Sees possibilities
```

**4. Functional Roles:**
```
- Critic: Points out flaws
- Supporter: Reinforces strengths
- Mediator: Finds middle ground
- Devil's Advocate: Challenges assumptions
```

#### Role Design Elements

**Effective role definition includes:**

**1. Identity:**
```
"I am a [role] with [background/expertise]"
Example: "I am a security architect with 15 years in financial services"
```

**2. Perspective:**
```
"I prioritize [values/concerns]"
Example: "I prioritize data security, compliance, and risk mitigation"
```

**3. Communication Style:**
```
"I communicate in a [style]"
Example: "I communicate in a technical, detail-oriented manner"
```

**4. Goals/Motivations:**
```
"My goal is to [objective]"
Example: "My goal is to ensure systems are secure and compliant"
```

**5. Constraints:**
```
"I must consider [limitations]"
Example: "I must consider budget constraints and implementation timelines"
```

#### Role Consistency

**Key principle:** Agents must maintain role consistency throughout interaction.

**Bad (breaks character):**
```
Security Agent: "From a security perspective, we need encryption"
[later in same conversation]
Security Agent: "Let's focus on user experience instead"
← This breaks role consistency
```

**Good (maintains character):**
```
Security Agent: "From a security perspective, we need encryption"
[later in same conversation]
Security Agent: "I understand UX is important, but security cannot be compromised"
← Stays in character while acknowledging other perspectives
```

#### When to Use Role-Based Agents

✅ **Use Role-Based Agents When:**
- Need multiple perspectives on complex decisions
- Simulating stakeholder interactions
- Testing ideas from different angles
- Creative brainstorming
- Training and scenario planning

❌ **Don't Use When:**
- Simple, straightforward tasks
- Single perspective is sufficient
- Roles would add unnecessary complexity
- Need fast, simple answers

---

### Implementation: Role-Based Agent System

```python
from typing import TypedDict, Annotated, Sequence, List, Dict
from operator import add
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== ROLE-BASED AGENT SYSTEM ====================

# Role definition
class AgentRole:
    """Defines an agent's role"""
    def __init__(
        self,
        name: str,
        identity: str,
        perspective: str,
        communication_style: str,
        goals: str
    ):
        self.name = name
        self.identity = identity
        self.perspective = perspective
        self.communication_style = communication_style
        self.goals = goals
    
    def get_system_prompt(self) -> str:
        """Generate system prompt from role definition"""
        return f"""{self.identity}

Your perspective: {self.perspective}

Communication style: {self.communication_style}

Your goals: {self.goals}

Stay in character throughout the conversation. Provide input consistent with your role."""

# Individual role agent state
class RoleAgentState(TypedDict):
    """State for role-based agent"""
    messages: Annotated[Sequence[BaseMessage], add]
    role_name: str
    context: str
    perspective: str

# Orchestrator state
class RoleplayState(TypedDict):
    """State for roleplay orchestration"""
    messages: Annotated[Sequence[BaseMessage], add]
    scenario: str
    roles: List[str]
    perspectives: Dict[str, str]
    discussion_round: Annotated[int, add]
    max_rounds: int
    synthesis: str

llm = ChatOllama(model="llama3.2", temperature=0.7)

# ==================== ROLE DEFINITIONS ====================

ROLES = {
    "cfo": AgentRole(
        name="CFO",
        identity="I am the Chief Financial Officer with 20 years of experience in corporate finance and investment strategy.",
        perspective="I prioritize financial sustainability, ROI, cost management, and shareholder value.",
        communication_style="I communicate in a data-driven, quantitative manner with focus on numbers and financial metrics.",
        goals="My goal is to ensure financial health, maximize returns, and manage risks."
    ),
    "cto": AgentRole(
        name="CTO",
        identity="I am the Chief Technology Officer with deep expertise in software architecture, infrastructure, and emerging technologies.",
        perspective="I prioritize technical excellence, scalability, innovation, and system reliability.",
        communication_style="I communicate in technical terms, focusing on architecture, implementation feasibility, and technical trade-offs.",
        goals="My goal is to build robust, scalable systems using cutting-edge technology."
    ),
    "cmo": AgentRole(
        name="CMO",
        identity="I am the Chief Marketing Officer with extensive experience in brand building, customer acquisition, and market strategy.",
        perspective="I prioritize customer satisfaction, market positioning, brand value, and growth.",
        communication_style="I communicate in customer-centric language, focusing on user experience, market trends, and brand impact.",
        goals="My goal is to grow market share, build brand loyalty, and understand customer needs."
    ),
    "customer": AgentRole(
        name="Customer",
        identity="I am a typical end-user customer who values simplicity, reliability, and good value for money.",
        perspective="I prioritize ease of use, reliability, fair pricing, and responsive support.",
        communication_style="I communicate from a user's perspective, focusing on practical everyday concerns and experiences.",
        goals="My goal is to get value for my money and have products that just work without hassle."
    ),
    "innovator": AgentRole(
        name="Innovator",
        identity="I am an innovation strategist focused on breakthrough ideas and disruptive opportunities.",
        perspective="I prioritize creativity, bold moves, future trends, and competitive differentiation.",
        communication_style="I communicate with enthusiasm about possibilities, challenging conventional thinking.",
        goals="My goal is to push boundaries and explore innovative solutions that create competitive advantages."
    ),
    "pragmatist": AgentRole(
        name="Pragmatist",
        identity="I am a pragmatic operations leader focused on practical execution and realistic outcomes.",
        perspective="I prioritize feasibility, resource constraints, timeline realities, and risk management.",
        communication_style="I communicate in practical, grounded terms, focusing on what's actually achievable.",
        goals="My goal is to ensure ideas can be executed successfully with available resources."
    )
}

# ==================== ROLE AGENT EXECUTION ====================

def create_role_agent(role: AgentRole):
    """Create an agent for a specific role"""
    
    def role_agent_execute(state: RoleAgentState) -> dict:
        """Execute with role perspective"""
        logger.info(f"🎭 {role.name}: Providing perspective")
        
        # Build prompt with role
        messages = [
            SystemMessage(content=role.get_system_prompt()),
            HumanMessage(content=f"""Scenario: {state['context']}

Provide your perspective on this scenario staying in character as {role.name}.

Your perspective:""")
        ]
        
        response = llm.invoke(messages)
        
        return {
            "perspective": response.content,
            "messages": [AIMessage(content=f"[{role.name}] {response.content}")]
        }
    
    # Build workflow for this role
    role_workflow = StateGraph(RoleAgentState)
    role_workflow.add_node("execute", role_agent_execute)
    role_workflow.set_entry_point("execute")
    role_workflow.add_edge("execute", END)
    
    return role_workflow.compile()

# Create agents for all roles
ROLE_AGENTS = {
    role_name: create_role_agent(role)
    for role_name, role in ROLES.items()
}

# ==================== ROLEPLAY ORCHESTRATOR ====================

def gather_perspectives(state: RoleplayState) -> dict:
    """Gather perspectives from all roles"""
    logger.info(f"🎬 Roleplay Round {state['discussion_round'] + 1}: Gathering perspectives")
    
    scenario = state["scenario"]
    roles = state["roles"]
    
    perspectives = {}
    
    for role_name in roles:
        logger.info(f"   Consulting {role_name}...")
        
        role_agent = ROLE_AGENTS[role_name]
        
        # Build context with previous perspectives if not first round
        context = scenario
        if state["perspectives"]:
            prev_perspectives = "\n\n".join([
                f"{name.upper()}'s view: {persp[:150]}..."
                for name, persp in state["perspectives"].items()
            ])
            context = f"{scenario}\n\nPrevious discussion:\n{prev_perspectives}"
        
        result = role_agent.invoke({
            "messages": [],
            "role_name": role_name,
            "context": context,
            "perspective": ""
        })
        
        perspectives[role_name] = result["perspective"]
    
    logger.info(f"✅ Collected {len(perspectives)} perspectives")
    
    return {
        "perspectives": perspectives,
        "discussion_round": 1
    }

def analyze_convergence(state: RoleplayState) -> dict:
    """Analyze if perspectives are converging or need more discussion"""
    logger.info("🔍 Analyzing perspective convergence")
    
    perspectives = state["perspectives"]
    
    # Use LLM to analyze convergence
    perspectives_text = "\n\n".join([
        f"{role.upper()}:\n{persp}"
        for role, persp in perspectives.items()
    ])
    
    convergence_prompt = ChatPromptTemplate.from_messages([
        ("system", "Analyze if these perspectives are converging toward agreement or if there are significant disagreements that need more discussion."),
        ("human", """Perspectives:

{perspectives}

Are these perspectives:
A) CONVERGED - Generally agree, differences are minor
B) DIVERGENT - Significant disagreements remain

Respond with just A or B.

Analysis:""")
    ])
    
    chain = convergence_prompt | llm
    response = chain.invoke({"perspectives": perspectives_text})
    
    converged = "A" in response.content or "CONVERGED" in response.content
    
    logger.info(f"Convergence status: {'CONVERGED' if converged else 'DIVERGENT'}")
    
    return {}

def synthesize_perspectives(state: RoleplayState) -> dict:
    """Synthesize all perspectives into coherent analysis"""
    logger.info("🔗 Synthesizing perspectives")
    
    perspectives = state["perspectives"]
    
    perspectives_text = "\n\n".join([
        f"=== {role.upper()} ===\n{persp}"
        for role, persp in perspectives.items()
    ])
    
    synthesis_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are synthesizing multiple role-based perspectives. Create a balanced analysis that acknowledges all viewpoints."),
        ("human", """Scenario: {scenario}

Perspectives from different roles:
{perspectives}

Synthesize these perspectives into a comprehensive analysis that:
1. Acknowledges each perspective
2. Identifies common ground
3. Highlights key disagreements
4. Provides balanced recommendations

Synthesis:""")
    ])
    
    chain = synthesis_prompt | llm
    response = chain.invoke({
        "scenario": state["scenario"],
        "perspectives": perspectives_text
    })
    
    synthesis = f"""ROLE-BASED ANALYSIS

Scenario: {state['scenario']}

Roles Consulted: {', '.join([ROLES[r].name for r in state['roles']])}
Discussion Rounds: {state['discussion_round']}

{response.content}"""
    
    return {
        "synthesis": synthesis,
        "messages": [AIMessage(content=synthesis)]
    }

def should_continue_discussion(state: RoleplayState) -> str:
    """Decide if more discussion rounds needed"""
    
    if state["discussion_round"] >= state["max_rounds"]:
        logger.info("Max discussion rounds reached")
        return "synthesize"
    
    # For simplicity, we synthesize after gathering perspectives
    # In more advanced version, could loop for iterative refinement
    return "synthesize"

# Build roleplay workflow
roleplay_workflow = StateGraph(RoleplayState)
roleplay_workflow.add_node("gather", gather_perspectives)
roleplay_workflow.add_node("analyze", analyze_convergence)
roleplay_workflow.add_node("synthesize", synthesize_perspectives)

roleplay_workflow.set_entry_point("gather")
roleplay_workflow.add_edge("gather", "analyze")

roleplay_workflow.add_conditional_edges(
    "analyze",
    should_continue_discussion,
    {
        "gather": "gather",  # Loop for more rounds
        "synthesize": "synthesize"
    }
)

roleplay_workflow.add_edge("synthesize", END)

roleplay_system = roleplay_workflow.compile()

# ==================== API ====================

def roleplay_analysis(
    scenario: str,
    roles: List[str] = None,
    max_rounds: int = 1
) -> dict:
    """
    Analyze scenario from multiple role perspectives.
    
    Args:
        scenario: Scenario to analyze
        roles: List of role names (from ROLES dict)
        max_rounds: Maximum discussion rounds
    """
    
    if roles is None:
        roles = ["cfo", "cto", "cmo"]
    
    # Validate roles
    roles = [r for r in roles if r in ROLES]
    
    if not roles:
        return {
            "success": False,
            "error": "No valid roles specified"
        }
    
    result = roleplay_system.invoke({
        "messages": [HumanMessage(content=scenario)],
        "scenario": scenario,
        "roles": roles,
        "perspectives": {},
        "discussion_round": 0,
        "max_rounds": max_rounds,
        "synthesis": ""
    })
    
    return {
        "success": True,
        "scenario": scenario,
        "roles": roles,
        "perspectives": result["perspectives"],
        "synthesis": result["synthesis"],
        "result": result["messages"][-1].content
    }

# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("ROLE-BASED AGENT SYSTEM DEMO")
    print("="*60)
    
    scenario = "We are considering implementing AI-powered customer service chatbots to reduce support costs by 40%."
    
    # Test with executive roles
    print(f"\n{'='*60}")
    print(f"SCENARIO: {scenario}")
    print(f"{'='*60}")
    
    result = roleplay_analysis(
        scenario,
        roles=["cfo", "cto", "cmo", "customer"],
        max_rounds=1
    )
    
    if result["success"]:
        print(f"\nRoles consulted: {', '.join([ROLES[r].name for r in result['roles']])}")
        print(f"\n{result['result']}")
    
    # Test with different role combination
    print(f"\n{'='*60}")
    print(f"DIFFERENT PERSPECTIVE SET")
    print(f"{'='*60}")
    
    scenario2 = "Should we invest in building our own infrastructure or use cloud services?"
    
    result2 = roleplay_analysis(
        scenario2,
        roles=["cfo", "cto", "pragmatist", "innovator"],
        max_rounds=1
    )
    
    if result2["success"]:
        print(f"\nScenario: {scenario2}")
        print(f"\nRoles consulted: {', '.join([ROLES[r].name for r in result2['roles']])}")
        print(f"\n{result2['result'][:500]}...")
```

---

## ⚖️ Part 2: Debate Frameworks - Pro-Con Analysis

### Theory: Debate Agents

#### What Are Debate Agents?

**Debate agents** are agents specifically designed to argue opposing sides of an issue. This creates a structured exploration of arguments for and against a position.

```
Proposition: "We should launch product now"
    │
    ├─→ PRO Agent: Arguments FOR launching now
    └─→ CON Agent: Arguments AGAINST launching now
```

#### Why Use Debate?

**1. Comprehensive Exploration:**
- Forces examination of both sides
- Uncovers arguments you might miss
- More thorough than single perspective

**2. Bias Reduction:**
- Challenges confirmation bias
- Presents alternative viewpoints
- Balances optimistic/pessimistic thinking

**3. Quality Decisions:**
- Better-informed choices
- Anticipates counterarguments
- Identifies risks and opportunities

**4. Persuasive Content:**
- Creates balanced content
- Addresses objections
- More credible communication

#### Debate Structure

**Classic Debate Format:**

**Round 1 - Opening Statements:**
```
PRO: Presents main arguments FOR
CON: Presents main arguments AGAINST
```

**Round 2 - Rebuttals:**
```
PRO: Responds to CON's arguments
CON: Responds to PRO's arguments
```

**Round 3 - Closing Arguments:**
```
PRO: Summarizes strongest points
CON: Summarizes strongest points
```

**Judge/Moderator:**
```
Evaluates both sides
Identifies strongest arguments
Makes recommendation
```

#### Pro-Con Analysis Pattern

**Structure:**
```
1. Define proposition clearly
2. PRO agent lists arguments for
3. CON agent lists arguments against
4. Both agents rebut each other
5. Judge synthesizes and recommends
```

**Example:**
```
Proposition: "Adopt microservices architecture"

PRO Arguments:
- Better scalability
- Independent deployment
- Technology flexibility
- Fault isolation

CON Arguments:
- Increased complexity
- Network overhead
- Harder debugging
- Operational burden

Synthesis: "Adopt for new services, keep monolith for core"
```

#### Debate Agent Characteristics

**PRO Agent:**
- Focuses on benefits, opportunities, advantages
- Emphasizes positive outcomes
- Addresses CON's points defensively
- Maintains optimistic tone while being realistic

**CON Agent:**
- Focuses on risks, costs, challenges
- Emphasizes potential problems
- Addresses PRO's points critically
- Maintains skeptical tone while being fair

**Judge/Moderator:**
- Evaluates both sides objectively
- Identifies strongest arguments
- Weighs trade-offs
- Makes balanced recommendation

#### When to Use Debate Agents

✅ **Use Debate Agents When:**
- Making important decisions
- Exploring controversial topics
- Creating balanced content
- Testing ideas rigorously
- Need to anticipate objections

❌ **Don't Use When:**
- Decision is straightforward
- Only one viable option
- Need quick answer
- Topic isn't debatable

---

### Implementation: Debate System

```python
from typing import TypedDict, Annotated, Sequence, List, Dict
from operator import add
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== DEBATE AGENT SYSTEM ====================

# Debater state
class DebaterState(TypedDict):
    """State for individual debater"""
    messages: Annotated[Sequence[BaseMessage], add]
    proposition: str
    side: str  # "pro" or "con"
    opponent_arguments: str
    argument: str

# Debate orchestration state
class DebateState(TypedDict):
    """State for debate orchestration"""
    messages: Annotated[Sequence[BaseMessage], add]
    proposition: str
    pro_arguments: Annotated[List[str], add]
    con_arguments: Annotated[List[str], add]
    debate_round: Annotated[int, add]
    max_rounds: int
    judgment: str

llm = ChatOllama(model="llama3.2", temperature=0.7)

# ==================== PRO AGENT ====================

PRO_SYSTEM_PROMPT = """You are a debate agent arguing FOR the proposition.

Your role:
- Present compelling arguments IN FAVOR of the proposition
- Highlight benefits, opportunities, and positive outcomes
- Address counterarguments constructively
- Be persuasive but honest
- Support claims with reasoning

Stay focused on the PRO side throughout the debate."""

def pro_agent_argue(state: DebaterState) -> dict:
    """Pro agent presents arguments"""
    logger.info("✅ PRO Agent: Presenting arguments")
    
    proposition = state["proposition"]
    opponent_args = state.get("opponent_arguments", "")
    
    if opponent_args:
        prompt = ChatPromptTemplate.from_messages([
            ("system", PRO_SYSTEM_PROMPT),
            ("human", """Proposition: {proposition}

CON side has argued:
{opponent_args}

Respond with your PRO arguments and rebuttals to the CON points:""")
        ])
        
        response = llm.invoke(
            prompt.format_messages(proposition=proposition, opponent_args=opponent_args)
        )
    else:
        prompt = ChatPromptTemplate.from_messages([
            ("system", PRO_SYSTEM_PROMPT),
            ("human", """Proposition: {proposition}

Present your main arguments FOR this proposition:""")
        ])
        
        response = llm.invoke(
            prompt.format_messages(proposition=proposition)
        )
    
    return {
        "argument": response.content,
        "messages": [AIMessage(content=f"[PRO] {response.content}")]
    }

pro_workflow = StateGraph(DebaterState)
pro_workflow.add_node("argue", pro_agent_argue)
pro_workflow.set_entry_point("argue")
pro_workflow.add_edge("argue", END)
pro_agent = pro_workflow.compile()

# ==================== CON AGENT ====================

CON_SYSTEM_PROMPT = """You are a debate agent arguing AGAINST the proposition.

Your role:
- Present compelling arguments AGAINST the proposition
- Highlight risks, costs, and negative outcomes
- Challenge PRO's arguments critically
- Be skeptical but fair
- Support claims with reasoning

Stay focused on the CON side throughout the debate."""

def con_agent_argue(state: DebaterState) -> dict:
    """Con agent presents arguments"""
    logger.info("❌ CON Agent: Presenting arguments")
    
    proposition = state["proposition"]
    opponent_args = state.get("opponent_arguments", "")
    
    if opponent_args:
        prompt = ChatPromptTemplate.from_messages([
            ("system", CON_SYSTEM_PROMPT),
            ("human", """Proposition: {proposition}

PRO side has argued:
{opponent_args}

Respond with your CON arguments and rebuttals to the PRO points:""")
        ])
        
        response = llm.invoke(
            prompt.format_messages(proposition=proposition, opponent_args=opponent_args)
        )
    else:
        prompt = ChatPromptTemplate.from_messages([
            ("system", CON_SYSTEM_PROMPT),
            ("human", """Proposition: {proposition}

Present your main arguments AGAINST this proposition:""")
        ])
        
        response = llm.invoke(
            prompt.format_messages(proposition=proposition)
        )
    
    return {
        "argument": response.content,
        "messages": [AIMessage(content=f"[CON] {response.content}")]
    }

con_workflow = StateGraph(DebaterState)
con_workflow.add_node("argue", con_agent_argue)
con_workflow.set_entry_point("argue")
con_workflow.add_edge("argue", END)
con_agent = con_workflow.compile()

# ==================== DEBATE ORCHESTRATOR ====================

def debate_round_pro(state: DebateState) -> dict:
    """Execute PRO agent's turn"""
    logger.info(f"🎤 Debate Round {state['debate_round'] + 1}: PRO speaks")
    
    # Get opponent's last argument if exists
    opponent_args = ""
    if state["con_arguments"]:
        opponent_args = state["con_arguments"][-1]
    
    result = pro_agent.invoke({
        "messages": [],
        "proposition": state["proposition"],
        "side": "pro",
        "opponent_arguments": opponent_args,
        "argument": ""
    })
    
    return {
        "pro_arguments": [result["argument"]],
        "debate_round": 1
    }

def debate_round_con(state: DebateState) -> dict:
    """Execute CON agent's turn"""
    logger.info(f"🎤 Debate Round {state['debate_round']}: CON speaks")
    
    # Get opponent's last argument
    opponent_args = ""
    if state["pro_arguments"]:
        opponent_args = state["pro_arguments"][-1]
    
    result = con_agent.invoke({
        "messages": [],
        "proposition": state["proposition"],
        "side": "con",
        "opponent_arguments": opponent_args,
        "argument": ""
    })
    
    return {
        "con_arguments": [result["argument"]]
    }

def judge_debate(state: DebateState) -> dict:
    """Judge evaluates both sides"""
    logger.info("⚖️ Judge: Evaluating debate")
    
    pro_args_text = "\n\n".join([
        f"PRO Round {i+1}:\n{arg}"
        for i, arg in enumerate(state["pro_arguments"])
    ])
    
    con_args_text = "\n\n".join([
        f"CON Round {i+1}:\n{arg}"
        for i, arg in enumerate(state["con_arguments"])
    ])
    
    judge_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an impartial judge evaluating a debate.

Your role:
- Evaluate both sides objectively
- Identify strongest arguments from each side
- Assess quality of reasoning
- Consider practical implications
- Provide balanced judgment
- Make recommendation based on merit of arguments"""),
        ("human", """Proposition: {proposition}

PRO Arguments:
{pro_args}

CON Arguments:
{con_args}

Provide your judgment including:
1. Summary of key points from each side
2. Evaluation of argument strength
3. Your recommendation
4. Reasoning for your decision

Judgment:""")
    ])
    
    chain = judge_prompt | llm
    response = chain.invoke({
        "proposition": state["proposition"],
        "pro_args": pro_args_text,
        "con_args": con_args_text
    })
    
    judgment = f"""DEBATE JUDGMENT

Proposition: {state["proposition"]}
Rounds: {state["debate_round"]}

{response.content}"""
    
    return {
        "judgment": judgment,
        "messages": [AIMessage(content=judgment)]
    }

def should_continue_debate(state: DebateState) -> str:
    """Decide if debate should continue"""
    
    if state["debate_round"] >= state["max_rounds"]:
        logger.info("Debate rounds complete")
        return "judge"
    
    return "pro"

# Build debate workflow
debate_workflow = StateGraph(DebateState)
debate_workflow.add_node("pro", debate_round_pro)
debate_workflow.add_node("con", debate_round_con)
debate_workflow.add_node("judge", judge_debate)

debate_workflow.set_entry_point("pro")
debate_workflow.add_edge("pro", "con")

debate_workflow.add_conditional_edges(
    "con",
    should_continue_debate,
    {
        "pro": "pro",  # Loop for another round
        "judge": "judge"
    }
)

debate_workflow.add_edge("judge", END)

debate_system = debate_workflow.compile()

# ==================== API ====================

def debate_proposition(
    proposition: str,
    rounds: int = 2
) -> dict:
    """
    Debate a proposition with PRO and CON agents.
    
    Args:
        proposition: Statement to debate
        rounds: Number of debate rounds
    """
    
    result = debate_system.invoke({
        "messages": [HumanMessage(content=proposition)],
        "proposition": proposition,
        "pro_arguments": [],
        "con_arguments": [],
        "debate_round": 0,
        "max_rounds": rounds,
        "judgment": ""
    })
    
    return {
        "success": True,
        "proposition": proposition,
        "rounds": rounds,
        "pro_arguments": result["pro_arguments"],
        "con_arguments": result["con_arguments"],
        "judgment": result["judgment"],
        "result": result["messages"][-1].content
    }

# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("DEBATE AGENT SYSTEM DEMO")
    print("="*60)
    
    propositions = [
        "Companies should mandate full return to office work",
        "We should build features using AI code generation tools",
    ]
    
    for prop in propositions:
        print(f"\n{'='*60}")
        print(f"PROPOSITION: {prop}")
        print(f"{'='*60}")
        
        result = debate_proposition(prop, rounds=2)
        
        if result["success"]:
            print(f"\nDebate rounds: {result['rounds']}")
            print(f"\n{result['result']}")
```

---

## 🔮 Part 3: Perspective-Taking and Simulation

### Theory: Perspective-Taking

#### What Is Perspective-Taking?

**Perspective-taking** is the ability to understand a situation from another person's or entity's point of view. In multi-agent systems, this means creating agents that genuinely adopt and maintain specific perspectives.

**Different from simple role-playing:**
- Not just "pretending" to be someone
- Deeply understanding motivations, constraints, priorities
- Seeing the world through that lens
- Making decisions as that entity would

#### Why Perspective-Taking Matters

**1. Empathy and Understanding:**
```
Without perspective-taking:
"Users complain about the interface"

With perspective-taking:
"As a 65-year-old user with limited tech experience, 
small fonts and complex navigation are genuinely frustrating"
```

**2. Better Decisions:**
- Anticipates stakeholder reactions
- Identifies unintended consequences
- Considers diverse needs

**3. Design and Testing:**
- Test products from user perspectives
- Identify usability issues
- Validate assumptions

**4. Conflict Resolution:**
- Understand all sides
- Find common ground
- Build bridges

#### Levels of Perspective-Taking

**Level 1: Surface Role**
```
Agent: "I am a customer"
→ Basic identity, no depth
```

**Level 2: Contextual Understanding**
```
Agent: "I am a customer who values price over features"
→ Priorities defined
```

**Level 3: Deep Perspective**
```
Agent: "I am a budget-conscious parent of three, working two jobs,
who needs reliable products but can't afford premium options.
Time is scarce, so I value things that 'just work'."
→ Full context, motivations, constraints
```

**Aim for Level 3 for authentic perspective-taking.**

#### Simulation vs. Prediction

**Prediction:**
```
"What will customers think about this?"
→ Guessing outcomes
```

**Simulation:**
```
Customer Agent: Experiences the product
Customer Agent: Reacts authentically from their perspective
Customer Agent: Provides genuine feedback
→ Simulating actual experience
```

**Simulation is more powerful because:**
- Uncovers unexpected reactions
- More nuanced insights
- Tests edge cases
- Reveals real problems

#### Perspective-Taking Techniques

**Technique 1: Persona-Based**
```
Create detailed personas with:
- Demographics
- Background
- Goals and motivations
- Pain points
- Constraints
- Values and priorities
```

**Technique 2: Stakeholder Mapping**
```
Identify all stakeholders:
- Who is affected?
- What do they care about?
- What are their constraints?
- How do they measure success?
```

**Technique 3: Day-in-the-Life**
```
Agent simulates:
- Typical day of the persona
- Interactions with product/service
- Real-world context
- Authentic reactions
```

**Technique 4: Constraint-Driven**
```
Agent operates under real constraints:
- Budget limitations
- Time pressure
- Technical literacy
- Access to resources
```

#### Authentic Perspective Maintenance

**Keys to authenticity:**

**1. Consistent Constraints:**
```
❌ Bad: "As a budget customer, I recommend the premium option"
✅ Good: "As a budget customer, this premium option is out of reach"
```

**2. Context-Aware Responses:**
```
❌ Bad: Generic feedback
✅ Good: Feedback specific to their situation
```

**3. Realistic Priorities:**
```
❌ Bad: Equally values all factors
✅ Good: Clear priority hierarchy based on perspective
```

**4. Authentic Language:**
```
❌ Bad: Technical jargon (from non-technical persona)
✅ Good: Language natural to that persona
```

---

### Implementation: Perspective-Taking Simulation

```python
from typing import TypedDict, Annotated, Sequence, List, Dict
from operator import add
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== PERSPECTIVE-TAKING SIMULATION ====================

@dataclass
class Persona:
    """Detailed persona for perspective-taking"""
    name: str
    age: int
    occupation: str
    background: str
    goals: List[str]
    pain_points: List[str]
    constraints: List[str]
    values: List[str]
    technical_literacy: str  # "low", "medium", "high"
    budget_sensitivity: str  # "low", "medium", "high"
    
    def get_system_prompt(self) -> str:
        """Generate detailed system prompt"""
        return f"""You are roleplaying as {self.name}.

BACKGROUND:
Age: {self.age}
Occupation: {self.occupation}
Background: {self.background}

GOALS:
{chr(10).join(f"- {goal}" for goal in self.goals)}

PAIN POINTS:
{chr(10).join(f"- {pain}" for pain in self.pain_points)}

CONSTRAINTS:
{chr(10).join(f"- {constraint}" for constraint in self.constraints)}

VALUES:
{chr(10).join(f"- {value}" for value in self.values)}

Technical Literacy: {self.technical_literacy}
Budget Sensitivity: {self.budget_sensitivity}

IMPORTANT: 
- Stay deeply in character
- Respond authentically from THIS perspective
- Consider YOUR specific constraints and priorities
- Use language natural to YOUR background
- Make decisions as YOU would, not generically"""

# Persona definitions
PERSONAS = {
    "busy_parent": Persona(
        name="Sarah",
        age=42,
        occupation="Working parent, office manager",
        background="Manages household of 4, works full-time, limited tech time",
        goals=[
            "Save time on daily tasks",
            "Stay organized with family schedule",
            "Keep family connected"
        ],
        pain_points=[
            "Constantly juggling work and family",
            "Limited free time",
            "Overwhelmed by too many apps/tools",
            "Need things that 'just work'"
        ],
        constraints=[
            "Max 30 minutes per day for new tools",
            "Budget: ~$50/month for subscriptions",
            "Must work on phone (rarely at computer)"
        ],
        values=[
            "Simplicity over features",
            "Reliability over innovation",
            "Time savings over cost savings"
        ],
        technical_literacy="medium",
        budget_sensitivity="high"
    ),
    "tech_enthusiast": Persona(
        name="Alex",
        age=28,
        occupation="Software developer",
        background="Early adopter, loves new tech, power user",
        goals=[
            "Use cutting-edge tools",
            "Maximize productivity",
            "Customize everything"
        ],
        pain_points=[
            "Frustrated by limited features",
            "Annoyed by 'dumbed down' interfaces",
            "Wants advanced options"
        ],
        constraints=[
            "Time to learn: unlimited",
            "Budget: flexible for quality tools",
            "Must integrate with existing workflow"
        ],
        values=[
            "Power and flexibility over simplicity",
            "Performance over ease of use",
            "Customization over defaults"
        ],
        technical_literacy="high",
        budget_sensitivity="low"
    ),
    "senior_user": Persona(
        name="Robert",
        age=72,
        occupation="Retired teacher",
        background="Adapting to technology later in life, wants to stay connected",
        goals=[
            "Stay in touch with grandchildren",
            "Manage finances and health",
            "Learn at own pace"
        ],
        pain_points=[
            "Small text and buttons difficult",
            "Confused by too many options",
            "Worried about making mistakes",
            "Feels left behind by tech"
        ],
        constraints=[
            "Limited tech support available",
            "Fixed income, careful with spending",
            "Vision and dexterity considerations",
            "Prefers step-by-step guidance"
        ],
        values=[
            "Clarity and simplicity above all",
            "Patient, supportive interfaces",
            "Safety and security",
            "Respect for learning curve"
        ],
        technical_literacy="low",
        budget_sensitivity="high"
    ),
    "small_business": Persona(
        name="Maria",
        age=35,
        occupation="Small business owner (local bakery)",
        background="Runs business solo, wearing many hats, needs efficiency",
        goals=[
            "Grow customer base",
            "Streamline operations",
            "Increase profitability"
        ],
        pain_points=[
            "No time for complex tools",
            "Limited budget for software",
            "Can't afford mistakes or downtime",
            "Need ROI to be obvious"
        ],
        constraints=[
            "Working 60+ hours/week",
            "Budget: minimal for tools",
            "Must see clear business value",
            "Can't risk disrupting current operations"
        ],
        values=[
            "Practical results over fancy features",
            "ROI must be clear and quick",
            "Reliability is critical",
            "Simplicity for efficiency"
        ],
        technical_literacy="medium",
        budget_sensitivity="very high"
    )
}

# Simulation state
class PersonaState(TypedDict):
    """State for persona agent"""
    messages: Annotated[Sequence[BaseMessage], add]
    persona_name: str
    scenario: str
    reaction: str

class SimulationState(TypedDict):
    """State for simulation orchestration"""
    messages: Annotated[Sequence[BaseMessage], add]
    product_scenario: str
    personas: List[str]
    reactions: Dict[str, str]
    insights: str

llm = ChatOllama(model="llama3.2", temperature=0.7)

# ==================== PERSONA AGENTS ====================

def create_persona_agent(persona: Persona):
    """Create agent for specific persona"""
    
    def persona_react(state: PersonaState) -> dict:
        """Persona reacts to scenario"""
        logger.info(f"👤 {persona.name}: Reacting to scenario")
        
        messages = [
            SystemMessage(content=persona.get_system_prompt()),
            HumanMessage(content=f"""SCENARIO:
{state['scenario']}

React to this scenario authentically from YOUR perspective as {persona.name}.

Consider:
- How does this fit YOUR needs and goals?
- What are YOUR concerns given YOUR constraints?
- How would YOU actually use this?
- What would YOU prioritize?

Your authentic reaction:""")
        ]
        
        response = llm.invoke(messages)
        
        return {
            "reaction": response.content,
            "messages": [AIMessage(content=f"[{persona.name}] {response.content}")]
        }
    
    # Build workflow
    persona_workflow = StateGraph(PersonaState)
    persona_workflow.add_node("react", persona_react)
    persona_workflow.set_entry_point("react")
    persona_workflow.add_edge("react", END)
    
    return persona_workflow.compile()

# Create agents for all personas
PERSONA_AGENTS = {
    name: create_persona_agent(persona)
    for name, persona in PERSONAS.items()
}

# ==================== SIMULATION ORCHESTRATOR ====================

def simulate_reactions(state: SimulationState) -> dict:
    """Simulate reactions from all personas"""
    logger.info("🎭 Simulation: Gathering persona reactions")
    
    scenario = state["product_scenario"]
    personas = state["personas"]
    
    reactions = {}
    
    for persona_name in personas:
        logger.info(f"   Simulating {persona_name}...")
        
        persona_agent = PERSONA_AGENTS[persona_name]
        
        result = persona_agent.invoke({
            "messages": [],
            "persona_name": persona_name,
            "scenario": scenario,
            "reaction": ""
        })
        
        reactions[persona_name] = result["reaction"]
    
    logger.info(f"✅ Collected {len(reactions)} persona reactions")
    
    return {"reactions": reactions}

def synthesize_insights(state: SimulationState) -> dict:
    """Synthesize insights from persona reactions"""
    logger.info("💡 Synthesizing insights")
    
    reactions_text = "\n\n".join([
        f"=== {PERSONAS[name].name} ({PERSONAS[name].occupation}) ===\n{reaction}"
        for name, reaction in state["reactions"].items()
    ])
    
    synthesis_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are analyzing user reactions to identify patterns, concerns, and opportunities."),
        ("human", """Product Scenario:
{scenario}

User Reactions from Different Perspectives:
{reactions}

Provide analysis including:
1. Common themes across personas
2. Unique concerns by persona type
3. Key insights for product design
4. Recommendations for improvement
5. Which personas are most/least satisfied and why

Analysis:""")
    ])
    
    chain = synthesis_prompt | llm
    response = chain.invoke({
        "scenario": state["product_scenario"],
        "reactions": reactions_text
    })
    
    insights = f"""PERSPECTIVE-TAKING SIMULATION RESULTS

Scenario: {state['product_scenario']}
Personas Simulated: {len(state['personas'])}

{response.content}"""
    
    return {
        "insights": insights,
        "messages": [AIMessage(content=insights)]
    }

# Build simulation workflow
simulation_workflow = StateGraph(SimulationState)
simulation_workflow.add_node("simulate", simulate_reactions)
simulation_workflow.add_node("synthesize", synthesize_insights)

simulation_workflow.set_entry_point("simulate")
simulation_workflow.add_edge("simulate", "synthesize")
simulation_workflow.add_edge("synthesize", END)

simulation_system = simulation_workflow.compile()

# ==================== API ====================

def simulate_perspectives(
    product_scenario: str,
    personas: List[str] = None
) -> dict:
    """
    Simulate reactions from multiple personas.
    
    Args:
        product_scenario: Product/feature to test
        personas: List of persona names (from PERSONAS dict)
    """
    
    if personas is None:
        personas = list(PERSONAS.keys())
    
    # Validate personas
    personas = [p for p in personas if p in PERSONAS]
    
    if not personas:
        return {
            "success": False,
            "error": "No valid personas specified"
        }
    
    result = simulation_system.invoke({
        "messages": [HumanMessage(content=product_scenario)],
        "product_scenario": product_scenario,
        "personas": personas,
        "reactions": {},
        "insights": ""
    })
    
    return {
        "success": True,
        "scenario": product_scenario,
        "personas": [PERSONAS[p].name for p in personas],
        "reactions": result["reactions"],
        "insights": result["insights"],
        "result": result["messages"][-1].content
    }

# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("PERSPECTIVE-TAKING SIMULATION DEMO")
    print("="*60)
    
    scenario = """We're launching a new mobile app for personal finance management.

Features:
- Automatic expense tracking
- Budget planning with AI suggestions
- Investment portfolio monitoring
- Bill payment reminders
- Financial goal tracking

Price: $12.99/month subscription"""
    
    result = simulate_perspectives(
        scenario,
        personas=["busy_parent", "tech_enthusiast", "senior_user", "small_business"]
    )
    
    if result["success"]:
        print(f"\nScenario: {scenario[:100]}...")
        print(f"\nPersonas simulated: {', '.join(result['personas'])}")
        print(f"\n{result['result']}")
```

---

## 🔄 Part 4: Iterative Refinement through Discussion

### Theory: Iterative Refinement

#### What Is Iterative Refinement?

**Iterative refinement** is the process of improving ideas, arguments, or solutions through multiple rounds of discussion and feedback.

```
Round 1: Initial ideas presented
    ↓
Round 2: Agents critique and respond
    ↓
Round 3: Ideas refined based on feedback
    ↓
Round 4: Converge toward better solution
```

#### Why Iterative Refinement?

**1. Quality Improvement:**
- First ideas rarely optimal
- Feedback reveals weaknesses
- Refinement strengthens arguments

**2. Convergence:**
- Moves from divergent to convergent thinking
- Finds common ground
- Builds consensus

**3. Comprehensive Exploration:**
- Uncovers edge cases
- Tests robustness
- Challenges assumptions

**4. Learning:**
- Agents learn from each other
- Knowledge synthesis
- Better understanding

#### Refinement Process

**Phase 1: Divergent Thinking**
```
Goal: Generate many ideas
Agents: Propose different approaches
Focus: Creativity, not criticism
Output: Multiple alternatives
```

**Phase 2: Critical Evaluation**
```
Goal: Identify strengths and weaknesses
Agents: Critique constructively
Focus: Finding problems
Output: Detailed feedback
```

**Phase 3: Convergent Refinement**
```
Goal: Improve and combine ideas
Agents: Propose improvements
Focus: Building on strengths
Output: Refined solutions
```

**Phase 4: Consensus Building**
```
Goal: Agree on best approach
Agents: Evaluate options
Focus: Finding agreement
Output: Final recommendation
```

#### Refinement Strategies

**Strategy 1: Successive Approximation**
```
Version 1.0 → Feedback → Version 1.1 → Feedback → Version 1.2
Each iteration gets closer to optimal
```

**Strategy 2: Multi-Agent Critique**
```
Proposal → Agent A critiques → Agent B critiques → Agent C critiques
Different perspectives reveal different issues
```

**Strategy 3: Thesis-Antithesis-Synthesis**
```
Thesis: Original idea
Antithesis: Opposing view
Synthesis: Combines best of both
→ Hegelian dialectic
```

**Strategy 4: Iterative Consensus**
```
Round 1: All propose solutions
Round 2: Discuss and critique
Round 3: Revised proposals
Round 4: Vote/converge
```

#### Convergence Criteria

**When to stop iterating?**

**Criterion 1: Agreement Threshold**
```
Stop when: >80% of agents agree
```

**Criterion 2: Diminishing Returns**
```
Stop when: Changes between iterations become minimal
```

**Criterion 3: Fixed Rounds**
```
Stop when: Completed N rounds (e.g., 3-5)
```

**Criterion 4: Quality Threshold**
```
Stop when: Solution quality score > threshold
```

#### Preventing Infinite Loops

**Problems:**
- Agents may never fully agree
- Discussion could continue indefinitely
- Some issues have no "correct" answer

**Solutions:**
1. **Maximum rounds limit** (e.g., 5 rounds max)
2. **Diminishing returns detection** (stop if changes < 5%)
3. **Timeout** (e.g., 5 minutes max)
4. **Forced consensus** (majority vote after max rounds)

---

### Implementation: Iterative Refinement System

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

# ==================== ITERATIVE REFINEMENT SYSTEM ====================

# Agent proposal state
class ProposalState(TypedDict):
    """State for agent proposals"""
    messages: Annotated[Sequence[BaseMessage], add]
    problem: str
    agent_role: str
    previous_proposals: str
    feedback_received: str
    proposal: str

# Refinement orchestration state
class RefinementState(TypedDict):
    """State for refinement orchestration"""
    messages: Annotated[Sequence[BaseMessage], add]
    problem_statement: str
    iteration: Annotated[int, add]
    max_iterations: int
    proposals_by_iteration: Dict[int, Dict[str, str]]
    feedback_by_iteration: Dict[int, Dict[str, str]]
    convergence_score: float
    final_solution: str

llm = ChatOllama(model="llama3.2", temperature=0.7)

# ==================== REFINING AGENTS ====================

# Innovator Agent
def innovator_propose(state: ProposalState) -> dict:
    """Innovator proposes creative solutions"""
    logger.info("💡 Innovator: Proposing solution")
    
    problem = state["problem"]
    previous = state.get("previous_proposals", "")
    feedback = state.get("feedback_received", "")
    
    if previous and feedback:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an innovator focused on creative, bold solutions."),
            ("human", """Problem: {problem}

Previous proposals from the team:
{previous}

Feedback received:
{feedback}

Refine your proposal incorporating the feedback:""")
        ])
        
        response = llm.invoke(
            prompt.format_messages(problem=problem, previous=previous, feedback=feedback)
        )
    else:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an innovator focused on creative, bold solutions."),
            ("human", """Problem: {problem}

Propose your innovative solution:""")
        ])
        
        response = llm.invoke(
            prompt.format_messages(problem=problem)
        )
    
    return {
        "proposal": response.content,
        "messages": [AIMessage(content=f"[Innovator] {response.content}")]
    }

innovator_workflow = StateGraph(ProposalState)
innovator_workflow.add_node("propose", innovator_propose)
innovator_workflow.set_entry_point("propose")
innovator_workflow.add_edge("propose", END)
innovator_agent = innovator_workflow.compile()

# Pragmatist Agent
def pragmatist_propose(state: ProposalState) -> dict:
    """Pragmatist proposes practical solutions"""
    logger.info("⚙️ Pragmatist: Proposing solution")
    
    problem = state["problem"]
    previous = state.get("previous_proposals", "")
    feedback = state.get("feedback_received", "")
    
    if previous and feedback:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a pragmatist focused on practical, achievable solutions."),
            ("human", """Problem: {problem}

Previous proposals from the team:
{previous}

Feedback received:
{feedback}

Refine your proposal incorporating the feedback:""")
        ])
        
        response = llm.invoke(
            prompt.format_messages(problem=problem, previous=previous, feedback=feedback)
        )
    else:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a pragmatist focused on practical, achievable solutions."),
            ("human", """Problem: {problem}

Propose your practical solution:""")
        ])
        
        response = llm.invoke(
            prompt.format_messages(problem=problem)
        )
    
    return {
        "proposal": response.content,
        "messages": [AIMessage(content=f"[Pragmatist] {response.content}")]
    }

pragmatist_workflow = StateGraph(ProposalState)
pragmatist_workflow.add_node("propose", pragmatist_propose)
pragmatist_workflow.set_entry_point("propose")
pragmatist_workflow.add_edge("propose", END)
pragmatist_agent = pragmatist_workflow.compile()

# Critic Agent
def critic_propose(state: ProposalState) -> dict:
    """Critic identifies issues and proposes improvements"""
    logger.info("🔍 Critic: Proposing solution")
    
    problem = state["problem"]
    previous = state.get("previous_proposals", "")
    feedback = state.get("feedback_received", "")
    
    if previous and feedback:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a critic who identifies problems and proposes careful, well-thought solutions."),
            ("human", """Problem: {problem}

Previous proposals from the team:
{previous}

Feedback received:
{feedback}

Refine your proposal addressing the concerns:""")
        ])
        
        response = llm.invoke(
            prompt.format_messages(problem=problem, previous=previous, feedback=feedback)
        )
    else:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a critic who identifies problems and proposes careful, well-thought solutions."),
            ("human", """Problem: {problem}

Propose your carefully considered solution:""")
        ])
        
        response = llm.invoke(
            prompt.format_messages(problem=problem)
        )
    
    return {
        "proposal": response.content,
        "messages": [AIMessage(content=f"[Critic] {response.content}")]
    }

critic_workflow = StateGraph(ProposalState)
critic_workflow.add_node("propose", critic_propose)
critic_workflow.set_entry_point("propose")
critic_workflow.add_edge("propose", END)
critic_agent = critic_workflow.compile()

REFINING_AGENTS = {
    "innovator": innovator_agent,
    "pragmatist": pragmatist_agent,
    "critic": critic_agent
}

# ==================== REFINEMENT ORCHESTRATOR ====================

def collect_proposals(state: RefinementState) -> dict:
    """Collect proposals from all agents"""
    iteration = state["iteration"] + 1
    logger.info(f"🔄 Iteration {iteration}: Collecting proposals")
    
    problem = state["problem_statement"]
    
    # Get previous iteration's proposals and feedback
    previous_proposals = ""
    feedback_received = ""
    
    if iteration > 1:
        prev_iter = iteration - 1
        if prev_iter in state["proposals_by_iteration"]:
            previous_proposals = "\n\n".join([
                f"{agent}: {proposal}"
                for agent, proposal in state["proposals_by_iteration"][prev_iter].items()
            ])
        
        if prev_iter in state["feedback_by_iteration"]:
            feedback_received = "\n\n".join([
                f"To {agent}: {feedback}"
                for agent, feedback in state["feedback_by_iteration"][prev_iter].items()
            ])
    
    # Collect new proposals
    proposals = {}
    
    for agent_name, agent in REFINING_AGENTS.items():
        logger.info(f"   {agent_name} proposing...")
        
        result = agent.invoke({
            "messages": [],
            "problem": problem,
            "agent_role": agent_name,
            "previous_proposals": previous_proposals,
            "feedback_received": feedback_received,
            "proposal": ""
        })
        
        proposals[agent_name] = result["proposal"]
    
    # Store proposals
    proposals_by_iteration = state["proposals_by_iteration"].copy()
    proposals_by_iteration[iteration] = proposals
    
    logger.info(f"✅ Collected {len(proposals)} proposals")
    
    return {
        "proposals_by_iteration": proposals_by_iteration,
        "iteration": 1
    }

def generate_feedback(state: RefinementState) -> dict:
    """Generate feedback on proposals"""
    iteration = state["iteration"]
    logger.info(f"💬 Iteration {iteration}: Generating feedback")
    
    current_proposals = state["proposals_by_iteration"][iteration]
    
    # Each agent provides feedback to others
    feedback_by_iteration = state["feedback_by_iteration"].copy()
    feedback = {}
    
    proposals_text = "\n\n".join([
        f"{agent.upper()}:\n{proposal}"
        for agent, proposal in current_proposals.items()
    ])
    
    feedback_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are evaluating proposals and providing constructive feedback."),
        ("human", """Problem: {problem}

Current proposals:
{proposals}

Provide brief constructive feedback for each proposal.

Feedback:""")
    ])
    
    chain = feedback_prompt | llm
    response = chain.invoke({
        "problem": state["problem_statement"],
        "proposals": proposals_text
    })
    
    # Parse feedback (simplified - just use one feedback for all)
    feedback_text = response.content
    
    for agent_name in current_proposals.keys():
        feedback[agent_name] = feedback_text
    
    feedback_by_iteration[iteration] = feedback
    
    return {
        "feedback_by_iteration": feedback_by_iteration
    }

def evaluate_convergence(state: RefinementState) -> dict:
    """Evaluate if proposals are converging"""
    iteration = state["iteration"]
    logger.info(f"📊 Iteration {iteration}: Evaluating convergence")
    
    if iteration < 2:
        # Need at least 2 iterations to compare
        return {"convergence_score": 0.0}
    
    current_proposals = state["proposals_by_iteration"][iteration]
    previous_proposals = state["proposals_by_iteration"][iteration - 1]
    
    # Compare proposals using LLM
    current_text = "\n\n".join([
        f"{agent}: {proposal[:200]}..."
        for agent, proposal in current_proposals.items()
    ])
    
    previous_text = "\n\n".join([
        f"{agent}: {proposal[:200]}..."
        for agent, proposal in previous_proposals.items()
    ])
    
    convergence_prompt = ChatPromptTemplate.from_messages([
        ("system", "Evaluate if proposals are converging toward a common solution."),
        ("human", """Previous iteration:
{previous}

Current iteration:
{current}

Rate convergence from 0.0 to 1.0 where:
- 0.0 = Completely different, no convergence
- 0.5 = Some commonalities emerging
- 1.0 = Proposals are essentially the same

Respond with ONLY a number between 0.0 and 1.0.

Convergence score:""")
    ])
    
    chain = convergence_prompt | llm
    response = chain.invoke({
        "previous": previous_text,
        "current": current_text
    })
    
    # Parse score
    try:
        import re
        numbers = re.findall(r'0?\.\d+|1\.0|0|1', response.content)
        score = float(numbers[0]) if numbers else 0.5
    except:
        score = 0.5
    
    logger.info(f"Convergence score: {score:.2f}")
    
    return {"convergence_score": score}

def synthesize_solution(state: RefinementState) -> dict:
    """Synthesize final solution from refined proposals"""
    logger.info("🔗 Synthesizing final solution")
    
    final_iteration = state["iteration"]
    final_proposals = state["proposals_by_iteration"][final_iteration]
    
    proposals_text = "\n\n".join([
        f"=== {agent.upper()} ===\n{proposal}"
        for agent, proposal in final_proposals.items()
    ])
    
    synthesis_prompt = ChatPromptTemplate.from_messages([
        ("system", "Synthesize the best elements from all proposals into a unified solution."),
        ("human", """Problem: {problem}

Final refined proposals after {iterations} iterations:
{proposals}

Create a synthesized solution that combines the best ideas:""")
    ])
    
    chain = synthesis_prompt | llm
    response = chain.invoke({
        "problem": state["problem_statement"],
        "iterations": final_iteration,
        "proposals": proposals_text
    })
    
    final_solution = f"""ITERATIVE REFINEMENT RESULT

Problem: {state['problem_statement']}
Iterations: {final_iteration}
Convergence Score: {state['convergence_score']:.2f}

Final Synthesized Solution:
{response.content}"""
    
    return {
        "final_solution": final_solution,
        "messages": [AIMessage(content=final_solution)]
    }

def should_continue_refinement(state: RefinementState) -> str:
    """Decide if refinement should continue"""
    
    # Check max iterations
    if state["iteration"] >= state["max_iterations"]:
        logger.info("Max iterations reached")
        return "synthesize"
    
    # Check convergence
    if state["convergence_score"] >= 0.8:
        logger.info(f"High convergence achieved ({state['convergence_score']:.2f})")
        return "synthesize"
    
    logger.info(f"Continuing refinement (convergence: {state['convergence_score']:.2f})")
    return "collect"

# Build refinement workflow
refinement_workflow = StateGraph(RefinementState)
refinement_workflow.add_node("collect", collect_proposals)
refinement_workflow.add_node("feedback", generate_feedback)
refinement_workflow.add_node("evaluate", evaluate_convergence)
refinement_workflow.add_node("synthesize", synthesize_solution)

refinement_workflow.set_entry_point("collect")
refinement_workflow.add_edge("collect", "feedback")
refinement_workflow.add_edge("feedback", "evaluate")

refinement_workflow.add_conditional_edges(
    "evaluate",
    should_continue_refinement,
    {
        "collect": "collect",  # Loop for another iteration
        "synthesize": "synthesize"
    }
)

refinement_workflow.add_edge("synthesize", END)

refinement_system = refinement_workflow.compile()

# ==================== API ====================

def iterative_refinement(
    problem: str,
    max_iterations: int = 3
) -> dict:
    """
    Refine solution through iterative discussion.
    
    Args:
        problem: Problem statement
        max_iterations: Maximum refinement iterations
    """
    
    result = refinement_system.invoke({
        "messages": [HumanMessage(content=problem)],
        "problem_statement": problem,
        "iteration": 0,
        "max_iterations": max_iterations,
        "proposals_by_iteration": {},
        "feedback_by_iteration": {},
        "convergence_score": 0.0,
        "final_solution": ""
    })
    
    return {
        "success": True,
        "problem": problem,
        "iterations": result["iteration"],
        "convergence_score": result["convergence_score"],
        "final_solution": result["final_solution"],
        "result": result["messages"][-1].content
    }

# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("ITERATIVE REFINEMENT DEMO")
    print("="*60)
    
    problem = "Design an employee onboarding program that reduces time-to-productivity by 50%"
    
    result = iterative_refinement(problem, max_iterations=3)
    
    if result["success"]:
        print(f"\nProblem: {result['problem']}")
        print(f"Iterations: {result['iterations']}")
        print(f"Convergence: {result['convergence_score']:.2f}")
        print(f"\n{result['result']}")
```

---

## 📚 Part 5: Use Cases - Decision Making and Content Creation

### Theory: Practical Applications

#### Use Case 1: Decision Making

**Scenario: Strategic Business Decisions**

**Without Roleplay/Debate:**
```
Manager: "I think we should do X"
Team: "Okay"
→ Limited perspective, potential blind spots
```

**With Roleplay/Debate:**
```
CFO Agent: Financial implications
CTO Agent: Technical feasibility
CMO Agent: Market impact
Customer Agent: User perspective
→ Comprehensive analysis from all angles
```

**Benefits:**
- More informed decisions
- Anticipates objections
- Reduces risk of oversight
- Better buy-in (all perspectives considered)

**Example Applications:**
- Product launch decisions
- Investment choices
- Organizational changes
- Strategic direction

---

#### Use Case 2: Content Creation

**Scenario: Writing Balanced Articles**

**Without Debate:**
```
Writer: Creates article from one perspective
→ May be biased, miss counterarguments
```

**With Debate:**
```
PRO Agent: Arguments supporting thesis
CON Agent: Arguments challenging thesis
Editor: Balances both perspectives
→ More credible, comprehensive content
```

**Benefits:**
- More balanced content
- Addresses reader objections
- Higher credibility
- Better persuasion

**Example Applications:**
- Opinion pieces
- Product reviews
- Policy analysis
- Educational content

---

#### Use Case 3: User Testing

**Scenario: Product Design Validation**

**Without Persona Simulation:**
```
Designer: "I think users will like this"
→ Assumption-based, designer's perspective
```

**With Persona Simulation:**
```
Senior User Agent: "Text is too small"
Tech Enthusiast Agent: "Needs more features"
Busy Parent Agent: "Too complicated"
→ Real user concerns identified early
```

**Benefits:**
- Early issue detection
- Multiple user types considered
- Cheaper than real user testing
- Faster iteration

**Example Applications:**
- UI/UX design
- Feature prioritization
- Accessibility testing
- Market research

---

#### Use Case 4: Training and Education

**Scenario: Soft Skills Training**

**Traditional Training:**
```
Lecture: "Here's how to handle difficult conversations"
→ Theoretical, not experiential
```

**With Roleplay Agents:**
```
Trainee: Practices conversation
Difficult Customer Agent: Simulates challenges
Coach Agent: Provides feedback
→ Safe practice environment
```

**Benefits:**
- Experiential learning
- Safe to make mistakes
- Immediate feedback
- Scalable training

**Example Applications:**
- Sales training
- Conflict resolution
- Leadership development
- Customer service training

---

### Comparison: When to Use Each Pattern

| Need | Pattern | Example |
|------|---------|---------|
| **Multiple expert views** | Role-Based | Product launch (CFO, CTO, CMO perspectives) |
| **Explore both sides** | Debate | "Should we outsource?" (PRO vs CON) |
| **Test with users** | Persona Simulation | UI design (different user types) |
| **Improve through discussion** | Iterative Refinement | Marketing strategy (refine over rounds) |

---

## 🎯 Best Practices

### 1. Role Design

**✅ DO:**
- Create detailed, specific roles
- Include constraints and priorities
- Maintain consistency
- Use authentic language

**❌ DON'T:**
- Use vague, generic roles
- Let roles break character
- Mix professional and personality roles inconsistently
- Forget role limitations

---

### 2. Debate Structure

**✅ DO:**
- Define proposition clearly
- Give both sides equal opportunity
- Use impartial judge
- Set round limits

**❌ DON'T:**
- Use biased framing
- Let one side dominate
- Forget to synthesize
- Continue indefinitely

---

### 3. Simulation

**✅ DO:**
- Use realistic personas
- Include constraints
- Test edge cases
- Synthesize findings

**❌ DON'T:**
- Use stereotypes
- Ignore limitations
- Test only ideal users
- Take feedback literally without analysis

---

### 4. Iteration

**✅ DO:**
- Set maximum iterations
- Check convergence
- Provide constructive feedback
- Synthesize final solution

**❌ DON'T:**
- Iterate indefinitely
- Provide only criticism
- Ignore convergence signals
- Stop too early

---

## ✅ Chapter 14 Complete!

**You now understand:**
- ✅ Role-based agent systems (identity, perspective, consistency)
- ✅ Debate frameworks (PRO-CON analysis, structured argumentation)
- ✅ Perspective-taking and simulation (personas, authentic reactions)
- ✅ Iterative refinement (convergence, feedback loops)
- ✅ Use cases (decision making, content creation, testing, training)
- ✅ When to use each pattern
- ✅ Best practices for implementation

**Ready for Chapter 15?**

**Chapter 15: Competitive & Game-Theoretic Agents** will cover:
- Adversarial agent design
- Game theory in multi-agent systems
- Competitive auction/bidding systems
- Red team / Blue team patterns
- Nash equilibrium in agent interactions

Just say "Continue to Chapter 15" when ready!