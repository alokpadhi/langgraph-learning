# Chapter 15: Competitive & Game-Theoretic Agents

## 🎮 Introduction: Competition in Multi-Agent Systems

**Competitive multi-agent systems** are fundamentally different from cooperative systems. Instead of working together toward shared goals, agents:
- Compete for resources
- Pursue conflicting objectives
- Try to outperform each other
- May work against each other

Think of **markets, games, security** - contexts where competition drives behavior.

**Key Distinction:**

```
Cooperative: All agents win together or lose together
Competitive: One agent's gain may be another's loss
Mixed: Some cooperation, some competition (most realistic)
```

---

## ⚔️ Part 1: Adversarial Agent Design

### Theory: Adversarial Agents

#### What Are Adversarial Agents?

**Adversarial agents** are designed to work **against** another agent or system. Their goal is to:
- Find weaknesses
- Exploit vulnerabilities
- Defeat opponents
- Maximize their own outcome at the expense of others

**Not necessarily "evil"** - adversarial agents serve important purposes:
- Security testing (find vulnerabilities before attackers do)
- Quality assurance (stress test systems)
- Training (create challenging opponents)
- Strategic planning (anticipate adversary moves)

#### Types of Adversarial Relationships

**1. Zero-Sum Competition:**
```
Agent A's gain = Agent B's loss
Example: Chess - one wins, one loses
Total outcome: +1 + (-1) = 0
```

**2. Negative-Sum Competition:**
```
Both agents lose in aggregate
Example: Arms race - both waste resources
Total outcome: -5 + (-3) = -8
```

**3. Positive-Sum with Competition:**
```
Both can gain, but compete for larger share
Example: Business competition - market grows, but fight for share
Total outcome: +5 + +3 = +8 (both positive, but competing)
```

#### Adversarial Agent Design Principles

**Principle 1: Know Your Opponent**
```
Effective adversaries model their opponent:
- What are they trying to achieve?
- What are their weaknesses?
- How do they make decisions?
- What patterns do they follow?
```

**Principle 2: Adaptive Strategy**
```
Static strategies fail against smart opponents
Must adapt:
- Observe opponent behavior
- Update strategy model
- Exploit discovered patterns
- Avoid becoming predictable
```

**Principle 3: Resource Management**
```
Adversarial competition requires resources:
- Time
- Energy/computation
- Information
- Position/advantage

Manage resources strategically
```

**Principle 4: Risk Assessment**
```
Aggressive moves may backfire
Balance:
- Expected gain
- Probability of success
- Cost of failure
- Opponent's likely response
```

#### Adversarial Strategies

**Strategy 1: Aggressive/Exploitative**
```
Maximize immediate advantage
Attack opponent weaknesses
High risk, high reward
Example: All-in poker bet
```

**Strategy 2: Defensive/Conservative**
```
Minimize losses
Protect weaknesses
Low risk, low reward
Example: Castle defense in chess
```

**Strategy 3: Adaptive/Balanced**
```
Mix strategies based on context
Respond to opponent behavior
Medium risk, medium reward
Example: Tit-for-tat in iterated games
```

**Strategy 4: Deceptive/Misleading**
```
Misdirect opponent
Hide true intentions
Create false patterns
Example: Feint in boxing
```

#### Adversarial Design Patterns

**Pattern 1: Attacker-Defender**
```
Attacker: Tries to breach system
Defender: Tries to protect system
Example: Security testing
```

**Pattern 2: Competing Optimizers**
```
Both optimize for same scarce resource
Example: Bidding for limited supply
```

**Pattern 3: Predator-Prey**
```
One agent hunts, other evades
Example: Anti-virus vs malware
```

**Pattern 4: Strategic Opponents**
```
Both plan multiple moves ahead
Example: Chess engines
```

#### When to Use Adversarial Agents

✅ **Use Adversarial Agents When:**
- Testing system robustness
- Security validation
- Training competitive skills
- Strategy development
- Realistic simulation of conflicts

❌ **Don't Use When:**
- Cooperation would achieve better outcomes
- Adversarial dynamic is unnecessary
- Risk of destructive behavior
- Need trust and collaboration

---

### Implementation: Adversarial Agent System

```python
from typing import TypedDict, Annotated, Sequence, List, Dict, Literal
from operator import add
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from dataclasses import dataclass
import random
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== ADVERSARIAL AGENT SYSTEM ====================

@dataclass
class AgentStatus:
    """Track agent status in competition"""
    name: str
    health: int  # 0-100
    resources: int
    position: str
    score: int

# Adversarial agent state
class AdversarialAgentState(TypedDict):
    """State for adversarial agent"""
    messages: Annotated[Sequence[BaseMessage], add]
    agent_name: str
    agent_status: AgentStatus
    opponent_status: AgentStatus
    game_context: str
    history: List[str]
    action: str
    strategy: str

# Game orchestration state
class AdversarialGameState(TypedDict):
    """State for adversarial game"""
    messages: Annotated[Sequence[BaseMessage], add]
    game_type: str
    agent_a_status: AgentStatus
    agent_b_status: AgentStatus
    turn_count: Annotated[int, add]
    max_turns: int
    action_history: Annotated[List[str], add]
    winner: str
    outcome_description: str

llm = ChatOllama(model="llama3.2", temperature=0.8)

# ==================== ATTACKER AGENT ====================

ATTACKER_SYSTEM = """You are an ATTACKER agent in a security simulation.

Your goal: Find and exploit vulnerabilities in the defender's system.

Capabilities:
- Probe for weaknesses
- Launch attacks
- Adapt based on defender responses
- Use resources strategically

You have {resources} resources and {health} health.

Be strategic, adaptive, and persistent."""

def attacker_decide(state: AdversarialAgentState) -> dict:
    """Attacker decides action"""
    logger.info("⚔️ ATTACKER: Deciding action")
    
    agent_status = state["agent_status"]
    opponent_status = state["opponent_status"]
    history = state.get("history", [])
    
    history_text = "\n".join(history[-5:]) if history else "No history yet"
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", ATTACKER_SYSTEM.format(
            resources=agent_status.resources,
            health=agent_status.health
        )),
        ("human", """Current Situation:
Your status: Health {your_health}, Resources {your_resources}, Score {your_score}
Defender status: Health {opp_health}, Resources {opp_resources}, Score {opp_score}

Recent history:
{history}

Choose your action. Options:
1. PROBE: Scan for vulnerabilities (cost: 5 resources)
2. LIGHT_ATTACK: Quick attack (cost: 10 resources, damage: 10-20)
3. HEAVY_ATTACK: Powerful attack (cost: 20 resources, damage: 20-40)
4. RECHARGE: Restore resources (gain: 15 resources)

Respond with: ACTION: [choice] | REASONING: [your reasoning]

Your decision:""")
    ])
    
    response = llm.invoke(
        prompt.format_messages(
            your_health=agent_status.health,
            your_resources=agent_status.resources,
            your_score=agent_status.score,
            opp_health=opponent_status.health,
            opp_resources=opponent_status.resources,
            opp_score=opponent_status.score,
            history=history_text
        )
    )
    
    # Parse action
    content = response.content
    action = "PROBE"  # default
    
    if "ACTION:" in content:
        action_part = content.split("ACTION:")[1].split("|")[0].strip()
        action = action_part.split()[0].upper()
    
    logger.info(f"⚔️ ATTACKER chose: {action}")
    
    return {
        "action": action,
        "strategy": content,
        "messages": [AIMessage(content=f"[ATTACKER] {action}")]
    }

attacker_workflow = StateGraph(AdversarialAgentState)
attacker_workflow.add_node("decide", attacker_decide)
attacker_workflow.set_entry_point("decide")
attacker_workflow.add_edge("decide", END)
attacker_agent = attacker_workflow.compile()

# ==================== DEFENDER AGENT ====================

DEFENDER_SYSTEM = """You are a DEFENDER agent in a security simulation.

Your goal: Protect your system from the attacker.

Capabilities:
- Detect attacks
- Block attacks
- Counterattack
- Reinforce defenses

You have {resources} resources and {health} health.

Be vigilant, strategic, and adaptive."""

def defender_decide(state: AdversarialAgentState) -> dict:
    """Defender decides action"""
    logger.info("🛡️ DEFENDER: Deciding action")
    
    agent_status = state["agent_status"]
    opponent_status = state["opponent_status"]
    history = state.get("history", [])
    
    history_text = "\n".join(history[-5:]) if history else "No history yet"
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", DEFENDER_SYSTEM.format(
            resources=agent_status.resources,
            health=agent_status.health
        )),
        ("human", """Current Situation:
Your status: Health {your_health}, Resources {your_resources}, Score {your_score}
Attacker status: Health {opp_health}, Resources {opp_resources}, Score {opp_score}

Recent history:
{history}

Choose your action. Options:
1. MONITOR: Watch for attacks (cost: 5 resources)
2. BLOCK: Defend against attacks (cost: 10 resources, blocks 15-25 damage)
3. COUNTERATTACK: Strike back (cost: 15 resources, damage: 10-20)
4. REINFORCE: Strengthen defenses (cost: 20 resources, gain: +10 health)

Respond with: ACTION: [choice] | REASONING: [your reasoning]

Your decision:""")
    ])
    
    response = llm.invoke(
        prompt.format_messages(
            your_health=agent_status.health,
            your_resources=agent_status.resources,
            your_score=agent_status.score,
            opp_health=opponent_status.health,
            opp_resources=opponent_status.resources,
            opp_score=opponent_status.score,
            history=history_text
        )
    )
    
    # Parse action
    content = response.content
    action = "MONITOR"  # default
    
    if "ACTION:" in content:
        action_part = content.split("ACTION:")[1].split("|")[0].strip()
        action = action_part.split()[0].upper()
    
    logger.info(f"🛡️ DEFENDER chose: {action}")
    
    return {
        "action": action,
        "strategy": content,
        "messages": [AIMessage(content=f"[DEFENDER] {action}")]
    }

defender_workflow = StateGraph(AdversarialAgentState)
defender_workflow.add_node("decide", defender_decide)
defender_workflow.set_entry_point("decide")
defender_workflow.add_edge("decide", END)
defender_agent = defender_workflow.compile()

# ==================== GAME ORCHESTRATOR ====================

def execute_turn(state: AdversarialGameState) -> dict:
    """Execute one turn of adversarial game"""
    turn = state["turn_count"] + 1
    logger.info(f"🎮 Turn {turn}: Executing")
    
    agent_a_status = state["agent_a_status"]
    agent_b_status = state["agent_b_status"]
    
    # Agent A (Attacker) acts
    attacker_result = attacker_agent.invoke({
        "messages": [],
        "agent_name": "attacker",
        "agent_status": agent_a_status,
        "opponent_status": agent_b_status,
        "game_context": "",
        "history": state["action_history"],
        "action": "",
        "strategy": ""
    })
    
    attacker_action = attacker_result["action"]
    
    # Agent B (Defender) acts
    defender_result = defender_agent.invoke({
        "messages": [],
        "agent_name": "defender",
        "agent_status": agent_b_status,
        "opponent_status": agent_a_status,
        "game_context": "",
        "history": state["action_history"],
        "action": "",
        "strategy": ""
    })
    
    defender_action = defender_result["action"]
    
    # Resolve actions
    action_log = []
    
    # Process attacker action
    if attacker_action == "PROBE":
        if agent_a_status.resources >= 5:
            agent_a_status.resources -= 5
            agent_a_status.score += 2
            action_log.append("ATTACKER probed defenses")
        else:
            action_log.append("ATTACKER tried to probe but lacks resources")
    
    elif attacker_action == "LIGHT_ATTACK":
        if agent_a_status.resources >= 10:
            agent_a_status.resources -= 10
            damage = random.randint(10, 20)
            
            if defender_action == "BLOCK" and agent_b_status.resources >= 10:
                block_amount = random.randint(15, 25)
                actual_damage = max(0, damage - block_amount)
                agent_b_status.health -= actual_damage
                agent_a_status.score += actual_damage
                action_log.append(f"ATTACKER light attack ({damage}) - DEFENDER blocked ({block_amount}) - Net damage: {actual_damage}")
            else:
                agent_b_status.health -= damage
                agent_a_status.score += damage
                action_log.append(f"ATTACKER light attack hit for {damage} damage")
        else:
            action_log.append("ATTACKER tried light attack but lacks resources")
    
    elif attacker_action == "HEAVY_ATTACK":
        if agent_a_status.resources >= 20:
            agent_a_status.resources -= 20
            damage = random.randint(20, 40)
            
            if defender_action == "BLOCK" and agent_b_status.resources >= 10:
                block_amount = random.randint(15, 25)
                actual_damage = max(0, damage - block_amount)
                agent_b_status.health -= actual_damage
                agent_a_status.score += actual_damage
                action_log.append(f"ATTACKER heavy attack ({damage}) - DEFENDER blocked ({block_amount}) - Net damage: {actual_damage}")
            else:
                agent_b_status.health -= damage
                agent_a_status.score += damage
                action_log.append(f"ATTACKER heavy attack hit for {damage} damage")
        else:
            action_log.append("ATTACKER tried heavy attack but lacks resources")
    
    elif attacker_action == "RECHARGE":
        agent_a_status.resources += 15
        action_log.append("ATTACKER recharged resources")
    
    # Process defender action
    if defender_action == "MONITOR":
        if agent_b_status.resources >= 5:
            agent_b_status.resources -= 5
            agent_b_status.score += 1
            action_log.append("DEFENDER monitored for threats")
        else:
            action_log.append("DEFENDER tried to monitor but lacks resources")
    
    elif defender_action == "BLOCK":
        if agent_b_status.resources >= 10:
            agent_b_status.resources -= 10
            # Block already processed in attacker actions
            pass
        else:
            action_log.append("DEFENDER tried to block but lacks resources")
    
    elif defender_action == "COUNTERATTACK":
        if agent_b_status.resources >= 15:
            agent_b_status.resources -= 15
            damage = random.randint(10, 20)
            agent_a_status.health -= damage
            agent_b_status.score += damage
            action_log.append(f"DEFENDER counterattacked for {damage} damage")
        else:
            action_log.append("DEFENDER tried to counterattack but lacks resources")
    
    elif defender_action == "REINFORCE":
        if agent_b_status.resources >= 20:
            agent_b_status.resources -= 20
            agent_b_status.health = min(100, agent_b_status.health + 10)
            action_log.append("DEFENDER reinforced defenses (+10 health)")
        else:
            action_log.append("DEFENDER tried to reinforce but lacks resources")
    
    # Ensure health doesn't go negative
    agent_a_status.health = max(0, agent_a_status.health)
    agent_b_status.health = max(0, agent_b_status.health)
    
    # Natural resource generation
    agent_a_status.resources = min(100, agent_a_status.resources + 5)
    agent_b_status.resources = min(100, agent_b_status.resources + 5)
    
    turn_summary = f"Turn {turn}: " + " | ".join(action_log)
    
    logger.info(f"Turn complete: A health={agent_a_status.health}, B health={agent_b_status.health}")
    
    return {
        "agent_a_status": agent_a_status,
        "agent_b_status": agent_b_status,
        "turn_count": 1,
        "action_history": [turn_summary]
    }

def check_winner(state: AdversarialGameState) -> dict:
    """Check if game is over"""
    
    agent_a = state["agent_a_status"]
    agent_b = state["agent_b_status"]
    
    winner = ""
    outcome = ""
    
    # Check if either agent is defeated
    if agent_a.health <= 0:
        winner = "DEFENDER"
        outcome = f"DEFENDER wins! ATTACKER defeated after {state['turn_count']} turns."
    elif agent_b.health <= 0:
        winner = "ATTACKER"
        outcome = f"ATTACKER wins! DEFENDER defeated after {state['turn_count']} turns."
    elif state["turn_count"] >= state["max_turns"]:
        # Time limit - winner by score
        if agent_a.score > agent_b.score:
            winner = "ATTACKER"
            outcome = f"ATTACKER wins on points ({agent_a.score} vs {agent_b.score}) after {state['turn_count']} turns."
        elif agent_b.score > agent_a.score:
            winner = "DEFENDER"
            outcome = f"DEFENDER wins on points ({agent_b.score} vs {agent_a.score}) after {state['turn_count']} turns."
        else:
            winner = "DRAW"
            outcome = f"Draw! Both scored {agent_a.score} points after {state['turn_count']} turns."
    
    if winner:
        logger.info(f"🏆 Game Over: {outcome}")
    
    return {
        "winner": winner,
        "outcome_description": outcome
    }

def format_game_result(state: AdversarialGameState) -> dict:
    """Format final game result"""
    
    history_text = "\n".join(state["action_history"])
    
    result = f"""ADVERSARIAL GAME RESULT

Game Type: Attacker vs Defender
Turns: {state['turn_count']}

Final Status:
ATTACKER: Health {state['agent_a_status'].health}, Score {state['agent_a_status'].score}
DEFENDER: Health {state['agent_b_status'].health}, Score {state['agent_b_status'].score}

{state['outcome_description']}

Action History:
{history_text}"""
    
    return {
        "messages": [AIMessage(content=result)]
    }

def should_continue_game(state: AdversarialGameState) -> str:
    """Decide if game should continue"""
    
    if state["winner"]:
        return "format"
    
    return "execute"

# Build adversarial game workflow
adversarial_workflow = StateGraph(AdversarialGameState)
adversarial_workflow.add_node("execute", execute_turn)
adversarial_workflow.add_node("check", check_winner)
adversarial_workflow.add_node("format", format_game_result)

adversarial_workflow.set_entry_point("execute")
adversarial_workflow.add_edge("execute", "check")

adversarial_workflow.add_conditional_edges(
    "check",
    should_continue_game,
    {
        "execute": "execute",  # Loop
        "format": "format"
    }
)

adversarial_workflow.add_edge("format", END)

adversarial_game_system = adversarial_workflow.compile()

# ==================== API ====================

def run_adversarial_game(
    game_type: str = "attacker_defender",
    max_turns: int = 10
) -> dict:
    """
    Run adversarial game between agents.
    
    Args:
        game_type: Type of adversarial game
        max_turns: Maximum turns before timeout
    """
    
    # Initialize agents
    agent_a = AgentStatus(
        name="ATTACKER",
        health=100,
        resources=50,
        position="attacking",
        score=0
    )
    
    agent_b = AgentStatus(
        name="DEFENDER",
        health=100,
        resources=50,
        position="defending",
        score=0
    )
    
    result = adversarial_game_system.invoke({
        "messages": [HumanMessage(content=f"Starting {game_type} game")],
        "game_type": game_type,
        "agent_a_status": agent_a,
        "agent_b_status": agent_b,
        "turn_count": 0,
        "max_turns": max_turns,
        "action_history": [],
        "winner": "",
        "outcome_description": ""
    })
    
    return {
        "success": True,
        "game_type": game_type,
        "turns": result["turn_count"],
        "winner": result["winner"],
        "attacker_final": {
            "health": result["agent_a_status"].health,
            "score": result["agent_a_status"].score
        },
        "defender_final": {
            "health": result["agent_b_status"].health,
            "score": result["agent_b_status"].score
        },
        "result": result["messages"][-1].content
    }

# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("ADVERSARIAL AGENT SYSTEM DEMO")
    print("="*60)
    
    result = run_adversarial_game(
        game_type="attacker_defender",
        max_turns=10
    )
    
    if result["success"]:
        print(f"\nGame type: {result['game_type']}")
        print(f"Turns: {result['turns']}")
        print(f"Winner: {result['winner']}")
        print(f"\n{result['result']}")
```

---

## 🎲 Part 2: Game Theory in Multi-Agent Systems

### Theory: Game Theory Basics

#### What Is Game Theory?

**Game theory** is the mathematical study of strategic interaction between rational decision-makers. It provides frameworks for analyzing situations where:
- Multiple agents make decisions
- Each agent's outcome depends on others' choices
- Agents reason about what others will do

**Core Concepts:**

**1. Players:** The agents making decisions

**2. Strategies:** Available choices for each player

**3. Payoffs:** Outcomes (rewards/costs) for each combination of strategies

**4. Information:** What each player knows when deciding

#### Types of Games

**1. Cooperative vs Non-Cooperative:**
```
Cooperative: Players can form binding agreements
Non-Cooperative: Players act independently
```

**2. Zero-Sum vs Non-Zero-Sum:**
```
Zero-Sum: One's gain = others' loss (poker)
Non-Zero-Sum: All can gain or lose (trade)
```

**3. Simultaneous vs Sequential:**
```
Simultaneous: Players choose simultaneously (rock-paper-scissors)
Sequential: Players take turns (chess)
```

**4. Perfect vs Imperfect Information:**
```
Perfect: All players know all past moves (chess)
Imperfect: Some information hidden (poker)
```

#### Classic Game Theory Examples

**Example 1: Prisoner's Dilemma**

```
Two suspects caught, interrogated separately.

Payoffs:
                Suspect B Cooperates  |  Suspect B Defects
Suspect A 
Cooperates         (-1, -1)          |      (-3, 0)
Suspect A
Defects            (0, -3)           |      (-2, -2)

Numbers = years in prison
```

**Key Insight:** Rational self-interest leads both to defect, even though mutual cooperation is better.

**In multi-agent systems:**
- Competing services may overload shared infrastructure
- Agents may hoard resources "just in case"
- Lack of trust prevents cooperation

---

**Example 2: Coordination Game**

```
Two drivers approaching intersection.

Payoffs:
                Driver B Goes Left  |  Driver B Goes Right
Driver A
Goes Left           (1, 1)         |      (-10, -10)
Driver A  
Goes Right      (-10, -10)         |       (1, 1)

Numbers = utility (negative = crash)
```

**Key Insight:** Multiple good outcomes (Nash equilibria), but need coordination to reach them.

**In multi-agent systems:**
- Choosing communication protocols
- Standardizing interfaces
- Synchronizing actions

---

**Example 3: Hawk-Dove Game**

```
Two animals compete for resource.

Payoffs:
                Animal B Hawk    |  Animal B Dove
Animal A
Hawk               (-5, -5)      |     (10, 0)
Animal A
Dove                (0, 10)      |     (5, 5)

Hawk = aggressive, Dove = peaceful
```

**Key Insight:** Mixed strategies emerge - sometimes aggressive, sometimes peaceful.

**In multi-agent systems:**
- Bandwidth competition
- CPU resource allocation
- API rate limiting

---

#### Game Theory in Agent Design

**Principle 1: Model Other Agents**
```
Rational agents reason about opponents:
- What are their payoffs?
- What strategies are available to them?
- How might they respond to my actions?
```

**Principle 2: Best Response**
```
Choose strategy that's optimal given what others will do:
If opponent plays X, I should play Y
```

**Principle 3: Equilibrium Concepts**
```
Nash Equilibrium: No player can improve by unilaterally changing
Dominant Strategy: Best choice regardless of what others do
```

**Principle 4: Mixed Strategies**
```
Sometimes randomize between strategies:
- Makes you unpredictable
- Can achieve better expected payoff
```

#### Applying Game Theory to Multi-Agent Systems

**Application 1: Resource Allocation**
```
Multiple agents need shared CPU/memory
Game theory determines:
- Fair allocation
- Efficiency
- Preventing gaming the system
```

**Application 2: Auction Design**
```
Agents bid for resources
Auction mechanism design:
- Truth-telling incentives
- Efficient allocation
- Revenue maximization
```

**Application 3: Security**
```
Defender vs Attacker
Game theory analyzes:
- Optimal defense strategies
- Where attacker will strike
- Resource allocation for defense
```

**Application 4: Negotiation**
```
Agents negotiate terms
Game theory guides:
- Bargaining strategies
- Fair divisions
- Threat credibility
```

---

### Implementation: Game Theory System

```python
from typing import TypedDict, Annotated, Sequence, List, Dict, Tuple
from operator import add
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from dataclasses import dataclass
import random
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== GAME THEORY SYSTEM ====================

@dataclass
class GamePayoff:
    """Payoff matrix for a game"""
    name: str
    strategies_a: List[str]
    strategies_b: List[str]
    payoff_matrix: List[List[Tuple[int, int]]]  # (A's payoff, B's payoff)
    
    def get_payoff(self, strategy_a: str, strategy_b: str) -> Tuple[int, int]:
        """Get payoff for strategy combination"""
        idx_a = self.strategies_a.index(strategy_a)
        idx_b = self.strategies_b.index(strategy_b)
        return self.payoff_matrix[idx_a][idx_b]

# Predefined games
PRISONERS_DILEMMA = GamePayoff(
    name="Prisoner's Dilemma",
    strategies_a=["COOPERATE", "DEFECT"],
    strategies_b=["COOPERATE", "DEFECT"],
    payoff_matrix=[
        [(-1, -1), (-3, 0)],   # A cooperates
        [(0, -3), (-2, -2)]     # A defects
    ]
)

COORDINATION_GAME = GamePayoff(
    name="Coordination Game",
    strategies_a=["LEFT", "RIGHT"],
    strategies_b=["LEFT", "RIGHT"],
    payoff_matrix=[
        [(10, 10), (-5, -5)],   # A goes left
        [(-5, -5), (10, 10)]    # A goes right
    ]
)

HAWK_DOVE_GAME = GamePayoff(
    name="Hawk-Dove Game",
    strategies_a=["HAWK", "DOVE"],
    strategies_b=["HAWK", "DOVE"],
    payoff_matrix=[
        [(-5, -5), (10, 0)],    # A plays hawk
        [(0, 10), (5, 5)]       # A plays dove
    ]
)

# Game-playing agent state
class GameAgentState(TypedDict):
    """State for game-playing agent"""
    messages: Annotated[Sequence[BaseMessage], add]
    agent_name: str
    game: GamePayoff
    opponent_history: List[str]
    own_history: List[str]
    strategy: str
    reasoning: str

# Game state
class GameTheoryState(TypedDict):
    """State for game theory simulation"""
    messages: Annotated[Sequence[BaseMessage], add]
    game_name: str
    game: GamePayoff
    agent_a_history: Annotated[List[str], add]
    agent_b_history: Annotated[List[str], add]
    payoff_a_total: Annotated[int, add]
    payoff_b_total: Annotated[int, add]
    round_count: Annotated[int, add]
    max_rounds: int
    results: str

llm = ChatOllama(model="llama3.2", temperature=0.7)

# ==================== GAME-PLAYING AGENTS ====================

def create_game_agent(agent_name: str, agent_type: str = "rational"):
    """Create game-playing agent"""
    
    def agent_play(state: GameAgentState) -> dict:
        """Agent chooses strategy"""
        logger.info(f"🎮 {agent_name}: Choosing strategy")
        
        game = state["game"]
        opponent_history = state.get("opponent_history", [])
        own_history = state.get("own_history", [])
        
        # Get available strategies
        if agent_name == "AGENT_A":
            strategies = game.strategies_a
        else:
            strategies = game.strategies_b
        
        # Build history context
        if opponent_history:
            history_text = "Previous rounds:\n" + "\n".join([
                f"Round {i+1}: You played {own_history[i]}, Opponent played {opponent_history[i]}"
                for i in range(len(opponent_history))
            ])
        else:
            history_text = "This is the first round."
        
        if agent_type == "rational":
            system_prompt = f"""You are {agent_name}, a rational game-playing agent.

Game: {game.name}

Payoff Matrix:
{format_payoff_matrix(game)}

Analyze the game strategically:
- Consider opponent's likely moves
- Calculate expected payoffs
- Look for Nash equilibrium
- Consider opponent's history

Be strategic and rational."""

        elif agent_type == "cooperative":
            system_prompt = f"""You are {agent_name}, a cooperative agent.

You prefer mutual benefit over individual gain.
You trust your opponent unless they betray you.
You value long-term relationships.

Game: {game.name}

Payoff Matrix:
{format_payoff_matrix(game)}"""

        else:  # competitive
            system_prompt = f"""You are {agent_name}, a competitive agent.

You maximize your own payoff above all else.
You don't trust your opponent.
You exploit any weakness.

Game: {game.name}

Payoff Matrix:
{format_payoff_matrix(game)}"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", """{history}

Available strategies: {strategies}

Choose your strategy. Format your response as:
STRATEGY: [your choice]
REASONING: [brief explanation]

Your decision:""")
        ])
        
        response = llm.invoke(
            prompt.format_messages(
                history=history_text,
                strategies=", ".join(strategies)
            )
        )
        
        # Parse strategy
        content = response.content
        strategy = strategies[0]  # default
        
        if "STRATEGY:" in content:
            strategy_part = content.split("STRATEGY:")[1].split("REASONING:")[0].strip()
            # Find which strategy matches
            for s in strategies:
                if s in strategy_part.upper():
                    strategy = s
                    break
        
        logger.info(f"🎮 {agent_name} chose: {strategy}")
        
        return {
            "strategy": strategy,
            "reasoning": content
        }
    
    # Build workflow
    agent_workflow = StateGraph(GameAgentState)
    agent_workflow.add_node("play", agent_play)
    agent_workflow.set_entry_point("play")
    agent_workflow.add_edge("play", END)
    
    return agent_workflow.compile()

def format_payoff_matrix(game: GamePayoff) -> str:
    """Format payoff matrix for display"""
    lines = []
    
    # Header
    header = "              " + "  |  ".join(game.strategies_b)
    lines.append(header)
    lines.append("-" * len(header))
    
    # Rows
    for i, strat_a in enumerate(game.strategies_a):
        row_payoffs = []
        for j in range(len(game.strategies_b)):
            payoff = game.payoff_matrix[i][j]
            row_payoffs.append(f"({payoff[0]}, {payoff[1]})")
        
        row = f"{strat_a:12s}  " + "  |  ".join(row_payoffs)
        lines.append(row)
    
    return "\n".join(lines)

# Create agents
rational_agent_a = create_game_agent("AGENT_A", "rational")
rational_agent_b = create_game_agent("AGENT_B", "rational")

cooperative_agent_a = create_game_agent("AGENT_A", "cooperative")
cooperative_agent_b = create_game_agent("AGENT_B", "cooperative")

competitive_agent_a = create_game_agent("AGENT_A", "competitive")
competitive_agent_b = create_game_agent("AGENT_B", "competitive")

# ==================== GAME ORCHESTRATOR ====================

def play_round(state: GameTheoryState, agent_a, agent_b) -> dict:
    """Play one round of the game"""
    round_num = state["round_count"] + 1
    logger.info(f"🎲 Round {round_num}")
    
    game = state["game"]
    
    # Agent A plays
    result_a = agent_a.invoke({
        "messages": [],
        "agent_name": "AGENT_A",
        "game": game,
        "opponent_history": state["agent_b_history"],
        "own_history": state["agent_a_history"],
        "strategy": "",
        "reasoning": ""
    })
    
    strategy_a = result_a["strategy"]
    
    # Agent B plays
    result_b = agent_b.invoke({
        "messages": [],
        "agent_name": "AGENT_B",
        "game": game,
        "opponent_history": state["agent_a_history"],
        "own_history": state["agent_b_history"],
        "strategy": "",
        "reasoning": ""
    })
    
    strategy_b = result_b["strategy"]
    
    # Get payoffs
    payoff_a, payoff_b = game.get_payoff(strategy_a, strategy_b)
    
    logger.info(f"Round {round_num}: A={strategy_a}, B={strategy_b} → Payoffs: A={payoff_a}, B={payoff_b}")
    
    return {
        "agent_a_history": [strategy_a],
        "agent_b_history": [strategy_b],
        "payoff_a_total": payoff_a,
        "payoff_b_total": payoff_b,
        "round_count": 1
    }

def simulate_game(
    game: GamePayoff,
    agent_a,
    agent_b,
    rounds: int
) -> dict:
    """Simulate game for multiple rounds"""
    
    # Build workflow dynamically for this game
    def execute_round(state: GameTheoryState) -> dict:
        return play_round(state, agent_a, agent_b)
    
    def check_completion(state: GameTheoryState) -> dict:
        """Check if game is complete"""
        return {}
    
    def format_results(state: GameTheoryState) -> dict:
        """Format game results"""
        
        # Build round-by-round summary
        round_summaries = []
        for i in range(state["round_count"]):
            round_summaries.append(
                f"Round {i+1}: A={state['agent_a_history'][i]}, "
                f"B={state['agent_b_history'][i]}"
            )
        
        results = f"""GAME THEORY SIMULATION RESULTS

Game: {state['game_name']}
Rounds: {state['round_count']}

Final Scores:
AGENT_A: {state['payoff_a_total']}
AGENT_B: {state['payoff_b_total']}

Round History:
{chr(10).join(round_summaries)}

Analysis:
Average payoff per round:
- AGENT_A: {state['payoff_a_total'] / state['round_count']:.2f}
- AGENT_B: {state['payoff_b_total'] / state['round_count']:.2f}"""
        
        return {
            "results": results,
            "messages": [AIMessage(content=results)]
        }
    
    def should_continue(state: GameTheoryState) -> str:
        if state["round_count"] >= state["max_rounds"]:
            return "format"
        return "execute"
    
    # Build workflow
    game_workflow = StateGraph(GameTheoryState)
    game_workflow.add_node("execute", execute_round)
    game_workflow.add_node("check", check_completion)
    game_workflow.add_node("format", format_results)
    
    game_workflow.set_entry_point("execute")
    game_workflow.add_edge("execute", "check")
    
    game_workflow.add_conditional_edges(
        "check",
        should_continue,
        {
            "execute": "execute",
            "format": "format"
        }
    )
    
    game_workflow.add_edge("format", END)
    
    game_system = game_workflow.compile()
    
    # Run simulation
    result = game_system.invoke({
        "messages": [HumanMessage(content=f"Playing {game.name}")],
        "game_name": game.name,
        "game": game,
        "agent_a_history": [],
        "agent_b_history": [],
        "payoff_a_total": 0,
        "payoff_b_total": 0,
        "round_count": 0,
        "max_rounds": rounds,
        "results": ""
    })
    
    return {
        "success": True,
        "game": game.name,
        "rounds": result["round_count"],
        "agent_a_total": result["payoff_a_total"],
        "agent_b_total": result["payoff_b_total"],
        "result": result["messages"][-1].content
    }

# ==================== API ====================

def run_game_theory_simulation(
    game_name: str = "prisoners_dilemma",
    agent_a_type: str = "rational",
    agent_b_type: str = "rational",
    rounds: int = 5
) -> dict:
    """
    Run game theory simulation.
    
    Args:
        game_name: "prisoners_dilemma", "coordination", or "hawk_dove"
        agent_a_type: "rational", "cooperative", or "competitive"
        agent_b_type: "rational", "cooperative", or "competitive"
        rounds: Number of rounds to play
    """
    
    # Select game
    games = {
        "prisoners_dilemma": PRISONERS_DILEMMA,
        "coordination": COORDINATION_GAME,
        "hawk_dove": HAWK_DOVE_GAME
    }
    
    game = games.get(game_name, PRISONERS_DILEMMA)
    
    # Select agents
    agent_types = {
        "rational": (rational_agent_a, rational_agent_b),
        "cooperative": (cooperative_agent_a, cooperative_agent_b),
        "competitive": (competitive_agent_a, competitive_agent_b)
    }
    
    agents_a = {
        "rational": rational_agent_a,
        "cooperative": cooperative_agent_a,
        "competitive": competitive_agent_a
    }
    
    agents_b = {
        "rational": rational_agent_b,
        "cooperative": cooperative_agent_b,
        "competitive": competitive_agent_b
    }
    
    agent_a = agents_a.get(agent_a_type, rational_agent_a)
    agent_b = agents_b.get(agent_b_type, rational_agent_b)
    
    return simulate_game(game, agent_a, agent_b, rounds)

# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("GAME THEORY SIMULATION DEMO")
    print("="*60)
    
    # Test Prisoner's Dilemma with different agent types
    scenarios = [
        ("prisoners_dilemma", "rational", "rational"),
        ("prisoners_dilemma", "cooperative", "cooperative"),
        ("prisoners_dilemma", "cooperative", "competitive"),
    ]
    
    for game, type_a, type_b in scenarios:
        print(f"\n{'='*60}")
        print(f"SCENARIO: {game.upper()} - {type_a} vs {type_b}")
        print(f"{'='*60}")
        
        result = run_game_theory_simulation(
            game_name=game,
            agent_a_type=type_a,
            agent_b_type=type_b,
            rounds=5
        )
        
        if result["success"]:
            print(f"\n{result['result']}")
```

---

## 💰 Part 3: Competitive Auction/Bidding Systems

### Theory: Auction Mechanisms

#### What Are Auction Systems?

**Auctions** are mechanisms for allocating resources among competing agents through bidding. Key elements:
- **Seller**: Has resource to allocate
- **Bidders**: Agents competing for resource
- **Mechanism**: Rules determining winner and price
- **Valuations**: How much each bidder values the resource

#### Types of Auctions

**1. English Auction (Ascending Bid)**
```
Start: Low price
Process: Bidders raise bids publicly
Winner: Highest bidder
Price: Final bid amount

Example: Art auctions, eBay
```

**2. Dutch Auction (Descending Bid)**
```
Start: High price
Process: Price drops until someone accepts
Winner: First to accept
Price: Current price when accepted

Example: Flower markets, some IPOs
```

**3. First-Price Sealed-Bid**
```
Process: All bidders submit sealed bids simultaneously
Winner: Highest bidder
Price: Their own bid

Strategy: Bid below true value (shade bid)
```

**4. Second-Price Sealed-Bid (Vickrey Auction)**
```
Process: All bidders submit sealed bids simultaneously
Winner: Highest bidder
Price: Second-highest bid

Strategy: Bid true value (truthful bidding)
```

**5. All-Pay Auction**
```
Process: All bidders pay their bid
Winner: Highest bidder gets resource
Price: Everyone pays their bid

Example: Political lobbying, R&D races
```

#### Auction Design Goals

**Goal 1: Efficiency**
```
Allocate resource to bidder who values it most
Maximizes social welfare
```

**Goal 2: Revenue Maximization**
```
Maximize seller's revenue
Important for seller's objectives
```

**Goal 3: Truth-Telling (Incentive Compatibility)**
```
Bidders should bid their true valuation
Prevents strategic manipulation
```

**Goal 4: Fairness**
```
Equal opportunity for all bidders
No bias toward certain participants
```

#### Why Second-Price Auctions Are Special

**Vickrey Auction Property:**
```
Dominant strategy: Bid your true value

Why?
- If you bid more than value: Risk paying more than it's worth
- If you bid less than value: Miss opportunities to win profitably
- Bidding true value: Always optimal regardless of others
```

**Example:**
```
Your true value: $100
Second-highest bid: $80

If you bid $100: Win, pay $80, profit $20 ✓
If you bid $90: Win, pay $80, profit $20 ✓ (same outcome)
If you bid $70: Lose, profit $0 ✗ (missed opportunity)
If you bid $110: Win, pay $80, profit $20 ✓ (same outcome)

Bidding true value is safe and optimal
```

#### Bidding Strategies

**Strategy 1: Truthful Bidding**
```
Bid your actual valuation
Works in: Vickrey auctions
Advantage: Simple, no need to predict others
```

**Strategy 2: Bid Shading**
```
Bid less than true value
Works in: First-price auctions
Goal: Win while paying less
Risk: Might lose auction
```

**Strategy 3: Incremental Bidding**
```
Raise bid gradually in English auction
Goal: Discover minimum winning price
Risk: Others might outbid at last second
```

**Strategy 4: Sniping**
```
Wait until last moment to bid
Works in: Online timed auctions
Goal: Prevent bidding wars
Risk: Technical failures
```

**Strategy 5: Budget Constraints**
```
Set maximum bid (reservation price)
Don't exceed budget regardless of competition
Prevents overpaying in competitive bidding
```

#### Multi-Unit Auctions

**Scenario:** Multiple identical items to allocate

**Uniform Price Auction:**
```
All winners pay same price (k+1-th highest bid)
Example: Treasury bonds, Google AdWords
```

**Discriminatory Price Auction:**
```
Each winner pays their own bid
Higher revenue but less truth-telling
```

**Combinatorial Auctions:**
```
Bidders can bid on bundles of items
Complex but allows expressing preferences
Example: Spectrum auctions
```

#### Auction Applications in Multi-Agent Systems

**1. Cloud Resource Allocation:**
```
VMs/containers allocated via auction
Bidders: Different applications
Price: Based on demand/supply
```

**2. Ad Placement:**
```
Ad slots allocated to highest bidders
Google AdWords, Facebook Ads
Second-price auction for truth-telling
```

**3. Task Assignment:**
```
Tasks assigned to agents via bidding
Agents bid based on capability/cost
Efficient task distribution
```

**4. Bandwidth Allocation:**
```
Network bandwidth auctioned
Agents bid for priority
Dynamic pricing based on congestion
```

---

### Implementation: Auction System

```python
from typing import TypedDict, Annotated, Sequence, List, Dict, Optional
from operator import add
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from dataclasses import dataclass
import random
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== AUCTION SYSTEM ====================

@dataclass
class Bidder:
    """Represents a bidding agent"""
    name: str
    budget: int
    valuation: int  # True value for the item
    strategy: str  # "truthful", "shade", "aggressive"
    
@dataclass
class Bid:
    """Represents a bid"""
    bidder_name: str
    amount: int
    timestamp: int

# Bidder agent state
class BidderState(TypedDict):
    """State for bidding agent"""
    messages: Annotated[Sequence[BaseMessage], add]
    bidder: Bidder
    auction_type: str
    item_description: str
    current_highest_bid: Optional[int]
    bid_history: List[Bid]
    bid_amount: int
    reasoning: str

# Auction state
class AuctionState(TypedDict):
    """State for auction orchestration"""
    messages: Annotated[Sequence[BaseMessage], add]
    auction_type: str
    item_description: str
    reserve_price: int
    bidders: List[Bidder]
    bids: Annotated[List[Bid], add]
    current_highest_bid: int
    round_count: Annotated[int, add]
    max_rounds: int
    winner: Optional[str]
    winning_bid: int
    price_paid: int
    auction_result: str

llm = ChatOllama(model="llama3.2", temperature=0.7)

# ==================== BIDDING AGENTS ====================

def create_bidding_agent(bidder: Bidder):
    """Create bidding agent with specific strategy"""
    
    def agent_bid(state: BidderState) -> dict:
        """Agent decides bid amount"""
        logger.info(f"💰 {bidder.name}: Deciding bid")
        
        auction_type = state["auction_type"]
        item = state["item_description"]
        current_high = state.get("current_highest_bid", 0)
        
        # Strategy-based bidding
        if bidder.strategy == "truthful":
            # Vickrey-style: bid true value
            bid_amount = bidder.valuation
            reasoning = f"Bidding true valuation (${bid_amount})"
            
        elif bidder.strategy == "shade":
            # First-price: shade bid below true value
            shade_factor = random.uniform(0.7, 0.9)
            bid_amount = int(bidder.valuation * shade_factor)
            reasoning = f"Shading bid to ${bid_amount} (true value: ${bidder.valuation})"
            
        elif bidder.strategy == "aggressive":
            # Bid above current high by margin
            if current_high > 0:
                increment = random.randint(10, 30)
                bid_amount = min(current_high + increment, bidder.budget)
            else:
                bid_amount = min(bidder.valuation, bidder.budget)
            reasoning = f"Aggressive bid: ${bid_amount} (outbidding current ${current_high})"
            
        else:  # LLM-based reasoning
            prompt = ChatPromptTemplate.from_messages([
                ("system", f"""You are bidder {bidder.name} in an auction.

Your information:
- Budget: ${bidder.budget}
- True valuation: ${bidder.valuation}
- Strategy preference: {bidder.strategy}

Auction type: {auction_type}

In second-price auctions, bid your true value.
In first-price auctions, consider bidding below true value."""),
                ("human", """Item: {item}
Current highest bid: ${current_high}

What amount will you bid? Consider:
- Your valuation and budget
- Auction type
- Current competition

Respond with: BID: [amount]

Your bid:""")
            ])
            
            response = llm.invoke(
                prompt.format_messages(
                    item=item,
                    current_high=current_high if current_high else 0
                )
            )
            
            # Parse bid
            content = response.content
            bid_amount = bidder.valuation  # default
            
            if "BID:" in content:
                try:
                    bid_str = content.split("BID:")[1].strip().split()[0]
                    bid_str = bid_str.replace("$", "").replace(",", "")
                    bid_amount = int(float(bid_str))
                except:
                    pass
            
            reasoning = content
        
        # Enforce budget constraint
        bid_amount = min(bid_amount, bidder.budget)
        
        logger.info(f"💰 {bidder.name} bids: ${bid_amount}")
        
        return {
            "bid_amount": bid_amount,
            "reasoning": reasoning
        }
    
    # Build workflow
    bidder_workflow = StateGraph(BidderState)
    bidder_workflow.add_node("bid", agent_bid)
    bidder_workflow.set_entry_point("bid")
    bidder_workflow.add_edge("bid", END)
    
    return bidder_workflow.compile()

# ==================== AUCTION ORCHESTRATORS ====================

def run_sealed_bid_auction(
    auction_state: AuctionState,
    bidding_agents: Dict[str, any]
) -> dict:
    """Run sealed-bid auction (first or second price)"""
    
    logger.info(f"🔨 Running {auction_state['auction_type']} auction")
    
    auction_type = auction_state["auction_type"]
    item = auction_state["item_description"]
    bidders = auction_state["bidders"]
    
    # Collect sealed bids
    bids = []
    
    for bidder in bidders:
        agent = bidding_agents[bidder.name]
        
        result = agent.invoke({
            "messages": [],
            "bidder": bidder,
            "auction_type": auction_type,
            "item_description": item,
            "current_highest_bid": None,  # Sealed bid - no info
            "bid_history": [],
            "bid_amount": 0,
            "reasoning": ""
        })
        
        bid = Bid(
            bidder_name=bidder.name,
            amount=result["bid_amount"],
            timestamp=0
        )
        
        bids.append(bid)
        logger.info(f"  {bidder.name} bid: ${bid.amount}")
    
    # Sort bids descending
    sorted_bids = sorted(bids, key=lambda b: b.amount, reverse=True)
    
    if not sorted_bids or sorted_bids[0].amount < auction_state["reserve_price"]:
        logger.info("No bids meet reserve price")
        return {
            "bids": bids,
            "winner": None,
            "winning_bid": 0,
            "price_paid": 0,
            "auction_result": "No winner - reserve price not met"
        }
    
    winner_bid = sorted_bids[0]
    
    # Determine price based on auction type
    if auction_type == "second_price":
        # Winner pays second-highest bid
        price_paid = sorted_bids[1].amount if len(sorted_bids) > 1 else auction_state["reserve_price"]
    else:  # first_price
        # Winner pays their own bid
        price_paid = winner_bid.amount
    
    logger.info(f"🏆 Winner: {winner_bid.bidder_name}, Price: ${price_paid}")
    
    return {
        "bids": bids,
        "winner": winner_bid.bidder_name,
        "winning_bid": winner_bid.amount,
        "price_paid": price_paid,
        "round_count": 1
    }

def run_english_auction(
    auction_state: AuctionState,
    bidding_agents: Dict[str, any]
) -> dict:
    """Run English (ascending) auction"""
    
    logger.info("🔨 Running English auction")
    
    item = auction_state["item_description"]
    bidders = auction_state["bidders"]
    current_high = auction_state["reserve_price"]
    
    all_bids = []
    active_bidders = list(bidders)
    round_num = 0
    
    while len(active_bidders) > 1 and round_num < auction_state["max_rounds"]:
        round_num += 1
        logger.info(f"Round {round_num}: Current high bid ${current_high}")
        
        round_bids = []
        
        for bidder in active_bidders:
            agent = bidding_agents[bidder.name]
            
            result = agent.invoke({
                "messages": [],
                "bidder": bidder,
                "auction_type": "english",
                "item_description": item,
                "current_highest_bid": current_high,
                "bid_history": all_bids,
                "bid_amount": 0,
                "reasoning": ""
            })
            
            bid_amount = result["bid_amount"]
            
            # Only bids above current high are valid
            if bid_amount > current_high:
                bid = Bid(
                    bidder_name=bidder.name,
                    amount=bid_amount,
                    timestamp=round_num
                )
                round_bids.append(bid)
                logger.info(f"  {bidder.name} bid: ${bid_amount}")
        
        if not round_bids:
            # No new bids - auction ends
            logger.info("No new bids - auction ending")
            break
        
        # Update current high
        highest_this_round = max(round_bids, key=lambda b: b.amount)
        current_high = highest_this_round.amount
        all_bids.extend(round_bids)
        
        # Remove bidders who didn't bid this round (dropped out)
        bidders_who_bid = {b.bidder_name for b in round_bids}
        active_bidders = [b for b in active_bidders if b.name in bidders_who_bid]
    
    if all_bids:
        winner_bid = max(all_bids, key=lambda b: b.amount)
        
        return {
            "bids": all_bids,
            "winner": winner_bid.bidder_name,
            "winning_bid": winner_bid.amount,
            "price_paid": winner_bid.amount,
            "round_count": round_num
        }
    else:
        return {
            "bids": [],
            "winner": None,
            "winning_bid": 0,
            "price_paid": 0,
            "round_count": round_num,
            "auction_result": "No winner - no bids above reserve"
        }

def format_auction_results(state: AuctionState) -> dict:
    """Format auction results"""
    
    # Build bid summary
    bid_summary = []
    for bid in state["bids"]:
        bid_summary.append(f"  {bid.bidder_name}: ${bid.amount}")
    
    # Build bidder info
    bidder_info = []
    for bidder in state["bidders"]:
        is_winner = bidder.name == state["winner"]
        profit = 0
        if is_winner:
            profit = bidder.valuation - state["price_paid"]
        
        bidder_info.append(
            f"  {bidder.name}: "
            f"Valuation=${bidder.valuation}, "
            f"Strategy={bidder.strategy}"
            f"{' (WINNER - Profit: $' + str(profit) + ')' if is_winner else ''}"
        )
    
    result = f"""AUCTION RESULTS

Type: {state['auction_type'].upper().replace('_', ' ')}
Item: {state['item_description']}
Reserve Price: ${state['reserve_price']}
Rounds: {state['round_count']}

Bidders:
{chr(10).join(bidder_info)}

Bids:
{chr(10).join(bid_summary) if bid_summary else '  No bids'}

Outcome:
Winner: {state['winner'] if state['winner'] else 'None'}
Winning Bid: ${state['winning_bid']}
Price Paid: ${state['price_paid']}

Analysis:
- Revenue to Seller: ${state['price_paid']}
- Winner's Surplus: ${state['bidders'][0].valuation - state['price_paid'] if state['winner'] else 0}
- Efficiency: {'Item allocated to highest valuer' if state['winner'] else 'No allocation'}"""
    
    return {
        "auction_result": result,
        "messages": [AIMessage(content=result)]
    }

# Build auction workflow
def create_auction_workflow(auction_type: str):
    """Create auction workflow for specific type"""
    
    def execute_auction(state: AuctionState) -> dict:
        """Execute the auction"""
        
        # Create bidding agents
        bidding_agents = {
            bidder.name: create_bidding_agent(bidder)
            for bidder in state["bidders"]
        }
        
        # Run appropriate auction type
        if auction_type in ["first_price", "second_price"]:
            result = run_sealed_bid_auction(state, bidding_agents)
        elif auction_type == "english":
            result = run_english_auction(state, bidding_agents)
        else:
            result = run_sealed_bid_auction(state, bidding_agents)
        
        return result
    
    auction_workflow = StateGraph(AuctionState)
    auction_workflow.add_node("execute", execute_auction)
    auction_workflow.add_node("format", format_auction_results)
    
    auction_workflow.set_entry_point("execute")
    auction_workflow.add_edge("execute", "format")
    auction_workflow.add_edge("format", END)
    
    return auction_workflow.compile()

# ==================== API ====================

def run_auction(
    auction_type: str = "second_price",
    item_description: str = "Vintage collectible item",
    reserve_price: int = 50,
    bidders: List[Dict] = None
) -> dict:
    """
    Run auction simulation.
    
    Args:
        auction_type: "first_price", "second_price", or "english"
        item_description: Description of item being auctioned
        reserve_price: Minimum acceptable price
        bidders: List of bidder configs with {name, budget, valuation, strategy}
    """
    
    if bidders is None:
        # Default bidders
        bidders = [
            {"name": "Bidder_A", "budget": 200, "valuation": 150, "strategy": "truthful"},
            {"name": "Bidder_B", "budget": 180, "valuation": 130, "strategy": "shade"},
            {"name": "Bidder_C", "budget": 220, "valuation": 170, "strategy": "aggressive"},
        ]
    
    # Create Bidder objects
    bidder_objects = [
        Bidder(
            name=b["name"],
            budget=b["budget"],
            valuation=b["valuation"],
            strategy=b["strategy"]
        )
        for b in bidders
    ]
    
    # Create and run auction
    auction_system = create_auction_workflow(auction_type)
    
    result = auction_system.invoke({
        "messages": [HumanMessage(content=f"Running {auction_type} auction")],
        "auction_type": auction_type,
        "item_description": item_description,
        "reserve_price": reserve_price,
        "bidders": bidder_objects,
        "bids": [],
        "current_highest_bid": reserve_price,
        "round_count": 0,
        "max_rounds": 5,
        "winner": None,
        "winning_bid": 0,
        "price_paid": 0,
        "auction_result": ""
    })
    
    return {
        "success": True,
        "auction_type": auction_type,
        "winner": result["winner"],
        "price_paid": result["price_paid"],
        "result": result["messages"][-1].content
    }

# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("AUCTION SYSTEM DEMO")
    print("="*60)
    
    # Test different auction types
    for auction_type in ["second_price", "first_price", "english"]:
        print(f"\n{'='*60}")
        print(f"AUCTION TYPE: {auction_type.upper().replace('_', ' ')}")
        print(f"{'='*60}")
        
        result = run_auction(
            auction_type=auction_type,
            item_description="Premium cloud compute instance (1 hour)",
            reserve_price=50
        )
        
        if result["success"]:
            print(f"\n{result['result']}")
```

---

## 🛡️ Part 4: Red Team / Blue Team Patterns

### Theory: Red Team vs Blue Team

#### What Are Red Team / Blue Team Exercises?

**Red Team / Blue Team** is a cybersecurity practice where:
- **Red Team**: Simulates attackers/adversaries
- **Blue Team**: Defends against attacks
- **Goal**: Test and improve security through realistic simulation

**Origins:** Military training exercises where opposing forces test each other.

#### Why Red Team / Blue Team?

**1. Proactive Security Testing:**
```
Don't wait for real attackers
Find vulnerabilities first
Test defenses realistically
```

**2. Continuous Improvement:**
```
Red team finds weaknesses → Blue team fixes → Red team tries again
Iterative improvement cycle
```

**3. Training:**
```
Blue team practices response
Red team practices attack techniques
Both improve skills
```

**4. Validation:**
```
Verify security controls work
Test incident response procedures
Identify gaps in coverage
```

#### Red Team Characteristics

**Objectives:**
- Achieve specific goals (e.g., access data, disrupt service)
- Use any means necessary within rules
- Think creatively and unconventionally
- Exploit any weakness

**Tactics:**
- Reconnaissance (gather information)
- Initial compromise (gain foothold)
- Privilege escalation (gain more access)
- Lateral movement (spread through system)
- Persistence (maintain access)
- Exfiltration (steal data) or Impact (cause damage)

**Mindset:**
- Adversarial thinking
- Question assumptions
- Look for shortcuts
- Exploit human factors

#### Blue Team Characteristics

**Objectives:**
- Prevent attacks
- Detect intrusions quickly
- Respond effectively
- Maintain operations

**Tactics:**
- Defense in depth (multiple layers)
- Monitoring and detection
- Incident response
- Patch management
- Access control

**Mindset:**
- Defensive thinking
- Assume breach
- Continuous vigilance
- Systematic approach

#### Red Team / Blue Team Cycle

**Phase 1: Planning**
```
Define scope and rules
Set objectives for red team
Brief both teams
```

**Phase 2: Reconnaissance**
```
Red team gathers intelligence
Blue team monitors for suspicious activity
```

**Phase 3: Attack**
```
Red team attempts to achieve objectives
Blue team defends and responds
```

**Phase 4: Debrief**
```
Both teams share what happened
Analyze successes and failures
Document lessons learned
```

**Phase 5: Improvement**
```
Blue team implements fixes
Update defenses and procedures
Prepare for next exercise
```

#### Purple Team

**Purple Team** = Collaboration between red and blue

Instead of pure adversarial:
- Share techniques in real-time
- Explain how attacks work
- Jointly improve defenses
- Faster learning cycle

**When to use:**
- Training new defenders
- Testing specific controls
- Limited time/resources
- Building capabilities

#### Applications Beyond Security

**1. Business Strategy:**
```
Red team: Simulates competitors
Blue team: Defends market position
```

**2. Product Design:**
```
Red team: Finds usability issues
Blue team: Improves design
```

**3. System Testing:**
```
Red team: Stress tests system
Blue team: Ensures reliability
```

**4. Decision Making:**
```
Red team: Challenges plan
Blue team: Defends/improves plan
```

---

### Implementation: Red Team / Blue Team System

```python
from typing import TypedDict, Annotated, Sequence, List, Dict, Literal
from operator import add
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from dataclasses import dataclass
import random
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== RED TEAM / BLUE TEAM SYSTEM ====================

@dataclass
class SystemState:
    """State of the system being tested"""
    security_level: int  # 0-100
    vulnerabilities: List[str]
    patches_applied: List[str]
    alerts_triggered: int

# Team agent state
class TeamAgentState(TypedDict):
    """State for red/blue team agent"""
    messages: Annotated[Sequence[BaseMessage], add]
    team_name: str
    system_state: SystemState
    opponent_actions: List[str]
    action: str
    reasoning: str

# Exercise state
class RedBlueExerciseState(TypedDict):
    """State for red/blue team exercise"""
    messages: Annotated[Sequence[BaseMessage], add]
    exercise_scenario: str
    system_state: SystemState
    red_team_objective: str
    red_team_actions: Annotated[List[str], add]
    blue_team_actions: Annotated[List[str], add]
    round_count: Annotated[int, add]
    max_rounds: int
    objective_achieved: bool
    exercise_log: Annotated[List[str], add]
    debrief: str

llm = ChatOllama(model="llama3.2", temperature=0.8)

# ==================== RED TEAM AGENT ====================

RED_TEAM_SYSTEM = """You are a RED TEAM agent in a cybersecurity exercise.

Your role: Simulate an attacker trying to compromise the system.

Current objective: {objective}

System status:
- Security level: {security_level}/100
- Known vulnerabilities: {vulnerabilities}
- Patches applied: {patches}

Your capabilities:
1. SCAN: Probe for vulnerabilities
2. EXPLOIT: Attempt to exploit known vulnerability
3. ESCALATE: Try to gain higher privileges
4. PERSIST: Establish persistent access
5. EXFILTRATE: Attempt to steal data

Be creative, strategic, and persistent. Your goal is to succeed before blue team stops you."""

def red_team_act(state: TeamAgentState) -> dict:
    """Red team takes action"""
    logger.info("🔴 RED TEAM: Planning attack")
    
    system_state = state["system_state"]
    blue_actions = state.get("opponent_actions", [])
    
    blue_summary = "\n".join(blue_actions[-3:]) if blue_actions else "No blue team actions yet"
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", RED_TEAM_SYSTEM.format(
            objective=state.get("objective", "Compromise system"),
            security_level=system_state.security_level,
            vulnerabilities=", ".join(system_state.vulnerabilities) if system_state.vulnerabilities else "None discovered",
            patches=", ".join(system_state.patches_applied) if system_state.patches_applied else "None"
        )),
        ("human", """Recent blue team actions:
{blue_actions}

Choose your action: SCAN, EXPLOIT, ESCALATE, PERSIST, or EXFILTRATE

Respond with:
ACTION: [your choice]
TARGET: [what you're targeting]
REASONING: [your strategy]

Your action:""")
    ])
    
    response = llm.invoke(
        prompt.format_messages(blue_actions=blue_summary)
    )
    
    # Parse action
    content = response.content
    action = "SCAN"  # default
    
    if "ACTION:" in content:
        action_part = content.split("ACTION:")[1].split("TARGET:")[0].strip()
        action = action_part.split()[0].upper()
    
    logger.info(f"🔴 RED TEAM action: {action}")
    
    return {
        "action": action,
        "reasoning": content,
        "messages": [AIMessage(content=f"[RED TEAM] {action}")]
    }

red_team_workflow = StateGraph(TeamAgentState)
red_team_workflow.add_node("act", red_team_act)
red_team_workflow.set_entry_point("act")
red_team_workflow.add_edge("act", END)
red_team_agent = red_team_workflow.compile()

# ==================== BLUE TEAM AGENT ====================

BLUE_TEAM_SYSTEM = """You are a BLUE TEAM agent in a cybersecurity exercise.

Your role: Defend the system against red team attacks.

System status:
- Security level: {security_level}/100
- Known vulnerabilities: {vulnerabilities}
- Patches applied: {patches}
- Alerts triggered: {alerts}

Your capabilities:
1. MONITOR: Watch for suspicious activity
2. PATCH: Fix known vulnerabilities
3. BLOCK: Block detected attack
4. INVESTIGATE: Analyze suspicious behavior
5. HARDEN: Strengthen security controls

Be vigilant, systematic, and proactive. Protect the system."""

def blue_team_act(state: TeamAgentState) -> dict:
    """Blue team takes action"""
    logger.info("🔵 BLUE TEAM: Planning defense")
    
    system_state = state["system_state"]
    red_actions = state.get("opponent_actions", [])
    
    red_summary = "\n".join(red_actions[-3:]) if red_actions else "No red team actions yet"
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", BLUE_TEAM_SYSTEM.format(
            security_level=system_state.security_level,
            vulnerabilities=", ".join(system_state.vulnerabilities) if system_state.vulnerabilities else "None known",
            patches=", ".join(system_state.patches_applied) if system_state.patches_applied else "None",
            alerts=system_state.alerts_triggered
        )),
        ("human", """Recent red team actions (detected):
{red_actions}

Choose your action: MONITOR, PATCH, BLOCK, INVESTIGATE, or HARDEN

Respond with:
ACTION: [your choice]
TARGET: [what you're securing]
REASONING: [your strategy]

Your action:""")
    ])
    
    response = llm.invoke(
        prompt.format_messages(red_actions=red_summary)
    )
    
    # Parse action
    content = response.content
    action = "MONITOR"  # default
    
    if "ACTION:" in content:
        action_part = content.split("ACTION:")[1].split("TARGET:")[0].strip()
        action = action_part.split()[0].upper()
    
    logger.info(f"🔵 BLUE TEAM action: {action}")
    
    return {
        "action": action,
        "reasoning": content,
        "messages": [AIMessage(content=f"[BLUE TEAM] {action}")]
    }

blue_team_workflow = StateGraph(TeamAgentState)
blue_team_workflow.add_node("act", blue_team_act)
blue_team_workflow.set_entry_point("act")
blue_team_workflow.add_edge("act", END)
blue_team_agent = blue_team_workflow.compile()

# ==================== EXERCISE ORCHESTRATOR ====================

def execute_round(state: RedBlueExerciseState) -> dict:
    """Execute one round of red/blue exercise"""
    round_num = state["round_count"] + 1
    logger.info(f"⚔️ Round {round_num}")
    
    system_state = state["system_state"]
    
    # Red team acts
    red_result = red_team_agent.invoke({
        "messages": [],
        "team_name": "RED",
        "system_state": system_state,
        "opponent_actions": state["blue_team_actions"],
        "objective": state["red_team_objective"],
        "action": "",
        "reasoning": ""
    })
    
    red_action = red_result["action"]
    
    # Process red team action
    round_log = [f"Round {round_num}:"]
    
    if red_action == "SCAN":
        # Discover vulnerability
        if random.random() > 0.5:
            vuln = f"vuln_{round_num}"
            system_state.vulnerabilities.append(vuln)
            round_log.append(f"  🔴 RED discovered vulnerability: {vuln}")
        else:
            round_log.append(f"  🔴 RED scan found nothing")
    
    elif red_action == "EXPLOIT":
        if system_state.vulnerabilities:
            # Attempt exploit
            if random.random() > 0.6:  # 40% success rate
                system_state.security_level -= 20
                system_state.alerts_triggered += 1
                round_log.append(f"  🔴 RED exploited vulnerability! Security: {system_state.security_level}")
            else:
                system_state.alerts_triggered += 1
                round_log.append(f"  🔴 RED exploit failed (detected)")
        else:
            round_log.append(f"  🔴 RED tried to exploit but no known vulnerabilities")
    
    elif red_action == "EXFILTRATE":
        if system_state.security_level < 50:
            # Objective achieved
            round_log.append(f"  🔴 RED successfully exfiltrated data! OBJECTIVE ACHIEVED")
            objective_achieved = True
        else:
            round_log.append(f"  🔴 RED exfiltration blocked (security too high)")
            objective_achieved = False
    
    else:
        round_log.append(f"  🔴 RED action: {red_action}")
    
    # Blue team acts
    blue_result = blue_team_agent.invoke({
        "messages": [],
        "team_name": "BLUE",
        "system_state": system_state,
        "opponent_actions": state["red_team_actions"],
        "action": "",
        "reasoning": ""
    })
    
    blue_action = blue_result["action"]
    
    # Process blue team action
    if blue_action == "PATCH":
        if system_state.vulnerabilities:
            patched = system_state.vulnerabilities.pop(0)
            system_state.patches_applied.append(patched)
            round_log.append(f"  🔵 BLUE patched vulnerability: {patched}")
        else:
            round_log.append(f"  🔵 BLUE tried to patch but no known vulnerabilities")
    
    elif blue_action == "BLOCK":
        if system_state.alerts_triggered > 0:
            system_state.security_level = min(100, system_state.security_level + 10)
            round_log.append(f"  🔵 BLUE blocked attack! Security: {system_state.security_level}")
        else:
            round_log.append(f"  🔵 BLUE on alert but nothing to block")
    
    elif blue_action == "HARDEN":
        system_state.security_level = min(100, system_state.security_level + 5)
        round_log.append(f"  🔵 BLUE hardened defenses! Security: {system_state.security_level}")
    
    else:
        round_log.append(f"  🔵 BLUE action: {blue_action}")
    
    # Check if red team achieved objective
    objective_achieved = state.get("objective_achieved", False)
    if red_action == "EXFILTRATE" and system_state.security_level < 50:
        objective_achieved = True
    
    return {
        "system_state": system_state,
        "red_team_actions": [f"Round {round_num}: {red_action}"],
        "blue_team_actions": [f"Round {round_num}: {blue_action}"],
        "round_count": 1,
        "objective_achieved": objective_achieved,
        "exercise_log": ["\n".join(round_log)]
    }

def create_debrief(state: RedBlueExerciseState) -> dict:
    """Create exercise debrief"""
    logger.info("📋 Creating debrief")
    
    system_state = state["system_state"]
    
    debrief = f"""RED TEAM / BLUE TEAM EXERCISE DEBRIEF

Scenario: {state['exercise_scenario']}
Rounds: {state['round_count']}

Red Team Objective: {state['red_team_objective']}
Objective Achieved: {state['objective_achieved']}

Final System State:
- Security Level: {system_state.security_level}/100
- Unpatched Vulnerabilities: {len(system_state.vulnerabilities)}
- Patches Applied: {len(system_state.patches_applied)}
- Total Alerts: {system_state.alerts_triggered}

Exercise Log:
{chr(10).join(state['exercise_log'])}

Key Findings:
- {'RED TEAM succeeded in achieving objective' if state['objective_achieved'] else 'BLUE TEAM successfully defended system'}
- System security {'compromised' if system_state.security_level < 50 else 'maintained'}
- {'Multiple vulnerabilities remain unpatched' if system_state.vulnerabilities else 'All discovered vulnerabilities patched'}

Recommendations:
- Blue Team: {'Focus on faster patching cycle' if system_state.vulnerabilities else 'Maintain current vigilance'}
- Red Team: {'Successful tactics can be used in future exercises' if state['objective_achieved'] else 'Explore alternative attack vectors'}"""
    
    return {
        "debrief": debrief,
        "messages": [AIMessage(content=debrief)]
    }

def should_continue_exercise(state: RedBlueExerciseState) -> str:
    """Decide if exercise should continue"""
    
    if state["objective_achieved"]:
        logger.info("Red team achieved objective - exercise complete")
        return "debrief"
    
    if state["round_count"] >= state["max_rounds"]:
        logger.info("Max rounds reached - exercise complete")
        return "debrief"
    
    return "execute"

# Build red/blue exercise workflow
redblue_workflow = StateGraph(RedBlueExerciseState)
redblue_workflow.add_node("execute", execute_round)
redblue_workflow.add_node("debrief", create_debrief)

redblue_workflow.set_entry_point("execute")

redblue_workflow.add_conditional_edges(
    "execute",
    should_continue_exercise,
    {
        "execute": "execute",  # Loop
        "debrief": "debrief"
    }
)

redblue_workflow.add_edge("debrief", END)

redblue_system = redblue_workflow.compile()

# ==================== API ====================

def run_redblue_exercise(
    scenario: str = "Web application security test",
    red_objective: str = "Exfiltrate customer database",
    max_rounds: int = 8
) -> dict:
    """
    Run red team / blue team exercise.
    
    Args:
        scenario: Description of the scenario
        red_objective: What red team is trying to achieve
        max_rounds: Maximum exercise rounds
    """
    
    # Initialize system
    initial_system = SystemState(
        security_level=80,
        vulnerabilities=[],
        patches_applied=[],
        alerts_triggered=0
    )
    
    result = redblue_system.invoke({
        "messages": [HumanMessage(content=f"Starting red/blue exercise: {scenario}")],
        "exercise_scenario": scenario,
        "system_state": initial_system,
        "red_team_objective": red_objective,
        "red_team_actions": [],
        "blue_team_actions": [],
        "round_count": 0,
        "max_rounds": max_rounds,
        "objective_achieved": False,
        "exercise_log": [],
        "debrief": ""
    })
    
    return {
        "success": True,
        "scenario": scenario,
        "rounds": result["round_count"],
        "objective_achieved": result["objective_achieved"],
        "final_security": result["system_state"].security_level,
        "result": result["messages"][-1].content
    }

# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("RED TEAM / BLUE TEAM EXERCISE DEMO")
    print("="*60)
    
    result = run_redblue_exercise(
        scenario="Enterprise cloud infrastructure security assessment",
        red_objective="Gain access to production database",
        max_rounds=8
    )
    
    if result["success"]:
        print(f"\nScenario: {result['scenario']}")
        print(f"Rounds: {result['rounds']}")
        print(f"Red team success: {result['objective_achieved']}")
        print(f"Final security level: {result['final_security']}/100")
        print(f"\n{result['result']}")
```

---

## ⚖️ Part 5: Nash Equilibrium in Agent Interactions

### Theory: Nash Equilibrium

#### What Is Nash Equilibrium?

**Nash Equilibrium** is a solution concept where:
- Each agent is doing the best they can
- Given what other agents are doing
- No agent can improve by unilaterally changing strategy

**Formally:**
```
A strategy profile (s₁*, s₂*, ..., sₙ*) is a Nash Equilibrium if:
For each agent i:
  payoff(sᵢ*, s₋ᵢ*) ≥ payoff(sᵢ, s₋ᵢ*)
  for any alternative strategy sᵢ

Where s₋ᵢ* = strategies of all other agents
```

**In plain English:**
```
"I'm playing my best response to what you're playing,
and you're playing your best response to what I'm playing."
```

#### Nash Equilibrium Examples

**Example 1: Prisoner's Dilemma**
```
                Suspect B Cooperates  |  Suspect B Defects
Suspect A 
Cooperates         (-1, -1)          |      (-3, 0)
Suspect A
Defects            (0, -3)           |      (-2, -2)  ← Nash Equilibrium
```

**Nash Equilibrium: (Defect, Defect)**

Why?
- If A cooperates, B should defect (0 > -1)
- If A defects, B should defect (-2 > -3)
- If B cooperates, A should defect (0 > -1)
- If B defects, A should defect (-2 > -3)

Both defecting is stable - no one benefits from changing alone.

---

**Example 2: Coordination Game**
```
                Driver B Left  |  Driver B Right
Driver A
Left                (1, 1)     |    (-10, -10)
                      ↑
                Nash Equilibrium

Driver A  
Right           (-10, -10)     |      (1, 1)
                                        ↑
                                Nash Equilibrium
```

**Two Nash Equilibria: (Left, Left) and (Right, Right)**

Both are stable - if both go left, neither wants to switch to right.

---

**Example 3: Matching Pennies (No Pure Nash Equilibrium)**
```
                Player B Heads  |  Player B Tails
Player A
Heads              (1, -1)      |     (-1, 1)
Player A
Tails             (-1, 1)       |      (1, -1)
```

**No pure strategy Nash Equilibrium**

But there IS a **mixed strategy Nash Equilibrium**: Both players randomize 50-50.

---

#### Properties of Nash Equilibrium

**1. Existence:**
- Every finite game has at least one Nash Equilibrium (possibly mixed)
- May have multiple equilibria

**2. Not Always Optimal:**
- Nash Equilibrium ≠ best outcome for all
- Prisoner's Dilemma: equilibrium is worse than mutual cooperation

**3. Self-Enforcing:**
- No need for external enforcement
- Each agent's self-interest maintains equilibrium

**4. Predictive Power:**
- Rational agents converge to Nash Equilibrium
- Useful for predicting behavior

#### Finding Nash Equilibrium

**Method 1: Best Response Analysis**
```
For each strategy combination:
1. Check if player 1 can improve by changing
2. Check if player 2 can improve by changing
3. If neither can improve → Nash Equilibrium
```

**Method 2: Iterated Elimination**
```
1. Remove strictly dominated strategies
2. Repeat until no more eliminations
3. Check remaining for Nash Equilibrium
```

**Method 3: Computational Search**
```
For complex games:
- Use algorithms (Lemke-Howson, support enumeration)
- Or approximate with learning agents
```

#### Nash Equilibrium in Multi-Agent Systems

**Application 1: Load Balancing**
```
Agents choose servers to use
Nash Equilibrium: Load distributed so no agent benefits from switching
```

**Application 2: Pricing**
```
Competing services set prices
Nash Equilibrium: Prices where neither benefits from changing alone
```

**Application 3: Resource Allocation**
```
Agents compete for shared resources
Nash Equilibrium: Allocation where no one benefits from changing bid
```

**Application 4: Network Routing**
```
Packets choose routes through network
Nash Equilibrium: Routes where no packet improves travel time by switching
```

#### Limitations of Nash Equilibrium

**1. Multiple Equilibria:**
```
Which one will agents reach?
Coordination problem remains
```

**2. Computation:**
```
Finding Nash Equilibrium is hard (PPAD-complete)
May require approximations
```

**3. Rationality Assumption:**
```
Assumes perfect rationality
Real agents may not find equilibrium
```

**4. Static Analysis:**
```
Doesn't capture learning dynamics
How do agents reach equilibrium?
```

---

### Implementation: Nash Equilibrium Finder

```python
from typing import TypedDict, List, Tuple, Dict
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== NASH EQUILIBRIUM FINDER ====================

@dataclass
class Game2x2:
    """2x2 game representation"""
    name: str
    strategy_a: List[str]  # Two strategies for player A
    strategy_b: List[str]  # Two strategies for player B
    payoffs: List[List[Tuple[float, float]]]  # (A's payoff, B's payoff)
    
    def get_payoff(self, i: int, j: int) -> Tuple[float, float]:
        """Get payoff for strategy indices"""
        return self.payoffs[i][j]

def find_pure_nash_equilibria(game: Game2x2) -> List[Tuple[int, int]]:
    """Find pure strategy Nash equilibria"""
    
    logger.info(f"Finding Nash equilibria for: {game.name}")
    
    equilibria = []
    
    # Check each strategy combination
    for i in range(2):  # Player A's strategies
        for j in range(2):  # Player B's strategies
            
            payoff_a, payoff_b = game.get_payoff(i, j)
            
            # Check if A can improve by switching
            alt_i = 1 - i
            alt_payoff_a, _ = game.get_payoff(alt_i, j)
            
            if alt_payoff_a > payoff_a:
                # A would benefit from switching - not equilibrium
                continue
            
            # Check if B can improve by switching
            alt_j = 1 - j
            _, alt_payoff_b = game.get_payoff(i, alt_j)
            
            if alt_payoff_b > payoff_b:
                # B would benefit from switching - not equilibrium
                continue
            
            # Neither can improve - this is Nash Equilibrium
            equilibria.append((i, j))
            logger.info(f"  Found equilibrium: ({game.strategy_a[i]}, {game.strategy_b[j]}) = ({payoff_a}, {payoff_b})")
    
    if not equilibria:
        logger.info("  No pure strategy Nash equilibria found")
    
    return equilibria

def analyze_game(game: Game2x2) -> str:
    """Analyze game and find Nash equilibria"""
    
    # Display game matrix
    matrix = f"""
Game: {game.name}

Payoff Matrix:
                    {game.strategy_b[0]:15s} | {game.strategy_b[1]:15s}
{game.strategy_a[0]:15s}   {str(game.payoffs[0][0]):15s} | {str(game.payoffs[0][1]):15s}
{game.strategy_a[1]:15s}   {str(game.payoffs[1][0]):15s} | {str(game.payoffs[1][1]):15s}
"""
    
    # Find equilibria
    equilibria = find_pure_nash_equilibria(game)
    
    if equilibria:
        eq_descriptions = []
        for i, j in equilibria:
            payoff_a, payoff_b = game.get_payoff(i, j)
            eq_descriptions.append(
                f"  - ({game.strategy_a[i]}, {game.strategy_b[j]}) with payoffs ({payoff_a}, {payoff_b})"
            )
        
        equilibria_text = "\n".join(eq_descriptions)
        
        analysis = f"""{matrix}
Nash Equilibria Found: {len(equilibria)}
{equilibria_text}

Interpretation:
These are stable outcomes where neither player benefits from unilaterally changing strategy.
"""
    else:
        analysis = f"""{matrix}
Nash Equilibria Found: 0

No pure strategy Nash equilibria exist. The game may have a mixed strategy equilibrium
where players randomize between strategies.
"""
    
    return analysis

# ==================== EXAMPLE GAMES ====================

# Prisoner's Dilemma
prisoners_dilemma = Game2x2(
    name="Prisoner's Dilemma",
    strategy_a=["Cooperate", "Defect"],
    strategy_b=["Cooperate", "Defect"],
    payoffs=[
        [(-1, -1), (-3, 0)],
        [(0, -3), (-2, -2)]
    ]
)

# Coordination Game
coordination = Game2x2(
    name="Coordination Game",
    strategy_a=["Left", "Right"],
    strategy_b=["Left", "Right"],
    payoffs=[
        [(10, 10), (-5, -5)],
        [(-5, -5), (10, 10)]
    ]
)

# Battle of Sexes
battle_of_sexes = Game2x2(
    name="Battle of the Sexes",
    strategy_a=["Opera", "Football"],
    strategy_b=["Opera", "Football"],
    payoffs=[
        [(2, 1), (0, 0)],
        [(0, 0), (1, 2)]
    ]
)

# Matching Pennies
matching_pennies = Game2x2(
    name="Matching Pennies",
    strategy_a=["Heads", "Tails"],
    strategy_b=["Heads", "Tails"],
    payoffs=[
        [(1, -1), (-1, 1)],
        [(-1, 1), (1, -1)]
    ]
)

# Hawk-Dove
hawk_dove = Game2x2(
    name="Hawk-Dove Game",
    strategy_a=["Hawk", "Dove"],
    strategy_b=["Hawk", "Dove"],
    payoffs=[
        [(-5, -5), (10, 0)],
        [(0, 10), (5, 5)]
    ]
)

# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("NASH EQUILIBRIUM ANALYSIS")
    print("="*60)
    
    games = [
        prisoners_dilemma,
        coordination,
        battle_of_sexes,
        matching_pennies,
        hawk_dove
    ]
    
    for game in games:
        print("\n" + "="*60)
        result = analyze_game(game)
        print(result)
```

---

## 📋 Best Practices and Summary

### Best Practices for Competitive Multi-Agent Systems

**1. Adversarial Design:**
- Model opponent strategies
- Test with realistic adversaries
- Update defenses continuously
- Document attack patterns

**2. Game Theory Application:**
- Identify game structure
- Find equilibria
- Design incentive-compatible mechanisms
- Consider repeated interactions

**3. Auction Design:**
- Choose mechanism for goals (efficiency, revenue, truth-telling)
- Test with various bidder strategies
- Handle edge cases (no bids, ties)
- Monitor for collusion

**4. Red Team / Blue Team:**
- Define clear objectives and rules
- Debrief thoroughly after exercises
- Implement lessons learned
- Cycle regularly

**5. Nash Equilibrium:**
- Verify equilibria exist
- Handle multiple equilibria
- Consider learning dynamics
- Don't assume perfect rationality

---

## ✅ Chapter 15 Complete!

**You now understand:**
- ✅ Adversarial agent design (attacker-defender dynamics)
- ✅ Game theory fundamentals (payoffs, strategies, equilibria)
- ✅ Auction mechanisms (first-price, second-price, English)
- ✅ Bidding strategies (truthful, shading, aggressive)
- ✅ Red team / blue team patterns (security testing)
- ✅ Nash equilibrium (stability, finding, applications)
- ✅ When to use competitive vs cooperative agents
- ✅ Best practices for competitive systems

**Key Takeaways:**
- Competition drives different dynamics than cooperation
- Game theory provides formal framework for analysis
- Mechanism design shapes agent behavior
- Nash equilibrium predicts stable outcomes
- Red team / blue team improves security
- Choose mechanisms aligned with goals

---

**Ready for Chapter 16?**

**Chapter 16: Advanced State Management** will cover:
- Custom reducers and state updates
- Branching and parallel execution
- Sub-graphs and nested workflows
- Dynamic graph construction
- Streaming and async patterns

Just say "Continue to Chapter 16" when ready!