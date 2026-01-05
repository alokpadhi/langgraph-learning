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