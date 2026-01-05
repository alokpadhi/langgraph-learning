# LangGraph Course - Complete Reference Cheat Sheet

## 📚 Table of Contents
- [Ch 1-2: Foundations](#ch-1-2-foundations)
- [Ch 3: Core Concepts](#ch-3-core-concepts)
- [Ch 4: Tools & Function Calling](#ch-4-tools--function-calling)
- [Ch 5-6: State Management](#ch-5-6-state-management)
- [Ch 7: Reasoning Paradigms I](#ch-7-reasoning-paradigms-i)
- [Ch 8: Reasoning Paradigms II](#ch-8-reasoning-paradigms-ii)
- [Ch 9: Agentic RAG](#ch-9-agentic-rag)
- [Ch 10: Memory Systems](#ch-10-memory-systems)
- [Ch 11: Agent Design Patterns](#ch-11-agent-design-patterns)
- [Ch 12: Communication & Coordination](#ch-12-communication--coordination)
- [Ch 13: Cooperative Agents](#ch-13-cooperative-agents)
- [Ch 14: Roleplay & Debate](#ch-14-roleplay--debate)
- [Ch 15: Competitive & Game Theory](#ch-15-competitive--game-theory)
- [Ch 16: Advanced State Management](#ch-16-advanced-state-management)
- [Ch 17: Human-in-the-Loop](#ch-17-human-in-the-loop)
- [Ch 18: Production Engineering](#ch-18-production-engineering)

---

# Ch 1-2: Foundations

## Key Concepts

**What is LangGraph?**
- Framework for building stateful, multi-step AI agents
- Built on LangChain, adds state management and control flow
- Graph-based: nodes (functions) + edges (connections)

**Core Components:**
1. **State**: Data that flows through the graph
2. **Nodes**: Functions that process state
3. **Edges**: Connections between nodes
4. **Graph**: Complete workflow

**Why Use LangGraph?**
- ✅ Multi-step workflows with branching logic
- ✅ Stateful applications (memory, context)
- ✅ Complex agent behaviors
- ✅ Human-in-the-loop systems
- ❌ NOT for: Simple single-prompt tasks

---

# Ch 3: Core Concepts

## OODA Loop Framework

**O**bserve → **O**rient → **D**ecide → **A**ct

**Observe:**
- Gather information from environment
- User input, tool results, context
- When: Start of every agent cycle

**Orient:**
- Analyze and understand information
- Contextualize, identify patterns
- When: After gathering data

**Decide:**
- Determine best action
- Routing, prioritization
- When: Need to choose path

**Act:**
- Execute decision
- Call tools, generate response
- When: Ready to take action

**Use OODA When:**
- Building autonomous agents
- Need structured decision-making
- Complex environments
- Adaptive behavior required

## Basic Workflow Patterns

**Linear:**
```
A → B → C → End
```
Use: Sequential processing, simple pipelines

**Branching:**
```
A → [B or C] → End
```
Use: Conditional logic, routing

**Looping:**
```
A → B → [check] → Back to A or End
```
Use: Iterative refinement, retry logic

---

# Ch 4: Tools & Function Calling

## Tool Types

**1. Information Retrieval**
- Web search, database queries, API calls
- When: Need external data
- Example: Search engines, knowledge bases

**2. Action/Execution**
- Send emails, update databases, file operations
- When: Need to make changes
- Example: CRUD operations

**3. Computation**
- Math, data processing, analysis
- When: Need to calculate
- Example: Statistical analysis, formatting

**4. Validation**
- Check formats, verify data, test conditions
- When: Need to ensure correctness
- Example: Email validation, data checks

## Tool Design Principles

**Good Tool:**
- ✅ Single, clear purpose
- ✅ Well-defined inputs/outputs
- ✅ Handles errors gracefully
- ✅ Returns structured data
- ✅ Has descriptive name and description

**Bad Tool:**
- ❌ Does multiple unrelated things
- ❌ Unclear parameters
- ❌ Returns inconsistent formats
- ❌ No error handling

## When to Use Tools

**Use Tools When:**
- LLM doesn't have the information
- Need real-time/current data
- Need to take actions
- Need deterministic operations

**Don't Use Tools When:**
- LLM already knows the answer
- Simple text generation
- No external interaction needed

---

# Ch 5-6: State Management

## State Fundamentals

**What is State?**
- Data that persists across nodes
- Carries context through workflow
- TypedDict in Python

**State Design Principles:**
1. Include only necessary data
2. Use typed schemas
3. Make it serializable
4. Consider state size

## Reducers

**What:** Functions that control how state updates merge

**Common Reducers:**
- `add`: Accumulate (lists, numbers, strings)
- `replace`: Overwrite (default behavior)
- `max`/`min`: Keep extreme value
- Custom: Any merge logic you need

**When to Use:**
- `add`: Accumulating messages, events, logs
- `replace`: Single values that update
- `max`/`min`: Tracking metrics
- Custom: Complex merge requirements

## State Patterns

**Shared State:**
- All nodes access same state
- Use: Collaborative workflows
- Caution: Careful with concurrent updates

**Scoped State:**
- Nodes have local state
- Use: Isolation, modularity
- Caution: Need explicit passing

**Checkpointing:**
- Persist state to storage
- Use: Long-running workflows, recovery
- Benefit: Resume after failures

---

# Ch 7: Reasoning Paradigms I

## Chain-of-Thought (CoT)

**What:** Step-by-step reasoning before answer

**How:** Prompt with "Let's think step by step" or similar

**When to Use:**
- ✅ Math problems
- ✅ Logic puzzles
- ✅ Complex reasoning
- ✅ Need to show work
- ❌ Simple direct questions
- ❌ Time-critical responses

**Benefits:**
- Improves accuracy
- Explainable reasoning
- Catches errors early

**Variations:**
- Zero-shot CoT: Just add "think step by step"
- Few-shot CoT: Provide examples with reasoning

## Tree-of-Thoughts (ToT)

**What:** Explore multiple reasoning paths, evaluate, choose best

**Structure:**
```
Generate multiple options
    ↓
Evaluate each option
    ↓
Select best path(s)
    ↓
Continue or explore deeper
```

**When to Use:**
- ✅ Creative tasks (writing, design)
- ✅ Strategic planning
- ✅ Complex problems with multiple solutions
- ✅ Need to explore alternatives
- ❌ Single correct answer
- ❌ Time/cost constrained

**Key Difference from CoT:**
- CoT: Single path reasoning
- ToT: Multiple paths, evaluated

---

# Ch 8: Reasoning Paradigms II

## ReAct (Reason + Act)

**What:** Interleave reasoning and actions

**Pattern:**
```
Thought: What should I do?
Action: Use a tool
Observation: Tool result
Thought: What does this mean?
Action: Next tool
... (repeat)
Answer: Final response
```

**When to Use:**
- ✅ Need to use tools/APIs
- ✅ Information needs gathering
- ✅ Multi-step investigations
- ✅ Web search, data retrieval
- ❌ No external tools needed
- ❌ LLM has all info

**Key Components:**
- Thought: Reasoning step
- Action: Tool to use + input
- Observation: Tool result

## Reflection

**What:** Generate → Critique → Improve (iterative)

**Pattern:**
```
Generate initial output
    ↓
Critic reviews and provides feedback
    ↓
Generator improves based on feedback
    ↓
Repeat until quality threshold or max iterations
```

**When to Use:**
- ✅ Code generation
- ✅ Content creation
- ✅ Quality-critical outputs
- ✅ Have time for iteration
- ❌ First draft is good enough
- ❌ Strict time limits
- ❌ Cost-sensitive (multiple LLM calls)

**Roles:**
- Generator: Creates output
- Critic: Identifies issues
- (Can be same or different LLMs)

**Stopping Criteria:**
- Quality threshold met
- Max iterations reached
- No more improvements
- Diminishing returns

---

# Ch 9: Agentic RAG

## RAG Fundamentals

**Basic RAG:**
```
Query → Retrieve docs → Generate answer
```

**Agentic RAG:**
```
Query → Decide: need docs? → Route to appropriate retrieval
     → Evaluate results → Re-query if needed
     → Generate answer
```

## RAG Patterns

**1. Simple RAG**
- Retrieve once, generate answer
- Use: Straightforward Q&A
- Limitation: No retry, fixed retrieval

**2. Corrective RAG**
- Retrieve → Grade relevance → Re-retrieve if poor
- Use: Need high quality docs
- Benefit: Self-correcting

**3. Self-RAG**
- Generate → Check if needs retrieval → Retrieve → Generate
- Use: Uncertain if docs needed
- Benefit: Dynamic retrieval

**4. Adaptive RAG**
- Route to appropriate strategy based on query type
- Use: Diverse query types
- Benefit: Optimal strategy per query

**5. Multi-Source RAG**
- Retrieve from multiple sources (docs + web + database)
- Use: Need comprehensive info
- Benefit: Broader coverage

## RAG Components

**Retriever:**
- Vector search (similarity)
- Keyword search (BM25)
- Hybrid (combine both)

**Grading:**
- Relevance scoring
- Filter low-quality docs
- Use LLM or classifier

**Query Transformation:**
- Rewrite unclear queries
- Decompose complex queries
- Generate sub-queries

**When to Use RAG:**
- ✅ Need to reference specific documents
- ✅ Information changes frequently
- ✅ Can't fit all data in context
- ✅ Need citations/sources
- ❌ LLM knowledge sufficient
- ❌ Static, limited information
- ❌ No documents to retrieve from

---

# Ch 10: Memory Systems

## Memory Types

**1. Short-Term (Working) Memory**
- Current conversation context
- Cleared after session
- Use: Single conversation
- Size: Recent messages only

**2. Long-Term Memory**
- Persisted across sessions
- Stored in database
- Use: Personalization, learning
- Size: Unlimited (with retrieval)

**3. Episodic Memory**
- Specific past events/conversations
- Searchable by context
- Use: "Remember when we discussed X?"
- Access: Semantic search

**4. Semantic Memory**
- Facts and knowledge learned
- Organized by concepts
- Use: User preferences, facts
- Access: Key-value or graph

## Memory Strategies

**Buffer Memory:**
- Keep last N messages
- Simple, fast
- Use: Short conversations
- Limitation: No long-term retention

**Summary Memory:**
- Periodically summarize and compress
- Saves context space
- Use: Long conversations
- Caution: May lose details

**Entity Memory:**
- Track entities (people, places, things)
- Extract and update facts
- Use: Personalization
- Benefit: Structured knowledge

**Vector Memory:**
- Embed and retrieve similar past interactions
- Semantic search
- Use: Find relevant history
- Benefit: Context-aware retrieval

## Memory Management

**When to Clear:**
- New topic/conversation
- Privacy requirements
- User request
- Storage limits

**When to Persist:**
- User preferences
- Learned facts
- Important history
- Cross-session context

**Memory Pitfalls:**
- ⚠️ Outdated information
- ⚠️ Privacy concerns
- ⚠️ Context window overflow
- ⚠️ Conflicting memories

---

# Ch 11: Agent Design Patterns

## Single Agent

**When:** 
- One clear task
- Simple workflow
- No need for specialization

**Pattern:**
```
Input → Process → Output
```

**Example:** Document summarization

## Router Pattern

**When:**
- 2-5 distinct task types
- Classification needed
- Different handlers per type

**Pattern:**
```
Input → Classify → Route to specialist → Output
```

**Example:** Customer support routing

## Supervisor Pattern

**When:**
- Multiple specialist agents
- Need coordination
- 3-7 subtasks

**Pattern:**
```
Supervisor coordinates workers
Workers report back to supervisor
Supervisor synthesizes results
```

**Example:** Research team with specialists

## Hierarchical Pattern

**When:**
- Complex organization
- Multiple management levels
- >7 tasks

**Pattern:**
```
Top supervisor
  ├─ Mid-level manager 1
  │   ├─ Worker A
  │   └─ Worker B
  └─ Mid-level manager 2
      ├─ Worker C
      └─ Worker D
```

**Example:** Enterprise workflow automation

## Pipeline Pattern

**When:**
- Sequential processing stages
- Each stage transforms data
- Linear flow

**Pattern:**
```
Stage 1 → Stage 2 → Stage 3 → Output
```

**Example:** ETL (Extract-Transform-Load)

## Subgraph Pattern

**When:**
- Reusable components
- Modular workflows
- Need encapsulation

**Pattern:**
```
Main graph uses compiled subgraphs as nodes
```

**Example:** Shared validation/processing modules

---

# Ch 12: Communication & Coordination

## Communication Patterns

**1. Direct Invocation**
- Agent A directly calls Agent B
- **Use:** Same process, synchronous, simple
- **Avoid:** Distributed systems, need async
- **Benefit:** Simple, fast
- **Drawback:** Tight coupling

**2. Shared State**
- Agents read/write common state
- **Use:** Collaborative workflows, same process
- **Avoid:** Need isolation, testing
- **Benefit:** Easy data sharing
- **Drawback:** Concurrency issues

**3. Message Passing**
- Agents exchange messages via queue/bus
- **Use:** Distributed, async, decoupled
- **Avoid:** Simple same-process tasks
- **Benefit:** Loose coupling, scalable
- **Drawback:** Complexity, overhead

**4. Isolated State**
- Each agent has own state
- **Use:** Independence, privacy, testing
- **Avoid:** Need shared data access
- **Benefit:** No interference
- **Drawback:** Harder coordination

## Coordination Strategies

**Centralized Coordination:**
- One coordinator manages all agents
- **Use:** Clear hierarchy needed
- **Example:** Supervisor pattern

**Decentralized Coordination:**
- Agents self-organize
- **Use:** Peer-to-peer collaboration
- **Example:** Consensus building

**Hybrid Coordination:**
- Mix of centralized and decentralized
- **Use:** Complex systems
- **Example:** Hierarchical with team autonomy

## Synchronization

**Synchronous:**
- Wait for response before continuing
- **Use:** Sequential dependencies
- **Drawback:** Blocking

**Asynchronous:**
- Continue without waiting
- **Use:** Independent tasks
- **Benefit:** Parallel execution

---

# Ch 13: Cooperative Agents

## Team Collaboration

**When:** Multiple agents with different perspectives/expertise

**Pattern:**
- Agents work together toward shared goal
- Each contributes unique value
- Results combined/synthesized

**Example:** Document review by multiple specialists

**Keys:**
- Define clear roles
- Coordinate information sharing
- Synthesize contributions

## Supervisor-Worker

**When:** One agent manages, others execute

**Pattern:**
```
Supervisor: Plans, delegates, synthesizes
Workers: Execute assigned tasks
```

**Example:** Research project with coordinator

**Keys:**
- Supervisor doesn't do work, coordinates
- Workers report back
- Clear task assignments

## Consensus Building

**When:** Need agreement among agents

**Patterns:**

**Voting:**
- Majority wins
- Use: Democratic decisions
- Limitation: May miss nuance

**Weighted Voting:**
- Some agents have more influence
- Use: Expertise-based decisions
- Limitation: Bias toward experts

**Unanimous:**
- All must agree
- Use: Critical decisions
- Limitation: Slow, may deadlock

**Keys:**
- Define voting mechanism upfront
- Handle ties/disagreements
- Consider confidence levels

## Task Allocation

**Static Allocation:**
- Pre-assigned responsibilities
- Use: Known task distribution
- Benefit: Simple, predictable

**Dynamic Allocation:**
- Assign based on runtime factors
- Use: Workload balancing, availability
- Benefit: Flexible, optimal

**Auction-Based:**
- Agents bid for tasks
- Use: Resource competition
- Benefit: Market-driven efficiency

---

# Ch 14: Roleplay & Debate

## Role-Based Agents

**What:** Agents embody specific roles/personas

**When to Use:**
- ✅ Need diverse perspectives
- ✅ Domain expertise simulation
- ✅ User testing (personas)
- ✅ Creative exploration

**Role Design:**
1. Identity: "I am a [role] with [background]"
2. Perspective: "I prioritize [values/concerns]"
3. Communication style
4. Goals and motivations
5. Constraints

**Example Roles:**
- CFO (financial focus)
- CTO (technical focus)
- Customer (user perspective)
- Critic (find issues)

**Keys:**
- Maintain role consistency
- Define clear perspectives
- Avoid role bleed

## Debate Framework

**What:** Agents argue opposing sides

**Structure:**
```
Opening: PRO presents, CON presents
Rebuttal: Each responds to other
Closing: Final arguments
Judge: Evaluates, recommends
```

**When to Use:**
- ✅ Explore both sides of decision
- ✅ Reduce confirmation bias
- ✅ Complex trade-offs
- ✅ Content creation (balanced articles)

**Roles:**
- PRO: Argues for proposition
- CON: Argues against proposition
- Judge: Objective evaluation

**Keys:**
- Set round limits
- Define proposition clearly
- Use impartial judge

## Perspective-Taking

**Levels:**
1. Surface: "I am a customer"
2. Contextual: "I value price over features"
3. Deep: Full background, motivations, constraints

**Simulation vs Prediction:**
- Prediction: Guess outcomes
- Simulation: Experience scenario, react authentically

**Persona Design:**
- Demographics
- Background
- Goals
- Pain points
- Constraints
- Values
- Technical literacy

## Iterative Refinement

**Pattern:**
```
1. Divergent: Generate many ideas
2. Critical: Evaluate strengths/weaknesses
3. Convergent: Improve and combine
4. Consensus: Agree on best approach
```

**Stopping Criteria:**
- Agreement threshold (>80%)
- Diminishing returns
- Max iterations
- Quality threshold

---

# Ch 15: Competitive & Game Theory

## Adversarial Design

**Types:**

**Zero-Sum:**
- One's gain = other's loss
- Example: Chess
- Use: Pure competition

**Negative-Sum:**
- Both lose in aggregate
- Example: Arms race
- Use: Model conflict

**Positive-Sum with Competition:**
- Both can gain, compete for share
- Example: Business competition
- Use: Realistic markets

**Design Principles:**
1. Know opponent (goals, patterns)
2. Adaptive strategy
3. Resource management
4. Risk assessment

**Strategies:**
- Aggressive: Maximize advantage, high risk
- Defensive: Minimize losses, low risk
- Adaptive: Mix based on context
- Deceptive: Mislead opponent

## Game Theory

**Key Concepts:**
- Players: Decision-makers
- Strategies: Available choices
- Payoffs: Outcomes
- Nash Equilibrium: No one can improve by changing alone

**Classic Games:**

**Prisoner's Dilemma:**
- Best individual choice ≠ best collective
- Use: Cooperation vs self-interest

**Coordination Game:**
- Multiple good outcomes, need agreement
- Use: Team coordination

**Hawk-Dove:**
- Aggression vs peaceful
- Use: Resource competition

**Applications:**
- Resource allocation
- Auction design
- Security (attacker vs defender)
- Negotiation

## Auctions

**Types:**

**English (Ascending):**
- Start low, raise until one bidder left
- Use: Maximize seller revenue
- Strategy: Bid incrementally

**Dutch (Descending):**
- Start high, drop until accepted
- Use: Quick sales
- Strategy: Wait for acceptable price

**First-Price Sealed-Bid:**
- All bid secretly, highest wins, pays own bid
- Use: Single-round efficiency
- Strategy: Shade bid below value

**Second-Price (Vickrey):**
- Highest wins, pays second-highest bid
- Use: Truth-telling incentive
- Strategy: Bid true value (dominant)

**Bidding Strategies:**
- Truthful: Bid actual value
- Shading: Bid below value
- Sniping: Wait until last moment

## Red Team / Blue Team

**Red Team (Attackers):**
- Objectives: Breach, disrupt, extract
- Mindset: Adversarial, question assumptions
- Tactics: Reconnaissance, exploitation, persistence

**Blue Team (Defenders):**
- Objectives: Prevent, detect, respond
- Mindset: Defensive, assume breach
- Tactics: Defense in depth, monitoring, patching

**Cycle:**
```
Planning → Reconnaissance → Attack → Debrief → Improve
```

**Purple Team:**
- Collaboration between red and blue
- Share techniques real-time
- Faster learning

**Beyond Security:**
- Business strategy (simulate competitors)
- Product design (find usability issues)
- Testing (stress tests)

## Nash Equilibrium

**Definition:** 
Strategy profile where no agent benefits from unilateral change

**Properties:**
- Self-enforcing (no external enforcement)
- Not always optimal (collective vs individual)
- Every finite game has at least one (possibly mixed)

**Finding Equilibria:**
1. Check each strategy combination
2. See if either player can improve by switching
3. If neither can improve → Nash Equilibrium

**Applications:**
- Load balancing (server selection)
- Pricing (competing services)
- Resource allocation
- Network routing

**Limitations:**
- Multiple equilibria (which one?)
- Assumes perfect rationality
- Computationally hard for complex games

---

# Ch 16: Advanced State Management

## Custom Reducers

**What:** Functions controlling how state updates merge

**Built-in:**
- `operator.add`: Concatenate/sum
- `lambda x, y: max(x, y)`: Keep maximum
- `lambda x, y: min(x, y)`: Keep minimum

**Custom Patterns:**
- Unique lists: Merge without duplicates
- Deep merge: Recursively merge dicts
- Limited buffer: Keep last N items
- Conditional update: Update only if valid

**When to Use:**
- Accumulating values
- Complex merge logic
- Maintaining invariants
- Custom semantics

## Branching & Parallel Execution

**Branching:**
```
One path chosen based on condition
```
**Use:** Conditional logic, routing
**Pattern:** `add_conditional_edges`

**Parallel:**
```
Multiple paths execute simultaneously
```
**Use:** Independent tasks, speedup
**Pattern:** Multiple edges from same node

**Trade-offs:**

| Aspect | Sequential | Parallel |
|--------|-----------|----------|
| Speed | Slower | Faster |
| Complexity | Simple | Complex |
| Debugging | Easy | Hard |
| Resources | Low | High |

## Sub-Graphs

**What:** Compiled workflows used as nodes in parent graph

**When to Use:**
- ✅ Logical modularity
- ✅ Reusable components
- ✅ Team collaboration
- ✅ Testing isolation
- ❌ Over-nesting (>2-3 levels)

**Benefits:**
- Encapsulation
- Reusability
- Independent testing
- Clear abstraction

**State Mapping:**
- Parent and sub-graph have different schemas
- Use wrapper node to map between them

## Dynamic Graph Construction

**What:** Build graph structure at runtime

**Approaches:**

**Conditional Structure:**
```python
if task_type == "simple":
    add simple nodes
else:
    add complex nodes
```

**Pipeline Builder:**
```python
for step in steps:
    add node for step
connect sequentially
```

**Template-Based:**
```python
base = TEMPLATES[template_name]
customize(base, options)
```

**When to Use:**
- ✅ Workflow varies by input
- ✅ User configures workflow
- ✅ Plugin architecture
- ✅ A/B testing
- ❌ Static workflow sufficient

**Challenges:**
- Validation at runtime
- Testing all configurations
- Debugging different structures
- Compilation overhead

## Streaming & Async

**Streaming:**
- Produce output incrementally
- **Use:** Long workflows, user feedback
- **Modes:** values, updates, messages

**When to Stream:**
- ✅ Long-running (>5s)
- ✅ User-facing applications
- ✅ Show progress
- ✅ Real-time data
- ❌ Fast operations (<1s)

**Async:**
- Non-blocking I/O operations
- **Use:** Multiple API calls, high throughput

**When to Use Async:**
- ✅ I/O-bound tasks
- ✅ Concurrent execution
- ✅ Many simultaneous requests
- ❌ CPU-bound tasks

---

# Ch 17: Human-in-the-Loop

## Interrupts & Breakpoints

**What:** Pause execution, wait for human input

**Types:**

**Approval Interrupts:**
```
Pause before critical action
Human approves/rejects
Continue or stop
```

**Review Interrupts:**
```
Pause after generation
Human reviews output
Can edit and resume
```

**Conditional Interrupts:**
```
Pause only if condition met
Example: Low confidence
```

**When to Use:**
- ✅ Critical/irreversible actions
- ✅ Quality control
- ✅ Compliance requirements
- ✅ Error recovery
- ❌ High-frequency operations
- ❌ Fully autonomous needed

**Checkpointing:**
- Persist state to storage
- Resume after restart
- Essential for production HITL

## Approval Workflows

**Pattern:**
```
Generate proposal
  ↓
Request approval
  ↓
Wait for human decision
  ↓
Execute if approved / Cancel if rejected
```

**Approval Patterns:**

**Single Approver:**
- One person approves
- Use: Simple decisions

**Multi-Level:**
- Manager → VP → Executive
- Use: Escalating authority

**Threshold-Based:**
- Auto-approve if < threshold
- Require approval if > threshold
- Use: Risk-based

**Committee:**
- Multiple reviewers vote
- Use: Consensus decisions

**Approval Metadata:**
- Who approved
- When approved
- Why (comments)
- Approval level

**Timeout Handling:**
- Expiration (auto-reject)
- Escalation (notify manager)
- Default action

## Dynamic User Input

**What:** Collect information during execution

**Pattern:**
```
Start workflow
  ↓
Determine what info needed
  ↓
Ask user → Wait → Receive answer
  ↓
Update state
  ↓
Repeat if more info needed
```

**Use Cases:**
- Progressive forms
- Troubleshooting (questions depend on answers)
- Conditional information gathering
- Interactive configuration

**Best Practices:**
- Validate input immediately
- Provide clear context
- Allow editing previous answers
- Show progress
- Make questions conditional

## State Editing

**What:** Modify state while paused, then resume

**When to Use:**
- Human corrects errors
- Human refines output
- Circumstances changed
- Learning from corrections

**What Can Be Edited:**
- ✅ Generated content
- ✅ Parameters
- ✅ Decisions
- ✅ Data
- ❌ Execution history
- ❌ Timestamps
- ❌ Immutable audit logs

**Resume Strategies:**
- Continue forward (most common)
- Replay from edit point
- Branch to alternative path
- Restart with edits

**Safety:**
- Validate edits
- Track changes
- Audit all modifications

## Audit Trails

**What:** Comprehensive record of all events

**What to Log:**
- **Who:** Actor (human/AI/system)
- **What:** Action taken
- **When:** Timestamp
- **Why:** Reason/context
- **Result:** Outcome

**Event Types:**
- Workflow started/completed
- Node executions
- Human approvals/rejections
- State edits
- Errors

**Storage Requirements:**
- Immutable
- Tamper-proof
- Searchable
- Secure
- Durable

**Compliance:**
- GDPR: Data access logs
- SOC 2: Security events
- HIPAA: Medical record access
- SOX: Financial transactions

**Retention:**
- Operational: 30 days
- Security: 1 year
- Compliance: 7+ years
- Financial: 10 years

---

# Ch 18: Production Engineering

## Error Handling

**Error Types:**

**Transient (Recoverable):**
- Network timeouts
- Rate limits
- Temporary unavailability
- **Strategy:** Retry with backoff

**Permanent (Non-recoverable):**
- Invalid credentials
- Malformed input
- Authorization failures
- **Strategy:** Fail fast, log, alert

**Strategies:**

**Retry with Exponential Backoff:**
- Wait increases exponentially: 1s, 2s, 4s, 8s...
- Add jitter (randomness)
- **Use:** Transient errors

**Circuit Breaker:**
- Track failures
- Open circuit after threshold
- Prevent cascading failures
- **Use:** Protect failing services

**Fallback:**
- Try primary, use fallback if fails
- **Use:** Degraded functionality OK

**Timeout:**
- Limit execution time
- Prevent hanging
- **Use:** Always

**Graceful Degradation:**
- Reduce functionality but continue
- **Use:** Non-critical features

**What to Log:**
- Error type and message
- Stack trace
- State snapshot (sanitized)
- Context (workflow_id, user_id)
- Environment info

## Performance Optimization

**Caching:**

**Response Caching:**
- Cache LLM responses
- Key by prompt + parameters
- **Use:** Repeated queries

**Semantic Caching:**
- Cache by meaning (embeddings)
- Match similar queries
- **Use:** Variations of same question

**Tiered Caching:**
- Memory → Disk → Remote
- **Use:** Different access patterns

**What to Cache:**
- ✅ LLM responses (expensive)
- ✅ Database queries (frequent reads)
- ✅ External APIs (rate limited)
- ❌ User-specific real-time data
- ❌ Critical transactions

**Other Optimizations:**
- **Batching:** Process multiple together
- **Parallel:** Execute independently
- **Connection Pooling:** Reuse connections
- **Streaming:** Don't load all in memory
- **Generators:** Lazy evaluation

## Monitoring

**Three Pillars:**

**1. Metrics (What?):**
- Request count
- Response time (avg, p95, p99)
- Error rate
- Resource usage

**2. Logs (Details):**
- Structured events
- Error messages
- Debug info
- Audit trails

**3. Traces (How?):**
- Request path through system
- Time in each component
- Dependencies

**Key Metrics:**

**System Health:**
- Uptime
- Response time
- Error rate
- Throughput

**Business:**
- Active users
- Workflows completed
- Cost per request
- User satisfaction

**Agent-Specific:**
- Nodes per workflow
- Node execution times
- Approval rates
- Cache hit rate

**Alerting:**

**Threshold:**
```
if metric > threshold:
    alert
```

**Anomaly:**
```
if metric > historical_p95:
    alert
```

**Composite:**
```
if errors high AND latency high:
    alert
```

**Severity:**
- CRITICAL: Immediate action
- WARNING: Investigate soon
- INFO: Awareness

## Deployment

**Models:**

**Monolithic:**
- Single service
- **Pros:** Simple
- **Cons:** Scales as unit

**Microservices:**
- Separate services
- **Pros:** Independent scaling
- **Cons:** Complexity

**Serverless:**
- Cloud functions
- **Pros:** Auto-scaling, pay-per-use
- **Cons:** Cold starts, limits

**Strategies:**

**Blue-Green:**
- Two environments
- Switch traffic when ready
- Easy rollback

**Rolling:**
- Gradual replacement
- Deploy to subset, expand

**Canary:**
- Route small % to new version
- Increase if metrics good
- Immediate rollback if bad

**Containers:**
- Docker for packaging
- Kubernetes for orchestration
- Ensures consistency

**Environment Management:**
- Development: Verbose, test keys
- Staging: Similar to production
- Production: Minimal logging, prod keys

## Testing

**Test Pyramid:**
```
     E2E Tests (Few)
   Integration Tests (Some)
  Unit Tests (Many)
```

**Types:**

**Unit:**
- Test individual nodes
- Fast, isolated
- Mock dependencies

**Integration:**
- Test workflow with real components
- Medium speed
- Real integrations

**End-to-End:**
- Test entire system
- Slow, expensive
- Real environment

**LLM Testing Challenges:**
- Non-deterministic outputs
- **Solutions:**
  - Mock LLM responses
  - Test behavior, not exact output
  - Use temperature=0 + seed
  - Regression tests with golden outputs

**HITL Testing:**
- Mock human decisions
- Auto-approve in tests
- Test both approval/rejection paths

**Load Testing:**
- Simulate production traffic
- Find bottlenecks
- Ensure scalability

## Scalability

**Dimensions:**

**Vertical (Scale Up):**
- More CPU/RAM on single machine
- **Pros:** Simple
- **Cons:** Limited, expensive

**Horizontal (Scale Out):**
- Add more machines
- **Pros:** Unlimited, cost-effective
- **Cons:** Complexity

**Bottlenecks:**

**LLM API Rate Limits:**
- **Solutions:** Batching, caching, multiple keys

**Stateful Workflows:**
- **Solutions:** External state (Redis, DB), sticky sessions

**Database Connections:**
- **Solutions:** Connection pooling, read replicas, caching

**Memory:**
- **Solutions:** Streaming, lazy loading, compression

**Patterns:**

**Stateless Workers:**
```
Load Balancer → Worker 1
              → Worker 2
              → Worker 3
State in external DB
```

**Queue-Based:**
```
Requests → Queue → Workers pull
Decouples submission from execution
```

**Load Balancing:**
- Round Robin: Equal distribution
- Least Connections: Route to least busy
- Weighted: Based on capacity
- Geographic: Route by location

**Async Processing:**
```
Sync: User waits for result (blocking)
Async: User gets job_id, polls status (non-blocking)
```

---

# Quick Reference Tables

## When to Use What Reasoning

| Need | Use |
|------|-----|
| Step-by-step logic | Chain-of-Thought |
| Explore multiple solutions | Tree-of-Thoughts |
| Use tools/APIs | ReAct |
| Improve quality iteratively | Reflection |
| Simple input→output | Direct Prompting |

## When to Use What Pattern

| Situation | Pattern |
|-----------|---------|
| 1 task, simple | Single Agent |
| 2-3 tasks, classify & route | Router |
| 3-7 tasks, coordinate | Supervisor |
| >7 tasks, complex org | Hierarchical |
| Reusable components | Subgraphs |
| Sequential stages | Pipeline |

## When to Add What Component

| Need | Component |
|------|-----------|
| Reference documents | RAG |
| Conversation history | Memory |
| High-stakes decisions | HITL |
| Error resilience | Error Handling |
| Performance | Caching |
| Debugging | Monitoring |
| Scale | Load Balancing |

---

# The Golden Rules

1. **Start Simple:** Use simplest architecture that works
2. **Add Complexity Only When Needed:** Don't over-engineer
3. **Measure, Don't Assume:** Profile before optimizing
4. **Design for Failure:** Errors will happen
5. **Make it Observable:** Can't fix what you can't see
6. **Test Early and Often:** Catch issues before production
7. **Think About Scale:** Even if starting small
8. **Keep Human in Control:** For critical systems
9. **Document Your Decisions:** Future you will thank you
10. **Iterate and Improve:** First version won't be perfect

---

**Print this entire cheat sheet and keep it handy when designing agent systems!**