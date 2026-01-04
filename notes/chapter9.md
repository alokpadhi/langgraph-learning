# Chapter 9: State Management & Memory

## 🎯 The Problem: Agents Need to Remember

Without memory, agents:
- ❌ Lose context between messages
- ❌ Can't recall past interactions
- ❌ Repeat questions they already asked
- ❌ Don't learn from experience
- ❌ Can't maintain long conversations

**Memory systems** give agents the ability to remember, recall, and learn.

---

## 🧠 Part 1: Understanding Memory Types

### Memory Taxonomy

```
┌─────────────────────────────────────────────────┐
│              AGENT MEMORY TYPES                  │
│                                                  │
│  ┌──────────────────┐  ┌──────────────────┐   │
│  │  SHORT-TERM      │  │   LONG-TERM      │   │
│  │   MEMORY         │  │    MEMORY        │   │
│  │                  │  │                  │   │
│  │ - Conversation   │  │ - Vector stores  │   │
│  │   buffer         │  │ - Database       │   │
│  │ - Working memory │  │ - Knowledge base │   │
│  │ - Current state  │  │ - Facts/lessons  │   │
│  │                  │  │                  │   │
│  │ Duration: Hours  │  │ Duration: Forever│   │
│  └──────────────────┘  └──────────────────┘   │
│                                                  │
│  ┌──────────────────┐  ┌──────────────────┐   │
│  │  SEMANTIC        │  │   EPISODIC       │   │
│  │   MEMORY         │  │    MEMORY        │   │
│  │                  │  │                  │   │
│  │ - Facts          │  │ - Events         │   │
│  │ - Concepts       │  │ - Conversations  │   │
│  │ - Knowledge      │  │ - Experiences    │   │
│  │                  │  │                  │   │
│  │ What: "Python    │  │ When: "I talked  │   │
│  │ is a language"   │  │ to user on 1/1"  │   │
│  └──────────────────┘  └──────────────────┘   │
└─────────────────────────────────────────────────┘
```

### Memory Characteristics

| Type | Capacity | Duration | Content | Retrieval |
|------|----------|----------|---------|-----------|
| **Short-term** | Limited (~10 messages) | Session | Recent conversation | Sequential |
| **Long-term** | Unlimited | Persistent | Facts, experiences | Semantic search |
| **Semantic** | Large | Persistent | Knowledge, facts | Similarity |
| **Episodic** | Large | Persistent | Events, timeline | Temporal/Semantic |

---

### LangChain's Built-in Utilities

1. **`trim_messages`** - Intelligently manages message history with token limits
2. **`RemoveMessage`** - Precisely removes specific messages by ID
3. **Better token management** - Counts tokens accurately
4. **Preserves system messages** - Keeps important context
5. **Production-ready** - Battle-tested by the community

## 💬 Part 2: Short-Term Memory (Conversation Buffers)

### Simple Conversation Buffer (Manual way)

```python
from typing import TypedDict, Annotated, Sequence
from operator import add
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, trim_messages
from langchain_core.prompts import ChatPromptTemplate
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# State with conversation buffer
class ConversationState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add]
    buffer_size: int  # Max messages to keep

# Initialize LLM
llm = ChatOllama(model="llama3.2", temperature=0.7)

# System prompt
SYSTEM_PROMPT = """You are a helpful assistant with memory of the conversation.
Reference previous messages when relevant to provide continuity."""

# Prompt template
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("placeholder", "{messages}")
])

# Create chain
chain = chat_prompt | llm

# Node: Chat with buffer management
def chat_node(state: ConversationState) -> dict:
    """Chat node with automatic buffer management"""
    
    try:
        logger.info(f"Processing with {len(state['messages'])} messages in buffer")
        
        # Trim messages if buffer is too large
        buffer_size = state.get("buffer_size", 10)
        messages = state["messages"]
        
        if len(messages) > buffer_size:
            # Keep most recent messages
            messages = messages[-buffer_size:]
            logger.info(f"Trimmed buffer to {buffer_size} messages")
        
        # Invoke LLM
        response = chain.invoke({"messages": messages})
        
        return {
            "messages": [response]
        }
    
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return {
            "messages": [AIMessage(content=f"Error: {str(e)}")]
        }

# Build graph
workflow = StateGraph(ConversationState)
workflow.add_node("chat", chat_node)
workflow.set_entry_point("chat")
workflow.add_edge("chat", END)

# Compile with checkpointing
checkpointer = SqliteSaver.from_conn_string("./conversation_memory.db")
conversation_agent = workflow.compile(checkpointer=checkpointer)

# API
def chat_with_memory(
    message: str,
    session_id: str = "default",
    buffer_size: int = 10
) -> dict:
    """Chat with conversation buffer"""
    
    config = {
        "configurable": {
            "thread_id": f"conv-{session_id}"
        }
    }
    
    initial_state = {
        "messages": [HumanMessage(content=message)],
        "buffer_size": buffer_size
    }
    
    try:
        result = conversation_agent.invoke(initial_state, config=config)
        
        final_message = result["messages"][-1]
        
        return {
            "success": True,
            "response": final_message.content,
            "total_messages": len(result["messages"])
        }
    
    except Exception as e:
        logger.error(f"Chat failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }

# Example usage
if __name__ == "__main__":
    session = "user-123"
    
    conversations = [
        "Hi, my name is Alice",
        "What's my name?",
        "I work as a software engineer",
        "What do I do for work?",
        "I'm interested in AI and machine learning",
        "What are my interests?"
    ]
    
    print("\n" + "="*60)
    print("CONVERSATION BUFFER DEMO")
    print("="*60)
    
    for i, msg in enumerate(conversations, 1):
        print(f"\n--- Turn {i} ---")
        print(f"User: {msg}")
        
        result = chat_with_memory(msg, session_id=session, buffer_size=10)
        
        if result["success"]:
            print(f"Agent: {result['response']}")
            print(f"(Buffer size: {result['total_messages']} messages)")
        else:
            print(f"Error: {result['error']}")
```

---

### Using `trim_messages` (The Right Way)

```python
from typing import TypedDict, Annotated, Sequence
from operator import add
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage, trim_messages
from langchain_core.prompts import ChatPromptTemplate
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== PROPER SHORT-TERM MEMORY WITH trim_messages ====================

class ConversationState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add]

# Initialize LLM
llm = ChatOllama(model="llama3.2", temperature=0.7)

# System message (will be preserved)
SYSTEM_MESSAGE = SystemMessage(
    content="You are a helpful assistant with memory of the conversation. Reference previous messages when relevant."
)

# Node: Chat with proper message trimming
def chat_with_trimming(state: ConversationState) -> dict:
    """Chat node with automatic message trimming"""
    
    try:
        # Get messages
        messages = list(state["messages"])
        
        logger.info(f"Processing with {len(messages)} messages")
        
        # Add system message if not present
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SYSTEM_MESSAGE] + messages
        
        # Use trim_messages to intelligently manage history
        # This keeps system message, trims middle, keeps recent messages
        trimmed_messages = trim_messages(
            messages,
            max_tokens=1000,  # Keep last ~1000 tokens
            strategy="last",  # Keep most recent messages
            token_counter=llm,  # Use LLM's token counter
            include_system=True,  # Always keep system message
            allow_partial=False,  # Don't split messages
            start_on="human"  # Start on human message for context
        )
        
        logger.info(f"Trimmed to {len(trimmed_messages)} messages")
        
        # Invoke LLM with trimmed messages
        response = llm.invoke(trimmed_messages)
        
        return {
            "messages": [response]
        }
    
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return {
            "messages": [AIMessage(content=f"Error: {str(e)}")]
        }

# Build graph
workflow = StateGraph(ConversationState)
workflow.add_node("chat", chat_with_trimming)
workflow.set_entry_point("chat")
workflow.add_edge("chat", END)

# Compile with checkpointing
checkpointer = SqliteSaver.from_conn_string("./proper_conversation_memory.db")
conversation_agent = workflow.compile(checkpointer=checkpointer)

# API
def chat_with_proper_memory(
    message: str,
    session_id: str = "default"
) -> dict:
    """Chat with proper memory management using trim_messages"""
    
    config = {
        "configurable": {
            "thread_id": f"conv-{session_id}"
        }
    }
    
    initial_state = {
        "messages": [HumanMessage(content=message)]
    }
    
    try:
        result = conversation_agent.invoke(initial_state, config=config)
        
        final_message = result["messages"][-1]
        
        return {
            "success": True,
            "response": final_message.content,
            "total_messages": len(result["messages"])
        }
    
    except Exception as e:
        logger.error(f"Chat failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }

# Example usage
if __name__ == "__main__":
    session = "user-123"
    
    conversations = [
        "Hi, my name is Alice",
        "What's my name?",
        "I work as a software engineer at Google",
        "What do I do for work?",
        "I'm interested in AI and machine learning",
        "What are my interests?",
        "I live in San Francisco",
        "Where do I live?",
    ]
    
    print("\n" + "="*60)
    print("PROPER CONVERSATION BUFFER WITH trim_messages")
    print("="*60)
    
    for i, msg in enumerate(conversations, 1):
        print(f"\n--- Turn {i} ---")
        print(f"User: {msg}")
        
        result = chat_with_proper_memory(msg, session_id=session)
        
        if result["success"]:
            print(f"Agent: {result['response']}")
            print(f"(Total messages in state: {result['total_messages']})")
        else:
            print(f"Error: {result['error']}")
```

---
### Advanced `trim_messages` Strategies

```python
from langchain_core.messages import trim_messages

# ==================== DIFFERENT TRIMMING STRATEGIES ====================

def demo_trimming_strategies():
    """Demonstrate different trim_messages strategies"""
    
    # Create sample conversation
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="Hi, I'm Alice"),
        AIMessage(content="Hello Alice!"),
        HumanMessage(content="I work at Google"),
        AIMessage(content="That's great!"),
        HumanMessage(content="I live in SF"),
        AIMessage(content="San Francisco is beautiful!"),
        HumanMessage(content="What's my name?"),
    ]
    
    print("\n" + "="*60)
    print("TRIM_MESSAGES STRATEGIES")
    print("="*60)
    
    # Strategy 1: Keep last N messages
    print("\n--- Strategy: LAST (keep most recent) ---")
    trimmed = trim_messages(
        messages,
        max_tokens=500,
        strategy="last",
        token_counter=llm,
        include_system=True
    )
    print(f"Original: {len(messages)} messages")
    print(f"Trimmed: {len(trimmed)} messages")
    for msg in trimmed:
        print(f"  - {type(msg).__name__}: {msg.content[:50]}")
    
    # Strategy 2: Keep first N messages
    print("\n--- Strategy: FIRST (keep oldest) ---")
    trimmed = trim_messages(
        messages,
        max_tokens=500,
        strategy="first",
        token_counter=llm,
        include_system=True
    )
    print(f"Trimmed: {len(trimmed)} messages")
    
    # Strategy 3: Start on human message
    print("\n--- Strategy: START ON HUMAN ---")
    trimmed = trim_messages(
        messages,
        max_tokens=500,
        strategy="last",
        token_counter=llm,
        include_system=True,
        start_on="human"  # Ensures we don't start mid-conversation
    )
    print(f"Trimmed: {len(trimmed)} messages")
    print(f"First user message: {type(trimmed[1]).__name__}")

if __name__ == "__main__":
    demo_trimming_strategies()
```

---

### Using `RemoveMessage` for Precise Control

```python
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import RemoveMessage
from typing import TypedDict, Annotated, Sequence
from operator import add
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== USING RemoveMessage ====================

class PreciseMemoryState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add]

def chat_node(state: PreciseMemoryState) -> dict:
    """Chat with message retention"""
    
    messages = state["messages"]
    
    # Process with LLM
    response = llm.invoke(messages)
    
    return {"messages": [response]}

def cleanup_old_messages(state: PreciseMemoryState) -> dict:
    """Remove old messages by ID"""
    
    messages = state["messages"]
    
    # Keep system message + last 6 messages (3 exchanges)
    # Remove everything else
    
    to_remove = []
    
    # Skip system message (index 0)
    # Keep last 6 messages
    # Remove everything in between
    if len(messages) > 7:  # System + 6 messages
        # Get IDs of messages to remove
        for msg in messages[1:-6]:  # Skip system, keep last 6
            to_remove.append(RemoveMessage(id=msg.id))
        
        logger.info(f"Removing {len(to_remove)} old messages")
        
        return {"messages": to_remove}
    
    return {}

# Build graph with cleanup
workflow = StateGraph(PreciseMemoryState)
workflow.add_node("chat", chat_node)
workflow.add_node("cleanup", cleanup_old_messages)

workflow.set_entry_point("chat")
workflow.add_edge("chat", "cleanup")
workflow.add_edge("cleanup", END)

precise_agent = workflow.compile(checkpointer=SqliteSaver.from_conn_string("./precise_memory.db"))

# Example
def test_remove_message():
    """Test RemoveMessage"""
    
    session = "test-remove"
    config = {"configurable": {"thread_id": session}}
    
    # Have a long conversation
    for i in range(10):
        result = precise_agent.invoke(
            {"messages": [HumanMessage(content=f"Message {i+1}")]},
            config=config
        )
        
        print(f"After message {i+1}: {len(result['messages'])} total messages in state")

if __name__ == "__main__":
    test_remove_message()
```

---

### Automatic Summarization Pattern

```python
from typing import TypedDict, Annotated, Sequence
from operator import add
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage, RemoveMessage
from langchain_core.prompts import ChatPromptTemplate
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== SUMMARIZATION PATTERN ====================

class SummarizationState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add]
    summary: str

# Initialize
llm = ChatOllama(model="llama3.2", temperature=0.3)

# Summarization prompt
summarize_prompt = ChatPromptTemplate.from_messages([
    ("human", """Summarize this conversation concisely, preserving key information:

{conversation}

Provide a 2-3 sentence summary:""")
])

summarize_chain = summarize_prompt | llm

def chat_with_summary(state: SummarizationState) -> dict:
    """Chat using summary + recent messages"""
    
    messages = list(state["messages"])
    
    # Build context with summary
    context_messages = []
    
    if state.get("summary"):
        context_messages.append(
            SystemMessage(content=f"Conversation summary: {state['summary']}")
        )
    
    context_messages.extend(messages)
    
    # Invoke LLM
    response = llm.invoke(context_messages)
    
    return {"messages": [response]}

def summarize_and_trim(state: SummarizationState) -> dict:
    """Summarize old messages and trim"""
    
    messages = [
        m for m in state["messages"]
        if isinstance(m, (HumanMessage, AIMessage, SystemMessage))
    ]
    
    # Threshold for summarization
    if len(messages) < 10:
        return {}
    
    logger.info("Creating summary and trimming messages...")
    
    # Messages to summarize (all except last 4)
    old_messages = messages[:-4]
    
    # Create summary
    conversation_text = "\n".join([
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
        for m in old_messages
    ])
    
    summary_response = summarize_chain.invoke({"conversation": conversation_text})
    new_summary = summary_response.content
    
    # Combine with old summary if exists
    if state.get("summary"):
        combined_summary = f"{state['summary']}\n\nRecent: {new_summary}"
    else:
        combined_summary = new_summary
    
    logger.info(f"Created summary: {new_summary[:100]}...")
    
    # Remove old messages
    messages_to_remove = [RemoveMessage(id=m.id) for m in old_messages]
    
    return {
        "summary": combined_summary,
        "messages": messages_to_remove
    }

# Build graph
workflow = StateGraph(SummarizationState)
workflow.add_node("chat", chat_with_summary)
workflow.add_node("summarize", summarize_and_trim)

workflow.set_entry_point("chat")
workflow.add_edge("chat", "summarize")
workflow.add_edge("summarize", END)

summarization_agent = workflow.compile(
    checkpointer=SqliteSaver.from_conn_string("./summarization_memory.db")
)

# API
def chat_with_summarization(message: str, session_id: str = "default") -> dict:
    """Chat with automatic summarization"""
    
    config = {"configurable": {"thread_id": f"sum-{session_id}"}}
    
    result = summarization_agent.invoke(
        {"messages": [HumanMessage(content=message)], "summary": ""},
        config=config
    )
    
    return {
        "response": result["messages"][-1].content,
        "summary": result.get("summary", ""),
        "message_count": len(result["messages"])
    }

# Example
if __name__ == "__main__":
    print("\n" + "="*60)
    print("SUMMARIZATION PATTERN")
    print("="*60)
    
    session = "sum-test"
    
    # Long conversation
    for i in range(15):
        msg = f"This is message {i+1} with some content to remember"
        result = chat_with_summarization(msg, session_id=session)
        
        if i % 5 == 0:
            print(f"\nAfter message {i+1}:")
            print(f"  Messages in buffer: {result['message_count']}")
            print(f"  Has summary: {bool(result['summary'])}")
            if result['summary']:
                print(f"  Summary: {result['summary'][:100]}...")
```

---

## 📊 Part 3: Long-Term Memory with Vector Stores

### Production Vector Store Integration

```python
from typing import TypedDict, Annotated, Sequence, List
from operator import add
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import logging
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# State with vector memory
class VectorMemoryState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add]
    retrieved_memories: List[str]
    user_id: str

# Initialize components
llm = ChatOllama(model="llama3.2", temperature=0.7)
embeddings = OllamaEmbeddings(model="llama3.2")

# Create vector store (initialize once)
try:
    vector_store = FAISS.load_local(
        "./faiss_memory",
        embeddings,
        allow_dangerous_deserialization=True
    )
    logger.info("Loaded existing vector store")
except:
    # Create new vector store
    initial_docs = [
        Document(
            page_content="System initialized",
            metadata={"timestamp": datetime.now().isoformat(), "type": "system"}
        )
    ]
    vector_store = FAISS.from_documents(initial_docs, embeddings)
    vector_store.save_local("./faiss_memory")
    logger.info("Created new vector store")

# Prompts
MEMORY_SYSTEM_PROMPT = """You are a helpful assistant with long-term memory.

Retrieved relevant memories:
{memories}

Use these memories to provide context-aware responses. Reference past conversations naturally."""

memory_prompt = ChatPromptTemplate.from_messages([
    ("system", MEMORY_SYSTEM_PROMPT),
    ("placeholder", "{messages}")
])

memory_chain = memory_prompt | llm

# Node: Retrieve relevant memories
def retrieve_memories(state: VectorMemoryState) -> dict:
    """Retrieve relevant memories from vector store"""
    
    try:
        # Get current message
        current_message = state["messages"][-1].content
        
        logger.info(f"Retrieving memories for: {current_message[:50]}...")
        
        # Search vector store
        docs = vector_store.similarity_search(
            current_message,
            k=3,  # Top 3 most relevant memories
            filter={"user_id": state["user_id"]} if state.get("user_id") else None
        )
        
        # Format memories
        memories = []
        for doc in docs:
            timestamp = doc.metadata.get("timestamp", "unknown")
            memory_type = doc.metadata.get("type", "conversation")
            memories.append(f"[{timestamp}] ({memory_type}): {doc.page_content}")
        
        logger.info(f"Retrieved {len(memories)} memories")
        
        return {
            "retrieved_memories": memories
        }
    
    except Exception as e:
        logger.error(f"Memory retrieval error: {e}")
        return {
            "retrieved_memories": []
        }

# Node: Generate response with memory
def respond_with_memory(state: VectorMemoryState) -> dict:
    """Generate response using retrieved memories"""
    
    try:
        # Format memories
        memories_text = "\n".join(state["retrieved_memories"]) if state["retrieved_memories"] else "No relevant memories found."
        
        # Invoke LLM
        response = memory_chain.invoke({
            "memories": memories_text,
            "messages": state["messages"]
        })
        
        return {
            "messages": [response]
        }
    
    except Exception as e:
        logger.error(f"Response generation error: {e}")
        return {
            "messages": [AIMessage(content=f"Error: {str(e)}")]
        }

# Node: Store new memory
def store_memory(state: VectorMemoryState) -> dict:
    """Store important information in long-term memory"""
    
    try:
        # Get last few messages
        recent_messages = state["messages"][-2:]  # Last user message and AI response
        
        # Extract user message and AI response
        user_msg = None
        ai_msg = None
        
        for msg in recent_messages:
            if isinstance(msg, HumanMessage):
                user_msg = msg.content
            elif isinstance(msg, AIMessage):
                ai_msg = msg.content
        
        if user_msg and ai_msg:
            # Check if this is important to remember
            # Simple heuristic: if user shares personal info
            important_keywords = ["my name is", "i am", "i work", "i like", "i live", "i have"]
            
            should_store = any(keyword in user_msg.lower() for keyword in important_keywords)
            
            if should_store:
                # Create memory document
                memory_content = f"User: {user_msg}\nAssistant: {ai_msg}"
                
                doc = Document(
                    page_content=memory_content,
                    metadata={
                        "timestamp": datetime.now().isoformat(),
                        "user_id": state.get("user_id", "unknown"),
                        "type": "conversation"
                    }
                )
                
                # Add to vector store
                vector_store.add_documents([doc])
                vector_store.save_local("./faiss_memory")
                
                logger.info("Stored new memory")
        
        return {}
    
    except Exception as e:
        logger.error(f"Memory storage error: {e}")
        return {}

# Build graph
workflow = StateGraph(VectorMemoryState)

workflow.add_node("retrieve", retrieve_memories)
workflow.add_node("respond", respond_with_memory)
workflow.add_node("store", store_memory)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "respond")
workflow.add_edge("respond", "store")
workflow.add_edge("store", END)

# Compile
checkpointer = SqliteSaver.from_conn_string("./vector_memory_agent.db")
vector_memory_agent = workflow.compile(checkpointer=checkpointer)

# API
def chat_with_vector_memory(
    message: str,
    user_id: str = "default"
) -> dict:
    """Chat with long-term vector memory"""
    
    config = {
        "configurable": {
            "thread_id": f"vector-mem-{user_id}"
        }
    }
    
    initial_state = {
        "messages": [HumanMessage(content=message)],
        "retrieved_memories": [],
        "user_id": user_id
    }
    
    try:
        result = vector_memory_agent.invoke(initial_state, config=config)
        
        final_message = result["messages"][-1]
        
        return {
            "success": True,
            "response": final_message.content,
            "memories_retrieved": len(result["retrieved_memories"])
        }
    
    except Exception as e:
        logger.error(f"Chat failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }

# Example usage
if __name__ == "__main__":
    user = "alice-123"
    
    # First conversation
    print("\n" + "="*60)
    print("VECTOR MEMORY DEMO - First Conversation")
    print("="*60)
    
    messages_1 = [
        "Hi, my name is Alice and I'm a software engineer",
        "I work primarily with Python and I love building AI applications",
        "My favorite hobby is rock climbing"
    ]
    
    for msg in messages_1:
        print(f"\nUser: {msg}")
        result = chat_with_vector_memory(msg, user_id=user)
        if result["success"]:
            print(f"Agent: {result['response']}")
    
    print("\n" + "="*60)
    print("VECTOR MEMORY DEMO - Later Conversation")
    print("="*60)
    
    # Later conversation - test memory recall
    messages_2 = [
        "What's my name?",
        "What do I do for work?",
        "What are my hobbies?"
    ]
    
    for msg in messages_2:
        print(f"\nUser: {msg}")
        result = chat_with_vector_memory(msg, user_id=user)
        if result["success"]:
            print(f"Agent: {result['response']}")
            print(f"(Retrieved {result['memories_retrieved']} relevant memories)")
```

---

## 🔀 Part 4: Semantic vs Episodic Memory

### Hybrid Memory System

```python
from typing import TypedDict, Annotated, Sequence, List, Literal
from operator import add
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# State with hybrid memory
class HybridMemoryState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add]
    semantic_memories: List[str]  # Facts, knowledge
    episodic_memories: List[str]  # Events, conversations
    user_id: str
    memory_type_needed: Literal["semantic", "episodic", "both"]

# Initialize
llm = ChatOllama(model="llama3.2", temperature=0.7)
embeddings = OllamaEmbeddings(model="llama3.2")

# Create separate vector stores
try:
    semantic_store = FAISS.load_local(
        "./semantic_memory",
        embeddings,
        allow_dangerous_deserialization=True
    )
    episodic_store = FAISS.load_local(
        "./episodic_memory",
        embeddings,
        allow_dangerous_deserialization=True
    )
    logger.info("Loaded existing memory stores")
except:
    # Create new stores
    init_doc = [Document(page_content="System initialized", metadata={"timestamp": datetime.now().isoformat()})]
    semantic_store = FAISS.from_documents(init_doc, embeddings)
    episodic_store = FAISS.from_documents(init_doc, embeddings)
    semantic_store.save_local("./semantic_memory")
    episodic_store.save_local("./episodic_memory")
    logger.info("Created new memory stores")

# Classifier prompt
classifier_prompt = ChatPromptTemplate.from_messages([
    ("human", """Analyze this query and determine what type of memory is needed:

Query: {query}

Options:
- SEMANTIC: For factual questions (What is X? How does Y work? Define Z)
- EPISODIC: For personal/temporal questions (When did I? What did we discuss? Remember when?)
- BOTH: For questions needing both types

Respond with ONLY one word: SEMANTIC, EPISODIC, or BOTH

Classification:""")
])

classifier_chain = classifier_prompt | llm

# Nodes
def classify_memory_need(state: HybridMemoryState) -> dict:
    """Classify what type of memory is needed"""
    
    try:
        current_message = state["messages"][-1].content
        
        response = classifier_chain.invoke({"query": current_message})
        classification = response.content.strip().upper()
        
        # Map to our types
        if "SEMANTIC" in classification and "EPISODIC" not in classification:
            memory_type = "semantic"
        elif "EPISODIC" in classification and "SEMANTIC" not in classification:
            memory_type = "episodic"
        else:
            memory_type = "both"
        
        logger.info(f"Memory type needed: {memory_type}")
        
        return {
            "memory_type_needed": memory_type
        }
    
    except Exception as e:
        logger.error(f"Classification error: {e}")
        return {
            "memory_type_needed": "both"  # Default to both
        }

def retrieve_semantic_memory(state: HybridMemoryState) -> dict:
    """Retrieve factual/knowledge memories"""
    
    try:
        if state["memory_type_needed"] not in ["semantic", "both"]:
            return {"semantic_memories": []}
        
        current_message = state["messages"][-1].content
        
        docs = semantic_store.similarity_search(
            current_message,
            k=3,
            filter={"user_id": state.get("user_id")} if state.get("user_id") else None
        )
        
        memories = [f"Fact: {doc.page_content}" for doc in docs]
        
        logger.info(f"Retrieved {len(memories)} semantic memories")
        
        return {
            "semantic_memories": memories
        }
    
    except Exception as e:
        logger.error(f"Semantic retrieval error: {e}")
        return {"semantic_memories": []}

def retrieve_episodic_memory(state: HybridMemoryState) -> dict:
    """Retrieve event/conversation memories"""
    
    try:
        if state["memory_type_needed"] not in ["episodic", "both"]:
            return {"episodic_memories": []}
        
        current_message = state["messages"][-1].content
        
        docs = episodic_store.similarity_search(
            current_message,
            k=3,
            filter={"user_id": state.get("user_id")} if state.get("user_id") else None
        )
        
        memories = []
        for doc in docs:
            timestamp = doc.metadata.get("timestamp", "unknown")
            memories.append(f"[{timestamp}] {doc.page_content}")
        
        logger.info(f"Retrieved {len(memories)} episodic memories")
        
        return {
            "episodic_memories": memories
        }
    
    except Exception as e:
        logger.error(f"Episodic retrieval error: {e}")
        return {"episodic_memories": []}

def respond_with_hybrid_memory(state: HybridMemoryState) -> dict:
    """Generate response using both memory types"""
    
    try:
        # Combine memories
        all_memories = []
        
        if state.get("semantic_memories"):
            all_memories.append("=== Factual Knowledge ===")
            all_memories.extend(state["semantic_memories"])
        
        if state.get("episodic_memories"):
            all_memories.append("\n=== Past Conversations ===")
            all_memories.extend(state["episodic_memories"])
        
        memories_text = "\n".join(all_memories) if all_memories else "No relevant memories."
        
        # Create prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant with two types of memory:
1. Semantic memory: Facts and knowledge
2. Episodic memory: Past conversations and events

Retrieved memories:
{memories}

Use these memories to provide informed responses."""),
            ("placeholder", "{messages}")
        ])
        
        chain = prompt | llm
        response = chain.invoke({
            "memories": memories_text,
            "messages": state["messages"]
        })
        
        return {
            "messages": [response]
        }
    
    except Exception as e:
        logger.error(f"Response error: {e}")
        return {
            "messages": [AIMessage(content=f"Error: {str(e)}")]
        }

def store_hybrid_memory(state: HybridMemoryState) -> dict:
    """Store information in appropriate memory type"""
    
    try:
        recent_messages = state["messages"][-2:]
        
        user_msg = None
        ai_msg = None
        
        for msg in recent_messages:
            if isinstance(msg, HumanMessage):
                user_msg = msg.content
            elif isinstance(msg, AIMessage):
                ai_msg = msg.content
        
        if not (user_msg and ai_msg):
            return {}
        
        # Determine what to store
        user_lower = user_msg.lower()
        
        # Semantic: Facts, definitions, knowledge
        semantic_keywords = ["is a", "means", "definition", "how to", "what is", "explain"]
        is_semantic = any(keyword in user_lower for keyword in semantic_keywords)
        
        # Episodic: Personal info, events
        episodic_keywords = ["my name", "i am", "i did", "yesterday", "last week", "i like", "i work"]
        is_episodic = any(keyword in user_lower for keyword in episodic_keywords)
        
        timestamp = datetime.now().isoformat()
        user_id = state.get("user_id", "unknown")
        
        # Store in semantic memory
        if is_semantic:
            doc = Document(
                page_content=f"Q: {user_msg}\nA: {ai_msg}",
                metadata={
                    "timestamp": timestamp,
                    "user_id": user_id,
                    "type": "semantic"
                }
            )
            semantic_store.add_documents([doc])
            semantic_store.save_local("./semantic_memory")
            logger.info("Stored semantic memory")
        
        # Store in episodic memory
        if is_episodic:
            doc = Document(
                page_content=f"User said: {user_msg}",
                metadata={
                    "timestamp": timestamp,
                    "user_id": user_id,
                    "type": "episodic"
                }
            )
            episodic_store.add_documents([doc])
            episodic_store.save_local("./episodic_memory")
            logger.info("Stored episodic memory")
        
        return {}
    
    except Exception as e:
        logger.error(f"Storage error: {e}")
        return {}

# Build graph
workflow = StateGraph(HybridMemoryState)

workflow.add_node("classify", classify_memory_need)
workflow.add_node("retrieve_semantic", retrieve_semantic_memory)
workflow.add_node("retrieve_episodic", retrieve_episodic_memory)
workflow.add_node("respond", respond_with_hybrid_memory)
workflow.add_node("store", store_hybrid_memory)

workflow.set_entry_point("classify")
workflow.add_edge("classify", "retrieve_semantic")
workflow.add_edge("retrieve_semantic", "retrieve_episodic")
workflow.add_edge("retrieve_episodic", "respond")
workflow.add_edge("respond", "store")
workflow.add_edge("store", END)

# Compile
hybrid_memory_agent = workflow.compile()

# API
def chat_with_hybrid_memory(message: str, user_id: str = "default") -> dict:
    """Chat with hybrid semantic/episodic memory"""
    
    initial_state = {
        "messages": [HumanMessage(content=message)],
        "semantic_memories": [],
        "episodic_memories": [],
        "user_id": user_id,
        "memory_type_needed": "both"
    }
    
    try:
        result = hybrid_memory_agent.invoke(initial_state)
        
        final_message = result["messages"][-1]
        
        return {
            "success": True,
            "response": final_message.content,
            "memory_type_used": result["memory_type_needed"],
            "semantic_count": len(result["semantic_memories"]),
            "episodic_count": len(result["episodic_memories"])
        }
    
    except Exception as e:
        logger.error(f"Chat failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }

# Example usage
if __name__ == "__main__":
    user = "bob-456"
    
    print("\n" + "="*60)
    print("HYBRID MEMORY DEMO")
    print("="*60)
    
    # Store some facts (semantic)
    print("\n--- Storing Semantic Memories ---")
    semantic_queries = [
        "What is Python? Python is a high-level programming language",
        "How does vector search work? It uses embeddings to find similar content"
    ]
    
    for query in semantic_queries:
        print(f"\nUser: {query}")
        result = chat_with_hybrid_memory(query, user_id=user)
        if result["success"]:
            print(f"Agent: {result['response'][:100]}...")
    
    # Store personal info (episodic)
    print("\n--- Storing Episodic Memories ---")
    episodic_queries = [
        "My name is Bob and I'm learning about AI",
        "Yesterday I built my first LangGraph agent"
    ]
    
    for query in episodic_queries:
        print(f"\nUser: {query}")
        result = chat_with_hybrid_memory(query, user_id=user)
        if result["success"]:
            print(f"Agent: {result['response'][:100]}...")
    
    # Test recall
    print("\n--- Testing Memory Recall ---")
    test_queries = [
        "What is Python?",  # Should use semantic
        "What's my name?",  # Should use episodic
        "Tell me about what I did yesterday and also explain vector search"  # Should use both
    ]
    
    for query in test_queries:
        print(f"\nUser: {query}")
        result = chat_with_hybrid_memory(query, user_id=user)
        if result["success"]:
            print(f"Agent: {result['response']}")
            print(f"Memory type: {result['memory_type_used']}")
            print(f"Retrieved: {result['semantic_count']} semantic, {result['episodic_count']} episodic")
```

---

## ✂️ Part 5: State Pruning and Compression

### 5(a): Production Memory Management (Manual Way)

```python
from typing import TypedDict, Annotated, Sequence
from operator import add
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage, trim_messages
from langchain_core.prompts import ChatPromptTemplate
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# State with pruning config
class PrunedMemoryState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add]
    summary: str  # Compressed summary of old messages
    max_messages: int
    summarization_threshold: int

# Initialize
llm = ChatOllama(model="llama3.2", temperature=0.3)

# Summarization prompt
summarize_prompt = ChatPromptTemplate.from_messages([
    ("human", """Summarize this conversation concisely, preserving key information:

{conversation}

Summary (2-3 sentences):""")
])

summarize_chain = summarize_prompt | llm

# Node: Prune and compress
def prune_and_compress(state: PrunedMemoryState) -> dict:
    """Prune old messages and compress into summary"""
    
    try:
        messages = state["messages"]
        max_msgs = state.get("max_messages", 10)
        threshold = state.get("summarization_threshold", 20)
        
        # Check if we need to compress
        if len(messages) < threshold:
            logger.info(f"No compression needed ({len(messages)}/{threshold})")
            return {}
        
        logger.info(f"Compressing {len(messages)} messages...")
        
        # Keep most recent messages
        recent_messages = messages[-max_msgs:]
        old_messages = messages[:-max_msgs]
        
        # Compress old messages into summary
        if old_messages and not state.get("summary"):
            # Format for summarization
            conversation_text = "\n".join([
                f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
                for m in old_messages
            ])
            
            summary_response = summarize_chain.invoke({
                "conversation": conversation_text
            })
            
            new_summary = summary_response.content
            
            logger.info(f"Created summary: {new_summary[:100]}...")
            
            return {
                "summary": new_summary,
                "messages": recent_messages
            }
        
        return {}
    
    except Exception as e:
        logger.error(f"Pruning error: {e}")
        return {}

# Node: Chat with compressed memory
def chat_with_pruned_memory(state: PrunedMemoryState) -> dict:
    """Chat using summary and recent messages"""
    
    try:
        # Build context with summary
        context_messages = []
        
        if state.get("summary"):
            context_messages.append(
                SystemMessage(content=f"Previous conversation summary: {state['summary']}")
            )
        
        context_messages.extend(state["messages"])
        
        # Create prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Use the conversation history and summary to provide continuity."),
            ("placeholder", "{messages}")
        ])
        
        chain = prompt | llm
        response = chain.invoke({"messages": context_messages})
        
        return {
            "messages": [response]
        }
    
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return {
            "messages": [AIMessage(content=f"Error: {str(e)}")]
        }

# Build graph
workflow = StateGraph(PrunedMemoryState)

workflow.add_node("prune", prune_and_compress)
workflow.add_node("chat", chat_with_pruned_memory)

workflow.set_entry_point("prune")
workflow.add_edge("prune", "chat")
workflow.add_edge("chat", END)

# Compile
pruned_memory_agent = workflow.compile()

# API
def chat_with_pruned_memory_api(
    message: str,
    max_messages: int = 10,
    summarization_threshold: int = 20
) -> dict:
    """Chat with automatic memory pruning"""
    
    initial_state = {
        "messages": [HumanMessage(content=message)],
        "summary": "",
        "max_messages": max_messages,
        "summarization_threshold": summarization_threshold
    }
    
    try:
        result = pruned_memory_agent.invoke(initial_state)
        
        final_message = result["messages"][-1]
        
        return {
            "success": True,
            "response": final_message.content,
            "has_summary": bool(result.get("summary")),
            "message_count": len(result["messages"])
        }
    
    except Exception as e:
        logger.error(f"Chat failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }

# Example
if __name__ == "__main__":
    print("\n" + "="*60)
    print("MEMORY PRUNING DEMO")
    print("="*60)
    
    # Simulate long conversation
    for i in range(25):
        msg = f"Message {i+1}: This is a test message"
        result = chat_with_pruned_memory_api(
            msg,
            max_messages=5,
            summarization_threshold=15
        )
        
        if i % 5 == 0:
            print(f"\nTurn {i+1}:")
            print(f"  Has summary: {result.get('has_summary')}")
            print(f"  Messages in buffer: {result.get('message_count')}")
```

---
### Part 5(b): Production Memory with All Utilities

```python
from typing import TypedDict, Annotated, Sequence, Literal
from operator import add
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_ollama import ChatOllama
from langchain_core.messages import (
    HumanMessage, 
    AIMessage, 
    BaseMessage, 
    SystemMessage, 
    RemoveMessage,
    trim_messages
)
from langchain_core.prompts import ChatPromptTemplate
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== PRODUCTION MEMORY SYSTEM ====================

class ProductionMemoryState(TypedDict):
    """Production state with all memory management"""
    messages: Annotated[Sequence[BaseMessage], add]
    summary: str
    memory_strategy: Literal["trim", "summarize", "hybrid"]

# Initialize
llm = ChatOllama(model="llama3.2", temperature=0.7)

SYSTEM_PROMPT = "You are a helpful assistant with conversation memory."

def chat_node(state: ProductionMemoryState) -> dict:
    """Chat with context"""
    
    messages = list(state["messages"])
    
    # Add system message with summary if exists
    if state.get("summary"):
        system_msg = SystemMessage(
            content=f"{SYSTEM_PROMPT}\n\nPrevious conversation summary: {state['summary']}"
        )
    else:
        system_msg = SystemMessage(content=SYSTEM_PROMPT)
    
    # Ensure system message is first
    if messages and isinstance(messages[0], SystemMessage):
        messages = [system_msg] + messages[1:]
    else:
        messages = [system_msg] + messages
    
    # Trim messages using trim_messages
    trimmed = trim_messages(
        messages,
        max_tokens=2000,
        strategy="last",
        token_counter=llm,
        include_system=True,
        start_on="human"
    )
    
    logger.info(f"Trimmed from {len(messages)} to {len(trimmed)} messages")
    
    # Invoke LLM
    response = llm.invoke(trimmed)
    
    return {"messages": [response]}

def memory_management(state: ProductionMemoryState) -> dict:
    """Intelligent memory management"""
    
    messages = list(state["messages"])
    strategy = state.get("memory_strategy", "hybrid")
    
    # Remove system messages from count
    user_messages = [m for m in messages if not isinstance(m, SystemMessage)]
    
    logger.info(f"Managing {len(user_messages)} user messages with '{strategy}' strategy")
    
    # Hybrid strategy: summarize when > 20, trim when > 10
    if strategy == "hybrid":
        if len(user_messages) > 20:
            return summarize_old_messages(state)
        elif len(user_messages) > 10:
            return trim_old_messages(state)
        else:
            return {}
    
    elif strategy == "summarize" and len(user_messages) > 10:
        return summarize_old_messages(state)
    
    elif strategy == "trim" and len(user_messages) > 10:
        return trim_old_messages(state)
    
    return {}

def summarize_old_messages(state: ProductionMemoryState) -> dict:
    """Summarize and remove old messages"""
    
    messages = list(state["messages"])
    
    # Keep last 6 messages, summarize the rest
    if len(messages) <= 6:
        return {}
    
    old_messages = messages[:-6]
    
    # Only summarize user/AI messages
    to_summarize = [m for m in old_messages if not isinstance(m, SystemMessage)]
    
    if not to_summarize:
        return {}
    
    logger.info(f"Summarizing {len(to_summarize)} messages")
    
    # Create summary
    conv_text = "\n".join([
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
        for m in to_summarize
    ])
    
    summarize_prompt = ChatPromptTemplate.from_messages([
        ("human", "Summarize this conversation:\n\n{conversation}\n\nSummary:")
    ])
    
    chain = summarize_prompt | llm
    response = chain.invoke({"conversation": conv_text})
    
    new_summary = response.content
    
    # Combine with existing summary
    if state.get("summary"):
        combined = f"{state['summary']}\n\n{new_summary}"
    else:
        combined = new_summary
    
    # Remove old messages
    to_remove = [RemoveMessage(id=m.id) for m in old_messages]
    
    logger.info(f"Created summary, removing {len(to_remove)} messages")
    
    return {
        "summary": combined,
        "messages": to_remove
    }

def trim_old_messages(state: ProductionMemoryState) -> dict:
    """Remove old messages without summarizing"""
    
    messages = list(state["messages"])
    
    # Keep last 10 messages
    if len(messages) <= 10:
        return {}
    
    to_remove = messages[:-10]
    
    # Don't remove system messages
    to_remove = [m for m in to_remove if not isinstance(m, SystemMessage)]
    
    logger.info(f"Trimming {len(to_remove)} old messages")
    
    return {
        "messages": [RemoveMessage(id=m.id) for m in to_remove]
    }

# Build graph
workflow = StateGraph(ProductionMemoryState)
workflow.add_node("chat", chat_node)
workflow.add_node("memory_mgmt", memory_management)

workflow.set_entry_point("chat")
workflow.add_edge("chat", "memory_mgmt")
workflow.add_edge("memory_mgmt", END)

# Compile
production_memory_agent = workflow.compile(
    checkpointer=SqliteSaver.from_conn_string("./production_proper_memory.db")
)

# API
def chat_production(
    message: str,
    session_id: str = "default",
    strategy: Literal["trim", "summarize", "hybrid"] = "hybrid"
) -> dict:
    """Production chat with proper memory management"""
    
    config = {"configurable": {"thread_id": f"prod-{session_id}"}}
    
    result = production_memory_agent.invoke(
        {
            "messages": [HumanMessage(content=message)],
            "summary": "",
            "memory_strategy": strategy
        },
        config=config
    )
    
    return {
        "response": result["messages"][-1].content,
        "summary": result.get("summary", ""),
        "message_count": len([m for m in result["messages"] if not isinstance(m, SystemMessage)])
    }

# Example
if __name__ == "__main__":
    print("\n" + "="*60)
    print("PRODUCTION MEMORY MANAGEMENT")
    print("="*60)
    
    session = "prod-test"
    
    # Long conversation
    for i in range(25):
        msg = f"Message {i+1}: Tell me something interesting"
        result = chat_production(msg, session_id=session, strategy="hybrid")
        
        if i % 5 == 0:
            print(f"\nAfter message {i+1}:")
            print(f"  Messages: {result['message_count']}")
            print(f"  Has summary: {bool(result['summary'])}")
```

---

## 🎯 Part 6: Advanced Memory Retrieval Strategies

### Multi-Strategy Retrieval

```python
from typing import TypedDict, Annotated, Sequence, List, Literal
from operator import add
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# State
class MultiStrategyMemoryState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add]
    retrieved_memories: List[dict]  # {content: str, score: float, strategy: str}
    retrieval_strategy: Literal["similarity", "recency", "hybrid"]

# Initialize
llm = ChatOllama(model="llama3.2", temperature=0.7)
embeddings = OllamaEmbeddings(model="llama3.2")

# Vector store
try:
    memory_store = FAISS.load_local(
        "./multi_strategy_memory",
        embeddings,
        allow_dangerous_deserialization=True
    )
except:
    init_doc = [Document(
        page_content="System initialized",
        metadata={"timestamp": datetime.now().isoformat(), "importance": 1.0}
    )]
    memory_store = FAISS.from_documents(init_doc, embeddings)
    memory_store.save_local("./multi_strategy_memory")

# Retrieval strategies
def similarity_retrieval(query: str, k: int = 5) -> List[dict]:
    """Pure similarity-based retrieval"""
    docs = memory_store.similarity_search_with_score(query, k=k)
    
    return [
        {
            "content": doc.page_content,
            "score": float(score),
            "strategy": "similarity",
            "timestamp": doc.metadata.get("timestamp")
        }
        for doc, score in docs
    ]

def recency_retrieval(k: int = 5) -> List[dict]:
    """Retrieve most recent memories"""
    # Get all documents (in production, use a better method)
    docs = memory_store.similarity_search("", k=k*2)
    
    # Sort by timestamp
    sorted_docs = sorted(
        docs,
        key=lambda d: d.metadata.get("timestamp", ""),
        reverse=True
    )
    
    return [
        {
            "content": doc.page_content,
            "score": 1.0,  # Recency doesn't have similarity score
            "strategy": "recency",
            "timestamp": doc.metadata.get("timestamp")
        }
        for doc in sorted_docs[:k]
    ]

def hybrid_retrieval(query: str, k: int = 5) -> List[dict]:
    """Combine similarity and recency"""
    # Get similarity results
    sim_results = similarity_retrieval(query, k=k*2)
    
    # Calculate time decay
    now = datetime.now()
    
    for result in sim_results:
        try:
            timestamp = datetime.fromisoformat(result["timestamp"])
            days_old = (now - timestamp).days
            
            # Time decay: recent memories get boost
            time_factor = 1.0 / (1.0 + days_old * 0.1)  # Decays over time
            
            # Combine similarity score with time factor
            result["score"] = result["score"] * 0.7 + time_factor * 0.3
            result["strategy"] = "hybrid"
        except:
            pass
    
    # Sort by combined score
    sorted_results = sorted(sim_results, key=lambda x: x["score"], reverse=True)
    
    return sorted_results[:k]

# Node: Smart retrieval
def smart_retrieve(state: MultiStrategyMemoryState) -> dict:
    """Use appropriate retrieval strategy"""
    
    try:
        query = state["messages"][-1].content
        strategy = state.get("retrieval_strategy", "hybrid")
        
        logger.info(f"Using {strategy} retrieval strategy")
        
        if strategy == "similarity":
            memories = similarity_retrieval(query)
        elif strategy == "recency":
            memories = recency_retrieval()
        else:  # hybrid
            memories = hybrid_retrieval(query)
        
        logger.info(f"Retrieved {len(memories)} memories")
        
        return {
            "retrieved_memories": memories
        }
    
    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        return {
            "retrieved_memories": []
        }

# Node: Respond
def respond_with_retrieved_memories(state: MultiStrategyMemoryState) -> dict:
    """Generate response using retrieved memories"""
    
    try:
        # Format memories
        memories_text = []
        for mem in state["retrieved_memories"]:
            strategy = mem["strategy"]
            score = mem["score"]
            content = mem["content"]
            memories_text.append(f"[{strategy}, score: {score:.2f}] {content}")
        
        memories_str = "\n".join(memories_text) if memories_text else "No memories retrieved."
        
        # Create prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant with memory.

Retrieved memories:
{memories}

Use these memories to provide informed responses."""),
            ("placeholder", "{messages}")
        ])
        
        chain = prompt | llm
        response = chain.invoke({
            "memories": memories_str,
            "messages": state["messages"]
        })
        
        return {
            "messages": [response]
        }
    
    except Exception as e:
        logger.error(f"Response error: {e}")
        return {
            "messages": [AIMessage(content=f"Error: {str(e)}")]
        }

# Build graph
workflow = StateGraph(MultiStrategyMemoryState)

workflow.add_node("retrieve", smart_retrieve)
workflow.add_node("respond", respond_with_retrieved_memories)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "respond")
workflow.add_edge("respond", END)

# Compile
multi_strategy_agent = workflow.compile()

# API
def chat_with_strategy(
    message: str,
    strategy: Literal["similarity", "recency", "hybrid"] = "hybrid"
) -> dict:
    """Chat with specific retrieval strategy"""
    
    initial_state = {
        "messages": [HumanMessage(content=message)],
        "retrieved_memories": [],
        "retrieval_strategy": strategy
    }
    
    try:
        result = multi_strategy_agent.invoke(initial_state)
        
        final_message = result["messages"][-1]
        
        return {
            "success": True,
            "response": final_message.content,
            "strategy_used": strategy,
            "memories_retrieved": len(result["retrieved_memories"])
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

# Example
if __name__ == "__main__":
    # Add some test memories
    test_memories = [
        ("Python is a programming language", 10),  # 10 days ago
        ("I went hiking last weekend", 2),  # 2 days ago
        ("Machine learning uses neural networks", 5),  # 5 days ago
        ("I had coffee this morning", 0),  # Today
    ]
    
    for content, days_ago in test_memories:
        timestamp = (datetime.now() - timedelta(days=days_ago)).isoformat()
        doc = Document(
            page_content=content,
            metadata={"timestamp": timestamp}
        )
        memory_store.add_documents([doc])
    
    memory_store.save_local("./multi_strategy_memory")
    
    print("\n" + "="*60)
    print("RETRIEVAL STRATEGY COMPARISON")
    print("="*60)
    
    query = "Tell me about programming"
    
    for strategy in ["similarity", "recency", "hybrid"]:
        print(f"\n--- Using {strategy.upper()} strategy ---")
        result = chat_with_strategy(query, strategy=strategy)
        if result["success"]:
            print(f"Response: {result['response'][:150]}...")
            print(f"Memories retrieved: {result['memories_retrieved']}")
```

## 🏆 Part 7: Complete Production Memory System

### Enterprise-Grade Memory Architecture

```python
from typing import TypedDict, Annotated, Sequence, List, Dict, Optional, Literal
from operator import add
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import logging
import json
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== DATA MODELS ====================

@dataclass
class Memory:
    """Structured memory object"""
    content: str
    memory_type: Literal["semantic", "episodic", "procedural"]
    importance: float  # 0.0 to 1.0
    timestamp: str
    user_id: str
    metadata: Dict
    
    def to_document(self) -> Document:
        """Convert to LangChain Document"""
        return Document(
            page_content=self.content,
            metadata={
                "memory_type": self.memory_type,
                "importance": self.importance,
                "timestamp": self.timestamp,
                "user_id": self.user_id,
                **self.metadata
            }
        )
    
    @classmethod
    def from_document(cls, doc: Document) -> "Memory":
        """Create from LangChain Document"""
        metadata = doc.metadata.copy()
        return cls(
            content=doc.page_content,
            memory_type=metadata.pop("memory_type", "semantic"),
            importance=metadata.pop("importance", 0.5),
            timestamp=metadata.pop("timestamp", datetime.now().isoformat()),
            user_id=metadata.pop("user_id", "unknown"),
            metadata=metadata
        )

# ==================== STATE ====================

class ProductionMemoryState(TypedDict):
    """Complete production memory state"""
    # Messages
    messages: Annotated[Sequence[BaseMessage], add]
    
    # Memory retrieval
    retrieved_memories: List[Memory]
    memory_query: str
    
    # User context
    user_id: str
    session_id: str
    
    # Memory management
    buffer_size: int
    compression_threshold: int
    conversation_summary: str
    
    # Metadata
    turn_count: Annotated[int, add]
    total_memories_stored: Annotated[int, add]

# ==================== MEMORY MANAGER ====================

class MemoryManager:
    """Centralized memory management system"""
    
    def __init__(self, storage_path: str = "./production_memory"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        self.embeddings = OllamaEmbeddings(model="llama3.2")
        
        # Initialize vector stores for different memory types
        self.stores = {}
        for memory_type in ["semantic", "episodic", "procedural"]:
            store_path = self.storage_path / memory_type
            try:
                self.stores[memory_type] = FAISS.load_local(
                    str(store_path),
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info(f"Loaded {memory_type} memory store")
            except:
                # Create new store
                init_doc = [Document(
                    page_content=f"{memory_type} memory initialized",
                    metadata={
                        "timestamp": datetime.now().isoformat(),
                        "memory_type": memory_type
                    }
                )]
                self.stores[memory_type] = FAISS.from_documents(init_doc, self.embeddings)
                self.stores[memory_type].save_local(str(store_path))
                logger.info(f"Created {memory_type} memory store")
    
    def store_memory(self, memory: Memory) -> None:
        """Store memory in appropriate vector store"""
        try:
            store = self.stores[memory.memory_type]
            doc = memory.to_document()
            store.add_documents([doc])
            
            # Save to disk
            store_path = self.storage_path / memory.memory_type
            store.save_local(str(store_path))
            
            logger.info(f"Stored {memory.memory_type} memory for user {memory.user_id}")
        
        except Exception as e:
            logger.error(f"Error storing memory: {e}")
    
    def retrieve_memories(
        self,
        query: str,
        user_id: str,
        memory_types: List[str] = None,
        k: int = 5,
        importance_threshold: float = 0.3
    ) -> List[Memory]:
        """Retrieve relevant memories"""
        
        if memory_types is None:
            memory_types = ["semantic", "episodic", "procedural"]
        
        all_memories = []
        
        for memory_type in memory_types:
            if memory_type not in self.stores:
                continue
            
            try:
                store = self.stores[memory_type]
                
                # Search with filters
                docs = store.similarity_search(
                    query,
                    k=k,
                    filter={"user_id": user_id}
                )
                
                # Convert to Memory objects
                memories = [Memory.from_document(doc) for doc in docs]
                
                # Filter by importance
                memories = [m for m in memories if m.importance >= importance_threshold]
                
                all_memories.extend(memories)
            
            except Exception as e:
                logger.error(f"Error retrieving {memory_type} memories: {e}")
        
        # Sort by importance and recency
        now = datetime.now()
        
        def score_memory(mem: Memory) -> float:
            """Calculate memory relevance score"""
            try:
                timestamp = datetime.fromisoformat(mem.timestamp)
                days_old = (now - timestamp).days
                recency_factor = 1.0 / (1.0 + days_old * 0.1)
                
                return mem.importance * 0.6 + recency_factor * 0.4
            except:
                return mem.importance
        
        all_memories.sort(key=score_memory, reverse=True)
        
        return all_memories[:k]
    
    def cleanup_old_memories(
        self,
        user_id: str,
        days_old: int = 90,
        importance_threshold: float = 0.5
    ):
        """Remove old, unimportant memories"""
        # This is a simplified version
        # In production, you'd implement proper cleanup logic
        logger.info(f"Cleanup old memories for user {user_id}")
        pass

# ==================== MEMORY ANALYSIS ====================

class MemoryAnalyzer:
    """Analyze conversations and extract memories"""
    
    def __init__(self):
        self.llm = ChatOllama(model="llama3.2", temperature=0.3)
    
    def analyze_conversation(
        self,
        user_message: str,
        ai_message: str
    ) -> List[Memory]:
        """Analyze conversation and extract memories"""
        
        analysis_prompt = ChatPromptTemplate.from_messages([
            ("human", """Analyze this conversation and extract important information to remember.

User: {user_message}
Assistant: {ai_message}

Extract:
1. Facts (semantic memory) - general knowledge, definitions
2. Personal info (episodic memory) - user's experiences, preferences
3. Procedures (procedural memory) - how to do things, workflows

For each item, provide:
- content: what to remember
- type: semantic, episodic, or procedural
- importance: 0.0 to 1.0

Respond in JSON format:
{{"memories": [{{"content": "...", "type": "...", "importance": 0.8}}]}}

Analysis:""")
        ])
        
        try:
            chain = analysis_prompt | self.llm
            response = chain.invoke({
                "user_message": user_message,
                "ai_message": ai_message
            })
            
            # Parse JSON
            response_text = response.content.strip()
            
            # Remove markdown code blocks if present
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            data = json.loads(response_text)
            
            memories = []
            for item in data.get("memories", []):
                memory = Memory(
                    content=item["content"],
                    memory_type=item["type"],
                    importance=item["importance"],
                    timestamp=datetime.now().isoformat(),
                    user_id="",  # Will be set by caller
                    metadata={}
                )
                memories.append(memory)
            
            logger.info(f"Extracted {len(memories)} memories")
            return memories
        
        except Exception as e:
            logger.error(f"Memory analysis error: {e}")
            return []

# ==================== CONVERSATION SUMMARIZER ====================

class ConversationSummarizer:
    """Summarize long conversations"""
    
    def __init__(self):
        self.llm = ChatOllama(model="llama3.2", temperature=0.3)
    
    def summarize(self, messages: Sequence[BaseMessage]) -> str:
        """Create conversation summary"""
        
        if len(messages) < 5:
            return ""
        
        summary_prompt = ChatPromptTemplate.from_messages([
            ("human", """Summarize this conversation, preserving key facts and context:

{conversation}

Provide a concise 2-3 sentence summary.

Summary:""")
        ])
        
        try:
            # Format conversation
            conv_text = "\n".join([
                f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
                for m in messages
            ])
            
            chain = summary_prompt | self.llm
            response = chain.invoke({"conversation": conv_text})
            
            return response.content.strip()
        
        except Exception as e:
            logger.error(f"Summarization error: {e}")
            return ""

# ==================== PRODUCTION AGENT NODES ====================

# Initialize components
memory_manager = MemoryManager()
memory_analyzer = MemoryAnalyzer()
conversation_summarizer = ConversationSummarizer()
llm = ChatOllama(model="llama3.2", temperature=0.7)

def retrieve_relevant_memories(state: ProductionMemoryState) -> dict:
    """Retrieve relevant memories for current query"""
    
    try:
        current_message = state["messages"][-1].content
        user_id = state["user_id"]
        
        logger.info(f"Retrieving memories for user {user_id}")
        
        # Retrieve memories
        memories = memory_manager.retrieve_memories(
            query=current_message,
            user_id=user_id,
            k=5
        )
        
        return {
            "retrieved_memories": memories,
            "memory_query": current_message
        }
    
    except Exception as e:
        logger.error(f"Memory retrieval error: {e}")
        return {
            "retrieved_memories": [],
            "memory_query": ""
        }

def manage_conversation_buffer(state: ProductionMemoryState) -> dict:
    """Manage conversation buffer with summarization"""
    
    try:
        messages = state["messages"]
        buffer_size = state.get("buffer_size", 10)
        threshold = state.get("compression_threshold", 20)
        
        # Check if summarization needed
        if len(messages) >= threshold:
            logger.info("Conversation threshold reached - creating summary")
            
            # Summarize old messages
            old_messages = messages[:-buffer_size]
            summary = conversation_summarizer.summarize(old_messages)
            
            # Keep only recent messages
            recent_messages = messages[-buffer_size:]
            
            return {
                "conversation_summary": summary,
                "messages": recent_messages
            }
        
        return {}
    
    except Exception as e:
        logger.error(f"Buffer management error: {e}")
        return {}

def generate_response(state: ProductionMemoryState) -> dict:
    """Generate response using memories"""
    
    try:
        # Format memories
        memories_text = []
        
        if state.get("conversation_summary"):
            memories_text.append(f"Previous conversation: {state['conversation_summary']}")
        
        for memory in state.get("retrieved_memories", []):
            mem_type = memory.memory_type
            importance = memory.importance
            content = memory.content
            memories_text.append(f"[{mem_type}, importance: {importance:.1f}] {content}")
        
        memories_str = "\n".join(memories_text) if memories_text else "No prior context."
        
        # Create prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant with memory of past conversations.

Relevant memories and context:
{memories}

Use this context naturally to provide informed, personalized responses."""),
            ("placeholder", "{messages}")
        ])
        
        chain = prompt | llm
        response = chain.invoke({
            "memories": memories_str,
            "messages": state["messages"]
        })
        
        return {
            "messages": [response],
            "turn_count": 1
        }
    
    except Exception as e:
        logger.error(f"Response generation error: {e}")
        return {
            "messages": [AIMessage(content=f"Error: {str(e)}")]
        }

def extract_and_store_memories(state: ProductionMemoryState) -> dict:
    """Extract important information and store as memories"""
    
    try:
        # Get last exchange
        recent_messages = state["messages"][-2:]
        
        user_msg = None
        ai_msg = None
        
        for msg in recent_messages:
            if isinstance(msg, HumanMessage):
                user_msg = msg.content
            elif isinstance(msg, AIMessage):
                ai_msg = msg.content
        
        if not (user_msg and ai_msg):
            return {}
        
        # Analyze and extract memories
        memories = memory_analyzer.analyze_conversation(user_msg, ai_msg)
        
        # Set user_id and store
        user_id = state["user_id"]
        stored_count = 0
        
        for memory in memories:
            memory.user_id = user_id
            memory_manager.store_memory(memory)
            stored_count += 1
        
        logger.info(f"Stored {stored_count} new memories")
        
        return {
            "total_memories_stored": stored_count
        }
    
    except Exception as e:
        logger.error(f"Memory storage error: {e}")
        return {}

# ==================== BUILD PRODUCTION GRAPH ====================

workflow = StateGraph(ProductionMemoryState)

# Add nodes
workflow.add_node("retrieve_memories", retrieve_relevant_memories)
workflow.add_node("manage_buffer", manage_conversation_buffer)
workflow.add_node("generate", generate_response)
workflow.add_node("store_memories", extract_and_store_memories)

# Define flow
workflow.set_entry_point("retrieve_memories")
workflow.add_edge("retrieve_memories", "manage_buffer")
workflow.add_edge("manage_buffer", "generate")
workflow.add_edge("generate", "store_memories")
workflow.add_edge("store_memories", END)

# Compile with checkpointing
checkpointer = SqliteSaver.from_conn_string("./production_memory_agent.db")
production_memory_agent = workflow.compile(checkpointer=checkpointer)

# ==================== PRODUCTION API ====================

def chat_with_production_memory(
    message: str,
    user_id: str,
    session_id: str = None,
    buffer_size: int = 10,
    compression_threshold: int = 20
) -> dict:
    """
    Production-ready chat with comprehensive memory.
    
    Args:
        message: User message
        user_id: User identifier
        session_id: Session identifier (optional)
        buffer_size: Max messages to keep in buffer
        compression_threshold: When to compress conversation
    
    Returns:
        dict with response and metadata
    """
    
    if session_id is None:
        session_id = f"{user_id}-{datetime.now().strftime('%Y%m%d')}"
    
    config = {
        "configurable": {
            "thread_id": f"prod-mem-{session_id}"
        }
    }
    
    initial_state = {
        "messages": [HumanMessage(content=message)],
        "retrieved_memories": [],
        "memory_query": "",
        "user_id": user_id,
        "session_id": session_id,
        "buffer_size": buffer_size,
        "compression_threshold": compression_threshold,
        "conversation_summary": "",
        "turn_count": 0,
        "total_memories_stored": 0
    }
    
    try:
        result = production_memory_agent.invoke(initial_state, config=config)
        
        final_message = result["messages"][-1]
        
        return {
            "success": True,
            "response": final_message.content,
            "metadata": {
                "user_id": user_id,
                "session_id": session_id,
                "turn_count": result["turn_count"],
                "memories_retrieved": len(result["retrieved_memories"]),
                "memories_stored": result["total_memories_stored"],
                "has_summary": bool(result.get("conversation_summary")),
                "buffer_size": len(result["messages"])
            }
        }
    
    except Exception as e:
        logger.error(f"Chat failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }

# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    user_id = "alice-123"
    
    print("\n" + "="*60)
    print("PRODUCTION MEMORY SYSTEM DEMO")
    print("="*60)
    
    # Conversation 1: Build initial memory
    print("\n--- Building Initial Memory ---")
    conv1 = [
        "Hi, my name is Alice and I'm a data scientist",
        "I work with Python and specialize in machine learning",
        "I'm currently learning about LangGraph and agent systems",
        "My favorite ML framework is PyTorch"
    ]
    
    for msg in conv1:
        print(f"\nUser: {msg}")
        result = chat_with_production_memory(msg, user_id)
        if result["success"]:
            print(f"Agent: {result['response'][:150]}...")
            print(f"Stats: {result['metadata']['memories_stored']} memories stored")
    
    # Conversation 2: Test memory recall
    print("\n" + "="*60)
    print("--- Testing Memory Recall ---")
    
    conv2 = [
        "What's my name?",
        "What do I do for work?",
        "What ML framework do I prefer?",
        "What am I currently learning about?"
    ]
    
    for msg in conv2:
        print(f"\nUser: {msg}")
        result = chat_with_production_memory(msg, user_id)
        if result["success"]:
            print(f"Agent: {result['response']}")
            meta = result['metadata']
            print(f"Retrieved: {meta['memories_retrieved']} memories")
    
    # Conversation 3: Long conversation with compression
    print("\n" + "="*60)
    print("--- Testing Conversation Compression ---")
    
    for i in range(15):
        msg = f"Message {i+1}: Tell me something interesting about AI"
        result = chat_with_production_memory(
            msg,
            user_id,
            buffer_size=5,
            compression_threshold=10
        )
        
        if i % 5 == 0:
            meta = result['metadata']
            print(f"\nTurn {i+1}:")
            print(f"  Buffer size: {meta['buffer_size']}")
            print(f"  Has summary: {meta['has_summary']}")
```

---

## 📋 Part 8: Best Practices for Production Memory

### 1. Memory Storage Best Practices

```python
# ✅ GOOD: Structured memory with metadata
@dataclass
class StructuredMemory:
    content: str
    memory_type: str
    importance: float
    timestamp: str
    user_id: str
    tags: List[str]
    source: str  # "conversation", "upload", "external"

# ❌ BAD: Just storing raw text
memories = ["user likes pizza", "user works at Google"]
```

### 2. Retrieval Optimization

```python
# ✅ GOOD: Multi-factor retrieval scoring
def score_memory(memory: Memory, query: str) -> float:
    """Score memory relevance"""
    
    # Semantic similarity (from vector search)
    semantic_score = memory.similarity_score
    
    # Recency factor
    days_old = (datetime.now() - datetime.fromisoformat(memory.timestamp)).days
    recency_score = 1.0 / (1.0 + days_old * 0.1)
    
    # Importance factor
    importance_score = memory.importance
    
    # Combine
    final_score = (
        semantic_score * 0.5 +
        recency_score * 0.3 +
        importance_score * 0.2
    )
    
    return final_score

# ❌ BAD: Only using similarity
def bad_scoring(memory: Memory) -> float:
    return memory.similarity_score
```

### 3. Memory Cleanup and Maintenance

```python
# ✅ GOOD: Periodic memory cleanup
class MemoryMaintenance:
    """Periodic memory maintenance tasks"""
    
    def cleanup_old_memories(
        self,
        user_id: str,
        days_threshold: int = 90,
        importance_threshold: float = 0.3
    ):
        """Remove old, low-importance memories"""
        
        cutoff_date = datetime.now() - timedelta(days=days_threshold)
        
        # Only delete if:
        # 1. Old enough
        # 2. Low importance
        # 3. Not recently accessed
        
        logger.info(f"Cleanup: Removing memories older than {days_threshold} days with importance < {importance_threshold}")
    
    def consolidate_similar_memories(self, user_id: str):
        """Merge similar memories to reduce redundancy"""
        # Implementation would use clustering
        pass
    
    def rebuild_indexes(self):
        """Rebuild vector store indexes for performance"""
        logger.info("Rebuilding memory indexes...")
        # FAISS index optimization
        pass

# ❌ BAD: Never cleaning up, letting memory grow unbounded
```

### 4. Privacy and Security

```python
# ✅ GOOD: User data isolation
class SecureMemoryManager:
    """Memory manager with privacy controls"""
    
    def store_memory(self, memory: Memory, user_id: str):
        """Store with user isolation"""
        
        # Validate user owns this memory
        if memory.user_id != user_id:
            raise PermissionError("Cannot store memory for different user")
        
        # Hash sensitive info
        if memory.contains_sensitive_data:
            memory.content = self.hash_sensitive_fields(memory.content)
        
        # Store with encryption at rest
        self._store_encrypted(memory)
    
    def retrieve_memories(self, query: str, user_id: str):
        """Retrieve with strict user filtering"""
        
        # ALWAYS filter by user_id
        results = self.vector_store.similarity_search(
            query,
            filter={"user_id": user_id}  # Critical!
        )
        
        return results
    
    def delete_user_data(self, user_id: str):
        """GDPR compliance - delete all user data"""
        logger.info(f"Deleting all data for user {user_id}")
        # Implementation...

# ❌ BAD: Mixing user data without isolation
def bad_retrieve(query: str):
    # No user filtering - retrieves from all users!
    return vector_store.similarity_search(query)
```

### 5. Performance Optimization

```python
# ✅ GOOD: Caching and batching
class OptimizedMemoryManager:
    """Optimized memory operations"""
    
    def __init__(self):
        self.cache = {}  # Simple cache
        self.batch_buffer = []
        self.batch_size = 10
    
    def retrieve_with_cache(self, query: str, user_id: str) -> List[Memory]:
        """Retrieve with caching"""
        
        cache_key = f"{user_id}:{hash(query)}"
        
        # Check cache
        if cache_key in self.cache:
            cache_time, cached_result = self.cache[cache_key]
            
            # Cache valid for 5 minutes
            if (datetime.now() - cache_time).seconds < 300:
                logger.info("Cache hit")
                return cached_result
        
        # Cache miss - retrieve
        result = self._retrieve_from_store(query, user_id)
        
        # Update cache
        self.cache[cache_key] = (datetime.now(), result)
        
        return result
    
    def batch_store(self, memory: Memory):
        """Batch multiple stores for efficiency"""
        
        self.batch_buffer.append(memory)
        
        # Flush when batch is full
        if len(self.batch_buffer) >= self.batch_size:
            self.flush_batch()
    
    def flush_batch(self):
        """Flush batched stores"""
        if not self.batch_buffer:
            return
        
        logger.info(f"Flushing {len(self.batch_buffer)} memories")
        
        # Convert to documents
        docs = [m.to_document() for m in self.batch_buffer]
        
        # Single batch operation
        self.vector_store.add_documents(docs)
        
        # Clear buffer
        self.batch_buffer = []

# ❌ BAD: Every operation hits the database
def bad_storage(memory: Memory):
    # Immediate write, no batching
    vector_store.add_documents([memory.to_document()])
    vector_store.save_local("./store")  # Slow!
```

### 6. Monitoring and Observability

```python
# ✅ GOOD: Comprehensive logging and metrics
from dataclasses import dataclass, field
from typing import Dict
import time

@dataclass
class MemoryMetrics:
    """Track memory system metrics"""
    
    total_retrievals: int = 0
    total_stores: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    avg_retrieval_time: float = 0.0
    avg_store_time: float = 0.0
    memory_count_by_type: Dict[str, int] = field(default_factory=dict)
    
    def log_retrieval(self, duration: float, cache_hit: bool):
        """Log a retrieval operation"""
        self.total_retrievals += 1
        
        if cache_hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
        
        # Update average
        self.avg_retrieval_time = (
            (self.avg_retrieval_time * (self.total_retrievals - 1) + duration) /
            self.total_retrievals
        )
    
    def get_report(self) -> str:
        """Generate metrics report"""
        cache_hit_rate = (
            self.cache_hits / self.total_retrievals * 100
            if self.total_retrievals > 0 else 0
        )
        
        return f"""
Memory System Metrics:
- Total Retrievals: {self.total_retrievals}
- Total Stores: {self.total_stores}
- Cache Hit Rate: {cache_hit_rate:.1f}%
- Avg Retrieval Time: {self.avg_retrieval_time:.3f}s
- Avg Store Time: {self.avg_store_time:.3f}s
- Memory Counts: {self.memory_count_by_type}
"""

class MonitoredMemoryManager:
    """Memory manager with observability"""
    
    def __init__(self):
        self.metrics = MemoryMetrics()
        # ... other initialization
    
    def retrieve_memories(self, query: str, user_id: str) -> List[Memory]:
        """Retrieve with timing"""
        start = time.time()
        
        try:
            # Try cache first
            cache_key = f"{user_id}:{hash(query)}"
            if cache_key in self.cache:
                duration = time.time() - start
                self.metrics.log_retrieval(duration, cache_hit=True)
                return self.cache[cache_key]
            
            # Cache miss - retrieve from store
            result = self._retrieve_from_store(query, user_id)
            
            duration = time.time() - start
            self.metrics.log_retrieval(duration, cache_hit=False)
            
            logger.info(f"Retrieved {len(result)} memories in {duration:.3f}s")
            
            return result
        
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            raise
    
    def get_metrics(self) -> str:
        """Get current metrics"""
        return self.metrics.get_report()

# ❌ BAD: No monitoring or metrics
def unmonitored_retrieve(query: str):
    return vector_store.search(query)  # No timing, no logging
```

---

## 🧠 Key Concepts to Remember

1. **Short-term memory** = conversation buffer (limited capacity)
2. **Long-term memory** = vector stores (unlimited capacity)
3. **Semantic memory** = facts and knowledge
4. **Episodic memory** = events and experiences
5. **Always compress** old conversations to save context
6. **Multi-factor retrieval** (similarity + recency + importance)
7. **User isolation** is critical for privacy
8. **Batch operations** for performance
9. **Cache frequently accessed** memories
10. **Monitor and measure** everything
11. **Clean up periodically** to manage growth
12. **Structure memory** with proper metadata

---

## 🎯 Production Checklist

### Before Deploying Memory System:

- ✅ User data is isolated (filter by user_id)
- ✅ Sensitive data is handled securely
- ✅ Memory cleanup strategy in place
- ✅ Caching implemented for hot paths
- ✅ Batch operations where possible
- ✅ Monitoring and logging configured
- ✅ Error handling comprehensive
- ✅ Backup and recovery plan
- ✅ GDPR compliance (user data deletion)
- ✅ Performance tested under load
- ✅ Memory limits configured
- ✅ Vector store indexes optimized

---

## 🚀 What's Next?

In **Chapter 10**, we'll explore:
- **Agentic RAG Systems**
- Traditional RAG vs Agentic RAG
- Query analysis and routing
- Self-RAG with reflection
- Corrective RAG (CRAG)
- Adaptive RAG strategies
- Production RAG patterns

---

## ✅ Chapter 9 Complete!

**You now understand:**
- ✅ Memory types (short-term, long-term, semantic, episodic)
- ✅ Conversation buffers and compression
- ✅ Vector stores for long-term memory
- ✅ Multi-strategy retrieval (similarity, recency, hybrid)
- ✅ Production memory architecture
- ✅ Memory analysis and extraction
- ✅ Best practices (privacy, performance, monitoring)
- ✅ Complete production-ready memory system

**Ready for Chapter 10?** Just say "Continue to Chapter 10" or ask any questions!