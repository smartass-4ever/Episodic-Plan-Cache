# Episodic Plan Cache - Project Summary

## Origin Story

This implementation is based on my RFC posted to Aden Hive:  
**[Issue #3749: Architectural Implementation of an Episodic "Plan Cache" for Deterministic Orchestration](https://github.com/adenhq/hive/issues/3749)**

### The Problem I Identified

While studying Aden Hive's agent orchestrator, I noticed a critical inefficiency: the Queen Bee (LLM orchestrator) regenerates execution plans from scratch for EVERY user intent, even recurring tasks like "monthly budgeting."

This creates three production bottlenecks:
1. **15-30 second latency** per planning session
2. **2k-5k tokens burned** per plan generation
3. **Non-deterministic behavior** (same intent → different plans)

### My Proposed Solution

Implement an "Episodic Memory Layer" that learns from successful executions:
- Cache successful plans with vector indexing
- Reuse plans via similarity search (>0.95 threshold)
- Hydrate cached plans with new variables (no LLM call)
- Trigger evolution only when cached plans fail

## Implementation

I built a generic version that works with ANY agent orchestrator (Aden Hive, LangGraph, CrewAI, etc.).

### Key Components

**1. Semantic Plan Storage**
```python
cache.store_plan(
    intent="Generate monthly budget report",
    plan_data={...},  # DAG/workflow
    variables={...}   # Template vars
)
```

**2. Similarity Search**
```python
cached = cache.search("Create budget report")
# Returns match if similarity > 0.95
```

**3. Plan Hydration**
```python
plan = cache.hydrate(cached_plan, {"month": "March"})
# Instant, deterministic, no LLM call
```

**4. Evolution Trigger**
```python
if plan_fails:
    cache.mark_failure(plan_id)
    # Auto-evicts if failure rate > 50%
```


## Architecture Highlights

### What Makes This Production-Ready

1. **Fallback Safety**: Cache miss → LLM planning (graceful degradation)
2. **Self-Healing**: Failed plans auto-evict, trigger re-planning
3. **Observability**: Built-in metrics (hit rate, latency saved, tokens saved)
4. **Modularity**: Drop-in integration for any orchestrator
5. **Dependency-Free**: Core implementation requires only numpy

### Production Deployment Path

**Phase 1: Development** (current)
- In-memory storage
- Simple TF-IDF embeddings
- Single-process only

**Phase 2: Production**
- Redis/PostgreSQL storage
- OpenAI/Cohere embeddings
- Multi-process safe

**Phase 3: Scale**
- Distributed vector DB (Pinecone/Weaviate)
- Plan versioning & migration
- Multi-tenant isolation

## Integration Examples

### For Aden Hive

```python
class HiveWithCache(HiveOrchestrator):
    def __init__(self):
        super().__init__()
        self.cache = PlanCache()
    
    def orchestrate(self, intent):
        # Try cache first
        cached = self.cache.search(intent)
        if cached:
            plan, _ = cached
            return self.execute(self.cache.hydrate(plan, vars))
        
        # Cache miss - Queen Bee plans
        dag = self.queen_bee.plan(intent)
        result = self.execute(dag)
        
        if result.success:
            self.cache.store_plan(intent, dag, vars)
        
        return result
```

### For LangGraph

```python
cache = PlanCache()

def workflow(query):
    cached = cache.search(query)
    if cached:
        graph = cache.hydrate(cached[0], {"query": query})
        return execute_graph(graph)
    
    graph = build_graph(query)
    result = execute_graph(graph)
    cache.store_plan(query, graph, {"query": query})
    return result
```

## Why This Matters

### For Agent Orchestrators

Every major orchestrator has this problem:
- Aden Hive: LLM plans every DAG
- LangGraph: Builds graphs from scratch
- CrewAI: Generates task lists per request
- AutoGPT: Plans action sequences repeatedly

This pattern solves it universally.

### For Production Systems

Recurring business processes are common:
- Monthly reports
- Weekly summaries
- Daily standup notes
- Quarterly reviews

These should NOT require 20 seconds of LLM inference EVERY time.

### For Cost Optimization

At scale:
- 1,000 recurring tasks/month
- 20s latency each = 5.5 hours wasted
- 2M tokens = $20/month wasted
- **Solution: $0 recurring cost after first execution**

## Technical Decisions

### Why Vector Similarity?

Plans have semantic equivalence:
- "Generate monthly budget" ≈ "Create budget report"
- "Weekly sales summary" ≈ "Summarize weekly sales"

Keyword matching would miss these. Vector search captures semantic similarity.

### Why 0.95 Threshold?

Tested with various thresholds:
- 0.85: Too many false positives (wrong plans reused)
- 0.95: Optimal balance (catches similar, avoids wrong)
- 0.99: Too strict (cache rarely hits)

### Why Evolution Trigger?

Plans can become stale:
- API endpoints change
- Data schemas evolve
- Business logic updates

Automatic eviction + re-planning keeps cache fresh.


## Files

- `plan_cache.py` - Core implementation (400 lines)
- `plan_cache_demo.py` - Working demonstrations
- `plan_cache_README.md` - Full documentation
- `requirements_plan_cache.txt` - Dependencies


## Why I Built This

I'm exploring agent orchestration architecture and wanted to contribute something practical to the community. This solves a real problem I observed in production systems.

If you're building agent orchestrators and facing the "re-planning tax," this might help.

---

**Status**: Working proof-of-concept  
**License**: MIT  
**Origin**: RFC for Aden Hive improvement  
**Contact**: mahikajadhav22@gmail.com
