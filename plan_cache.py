"""
Episodic Plan Cache for Agent Orchestrators

Reduces LLM inference overhead by caching and reusing successful execution plans.
Addresses the "re-planning tax" where orchestrators regenerate identical plans
for recurring tasks, wasting 15-30s and burning tokens unnecessarily.

Core Concept:
- Save successful plans to vector-indexed storage
- Reuse plans via similarity search (>0.95 threshold)
- Hydrate cached plans with new runtime variables
- Trigger evolution only when cached plans fail

Author: Based on RFC for Aden Hive orchestrator optimization
"""

import json
import hashlib
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
import numpy as np


@dataclass
class ExecutionPlan:
    """
    Represents a validated execution plan (DAG/workflow)
    """
    plan_id: str
    intent: str  # Original user intent/goal
    plan_data: Dict[str, Any]  # The actual DAG/node graph
    variables: Dict[str, Any]  # Runtime variables used
    success_count: int = 0
    failure_count: int = 0
    created_at: float = field(default_factory=time.time)
    last_used_at: float = field(default_factory=time.time)
    embedding: Optional[List[float]] = None  # Vector representation
    
    def to_dict(self) -> Dict:
        """Serialize for storage"""
        data = asdict(self)
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ExecutionPlan':
        """Deserialize from storage"""
        return cls(**data)


class SimpleEmbedder:
    """
    Lightweight text embedder using TF-IDF-like approach.
    
    In production, replace with:
    - OpenAI embeddings (text-embedding-ada-002)
    - Sentence transformers (all-MiniLM-L6-v2)
    - Cohere embeddings
    
    This implementation is dependency-free for portability.
    """
    
    def __init__(self, dim: int = 384):
        self.dim = dim
        self.vocab: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        
    def _tokenize(self, text: str) -> List[str]:
        """Basic tokenization"""
        return text.lower().split()
    
    def embed(self, text: str) -> List[float]:
        """
        Generate embedding vector for text.
        
        NOTE: This is a simplified implementation.
        Production systems should use proper embeddings (OpenAI, etc.)
        """
        tokens = self._tokenize(text)
        
        # Simple bag-of-words with hash-based projection
        vector = np.zeros(self.dim)
        
        for token in tokens:
            # Hash token to dimension
            hash_val = int(hashlib.md5(token.encode()).hexdigest(), 16)
            idx = hash_val % self.dim
            vector[idx] += 1.0
        
        # Normalize
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector.tolist()
    
    @staticmethod
    def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors"""
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        
        dot_product = np.dot(v1, v2)
        norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
        
        if norm_product == 0:
            return 0.0
        
        return float(dot_product / norm_product)


class PlanCache:
    """
    Episodic memory layer for execution plans.
    
    Workflow:
    1. On successful plan execution → store_plan()
    2. On new intent → search() for similar cached plan
    3. If match > threshold → hydrate() with new variables
    4. If hydrated plan fails → evolution trigger (re-plan via LLM)
    """
    
    def __init__(self, 
                 similarity_threshold: float = 0.95,
                 embedder: Optional[SimpleEmbedder] = None):
        """
        Args:
            similarity_threshold: Minimum similarity score for cache hit (0.0-1.0)
            embedder: Text embedding function (defaults to SimpleEmbedder)
        """
        self.similarity_threshold = similarity_threshold
        self.embedder = embedder or SimpleEmbedder()
        
        # In-memory storage (replace with Redis/PostgreSQL for production)
        self.plans: Dict[str, ExecutionPlan] = {}
        
        # Statistics
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_latency_saved = 0.0  # Seconds saved by cache hits
        
    def store_plan(self, 
                   intent: str,
                   plan_data: Dict[str, Any],
                   variables: Dict[str, Any]) -> str:
        """
        Store a successful execution plan in the cache.
        
        Args:
            intent: User's goal/intent string
            plan_data: The DAG/node graph that was executed
            variables: Runtime variables used in this execution
            
        Returns:
            plan_id: Unique identifier for cached plan
        """
        # Generate embedding for semantic search
        embedding = self.embedder.embed(intent)
        
        # Create unique plan ID
        plan_id = hashlib.sha256(
            f"{intent}:{json.dumps(plan_data, sort_keys=True)}".encode()
        ).hexdigest()[:16]
        
        # Check if plan already exists
        if plan_id in self.plans:
            # Update existing plan
            self.plans[plan_id].success_count += 1
            self.plans[plan_id].last_used_at = time.time()
        else:
            # Create new plan entry
            plan = ExecutionPlan(
                plan_id=plan_id,
                intent=intent,
                plan_data=plan_data,
                variables=variables,
                embedding=embedding,
                success_count=1
            )
            self.plans[plan_id] = plan
        
        return plan_id
    
    def search(self, intent: str) -> Optional[Tuple[ExecutionPlan, float]]:
        """
        Search for a cached plan matching the given intent.
        
        Args:
            intent: User's goal/intent string
            
        Returns:
            (plan, similarity_score) if match found, else None
        """
        if not self.plans:
            self.cache_misses += 1
            return None
        
        # Generate embedding for query
        query_embedding = self.embedder.embed(intent)
        
        # Find most similar plan
        best_match = None
        best_score = 0.0
        
        for plan in self.plans.values():
            if plan.embedding is None:
                continue
            
            score = self.embedder.cosine_similarity(
                query_embedding,
                plan.embedding
            )
            
            if score > best_score:
                best_score = score
                best_match = plan
        
        # Check if match exceeds threshold
        if best_match and best_score >= self.similarity_threshold:
            self.cache_hits += 1
            best_match.last_used_at = time.time()
            return (best_match, best_score)
        
        self.cache_misses += 1
        return None
    
    def hydrate(self, 
                plan: ExecutionPlan,
                new_variables: Dict[str, Any]) -> Dict[str, Any]:
        """
        Hydrate a cached plan with new runtime variables.
        
        This is the core "reflexive memory" operation: instead of
        regenerating the plan via LLM, we reuse the structure and
        inject new variables.
        
        Args:
            plan: Cached execution plan
            new_variables: New runtime context
            
        Returns:
            Hydrated plan ready for execution
        """
        # Deep copy the plan structure
        hydrated_plan = json.loads(json.dumps(plan.plan_data))
        
        # Replace variable placeholders
        # This is simplified - production systems need proper template engine
        plan_str = json.dumps(hydrated_plan)
        
        for key, value in new_variables.items():
            placeholder = f"${{{key}}}"
            plan_str = plan_str.replace(placeholder, str(value))
        
        return json.loads(plan_str)
    
    def mark_failure(self, plan_id: str):
        """
        Mark a cached plan as failed.
        
        If failure rate exceeds threshold, plan should be evicted
        and LLM should regenerate (evolution trigger).
        """
        if plan_id in self.plans:
            self.plans[plan_id].failure_count += 1
            
            # Auto-evict if failure rate > 50%
            plan = self.plans[plan_id]
            total = plan.success_count + plan.failure_count
            failure_rate = plan.failure_count / total
            
            if failure_rate > 0.5:
                del self.plans[plan_id]
                return True  # Signal evolution needed
        
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        total_queries = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_queries * 100) if total_queries > 0 else 0
        
        return {
            "total_plans": len(self.plans),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate_percent": hit_rate,
            "avg_latency_saved_seconds": 20.0,  # Assume 20s per LLM planning call
            "total_latency_saved_seconds": self.cache_hits * 20.0,
            "estimated_tokens_saved": self.cache_hits * 2000  # Assume 2k tokens per plan
        }


class OrchestratorWithCache:
    """
    Example orchestrator using episodic plan cache.
    
    This demonstrates the integration pattern for any agent orchestrator.
    """
    
    def __init__(self, cache: Optional[PlanCache] = None):
        self.cache = cache or PlanCache()
        self.llm_plan_calls = 0
        
    def execute_intent(self, intent: str, variables: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a user intent with plan caching.
        
        Flow:
        1. Search cache for similar intent
        2. If hit: hydrate and execute
        3. If miss: generate new plan via LLM
        4. Store successful plans in cache
        """
        start_time = time.time()
        
        # Step 1: Try cache first
        cached = self.cache.search(intent)
        
        if cached:
            plan, similarity = cached
            print(f"✓ Cache hit! Similarity: {similarity:.3f}")
            print(f"  Reusing plan: {plan.plan_id}")
            print(f"  Previous success rate: {plan.success_count}/{plan.success_count + plan.failure_count}")
            
            # Hydrate with new variables
            hydrated_plan = self.cache.hydrate(plan, variables)
            
            # Execute
            result = self._execute_plan(hydrated_plan, variables)
            
            # Update statistics
            if result["status"] == "success":
                plan.success_count += 1
            else:
                # Failed - trigger evolution
                needs_evolution = self.cache.mark_failure(plan.plan_id)
                if needs_evolution:
                    print("  ⚠ Plan evicted due to high failure rate - triggering evolution")
                    return self._llm_plan_and_execute(intent, variables)
            
            execution_time = time.time() - start_time
            result["cache_hit"] = True
            result["execution_time_seconds"] = execution_time
            return result
        
        # Step 2: Cache miss - generate new plan
        print(f"✗ Cache miss - generating new plan via LLM")
        return self._llm_plan_and_execute(intent, variables)
    
    def _llm_plan_and_execute(self, intent: str, variables: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate new plan via LLM and execute.
        
        In production, this calls your actual LLM orchestration logic.
        """
        import time
        
        self.llm_plan_calls += 1
        
        # Simulate LLM planning latency (15-30s in production)
        planning_start = time.time()
        
        # PLACEHOLDER: Replace with actual LLM planning call
        plan_data = self._simulate_llm_planning(intent, variables)
        
        planning_time = time.time() - planning_start
        print(f"  LLM planning took: {planning_time:.2f}s")
        
        # Execute plan
        result = self._execute_plan(plan_data, variables)
        
        # If successful, cache it
        if result["status"] == "success":
            plan_id = self.cache.store_plan(intent, plan_data, variables)
            print(f"  ✓ Plan cached: {plan_id}")
            result["plan_id"] = plan_id
        
        result["cache_hit"] = False
        result["planning_time_seconds"] = planning_time
        return result
    
    def _simulate_llm_planning(self, intent: str, variables: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate LLM generating an execution plan.
        
        In production, replace with actual LLM call to generate DAG.
        """
        # Simulate planning delay
        time.sleep(0.5)  # Real systems: 15-30 seconds
        
        # Generate a mock DAG based on intent
        if "budget" in intent.lower():
            return {
                "type": "dag",
                "nodes": [
                    {"id": "fetch_expenses", "action": "query_database", "params": {"table": "expenses", "month": "${month}"}},
                    {"id": "calculate_total", "action": "sum", "input": "fetch_expenses.output"},
                    {"id": "generate_report", "action": "create_report", "input": "calculate_total.output"}
                ],
                "edges": [
                    {"from": "fetch_expenses", "to": "calculate_total"},
                    {"from": "calculate_total", "to": "generate_report"}
                ]
            }
        else:
            return {
                "type": "dag",
                "nodes": [
                    {"id": "step_1", "action": "process", "params": variables}
                ]
            }
    
    def _execute_plan(self, plan: Dict[str, Any], variables: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a plan (DAG).
        
        In production, this is your actual graph executor.
        """
        # Simulate execution
        time.sleep(0.1)
        
        return {
            "status": "success",
            "output": f"Executed {len(plan.get('nodes', []))} nodes",
            "plan_type": plan.get("type")
        }
