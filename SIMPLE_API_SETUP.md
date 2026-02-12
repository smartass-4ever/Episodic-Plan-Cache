# SIMPLE API SETUP - Plan Cache

## Strategy: Free API → Traction → Monetize Later

---

## WHAT YOU NEED:

1. ✅ **LICENSE** - MIT with attribution (you get credit)
2. ✅ **api.py** - Simple FastAPI wrapper (no auth, no billing)
3. ✅ **Deploy** - Free hosting (Railway/Render/Fly.io)
4. ✅ **Track usage** - Simple logs (see who uses it)

**That's it. No complicated stuff until you have users.**

---

## FILE 1: LICENSE

```
MIT License with Attribution

Copyright (c) 2026 [Your Name]

Permission is granted to use this software freely, with one requirement:

You must credit [Your Name] in any public use, documentation, or 
marketing materials.

Link to: https://github.com/[your-repo]
```

---

## FILE 2: api.py (Complete)

```python
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import uvicorn

# Import your plan cache
from plan_cache import PlanCache

app = FastAPI(title="Plan Cache API - Free Beta")

# Allow anyone to use it
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

cache = PlanCache()

# Request models
class StorePlan(BaseModel):
    task: str
    steps: List[str]
    tags: List[str] = []

class QueryPlan(BaseModel):
    task: str

# API endpoints
@app.get("/")
def home():
    return {
        "message": "Plan Cache API - Free during beta",
        "docs": "/docs",
        "github": "[your-repo]"
    }

@app.post("/v1/store")
async def store(req: StorePlan):
    plan_id = cache.store(req.task, req.steps, req.tags)
    return {"plan_id": plan_id, "status": "stored"}

@app.post("/v1/query")
async def query(req: QueryPlan):
    result = cache.query(req.task)
    if result:
        return {"found": True, "plan": result}
    return {"found": False}

@app.get("/v1/stats")
def stats():
    return cache.get_stats()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## FILE 3: requirements.txt

```
fastapi==0.109.0
uvicorn[standard]==0.27.0
pydantic==2.5.0
numpy==1.24.0
```

---

## DEPLOY (Choose One):

### Railway.app (Easiest)
```bash
railway init
railway up
```
Done. You get a URL.

### Render.com (Good Free Tier)
1. Connect GitHub repo
2. Select "Web Service"
3. Done

### Fly.io (Best Free Tier)
```bash
flyctl launch
flyctl deploy
```

**All are FREE and take 5 minutes.**

---

## UPDATE README:

Add this section:

```markdown
## 🚀 Free API (Beta)

No signup needed. Just use it.

**Store a plan:**
```bash
curl -X POST https://your-api.com/v1/store \
  -H "Content-Type: application/json" \
  -d '{"task": "monthly report", "steps": ["fetch", "calc", "pdf"]}'
```

**Query for plan:**
```bash
curl -X POST https://your-api.com/v1/query \
  -H "Content-Type: application/json" \
  -d '{"task": "quarterly report"}'
```

**See stats:**
```bash
curl https://your-api.com/v1/stats
```

## License
MIT with Attribution - Please credit [Your Name]
```

---

## THAT'S IT!

**Total files to add:** 3 (LICENSE, api.py, requirements.txt)  
**Total time:** 1 hour  
**Total cost:** $0  

---

## WHEN TO ADD BILLING:

**DON'T until you see:**
- 1,000+ API calls
- 10+ regular users
- Someone asks "can I pay for more?"

**THEN:**
- Add API keys
- Add free/paid tiers
- Add Stripe

**Not before.**

---

**Want me to create these exact files for you right now?**
