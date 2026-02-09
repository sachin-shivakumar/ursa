from pathlib import Path
from langchain_openai import ChatOpenAI
from dataclasses import asdict, is_dataclass
from langchain_core.runnables import RunnableConfig

from ursa.agents.optimization_agent import OptimizationAgent

problem_string = """
Consider the 1D1V Vlasov-Ampere Equations for electrons in a neutralizing ion background.
Start with a two-stream beam configuration and add some perturbation. There is an external
field control trying to suppress the instability. Using optimization formulation to find 
the minimum energy control that stabilizes the two-stream instability.

Notes:
- Please print progress during solving.
- At the end print a line containing one of: OPTIMAL, FEASIBLE, INFEASIBLE, UNBOUNDED, ERROR.
"""

llm = ChatOpenAI(model="gpt-5.2", max_tokens=10000, timeout=None, max_retries=2)
agent = OptimizationAgent(llm=llm)

ws = Path.cwd() / "ursa_workspace"
ws.mkdir(parents=True, exist_ok=True)
cfg = RunnableConfig(configurable={"workspace": str(ws)})
result = agent.invoke({"user_input": problem_string}, config=cfg)


# Convert final state to JSON-serializable dict
final_state = asdict(result) if is_dataclass(result) else dict(result)

# Write to workspace (BaseAgent already has write_state)
out_path = agent.workspace / "final_state.json"
agent.write_state(str(out_path), final_state)

print("\n=== FINAL STATE ===")
print("status:", result["status"])
print("solution.status:", result["solution"].status)
print("solution.obj:", result["solution"].obj)
print("solution.x:", result["solution"].x)
print("history:", [h.__dict__ for h in result["solution"].history])

print("\n=== AUDIT TRAIL ===")
for e in final_state["problem"].data.get("_audit", []):
    print(f'{e["ts"]}  {e["event"]}  { {k:v for k,v in e.items() if k not in ("ts","event")} }')
