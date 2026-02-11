from pathlib import Path
from langchain_openai import ChatOpenAI
from dataclasses import asdict, is_dataclass
from langchain_core.runnables import RunnableConfig

from ursa.agents.optimization_agent import OptimizationAgent


problem_string = f"""
Solve this optimization problem (AC Optimal Power Flow, IEEE 14-bus case):

Physical system:
An electric power transmission network operating in steady-state AC conditions.
We must supply loads while respecting Kirchhoff’s laws, generator limits, voltage limits, and line thermal limits.

Data:
Use the standard IEEE 14-bus test case in MATPOWER format (case14.m) provided in your workspace. Make sure that 
your python code loads it directly from that file.

Decision variables:
For each bus i:
- Voltage magnitude V_i (p.u.)
- Voltage angle theta_i (rad)
For each generator g:
- Real power P_g (p.u. on baseMVA)
- Reactive power Q_g (p.u. on baseMVA)

Objective (minimize):
Total generation cost:
min sum_g (a_g * P_g^2 + b_g * P_g + c_g)
Use the quadratic cost coefficients from the case file.

Constraints:
1) AC power balance at each bus i:
   P_gen_i - P_load_i = P_inj_i(V, theta)
   Q_gen_i - Q_load_i = Q_inj_i(V, theta)
   where injections are computed from the network admittance matrix Ybus:
   P_inj_i = sum_j V_i V_j ( G_ij cos(theta_i-theta_j) + B_ij sin(theta_i-theta_j) )
   Q_inj_i = sum_j V_i V_j ( G_ij sin(theta_i-theta_j) - B_ij cos(theta_i-theta_j) )

2) Generator operating limits (from case file):
   P_g_min <= P_g <= P_g_max
   Q_g_min <= Q_g <= Q_g_max

3) Voltage magnitude bounds (from case file):
   V_i_min <= V_i <= V_i_max

4) Branch thermal limits (from case file):
   For each branch (i,j), enforce apparent power flow limit:
   S_ij(V, theta) <= S_ij_max
   Use the case’s rateA (MVA) converted to p.u. on baseMVA.
   (If rateA is 0 or missing, omit that branch limit.)

5) Reference bus angle:
   Fix theta_ref = 0 for the slack/reference bus defined in the case.

Initialization:
- Start from a “flat start”: V_i = 1.0 p.u. for all buses, theta_i = 0 for all buses,
  P_g initialized to meet total load proportionally among generators (within limits),
  Q_g initialized to 0 (within limits).

Notes:
- This is a nonconvex nonlinear program (NLP).
- Print progress during solving (iteration number, objective, primal/dual infeasibility).
- At the end print a line containing exactly one of:
  OPTIMAL, FEASIBLE, INFEASIBLE, UNBOUNDED, ERROR.
- Also print:
  x=...
  obj=...
  max_power_mismatch=...
  max_line_loading=...
  (and optionally key generator dispatch values)

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
