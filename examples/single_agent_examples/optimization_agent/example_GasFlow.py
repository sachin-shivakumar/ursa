from pathlib import Path
from langchain_openai import ChatOpenAI
from dataclasses import asdict, is_dataclass
from langchain_core.runnables import RunnableConfig

from ursa.agents.optimization_agent import OptimizationAgent

problem_string = """
Solve this optimization problem (steady-state gas flow / gas network optimization): 

Physical system:
A steady-state natural gas transmission network with junctions (nodes), pipes, compressors, receipts (supplies), and deliveries (demands).
Gas flows satisfy mass balance at junctions and nonlinear pressure–flow relations on pipes, plus compressor pressure ratio limits.

Data source (read at runtime):
Data file name is `gaslib11.m' and is in your workspace. Load it programmatically inside the code.
Read network data from a file in a “GasLib-style” structure:
- Global parameters: gas_specific_gravity, specific_heat_capacity_ratio, temperature, compressibility_factor, base_pressure, base_flow, base_length, etc.
- Tables: mgc.junction, mgc.pipe, mgc.compressor, mgc.receipt, mgc.delivery
Assume the file path is provided at runtime, and your code should parse it and build the optimization model.

Units:
Use SI units consistent with the data. If mgc.is_per_unit == 1, interpret pressures/flows as per-unit on the given base values.

Decision variables:
For each junction j:
- p_j ≥ 0 : junction pressure (Pa, or p.u. if per-unit)

For each pipe ℓ connecting junction i -> k:
- q_ℓ : mass flow through pipe (kg/s, or p.u.)
  Sign convention: q_ℓ > 0 means flow from fr_junction to to_junction as listed in mgc.pipe.

For each compressor c connecting junction i -> k:
- q_c : mass flow through compressor (kg/s, or p.u.)
- r_c : compressor pressure ratio (dimensionless), r_c = p_out / p_in (use pressures at its endpoints)

For each receipt s (supply) at junction j:
- inj_s : injection (kg/s, or p.u.)

For each delivery d (demand) at junction j:
- wdr_d : withdrawal (kg/s, or p.u.)

Objective (minimize total operating cost minus served-value; choose one consistent economic model):
Primary objective:
min  sum_{s in receipts} ( offer_price_s * inj_s )
   + sum_{c in compressors} ( operating_cost_c * compressor_power_c )
   - sum_{d in deliveries} ( bid_price_d * wdr_d )

If you prefer a pure feasibility problem, set all prices/costs to 0 and only seek feasibility.

Constraints:

1) Junction pressure bounds:
For each junction j:
  p_min_j <= p_j <= p_max_j
Use the p_min and p_max columns from mgc.junction.
Only enforce for junctions with status=1.

2) Receipt (supply) bounds:
For each receipt s:
  injection_min_s <= inj_s <= injection_max_s
Only enforce if receipt status=1.
(If is_dispatchable=0, fix inj_s = injection_nominal_s.)

3) Delivery (demand) bounds:
For each delivery d:
  withdrawal_min_d <= wdr_d <= withdrawal_max_d
Only enforce if delivery status=1.
(If is_dispatchable=0, fix wdr_d = withdrawal_nominal_d.)

4) Mass balance at each junction (steady state):
For each junction j:
  sum_{s at j} inj_s
  - sum_{d at j} wdr_d
  + sum_{pipes ℓ with to_junction=j} q_ℓ
  - sum_{pipes ℓ with fr_junction=j} q_ℓ
  + sum_{compressors c with to_junction=j} q_c
  - sum_{compressors c with fr_junction=j} q_c
  = 0

(Only include active components with status=1. If a junction status=0, treat it as removed.)

5) Pipe pressure–flow equations (nonlinear steady-state gas physics):
For each active pipe ℓ connecting i -> k with diameter D, length L, friction factor f:
Use a Weymouth-type equation relating pressure drop to flow magnitude:
  p_i^2 - p_k^2 = K_ℓ * q_ℓ * |q_ℓ|
where K_ℓ is a constant computed from the pipe data and global gas properties.

Implementation requirement:
- Compute K_ℓ using a standard steady-state isothermal gas flow model consistent with mgc units.
- If you do not have a trusted closed-form K_ℓ, you may instead use a scaled form:
    p_i^2 - p_k^2 = K_ℓ * q_ℓ * |q_ℓ|
  with K_ℓ derived from (f, L, D, gas constants) and clearly document it in the output.
- Respect pipe status=1.

6) Compressor constraints:
For each active compressor c from i -> k:
- Pressure ratio bounds:
    c_ratio_min_c <= r_c <= c_ratio_max_c
- Pressure relation:
    p_k = r_c * p_i
- Flow bounds:
    flow_min_c <= q_c <= flow_max_c
- Power limit:
    compressor_power_c <= power_max_c

Compressor power model (steady-state):
Use a standard isentropic compression approximation:
  compressor_power_c = alpha * q_c * ( (r_c^((gamma-1)/gamma) - 1) )
where gamma = specific_heat_capacity_ratio, and alpha depends on temperature, gas properties, and unit conversions.
If needed, use a simplified convex/linear surrogate for power for numerical stability, but keep r_c within bounds and enforce p_out = r_c p_in.

7) Component bounds:
Optionally enforce:
- Pipe pressure bounds p_min/p_max columns if meaningful (often wide in GasLib); otherwise rely on junction bounds.
- Enforce nonnegativity of pressures and any required minimum inlet/outlet pressures from compressor data if provided.

Initialization (important for nonconvex solve):
- Start junction pressures at p_nominal from mgc.junction (or midpoint of [p_min,p_max]).
- Initialize all flows q_ℓ and q_c to 0.
- Initialize injections/withdrawals to their nominal values if provided, else midpoint of bounds.

Notes:
- This is a nonlinear, nonconvex steady-state network optimization problem.
- Print progress during solving (iterations, objective, and max constraint violation) with flush=True.
- After solving, print one line containing exactly one of: OPTIMAL, FEASIBLE, INFEASIBLE, UNBOUNDED, ERROR.
- Also print:
  obj=...
  max_mass_balance_violation=...
  max_pipe_residual=...
  max_compressor_residual=...
  and a few representative pressures and flows (e.g., top 5 largest |q| and min/max p).

Runtime requirements:
- Your code must read the network file at runtime, build the model from the tables, and solve it.
- If the solver fails due to numerical issues, your program should print the solver’s reason/error to STDERR and still print the final required status line.
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
