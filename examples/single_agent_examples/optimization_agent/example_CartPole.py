from pathlib import Path
from langchain_openai import ChatOpenAI
from dataclasses import asdict, is_dataclass
from langchain_core.runnables import RunnableConfig

from ursa.agents.optimization_agent import OptimizationAgent

problem_string = """
Solve this optimization problem (cart-pole swing-up optimal control):

Physical system:
A cart of mass M moves horizontally. A pendulum of mass m and length l is attached to the cart.
We want to swing the pendulum from hanging down to upright while bringing the cart to rest near the origin.

Parameters:
M = 1.0 kg
m = 0.1 kg
l = 0.5 m
g = 9.81 m/s^2
Time horizon T = 4.0 s
Use N = 80 uniform time steps (dt = T/N)

State variables at each time step k = 0..N:
x_k      = cart position (m)
xdot_k   = cart velocity (m/s)
theta_k  = pendulum angle (rad), measured from upright (theta=0 is upright, theta=pi is hanging down)
thetadot_k = angular velocity (rad/s)

Control variable at each time step k = 0..N-1:
u_k = horizontal force applied to cart (N)

Dynamics (discrete-time via forward Euler):
Let s = sin(theta_k), c = cos(theta_k)
Denominator D = M + m - m*c^2

x_{k+1}      = x_k      + dt * xdot_k
theta_{k+1}  = theta_k  + dt * thetadot_k

xdot_{k+1} = xdot_k + dt * ( (u_k + m*s*(l*thetadot_k^2 + g*c)) / D )

thetadot_{k+1} = thetadot_k + dt * ( (-u_k*c - m*l*thetadot_k^2*c*s - (M+m)*g*s) / (l*D) )

Objective (minimize):
J = sum_{k=0}^{N-1} [  1e-3*u_k^2  +  1e-2*x_k^2  +  1e-3*xdot_k^2  +  5.0*(theta_k)^2  +  1e-2*thetadot_k^2 ] * dt
    + terminal cost:
      50.0*x_N^2 + 1.0*xdot_N^2 + 200.0*theta_N^2 + 5.0*thetadot_N^2

Constraints:
Initial conditions (k=0):
x_0 = 0
xdot_0 = 0
theta_0 = pi
thetadot_0 = 0

Terminal target (soft via terminal cost, but also enforce tight bounds):
|x_N| <= 0.05
|xdot_N| <= 0.10
|theta_N| <= 0.05
|thetadot_N| <= 0.20

Path bounds for all k:
|x_k| <= 2.4
|xdot_k| <= 10.0
|thetadot_k| <= 20.0
Control bounds for all k:
|u_k| <= 15.0

Angle handling:
Keep theta_k continuous (do not wrap). Use theta=0 as upright and theta=pi as downward.

Notes:
- This is a nonlinear, nonconvex optimal control problem. Use a suitable NLP approach.
- Print progress during solving (iterations, objective, constraint violation) and flush output.
- At the end print one line containing exactly one of: OPTIMAL, FEASIBLE, INFEASIBLE, UNBOUNDED, ERROR.
- Also print representative values (at least x_N, theta_N, and the final objective) as:
  xN=...
  thetaN=...
  obj=...

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
