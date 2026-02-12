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
Use a stable time-stepping scheme.

State variables at each time step k = 0..N:
x(t)      = cart position (m)
xdot(t)   = cart velocity (m/s)
theta(t)  = pendulum angle (rad), measured from upright (theta=0 is upright, theta=pi is hanging down)
thetadot(t) = angular velocity (rad/s)
x_ddot(t) = cart acceleration (m/s^2)
theta_ddot(t) = angular acceleration (rad/s^2)

Control variable at each time step k = 0..N-1:
u_k = horizontal force applied to cart (N)

Dynamics (discrete-time via forward Euler):
Let s = sin(theta_k), c = cos(theta_k)
Denominator D = M + m - m*c^2

x_ddot(t) = ((u + m*s*(l*thetadot^2 + g*c)) / D )

theta_ddot(t) = ( (-u_k*c - m*l*thetadot^2*c*s - (M+m)*g*s) / (l*D) )

Objective (minimize):
Terminal state error + Overall control cost.

Constraints:
Initial conditions (k=0):
x(0) = 0
xdot(0) = 0
theta(0) = pi
thetadot(0) = 0

Terminal target (soft via terminal cost, but also enforce tight bounds):
|x(T)| <= 0.05
|xdot(T)| <= 0.10
|theta(T)| <= 0.05
|thetadot(T)| <= 0.20

Path bounds for all t:
|x(t)| <= 2.4
|xdot(t)| <= 10.0
|thetadot(t)| <= 20.0
Control bounds for all t:
|u(t)| <= 15.0

Angle handling:
Keep theta(t) continuous (do not wrap). Use theta=0 as upright and theta=pi as downward.

Notes:
- This is a nonlinear, nonconvex optimal control problem. Use a suitable NLP approach.
- Print progress during solving (iterations, objective, constraint violation) and flush output.
- At the end print one line containing exactly one of: OPTIMAL, FEASIBLE, INFEASIBLE, UNBOUNDED, ERROR.
- Also print representative values (at least x(T), theta(T), and the final objective) as:
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
