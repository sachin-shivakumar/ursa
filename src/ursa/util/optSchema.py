from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Literal

Status = Literal[
    "init", "formulated", "solving",
    "feasible", "optimal", "infeasible", "unbounded",
    "stopped", "error"
]

Sense = Literal["min", "max"]
VarType = Literal["continuous", "integer", "binary"]
ConstraintSense = Literal["<=", "==", ">="]


@dataclass
class Variable:
    name: str                       # variable identifier (string key used in x)
    type: VarType = "continuous"    # variable type: continuous|integer|binary
    lb: Optional[float] = None      # lower bound (None = -inf)
    ub: Optional[float] = None      # upper bound (None = +inf)
    x0: Optional[float] = None      # initial guess / warm start


@dataclass
class Objective:
    expr: Any                       # objective expression: callable|symbolic|AST|matrix-handle
    grad: Optional[Any] = None      # optional gradient provider (same form as expr)
    hess: Optional[Any] = None      # optional Hessian provider (same form as expr)


@dataclass
class Constraint:
    name: str                       # constraint identifier
    expr: Any                       # constraint expression: callable|symbolic|AST|matrix-handle (LHS)
    sense: ConstraintSense          # relation: <= | == | >=
    rhs: Any                        # right-hand side value(s): float|array|symbolic


@dataclass
class OptimizationProblem:
    sense: Sense = "min"                                # optimization direction: min|max
    vars: List[Variable] = field(default_factory=list)   # decision variables (ordered list)
    objective: Optional[Objective] = None                # objective definition (required before solve)
    constraints: List[Constraint] = field(default_factory=list)  # constraints (possibly empty)
    data: Dict[str, Any] = field(default_factory=dict)   # structured coefficients/data for model build (LLM-friendly JSON)


@dataclass
class SolverSpec:
    name: str                                            # solver backend name (e.g., scipy|ipopt|gurobi)
    method: Optional[str] = None                         # algorithm/method within solver (optional)
    options: Dict[str, Any] = field(default_factory=dict)  # solver parameters (LLM-friendly JSON)


@dataclass
class SolverPlan:
    primary: SolverSpec                                  # chosen solver config
    candidates: List[SolverSpec] = field(default_factory=list)  # fallback solver configs (optional)


@dataclass
class AttemptSummary:
    solver: str                                          # "name[:method]" for the attempt
    status: Status                                       # outcome status
    obj: Optional[float] = None                          # objective value (if available)


@dataclass
class SolutionState:
    status: Status = "init"                              # current solution status
    x: Optional[Dict[str, Any]] = None                   # variable assignment keyed by Variable.name (LLM-friendly JSON)
    obj: Optional[float] = None                          # objective value at x (if available)
    history: List[AttemptSummary] = field(default_factory=list)  # attempt log (optional)


@dataclass
class ToolIO:
    tool: str                                            # tool/node name ("llm_parse", "solve", "reformulate", ...)
    inp: Any                                             # tool input payload (must be JSON-serializable for logging)
    out: Any                                             # tool output payload (must be JSON-serializable for logging)
    err: Optional[str] = None                            # error string if tool failed


@dataclass
class Diagnostics:
    tool_calls: List[ToolIO] = field(default_factory=list)  # chronological tool call log


@dataclass
class OptimizerState:
    user_input: str                                    # raw user problem statement (LLM input)
    status: Status = "init"                              # overall run status
    problem: OptimizationProblem = field(default_factory=OptimizationProblem)  # canonical problem
    solver: SolverPlan = field(default_factory=lambda: SolverPlan(primary=SolverSpec(name="scipy")))  # solver plan
    solution: SolutionState = field(default_factory=SolutionState)             # current/best solution
    diagnostics: Diagnostics = field(default_factory=Diagnostics)              # tool call log
