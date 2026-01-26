from typing import Any, Literal, Optional, TypedDict


class DecisionVariableType(TypedDict):
    name: str  # decision variable name
    type: Literal[
        "continuous",
        "integer",
        "logical",
        "infinite-dimensional",
        "finite-dimensional",
    ]  # decision variable type
    domain: str  # allowable values of variable
    description: str  # natural language description


class ParameterType(TypedDict):
    name: str  # parameter name
    value: Optional[Any]  # parameter value; None
    description: str  # natural language description
    is_user_supplied: bool  # 1 if user supplied parameter


class ObjectiveType(TypedDict):
    sense: Literal["minimize", "maximize"]  # objective sense
    expression_nl: str  # sympy-representable mathematical expression
    tags: list[
        Literal["linear", "quadratic", "nonlinear", "convex", "nonconvex"]
    ]  # objective type


class ConstraintType(TypedDict):
    name: str  # constraint name
    expression_nl: str  # sympy-representable mathematical expression
    tags: list[
        Literal[
            "linear",
            "integer",
            "nonlinear",
            "equality",
            "inequality",
            "infinite-dimensional",
            "finite-dimensional",
        ]
    ]  # constraint type


class NotesSpec(TypedDict):
    description: str  # reformulation of the user input problem statement
    latex: str # latex typesetting of the optimization problem
    verifier: Literal["Solved",""]  # solved means stop iterating
    verifier_explanation: str # justification of verifier status


class ProblemSpec(TypedDict):
    decision_variables: list[
        DecisionVariableType
    ]  # list of all decision variables
    parameters: list[ParameterType]  # list of all parameters
    objective: ObjectiveType  # structred objective function details
    constraints: list[ConstraintType]  # structured constraint details
    problem_class: Optional[str]  # optimization problem class
    status: Literal["Feasible", "Infeasible", "Error", ""]  # problem status

class SolutionSpec(TypedDict):
    primal_solution: str # list of decision variable values for the primal problem, if primal is solved
    dual_solution: str # list of decision variable values for the dual problem, if dual is solved
    primal_objective: str # value of the primal objective
    dual_objective: str # value of the dual objective
    optimality_conditions: str # "Equation-only" mathematical expressions for KKT conditions or primal/dual first-order optimality conditions for the optimization problem

class SolverSpec(TypedDict):
    solver: str  # name of the solver, replace with Literal["Gurobi","Ipopt",...] to restrict solvers
    library: str  # library or relevant packages for the solver
    algorithm: Optional[str]  # algorithm used to solve the problem
    license: Optional[
        str
    ]  # License status of the solver (open-source, commercial,etc.)
    parameters: Optional[list[dict]]  # other parameters relevant to the problem
    notes: Optional[str]  # justifying the choice of solver
