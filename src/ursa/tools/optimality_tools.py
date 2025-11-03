from typing import List, Dict, Any, Optional, Tuple, Annotated
import math

import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations
from sympy.core.relational import Relational  # robust across SymPy versions
from langchain_core.tools import tool

# Optional deps (used when available)
try:
    import numpy as np
    _HAS_NUMPY = True
except Exception:
    _HAS_NUMPY = False

try:
    from ortools.linear_solver import pywraplp as _lp
    _HAS_LP = True
except Exception:
    _HAS_LP = False


# ----------------------------
# Helpers: parsing & building KKT
# ----------------------------

def _parse_symbols(variable_name: List[str]) -> List[sp.Symbol]:
    """Return SymPy symbols for variable names."""
    return sp.symbols(variable_name)


def _parse_exprs(exprs: List[str], varnames: List[str]) -> List[sp.Expr]:
    """Parse a list of string expressions into SymPy expressions using the same symbol table."""
    syms = _parse_symbols(varnames)
    local = {n: s for n, s in zip(varnames, syms)}
    return [
        parse_expr(e, local_dict=local, transformations=standard_transformations, evaluate=False)
        for e in exprs
    ]


def _normalize_constraints(
    constraints: List[sp.Expr]
) -> Tuple[List[sp.Expr], List[sp.Expr], List[str]]:
    """Normalize constraints into g(x) ≤ 0 (inequalities) and h(x) = 0 (equalities).

    Args:
        constraints: List of SymPy relational/boolean expressions.

    Returns:
        (g_list, h_list, flags)
        g_list: list of SymPy expressions with ≤ 0 sense (each is lhs - rhs or rhs - lhs).
        h_list: list of SymPy expressions with = 0 sense.
        flags:  parallel list of string tags for each input constraint indicating how it was mapped:
               'le', 'lt', 'ge', 'gt', 'eq', or 'other' (for non-relational/logic).
    """
    g_list, h_list, flags = [], [], []
    for c in constraints:
        if not isinstance(c, Relational):
            # Any logical composition makes KKT unsuitable
            flags.append("other")
            continue
        # Map to standard forms
        if isinstance(c, sp.Le):       # lhs ≤ rhs  => g := lhs - rhs ≤ 0
            g_list.append(sp.simplify(c.lhs - c.rhs)); flags.append("le")
        elif isinstance(c, sp.Lt):     # lhs < rhs  => treat as lhs - rhs ≤ 0 (strictness ignored in KKT)
            g_list.append(sp.simplify(c.lhs - c.rhs)); flags.append("lt")
        elif isinstance(c, sp.Ge):     # lhs ≥ rhs  => rhs - lhs ≤ 0
            g_list.append(sp.simplify(c.rhs - c.lhs)); flags.append("ge")
        elif isinstance(c, sp.Gt):     # lhs > rhs  => rhs - lhs ≤ 0
            g_list.append(sp.simplify(c.rhs - c.lhs)); flags.append("gt")
        elif isinstance(c, sp.Eq):     # lhs = rhs  => h := lhs - rhs = 0
            h_list.append(sp.simplify(c.lhs - c.rhs)); flags.append("eq")
        else:
            flags.append("other")
    return g_list, h_list, flags


def _grad(expr: sp.Expr, symbols: List[sp.Symbol]) -> sp.Matrix:
    """Gradient column vector."""
    return sp.Matrix([sp.diff(expr, s) for s in symbols])


def _hessian(expr: sp.Expr, symbols: List[sp.Symbol]) -> sp.Matrix:
    """Hessian matrix."""
    return sp.hessian(expr, symbols)


def _as_float(val) -> float:
    """Convert a SymPy value to python float."""
    return float(sp.N(val))


def _dict_to_env(symbols: List[sp.Symbol], candidate: Dict[str, Any]) -> Dict[sp.Symbol, Any]:
    """Map variable symbols to numeric values from candidate."""
    return {s: candidate[str(s)] for s in symbols if str(s) in candidate}


def _is_continuous_types(vtypes: List[str]) -> bool:
    """True if all vars are reals/continuous."""
    tl = [t.lower() for t in vtypes]
    return all(t in ("real", "float", "double", "continuous") for t in tl)


def _has_discrete_types(vtypes: List[str]) -> bool:
    """True if any var is integer/boolean."""
    tl = [t.lower() for t in vtypes]
    return any(t in ("int", "integer", "bool", "boolean", "logical") for t in tl)


def _feasible(constraints: List[sp.Expr], env: Dict[sp.Symbol, Any], tol: float) -> Tuple[bool, List[str]]:
    """Check feasibility of env against constraints; return (ok, messages)."""
    msgs = []
    ok_all = True
    for c in constraints:
        if isinstance(c, Relational):
            lhs = _as_float(c.lhs.subs(env))
            rhs = _as_float(c.rhs.subs(env))
            d = lhs - rhs
            if isinstance(c, sp.Eq):
                ok = abs(d) <= tol
                if not ok: msgs.append(f"Equality violated: {sp.srepr(c)} : |lhs-rhs|={abs(d)}")
            elif isinstance(c, sp.Le):
                ok = d <= tol
                if not ok: msgs.append(f"Ineq (<=) violated by {d}")
            elif isinstance(c, sp.Lt):
                ok = d < -tol
                if not ok: msgs.append(f"Strict (<) violated by {d}")
            elif isinstance(c, sp.Ge):
                ok = d >= -tol
                if not ok: msgs.append(f"Ineq (>=) violated by {d}")
            elif isinstance(c, sp.Gt):
                ok = d > tol
                if not ok: msgs.append(f"Strict (>) violated by {d}")
            else:
                ok = False
                msgs.append("Unknown relational type.")
        else:
            ok = False
            msgs.append("Non-relational/logic constraint — not supported for KKT.")
        ok_all = ok_all and ok
    return ok_all, msgs


# ----------------------------
# KKT generation & checking
# ----------------------------

def _generate_kkt(
    f: sp.Expr,
    constraints: List[sp.Expr],
    symbols: List[sp.Symbol],
    sense: str
) -> Dict[str, Any]:
    """Build KKT conditions (symbolic) for smooth problems.

    Returns:
        dict with keys:
          'objective' (possibly sign-flipped to minimization),
          'g', 'h' (normalized constraints),
          'lambda', 'mu' (multiplier symbols),
          'stationarity' (list of expressions == 0),
          'comp_slackness' (list of expressions == 0),
          'dual_feas' (list of inequalities lambda_i >= 0),
          'notes' (list of strings)
    """
    notes = []
    # minimize: if maximize, flip sign
    s = sense.lower().strip()
    if s not in ("min", "max", "minimize", "maximize"):
        s = "min"
        notes.append("Unknown sense; defaulting to 'min'.")
    f_min = f if s.startswith("min") else -f

    g_list, h_list, flags = _normalize_constraints(constraints)

    # Lagrangian: L = f + sum_i λ_i g_i + sum_j μ_j h_j
    lam_syms = sp.symbols([f"lam_{i+1}" for i in range(len(g_list))])
    mu_syms  = sp.symbols([f"mu_{j+1}"  for j in range(len(h_list))])

    L = f_min
    for lam, g in zip(lam_syms, g_list):
        L = L + lam * g
    for mu, h in zip(mu_syms, h_list):
        L = L + mu * h

    # Stationarity ∇x L = 0
    gradL = _grad(L, symbols)
    stationarity = [sp.simplify(gi) for gi in gradL]  # each should be == 0

    # Complementarity: lam_i * g_i(x) = 0
    comp = [sp.simplify(lam_syms[i] * g_list[i]) for i in range(len(g_list))]

    # Dual feasibility: lam_i >= 0
    dual_feas = [sp.Ge(lam, 0) for lam in lam_syms]

    return dict(
        objective=f_min,
        g=g_list,
        h=h_list,
        lambda_symbols=list(lam_syms),
        mu_symbols=list(mu_syms),
        stationarity=stationarity,
        complementarity=comp,
        dual_feas=dual_feas,
        flags=flags,
        notes=notes
    )


def _solve_kkt_multipliers_at_point(
    f: sp.Expr,
    g_list: List[sp.Expr],
    h_list: List[sp.Expr],
    symbols: List[sp.Symbol],
    x_env: Dict[sp.Symbol, Any],
    sense: str,
    tol_active: float = 1e-7
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, Any]]:
    """Given x*, compute multipliers (λ for active g’s, μ for all h’s) via stationarity.

    We evaluate gradients at x*:
        ∇f(x*) + Σ_{i in A} λ_i ∇g_i(x*) + Σ_j μ_j ∇h_j(x*) = 0
    where A = {i : g_i(x*) ≈ 0} (active inequalities).
    This is a linear system in unknown multipliers; we solve least-squares if over/underdetermined.

    Args:
        f: Objective (minimization form).
        g_list: Inequality functions (≤ 0).
        h_list: Equality functions (= 0).
        symbols: Variables.
        x_env: Mapping of variable symbols to candidate values.
        sense: "min"/"max" (already applied outside, but used for info).
        tol_active: Threshold for active set detection.

    Returns:
        (lam_map, mu_map, info) where lam_map/mu_map map symbol names to floats,
        and info has residual norm, active set indices, and rank diagnostics.
    """
    f_min = f if sense.lower().startswith("min") else -f

    # Active set detection
    g_vals = [ _as_float(g.subs(x_env)) for g in g_list ]
    active_idx = [i for i, gv in enumerate(g_vals) if abs(gv) <= tol_active]

    # Build matrix [G_active | H] and rhs = -∇f
    gradf = np.array([_as_float(sp.diff(f_min, s).subs(x_env)) for s in symbols], dtype=float).reshape(-1, 1)

    Gcols, Hcols = [], []
    for i in active_idx:
        ggrad = [_as_float(sp.diff(g_list[i], s).subs(x_env)) for s in symbols]
        Gcols.append(ggrad)
    for h in h_list:
        hgrad = [_as_float(sp.diff(h, s).subs(x_env)) for s in symbols]
        Hcols.append(hgrad)

    # Stack columns
    Acols = Gcols + Hcols
    if len(Acols) == 0:
        A = np.zeros((len(symbols), 0))
    else:
        A = np.array(Acols, dtype=float).T  # shape (n, m)

    # Solve A * y = -gradf  (y = [lam_active, mu])
    if A.shape[1] == 0:
        # No multipliers -> require gradf ≈ 0
        resid = float(np.linalg.norm(gradf, ord=2))
        lam_map, mu_map = {}, {}
        info = dict(residual=resid, active_set=active_idx, rank=0, ncols=0, nvars=len(symbols))
        return lam_map, mu_map, info

    y, residuals, rank, svals = np.linalg.lstsq(A, -gradf, rcond=None)
    y = y.flatten()
    # Split back to lam, mu
    k = len(active_idx)
    lam_vec = y[:k]
    mu_vec = y[k:]

    lam_map = { f"lam_{i+1}": float(lam_vec[j]) for j, i in enumerate(active_idx) }
    mu_map  = { f"mu_{j+1}": float(mu_vec[j]) for j in range(len(h_list)) }

    # Compute residual
    res_vec = A.dot(y.reshape(-1,1)) + gradf
    resid_norm = float(np.linalg.norm(res_vec, ord=2))

    info = dict(residual=resid_norm, active_set=active_idx, rank=int(rank), svals=[float(s) for s in svals],
                ncols=A.shape[1], nvars=A.shape[0])
    return lam_map, mu_map, info


def _second_order_check(
    f: sp.Expr,
    g_list: List[sp.Expr],
    h_list: List[sp.Expr],
    symbols: List[sp.Symbol],
    x_env: Dict[sp.Symbol, Any],
    lam_active: Dict[int, float],
    mu_all: Dict[int, float],
    tol_psd: float = 1e-8
) -> Dict[str, Any]:
    """Second-order necessary condition on tangent space (minimization).

    Projects the Lagrangian Hessian to the nullspace of active constraints’ gradients;
    checks PSD (all eigenvalues ≥ -tol_psd). If strictly positive (≥ tol_psd), it’s a
    local sufficient condition.

    Returns a dict with eigenvalues and boolean flags.
    """
    # Build Lagrangian
    L = f
    for i, g in enumerate(g_list):
        if i in lam_active:
            L += lam_active[i] * g
    for j, h in enumerate(h_list):
        L += mu_all.get(j, 0.0) * h

    # Hessian at x*
    H = np.array(_hessian(L, symbols).subs(x_env)).astype(float)

    # Tangent space basis: gradients of active inequalities + all equalities
    grads = []
    for i, g in enumerate(g_list):
        if i in lam_active:
            grads.append([_as_float(sp.diff(g, s).subs(x_env)) for s in symbols])
    for h in h_list:
        grads.append([_as_float(sp.diff(h, s).subs(x_env)) for s in symbols])
    G = np.array(grads, dtype=float) if grads else np.zeros((0, len(symbols)))

    # Nullspace via SVD
    if G.shape[0] == 0:
        # No active constraints -> whole space is tangent space
        B = np.eye(len(symbols))
    else:
        U, S, Vt = np.linalg.svd(G, full_matrices=True)
        r = (S > 1e-10).sum()
        B = Vt[r:].T  # columns span the nullspace of G

    Hproj = B.T @ H @ B
    vals = np.linalg.eigvalsh(Hproj) if B.size > 0 else np.array([])

    return dict(
        eigenvalues=[float(v) for v in (vals if vals.size else np.array([]))],
        psd=bool(vals.size == 0 or np.min(vals) >= -tol_psd),
        pd=bool(vals.size == 0 or np.min(vals) > tol_psd)
    )


def _objective_value(f: sp.Expr, x_env: Dict[sp.Symbol, Any], sense: str) -> float:
    """Evaluate objective (original sign) at x*."""
    val = _as_float(f.subs(x_env))
    return val if sense.lower().startswith("min") else val


# ----------------------------
# MILP/LP fallback for discrete or non-KKT cases
# ----------------------------

def _solve_with_cbc(
    objective: sp.Expr,
    sense: str,
    constraints: List[sp.Expr],
    variable_name: List[str],
    variable_type: List[str],
    variable_bounds: List[List[Optional[float]]]
) -> Tuple[str, Optional[Dict[str, float]], Optional[float]]:
    """Solve with OR-Tools CBC (LP/MILP) to certify optimality when KKT is N/A.

    Returns:
        (status, model, obj_value)
    """
    if not _HAS_LP:
        return ("CBC not installed", None, None)

    solver = _lp.Solver.CreateSolver("CBC_MIXED_INTEGER_PROGRAMMING")
    if solver is None:
        return ("CBC create failed", None, None)

    # Variables
    var = {}
    for (n, t), (lo, hi) in zip(zip(variable_name, variable_type), variable_bounds):
        t_l = t.lower()
        lo_v = -_lp.Solver.infinity() if lo is None else float(lo)
        hi_v =  _lp.Solver.infinity() if hi is None else float(hi)
        if t_l in ("boolean", "bool", "logical"):
            var[n] = solver.IntVar(0, 1, n)
        elif t_l in ("integer", "int"):
            var[n] = solver.IntVar(math.floor(lo_v), math.ceil(hi_v), n)
        else:
            var[n] = solver.NumVar(lo_v, hi_v, n)

    # Constraints
    for c in constraints:
        if not isinstance(c, Relational):
            return ("Non-relational constraints not supported by CBC", None, None)
        diff = sp.simplify(c.lhs - c.rhs)
        # Build lhs sum coeff*var and set bounds around -const
        coeffs = {n: float(sp.N(sp.diff(diff, sp.Symbol(n)))) for n in variable_name}
        const = float(sp.N(diff.subs({sp.Symbol(n): 0 for n in variable_name})))
        lb, ub = -_lp.Solver.infinity(), _lp.Solver.infinity()
        if isinstance(c, sp.Eq):
            lb = ub = -const
        elif isinstance(c, sp.Le):
            ub = -const
        elif isinstance(c, sp.Ge):
            lb = -const
        elif isinstance(c, sp.Lt):
            ub = -const - 1e-9
        elif isinstance(c, sp.Gt):
            lb = -const + 1e-9
        ct = solver.RowConstraint(lb, ub, "")
        for n in variable_name:
            ct.SetCoefficient(var[n], coeffs.get(n, 0.0))

    # Objective
    # Build c^T x + c0
    coeffs = {n: float(sp.N(sp.diff(objective, sp.Symbol(n)))) for n in variable_name}
    c0 = float(sp.N(objective.subs({sp.Symbol(n): 0 for n in variable_name})))
    expr = solver.Sum(coeffs[n]*var[n] for n in variable_name)
    if sense.lower().startswith("min"):
        solver.Minimize(expr + c0)
    else:
        solver.Maximize(expr + c0)

    status = solver.Solve()
    if status not in ( _lp.Solver.OPTIMAL, _lp.Solver.FEASIBLE ):
        return ("infeasible_or_error", None, None)
    model = {n: (int(round(var[n].solution_value())) if variable_type[i].lower() in ("integer","int","boolean","bool","logical") else float(var[n].solution_value()))
             for i, n in enumerate(variable_name)}
    obj = sum(coeffs[n]*model[n] for n in variable_name) + c0
    return ("optimal" if status == _lp.Solver.OPTIMAL else "feasible", model, float(obj))


# ----------------------------
# The Tool
# ----------------------------

@tool(parse_docstring=True)
def optimality_check(
    objective: Annotated[str, "Objective function as a string, e.g. 'x**2 + y**2'"],
    sense: Annotated[str, "'min' or 'max'"],
    constraints: Annotated[List[str], "Constraints like 'x + y <= 1', 'x - y == 0'"],
    variable_name: Annotated[List[str], "Variable names, e.g. ['x','y']"],
    variable_type: Annotated[List[str], "Per-variable types: 'real'|'integer'|'boolean'"],
    variable_bounds: Annotated[List[List[Optional[float]]], "Per-variable [low, high] (use None for unbounded)"],
    candidate: Annotated[Dict[str, float], "Feasible point mapping, e.g. {'x':0.3,'y':0.7}"],
    kkt_active_tol: Annotated[float, "Active-set tolerance for |g(x)|"] = 1e-7,
    kkt_stationarity_tol: Annotated[float, "Tolerance for stationarity residual"] = 1e-6,
    feasibility_tol: Annotated[float, "Tolerance to accept feasibility of candidate"] = 1e-7,
    second_order_check: Annotated[bool, "If True, run a PSD test on the tangent space"] = False,
) -> str:
    """Generate optimality conditions and check candidate optimality.

    Behavior:
      • For smooth continuous problems: builds KKT conditions symbolically (stationarity,
        complementary slackness, dual feasibility, primal feasibility) and tests them at
        the provided candidate (first-order KKT; optional 2nd-order PSD test).
      • If any integer/boolean variables or non-relational/logical constraints exist,
        KKT is marked "N/A". The tool can optionally verify optimality by solving the
        LP/MILP with CBC and comparing the candidate objective to the global optimum.

    Args:
        objective: Objective function as a string (SymPy syntax).
        sense: 'min' or 'max'.
        constraints: List of constraint strings (SymPy relational forms).
        variable_name: Variable names.
        variable_type: Variable types for each variable ('real', 'integer', 'boolean').
        variable_bounds: Per-variable [low, high] bounds; use None for unbounded side.
        candidate: A (supposedly) feasible solution mapping from name to numeric value.
        kkt_active_tol: |g(x*)| threshold to treat an inequality as active.
        kkt_stationarity_tol: ||∇f + Σλ∇g + Σμ∇h||₂ tolerance at x*.
        feasibility_tol: Feasibility tolerance for constraints at x*.
        second_order_check: If True, run a second-order (PSD) test on tangent space.

    Returns:
        A JSON-like string with:
          - 'conditions': { 'stationarity', 'complementarity', 'dual_feas', ... } (symbolic)
          - 'kkt_applicable': bool
          - 'candidate_feasible': bool and feasibility diagnostics
          - 'kkt_first_order_satisfied': bool and residual/dual/sign/complementarity diagnostics
          - 'second_order': (if requested) PSD/PD flags and eigenvalues
          - 'cbc_verification' (for discrete/logic cases): solver status and best objective
          - 'is_optimal' flag (best effort; sufficient if convex or MILP via CBC; necessary-only for general smooth nonconvex)

    Notes:
        • KKT are necessary conditions for local optima under constraint qualifications (e.g., LICQ).
          They are sufficient for convex problems (convex f, convex g, affine h).
        • For mixed-integer/discrete problems, KKT doesn't apply. We provide a solver-based
          certificate instead when possible.
    """
    # Parse symbols and expressions
    syms = _parse_symbols(variable_name)
    local = {n: s for n, s in zip(variable_name, syms)}
    try:
        f = parse_expr(objective, local_dict=local, transformations=standard_transformations, evaluate=False)
        cons = _parse_exprs(constraints, variable_name)
    except Exception as e:
        return f'{{"error":"Parse error: {str(e)}"}}'

    # Build candidate env and feasibility check
    try:
        x_env = _dict_to_env(syms, candidate)
    except Exception as e:
        return f'{{"error":"Candidate mapping error: {str(e)}"}}'

    feas_ok, feas_msgs = _feasible(cons, x_env, feasibility_tol)

    # Generate KKT if applicable
    has_discrete = _has_discrete_types(variable_type)
    has_logic = any(not isinstance(c, Relational) for c in cons)

    out: Dict[str, Any] = {}
    kkt_applicable = (not has_discrete) and (not has_logic)
    out["kkt_applicable"] = bool(kkt_applicable)

    if kkt_applicable:
        conds = _generate_kkt(f, cons, syms, sense)
        out["conditions"] = {
            "stationarity": [str(e) + " = 0" for e in conds["stationarity"]],
            "complementarity": [str(e) + " = 0" for e in conds["complementarity"]],
            "dual_feasibility": [str(ineq) for ineq in conds["dual_feas"]],
            "notes": conds["notes"],
        }
    else:
        out["conditions"] = {
            "stationarity": "N/A (discrete variables or logical constraints present)",
            "complementarity": "N/A",
            "dual_feasibility": "N/A",
            "notes": ["KKT not applicable; using solver-based verification if available."],
        }

    out["candidate_feasible"] = feas_ok
    if not feas_ok:
        out["feasibility_violations"] = feas_msgs
        out["is_optimal"] = False
        return str(out)

    # If KKT applies: compute multipliers and check FONC (+ optional SONC)
    if kkt_applicable:
        g_list, h_list, _ = _normalize_constraints(cons)
        if not _HAS_NUMPY:
            out["kkt_first_order_satisfied"] = False
            out["kkt_check_message"] = "NumPy not available to solve stationarity linear system."
            out["is_optimal"] = False
            return str(out)

        lam_map, mu_map, info = _solve_kkt_multipliers_at_point(
            f=f, g_list=g_list, h_list=h_list, symbols=syms, x_env=x_env, sense=sense,
            tol_active=kkt_active_tol
        )

        # Evaluate complementarity and dual feasibility
        g_vals = [ _as_float(g.subs(x_env)) for g in g_list ]
        # Build full λ vector (inactive λ=0)
        lam_full = [0.0]*len(g_list)
        for k, v in lam_map.items():
            # k is 'lam_i', index is i-1
            idx = int(k.split("_")[1]) - 1
            lam_full[idx] = v
        comp_ok = True
        comp_viol = []
        for i, gv in enumerate(g_vals):
            prod = lam_full[i] * gv
            if abs(prod) > max(1.0, abs(gv), abs(lam_full[i])) * 1e-6:
                comp_ok = False
                comp_viol.append(f"lam_{i+1}*g_{i+1}={prod} (lam={lam_full[i]}, g={gv})")

        dual_ok = all(lam >= -1e-9 for lam in lam_full)

        fonc_ok = (info["residual"] <= kkt_stationarity_tol) and comp_ok and dual_ok
        out["kkt_first_order_satisfied"] = bool(fonc_ok)
        out["kkt_diagnostics"] = dict(
            stationarity_residual=info["residual"],
            active_set=info["active_set"],
            dual_nonneg_all=dual_ok,
            complementarity_ok=comp_ok,
            complementarity_violations=comp_viol,
            multipliers_lambda={k: float(v) for k, v in lam_map.items()},
            multipliers_mu={k: float(v) for k, v in mu_map.items()},
        )

        # Optional second-order test on tangent space
        if second_order_check:
            lam_active_dict = {i: lam_full[i] for i in range(len(g_list)) if i in info["active_set"]}
            mu_all_dict  = {j: list(mu_map.values())[j] if j < len(mu_map) else 0.0 for j in range(len(h_list))}
            so = _second_order_check(
                f=(f if sense.lower().startswith("min") else -f),
                g_list=g_list,
                h_list=h_list,
                symbols=syms,
                x_env=x_env,
                lam_active=lam_active_dict,
                mu_all=mu_all_dict,
                tol_psd=1e-8
            )
            out["second_order"] = so

        # Optimality verdict (continuous):
        #  - If convex (not checked), KKT sufficient. Here we just report KKT FONC (+ optional SONC).
        #  - We expose best-effort flag: KKT satisfied => "locally optimal (necessary); sufficient if convex".
        out["is_optimal"] = bool(fonc_ok)  # caveat noted above
        out["verdict_note"] = "KKT FONC satisfied → locally optimal (necessary). Sufficient if the problem is convex."

        return str(out)

    # Otherwise: discrete / logic → certify via CBC when available
    status, model, opt_obj = _solve_with_cbc(
        objective=f, sense=sense, constraints=cons,
        variable_name=variable_name, variable_type=variable_type, variable_bounds=variable_bounds
    )
    cand_val = _objective_value(f, x_env, sense)
    out["cbc_verification"] = dict(status=status, best_solution=model, best_objective=opt_obj, candidate_objective=cand_val)

    if status == "optimal" and opt_obj is not None:
        out["is_optimal"] = bool(abs(cand_val - opt_obj) <= 1e-9)
        if not out["is_optimal"]:
            out["note"] = "Candidate is feasible but not optimal per MILP solve."
    else:
        out["is_optimal"] = False
        out["note"] = "Could not certify via CBC (not installed or solver error)."

    return str(out)
