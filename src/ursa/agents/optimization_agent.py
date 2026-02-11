# optimization_agent.py (refactor: combine Formulator + ReFormulator into one node)
import ast, pprint
import json
from pathlib import Path
from uuid import uuid4
from typing import Any, Dict, List, Literal
from datetime import datetime
import time

import warnings
from pydantic.json_schema import PydanticJsonSchemaWarning

warnings.filterwarnings(
    "ignore",
    category=PydanticJsonSchemaWarning,
    message=r"Default value <factory> is not JSON serializable; excluding default from JSON schema",
)

from pydantic import BaseModel, Field
from dataclasses import asdict, is_dataclass 

from langgraph.prebuilt import InjectedState
from langchain.tools import ToolRuntime
from langchain_core.runnables import RunnableConfig
from ursa.agents.base import AgentContext

from langchain.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, START, StateGraph

from ursa.prompt_library.optimization_prompts import (
    math_formulator_prompt,
    discretizer_prompt,
    feasibility_prompt,
    solver_selector_prompt,
    verifier_prompt,
    optimizer_prompt,
    code_generator_prompt,
)


from ursa.agents.base import BaseAgent
from ursa.tools.feasibility_tools import feasibility_check_auto as fca
from ursa.tools.write_code_tool import write_code
from ursa.tools.run_command_stream_tool import (
    run_command_stream_start,
    run_command_stream_poll,
    run_command_stream_cancel,
)
from ursa.util.helperFunctions import run_tool_calls
from ursa.util.optSchema import OptimizationProblem, SolverPlan, SolverSpec, ToolIO, AttemptSummary
from ursa.util.optSchema import OptimizerState, SolutionState


# -------------------- small structured LLM outputs --------------------

class DiscretizeDecision(BaseModel):
    discretize: bool        # should we discretize the variables and constraints or not?
    note: str = ""          # explanation for this discretization decision

class StreamRouting(BaseModel):
    route: Literal["continue", "terminate", "done"]  # should we stop the solve, continue, or finished?
    reason: str = ""                                 # explanation for this routing decision 

class AdjustDecision(BaseModel):
    action: Literal["proceed", "reformulate", "terminate"]                 # what to do next
    solver_options_patch: Dict[str, Any] = {}  # patch to merge into solver.primary.options
    note: str = ""

def _coerce_solver_plan(x: Any) -> SolverPlan:
    if isinstance(x, SolverPlan):
        return x

    # Dict -> dataclass
    if isinstance(x, dict):
        p = x.get("primary") or {}
        primary = SolverSpec(
            name=p.get("name", "scipy"),
            method=p.get("method"),
            options=p.get("options") or {},
        )

        cands = []
        for c in (x.get("candidates") or []):
            if isinstance(c, dict):
                cands.append(
                    SolverSpec(
                        name=c.get("name", "scipy"),
                        method=c.get("method"),
                        options=c.get("options") or {},
                    )
                )
        return SolverPlan(primary=primary, candidates=cands)

    # Fallback structural default
    return SolverPlan(primary=SolverSpec(name="scipy"), candidates=[])



def _audit(state: OptimizerState, event: str, **fields):
    rec = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "event": event,
        **fields,
    }
    # pprint.pprint(rec)
    state.problem.data.setdefault("_audit", []).append(rec)


# -------------------- robust tool-result extraction --------------------

def _parse_maybe_json(text: Any) -> Any:
    if isinstance(text, (dict, list)):
        return text
    if not isinstance(text, str):
        return text
    t = text.strip()
    if not t:
        return t
    try:
        return json.loads(t)
    except Exception:
        pass
    try:
        return ast.literal_eval(t)
    except Exception:
        return t

def _last_tool_payload(tool_msgs: List[Any]) -> Any:
    if not tool_msgs:
        return None
    last = tool_msgs[-1]
    content = getattr(last, "content", last)
    return _parse_maybe_json(content)

def _recent_polls(state: OptimizerState, k: int = 3) -> List[Any]:
    att = int(state.problem.data.get("_attempt", 0))
    out = []
    for tc in reversed(state.diagnostics.tool_calls):
        if tc.tool == "solve_poll" and (tc.inp or {}).get("attempt") == att:
            out.append(tc.out)
            if len(out) >= k:
                break
    return list(reversed(out))

def _as_tool_runtime(runtime):
    return ToolRuntime(
        context=runtime.context,
        store=getattr(runtime, "store", None),
        stream_writer=getattr(runtime, "stream_writer", None),
        state=getattr(runtime, "state", {}) or {},   # your runtime likely has no .state; make it empty
        tool_call_id=uuid4().hex,                   # required by ToolRuntime schema in your version
        config=getattr(runtime, "config", {}) or {},
    )
# -------------------- graph routers --------------------

def route_after_monitor(state: OptimizerState) -> Literal["continue", "terminate", "done"]:
    return state.problem.data.get("_stream_route", "continue")

def route_after_adjust(state: OptimizerState) -> Literal["proceed", "reformulate", "terminate"]:
    return state.problem.data.get("_adjust_action") or "proceed"


# -------------------- agent --------------------

class OptimizationAgent(BaseAgent[OptimizerState]):
    state_type = OptimizerState

    def __init__(self, llm: BaseChatModel, *args, **kwargs):
        super().__init__(llm, *args, **kwargs)
        self.llm = llm
        self.tools = [
            fca,
            write_code,
            run_command_stream_start,
            run_command_stream_poll,
            run_command_stream_cancel,
        ]
        self.llm_tools = self.llm.bind_tools(self.tools)
        self.tool_maps = {(getattr(t, "name", None) or getattr(t, "__name__", None)): t for t in self.tools}

        self._toolname_write = getattr(write_code, "name", None) or getattr(write_code, "__name__", "write_code")
        self._toolname_fca = getattr(fca, "name", None) or getattr(fca, "__name__", "feasibility_check_auto")
        self._toolname_start = getattr(run_command_stream_start, "name", None) or getattr(run_command_stream_start, "__name__", "run_command_stream_start")
        self._toolname_poll = getattr(run_command_stream_poll, "name", None) or getattr(run_command_stream_poll, "__name__", "run_command_stream_poll")
        self._toolname_cancel = getattr(run_command_stream_cancel, "name", None) or getattr(run_command_stream_cancel, "__name__", "run_command_stream_cancel")

    # -------- Nodes --------

    def Extractor(self, state: OptimizerState) -> OptimizerState:
        state.status = "init"
        state.solution.status = "init"
        state.solution.x = None
        state.solution.obj = None
        state.solution.history.clear()
        state.diagnostics.tool_calls.clear()
        state.problem = OptimizationProblem()
        state.problem.data.clear()
        state.problem.data["_job_id"] = None
        state.problem.data["_stream_route"] = None
        state.problem.data["_stream_reason"] = None
        state.problem.data["_adjust_action"] = None
        state.problem.data["_stream_log"] = []
        state.problem.data["_audit"] = []   # chronological audit trail
        state.problem.data["_solve_start_t"] = None
        state.problem.data["_last_output_t"] = None
        state.problem.data["_attempt"] = 0
        # used to signal whether the next formulation is a reformulation
        state.problem.data["_reformulate_requested"] = False

        print("Extractor Stage Complete")

        return state

    def Formulator(self, state: OptimizerState) -> OptimizerState:
        """
        Combined Formulator + ReFormulator:
        - If state.problem.data["_reformulate_requested"] is True, it reformulates using
          prior formulation + stream reason.
        - Otherwise, it formulates from scratch using state.user_input.

        Side effects:
        - Resets downstream artifacts that must be regenerated (code, job_id, discretization flag).
        """
        reform = bool(state.problem.data.get("_reformulate_requested", False))

        if reform and state.problem.data.get("formulation"):
            user_msg = {
                "formulation": state.problem.data["formulation"],
                "reason": state.problem.data.get("_stream_reason", ""),
                "request": "Reformulate to improve solvability; keep intent unchanged.",
            }
            inp = user_msg
            content = StrOutputParser().invoke(
                self.llm.invoke(
                    [SystemMessage(content=math_formulator_prompt), 
                     HumanMessage(content=str(user_msg)),
                     HumanMessage(content=str(state.user_input))]
                )
            )
            log_tool = "llm_reformulate"
        else:
            inp = state.user_input
            content = StrOutputParser().invoke(
                self.llm.invoke(
                    [SystemMessage(content=math_formulator_prompt), 
                     HumanMessage(content=state.user_input)]
                )
            )
            log_tool = "llm_formulate"

        state.problem.data["formulation"] = content
        state.status = "formulated"
        state.diagnostics.tool_calls.append(ToolIO(tool=log_tool, inp=inp, out=content))

        # reset/clear items that depend on formulation
        state.problem.data.pop("needs_discretization", None)
        state.problem.data.pop("discretization_note", None)

        state.problem.data.pop("solve_py", None)
        state.problem.data.pop("solve_cmd", None)

        state.problem.data["_job_id"] = None
        state.problem.data["_stream_route"] = None
        state.problem.data["_stream_reason"] = None

        # consume the request flag so subsequent passes are "fresh" unless re-set
        state.problem.data["_reformulate_requested"] = False

        print("Formulator Stage Complete")

        return state

    def Discretize(self, state: OptimizerState) -> OptimizerState:
        dec = self.llm.with_structured_output(DiscretizeDecision, method="function_calling").invoke(
            [SystemMessage(content=discretizer_prompt), 
            HumanMessage(content=state.problem.data["formulation"])]
        )
        state.problem.data["needs_discretization"] = dec.discretize
        if dec.note:
            state.problem.data["discretization_note"] = dec.note
        state.diagnostics.tool_calls.append(ToolIO(tool="llm_discretize_check", inp=None, out=dec.model_dump()))
        
        print("Discretizer Complete")

        return state

    def Verify(self, state: OptimizerState) -> OptimizerState:
        formulation = state.problem.data["formulation"]
        llm_out = self.llm_tools.bind(tool_choice={"type": "function", "function": {"name": self._toolname_fca}}).invoke(
            [
                SystemMessage(content=feasibility_prompt),
                HumanMessage(content=formulation),
            ]
        )
        tool_msgs = run_tool_calls(llm_out, self.tool_maps)
        payload = _last_tool_payload(tool_msgs)
        state.diagnostics.tool_calls.append(ToolIO(tool="feasibility_check", inp=None, out=payload))

        print("Verifier Complete")

        return state

    def select_solver(self, state: OptimizerState) -> OptimizerState:
        payload = {
            "formulation": state.problem.data.get("formulation", ""),
            "needs_discretization": state.problem.data.get("needs_discretization", False),
            "feasibility": next(
                (tc.out for tc in reversed(state.diagnostics.tool_calls) if tc.tool == "feasibility_check"),
                None,
            ),
        }

        llm_out = self.llm.with_structured_output(SolverPlan, method="function_calling", include_raw=True).invoke(
            [
                SystemMessage(content=solver_selector_prompt),
                HumanMessage(content=str(payload)),
            ]
        )

        plan = llm_out["parsed"]  # SolverPlan (dataclass) if supported

        state.solver = _coerce_solver_plan(plan)
        state.diagnostics.tool_calls.append(
            ToolIO(tool="llm_select_solver", inp=payload, out={"plan": plan, "raw": llm_out.get("raw")})
        )

        print("Solver Selected")
        return state

    def configure_solver(self, state: OptimizerState) -> OptimizerState:
        """
        Combined node:
        - First-time solver configuration (before codegen): chooses options patch only.
        - Post-cancel adjustment (after monitor->terminate): chooses action + options patch + sets reformulate flag.
        Controlled by state.problem.data["_adjust_action"] presence:
          - If absent/None: "initial configure" mode.
          - If present (set by adjust flow): "adjust" mode.
        """
        mode = "configure" if state.problem.data.get("_attempt", 0) == 0 else "adjust"
        
        payload = {
            "mode": mode,  # "configure" or "adjust"
            "solver": {
                "name": state.solver.primary.name,
                "method": state.solver.primary.method,
                "options": state.solver.primary.options,
                "candidates": [
                    {"name": s.name, "method": s.method, "options": s.options}
                    for s in state.solver.candidates
                ],
            },
            "formulation": state.problem.data.get("formulation", ""),
            "needs_discretization": state.problem.data.get("needs_discretization", False),
            "discretization_note": state.problem.data.get("discretization_note", ""),
            "feasibility": next(
                (tc.out for tc in reversed(state.diagnostics.tool_calls) if tc.tool == "feasibility_check"),
                None,
            ),
            "stream": {
                "reason": state.problem.data.get("_stream_reason", ""),
                "recent": _recent_polls(state, k=3),
            },
        }

        if mode == "configure":
            # Initial config: no branching decision needed; just patch options.
            options_patch = self.llm.invoke(
                [
                    SystemMessage(
                        content=optimizer_prompt
                        + "\n\nYou are configuring solver options before the first solve. "
                          "Return ONLY a JSON object of solver options to apply (keys/values)."
                    ),
                    HumanMessage(content=str(payload)),
                ]
            )

            options_patch = _parse_maybe_json(options_patch.content)

            if isinstance(options_patch, dict):
                state.solver.primary.options.update(options_patch)


            state.diagnostics.tool_calls.append(
                ToolIO(tool="llm_configure_solver", inp=payload, out=options_patch)
            )

            state.problem.data["_adjust_action"] = "proceed"

            print("Configured Solver")

            return state

        # Adjust mode: decide what to do next AND optionally patch options.
        decision = self.llm.with_structured_output(AdjustDecision, method="function_calling").invoke(
            [
                SystemMessage(
                    content=optimizer_prompt
                    + "\n\nYou are monitoring/adjusting a running solve that was terminated. "
                      "Choose action in {proceed, reformulate, terminate}. "
                      "If proceed, include solver_options_patch to improve performance; otherwise leave empty."
                ),
                HumanMessage(content=str(payload)),
            ]
        )
        _audit(state, "adjust_decision", action=decision.action, note=decision.note, patch=decision.solver_options_patch)


        if decision.solver_options_patch:
            state.solver.primary.options.update(decision.solver_options_patch)

        state.problem.data["_adjust_action"] = decision.action
        state.diagnostics.tool_calls.append(ToolIO(tool="llm_adjust", inp=payload, out=decision.model_dump()))

        if decision.action == "reformulate":
            _audit(state, "reformulate_requested", reason=state.problem.data.get("_stream_reason"))
            state.problem.data["_reformulate_requested"] = True

        print("Reconfigured solver")

        return state


    def codegen(self, state: OptimizerState) -> OptimizerState:
        code = StrOutputParser().invoke(
            self.llm.invoke(
                [
                    SystemMessage(content=code_generator_prompt),
                    HumanMessage(
                        content=str(state.user_input)+str(
                            {
                                "formulation": state.problem.data["formulation"],
                                "solver": state.solver.primary,
                                "requirements": [
                                    "Output ONLY valid python code (no markdown fences).",
                                    "Print iterative progress frequently and use flush=True.",
                                    "Print a final status line containing one of: OPTIMAL, FEASIBLE, INFEASIBLE, UNBOUNDED, ERROR.",
                                ],
                            }
                        )
                    ),
                ]
            )
        )

        att = int(state.problem.data.get("_attempt", 0)) + 1
        state.problem.data["_attempt"] = att
        state.problem.data["solve_filename"] = f"solve_attempt_{att}.py"
        state.problem.data["solve_py"] = code
        state.diagnostics.tool_calls.append(ToolIO(tool="llm_codegen", inp=None, out={"chars": len(code)}))

        print("Code Generated")

        return state


    def codewrite(
        self,
        state: OptimizerState,
        runtime: ToolRuntime[AgentContext] = InjectedState("runtime"),
    ) -> OptimizerState:
        code = state.problem.data["solve_py"]
        filename = state.problem.data.get("solve_filename", "solve.py")

        out = write_code.invoke({"code": code, "filename": filename, "runtime": _as_tool_runtime(runtime)})

        state.diagnostics.tool_calls.append(
            ToolIO(tool="write_code", inp={"filename": filename, "chars": len(code)}, out=out)
        )


        state.problem.data["solve_cmd"] = f'python -u "{filename}"'

        print("Code Written")

        return state

    def start_solve(
        self,
        state: OptimizerState,
        runtime: ToolRuntime[AgentContext] = InjectedState("runtime"),
    ) -> OptimizerState:
        if state.problem.data.get("_job_id"):
            return state

        cmd = state.problem.data["solve_cmd"]

        out = run_command_stream_start.invoke({"query": cmd, "runtime": _as_tool_runtime(runtime)})

        _audit(state, "solve_start", cmd=cmd, out=out)
        state.problem.data["_solve_start_t"] = time.time()
        state.problem.data["_last_output_t"] = time.time()

        state.diagnostics.tool_calls.append(ToolIO(tool="solve_start", inp={"cmd": cmd}, out=out))

        if isinstance(out, dict) and out.get("ok") and out.get("job_id"):
            state.problem.data["_job_id"] = out["job_id"]
            state.solution.status = "solving"
            state.problem.data["_stream_route"] = "continue"
        else:
            state.solution.status = "error"
            state.problem.data["_stream_route"] = "terminate"
            state.problem.data["_stream_reason"] = out.get("error") if isinstance(out, dict) else "start failed"
            _audit(state, "solve_start_failed", reason=state.problem.data.get("_stream_reason"))

        print("Starting Solve")

        return state

    def monitor_solve(
        self,
        state: OptimizerState,
        runtime: ToolRuntime[AgentContext] = InjectedState("runtime"),
    ) -> OptimizerState:
        job_id = state.problem.data.get("_job_id")
        if not job_id:
            state.problem.data["_stream_route"] = "terminate"
            state.problem.data["_stream_reason"] = "Missing job_id"
            return state

        poll = run_command_stream_poll.invoke({"job_id": str(job_id), "runtime": _as_tool_runtime(runtime)})

        state.diagnostics.tool_calls.append(ToolIO(tool="solve_poll", inp={"job_id": job_id}, out=poll))


        lines = poll.get("lines") if isinstance(poll, dict) else None
        if isinstance(lines, list) and lines:
            state.problem.data.setdefault("_stream_log", []).extend(lines)
            state.problem.data["_last_output_t"] = time.time()

        log = state.problem.data.get("_stream_log")
        if isinstance(log, list) and len(log) > 2000:
            state.problem.data["_stream_log"] = log[-2000:]
        
        _audit(state, "solve_poll", job_id=str(job_id), done=poll.get("done"), returncode=poll.get("returncode"), n_lines=len(lines or []))        

        # --- deterministic timeouts ---
        MAX_WALL_SEC = 3000       # total allowed solve time (e.g., 5 minutes)
        MAX_SILENCE_SEC = 120      # allowed time since last output line (e.g., 1 minute)

        now = time.time()
        t0 = state.problem.data.get("_solve_start_t")
        t_last = state.problem.data.get("_last_output_t")

        # If start time missing (shouldn't happen), initialize defensively
        if t0 is None:
            state.problem.data["_solve_start_t"] = now
            t0 = now
        if t_last is None:
            state.problem.data["_last_output_t"] = now
            t_last = now

        if now - t0 > MAX_WALL_SEC:
            state.solution.status = "error"
            state.problem.data["_stream_route"] = "terminate"
            state.problem.data["_stream_reason"] = f"Timeout: exceeded max wall time ({MAX_WALL_SEC}s)"
            return state

        if now - t_last > MAX_SILENCE_SEC:
            state.solution.status = "error"
            state.problem.data["_stream_route"] = "terminate"
            state.problem.data["_stream_reason"] = f"Timeout: no output for {MAX_SILENCE_SEC}s"
            return state

        if not isinstance(poll, dict) or not poll.get("ok"):
            state.problem.data["_stream_route"] = "terminate"
            state.problem.data["_stream_reason"] = poll.get("error") if isinstance(poll, dict) else "poll failed"
            _audit(state, "solve_poll_failed", reason=state.problem.data.get("_stream_reason"))
            return state

        if poll.get("done"):
            tail = "".join(state.problem.data.get("_stream_log", [])[-50:]).upper()
            rc = poll.get("returncode")

            # If the run ended in ERROR, treat it as "terminate" so we go to cancel_solve -> configure_solver
            if "ERROR" in tail or (rc not in (None, 0)):
                state.solution.status = "error"  # <--- add this
                state.problem.data["_stream_route"] = "terminate"
                state.problem.data["_stream_reason"] = f"Solver run failed (returncode={rc})"
                _audit(state, "solve_done_error", reason=state.problem.data.get("_stream_reason"))
            else:
                state.problem.data["_stream_route"] = "done"
                _audit(state, "solve_done_ok")
            return state

        # route decision (keep your existing logic)
        route = self.llm.with_structured_output(StreamRouting, method="function_calling").invoke(
            [
                SystemMessage(content=verifier_prompt),
                HumanMessage(content=str({"recent_stream": _recent_polls(state, k=3)})),
            ]
        )
        state.problem.data["_stream_route"] = route.route
        state.problem.data["_stream_reason"] = route.reason
        state.diagnostics.tool_calls.append(ToolIO(tool="llm_stream_router", inp=None, out=route.model_dump()))
        _audit(state, "route_stream", route=route.route, reason=route.reason)

        print("Monitored Run: Deciding next steps")

        return state


    def cancel_solve(
        self,
        state: OptimizerState,
        runtime: ToolRuntime[AgentContext] = InjectedState("runtime"),
    ) -> OptimizerState:
        job_id = state.problem.data.get("_job_id")
        if not job_id:
            return state

        out = run_command_stream_cancel.invoke({"job_id": str(job_id), "runtime": _as_tool_runtime(runtime)})

        state.diagnostics.tool_calls.append(ToolIO(tool="solve_cancel", inp={"job_id": job_id}, out=out))

        _audit(state, "solve_cancel", job_id=str(job_id), out=out)
        state.problem.data["_job_id"] = None


        print("Solve Cancelled: Going back to configure_solve")

        return state


    def finalize(self, state: OptimizerState) -> OptimizerState:
        stream_log = state.problem.data.get("_stream_log", [])
        tail = stream_log[-5000:] if isinstance(stream_log, list) else []

        # print(tail)

        payload = {
            "formulation": state.problem.data.get("formulation", ""),
            "solver": {
                "name": state.solver.primary.name,
                "method": state.solver.primary.method,
                "options": state.solver.primary.options,
            },
            "stream_tail": tail,
            "instructions": (
                "Extract final solution. Return SolutionState with fields:\n"
                "- status: one of {init, formulated, solving, feasible, optimal, infeasible, unbounded, stopped, error}\n"
                "- x: f stream contains a line like 'x=[...]', set x to that list of floats.\n"
                  "     Otherwise if stream contains scalar assignments like 'var=value', set x to a dict.\n"
                  "     Else null.\n"
                "- obj: number if available else null\n"
                "- history: leave empty (agent fills it)\n"
            ),
        }

        # Ask LLM to output a SolutionState-like object.
        # NOTE: This works only if your LangChain version supports dataclasses here.
        llm_out = self.llm.with_structured_output(SolutionState, method="function_calling", include_raw=True).invoke(
            [
                SystemMessage(content=verifier_prompt),
                HumanMessage(content=str(payload)),
            ]
        )
        sol = llm_out["parsed"]  # expected SolutionState

        print(sol)

        # write back into the canonical state.solution
        state.solution.status = sol["status"]
        state.solution.x = sol["x"]
        state.solution.obj = sol["obj"]

        # append attempt summary (agent-owned)
        solver_tag = state.solver.primary.name
        if state.solver.primary.method:
            solver_tag = f"{solver_tag}:{state.solver.primary.method}"

        state.solution.history.append(
            AttemptSummary(solver=solver_tag, status=state.solution.status, obj=state.solution.obj)
        )

        # log diagnostics (ToolIO.out must be JSON-serializable; dataclasses aren't always)
        state.diagnostics.tool_calls.append(
            ToolIO(
                tool="llm_finalize",
                inp={"tail_len": len(tail)},
                out={"parsed_solution": sol, "raw": str(llm_out.get("raw"))[:2000]},
            )
        )

        state.status = state.solution.status
        _audit(state, "finalize", parsed=sol, tail_len=len(tail))

        print("Exiting Summarizer")

        return state


    # -------- Graph --------

    def _build_graph(self):
        # g = StateGraph(OptimizerState)

        self.add_node(self.Extractor,"Extractor")
        self.add_node(self.Formulator,"Formulator")

        self.add_node(self.Discretize, "Discretize")
        self.add_node(self.Verify, "Verify")

        self.add_node(self.select_solver, "select_solver")
        self.add_node(self.configure_solver, "configure_solver")

        self.add_node(self.codegen, "codegen")
        self.add_node(self.codewrite, "codewrite")

        self.add_node(self.start_solve, "start_solve")
        self.add_node(self.monitor_solve, "monitor_solve")
        self.add_node(self.cancel_solve, "cancel_solve")

        self.add_node(self.finalize, "finalize")

        self.graph.add_edge(START, "Extractor")
        self.graph.add_edge("Extractor", "Formulator")

        self.graph.add_edge("Formulator", "Discretize")
        self.graph.add_edge("Discretize", "Verify")

        self.graph.add_edge("Verify", "select_solver")
        self.graph.add_edge("select_solver", "configure_solver")

        self.graph.add_edge("codegen", "codewrite")

        self.graph.add_edge("codewrite", "start_solve")
        self.graph.add_edge("start_solve", "monitor_solve")

        self.graph.add_conditional_edges(
            "monitor_solve",
            route_after_monitor,
            {"continue": "monitor_solve", "terminate": "cancel_solve", "done": "finalize"},
        )

        self.graph.add_edge("cancel_solve", "configure_solver")
        self.graph.add_conditional_edges(
            "configure_solver",
            route_after_adjust,
            {"proceed": "codegen", "reformulate": "Formulator", "terminate": "finalize"},
        )

        self.graph.add_edge("finalize", END)

                # --- graph plot (optional debug artifact) ---
        try:
            import os
            
            compiled = self.graph.compile()

            # 1) Mermaid source
            mermaid = compiled.get_graph().draw_mermaid()
            state_path = getattr(self, "context", None) and getattr(self.context, "workspace", None)
            out_dir = "ursa_workspace"

            os.makedirs(out_dir, exist_ok=True)

            mmd_path = f"{out_dir}/optimization_graph.mmd"
            with open(mmd_path, "w", encoding="utf-8") as f:
                f.write(mermaid)

            # 2) PNG (requires Mermaid rendering support in environment)
            if True:
                png_bytes = compiled.get_graph().draw_mermaid_png()
                png_path = f"{out_dir}/optimization_graph.png"
                with open(png_path, "wb") as f:
                    f.write(png_bytes)
            else:
                pass
        except Exception:
            # If rendering fails, still return a working graph
            pass


if __name__=="__main__":
    from langchain_openai import ChatOpenAI
    from langchain_core.runnables import RunnableConfig
    
    problem_string = """
    Solve this optimization problem:

    Decision variables: x, y (continuous)

    Objective: minimize (x - 1)^2 + (y + 2)^2

    Constraints:
    - x + y = 1
    - x >= 0
    - y >= 0

    Notes:
    - Please print progress during solving.
    - At the end print a line containing one of: OPTIMAL, FEASIBLE, INFEASIBLE, UNBOUNDED, ERROR.
    - Also print variable assignments as simple lines like "x=..." and "y=..." and objective like "obj=...".
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
    for e in final_state["problem"]["data"].get("_audit", []):
        print(f'{e["ts"]}  {e["event"]}  { {k:v for k,v in e.items() if k not in ("ts","event")} }')
