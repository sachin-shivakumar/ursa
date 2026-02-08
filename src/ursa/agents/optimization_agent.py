import os
import pprint
import subprocess
from typing import Annotated, Literal, TypedDict

from langchain.chat_models import BaseChatModel
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START
from langgraph.prebuilt import InjectedState

from ursa.prompt_library.optimization_prompts import (
    code_generator_prompt,
    discretizer_prompt,
    explainer_prompt,
    extractor_prompt,
    feasibility_prompt,
    math_formulator_prompt,
    solver_selector_prompt,
    verifier_prompt,
    optimizer_prompt,
)
from ..tools.feasibility_tools import feasibility_check_auto as fca
from ..tools.write_code_tool import write_code
from ..tools.run_command_tool import run_command
from ..util.helperFunctions import extract_tool_calls, run_tool_calls
from ..util.optimization_schema import ProblemSpec, SolverSpec, SolutionSpec

from .base import BaseAgent

# --- ANSI color codes ---
GREEN = "\033[92m"
BLUE = "\033[94m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"


class OptimizerState(TypedDict):
    user_input: str
    problem: ProblemSpec
    solver: SolverSpec
    solution: SolutionSpec
    notes: NotesSpec
    data: list[list[Any]]


class OptimizationAgent(BaseAgent[OptimizerState]):
    state_type = OptimizerState

    def __init__(self, llm: BaseChatModel, *args, **kwargs):
        super().__init__(llm, *args, **kwargs)
        self.extractor_prompt = extractor_prompt
        self.explainer_prompt = explainer_prompt
        self.verifier_prompt = verifier_prompt
        self.code_generator_prompt = code_generator_prompt
        self.solver_selector_prompt = solver_selector_prompt
        self.math_formulator_prompt = math_formulator_prompt
        self.discretizer_prompt = discretizer_prompt
        self.feasibility_prompt = feasibility_prompt
        self.optimizer_prompt = optimizer_prompt
        self.tools = [fca, write_code, run_command]  # [run_cmd, write_code, search_tool, fca, oc]
        self.llm = self.llm.bind_tools(self.tools)
        self.tool_maps = {
            (getattr(t, "name", None) or getattr(t, "__name__", None)): t
            for i, t in enumerate(self.tools)
        }

    # Define the function that calls the model
    def extractor(self, state: OptimizerState) -> OptimizerState:
        new_state = state.copy()
        new_state["problem"] = StrOutputParser().invoke(
            self.llm.invoke([
                SystemMessage(content=self.extractor_prompt),
                HumanMessage(content=new_state["user_input"]),
            ])
        )

        notes = self.llm.with_structured_output(NotesSpec).invoke([SystemMessage(content=self.extractor_prompt), 
            HumanMessage(content=new_state["user_input"]),
            ])

        print("Problem Extractor and Formulator:\n")
        pprint.pprint(problem)
        return {
        "problem": problem,
        "notes": notes,
        }


    def discretizer(self, state: OptimizerState) -> OptimizerState:
        new_state = state.copy()

        problem = self.llm.with_structured_output(
            ProblemSpec
        ).invoke([
            SystemMessage(content=self.discretizer_prompt),
            HumanMessage(content=str(state["problem"])),
        ])

        print("Discretizing Problem:\n")
        pprint.pprint(problem)

        return {
        "problem": problem
        }

    def tester(self, state: OptimizerState) -> OptimizerState:
        new_state = state.copy()

        llm_out = self.llm.bind(tool_choice="required").invoke([
            SystemMessage(content=self.feasibility_prompt),
            HumanMessage(content=str(state["problem"])),
        ])


        tool_log = run_tool_calls(llm_out, self.tool_maps)

        notes["diagnostic"] = []
        notes["diagnostic"].extend(tool_log)
        
        print("Feasibility Tester:\n")
        for msg in tool_log:
            msg.pretty_print()
        return {
        "notes": notes
        }

    def selector(self, state: OptimizerState) -> OptimizerState:
        new_state = state.copy()

        llm_out = self.llm.with_structured_output(
            SolverSpec, include_raw=True
        ).invoke([
            SystemMessage(content=self.solver_selector_prompt),
            HumanMessage(content=str(state["problem_spec"])),
        ])
        new_state["solver"] = llm_out["parsed"]

        print("Selector:\n ")
        pprint.pprint(new_state["solver"])
        return new_state

    def generator(self, state: OptimizerState) -> OptimizerState:
        new_state = state.copy()

        new_state["code"] = StrOutputParser().invoke(
            self.llm.invoke([
                SystemMessage(content=self.code_generator_prompt),
                HumanMessage(content=str(state["problem_spec"])),
            ])
        )

        print("Generator:\n")
        pprint.pprint(new_state["code"])
        return new_state

    def verifier(self, state: OptimizerState) -> OptimizerState:
        new_state = state.copy()

        llm_out = self.llm.with_structured_output(
            ProblemSpec, include_raw=True
        ).invoke([
            SystemMessage(content=self.verifier_prompt),
            HumanMessage(content=str(state["problem_spec"]) + state["code"]),
        ])
        new_state["problem_spec"] = llm_out["parsed"]
        if hasattr(llm_out, "tool_calls"):
            tool_log = run_tool_calls(llm_out, self.tool_maps)
            new_state["problem_diagnostic"].extend(tool_log)

        print("Verifier:\n ")
        pprint.pprint(new_state["problem_spec"])
        return new_state

    def optimizer(self, state:OptimizerState) -> OptimizerState:
        new_state = state.copy()

        llm_out = self.llm.with_structured_output(
            SolutionSpec, include_raw=True
        ).invoke([
            SystemMessage(content=self.optimizer_prompt),
            HumanMessage(content=str(state["problem_spec"]) + state["code"]),
        ])
        new_state["solution_spec"] = llm_out["parsed"]
        if hasattr(llm_out, "tool_calls"):
            tool_log = run_tool_calls(llm_out, self.tool_maps)
            new_state["problem_diagnostic"].extend(tool_log)

        print("Optimizer:\n ")
        pprint.pprint(new_state["solution_spec"])
        return new_state 

    def explainer(self, state: OptimizerState) -> OptimizerState:
        new_state = state.copy()

        new_state["summary"] = StrOutputParser().invoke(
            self.llm.invoke([
                SystemMessage(content=self.explainer_prompt),
                HumanMessage(
                    content=state["problem"] + str(state["problem_spec"])
                ),
                *state["problem_diagnostic"],
            ])
        )

        print("Summary:\n")
        pprint.pprint(new_state["summary"])
        return new_state

    def _build_graph(self):
        self.add_node(self.extractor, "Problem Extractor")
        self.add_node(self.formulator, "Math Formulator")
        self.add_node(self.selector, "Solver Selector")
        self.add_node(self.generator, "Code Generator")
        self.add_node(self.verifier, "Verifier")
        self.add_node(self.explainer, "Explainer")
        self.add_node(self.tester, "Feasibility Tester")
        self.add_node(self.discretizer, "Discretizer")

        self.graph.add_edge(START, "Problem Extractor")
        self.graph.add_edge("Problem Extractor", "Math Formulator")
        self.graph.add_conditional_edges(
            "Math Formulator",
            should_discretize,
            {"discretize": "Discretizer", "continue": "Solver Selector"},
        )
        self.graph.add_edge("Discretizer", "Solver Selector")
        self.graph.add_edge("Solver Selector", "Code Generator")
        self.graph.add_edge("Code Generator", "Feasibility Tester")
        self.graph.add_edge("Feasibility Tester", "Verifier")
        self.graph.add_conditional_edges(
            "Verifier",
            should_continue,
            {"continue": "Optimizer", "error": "Problem Extractor"},
        )

        self.graph.add_edge("Explainer", END)

search_tool = DuckDuckGoSearchResults(output_format="json", num_results=10)


# A function to test if discretization is needed
def should_discretize(
    state: OptimizerState,
) -> Literal["Discretize", "continue"]:
    cons = state["problem_spec"]["constraints"]
    decs = state["problem_spec"]["decision_variables"]

    if any("infinite-dimensional" in t["tags"] for t in cons) or any(
        "infinite-dimensional" in t["type"] for t in decs
    ):
        # print(f"Problem has infinite-dimensional constraints/decision variables. Needs to be discretized")
        return "discretize"

    return "continue"


# Define the function that determines whether to continue or not
def should_continue(state: OptimizerState) -> Literal["error", "continue"]:
    spec = state["problem_spec"]
    try:
        status = spec["status"].lower()
    except KeyError:
        status = spec["spec"]["status"].lower()
    if "VERIFIED".lower() in status:
        return "continue"
    # Otherwise if there is, we continue
    else:
        return "error"


def main():
    model = ChatOpenAI(
        model="gpt-5-mini", max_tokens=10000, timeout=None, max_retries=2
    )
    execution_agent = OptimizationAgent(llm=model)
    # execution_agent = execution_agent.bind_tools(feasibility_checker)
    problem_string = """
    Solve the following optimal power flow problem
    System topology and data:
        - Three buses (nodes) labeled 1, 2 and 3.
        - One generator at each bus; each can only inject power (no negative output).
        - Loads of 1 p.u. at bus 1, 2 p.u. at bus 2, and 4 p.u. at bus 3.
        - Transmission lines connecting every pair of buses, with susceptances (B):
            - Line 1–2: B₁₂ = 10
            - Line 1–3: B₁₃ = 20
            - Line 2–3: B₂₃ = 30

    Decision variables:
        - Voltage angles θ₁, θ₂, θ₃ (in radians) at buses 1–3.
        - Generator outputs Pᵍ₁, Pᵍ₂, Pᵍ₃ ≥ 0 (in per-unit).

    Reference angle:
        - To fix the overall angle‐shift ambiguity, we set θ₁ = 0 (“slack” or reference bus).

    Objective:
        - Minimize total generation cost with
            - 𝑐1 = 1
            - 𝑐2 = 10
            - 𝑐3 = 100

    Line‐flow limits
        - Lines 1-2 and 1-3 are thermal‐limited to ±0.5 p.u., line 2-3 is unconstrained.

    In words:
    We choose how much each generator should produce (at non-negative cost) and the voltage angles at each bus (with bus 1 set to zero) so that supply exactly meets demand, flows on the critical lines don’t exceed their limits, and the total cost is as small as possible.
    Use the tools at your disposal to check if your formulation is feasible.
    """
    inputs = {"user_input": problem_string}
    result = execution_agent.invoke(inputs)
    print(result["messages"][-1].text)
    return result


if __name__ == "__main__":
    main()


