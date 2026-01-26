import os
import pprint
import subprocess
import json
import operator
from typing import Annotated, Any, Literal, Mapping, TypedDict

# import langchain objects
from langchain.chat_models import BaseChatModel
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import InjectedState

# import relevant prompts
from ursa.prompt_library.optimization_prompts import *

# import relevant tools, functions, and schema
from ursa.tools.write_code_tool import write_code
from ursa.tools.run_command_tool import run_command
from ursa.tools.lammps_tool import run_lammps_tool 
from ursa.util.helperFunctions import extract_tool_calls, run_tool_calls
from ursa.util.optimization_schema import ProblemSpec, SolverSpec, SolutionSpec, NotesSpec

from ursa.agents.workspace_lammps.conv import execute_conv
from ursa.agents.workspace_lammps.predictor import predictor


from base import BaseAgent
from ursa.agents import LammpsAgent, ExecutionAgent

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
    diagnostic: Annotated[list[dict], operator.add]
    data: list[list[Any]]

class OptimizationAgent(BaseAgent):
    def __init__(
        self,
        llm: BaseChatModel,
        *args,
        **kwargs,
    ):
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
        self.tools = [write_code, run_command, run_lammps_tool, predictor]  # [run_cmd, write_code, search_tool, fca, oc]
        self.llm = self.llm.bind_tools(self.tools)
        self.tool_maps = {
            (getattr(t, "name", None) or getattr(t, "__name__", None)): t
            for i, t in enumerate(self.tools)
        }

        self._action = self._build_graph()

    # Define the function that calls the model
    def extractor(self, state: OptimizerState) -> OptimizerState:
        '''
        This stage only parses user input into a problem statement in the LLM.
        Additionally, if this verifier has some corrections, that is considered
        along with user input while generating the problem statement text.
        '''

        notes: NotesSpec = state.get("notes", {})

        # Build the messages depending on whether we have verifier info
        if "verifier" in notes and notes.get("verifier_explanation"):
            messages = [
                SystemMessage(content=self.extractor_prompt),
                SystemMessage(content=notes["verifier_explanation"]),  # <- field from Notes
                HumanMessage(content=state["user_input"]),
            ]
        else:
            messages = [
                SystemMessage(content=self.extractor_prompt),
                HumanMessage(content=state["user_input"]),
            ]

        notes["description"] = self.llm.invoke(messages).content

        diagnostic = state.get("diagnostic", [])

        print("Extractor:\n")
        # pprint.pprint(notes)
        
        data = []

        return {
            "notes": notes,
            "diagnostic": diagnostic,
            "data": data,
        }

    def formulator(self, state: OptimizerState) -> OptimizerState:
        '''
        This takes the problem statement and generates a structured output 
        prescribed by the ProblemSpec structure.
        '''

        problem = self.llm.with_structured_output(
            ProblemSpec).invoke([
            SystemMessage(content=self.math_formulator_prompt),
            HumanMessage(content=state["user_input"]),
            SystemMessage(content=state["notes"]["description"]),
        ])
        
        problem["feasibility"] = ""

        print("Formulator:\n")
        # pprint.pprint(problem)
        return {
            "problem": problem,
        }   

    def discretizer(self, state: OptimizerState) -> OptimizerState:
        '''
        This discretizes the problem if the original optimization problem was
        infinitedimensional.
        '''
        problem = self.llm.with_structured_output(
            ProblemSpec).invoke([
            SystemMessage(content=self.discretizer_prompt),
            HumanMessage(content=str(state["problem"])),
        ])
        problem["feasibility"] = ""

        print("Discretizer:\n")
        # pprint.pprint(problem)

        return {
            "problem": problem,
        }

    def tester(self, state: OptimizerState) -> OptimizerState:
        '''
        This step verifies if the formulated problem is mathematically sound
        by searching for the existence of a feasible solution.
        '''
        problem: ProblemSpec = state.get("problem", {})

        llm_out = self.llm.invoke([
            SystemMessage(content=self.feasibility_prompt),
            SystemMessage(content=str(state["problem"])),
        ])

        tool_log = run_tool_calls(llm_out, self.tool_maps)
        
        problem["feasibility"] = self.llm.invoke([
            SystemMessage(content="Read the tool log and return just one word: 'Feasible' or 'Infeasible'."),
            SystemMessage(content=str(tool_log)),
        ]).content

        print("Feasibility Tester:\n")
        # for msg in tool_log:
            # msg.pretty_print()
        return {
            "problem": problem,
            "diagnostic": tool_log,
        }

    def selector(self, state: OptimizerState) -> OptimizerState:
        '''
        Given ProblemSpec with a feasible solution, this decides the best
        solver to solve the problem.
        '''
        
        solver: SolverSpec = state.get("solver", {})

        solver = self.llm.with_structured_output(
            SolverSpec).invoke([
            SystemMessage(content=self.solver_selector_prompt),
            SystemMessage(content=str(state["problem"])),
        ])
        
        print("Selector:\n ")
        # pprint.pprint(solver)
        return {
            "solver": solver,
        }

    def generator(self, state: OptimizerState) -> OptimizerState:
        '''
        Generates code to solve the problem.
        '''
        data = state.get("data", [])

        code_prompt = '''Generate a candidate composition for the high entropy alloy with elements
        Co-Cr-Fe-Mn-Ni. Return only one candidate as a python list and no other output. Use the predictor tool whenever possible. Last entry should be predicted yield for the candidate composition. 
        Only return a list of numbers, compositions of elements in prescribed order followed by last entry, the predicted yield.
        '''

        composition = self.llm.bind(tool_choice="required").invoke([
            SystemMessage(content=code_prompt),
            SystemMessage(content=str(state["solver"])),
            HumanMessage(content=str(state["problem"])),
            SystemMessage(content=str(state["data"])),
        ])


        tool_log = run_tool_calls(composition, self.tool_maps)
        # pprint.pprint(tool_log)

        composition = [float(v) for v in json.loads(tool_log[-1].content)]
        # pprint.pprint(composition)

        # composition = float(self.tool_maps["predictor"].invoke(state["data"]).content)
        # pprint.pprint(composition)


        print("Candidate composition:\n")
        # pprint.pprint(composition)

        data.append(composition)

        # pprint.pprint(data)

        return {
                "data": data,
        }

    def verifier(self, state: OptimizerState) -> OptimizerState:
        
        notes: NotesSpec = state.get("notes", {})

        #llm_out = self.llm.invoke([
         #   SystemMessage(content="Read the data and decide if a composition whose "),
          #  HumanMessage(content=str(state["problem"])),
        #])

        #notes =llm_out["parsed"]

        print("Verifier:\n ")
        # pprint.pprint(notes["verifier"])
        # pprint.pprint(notes["verifier_explanation"])
        return {
            "notes": notes
        }

    def optimizer(self, state:OptimizerState) -> OptimizerState:
        
        workspace = "./workspace_lammps"

        with open("src/ursa/agents/workspace_lammps/template.txt", "r") as file:
            template = file.read()

        #simulation_task = "Carry out a LAMMPS simulation of the high entropy alloy Co-Cr-Fe-Mn-Ni to determine its yield strength. Composition percentages are in the following statement."
        
        simulation_task = """Carry out a LAMMPS simulation of the high entropy alloy Co-Cr-Fe-Mn-Ni to determine the following quantities: (1) Isotropic_Youngs_modulus."""
         # Make tool calls using availabe tools. The tool requires simulation task, composition data, and template data as inputs."""
        # pprint.pprint(simulation_task+str(state["data"][-1]))

        elements = ["Co", "Cr", "Fe", "Mn", "Ni"]

        agent = LammpsAgent(
            llm=llm,
            max_potentials=5,
            max_fix_attempts=15,
            find_potential_only=False,
            mpi_procs=8,
            workspace=workspace,
            lammps_cmd="lmp_mpi",
            mpirun_cmd="mpirun", 
        )

        if picked_potential is not None:
            inputs = {
            "simulation_task": simulation_task,
            "elements": elements,
            "template": template if template is not None else "No template provided.",
            "chosen_potential": picked_potential
            }    
        else:
            inputs = {
            "simulation_task": simulation_task,
            "elements": elements,
            "template": template if template is not None else "No template provided.",
         }

        # inputs = {
        #     "simulation_task": simulation_task,
        #     "elements": elements,
        #     "template": template if template is not None else "No template provided.",
        # }

        final_state = agent._invoke(inputs, recursion_limit=recursion_limit)

        picked_potential = getattr(final_state,"chosen_potential")


        if final_state.get("run_returncode") == 0:
            print("\n Lammps run successful. Now parsing the output.")

            #executor = ExecutionAgent(llm=self.llm.with_structured_output(SolutionSpec))
            #exe_plan = f"""
            #You are part of a larger scientific workflow whose purpose is to accomplish this task: {simulation_task}
            
            #A LAMMPS simulation has been done and the output is located in the file 'log.lammps'.
            
            #Extract the Yield Strength from the simulations log. Do not return any other information.
            #"""
            yield_strength = execute_conv(workspace+"/log.lammps")


        # final_lammps_state = self.llm.invoke([
        #     SystemMessage(content=simulation_task),
        #     HumanMessage(content="Composition is: "+ str(state["data"][-1][:5])),
        #     HumanMessage(content="Template data is: " + str(template)),
        #     ]
        # )
        # tool_log = run_tool_calls(final_summary, self.tool_maps) 
        
        # yield_strength = self.llm.invoke([
        #     SystemMessage(content="Read the tool log and return just the calculated yield_strength number."),
        #     SystemMessage(content=str(tool_log)),
        # ]).content

        print("Optimizer:\n ")
        pprint.pprint("Candidate Composition:"+str(state["data"][-1][:5])+"\n")
        pprint.pprint("Lammps Output"+ str(yield_strength)+"\n"+"Optimizer Guess:" + str(state["data"][-1][-1]))

        notes: NotesSpec = state.get("notes",{})

        if abs(state["data"][-1][-1] - yield_strength) < 1e-1:
            state["data"][-1][-1] = yield_strength
            notes["verifier"] = "Solved"
            return {"data": data, "notes": notes}
        else:            
            notes["verifier"] = ""
            return {"data": data, "notes": notes} 

    def explainer(self, state: OptimizerState) -> OptimizerState:
        new_state = state.copy()

        new_state["solution"] = self.llm.with_structred_output(SolutionSpec).invoke([
            SystemMessage(content=self.explainer_prompt),
            SystemMessage(content=str(state["problem"])), 
            SystemMessage(content=str(state["data"])),
            SystemMessage(content=str(state["notes"])),
        ]).content

        print("Summary:\n")
        pprint.pprint(new_state["solution"])
        return {
            "notes": new_state["solution"]
        }

    def _build_graph(self):
        graph = StateGraph(OptimizerState)

        self.add_node(graph, self.extractor, "Problem Extractor")
        self.add_node(graph, self.formulator, "Math Formulator")
        self.add_node(graph, self.discretizer, "Discretizer")
        self.add_node(graph, self.tester, "Feasibility Tester")
        self.add_node(graph, self.verifier, "Verifier")
        
        self.add_node(graph, self.selector, "Solver Selector")
        self.add_node(graph, self.generator, "Generator")
        self.add_node(graph, self.optimizer, "Optimizer")

        self.add_node(graph, self.explainer, "Explainer")

        
        graph.add_edge(START, "Problem Extractor")
        graph.add_edge("Problem Extractor", "Math Formulator")
        graph.add_conditional_edges(
            "Math Formulator",
            should_discretize,
            {"discretize": "Discretizer", "continue": "Feasibility Tester"},
        )
        graph.add_edge("Discretizer", "Feasibility Tester")
        # graph.add_edge("Feasibility Tester", "Generator")
        # graph.add_edge("Generator", "Optimizer")
        
        graph.add_edge("Feasibility Tester", "Solver Selector")
        graph.add_edge("Solver Selector", "Generator")
        graph.add_edge("Generator", "Optimizer")
        # graph.add_edge("Verifier", "Optimizer")
        
        graph.add_edge("Optimizer", "Verifier")
        graph.add_conditional_edges(
            "Verifier",
            should_continue,
            {"continue": "Explainer", "redo": "Generator"},
        )
        graph.add_edge("Explainer", END)

        compiled_graph = graph.compile()
        # try:
        #     import matplotlib.pyplot as plt
        #     import matplotlib.image as mpimg 
        #     import io
        #     png_bytes = compiled_graph.get_graph().draw_mermaid_png()
        #     img = mpimg.imread(io.BytesIO(png_bytes), format='png')  # decode bytes -> array

        #     plt.imshow(img)
        #     plt.axis('off')
        #     plt.show()
        # except Exception as e:
        #     # This requires some extra dependencies and is optional
        #     print(e)
        #     pass

        return compiled_graph

    def _invoke(
        self, inputs: Mapping[str, Any], recursion_limit: int = 100000, **_
    ):
        config = self.build_config(
            recursion_limit=recursion_limit, tags=["graph"]
        )
        if "user_input" not in inputs:
            try:
                inputs["user_input"] = inputs["messages"][0].content
            except KeyError:
                raise ("'user_input' is a required argument")

        return self._action.invoke(inputs, config)


# @tool
# def run_cmd(query: str, state: Annotated[dict, InjectedState]) -> str:
#     """
#     Run a commandline command from using the subprocess package in python

#     Args:
#         query: commandline command to be run as a string given to the subprocess.run command.
#     """
#     workspace_dir = state["workspace"]
#     print("RUNNING: ", query)
#     try:
#         process = subprocess.Popen(
#             query.split(" "),
#             stdout=subprocess.PIPE,
#             stderr=subprocess.PIPE,
#             text=True,
#             cwd=workspace_dir,
#         )

#         stdout, stderr = process.communicate(timeout=60000)
#     except KeyboardInterrupt:
#         print("Keyboard Interrupt of command: ", query)
#         stdout, stderr = "", "KeyboardInterrupt:"

#     print("STDOUT: ", stdout)
#     print("STDERR: ", stderr)

#     return f"STDOUT: {stdout} and STDERR: {stderr}"


# @tool
# def write_code(
#     code: str, filename: str, state: Annotated[dict, InjectedState]
# ) -> str:
#     """
#     Writes python or Julia code to a file in the given workspace as requested.

#     Args:
#         code: The code to write
#         filename: the filename with an appropriate extension for programming language (.py for python, .jl for Julia, etc.)

#     Returns:
#         Execution results
#     """
#     workspace_dir = state["workspace"]
#     print("Writing filename ", filename)
#     try:
#         # Extract code if wrapped in markdown code blocks
#         if "```" in code:
#             code_parts = code.split("```")
#             if len(code_parts) >= 3:
#                 # Extract the actual code
#                 if "\n" in code_parts[1]:
#                     code = "\n".join(code_parts[1].strip().split("\n")[1:])
#                 else:
#                     code = code_parts[2].strip()

#         # Write code to a file
#         code_file = os.path.join(workspace_dir, filename)

#         with open(code_file, "w") as f:
#             f.write(code)
#         print(f"Written code to file: {code_file}")

#         return f"File {filename} written successfully."

#     except Exception as e:
#         print(f"Error generating code: {str(e)}")
#         # Return minimal code that prints the error
#         return f"Failed to write {filename} successfully."


search_tool = DuckDuckGoSearchResults(output_format="json", num_results=10)
# search_tool = TavilySearchResults(max_results=10, search_depth="advanced", include_answer=True)


# A function to test if discretization is needed
def should_discretize(
    state: OptimizerState,
) -> Literal["Discretize", "continue"]:
    cons = state["problem"]["constraints"]
    decs = state["problem"]["decision_variables"]

    if any("infinite-dimensional" in t["tags"] for t in cons) or any(
        "infinite-dimensional" in t["type"] for t in decs
    ):
        # print(f"Problem has infinite-dimensional constraints/decision variables. Needs to be discretized")
        return "discretize"

    return "continue"


# Define the function that determines whether to continue or not
def should_continue(state: OptimizerState) -> Literal["error", "redo"]:
    spec = state["notes"]
    status = spec["verifier"].lower()

    if "SOLVED".lower() in status:
        return "continue"
    else:
        # Otherwise if there is, we continue
        return "redo"


def main():
    model = ChatOpenAI(
        model="gpt-5-mini", max_tokens=10000, timeout=None, max_retries=2
    )
    execution_agent = OptimizationAgent(llm=model)
    
    problem_string = """
    Find the composition of a high entropy alloy using the elements Co-Cr-Fe-Mn-Ni such that the
    Yield Strength is maximized. Minimum composition possible must be 5%.
    """
    inputs = {"user_input": problem_string}
    result = execution_agent.invoke(inputs)
    print(result["summary"])
    return result


if __name__ == "__main__":
    main()


#         min⁡ 𝑃𝑔  𝑐1*𝑃1 + 𝑐2 * 𝑃2 + 𝑐3 * 𝑃3
