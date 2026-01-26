# lammps_tool.py
from __future__ import annotations

import os
import tempfile
from typing import Any, Optional

from langchain_core.tools import tool

from ..agents.lammps_agent import LammpsAgent  # <-- adjust import

WORKSPACE_ROOT = "./workspace_lammps"

@tool("run_lammps_tool")
def run_lammps_tool(
    llm,
    simulation_task: str,
    elements: list[str],
    template: Optional[str] = None,
    recursion_limit: int = 999999,
    workspace: Optional[str] = None,
) -> dict[str, Any]:
    """Run the LammpsAgent graph and return final state + key outputs."""
    
    if workspace is None:
        os.makedirs(WORKSPACE_ROOT, exist_ok=True)
        workspace = tempfile.mkdtemp(dir=WORKSPACE_ROOT)
    else:
        os.makedirs(workspace, exist_ok=True)

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

    inputs = {
        "simulation_task": simulation_task,
        "elements": elements,
        "template": template if template is not None else "No template provided.",
    }

    final_state = agent._invoke(inputs, recursion_limit=recursion_limit)


    if final_state.get("run_returncode") == 0:
            print("\n Lammps run successful. Now parsing the output.")

            #executor = ExecutionAgent(llm=self.llm.with_structured_output(SolutionSpec))
            #exe_plan = f"""
            #You are part of a larger scientific workflow whose purpose is to accomplish this task: {simulation_task}
            
            #A LAMMPS simulation has been done and the output is located in the file 'log.lammps'.
            
            #Extract the Yield Strength from the simulations log. Do not return any other information.
            #"""
            yieldC = execute_conv(workspace+"/log.lammps")


    return {
        yieldC
    }
