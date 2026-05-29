import sqlite3
from pathlib import Path

from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
from langgraph.checkpoint.sqlite import SqliteSaver

from ursa.experimental.agents.simulator_agent import SimulatorAgent

problem = (
    "There is matlab library called PIETOOLS in your workspace. PIETOOLS defined a parser for simulation, analysis, estimation, and control of linear PDEs."
    "Your task is to figure out the way to run these codes, write and execute a matlab file to simulate an unstable reaction-diffusion equation and then design an $H_inf$ optimal controller that stabilizes the same PDE."
    "Compile your understanding of the library into a user guide of your own and save it in your workspace for future use. Execute using command matlab commands to check if everything works."
    "Run the open-loop and closed-loop simulation. Save the plots."
    "Write a second script, that does a sensitivity analysis of stability versus the reaction parameter in the reaction-diffusion equation."
    "Additional information: Sedumi solver can also be found in your workspace."
)
workspace = "ursa_workspace"

db_path = Path(workspace) / "checkpoint.db"
db_path.parent.mkdir(parents=True, exist_ok=True)
conn = sqlite3.connect(str(db_path), check_same_thread=False)
checkpointer = SqliteSaver(conn)

embedding = init_embeddings("openai:text-embedding-3-small")
model = init_chat_model(model="openai:gpt-5.2")

simulator = SimulatorAgent(
    llm=model,
    workspace=workspace,
    embedding=embedding,
    checkpointer=checkpointer,
    thread_id="dcopf_test_executor",
    use_web=True,
)

result = simulator.invoke(problem)
