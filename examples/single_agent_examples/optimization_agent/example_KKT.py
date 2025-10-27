import pprint

from langchain_openai import ChatOpenAI

from ursa.agents.optimization_agent import OptimizationAgent

model = ChatOpenAI(
    model="gpt-4o", max_tokens=10000, timeout=None, max_retries=2
)

problem_string = """
Here is an optimization problem with:
objective: minimize x^2+y^2-3xy
constraints: x+sin(y)=1

Extract the variables, parameters, and other relevant optimization problem information.
Write down the KKT conditions. Segregate them into dual-feasibility conditions, primal-feasibility
conditions, and complementarity slackness. Output in a structured JSON format. 
Additionally, write a latex file with the details in the folder:
examples/single_agent_examples/optimization_agent/out/
"""

execution_agent = OptimizationAgent(llm=model)

inputs = {"user_input": problem_string}


print("Started execution: \n")

result = execution_agent.invoke(inputs)
print("------------------------------------------\n")
print("------------------------------------------\n")
print("Output of the LLM:\n")
pprint.pprint(result)
print("------------------------------------------\n")
print("------------------------------------------\n")
print("Summary:\n")
print(f"{result['summary']}\n")


print("End execution\n")
