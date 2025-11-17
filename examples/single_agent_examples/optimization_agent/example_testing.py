import pprint

from langchain_openai import ChatOpenAI

from ursa.agents.optimization_agent import OptimizationAgent

model = ChatOpenAI(
    model="gpt-5", max_tokens=10000, timeout=None, max_retries=2
)

problem_string = """
Consider a statistical testing strategy problem. We have to test products by choosing
the test parameter values such that, 
a) at least 95 percent of the parameter space is covered (one-sided coverage).
b) test parameter values are within reasonable energy limits. In other words, if x_1 to x_N are
test parameters, the energy associated with this test is given by summation of exp(x_i) for all i. 

Given the N-dimensional distribution of x's, come up with the region that satisfies requirement (a).
Moreover, provide the candidate test parameter with highest likelihood and minimum energy on the 
boundary of this region.
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
