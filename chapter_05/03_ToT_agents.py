import asyncio
from agents import Agent, Runner
from setup_openai import model 

# Agent generating next-steps
generator = Agent(
    model=model,
    name="ToT-Generator", 
    instructions="Given the current situation, brainstorm a possible next step or action to reach the goal."
)

# Agent for evaluating partial solutions
evaluator = Agent(
    model=model,
    name="ToT-Evaluator", 
    instructions="Assess how likely the proposed plan will solve the problem. Respond with 'promising' or 'unlikely'"
)

problem = """
    You need to reach the year 1800 from 2025 using a time machine that can jump either -100 or -30 years. 
"""

async def main(): 
    # generate initial thought candidates 
    initial_thoughts = []
    for i in range(3): 
        res = await Runner.run(
            generator, input=f"Problem: {problem}\n Think of a first step."
        )
        initial_thoughts.append(res.final_output.strip())
        
    # Evaluate and expand each thought 
    promising_branches = []
    for thought in initial_thoughts: 
        eval_res = await Runner.run(
            evaluator, input=f"Plan: {thought}\nIs this promising?"
        )

        if "promising" in eval_res.final_output.lower(): 
            # Expand this thought
            next_step = await Runner.run(
                generator, input=f"Current idea: {thought}\nNext step?"
            )
            promising_branches.append(f"{thought} -> {next_step.final_output.strip()}")
            
    print("Initial thought candidates:", initial_thoughts)
    print(
        "Expanded promising branch:", 
        promising_branches[0] if promising_branches else "None"
    )
    

if __name__ == "__main__":
    asyncio.run(main())