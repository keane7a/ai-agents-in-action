import asyncio
from agents import Agent, RunContextWrapper, Runner 
from setup_openai import model

def get_reflexion_solver_instructions(run_context: RunContextWrapper[str], agent: Agent[str]) -> str:
    """
    Generates insturcitons for the reflexion solver agent. 
    """
    instructions = """
        You are a time-travel expert. Solve the problem step by step
        and be careful to avoild mistakes
    """
    
    return instructions + "\nHINT:\n" + run_context.context


# Base agent 
solver = Agent(
    model=model, 
    name="TimeTravelReflexion",
    instructions=get_reflexion_solver_instructions
)

critic = Agent(
    model=model,
    name="TimeTravelCritic", 
    instructions=(
        "You are an expert tutor. If the solution is wrong, explain the error "
        "and give a concise hint for improvement."
    )
)

# Problem spec
problem_1 = (
    "I left the eyar 2000 in a time machine, went forward 30 years,"
    "then back 40 years. I claim I'm now in 1990. Am I correct?"
)

problem_2 = """
In a sci-fi film, Alex is a time traveler who decides to go back in time
to witness a famous historical event that took place 100 years ago,
which lasted for 10 days. He arrives three days before the event starts.
However, after spending six days in the past, he jumps forward in time
by 50 years and stays there for 20 days. Then, he travels back to
witness the end of the end. 
How many days does Alex spend in the past before he sees the end of the event?
"""

TARGET_DAYS = "26"  # expected final answer
MAX_ATTEMPTS = 5  # fail-safe cap on retries


async def main(): 
    # Reflexion loop 
    feedback_hint = ""
    for attempt_no in range(1, MAX_ATTEMPTS + 1): 
        # Run the solver 
        result = await Runner.run(
            solver, input=problem_2, context=feedback_hint
        )
        answer = result.final_output.strip()
        print(f"\nAttempt {attempt_no}:\n{answer}")
        
        # correctness check
        has_correct_days = TARGET_DAYS in answer
        says_claim_correct = "yes" in answer.lower() or "correct" in answer.lower()
        solved = has_correct_days and says_claim_correct
        
        if solved: 
            print("solution accepted")
            break 
    
        # Not solved: Generate feedback & retry 
        feedback_prompt = f"""
            solution given: \n{answer}\n\n
            Expected final days: {TARGET_DAYS}\n
            Explain the error briefly and give helpful hint.
        """

        feedback_res = await Runner.run(critic, input=feedback_prompt)
        hint = feedback_res.final_output.strip()
        
        print(f"Feedback:\n{hint}")
        feedback_hint = hint
    
    else:
        print("Max atempts reached. Solution not found.") 
    
    
if __name__ == "__main__":
    asyncio.run(main())