import os 
from agents import Agent, Runner, trace, set_trace_processors
from phoenix.otel import register
from setup_openai import model

os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "http://localhost:6006"

set_trace_processors([]) # Disable default trace processors
# configure the Phoenix tracer
tracer_provider = register(
    project_name="agents",  # Default is 'default'
    auto_instrument=True,  # Auto-instrument your app based on installed dependencies
)


agent = Agent(model=model, name="Assistant", instructions="You are a helpful assistant")


async def main():
    with trace("Haiku Generator"):
        result = await Runner.run(
            agent, "Write a haiku about recursion in programming."
        )
        print(result.final_output)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
