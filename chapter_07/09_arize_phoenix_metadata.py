import os 
from agents import Agent, Runner, trace, set_trace_processors
from phoenix.otel import register
from setup_openai import model
from openinference.instrumentation import (
    using_metadata,
    using_session,
)

os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "http://localhost:6006"

set_trace_processors([]) # Disable default trace processors
# configure the Phoenix tracer
tracer_provider = register(
    project_name="agents",  # Default is 'default'
    auto_instrument=True,  # Auto-instrument your app based on installed dependencies
)


agent = Agent(model=model, name="Assistant", instructions="Always answer in a Haiku")


async def main():
    agent_input = dict(question="why is the sky blue?")
    metadata = dict(run_id="abc123", env="dev", customer_tier="pro", model=model)
    with (
        using_session("sess-42"),
        using_metadata(metadata)
    ):
        with trace("Haiku Generator"):
            result = await Runner.run(
                agent, str(agent_input)
            )
            print(result.final_output)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
