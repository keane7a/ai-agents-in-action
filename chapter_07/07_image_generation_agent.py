import asyncio 
import base64 
import os 
from setup_openai import model, client
from agents import Agent, ImageGenerationTool, Runner, result, trace, function_tool


def open_file(path: str) -> None: 
    os.startfile(path)

@function_tool
async def generate_image(prompt: str) -> str:
    """Generate an image return it as a base64 data URL."""
    response = await client.images.generate(
        model="gemini-3.1-flash-lite-image", 
        prompt=prompt,
        response_format="b64_json",
        n=1,
    )
    
    b64 = response.data[0].b64_json
    return f"data:image/png;base64,{b64}"
    

async def main(): 
    agent = Agent(
        model=model,
        name="Image generator",
        instructions="""
        ## Style Guidelines for All Images:

        - **Consistency**: Each image should maintain the same photographic quality with 3D-rendered elements seamlessly integrated
        - **Color Palette**: Use a consistent scheme of blues, purples, warm golds, and greens with pops of bright accent colors
        - **Lighting**: Professional photography lighting with dramatic but warm tones
        - **Infographics**: Semi-transparent holographic projections that don't overwhelm the main subjects
        - **Characters**: AI robots should be cute, approachable, and distinctly different from each other while maintaining a family resemblance
        - **Icons**: Use universally recognizable symbols (lightbulbs for ideas, gears for processing, hearts for alignment, etc.)
        - **Mood**: Optimistic, educational, and slightly futuristic without being cold or intimidating
        """,
        tools=[generate_image]
    )
    
    image_description= "An agent generating an image"
    image_name = "agent_image_generation"
    
    with trace("Image generation"): 
        result = await Runner.run(agent, input=image_description)
        print(result.final_output)
        for item in result.new_items:
            if (
                item.type == "tool_call_item"
                and item.raw_item.type == "image_generation_call"
                and (img_result := item.raw_item.result)
            ):
                os.makedirs("gen_images", exist_ok=True)
                image_path = os.path.join("gen_images", f"{image_name}.png")
                with open(image_path, "wb") as img_file:
                    img_file.write(base64.b64decode(img_result))
                open_file(image_path)


if __name__ == "__main__":
    asyncio.run(main())