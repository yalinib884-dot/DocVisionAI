# test_groq.py
from groq import Groq
from rich.console import Console
from rich.markdown import Markdown

# Create console for pretty printing
console = Console()

# Initialize Groq client
groq_client = Groq()  # or remove api_key if set in env

# Prompt to test Groq model
prompt = """
I need you to pick a business area that might be worth exploring for an Agentic AI opportunity.
Pick something that is not too obvious, but has potential.
Give me a short description of the area and why you think it is worth exploring.
"""

# Send the request to Groq API
response = groq_client.chat.completions.create(
    model="llama-3.1-8b-instant",   # ✅ Latest supported fast model
    messages=[
        {"role": "system", "content": "You are a helpful and creative business assistant."},
        {"role": "user", "content": prompt}
    ]
)

# Extract the model’s answer
answer = response.choices[0].message.content

# Print Markdown output beautifully
console.rule("[bold blue]Groq API Response[/bold blue]")
console.print(Markdown(answer))
console.rule("[bold blue]End of Response[/bold blue]")
