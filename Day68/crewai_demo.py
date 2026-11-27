
from crewai import Agent, Task, Crew, Process, LLM

# 1. Connect local Ollama model
llm = LLM(
    model="ollama/llama3.2:1b",
    base_url="http://localhost:11434",
    temperature=0.2
)

# 2. Create Researcher agent
researcher = Agent(
    role="Researcher",
    goal="Find important information about the given topic",
    backstory="You are good at researching technical topics.",
    llm=llm,
    verbose=True
)

# 3. Create Writer agent
writer = Agent(
    role="Technical Writer",
    goal="Write a clear and simple report using research",
    backstory="You explain complex topics in simple language.",
    llm=llm,
    verbose=True
)

# 4. Create Reviewer agent
reviewer = Agent(
    role="Reviewer",
    goal="Check the report and improve its accuracy",
    backstory="You are an expert editor who checks clarity and correctness.",
    llm=llm,
    verbose=True
)

# 5. Create Research task
research_task = Task(
    description="Research the topic: Artificial Intelligence. "
                "Provide its meaning, applications and examples.",
    expected_output="A structured research summary.",
    agent=researcher
)

# 6. Create Writing task
writing_task = Task(
    description="Write a simple report using the research provided. "
                "Include an introduction, applications and conclusion.",
    expected_output="A clear report about Artificial Intelligence.",
    agent=writer,
    context=[research_task]
)

# 7. Create Review task
review_task = Task(
    description="Review the report. Correct mistakes and improve clarity.",
    expected_output="A polished final report.",
    agent=reviewer,
    context=[writing_task]
)

# 8. Create Crew
crew = Crew(
    agents=[researcher, writer, reviewer],
    tasks=[research_task, writing_task, review_task],
    process=Process.sequential,
    verbose=True
)

# 9. Execute Crew
result = crew.kickoff()

print("\nFINAL REPORT:\n")
print(result)
