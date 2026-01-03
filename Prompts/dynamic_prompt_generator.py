from langchain_core.prompts import PromptTemplate

def get_prompt() -> PromptTemplate:
    template = "Write a topic on {topic} in a {style} style."
    prompt =  PromptTemplate(template=template, input_variables=["topic", "style"])
    return prompt


if __name__ == "__main__":
    prompt = get_prompt()
    print(prompt.format(topic="Ai in daily life", style="informal"))
    print(prompt.format(topic="Ai in health", style="normal"))
