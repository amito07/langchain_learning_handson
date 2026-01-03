from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

def get_chat_prompt(technology: str, sector: str) -> ChatPromptTemplate:
    chat_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(f"You are a helpful assistant knowledgeable about {technology}."),
        HumanMessagePromptTemplate.from_template(f"Explain the impact of {technology} in the {sector} sector."),
    ])

    return chat_prompt

if __name__ == "__main__":
    chat_prompt = get_chat_prompt("Artificial Intelligence", "healthcare")
    print(chat_prompt.format_messages())
