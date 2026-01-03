from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

def get_gemini_llm(model: str = "gemini-2.5-flash") -> ChatGoogleGenerativeAI:
    llm = ChatGoogleGenerativeAI(model=model)
    return llm

def get_chat_prompt(technology: str, sector: str) -> ChatPromptTemplate:
    chat_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(f"You are a professional blog writer and you will write blogs consciously. Blog would be written in short as like a paragraph about the topic provided by the user. The blog should be written about {technology}."),
        HumanMessagePromptTemplate.from_template(f"Write a blog on the topic of {technology} where instructions are {sector}."),
    ])

    return chat_prompt

if __name__ == "__main__":
    gemini_llm = get_gemini_llm()
    topic = input("Enter the blog topic: ")
    chat_history = []
    while True:
        user_input = input("Write instruction for blog (type 'exit' to quit):")
        if user_input.lower() == 'exit':
            break
        # Create initial system message and prompt
        chat_prompt = get_chat_prompt(topic, user_input)
        messages = chat_prompt.format_messages()

        # Add all previous conversation history
        messages.extend(chat_history)

        # Add current user message
        messages.append(HumanMessage(content=user_input))

        # Get response from LLM
        response = gemini_llm.invoke(messages)
        print("Blog Idea:", response.content)

         # Store the conversation in chat_history
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response.content))

        print("\n--- Conversation History ---")
        for msg in chat_history:
            print(f"{msg.type}: {msg.content}")