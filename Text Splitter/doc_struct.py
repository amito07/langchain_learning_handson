from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

code = '''def greet(name):
    return f"Hello, {name}! Welcome to Python."

users = ["Alice", "Bob", "Charlie"]

for user in users:
    message = greet(user)
    print(message)'''
spliter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=150,
    chunk_overlap=0
)

chunks = spliter.split_text(code)

print("Chunks ===> ", chunks)