from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

rag_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a helpful legal assistant. Always prioritize the given CONTEXT when answering. "
        "If the user asks something not covered in the context, you may use your general knowledge "
        "to provide a clear and helpful answer. "
        "If it's a casual message (like greetings or small talk), respond naturally.\n\n"
        "CONTEXT:\n{context}"
    )),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])


