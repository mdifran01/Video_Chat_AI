from utils.audio_processor import process_input
from core.transcribe import transcribe_all
from core.document_processor import segment_to_doc,split_document
from core.vector_store import create_vector_store
from core.tools import set_vector_store
from core.agent import agent 

source = input("Enter YouTube URL or local audio path: ")

chunks = process_input(source)

all_segments = transcribe_all(
    chunks,
    source=source
)
documents = segment_to_doc(all_segments)

rag_chunks = split_document(documents)
vector_store = create_vector_store(rag_chunks)

set_vector_store(vector_store)



# ==================================================
# CHAT HISTORY
# ==================================================

chat_history = []


print("\n===================================")
print("       RAG CHAT ASSISTANT")
print("===================================")
print("Ask multiple questions.")
print("Type 'quit' to exit.")
print("===================================\n")


# ==================================================
# CONTINUOUS CHAT LOOP
# ==================================================

while True:

    question = input("\nAsk a question: ").strip()


    # ----------------------------------------------
    # EXIT CONDITION
    # ----------------------------------------------

    if question.lower() == "quit":
        print("\nChat ended.")
        break


    # Ignore empty questions
    if not question:
        continue


    # ----------------------------------------------
    # ADD USER QUESTION TO HISTORY
    # ----------------------------------------------

    chat_history.append(
        {
            "role": "user",
            "content": question
        }
    )


    # ----------------------------------------------
    # SEND HISTORY TO AGENT
    # ----------------------------------------------

    response = agent.invoke(
        {
            "messages": chat_history
        }
    )


    # ----------------------------------------------
    # UPDATE HISTORY
    # ----------------------------------------------

    chat_history = response["messages"]


    # ----------------------------------------------
    # GET FINAL ANSWER
    # ----------------------------------------------

    answer = response["messages"][-1].content


    print("\nAssistant:")
    print(answer)
    print()