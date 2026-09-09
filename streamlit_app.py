import streamlit as st

from utils.audio_processor import process_input
from core.transcribe import transcribe_all
from core.document_processor import segment_to_doc, split_document
from core.vector_store import create_vector_store
from core.tools import set_vector_store
from core.agent import agent


# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 RAG Chat Assistant")
st.write("Ask questions about your YouTube video or audio file.")


# ==================================================
# SESSION STATE
# ==================================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None


# ==================================================
# INPUT SOURCE
# ==================================================

source = st.text_input(
    "Enter YouTube URL or local audio path:"
)


# ==================================================
# PROCESS BUTTON
# ==================================================

if st.button("Process Audio"):

    if not source:
        st.warning("Please enter a YouTube URL or audio file path.")

    else:

        with st.spinner("Processing audio..."):

            # Download / convert / chunk audio
            chunks = process_input(source)

            # Transcribe
            all_segments = transcribe_all(
                chunks,
                source=source
            )

            # Convert to documents
            documents = segment_to_doc(all_segments)

            # Split documents
            rag_chunks = split_document(documents)

            # Create vector store
            vector_store = create_vector_store(rag_chunks)

            # Set vector store for tool
            set_vector_store(vector_store)

            # Save in session
            st.session_state.vector_store = vector_store

        st.success("Audio processed successfully! You can now ask questions.")


# ==================================================
# CHAT HISTORY DISPLAY
# ==================================================

for message in st.session_state.chat_history:

    if message.type == "human":
        role = "user"

    elif message.type == "ai":
        role = "assistant"

    else:
        continue

    with st.chat_message(role):
        st.write(message.content)
        
# ==================================================
# CHAT HISTORY DISPLAY
# ==================================================

for message in st.session_state.chat_history:

    if message.type == "human":
        role = "user"

    elif message.type == "ai":
        role = "assistant"

    else:
        continue

    with st.chat_message(role):
        st.write(message.content)


# ==================================================
# CHAT INPUT
# ==================================================

question = st.chat_input("Ask a question...")


if question:

    if st.session_state.vector_store is None:

        st.warning("Please process a YouTube URL or audio file first.")

    else:

        with st.chat_message("user"):
            st.write(question)

        # Add new question
        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": question
            }
        )

        # Send conversation to agent
        response = agent.invoke(
            {
                "messages": st.session_state.chat_history
            }
        )

        # Store LangChain messages
        st.session_state.chat_history = response["messages"]

        # Get final answer
        answer = response["messages"][-1].content

        with st.chat_message("assistant"):
            st.write(answer)       
            
            
            