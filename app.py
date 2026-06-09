import streamlit as st
from graph import graph
from langchain_core.messages import HumanMessage


st.set_page_config(
    page_title="AI Workflow Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Workflow Assistant")

st.markdown(
    "Built using LangGraph + Groq + Streamlit"
)

with st.sidebar:

    st.header("Workflow")

    st.markdown("""
    1. Preprocess
    2. Sentiment Analysis
    3. Research
    4. Answer Generation
    5. Summary
    6. Chatbot
    """)

user_input = st.text_area(
    "Enter your query",
    height=150
)

if st.button("Run Workflow"):

    if not user_input.strip():

        st.warning("Please enter a query.")

    else:

        try:

            with st.spinner("Running AI Workflow..."):

                result = graph.invoke(
                    {
                        "messages": [
                            HumanMessage(
                                content=user_input
                            )
                        ]
                    }
                )

            col1, col2 = st.columns(2)

            with col1:

                st.subheader("📊 Sentiment")
                st.success(result["sentiment"])

                st.subheader("🔍 Research")
                st.write(result["research"])

            with col2:

                st.subheader("📝 Summary")
                st.info(result["summary"])

                st.subheader("💡 Final Answer")
                st.write(result["answer"])

            st.divider()

            st.subheader("🤖 Chatbot Response")

            st.write(
                result["messages"][-1].content
            )

        except Exception as e:

            st.error(f"Error: {str(e)}")
