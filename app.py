import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent
from langchain.callbacks import StreamlitCallbackHandler

# -------------------------------------------------
# Page configuration & global styles
# -------------------------------------------------
st.set_page_config(
    page_title="Math + Wiki Assistant",
    page_icon="🧮",
    layout="wide",
)

st.markdown(
    """
    <style>
        .main {
            background: linear-gradient(120deg, #f0f4ff, #ffffff);
        }
        header.st-emotion-cache-18ni7ap.ezrtsby0 {text-align:center;}
        .stButton>button {
            background-color:#4f46e5;
            color:white;
            border:none;
            border-radius:8px;
            padding:0.6em 1.4em;
            font-weight:600;
            transition:all 0.3s ease-in-out;
        }
        .stButton>button:hover {transform:scale(1.03);}
        .stChatMessage .stMarkdown {
            background: #f9fafb;
            color: black;
            border-radius:12px;
            padding:0.7em;
        }
        .stChatMessage.user .stMarkdown {
            background: #e0ecff;
        }
        textarea {
            border-radius:10px !important;
            background-color: #f8fafc !important;
            color: #111827 !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------
# Title & Sidebar
# -------------------------------------------------
st.title("🧮 Text‑to‑Math & Data Search Assistant")

st.sidebar.header("🔑 API Credentials")

groq_api_key = st.sidebar.text_input("Groq API Key", type="password")

st.sidebar.markdown("---")
st.sidebar.caption("Made with ❤️ & Streamlit • v1.0")

if not groq_api_key:
    st.info("Please add your Groq API key to continue ⬅️")
    st.stop()

# -------------------------------------------------
# LLM & Tools setup
# -------------------------------------------------
llm = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)

wikipedia_tool = Tool(
    name="Wikipedia",
    func=WikipediaAPIWrapper().run,
    description="Search Wikipedia for quick topic overviews",
)

math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
calculator_tool = Tool(
    name="Calculator",
    func=math_chain.run,
    description="Solve mathematical expressions",
)

prompt_template = PromptTemplate(
    input_variables=["question"],
    template="""
You are a helpful assistant for solving users' mathematical questions.
Please show your logical steps and detailed working using bullet points or numbered steps.
Question: {question}
Answer:
""",
)

chain = LLMChain(llm=llm, prompt=prompt_template, verbose=True)
reasoning_tool = Tool(
    name="Reasoning",
    func=chain.run,
    description="Logic‑based reasoning helper",
)

assistant_agent = initialize_agent(
    tools=[wikipedia_tool, calculator_tool, reasoning_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
)

# -------------------------------------------------
# Chat session state
# -------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "👋 Hi, I'm your Math & Knowledge bot! Ask me anything."}
    ]
if "question" not in st.session_state:
    st.session_state.question = ""

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------------------------------------------------
# Main Interaction UI
# -------------------------------------------------
with st.expander("💡 Need inspiration? Click to view an example question."):
    st.write(
        "I have 5 bananas and 7 grapes. I eat 2 bananas and give away 3 grapes. Then I buy a dozen apples and 2 packs of blueberries. Each pack of blueberries contains 25 berries. How many total pieces of fruit do I have at the end?"
    )

st.session_state.question = st.text_area(
    "Enter your question here 👇",
    value=st.session_state.question,
    placeholder="Type or paste your math/knowledge question...",
    height=120,
)

col1, col2 = st.columns([1, 5])
with col1:
    submit = st.button("🚀 Solve", use_container_width=True)
with col2:
    clear = st.button("🧹 Clear Chat", use_container_width=True)

if clear:
    st.session_state.messages = []
    st.session_state.question = ""
    st.rerun()

if submit:
    if st.session_state.question.strip():
        with st.spinner("Thinking..."):
            st.session_state.messages.append({"role": "user", "content": st.session_state.question})
            with st.chat_message("user"):
                st.markdown(st.session_state.question)

            cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True)
            response = assistant_agent.run(st.session_state.question, callbacks=[cb])

            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)

            st.balloons()
    else:
        st.warning("Please enter a question first!")
