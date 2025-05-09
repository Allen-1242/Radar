import streamlit as st
import pandas as pd
import seaborn as sns
from core.prompt_engine import prompt_to_json
from core.chart_render import render_chart
from core.vector_store import VectorStore

import openai, streamlit as st

openai.api_key = st.secrets["OPENAI_API_KEY"]

def inject_rag_context(user_prompt: str, rag_context: str) -> str:
    return f"""You are a dashboard assistant. Use the following data context to interpret the request.

DATA CONTEXT:
{rag_context}

USER PROMPT:
{user_prompt}"""

# --- Data loading and dynamic schema generation ---
uploaded_file = st.file_uploader("Upload your CSV", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = sns.load_dataset("iris")

SCHEMA = {
    "columns": [
        {"name": col, "dtype": str(df[col].dtype)}
        for col in df.columns
    ]
}

# --- Prepare schema chunks for vector store ---
schema_chunks = [
    f"Column: {col} | Type: {dtype} | Unique: {df[col].nunique()}"
    + (
        f" | Min: {df[col].min()} | Max: {df[col].max()}"
        if pd.api.types.is_numeric_dtype(df[col]) else ""
    )
    for col, dtype in zip(df.columns, df.dtypes)
]

@st.cache_resource
def get_vector_store(chunks):
    vs = VectorStore()
    vs.build(chunks)
    return vs

vs = get_vector_store(schema_chunks)

# --- Initialize session state for tiles ---
if "tiles" not in st.session_state:
    st.session_state.tiles = [None, None, None, None]

# --- Tabs for Dashboard and Data View ---
tab1, tab2 = st.tabs(["Dashboard", "Dataset View"])

with tab1:
    if st.button("Clear All Tiles"):
        st.session_state.tiles = [None, None, None, None]

    st.title("RADAR: Conversational Dashboard Builder")
    user_prompt = st.text_input(
        "Ask for a chart (e.g., 'Show me a bar chart of petal length per species')"
    )

    # Build + retrieve RAG context
    if st.button("Generate Chart") and user_prompt:
        try:
            tile_state = [i for i, t in enumerate(st.session_state.tiles) if t is not None]
            retrieved = vs.query(user_prompt, top_k=5)
            rag_context = "\n".join(retrieved)
            rag_prompt = inject_rag_context(user_prompt, rag_context)

            # Debug: full prompt
            with st.expander("View full LLM prompt"):
                st.code(rag_prompt, language="text")

            json_config = prompt_to_json(rag_prompt, SCHEMA, tile_state)
            if json_config.get("filter") in ["NULL","Null","none","None"]:
                json_config["filter"] = None

            fig = render_chart(df, json_config)
            action = json_config.get("action","add_chart")
            tile_id = json_config.get("target_tile","next_available")

            if action=="add_chart":
                for i in range(4):
                    if st.session_state.tiles[i] is None:
                        st.session_state.tiles[i] = fig
                        break
            elif action=="update_chart":
                if isinstance(tile_id,int) and 0<=tile_id<4:
                    st.session_state.tiles[tile_id] = fig
                else:
                    st.error(f"Invalid tile ID: {tile_id}")
            else:
                st.error(f"Unknown action: {action}")

        except Exception as e:
            st.error(f"Error: {e}")

    dashboard_title = st.text_input("Dashboard Title", value="Dashboard View")
    st.markdown(f"<h2 style='text-align: center'>{dashboard_title}</h2>", unsafe_allow_html=True)

    # --- Render the 2×2 grid with inner dividers as borders ---
    col1, col2 = st.columns(2)
    for i, col in enumerate([col1, col2, col1, col2]):
        with col:
            # Build per-tile style
            base = (
                "border:2px dashed #ccc; border-radius:5px; padding:20px; "
                "height:200px; background-color:rgba(224,224,224,0.4); "
                "display:flex; align-items:center; justify-content:center;"
            )
            inner = ""
            # vertical divider
            if i in (0,2):
                inner += "border-right:2px solid #aaa;"
            if i in (1,3):
                inner += "border-left:2px solid #aaa;"
            # horizontal divider
            if i in (0,1):
                inner += "border-bottom:2px solid #aaa;"
            if i in (2,3):
                inner += "border-top:2px solid #aaa;"

            style = base + inner

            if st.session_state.tiles[i] is not None:
                st.plotly_chart(st.session_state.tiles[i], use_container_width=True)
                if st.button(f"Reset Tile {i}"):
                    st.session_state.tiles[i] = None
            else:
                st.markdown(
                    f"<div style='{style}'>Tile {i}</div>",
                    unsafe_allow_html=True
                )

with tab2:
    st.subheader("Detected Schema")
    st.json(SCHEMA)
    st.subheader("Dataset Preview")
    st.dataframe(df)
    st.markdown(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
