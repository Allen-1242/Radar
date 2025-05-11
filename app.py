import streamlit as st
import pandas as pd
import seaborn as sns
import openai

from core.prompt_engine import prompt_to_json, inject_rag_context
from core.chart_render import render_chart
from core.vector_store import VectorStore

# — Set the OpenAI API key from Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

# — Data loading & dynamic schema generation
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

# — Prepare schema chunks for vector store embedding
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

# — Initialize multi-page state
if "pages" not in st.session_state:
    st.session_state.pages = [{
        "name": "Page 1",
        "tiles": [None, None, None, None],
        "title": "Dashboard View"
    }]
    st.session_state.current_page_index = 0

# — Sidebar: Page selector, new page, and assistant
with st.sidebar:
    # Page management
    page_names = [p["name"] for p in st.session_state.pages]
    selected = st.selectbox("Select Page", page_names,
                            index=st.session_state.current_page_index)
    st.session_state.current_page_index = page_names.index(selected)

    if st.button("Add New Page"):
        new_page = {
            "name": f"Page {len(st.session_state.pages) + 1}",
            "tiles": [None, None, None, None],
            "title": "New Dashboard"
        }
        st.session_state.pages.append(new_page)
        st.session_state.current_page_index = len(st.session_state.pages) - 1

    st.markdown("---")
    # Conversational assistant
    st.markdown("### Ask the Assistant")
    st.markdown("Pose a question about your dataset or chart context.")

    user_question = st.text_area(
        "Your question",
        placeholder="E.g., What patterns are worth exploring?"
    )
    selected_tile = st.selectbox(
        "Include chart from tile (optional)",
        options=["None", 0, 1, 2, 3]
    )

    if st.button("Ask") and user_question.strip():
        # Build assistant context from schema
        dataset_context = "\n".join(schema_chunks)
        chart_ctx = (
            f"Chart from Tile {selected_tile}"
            if selected_tile != "None" else "None selected."
        )
        system_prompt = f"""You are a data analysis assistant. Use the following context to respond.

DATASET STRUCTURE:
{dataset_context}

USER QUESTION:
{user_question}

CHART CONTEXT:
{chart_ctx}
"""
        try:
            resp = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_question}
                ]
            )
            assistant_response = resp["choices"][0]["message"]["content"]
            st.markdown("**Assistant Response:**")
            st.markdown(assistant_response)
        except Exception as e:
            st.error(f"Assistant error: {e}")

# — Sync current page state
page = st.session_state.pages[st.session_state.current_page_index]
st.session_state.tiles = page["tiles"]

# — Main area: Dashboard & Data View tabs
tab1, tab2 = st.tabs(["Dashboard", "Dataset View"])

with tab1:
    # Clear all tiles
    if st.button("Clear All Tiles"):
        st.session_state.tiles = [None, None, None, None]
        page["tiles"] = st.session_state.tiles

    st.title("RADAR: Conversational Dashboard Builder")
    user_prompt = st.text_input(
        "Ask for a chart (e.g., 'Show me a bar chart of petal length per species')"
    )

    if st.button("Generate Chart") and user_prompt.strip():
        try:
            used_tiles = [
                i for i, t in enumerate(st.session_state.tiles) if t is not None
            ]
            # RAG context for chart generation
            rag_chunks = vs.query(user_prompt, top_k=5)
            rag_ctx = "\n".join(rag_chunks)
            rag_prompt = inject_rag_context(user_prompt, rag_ctx)

            # Debug: full prompt
            with st.expander("View full LLM prompt"):
                st.code(rag_prompt, language="text")

            config = prompt_to_json(rag_prompt, SCHEMA, used_tiles)
            if config.get("filter") in ["NULL", "Null", "none", "None"]:
                config["filter"] = None

            fig = render_chart(df, config)
            action = config.get("action", "add_chart")
            tile_id = config.get("target_tile", "next_available")

            if action == "add_chart":
                for i in range(4):
                    if st.session_state.tiles[i] is None:
                        st.session_state.tiles[i] = fig
                        break
            elif action == "update_chart":
                if isinstance(tile_id, int) and 0 <= tile_id < 4:
                    st.session_state.tiles[tile_id] = fig
                else:
                    st.error(f"Invalid tile ID: {tile_id}")
            else:
                st.error(f"Unknown action: {action}")

            # Save back to page state
            page["tiles"] = st.session_state.tiles

        except Exception as e:
            st.error(f"Chart generation error: {e}")

    # Dashboard title per page
    page["title"] = st.text_input("Dashboard Title", value=page["title"])
    st.markdown(
        f"<h2 style='text-align: center'>{page['title']}</h2>",
        unsafe_allow_html=True
    )

    # Render 2×2 grid with quadrant borders
    col1, col2 = st.columns(2)
    for idx, col in enumerate([col1, col2, col1, col2]):
        with col:
            base_style = (
                "border:2px dashed #ccc; border-radius:5px; padding:20px; "
                "height:200px; background-color:rgba(224,224,224,0.4); "
                "display:flex; align-items:center; justify-content:center;"
            )
            inner_borders = ""
            if idx in (0, 2):
                inner_borders += "border-right:2px solid #aaa;"
            if idx in (1, 3):
                inner_borders += "border-left:2px solid #aaa;"
            if idx in (0, 1):
                inner_borders += "border-bottom:2px solid #aaa;"
            if idx in (2, 3):
                inner_borders += "border-top:2px solid #aaa;"

            style = base_style + inner_borders

            if st.session_state.tiles[idx] is not None:
                st.plotly_chart(st.session_state.tiles[idx], use_container_width=True)
                if st.button(f"Reset Tile {idx}"):
                    st.session_state.tiles[idx] = None
                    page["tiles"] = st.session_state.tiles
            else:
                st.markdown(
                    f"<div style='{style}'>Tile {idx}</div>",
                    unsafe_allow_html=True
                )

with tab2:
    st.subheader("Detected Schema")
    st.json(SCHEMA)

    st.subheader("Dataset Preview")
    st.dataframe(df)
    st.markdown(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
