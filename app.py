import streamlit as st
import pandas as pd
import seaborn as sns
import openai

from core.prompt_engine import prompt_to_json, inject_rag_context
from core.chart_render import render_chart
from core.vector_store import VectorStore

# — Set OpenAI key
openai.api_key = st.secrets["OPENAI_API_KEY"]

# — Load raw data
uploaded_file = st.file_uploader("Upload your CSV", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = sns.load_dataset("iris")

# — Sidebar filters
filters: dict = {}
with st.sidebar.expander("Filters", expanded=True):
    st.markdown("#### Narrow your data")
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            lo, hi = float(df[col].min()), float(df[col].max())
            sel = st.slider(col, lo, hi, (lo, hi))
            if sel != (lo, hi):
                filters[col] = {"type": "range", "value": sel}
        else:
            opts = df[col].dropna().unique().tolist()
            sel = st.multiselect(col, opts, default=opts)
            if set(sel) != set(opts):
                filters[col] = {"type": "category", "value": sel}

def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    for col, cfg in filters.items():
        if cfg["type"] == "range":
            lo, hi = cfg["value"]
            df = df[(df[col] >= lo) & (df[col] <= hi)]
        else:
            df = df[df[col].isin(cfg["value"])]
    return df

# — Apply filters
filtered_df = apply_filters(df, filters)

# — Dynamic schema from filtered data
SCHEMA = {
    "columns": [
        {"name": col, "dtype": str(filtered_df[col].dtype)}
        for col in filtered_df.columns
    ]
}

# — Build vector store (cached)
@st.cache_data(show_spinner=False)
def build_vector_store(chunks: tuple[str, ...]) -> VectorStore:
    vs = VectorStore()
    vs.build(list(chunks))
    return vs

schema_chunks = tuple(
    f"Column: {col} | Type: {dtype} | Unique: {filtered_df[col].nunique()}"
    + (
        f" | Min: {filtered_df[col].min()} | Max: {filtered_df[col].max()}"
        if pd.api.types.is_numeric_dtype(filtered_df[col]) else ""
    )
    for col, dtype in zip(filtered_df.columns, filtered_df.dtypes)
)

vs = build_vector_store(schema_chunks)

# — Initialize multi-page state
if "pages" not in st.session_state:
    st.session_state.pages = [{
        "name": "Page 1",
        "tiles": [None, None, None, None],
        "title": "Dashboard View"
    }]
    st.session_state.current_page_index = 0

# — Sidebar: page selector and assistant
with st.sidebar:
    # Page selector
    page_names = [p["name"] for p in st.session_state.pages]
    sel = st.selectbox("Select Page", page_names, index=st.session_state.current_page_index)
    st.session_state.current_page_index = page_names.index(sel)
    if st.button("Add New Page"):
        pg = {
            "name": f"Page {len(st.session_state.pages)+1}",
            "tiles": [None]*4,
            "title": "New Dashboard"
        }
        st.session_state.pages.append(pg)
        st.session_state.current_page_index = len(st.session_state.pages)-1

    st.markdown("---")
    # Assistant
    st.markdown("### Ask the Assistant")
    user_q = st.text_area("Question", placeholder="What should I explore next?")
    tile_choice = st.selectbox("Include tile (optional)", ["None", 0, 1, 2, 3])
    if st.button("Ask") and user_q.strip():
        # Retrieve top-5 schema chunks
        ctx = "\n".join(vs.query(user_q, top_k=5))
        chart_ctx = f"Chart from Tile {tile_choice}" if tile_choice != "None" else "None"
        sys_prompt = f"""You are a data analysis assistant. Use this dataset context:

{ctx}

User question:
{user_q}

Chart context:
{chart_ctx}"""
        try:
            resp = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role":"system","content":sys_prompt},
                          {"role":"user","content":user_q}]
            )
            st.markdown("**Assistant Response:**")
            st.markdown(resp.choices[0].message.content)
        except Exception as e:
            st.error(f"Assistant failed: {e}")

# — Sync current page
page = st.session_state.pages[st.session_state.current_page_index]
st.session_state.tiles = page["tiles"]

# — Main app
tab1, tab2 = st.tabs(["Dashboard","Dataset View"])

with tab1:
    if st.button("Clear All Tiles"):
        page["tiles"] = st.session_state.tiles = [None]*4

    st.title("RADAR: Conversational Dashboard Builder")
    up = st.text_input("Ask for a chart")
    if st.button("Generate Chart") and up.strip():
        try:
            used = [i for i,t in enumerate(st.session_state.tiles) if t]
            # RAG prompt
            chunks = vs.query(up, top_k=5)
            rag = "\n".join(chunks)
            prompt = inject_rag_context(up, rag)

            with st.expander("LLM Prompt"):
                st.code(prompt)

            cfg = prompt_to_json(prompt, SCHEMA, used)
            if cfg.get("filter") in ["NULL","Null","none","None"]:
                cfg["filter"] = None

            fig = render_chart(filtered_df, cfg)
            act = cfg.get("action","add_chart")
            tid = cfg.get("target_tile","next_available")

            if act=="add_chart":
                for i in range(4):
                    if not st.session_state.tiles[i]:
                        st.session_state.tiles[i] = fig
                        break
            elif act=="update_chart" and isinstance(tid,int) and 0<=tid<4:
                st.session_state.tiles[tid] = fig
            else:
                st.error("Unknown action or invalid tile")

            page["tiles"] = st.session_state.tiles
        except Exception as e:
            st.error(f"Error: {e}")

    # Page title
    page["title"] = st.text_input("Dashboard Title", value=page["title"])
    st.markdown(f"<h2 style='text-align:center'>{page['title']}</h2>", unsafe_allow_html=True)

    # Render 2×2 grid
    c1,c2 = st.columns(2)
    for idx,col in enumerate([c1,c2,c1,c2]):
        with col:
            base = "border:2px dashed #ccc; border-radius:5px; padding:20px; height:200px; background-color:rgba(224,224,224,0.4); display:flex; align-items:center; justify-content:center;"
            brd = ""
            if idx in (0,2): brd+="border-right:2px solid #aaa;"
            if idx in (1,3): brd+="border-left:2px solid #aaa;"
            if idx in (0,1): brd+="border-bottom:2px solid #aaa;"
            if idx in (2,3): brd+="border-top:2px solid #aaa;"
            style = base + brd

            if st.session_state.tiles[idx]:
                st.plotly_chart(st.session_state.tiles[idx],use_container_width=True)
                if st.button(f"Reset Tile {idx}"):
                    st.session_state.tiles[idx] = None
                    page["tiles"] = st.session_state.tiles
            else:
                st.markdown(f"<div style='{style}'>Tile {idx}</div>",
                            unsafe_allow_html=True)

with tab2:
    st.subheader("Detected Schema")
    st.json(SCHEMA)
    st.subheader("Dataset Preview")
    st.dataframe(filtered_df)
    st.markdown(f"**Shape:** {filtered_df.shape[0]} rows × {filtered_df.shape[1]} columns")
