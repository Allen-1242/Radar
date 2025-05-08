import streamlit as st

def init_tiles():
    if "tiles" not in st.session_state:
        st.session_state.tiles = [None, None, None, None]

def get_tile_summary():
    return [i for i, t in enumerate(st.session_state.tiles) if t is not None]
