import streamlit as st

def place_chart(fig, target_tile="next_available"):
    if target_tile == "next_available":
        for i in range(4):
            if st.session_state.tiles[i] is None:
                st.session_state.tiles[i] = fig
                return i
    elif isinstance(target_tile, int) and 0 <= target_tile < 4:
        st.session_state.tiles[target_tile] = fig
        return target_tile
    return None

def render_dashboard():
    col1, col2 = st.columns(2)

    with col1:
        if st.session_state.tiles[0]:
            st.plotly_chart(st.session_state.tiles[0], use_container_width=True)
        if st.session_state.tiles[2]:
            st.plotly_chart(st.session_state.tiles[2], use_container_width=True)

    with col2:
        if st.session_state.tiles[1]:
            st.plotly_chart(st.session_state.tiles[1], use_container_width=True)
        if st.session_state.tiles[3]:
            st.plotly_chart(st.session_state.tiles[3], use_container_width=True)
