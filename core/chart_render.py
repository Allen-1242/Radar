import streamlit as st
import plotly.express as px

# Dashboard wrapper for layout rendering
def render_chart(df, config):
    chart_type = config.get("chart_type")
    if not chart_type:
        raise ValueError("Missing 'chart_type' in chart configuration.")

    # Apply filters if present
    if "filter" in config and config["filter"]:
        for key, val in config["filter"].items():
            df = df[df[key] == val]

    # Get optional style config
    style = config.get("style", {})
    color = style.get("color")

    # Chart title
    title = config.get("title", "")

    if chart_type == "bar":
        return px.bar(
            df,
            x=config["x_axis"],
            y=config["y_axis"],
            color_discrete_sequence=[color] if color else None,
            title=title
        )
    elif chart_type == "line":
        return px.line(
            df,
            x=config["x_axis"],
            y=config["y_axis"],
            color_discrete_sequence=[color] if color else None,
            title=title
        )
    elif chart_type == "pie":
        return px.pie(
            df,
            names=config["x_axis"],
            values=config["y_axis"],
            color_discrete_sequence=[color] if color else None,
            title=title
        )
    elif chart_type == "scatter":
        return px.scatter(
            df,
            x=config["x_axis"],
            y=config["y_axis"],
            color_discrete_sequence=[color] if color else None,
            title=title
        )
    else:
        raise ValueError(f"Unsupported chart type: {chart_type}")

# Layout rendering with visual tile placeholders
def render_dashboard():
    col1, col2 = st.columns(2)

    for i, col in enumerate([col1, col2, col1, col2]):  # Tile 0-3
        with col:
            st.markdown(f"**Tile {i}**")
            tile_spot = st.empty()
            if st.session_state.tiles[i]:
                tile_spot.plotly_chart(st.session_state.tiles[i], use_container_width=True)
            else:
                tile_spot.markdown(f"""
                    <div style='
                        border: 2px dashed #ccc;
                        border-radius: 10px;
                        padding: 10px;
                        height: 400px;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        background-color: #000000;
                        color: #ffffff;
                        font-size: 16px;
                    '>
                    (empty)
                    </div>
                """, unsafe_allow_html=True)
