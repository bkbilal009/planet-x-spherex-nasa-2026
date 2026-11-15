

import streamlit as st
import pandas as pd
from pathlib import Path

# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="SPHEREx Explorer",
    page_icon="🌌",
    layout="wide"
)

ROOT = Path(__file__).parent

# ---------------- CUSTOM DESIGN ----------------

st.markdown("""
<style>
    .main-title {
        font-size: 42px;
        font-weight: 700;
        color: #7C83FD;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #A0A0B5;
        margin-bottom: 25px;
    }

    .info-card {
        background-color: rgba(124, 131, 253, 0.10);
        padding: 18px;
        border-radius: 12px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------

st.sidebar.title("🌌 SPHEREx Explorer")
st.sidebar.caption("NASA Space Apps Challenge 2026")

page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Observation Data", "Image Preview", "About Project"]
)

st.sidebar.divider()
st.sidebar.info(
    "Explore NASA SPHEREx observations "
    "and investigate changes in the infrared sky."
)

# ---------------- HEADER ----------------

st.markdown(
    '<div class="main-title">🌌 SPHEREx Explorer</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Explore the infrared universe with NASA SPHEREx</div>',
    unsafe_allow_html=True
)

st.write(
    "Welcome to SPHEREx Explorer! This tool provides access to "
    "astronomical observation metadata and preliminary image "
    "difference visualizations."
)

st.divider()

# ---------------- LOAD METADATA ----------------

metadata_path = ROOT / "spherex_metadata.csv"
metadata = None

if metadata_path.exists():
    try:
        metadata = pd.read_csv(metadata_path)
    except Exception as error:
        st.error(f"Unable to read metadata: {error}")
else:
    st.warning("Metadata file not found.")

# ---------------- LOAD IMAGE ----------------

preview = ROOT / "aligned_difference_preview.png"

# ---------------- OVERVIEW PAGE ----------------

if page == "Overview":

    st.header("Project Overview")

    st.write(
        "Explore SPHEREx observation records and view "
        "preliminary differences between aligned astronomical images."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Metadata Records",
            len(metadata) if metadata is not None else 0
        )

    with col2:
        st.metric(
            "Metadata Columns",
            len(metadata.columns) if metadata is not None else 0
        )

    with col3:
        st.metric(
            "Image Preview",
            "Available" if preview.exists() else "Unavailable"
        )

    st.divider()

    st.subheader("Explore the Project")

    st.markdown("""
    <div class="info-card">
        <b>🔭 Observation Data</b><br>
        Explore available SPHEREx observation metadata.
    </div>

    <div class="info-card">
        <b>🪐 Image Preview</b><br>
        View the preliminary aligned difference image.
    </div>

    <div class="info-card">
        <b>🌌 Project Information</b><br>
        Learn about the purpose of SPHEREx Explorer.
    </div>
    """, unsafe_allow_html=True)

# ---------------- OBSERVATION DATA PAGE ----------------

elif page == "Observation Data":

    st.header("📊 SPHEREx Observation Data")

    if metadata is not None:

        st.success(
            f"Successfully loaded {len(metadata)} metadata records."
        )

        search = st.text_input(
            "🔍 Search observations",
            placeholder="Enter a keyword to search metadata..."
        )

        filtered_metadata = metadata.copy()

        if search:
            matches = filtered_metadata.astype(str).apply(
                lambda column: column.str.contains(
                    search,
                    case=False,
                    na=False
                )
            ).any(axis=1)

            filtered_metadata = filtered_metadata[matches]

        st.caption(
            f"Showing {len(filtered_metadata)} of {len(metadata)} records."
        )

        st.dataframe(
            filtered_metadata,
            use_container_width=True,
            hide_index=True
        )

        with st.expander("About this dataset"):
            st.write(
                "This table contains metadata associated with "
                "SPHEREx astronomical observations."
            )

    else:
        st.warning("No observation metadata is currently available.")

# ---------------- IMAGE PREVIEW PAGE ----------------

elif page == "Image Preview":

    st.header("🖼️ Image Difference Preview")

    st.write(
        "View the preliminary difference visualization "
        "created from aligned SPHEREx observations."
    )

    if preview.exists():

        st.image(
            str(preview),
            caption="Preliminary SPHEREx aligned difference preview",
            use_container_width=True
        )

        st.info(
            "This is a preliminary visualization. Image differences "
            "should not automatically be interpreted as confirmed "
            "moving celestial objects."
        )

    else:
        st.warning("Image preview is not available yet.")

# ---------------- ABOUT PAGE ----------------

elif page == "About Project":

    st.header("🌌 About SPHEREx Explorer")

    st.write(
        "SPHEREx Explorer is a NASA Space Apps Challenge project "
        "that uses SPHEREx infrared observations to explore "
        "astronomical images and investigate possible changes "
        "in the sky."
    )

    st.subheader("Project Objectives")

    st.markdown("""
    - Explore NASA SPHEREx observation data.
    - Examine astronomical image differences.
    - Investigate potential moving celestial objects.
    - Explore possible Planet X candidates.
    """)

    st.subheader("Technology Used")

    st.markdown("""
    - **Python:** Application development
    - **Streamlit:** Interactive web interface
    - **Pandas:** Metadata handling
    - **NASA SPHEREx:** Astronomical observation data
    """)

    st.divider()

    st.caption(
        "NASA Space Apps Challenge 2026 | SPHEREx Explorer"
    )

# ---------------- FOOTER ----------------

st.divider()

st.caption(
    "SPHEREx Explorer | NASA Space Apps Challenge 2026"
)

