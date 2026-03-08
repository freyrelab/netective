from __future__ import annotations

import streamlit as st

from app_core import inject_base_styles, render_section_header, render_tool_card

st.set_page_config(page_title="Netective Studio", page_icon="N", layout="wide")
inject_base_styles()

render_section_header(
    title="Choose a workflow",
    description="The sidebar exposes each main Netective function as its own page. The cards below mirror those tools and point users to the right workflow before they touch any parameters.",
    eyebrow="Overview",
)

col1, col2 = st.columns(2, gap="large")
with col1:
    render_tool_card(
        kicker="Characterize",
        title="Profile one network or a batch",
        description="Compute scalar and node-level structural properties, preview the expected edgelist format, and inspect plots or raw values per network.",
    )
    st.page_link("pages/1_Characterize.py", label="Characterize")

    render_tool_card(
        kicker="Benchmark",
        title="Assess inferred networks against a gold standard",
        description="Load a gold standard plus scored predictions to inspect AUPR, AUROC, ROC and PR curves, with optional cutoff-driven metrics.",
    )
    st.page_link("pages/3_Benchmark.py", label="Benchmark")

with col2:
    render_tool_card(
        kicker="Compare",
        title="Measure topological similarity",
        description="Compare multiple networks, generate a clustered distance heatmap, and optionally include analog random models for side-by-side context.",
    )
    st.page_link("pages/2_Compare.py", label="Compare")

    render_tool_card(
        kicker="Classify",
        title="Cluster networks from files or distances",
        description="Classify networks from raw edgelists, existing Netective property outputs, or a precomputed distance matrix, then inspect cluster membership as a table.",
    )
    st.page_link("pages/4_Classify.py", label="Classify")

st.markdown(
    """
    <div class="section-card" style="margin-top:1rem;">
        <div class="eyebrow">Included examples</div>
    </div>
    """,
    unsafe_allow_html=True,
)
