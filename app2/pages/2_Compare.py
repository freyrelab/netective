from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from app_core import (
    ASSOCIATION_OPTIONS,
    BA_OPTIONS,
    DISTANCE_OPTIONS,
    EXAMPLE_LIBRARY,
    METHOD_OPTIONS,
    MODEL_OPTIONS,
    NETWORK_FORMAT_OPTIONS,
    NORMALIZATION_OPTIONS,
    VERBOSE_OPTIONS,
    clean_selected_props,
    compare_structure,
    example_bundle_bytes,
    get_property_names,
    inject_base_styles,
    mixed_ba_values,
    parse_structure_network,
    read_upload_preview,
    render_file_preview,
    render_section_header,
    stage_input_files,
    workers_value,
)

st.set_page_config(page_title="Compare | Netective", page_icon="N", layout="wide")
inject_base_styles()
render_section_header(
    title="Compare network structure",
    description="This page wraps the comparative topology workflow. Load multiple networks to compute a clustered association-distance heatmap and, if needed, add random analog models for context.",
    eyebrow="Tool 2",
)

property_names = get_property_names()

with st.sidebar:
    st.markdown("### Input source")
    source_mode = st.radio("Choose input type", ["Bundled example", "Upload files"], label_visibility="collapsed")
    selected_example = st.selectbox("Example dataset", list(EXAMPLE_LIBRARY["compare"].keys()), disabled=source_mode != "Bundled example")
    example_paths = EXAMPLE_LIBRARY["compare"][selected_example]
    st.download_button(
        "Download example bundle",
        data=example_bundle_bytes(example_paths),
        file_name="compare-example.zip",
        mime="application/zip",
        disabled=source_mode != "Bundled example",
    )
    st.markdown("### Parsing")
    directed = st.toggle("Directed networks", value=True)
    normalization = st.selectbox("Normalization", NORMALIZATION_OPTIONS, format_func=lambda value: "None" if value is None else value.title())
    net_format = st.selectbox("Network file format", NETWORK_FORMAT_OPTIONS)
    delimiter = st.text_input("Delimiter", value="\t")
    comments = st.text_input("Comment prefix", value="#")
    st.markdown("### Comparison")
    association = st.selectbox("Association metric", ASSOCIATION_OPTIONS)
    metric = st.selectbox("Distance metric", DISTANCE_OPTIONS, index=DISTANCE_OPTIONS.index("euclidean"))
    method = st.selectbox("Linkage method", METHOD_OPTIONS, index=METHOD_OPTIONS.index("ward"))
    use_auto_workers = st.toggle("Auto workers", value=True)
    workers = st.number_input("Workers", min_value=1, max_value=16, value=2, disabled=use_auto_workers)
    verbose = st.selectbox("Verbosity", VERBOSE_OPTIONS, index=0)

title = st.text_input("Plot title", value="Netective structural similarity")
selected_props = st.multiselect(
    "Selected properties",
    options=property_names,
    default=[],
    help="Leave empty to use every available property.",
)

model_col, model_col2 = st.columns(2, gap="large")
with model_col:
    include_models = st.multiselect("Include analog random models", MODEL_OPTIONS)
    compare_to_models = st.toggle("Compare only against model averages", value=False, disabled=not include_models)
with model_col2:
    n_models = st.number_input("Random models per input network", min_value=1, max_value=20, value=2, disabled=not include_models)
    directed_models = st.toggle("Directed model generation", value=False, disabled=not include_models)

keep_averages = st.toggle("Keep distribution averages in scalar arrays", value=True)
ba_raw = st.text_input("Barabasi-Albert m values", value="2", disabled="Barabasi Albert" not in include_models, help="Comma-separated integers or degree labels such as degree, in degree, out degree.")
st.caption(f"Recognized BA labels: {', '.join(BA_OPTIONS)}")

if source_mode == "Bundled example":
    file_paths = example_paths
    for example_path in file_paths:
        render_file_preview(example_path)
else:
    uploaded_files = st.file_uploader("Upload at least two network files", accept_multiple_files=True)
    file_paths = uploaded_files or []
    if file_paths:
        for uploaded in file_paths:
            st.caption(uploaded.name)
            st.code(read_upload_preview(uploaded), language="text")

if st.button("Run comparison", type="primary"):
    if len(file_paths) < 2:
        st.error("Provide at least two networks to compare.")
    else:
        try:
            with st.spinner("Computing structural similarity..."):
                with tempfile.TemporaryDirectory() as temp_dir_name:
                    temp_dir = Path(temp_dir_name)
                    prepared_paths = stage_input_files(file_paths, temp_dir)

                    graphs = {
                        path.name: parse_structure_network(
                            file_path=str(path),
                            comments=comments,
                            delimiter=delimiter,
                            directed=directed,
                            net_file_format=net_format,
                        )
                        for path in prepared_paths
                    }
                    heatmap, association_df = compare_structure(
                        networks=graphs,
                        directed=directed,
                        norm=normalization,
                        selected_props=clean_selected_props(selected_props),
                        workers=workers_value(use_auto_workers, int(workers)),
                        association_metric=association,
                        metric=metric,
                        method=method,
                        include_models=include_models or None,
                        compare_to_models=compare_to_models,
                        n_random_models=int(n_models),
                        directed_models=directed_models,
                        ba_m=mixed_ba_values(ba_raw),
                        keep_averages=keep_averages,
                        verbose=verbose,
                        title=title or None,
                    )
                    st.pyplot(heatmap)
                    st.dataframe(association_df, use_container_width=True)
                    st.download_button(
                        "Download association matrix",
                        data=association_df.to_csv(sep="\t"),
                        file_name="association_matrix.tsv",
                        mime="text/tab-separated-values",
                    )
        except Exception as exc:
            st.error("Comparison failed with the current inputs.")
            st.exception(exc)
