from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from app_core import (
    CHARACTERIZE_EXAMPLE_DEFAULTS,
    EXAMPLE_LIBRARY,
    NETWORK_FORMAT_OPTIONS,
    NORMALIZATION_OPTIONS,
    VERBOSE_OPTIONS,
    characterize_network,
    clean_selected_props,
    get_property_names,
    inject_base_styles,
    parse_structure_network,
    plot_distributions,
    plot_scalars,
    read_upload_preview,
    render_file_preview,
    render_section_header,
    stage_input_files,
    workers_value,
    compare_structure,
    example_bundle_bytes,
)

st.set_page_config(page_title="Characterize | Netective", page_icon="N", layout="wide")
inject_base_styles()
render_section_header(
    title="Characterize networks",
    description="Run the structural characterization workflow on one or more network files. Single-network runs show scalar and distribution plots directly; multi-network runs return the same characterization per file.",
    eyebrow="Tool 1",
)

property_names = get_property_names()

with st.sidebar:
    st.markdown("### Input source")
    source_mode = st.radio("Choose input type", ["Bundled example", "Upload files"], label_visibility="collapsed")
    selected_example = st.selectbox("Example dataset", list(EXAMPLE_LIBRARY["characterize"].keys()), disabled=source_mode != "Bundled example")
    example_paths = EXAMPLE_LIBRARY["characterize"][selected_example]
    example_defaults = CHARACTERIZE_EXAMPLE_DEFAULTS.get(selected_example, {})
    if source_mode == "Bundled example":
        if st.session_state.get("characterize_selected_example") != selected_example:
            st.session_state["characterize_directed"] = example_defaults.get("directed", True)
            st.session_state["characterize_selected_example"] = selected_example
    else:
        st.session_state.pop("characterize_selected_example", None)
        st.session_state.setdefault("characterize_directed", True)
    st.download_button(
        "Download example bundle",
        data=example_bundle_bytes(example_paths),
        file_name="characterize-example.zip",
        mime="application/zip",
        disabled=source_mode != "Bundled example",
    )
    st.markdown("### Parsing")
    directed = st.toggle("Directed network", key="characterize_directed")
    normalization = st.selectbox("Normalization", NORMALIZATION_OPTIONS, format_func=lambda value: "None" if value is None else value.title())
    net_format = st.selectbox("Network file format", NETWORK_FORMAT_OPTIONS)
    delimiter = st.text_input("Delimiter", value="\t")
    comments = st.text_input("Comment prefix", value="#")
    st.markdown("### Execution")
    use_auto_workers = st.toggle("Auto workers", value=True)
    workers = st.number_input("Workers", min_value=1, max_value=16, value=2, disabled=use_auto_workers)
    verbose = st.selectbox("Verbosity", VERBOSE_OPTIONS, index=0)

selected_props = st.multiselect(
    "Selected properties",
    options=property_names,
    default=[],
    help="Leave empty to compute every available property.",
)

if source_mode == "Bundled example":
    file_paths = example_paths
    upload_label = "Using bundled example"
else:
    uploaded_files = st.file_uploader("Upload one or more network files", accept_multiple_files=True)
    file_paths = uploaded_files or []
    upload_label = "Uploaded files"

st.markdown(f"<div class='caption-label'>{upload_label}</div>", unsafe_allow_html=True)

if source_mode == "Bundled example":
    for example_path in file_paths:
        render_file_preview(example_path)
else:
    if file_paths:
        for uploaded in file_paths:
            st.caption(uploaded.name)
            st.code(read_upload_preview(uploaded), language="text")

run_clicked = st.button("Run characterization", type="primary")

if run_clicked:
    if not file_paths:
        st.error("Provide at least one network file.")
    else:
        normalized_props = clean_selected_props(selected_props)
        resolved_workers = workers_value(use_auto_workers, int(workers))
        try:
            with st.spinner("Computing structural properties..."):
                with tempfile.TemporaryDirectory() as temp_dir_name:
                    temp_dir = Path(temp_dir_name)
                    prepared_paths = stage_input_files(file_paths, temp_dir)

                    if len(prepared_paths) == 1:
                        net_id = prepared_paths[0].name
                        graph = parse_structure_network(
                            file_path=str(prepared_paths[0]),
                            comments=comments,
                            delimiter=delimiter,
                            directed=directed,
                            net_file_format=net_format,
                        )
                        scalar_props, dist_props = characterize_network(
                            G=graph,
                            net_id=net_id,
                            norm=normalization,
                            selected_props=normalized_props,
                            verbose=verbose,
                            return_prop_dicts=True,
                        )
                        scalar_values = scalar_props.get(net_id, scalar_props)
                        dist_values = dist_props.get(net_id, dist_props)
                        fig_scalars, _ = plot_scalars(scalar_values, verbose=verbose)
                        fig_dists, _ = plot_distributions(dist_values, verbose=verbose)
                        left, right = st.columns(2, gap="large")
                        with left:
                            st.pyplot(fig_scalars)
                        with right:
                            if fig_dists is not None:
                                st.pyplot(fig_dists)
                            else:
                                st.info("No node-level distributions were produced for this network.")
                        tab1, tab2 = st.tabs(["Scalar properties", "Distribution values"])
                        with tab1:
                            scalar_df = pd.DataFrame.from_dict(scalar_values, orient="index", columns=["value"])
                            st.dataframe(scalar_df, use_container_width=True)
                        with tab2:
                            dist_df = pd.DataFrame({key: pd.Series(value) for key, value in dist_values.items()}) if dist_values else pd.DataFrame()
                            st.dataframe(dist_df, use_container_width=True)
                    else:
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
                        scalars_array, dist_array = compare_structure(
                            networks=graphs,
                            directed=directed,
                            norm=normalization,
                            selected_props=normalized_props,
                            workers=resolved_workers,
                            return_prop_dicts=True,
                            keep_averages=False,
                            verbose=verbose,
                        )
                        for network_name in scalars_array:
                            with st.expander(network_name, expanded=True):
                                left, right = st.columns(2, gap="large")
                                fig_scalars, _ = plot_scalars(scalars_array[network_name], verbose=verbose)
                                fig_dists, _ = plot_distributions(dist_array.get(network_name, {}), verbose=verbose)
                                with left:
                                    st.pyplot(fig_scalars)
                                with right:
                                    if fig_dists is not None:
                                        st.pyplot(fig_dists)
                                    else:
                                        st.info("No node-level distributions were produced for this network.")
                                scalar_df = pd.DataFrame.from_dict(scalars_array[network_name], orient="index", columns=["value"])
                                st.dataframe(scalar_df, use_container_width=True)
        except Exception as exc:
            st.error("Characterization failed with the current inputs.")
            st.exception(exc)
