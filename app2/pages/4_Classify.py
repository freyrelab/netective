from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from app_core import (
    ASSOCIATION_OPTIONS,
    CLASSIFICATION_DEFAULT_PROPS,
    DISTANCE_OPTIONS,
    EXAMPLE_LIBRARY,
    METHOD_OPTIONS,
    NETWORK_FORMAT_OPTIONS,
    NORMALIZATION_OPTIONS,
    VERBOSE_OPTIONS,
    clean_selected_props,
    classify_networks,
    clusters_to_frame,
    example_bundle_bytes,
    get_property_names,
    inject_base_styles,
    read_upload_preview,
    render_file_preview,
    render_section_header,
    stage_input_files,
    workers_value,
)

st.set_page_config(page_title="Classify | Netective", page_icon="N", layout="wide")
inject_base_styles()
render_section_header(
    title="Classify networks into clusters",
    description="Use raw network files, existing Netective property outputs, or a precomputed distance matrix to build flat clusters and review cluster assignments immediately.",
    eyebrow="Tool 4",
)

property_names = get_property_names()

with st.sidebar:
    input_mode = st.radio("Classification input", ["Network files", "Properties directory", "Distance dataframe"])
    directed = st.toggle("Directed networks", value=True, disabled=input_mode == "Distance dataframe")
    normalization = st.selectbox("Normalization", NORMALIZATION_OPTIONS, format_func=lambda value: "None" if value is None else value.title(), disabled=input_mode != "Network files")
    net_format = st.selectbox("Network file format", NETWORK_FORMAT_OPTIONS, disabled=input_mode != "Network files")
    delimiter = st.text_input("Delimiter", value="\t")
    comments = st.text_input("Comment prefix", value="#", disabled=input_mode == "Distance dataframe")
    use_auto_workers = st.toggle("Auto workers", value=True, disabled=input_mode != "Network files")
    workers = st.number_input("Workers", min_value=1, max_value=16, value=2, disabled=use_auto_workers or input_mode != "Network files")
    add_averages = st.toggle("Include local-property averages", value=False, disabled=input_mode == "Distance dataframe")
    association_metric = st.selectbox("Association metric", ASSOCIATION_OPTIONS, disabled=input_mode == "Distance dataframe")
    metric = st.selectbox("Distance metric", DISTANCE_OPTIONS, index=DISTANCE_OPTIONS.index("euclidean"))
    method = st.selectbox("Linkage method", METHOD_OPTIONS, index=METHOD_OPTIONS.index("ward"))
    map_ids = st.toggle("Return mapped cluster membership", value=True)
    verbose = st.selectbox("Verbosity", VERBOSE_OPTIONS, index=0)

selected_props = st.multiselect(
    "Selected properties",
    options=property_names,
    default=CLASSIFICATION_DEFAULT_PROPS,
)

cluster_mode = st.radio("Flat clustering criterion", ["Threshold", "Max clusters"], horizontal=True)
threshold = st.slider("Threshold", min_value=0.0, max_value=1.0, value=0.7, step=0.01, disabled=cluster_mode != "Threshold")
clusters = st.number_input("Max clusters", min_value=2, max_value=20, value=3, disabled=cluster_mode != "Max clusters")

if input_mode == "Network files":
    source_mode = st.radio("Source", ["Bundled example", "Upload files"], horizontal=True)
    example_paths = EXAMPLE_LIBRARY["classify_networks"][list(EXAMPLE_LIBRARY["classify_networks"].keys())[0]]
    if source_mode == "Bundled example":
        st.download_button(
            "Download example bundle",
            data=example_bundle_bytes(example_paths),
            file_name="classify-networks-example.zip",
            mime="application/zip",
        )
        for example_path in example_paths:
            render_file_preview(example_path)
        file_payload = example_paths
    else:
        uploads = st.file_uploader("Upload two or more networks", accept_multiple_files=True)
        file_payload = uploads or []
        if file_payload:
            for upload in file_payload:
                st.caption(upload.name)
                st.code(read_upload_preview(upload), language="text")
elif input_mode == "Distance dataframe":
    source_mode = st.radio("Source", ["Bundled example", "Upload file"], horizontal=True)
    distance_example = EXAMPLE_LIBRARY["classify_distance"][list(EXAMPLE_LIBRARY["classify_distance"].keys())[0]][0]
    if source_mode == "Bundled example":
        render_file_preview(distance_example)
        st.download_button(
            "Download distance example",
            data=distance_example.read_bytes(),
            file_name=distance_example.name,
            mime="text/tab-separated-values",
        )
        file_payload = distance_example
    else:
        upload = st.file_uploader("Upload a distance dataframe", accept_multiple_files=False)
        file_payload = upload
        if upload is not None:
            st.caption(upload.name)
            st.code(read_upload_preview(upload), language="text")
else:
    source_mode = "Upload files"
    uploads = st.file_uploader("Upload Netective property files", accept_multiple_files=True)
    file_payload = uploads or []
    if file_payload:
        for upload in file_payload:
            st.caption(upload.name)
            st.code(read_upload_preview(upload), language="text")

if st.button("Run classification", type="primary"):
    if input_mode != "Distance dataframe" and not file_payload:
        st.error("Provide the input files required for this classification mode.")
    elif input_mode == "Distance dataframe" and file_payload is None:
        st.error("Provide a distance dataframe.")
    else:
        try:
            with st.spinner("Classifying networks..."):
                with tempfile.TemporaryDirectory() as temp_dir_name:
                    temp_dir = Path(temp_dir_name)
                    classification_kwargs = {
                        "norm": normalization,
                        "directed": directed,
                        "selected_props": clean_selected_props(selected_props),
                        "workers": workers_value(use_auto_workers, int(workers)),
                        "add_averages": add_averages,
                        "association_metric": association_metric,
                        "clust_num": int(clusters) if cluster_mode == "Max clusters" else None,
                        "threshold": None if cluster_mode == "Max clusters" else threshold,
                        "metric": metric,
                        "method": method,
                        "map_ids": map_ids,
                        "verbose": verbose,
                        "nets_file_format": net_format,
                        "comments": comments,
                        "delimiter": delimiter,
                    }

                    if input_mode == "Network files":
                        if len(file_payload) < 2:
                            st.error("Provide at least two networks to form clusters.")
                            st.stop()
                        stage_input_files(file_payload, temp_dir)
                        clusters_output = classify_networks(networks=str(temp_dir), **classification_kwargs)
                    elif input_mode == "Properties directory":
                        stage_input_files(file_payload, temp_dir)
                        clusters_output = classify_networks(results_dir=str(temp_dir), **classification_kwargs)
                    else:
                        staged_paths = stage_input_files([file_payload], temp_dir)
                        distance_df = pd.read_csv(staged_paths[0], sep=delimiter, index_col=0)
                        clusters_output = classify_networks(distance_df=distance_df, **classification_kwargs)

                    clusters_df = clusters_to_frame(clusters_output)
                    st.dataframe(clusters_df, use_container_width=True)
                    st.download_button(
                        "Download cluster assignments",
                        data=clusters_df.to_csv(index=False),
                        file_name="network_clusters.csv",
                        mime="text/csv",
                    )
        except Exception as exc:
            st.error("Classification failed with the current inputs.")
            st.exception(exc)