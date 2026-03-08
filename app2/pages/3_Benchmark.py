from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from app_core import (
    Benchmark,
    EXAMPLE_LIBRARY,
    VERBOSE_OPTIONS,
    example_bundle_bytes,
    figure_from_plot,
    inject_base_styles,
    parse_benchmark_network,
    read_upload_preview,
    render_file_preview,
    render_section_header,
    stage_input_files,
)

st.set_page_config(page_title="Benchmark | Netective", page_icon="N", layout="wide")
inject_base_styles()
render_section_header(
    title="Benchmark inferred networks",
    description="Evaluate one or more inferred networks against a gold standard and inspect ranking-sensitive metrics such as AUPR, AUROC, F1 distributions, and optimal cutoff behavior.",
    eyebrow="Tool 3",
)

with st.sidebar:
    st.markdown("### Input source")
    source_mode = st.radio("Choose input type", ["Bundled example", "Upload files"], label_visibility="collapsed")
    selected_example = st.selectbox("Example dataset", list(EXAMPLE_LIBRARY["benchmark"].keys()), disabled=source_mode != "Bundled example")
    example_paths = EXAMPLE_LIBRARY["benchmark"][selected_example]
    st.download_button(
        "Download example bundle",
        data=example_bundle_bytes(example_paths),
        file_name="benchmark-example.zip",
        mime="application/zip",
        disabled=source_mode != "Bundled example",
    )
    st.markdown("### Parsing")
    directed = st.toggle("Directed networks", value=False)
    score_column = st.toggle("Inference files contain a score column", value=True)
    greater_is_better = st.toggle("Greater score is better", value=True)
    allow_self_loops = st.toggle("Allow self-loops", value=False)
    delimiter = st.text_input("Delimiter", value="\t")
    comments = st.text_input("Comment prefix", value="#")
    verbose = st.selectbox("Verbosity", VERBOSE_OPTIONS, index=0)

metric_col, metric_col2 = st.columns(2, gap="large")
with metric_col:
    baseline = st.toggle("Include baseline inference", value=False)
    optimal_cutoff = st.toggle("Show optimal cutoffs", value=True)
with metric_col2:
    f1_score = st.toggle("Show F1 score curves", value=True)
    mcc = st.toggle("Show MCC curves", value=True)

cutoff_enabled = st.toggle("Use explicit cutoff", value=False)
cutoff_value = st.slider("Cutoff", min_value=0.0, max_value=1.0, value=0.5, step=0.01, disabled=not cutoff_enabled)

if source_mode == "Bundled example":
    gold_standard_path = example_paths[0]
    inference_paths = example_paths[1:]
    st.markdown("<div class='caption-label'>Gold standard preview</div>", unsafe_allow_html=True)
    render_file_preview(gold_standard_path)
    st.markdown("<div class='caption-label'>Prediction previews</div>", unsafe_allow_html=True)
    for inference_path in inference_paths:
        render_file_preview(inference_path)
else:
    gold_standard_upload = st.file_uploader("Upload gold standard network", accept_multiple_files=False)
    inference_uploads = st.file_uploader("Upload one or more inference files", accept_multiple_files=True)
    gold_standard_path = gold_standard_upload
    inference_paths = inference_uploads or []
    if gold_standard_upload is not None:
        st.caption(gold_standard_upload.name)
        st.code(read_upload_preview(gold_standard_upload), language="text")
    if inference_paths:
        for uploaded in inference_paths:
            st.caption(uploaded.name)
            st.code(read_upload_preview(uploaded), language="text")

if st.button("Run benchmark", type="primary"):
    if gold_standard_path is None or not inference_paths:
        st.error("Provide one gold standard and at least one inferred network.")
    else:
        try:
            with st.spinner("Benchmarking predictions..."):
                with tempfile.TemporaryDirectory() as temp_dir_name:
                    temp_dir = Path(temp_dir_name)
                    staged_inputs = stage_input_files([gold_standard_path, *inference_paths], temp_dir)
                    gs_path = staged_inputs[0]
                    prepared_inferences = staged_inputs[1:]

                    gold_standard = parse_benchmark_network(
                        file_path=str(gs_path),
                        comments=comments,
                        delimiter=delimiter,
                        directed=directed,
                        score=False,
                        use_position_as_score=False,
                    )
                    inferences = {
                        path.name: parse_benchmark_network(
                            file_path=str(path),
                            comments=comments,
                            delimiter=delimiter,
                            directed=directed,
                            score=score_column,
                            use_position_as_score=not score_column,
                            greater_score_is_better=greater_is_better,
                        )
                        for path in prepared_inferences
                    }
                    benchmark = Benchmark(
                        gold_standard=gold_standard,
                        inferences=inferences,
                        greater_score_is_better=greater_is_better,
                        allow_self_loops=allow_self_loops,
                        cutoff=cutoff_value if cutoff_enabled else False,
                        baseline=baseline,
                    )
                    summary = benchmark.summarize
                    st.dataframe(summary, use_container_width=True)
                    tabs = ["AUPR", "Precision-Recall", "AUROC", "ROC"]
                    if optimal_cutoff:
                        tabs.append("Optimal cutoff")
                    if f1_score:
                        tabs.append("F1 score")
                    if mcc:
                        tabs.append("MCC")
                    rendered_tabs = st.tabs(tabs)
                    plot_index = 0
                    with rendered_tabs[plot_index]:
                        st.pyplot(figure_from_plot(benchmark.plot_aupr()))
                    plot_index += 1
                    with rendered_tabs[plot_index]:
                        st.pyplot(figure_from_plot(benchmark.plot_precision_recall_curves()))
                    plot_index += 1
                    with rendered_tabs[plot_index]:
                        st.pyplot(figure_from_plot(benchmark.plot_auroc()))
                    plot_index += 1
                    with rendered_tabs[plot_index]:
                        st.pyplot(figure_from_plot(benchmark.plot_roc_curves()))
                    plot_index += 1
                    if optimal_cutoff:
                        with rendered_tabs[plot_index]:
                            st.pyplot(figure_from_plot(benchmark.plot_optimal_cutoffs()))
                        plot_index += 1
                    if f1_score:
                        with rendered_tabs[plot_index]:
                            st.pyplot(figure_from_plot(benchmark.plot_f1_score()))
                        plot_index += 1
                    if mcc:
                        with rendered_tabs[plot_index]:
                            st.pyplot(figure_from_plot(benchmark.plot_mcc()))
        except Exception as exc:
            st.error("Benchmarking failed with the current inputs.")
            st.exception(exc)
