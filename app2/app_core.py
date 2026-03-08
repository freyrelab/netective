from __future__ import annotations

import base64
import inspect
import sys
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Iterable

import matplotlib.figure
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from netective import characterize_network, classify_networks, compare_structure  # noqa: E402
from netective.stats.stats import Benchmark, parse_network as parse_benchmark_network  # noqa: E402
from netective.structure import properties as structure_properties  # noqa: E402
from netective.structure.dataviz import plot_distributions, plot_scalars  # noqa: E402
from netective.utils import parse_network as parse_structure_network  # noqa: E402

EXAMPLES_DIR = APP_DIR / "examples"
ASSETS_DIR = APP_DIR / "assets"
BRAND_LOGO_PATH = REPO_ROOT / "app" / "assets" / "on_black.png"

NORMALIZATION_OPTIONS = [None, "network", "biological"]
VERBOSE_OPTIONS = ["WARNING", "INFO", "DEBUG", "ERROR", "CRITICAL"]
NETWORK_FORMAT_OPTIONS = ["edgelist", "graphml", "adj list", "multiline adj list"]
ASSOCIATION_OPTIONS = ["pearson", "spearman", "cosine"]
DISTANCE_OPTIONS = [
    "braycurtis",
    "canberra",
    "chebyshev",
    "cityblock",
    "correlation",
    "cosine",
    "dice",
    "euclidean",
    "hamming",
    "jaccard",
    "jensenshannon",
    "kulczynski1",
    "mahalanobis",
    "matching",
    "minkowski",
    "rogerstanimoto",
    "russellrao",
    "seuclidean",
    "sokalmichener",
    "sokalsneath",
    "sqeuclidean",
    "yule",
]
METHOD_OPTIONS = ["single", "complete", "average", "weighted", "centroid", "median", "ward"]
MODEL_OPTIONS = ["Erdos GNP", "Erdos GNM", "K Regular", "Barabasi Albert"]
BA_OPTIONS = ["in degree", "out degree", "degree"]
CLASSIFICATION_DEFAULT_PROPS = [
    "Average Local Efficiency",
    "Radius",
    "Center",
    "Periphery",
    "Complex Feed-Forward Circuits",
    "Feed-Forward Circuits",
    "Max Degree",
    "Gini Index",
    "Global Efficiency",
    "Undirected Gini Index",
    "Entropy of Degree Distribution",
    "Self-Loops",
]

EXAMPLE_LIBRARY = {
    "characterize": {
        "Zachary karate club network": [EXAMPLES_DIR / "characterize" / "karate_club.tsv"],
    },
    "compare": {
        "Three structural archetypes": [
            EXAMPLES_DIR / "compare" / "development.tsv",
            EXAMPLES_DIR / "compare" / "stress.tsv",
            EXAMPLES_DIR / "compare" / "metabolic.tsv",
        ],
    },
    "benchmark": {
        "Gold standard with two ranked predictions": [
            EXAMPLES_DIR / "benchmark" / "gold_standard.tsv",
            EXAMPLES_DIR / "benchmark" / "prediction_ranked.tsv",
            EXAMPLES_DIR / "benchmark" / "prediction_alt.tsv",
        ],
    },
    "classify_networks": {
        "Three networks to cluster": [
            EXAMPLES_DIR / "classify" / "cell_cycle.tsv",
            EXAMPLES_DIR / "classify" / "stress_response.tsv",
            EXAMPLES_DIR / "classify" / "metabolism.tsv",
        ],
    },
    "classify_distance": {
        "Precomputed distance matrix": [EXAMPLES_DIR / "classify" / "distance_dataframe.tsv"],
    },
}

CHARACTERIZE_EXAMPLE_DEFAULTS = {
    "Zachary karate club network": {
        "directed": False,
    },
}


def get_property_names() -> list[str]:
    names = []
    for _, obj in inspect.getmembers(structure_properties):
        if inspect.isclass(obj) and issubclass(obj, structure_properties._Property) and obj is not structure_properties._Property:
            names.append(obj.CLASS_NAME)
    return sorted(set(names))


def inject_base_styles() -> None:
    logo_markup = ""
    if BRAND_LOGO_PATH.exists():
        encoded_logo = base64.b64encode(BRAND_LOGO_PATH.read_bytes()).decode("ascii")
        logo_markup = f'<img src="data:image/png;base64,{encoded_logo}" alt="Netective logo" style="width:100%; height:auto; display:block; border-radius:18px;" />'
    st.markdown(
        f"""
        <style>
        :root {{
            --bg: #030303;
            --surface: rgba(17, 20, 23, 0.92);
            --surface-strong: rgba(10, 12, 14, 0.98);
            --border: rgba(217, 244, 255, 0.12);
            --accent: #d9f4ff;
            --accent-soft: rgba(143, 211, 232, 0.16);
            --text: #fffdf3;
            --muted: #9faab0;
        }}
        .stApp {{
            background:
                radial-gradient(circle at top left, rgba(143, 211, 232, 0.10), transparent 30%),
                radial-gradient(circle at bottom right, rgba(217, 244, 255, 0.08), transparent 28%),
                linear-gradient(180deg, #030303 0%, #06080a 100%);
            color: var(--text);
        }}
        [data-testid="stHeader"] {{
            background: transparent;
        }}
        [data-testid="stSidebar"] {{
            background: rgba(6, 8, 10, 0.94);
            border-right: 1px solid var(--border);
        }}
        .block-container {{
            padding-top: 1.6rem;
            padding-bottom: 2.4rem;
        }}
        div[data-testid="stVerticalBlock"] > div:has(> .hero-card),
        div[data-testid="stVerticalBlock"] > div:has(> .section-card),
        div[data-testid="stVerticalBlock"] > div:has(> .tool-card) {{
            width: 100%;
        }}
        .hero-card, .section-card, .tool-card {{
            border: 1px solid var(--border);
            background: linear-gradient(180deg, rgba(17, 20, 23, 0.96), rgba(11, 14, 16, 0.94));
            border-radius: 24px;
            padding: 1.2rem 1.25rem;
            box-shadow: 0 22px 80px rgba(0, 0, 0, 0.28);
        }}
        .hero-card {{
            padding: 1rem 1.2rem 1.35rem;
            margin-bottom: 1rem;
        }}
        .hero-grid {{
            display: grid;
            grid-template-columns: 120px 1fr;
            gap: 1rem;
            align-items: center;
        }}
        .hero-logo img {{
            width: 100%;
            height: auto;
            display: block;
        }}
        .eyebrow {{
            letter-spacing: 0.22em;
            text-transform: uppercase;
            color: var(--muted);
            font-size: 0.72rem;
            margin-bottom: 0.35rem;
        }}
        .hero-title {{
            font-family: Bahnschrift, "Segoe UI Variable Display", "Segoe UI", sans-serif;
            font-size: 2.2rem;
            line-height: 1;
            margin: 0 0 0.45rem;
            color: var(--text);
        }}
        .hero-copy, .tool-copy {{
            color: var(--muted);
            line-height: 1.55;
            font-size: 0.98rem;
            margin: 0;
        }}
        .metric-strip {{
            display: flex;
            gap: 0.6rem;
            flex-wrap: wrap;
            margin-top: 0.9rem;
        }}
        .metric-pill {{
            border-radius: 999px;
            padding: 0.38rem 0.75rem;
            background: var(--accent-soft);
            border: 1px solid rgba(217, 244, 255, 0.14);
            color: var(--accent);
            font-size: 0.82rem;
        }}
        .tool-card {{
            min-height: 220px;
        }}
        .tool-title {{
            font-family: Bahnschrift, "Segoe UI Variable Display", "Segoe UI", sans-serif;
            color: var(--text);
            font-size: 1.18rem;
            margin-bottom: 0.45rem;
        }}
        .tool-kicker {{
            color: var(--accent);
            font-size: 0.78rem;
            letter-spacing: 0.15em;
            text-transform: uppercase;
            margin-bottom: 0.55rem;
        }}
        .caption-label {{
            font-size: 0.74rem;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            color: var(--muted);
            margin-bottom: 0.3rem;
        }}
        .stTabs [data-baseweb="tab-list"] {{
            gap: 0.5rem;
        }}
        .stTabs [data-baseweb="tab"] {{
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border);
        }}
        .stButton > button, .stDownloadButton > button {{
            border-radius: 999px;
            border: 1px solid rgba(217, 244, 255, 0.18);
            background: linear-gradient(180deg, rgba(217, 244, 255, 0.12), rgba(143, 211, 232, 0.07));
            color: var(--text);
            font-weight: 600;
        }}
        .stButton > button:hover, .stDownloadButton > button:hover {{
            border-color: rgba(217, 244, 255, 0.32);
            color: var(--accent);
        }}
        div[data-testid="stPageLink"] {{
            margin-top: 0.9rem;
            margin-bottom: 1.4rem;
        }}
        div[data-testid="stPageLink"] a {{
            position: relative;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-width: 10.5rem;
            padding: 0.72rem 1.15rem;
            border-radius: 999px;
            border: 1px solid rgba(255, 76, 170, 0.34);
            background: linear-gradient(180deg, rgba(255, 76, 170, 0.14), rgba(255, 76, 170, 0.06));
            color: var(--text);
            font-weight: 700;
            letter-spacing: 0.03em;
            text-decoration: none !important;
            transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease, background 0.18s ease;
            overflow: visible;
        }}
        div[data-testid="stPageLink"] a:hover {{
            transform: translateY(-1px);
            border-color: rgba(255, 111, 191, 0.9);
            background: linear-gradient(180deg, rgba(255, 96, 184, 0.28), rgba(255, 76, 170, 0.12));
            box-shadow: 0 0 0 1px rgba(255, 111, 191, 0.16), 0 0 18px rgba(255, 58, 156, 0.55), 0 0 44px rgba(255, 58, 156, 0.34);
            color: #fff6fb;
        }}
        @media (max-width: 900px) {{
            .hero-grid {{
                grid-template-columns: 1fr;
            }}
            .hero-title {{
                font-size: 1.75rem;
            }}
        }}
        </style>
        <div class="hero-card">
            <div class="hero-grid">
                <div class="hero-logo">{logo_markup}</div>
                <div>
                    <div class="eyebrow">Netective Streamlit Studio</div>
                    <div class="hero-title">Interactive analysis for structure, similarity, benchmarking, and clustering</div>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(title: str, description: str, eyebrow: str) -> None:
    st.markdown(
        f"""
        <div class="section-card">
            <div class="eyebrow">{eyebrow}</div>
            <div class="hero-title" style="font-size:1.55rem; margin-bottom:0.5rem;">{title}</div>
            <p class="hero-copy">{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_tool_card(kicker: str, title: str, description: str) -> None:
    st.markdown(
        f"""
        <div class="tool-card">
            <div class="tool-kicker">{kicker}</div>
            <div class="tool-title">{title}</div>
            <p class="tool-copy">{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_file_preview(path: Path, lines: int = 8) -> None:
    preview = path.read_text(encoding="utf-8").splitlines()[:lines]
    st.caption(path.name)
    st.code("\n".join(preview) if preview else "<empty file>", language="text")


def example_bundle_bytes(paths: Iterable[Path]) -> bytes:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.writestr(path.name, path.read_bytes())
    buffer.seek(0)
    return buffer.getvalue()


def clean_selected_props(selected_props: list[str]) -> str | list[str]:
    return "all" if not selected_props else selected_props


def stage_input_files(file_inputs: Iterable, temp_dir: Path) -> list[Path]:
    staged_paths: list[Path] = []
    for file_input in file_inputs:
        if isinstance(file_input, Path):
            file_name = file_input.name
            file_bytes = file_input.read_bytes()
        else:
            file_name = file_input.name
            file_bytes = file_input.getvalue()

        suffix = Path(file_name).suffix.lower()
        if suffix in {".tsv", ".csv", ".txt", ".pred", ".graphml", ".edgelist"}:
            file_bytes = file_bytes.replace(b"\r\n", b"\n").replace(b"\r", b"\n")

        destination = temp_dir / file_name
        destination.write_bytes(file_bytes)
        staged_paths.append(destination)

    return staged_paths


def read_upload_preview(uploaded_file, lines: int = 8) -> str:
    preview_text = uploaded_file.getvalue().decode("utf-8", errors="replace")
    return "\n".join(preview_text.splitlines()[:lines]) or "<empty file>"


def workers_value(use_auto: bool, workers: int) -> str | int:
    return "auto" if use_auto else workers


def mixed_ba_values(raw_value: str) -> list[int | str]:
    values: list[int | str] = []
    for chunk in raw_value.split(","):
        item = chunk.strip()
        if not item:
            continue
        values.append(int(item) if item.isdigit() else item)
    return values or [2]


def figure_from_plot(plot_obj):
    if isinstance(plot_obj, matplotlib.figure.Figure):
        return plot_obj
    if hasattr(plot_obj, "get_figure"):
        return plot_obj.get_figure()
    raise TypeError("Object cannot be rendered as a matplotlib figure.")


def clusters_to_frame(clusters: dict) -> pd.DataFrame:
    rows = []
    for cluster_id, members in clusters.items():
        for member in members:
            rows.append({"network": member, "cluster": cluster_id})
    return pd.DataFrame(rows).sort_values(["cluster", "network"]).reset_index(drop=True)
