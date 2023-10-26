import os
import streamlit as st
import pandas as pd
from io import BytesIO
import networkx as nx
from PIL import Image
import inspect
import numpy as np
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from itertools import chain


from netective import characterize_network, compare_structure
from netective.structure.dataviz import (plot_distributions, plot_scalars)


# Set up the user interface
print("----------------------------path", os.getcwd())
image = Image.open(r"./assets/on_white.png")
width, height = image.size
st.image(image, width=int(width * 0.15))
st.title("Compute network structural properties")
uploaded_files = st.file_uploader("Choose up to five files", accept_multiple_files=True)




def main_single(G, norm="network"):
    fig_scalar, fig_dist = characterize_network(G, norm=norm)

    # Create two columns for plots
    col1, col2 = st.columns(2)

    with col1:
        st.pyplot(fig_scalar)  # , use_container_width=False)
    with col2:
        st.pyplot(fig_dist)


def main_multiple(uploaded_files, norm, directed):
    # Iterate over the uploaded files and perform the necessary operations
    networks = {}
    for i, uploaded_file in enumerate(uploaded_files):
        st.write(f"Processing file {i+1}: {uploaded_file.name}")
        file_bytes = uploaded_file.read()
        file_obj = BytesIO(file_bytes)
        G = nx.read_edgelist(file_obj, delimiter=" ", create_using=nx.DiGraph if directed else nx.Graph, data=False)
        networks[uploaded_file.name] = G

    
    fig_scalar = compare_structure(networks, norm=norm)

    st.pyplot(fig_scalar)  # , use_container_width=False)


# Define the backend functionality
norm_options = [None, "biological", "network"]
norm = st.selectbox("Normalization", norm_options)
directed = st.checkbox("Directed networks", value=True)

if uploaded_files:
    num_files = len(uploaded_files)
    if num_files == 1:
        # Process for a single file
        file_bytes = uploaded_files[0].read()
        file_obj = BytesIO(file_bytes)
        G = nx.read_edgelist(file_obj, delimiter=" ", create_using=nx.DiGraph if directed else nx.Graph, data=False)
        main_single(G, norm=norm)
    elif num_files <= 5:
        # Process for multiple files
        main_multiple(uploaded_files, norm=norm, directed=directed)
    else:
        # Show an error message if more than 3 files are uploaded
        st.error("You can upload up to five files.")
