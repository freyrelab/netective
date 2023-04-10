import streamlit as st
import pandas as pd
from io import BytesIO
import networkx as nx
from PIL import Image
from netective.struct import Structure
from netective.utils import parse_nets
import os

# Set up the user interface
print('----------------------------path',os.getcwd())
image = Image.open(r'./assets/on_black.png')
width, height = image.size
st.image(image, width=int(width*0.1))
st.title("Compute network structural properties")
uploaded_file = st.file_uploader("Choose a file")

# Define the backend functionality
if uploaded_file is not None:
    # Read file contents as bytes
    file_bytes = uploaded_file.read()

    file_obj = BytesIO(file_bytes)


    # Create a networkx graph from the input string
    network = nx.read_edgelist(
        file_obj,
        delimiter=' ',
        create_using=nx.DiGraph,
        data=False
    )

    #print(network.edges, 'egessssssssssssssssssssssssss----------------------------------------------')

    # Compute structural properties of the graph
    S = Structure(network, norm='biol', net_id='x', verbose=True)
    output_data = S.get_props()

    # Convert output to a Pandas dataframe
    df = pd.DataFrame(output_data)
    st.subheader("Output:")
    st.dataframe(df)
    st.bar_chart(df)
