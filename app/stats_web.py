import streamlit as st
import os
from shutil import copyfile
from PIL import Image
from netective.stats import stats
import matplotlib.pyplot as plt

# Header
image = Image.open(r"./assets/on_white.png")
width, height = image.size
st.image(image, width=int(width * 0.15))
st.title("Evaluate a predicted network")

# Function to copy the uploaded file to a specified directory
def copy_to_directory(file, destination_dir):
    file_path = os.path.join(destination_dir, file.name)
    with open(file_path, "wb") as dest_file:
        dest_file.write(file.read())
    return file_path


# File Uploads for Gold Standard and Prediction
gold_standard_file = st.file_uploader("Upload Gold Standard File", type=None)
prediction_file = st.file_uploader("Upload Prediction File", type=None)

# Specify paths explicitly
gold_standard_dir = os.getcwd()
prediction_dir = os.getcwd()

# Checkboxes and Float Input
greater_is_better = st.checkbox(
    "Greater is better",
    value=False,
    help="Check if a higher score means a more significant interaction. Only used if the inference file has a score column. Otherwise, it is ignored.",
)
directed = st.checkbox(
    "Directed network",
    value=False,
    help="Check if you want to consider the direction of the network in the assessment.",
)
allow_self_loops = st.checkbox(
    "Self-loops are considered",
    value=False,
    help="Check if you want to consider self-loops in the assessment.",
)
cutoff_enabled = st.checkbox("Cutoff Value", value=False, help="Check to enable the cutoff value.")

# Set cutoff_value to False if cutoff is not enabled
if cutoff_enabled:
    cutoff_value = st.number_input("Cutoff value", value=0.5, step=0.1)
else:
    cutoff_value = False

# Run Button
if st.button("Run"):
    if gold_standard_file is not None and prediction_file is not None:
        # Copy uploaded files to specified directories
        gold_standard_path = copy_to_directory(gold_standard_file, gold_standard_dir)
        prediction_path = copy_to_directory(prediction_file, prediction_dir)

        evaluation = stats.NetworkInferenceStats(
            gold_standard_path,
            prediction_path,
            greater_is_better,
            directed,
            allow_self_loops,
            cutoff_value,
        )
        fig, ax = plt.subplots(ncols=2, figsize=(6, 3))
        evaluation.plot_precision_recall_curve(ax=ax[0])
        evaluation.plot_roc_curve(ax=ax[1])
        plt.tight_layout()
        st.pyplot(fig)

        st.success("Evaluation completed successfully!")
