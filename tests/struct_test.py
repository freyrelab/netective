import pytest
import pandas as pd
import subprocess

# import netective as nt


def test_structural_properties_merenets():
    g_standard = pd.read_csv(
        r".\tests\structural_properties_merenets_er.tsv",
        sep="\t",
        comment="#",
        index_col=0,
        header=0,
    )
    mask = ~g_standard.index.str.contains("ER_avg")
    g_standard = g_standard.loc[mask]

    # Run the command in the terminal
    command = r"netective --comments # --delimiter \t --output_file tests\structural_properties_merenets_er_testing --workers 4 --erdos_renyi 2 --path .\tests\merenets"
    print(command)
    result = subprocess.run(command, shell=True, capture_output=False)

    new_df = pd.read_csv(
        r".\tests\structural_properties_merenets_er_testing.tsv",
        sep="\t",
        comment="#",
        index_col=0,
        header=0,
    )
    mask = ~new_df.index.str.contains("ER_avg")
    new_df = new_df.loc[mask]

    # Check if the output is the same as the standard
    assert g_standard.equals(
        new_df
    ), "----------------- !!!!!!!! The output of the command is not the same as the standard"
