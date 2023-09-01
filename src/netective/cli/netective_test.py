from pandas import DataFrame

from netective.cli import _arguments
from netective.structure.structure import save_strucs
from netective.structure import structure
from netective.utils import *

try:
    import pretty_traceback

    pretty_traceback.install()
except ImportError:
    pass


def main():

    ## parse arguments
    args = _arguments._parse_arguments()
    paths = args.path
    comments = args.comments
    delimiter = args.delimiter
    output = args.output
    output_file = args.output_file
    norm = args.norm
    erdos_renyi = args.erdos_renyi
    try:
        workers = int(args.workers)
    except ValueError:
        workers = args.workers
    verbose = args.verbose

    # collect data for parallel processing
    networks = parse_nets(paths, comments, delimiter)
    if len(networks.values()) > 1:
        fig_scalar, fig_dist = structure.compare_structure(networks, norm, workers=workers)
    else:
        networks = list(networks.values())
        structure.characterize_network(networks[0])

    ## save results
    cl = f"{comments} command: python {__file__} --path {paths} --norm {norm} --comments {comments} --delimiter {delimiter} --output {output} --output_file {output_file} --erdos_renyi {erdos_renyi} --workers {workers} --verbose {verbose}"
    save_strucs(fig_scalar, fig_dist, output, delimiter, cl, output_file)


## read arguments
if __name__ == "__main__":
    main()


# ..\..\data\test