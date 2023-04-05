from pandas import DataFrame

from netective.cli import _arguments
from netective.struct import struc_props_call, save_strucs
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
    workers = args.workers
    verbose = args.verbose

    # collect data for parallel processing
    networks = parse_nets(paths, comments, delimiter)
    net_names, Gs = zip(*networks.items())
    data = [Gs, net_names, [norm]*len(net_names), [erdos_renyi]*len(net_names), [verbose]*len(net_names)]
    net_props = dict(run_parallel(struc_props_call, data, workers=workers))

    # create DataFrame
    df = DataFrame.from_dict(net_props).T

    ## save results
    cl = f'{comments} command: python {__file__} --path {paths} --norm {norm} --comments {comments} --delimiter {delimiter} --output {output} --output_file {output_file} --erdos_renyi {erdos_renyi} --workers {workers} --verbose {verbose}'
    save_strucs(df, output, delimiter, cl, output_file)

## read arguments
if __name__ == '__main__':
    main()
