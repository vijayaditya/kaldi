#!/usr/bin/env python

from __future__ import print_function
import os
import argparse
import sys
import warnings
import copy
import imp
import ast

nodes = imp.load_source('', 'steps/nnet3/nodes.py')

def PrintConfig(file_name, config_lines):
    f = open(file_name, 'w')
    f.write("\n".join(config_lines['components'])+"\n")
    f.write("\n#Component nodes\n")
    f.write("\n".join(config_lines['component-nodes']))
    f.close()

def ParseSpliceString(splice_indexes, label_delay=None):
    ## Work out splice_array e.g. splice_array = [ [ -3,-2,...3 ], [0], [-2,2], .. [ -8,8 ] ]
    splice_array = []
    left_context = 0
    right_context = 0
    split1 = args.splice_indexes.split(" ");  # we already checked the string is nonempty.
    if len(split1) < 1:
        sys.exit("invalid --splice-indexes argument, too short: "
                 + args.splice_indexes)
    try:
        for string in split1:
            split2 = string.split(",")
            if len(split2) < 1:
                sys.exit("invalid --splice-indexes argument, too-short element: "
                         + args.splice_indexes)
            int_list = []
            for int_str in split2:
                int_list.append(int(int_str))
            if not int_list == sorted(int_list):
                sys.exit("elements of --splice-indexes must be sorted: "
                         + args.splice_indexes)
            left_context += -int_list[0]
            right_context += int_list[-1]
            splice_array.append(int_list)
    except ValueError as e:
        sys.exit("invalid --splice-indexes argument " + args.splice_indexes + e)
    left_context = max(0, left_context)
    right_context = max(0, right_context)
    num_hidden_layers = len(splice_array)
    input_dim = len(splice_array[0]) * args.feat_dim  +  args.ivector_dim

    return {'left_context':left_context,
            'right_context':right_context,
            'splice_indexes':splice_array,
            'num_hidden_layers':len(splice_array)
            }

if __name__ == "__main__":
    # we add compulsary arguments as named arguments for readability
    parser = argparse.ArgumentParser(description="Writes config files and variables "
                                                 "for CLSTMs creation and training",
                                     epilog="See steps/nnet3/lstm/train.sh for example.")
    # General neural network options
    parser.add_argument("--splice-indexes", type=str,
                        help="Splice indexes at input layer, e.g. '-3,-2,-1,0,1,2,3' [compulsary argument]", default="0")
    parser.add_argument("--feat-dim", type=int,
                        help="Raw feature dimension, e.g. 13")
    parser.add_argument("--ivector-dim", type=int,
                        help="iVector dimension, e.g. 100", default=0)
    parser.add_argument("--subset-dim", type=int, default=0,
                        help="dimension of the subset of units to be sent to the central frame")
    parser.add_argument("--pnorm-input-dim", type=int,
                        help="input dimension to p-norm nonlinearities")
    parser.add_argument("--pnorm-output-dim", type=int,
                        help="output dimension of p-norm nonlinearities")
    parser.add_argument("--relu-dim", type=int,
                        help="dimension of ReLU nonlinearities")
    parser.add_argument("--num-targets", type=int,
                        help="number of network targets (e.g. num-pdf-ids/num-leaves)")
    parser.add_argument("--use-presoftmax-prior-scale", type=str,
                        help="if true, a presoftmax-prior-scale is added",
                        choices=['true', 'false'], default = "true")
    parser.add_argument("config_dir",
                        help="Directory to write config files and variables")

    print(' '.join(sys.argv))

    args = parser.parse_args()
    print(args)
    if not os.path.exists(args.config_dir):
        os.makedirs(args.config_dir)

    ## Check arguments.
    if args.splice_indexes is None:
        sys.exit("--splice-indexes argument is required")
    if args.feat_dim is None or not (args.feat_dim > 0):
        sys.exit("--feat-dim argument is required")
    if args.num_targets is None or not (args.num_targets > 0):
        sys.exit("--num-targets argument is required")
    if (args.subset_dim < 0):
        sys.exit("--subset-dim has to be non-negative")

    if not args.relu_dim is None:
        if not args.pnorm_input_dim is None or not args.pnorm_output_dim is None:
            sys.exit("--relu-dim argument not compatible with "
                     "--pnorm-input-dim or --pnorm-output-dim options");
        nonlin_input_dim = args.relu_dim
        nonlin_output_dim = args.relu_dim
    else:
        if not args.pnorm_input_dim > 0 or not args.pnorm_output_dim > 0:
            sys.exit("--relu-dim not set, so expected --pnorm-input-dim and "
                     "--pnorm-output-dim to be provided.");
        nonlin_input_dim = args.pnorm_input_dim
        nonlin_output_dim = args.pnorm_output_dim

    prior_scale_file = '{0}/presoftmax_prior_scale.vec'.format(args.config_dir)
    if args.use_presoftmax_prior_scale == "true":
        use_presoftmax_prior_scale = True
    else:
        use_presoftmax_prior_scale = False

    parsed_splice_output = ParseSpliceString(args.splice_indexes.strip())
    num_hidden_layers = parsed_splice_output['num_hidden_layers']
    splice_indexes = parsed_splice_output['splice_indexes']

    config_lines = {'components':[], 'component-nodes':[]}

    config_files={}
    prev_layer_output = nodes.AddInputNode(config_lines, args.feat_dim, splice_indexes[0], args.ivector_dim)

    # Add the init config lines for estimating the preconditioning matrices
    init_config_lines = copy.deepcopy(config_lines)
    init_config_lines['components'].insert(0, '# Config file for initializing neural network prior to')
    init_config_lines['components'].insert(0, '# preconditioning matrix computation')
    nodes.AddOutputNode(init_config_lines, prev_layer_output)
    config_files[args.config_dir + '/init.config'] = init_config_lines

    prev_layer_output = nodes.AddLdaNode(config_lines, "L0", prev_layer_output, args.config_dir + '/lda.mat')

    # we moved the first splice layer to before the LDA..
    # so the input to the first affine layer is going to [0] index
    splice_indexes[0] = [0]
    for i in range(0, num_hidden_layers):
        # make the intermediate config file for layerwise discriminative
        # training
        # prepare the spliced input
        if not (len(splice_indexes[i]) == 1 and splice_indexes[i][0] == 0):
            try:
                zero_index = splice_indexes[i].index(0)
            except ValueError:
                zero_index = None
            # I just assume the prev_layer_output_descriptor is a simple forwarding descriptor
            prev_layer_output_descriptor = prev_layer_output['descriptor']
            subset_output = prev_layer_output
            if args.subset_dim > 0:
                # if subset_dim is specified the script expects a zero in the splice indexes
                assert(zero_index is not None)
                subset_node_config = "dim-range-node name=Tdnn_input_{0} input-node={1} dim-offset={2} dim={3}".format(i, prev_layer_output_descriptor, 0, args.subset_dim)
                subset_output = {'descriptor' : 'Tdnn_input_{0}'.format(i),
                                 'dimension' : args.subset_dim}
                config_lines['component-nodes'].append(subset_node_config)
            appended_descriptors = []
            appended_dimension = 0
            for j in range(len(splice_indexes[i])):
                if j == zero_index:
                    appended_descriptors.append(prev_layer_output['descriptor'])
                    appended_dimension += prev_layer_output['dimension']
                    continue
                appended_descriptors.append('Offset({0}, {1})'.format(subset_output['descriptor'], splice_indexes[i][j]))
                appended_dimension += subset_output['dimension']
            prev_layer_output = {'descriptor' : "Append({0})".format(" , ".join(appended_descriptors)),
                                 'dimension'  : appended_dimension}
        else:
            # this is a normal affine node
            pass
        prev_layer_output = nodes.AddAffRelNormNode(config_lines, "Tdnn_{0}".format(i),
                                                    prev_layer_output, nonlin_output_dim)
        nodes.AddFinalNode(config_lines, prev_layer_output, args.num_targets,
                           use_presoftmax_prior_scale = use_presoftmax_prior_scale,
                           prior_scale_file = prior_scale_file)

        config_files['{0}/layer{1}.config'.format(args.config_dir, i+1)] = config_lines
        config_lines = {'components':[], 'component-nodes':[]}

    left_context = int(parsed_splice_output['left_context'])
    right_context = int(parsed_splice_output['right_context'])

    # write the files used by other scripts like steps/nnet3/get_egs.sh
    f = open(args.config_dir + "/vars", "w")
    print('left_context=' + str(left_context), file=f)
    print('right_context=' + str(right_context), file=f)
    print('num_hidden_layers=' + str(num_hidden_layers), file=f)
    f.close()

    # printing out the configs
    # init.config used to train lda-mllt train
    for key in config_files.keys():
        PrintConfig(key, config_files[key])
