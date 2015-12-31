#!/usr/bin/env python

from __future__ import print_function
import os
import argparse
import sys
import warnings
import copy
import imp
import ast

nodes = imp.load_source('', 'steps/nnet3/components.py')

def CheckRatewiseParams(ratewise_params):
    #TODO : write this
    return True

def PrintConfig(file_name, config_lines):
    f = open(file_name, 'w')
    f.write("\n".join(config_lines['components'])+"\n")
    f.write("\n#Component nodes\n")
    f.write("\n".join(config_lines['component-nodes']))
    f.close()

def ParseSpliceString(splice_indexes, label_delay=None):
    ## Work out splice_array e.g. splice_array = [ [ -3,-2,...3 ], [0], [-2,2], .. [ -8,8 ] ]
    split1 = splice_indexes.split(" ");  # we already checked the string is nonempty.
    if len(split1) < 1:
        splice_indexes = "0"

    left_context = 0
    right_context = 0
    if label_delay is not None:
        left_context = -label_delay
        right_context = label_delay
    splice_array = []
    try:
        for i in range(len(split1)):
            indexes = map(lambda x: int(x), split1[i].strip().split(","))
            print(indexes)
            if len(indexes) < 1:
                raise ValueError("invalid --splice-indexes argument, too-short element: "
                                + splice_indexes)

            if (i > 0)  and ((len(indexes) != 1) or (indexes[0] != 0)):
                raise ValueError("elements of --splice-indexes splicing is only allowed initial layer.")

            if not indexes == sorted(indexes):
                raise ValueError("elements of --splice-indexes must be sorted: "
                                + splice_indexes)
            left_context += -indexes[0]
            right_context += indexes[-1]
            splice_array.append(indexes)
    except ValueError as e:
        raise ValueError("invalid --splice-indexes argument " + splice_indexes + str(e))

    left_context = max(0, left_context)
    right_context = max(0, right_context)

    return {'left_context':left_context,
            'right_context':right_context,
            'splice_indexes':splice_array,
            'num_hidden_layers':len(splice_array)
            }

if __name__ == "__main__":
    # we add compulsary arguments as named arguments for readability
    parser = argparse.ArgumentParser(description="Writes config files and variables "
                                                 "for CWRNNs creation and training",
                                     epilog="See steps/nnet3/lstm/train.sh for example.")
    # General neural network options
    parser.add_argument("--splice-indexes", type=str,
                        help="Splice indexes at input layer, e.g. '-3,-2,-1,0,1,2,3' [compulsary argument]", default="0")
    parser.add_argument("--feat-dim", type=int,
                        help="Raw feature dimension, e.g. 13")
    parser.add_argument("--ivector-dim", type=int,
                        help="iVector dimension, e.g. 100", default=0)
    parser.add_argument("--xent-regularize", type=float,
                        help="For chain models, if nonzero, add a separate output for cross-entropy "
                        "regularization (with learning-rate-factor equal to the inverse of this)",
                        default=0.0)
    parser.add_argument("--include-log-softmax", type=str,
                        help="add the final softmax layer ", default="true", choices = ["false", "true"])

    parser.add_argument("--num-cwrnn-layers", type=int,
                        help="Number of CWRNN layers to be stacked", default=1)
    parser.add_argument("--hidden-dim", type=int,
                        help="dimension of fully-connected layers")
    
    parser.add_argument("--ratewise-params", type=str, default=None,
                        help="the parameters for CWRNN units operating at different rates of operation in each clockwork-RNN")
    parser.add_argument("--num-lpfilter-taps", type=int, default=None,
                        help="number of taps in the low-pass filters used for the smoothing the clock-rnn inputs of different rates, before downsampling")
    parser.add_argument("--nonlinearity", type=str, default=None, choices = ['SigmoidComponent', 'TanhComponent', 'RectifiedLinearComponent+NormalizeComponent', 'RectifiedLinearComponent'],
                        help="type of non-linearity to be used in CWRNN")
    parser.add_argument("--diag-init-scaling-factor", type=float, default=0.0,
                        help="If non-zero the diagonal initialization of affinematrix is enabled, linear-params are diagonal scaled with the specified value and bias-params are 0")
    parser.add_argument("--projection-dim", type=int, default=0,
                        help="If non-zero the output of the CWRNN unit will be projected to this dimension")

    parser.add_argument("--input-type", type=str, default="smooth", choices = ["smooth", "stack", "sum", "none"],
                        help="""It can take one of the three values {'smooth', 'stack', 'sum', 'none'}.
                                smooth: input is low-pass filtered using a sinc filter with hamming window smoothing (non-causal)
                                stack:  input from the time steps skipped by the CWRNN unit is stacked at its input (this can get pretty huge for large time-periods)
                                sum:    input from the time steps skipped by the CWRNN unit is summed at its input
                                none:   input from the time steps skipped is ignored
                                """)

    parser.add_argument("--filter-input-step", type=int,
                        help="The distance between time steps used as input for splicing layers ", default=1)
    parser.add_argument("--subsample", type=str,
                        help="if true subsample the clockwork units, else operate at the same rate ", default="true", choices = ["false", "true"])
    parser.add_argument("--use-lstm", type=str,
                        help="if true clockwork units will be chosen to be LSTMs rather than simple RNNs", default="true", choices = ["false", "true"])
    # Natural gradient options
    parser.add_argument("--ng-per-element-scale-options", type=str,
                        help="options to be supplied to NaturalGradientPerElementScaleComponent", default="")
    parser.add_argument("--ng-affine-options", type=str,
                        help="options to be supplied to NaturalGradientAffineComponent", default="")

    # Gradient clipper options
    parser.add_argument("--norm-based-clipping", type=str,
                        help="use norm based clipping in ClipGradient components ", default="false", choices = ["false", "true"])
    parser.add_argument("--clipping-threshold", type=float,
                        help="clipping threshold used in ClipGradient components, if clipping-threshold=0 no clipping is done", default=15)

    parser.add_argument("--num-targets", type=int,
                        help="number of network targets (e.g. num-pdf-ids/num-leaves)")
    parser.add_argument("config_dir",
                        help="Directory to write config files and variables")

    # Delay options
    parser.add_argument("--label-delay", type=int, default=None,
                        help="option to delay the labels to make the lstm robust")

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
    if (args.num_cwrnn_layers < 1):
        sys.exit("--num-cwrnn-layers has to be a positive integer")
    if (args.clipping_threshold < 0):
        sys.exit("--clipping-threshold has to be a non-negative")
    if (args.projection_dim < 0):
        sys.exit("--projection-dim has to be non-negative")
    subsample = True
    if (args.subsample == "false"):
        subsample = False
    use_lstm = False
    if (args.use_lstm == "true"):
        use_lstm = True
    if args.ratewise_params is None:
        ratewise_params = {'T1': {'rate':1, 'dim':512},
                           'T2': {'rate':1.0/2, 'dim':256},
                           'T3': {'rate':1.0/4, 'dim':256},
                           'T4': {'rate':1.0/8, 'dim':256}
                          }
    else:
        ratewise_params = eval(args.ratewise_params)
        assert(CheckRatewiseParams(ratewise_params))
    if (args.filter_input_step <= 0):
        raise Exception("Filter input step should be greater than 0")

    include_log_softmax = True
    if (args.include_log_softmax == 'false'):
        include_log_softmax = False

    input_type = args.input_type

    for key in ratewise_params.keys():
        if ratewise_params[key]['rate'] > 1 :
            raise Exception("Rates cannot be greater than 1")

    if (args.num_lpfilter_taps is not None) and (args.num_lpfilter_taps % 2 == 0):
        warnings.warn("Only odd tap filters are allowed, adding a tap")
        args.num_lpfilter_taps += 1
    parsed_splice_output = ParseSpliceString(args.splice_indexes.strip(), args.label_delay)
    num_hidden_layers = parsed_splice_output['num_hidden_layers']
    splice_indexes = parsed_splice_output['splice_indexes']
    if (num_hidden_layers < args.num_cwrnn_layers):
        sys.exit("--num-cwrnn-layers : number of lstm layers has to be greater than number of layers, decided based on splice-indexes")

    config_lines = {'components':[], 'component-nodes':[]}

    config_files={}
    prev_layer_output = nodes.AddInputLayer(config_lines, args.feat_dim, splice_indexes[0], args.ivector_dim)

    # Add the init config lines for estimating the preconditioning matrices
    init_config_lines = copy.deepcopy(config_lines)
    init_config_lines['components'].insert(0, '# Config file for initializing neural network prior to')
    init_config_lines['components'].insert(0, '# preconditioning matrix computation')
    nodes.AddOutputLayer(init_config_lines, prev_layer_output)
    config_files[args.config_dir + '/init.config'] = init_config_lines

    prev_layer_output = nodes.AddLdaLayer(config_lines, "L0", prev_layer_output, args.config_dir + '/lda.mat')

    extra_left_context = 0
    for i in range(args.num_cwrnn_layers):
        warnings.warn("CWRNN units do not use the ng_affine_options specified")
        [prev_layer_output, num_lpfilter_taps, largest_time_period] = nodes.AddCwrnnLayer(config_lines, "Cwrnn{0}".format(i+1),
                                                                                         prev_layer_output, '{0}/cwrnn_layer{1}_lp_filts.txt'.format(args.config_dir, i),
                                                                                         args.num_lpfilter_taps,
                                                                                         clipping_threshold = args.clipping_threshold,
                                                                                         norm_based_clipping = args.norm_based_clipping,
                                                                                         ratewise_params = ratewise_params,
                                                                                         nonlinearity = args.nonlinearity,
                                                                                         input_type = input_type,
                                                                                         subsample = subsample,
                                                                                         filter_input_step = args.filter_input_step,
                                                                                         diag_init_scaling_factor = args.diag_init_scaling_factor,
                                                                                         use_lstm = use_lstm,
                                                                                         projection_dim = args.projection_dim)
        if input_type in set(["stack", "sum"]):
            extra_left_context += largest_time_period
        # make the intermediate config file for layerwise discriminative
        # training
        nodes.AddFinalLayer(config_lines, prev_layer_output, args.num_targets, label_delay = args.label_delay, include_log_softmax = include_log_softmax)

        if args.xent_regularize != 0.0:
            nodes.AddFinalLayer(config_lines, prev_layer_output, args.num_targets,
                                include_log_softmax = True,
                                name_affix = 'xent')

        config_files['{0}/layer{1}.config'.format(args.config_dir, i+1)] = config_lines
        config_lines = {'components':[], 'component-nodes':[]}

    for i in range(args.num_cwrnn_layers, num_hidden_layers):
        prev_layer_output = nodes.AddAffRelNormLayer(config_lines, "L{0}".format(i+1),
                                               prev_layer_output, args.hidden_dim)
        # make the intermediate config file for layerwise discriminative
        # training
        nodes.AddFinalLayer(config_lines, prev_layer_output, args.num_targets, label_delay = args.label_delay, include_log_softmax = include_log_softmax)

        if args.xent_regularize != 0.0:
            nodes.AddFinalLayer(config_lines, prev_layer_output, args.num_targets,
                                include_log_softmax = True,
                                name_affix = 'xent')

        config_files['{0}/layer{1}.config'.format(args.config_dir, i+1)] = config_lines
        config_lines = {'components':[], 'component-nodes':[]}


    filter_context = (num_lpfilter_taps - 1)/2 # assuming symmetric filters
    left_context = int(parsed_splice_output['left_context'] + extra_left_context)
    right_context = int(parsed_splice_output['right_context'])
    if input_type == "smooth":
        left_context += int(args.num_cwrnn_layers * filter_context)
        right_context += int(args.num_cwrnn_layers * filter_context)


    # write the files used by other scripts like steps/nnet3/get_egs.sh
    f = open(args.config_dir + "/vars", "w")
    print('model_left_context=' + str(left_context), file=f)
    print('model_right_context=' + str(right_context), file=f)
    print('num_hidden_layers=' + str(num_hidden_layers), file=f)
    # print('initial_right_context=' + str(splice_array[0][-1]), file=f)
    f.close()

    # printing out the configs
    # init.config used to train lda-mllt train
    for key in config_files.keys():
        PrintConfig(key, config_files[key])
