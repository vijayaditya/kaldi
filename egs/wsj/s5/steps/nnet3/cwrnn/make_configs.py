#!/usr/bin/env python

from __future__ import print_function
import os
import argparse
import sys
import warnings
import copy
import imp

nodes = imp.load_source('nodes', 'steps/nnet3/components.py')
nnet3_train_lib = imp.load_source('ntl', 'steps/nnet3/nnet3_train_lib.py')
chain_lib = imp.load_source('ncl', 'steps/nnet3/chain/nnet3_chain_lib.py')

def GetArgs():
    # we add compulsary arguments as named arguments for readability
    parser = argparse.ArgumentParser(description="Writes config files and variables "
                                                 "for CWRNNs creation and training",
                                     epilog="See swbd/s5c/local/nnet3/run_cwrnn*.sh for example.")

    # Only one of these arguments can be specified, and one of them has to
    # be compulsarily specified
    feat_group = parser.add_mutually_exclusive_group(required = True)
    feat_group.add_argument("--feat-dim", type=int,
                            help="Raw feature dimension, e.g. 13")
    feat_group.add_argument("--feat-dir", type=str,
                            help="Feature directory, from which we derive the feat-dim")

    # only one of these arguments can be specified
    ivector_group = parser.add_mutually_exclusive_group(required = False)
    ivector_group.add_argument("--ivector-dim", type=int,
                                help="iVector dimension, e.g. 100", default=0)
    ivector_group.add_argument("--ivector-dir", type=str,
                                help="iVector dir, which will be used to derive the ivector-dim  ", default=None)

    num_target_group = parser.add_mutually_exclusive_group(required = True)
    num_target_group.add_argument("--num-targets", type=int,
                                  help="number of network targets (e.g. num-pdf-ids/num-leaves)")
    num_target_group.add_argument("--ali-dir", type=str,
                                  help="alignment directory, from which we derive the num-targets")
    num_target_group.add_argument("--tree-dir", type=str,
                                  help="directory with final.mdl, from which we derive the num-targets")

    # General neural network options
    parser.add_argument("--splice-indexes", type=str,
                        help="Splice indexes at input layer, e.g. '-3,-2,-1,0,1,2,3'", required = True, default="0")
    parser.add_argument("--xent-regularize", type=float,
                        help="For chain models, if nonzero, add a separate output for cross-entropy "
                        "regularization (with learning-rate-factor equal to the inverse of this)",
                        default=0.0)
    parser.add_argument("--include-log-softmax", type=str, action=nnet3_train_lib.StrToBoolAction,
                        help="add the final softmax layer ", default=True, choices = ["false", "true"])

    # CWRNN options
    parser.add_argument("--num-cwrnn-layers", type=int,
                        help="Number of CWRNN layers to be stacked", default=1)
    parser.add_argument("--ratewise-params", type=str, default=None,
                        help="the parameters for CWRNN units operating at different rates of operation in each clockwork-RNN")
    parser.add_argument("--nonlinearity", type=str, default=None, choices = ['SigmoidComponent', 'TanhComponent', 'RectifiedLinearComponent+NormalizeComponent', 'RectifiedLinearComponent'],
                        help="type of non-linearity to be used in CWRNN")
    parser.add_argument("--diag-init-scaling-factor", type=float, default=0.0,
                        help="If non-zero the diagonal initialization of affinematrix is enabled, linear-params are diagonal scaled with the specified value and bias-params are 0")
    parser.add_argument("--projection-dim", type=int, default=0,
                        help="If non-zero the output of the CWRNN unit will be projected to this dimension")
    parser.add_argument("--input-type", type=str, default="per-dim-weighted-average", choices = ["per-dim-weighted-average", "stack", "sum", "none"],
                        help="""It can take one of the three values {'per-dim-weighted-average', 'stack', 'sum', 'none'}.
                                per-dim-weighted-average: temporal only filter along each dimension
                                stack:  input from the time steps skipped by the CWRNN unit is stacked at its input (this can get pretty huge for large time-periods)
                                sum:    input from the time steps skipped by the CWRNN unit is summed at its input
                                none:   input from the time steps skipped is ignored
                                """)
    parser.add_argument("--operating-time-period", type=int,
                        help="The distance between time steps used at CWRNN input", default=1)
    parser.add_argument("--hidden-dim", type=int,
                        help="dimension of fully-connected layers")

    # Natural gradient options
    parser.add_argument("--ng-affine-options", type=str,
                        help="options to be supplied to NaturalGradientAffineComponent", default="")

    # Gradient clipper options
    parser.add_argument("--norm-based-clipping", type=str, action=nnet3_train_lib.StrToBoolAction,
                        help="use norm based clipping in ClipGradient components ", default=True, choices = ["false", "true"])
    parser.add_argument("--clipping-threshold", type=float,
                        help="clipping threshold used in ClipGradient components, if clipping-threshold=0 no clipping is done", default=30)
    parser.add_argument("--self-repair-scale-nonlinearity", type=float,
                        help="A non-zero value activates the self-repair mechanism in the non-linearities of the CWRNN", default=0.00001)
    parser.add_argument("--self-repair-scale-clipgradient", type=float,
                        help="A non-zero value activates the self-repair mechanism in the ClipGradient component of the CWRNN", default=1.0)

    # Delay options
    parser.add_argument("--label-delay", type=int, default=None,
                        help="option to delay the labels to make the lstm robust")

    parser.add_argument("config_dir",
                        help="Directory to write config files and variables")

    print(' '.join(sys.argv))

    args = parser.parse_args()
    args = CheckArgs(args)

    return args

def CheckArgs(args):
    if not os.path.exists(args.config_dir):
        os.makedirs(args.config_dir)

    ## Check arguments.
    if args.feat_dir is not None:
        args.feat_dim = nnet3_train_lib.GetFeatDim(args.feat_dir)

    if args.ali_dir is not None:
        args.num_targets = nnet3_train_lib.GetNumberOfLeaves(args.ali_dir)
    elif args.tree_dir is not None:
        args.num_targets = chain_lib.GetNumberOfLeaves(args.tree_dir)

    if args.ivector_dir is not None:
        args.ivector_dim = nnet3_train_lib.GetIvectorDim(args.ivector_dir)

    if not args.feat_dim > 0:
        raise Exception("feat-dim has to be postive")

    if not args.num_targets > 0:
        print(args.num_targets)
        raise Exception("num_targets has to be positive")

    if not args.ivector_dim >= 0:
        raise Exception("ivector-dim has to be non-negative")

    if (args.num_cwrnn_layers < 1):
        sys.exit("--num-cwrnn-layers has to be a positive integer")
    if (args.clipping_threshold < 0):
        sys.exit("--clipping-threshold has to be a non-negative")

    if (args.projection_dim < 0):
        sys.exit("--projection-dim has to be non-negative")

    if args.ratewise_params is None:
        args.ratewise_params = {'T1': {'rate':1, 'dim':512},
                                'T2': {'rate':1.0/2, 'dim':256},
                                'T3': {'rate':1.0/4, 'dim':256},
                                'T4': {'rate':1.0/8, 'dim':256}
                                }
    else:
        args.ratewise_params = eval(args.ratewise_params)
        assert(CheckRatewiseParams(args.ratewise_params))
    if (args.operating_time_period <= 0):
        raise Exception("--operating-time-period should be greater than 0")

    for key in args.ratewise_params.keys():
        if args.ratewise_params[key]['rate'] > 1 :
            raise Exception("Rates cannot be greater than 1")

    return args

def CheckRatewiseParams(ratewise_params):
    #TODO : write this
    return True

def PrintConfig(file_name, config_lines):
    f = open(file_name, 'w')
    f.write("\n".join(config_lines['components'])+"\n")
    f.write("\n#Component nodes\n")
    f.write("\n".join(config_lines['component-nodes'])+"\n")
    f.close()

def ParseSpliceString(splice_indexes, label_delay=None):
    ## Work out splice_array e.g. splice_array = [ [ -3,-2,...3 ], [0], [-2,2], .. [ -8,8 ] ]
    split1 = splice_indexes.split(" ");  # we already checked the string is nonempty.
    if len(split1) < 1:
        splice_indexes = "0"

    left_context=0
    right_context=0
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

def ProcessSpliceIndexes(config_dir, splice_indexes, label_delay, num_cwrnn_layers):
    parsed_splice_output = ParseSpliceString(splice_indexes.strip(), label_delay)
    left_context = parsed_splice_output['left_context']
    right_context = parsed_splice_output['right_context']
    num_hidden_layers = parsed_splice_output['num_hidden_layers']
    splice_indexes = parsed_splice_output['splice_indexes']

    if (num_hidden_layers < num_cwrnn_layers):
        raise Exception("--num-cwrnn-layers : number cwrnn layers cannot be greater than number of layers (decided based on splice-indexes)")

    return [left_context, right_context, num_hidden_layers, splice_indexes]

def MakeConfigs(config_dir, feat_dim, ivector_dim, num_targets,
                splice_indexes_string,
                ratewise_params, operating_time_period,
                hidden_dim, projection_dim, num_cwrnn_layers,
                norm_based_clipping, clipping_threshold,
                ng_affine_options, nonlinearity, diag_init_scaling_factor,
                input_type,label_delay, include_log_softmax, xent_regularize,
                self_repair_scale_nonlinearity, self_repair_scale_clipgradient):

    [left_context, right_context, num_hidden_layers, splice_indexes] = ProcessSpliceIndexes(config_dir, splice_indexes_string, label_delay, num_cwrnn_layers)

    config_lines = {'components':[], 'component-nodes':[]}

    config_files={}

    # we will count the total number of learnable parameters in the model
    # we will count the xent branch parameters separately as these will be discarded
    # after training
    num_learnable_params = 0
    num_learnable_params_xent = 0

    prev_layer = nodes.AddInputLayer(config_lines, feat_dim, splice_indexes[0], ivector_dim)
    prev_layer_output = prev_layer['output']

    # Add the init config lines for estimating the preconditioning matrices
    init_config_lines = copy.deepcopy(config_lines)
    init_config_lines['components'].insert(0, '# Config file for initializing neural network prior to')
    init_config_lines['components'].insert(0, '# preconditioning matrix computation')
    nodes.AddOutputLayer(init_config_lines, prev_layer_output)
    config_files[config_dir + '/init.config'] = init_config_lines

    prev_layer = nodes.AddLdaLayer(config_lines, "L0", prev_layer_output, config_dir + '/lda.mat')
    prev_layer_output = prev_layer['output']

    extra_left_context = 0
    for i in range(num_cwrnn_layers):
        prev_layer = nodes.AddCwrnnLayer(config_lines,
                                       name = "Cwrnn{0}".format(i+1),
                                       input = prev_layer_output,
                                       operating_time_period = operating_time_period,
                                       clipping_threshold = clipping_threshold,
                                       norm_based_clipping = norm_based_clipping,
                                       ratewise_params = ratewise_params,
                                       nonlinearity = nonlinearity,
                                       diag_init_scaling_factor = diag_init_scaling_factor,
                                       input_type = input_type,
                                       projection_dim = projection_dim,
                                       self_repair_scale_nonlinearity = self_repair_scale_nonlinearity,
                                       self_repair_scale_clipgradient = self_repair_scale_clipgradient)

        prev_layer_output = prev_layer['output']
        largest_time_period = prev_layer['largest_time_period']
        num_learnable_params += prev_layer['num_learnable_params']

        if input_type in set(["stack", "sum", "per-dim-weighted-average"]):
            extra_left_context += largest_time_period
        # make the intermediate config file for layerwise discriminative
        # training
        num_learnable_params_final_layer = nodes.AddFinalLayer(config_lines, prev_layer_output, num_targets, ng_affine_options, label_delay = label_delay, include_log_softmax = include_log_softmax)


        if xent_regularize != 0.0:
            num_learnable_params_xent_final_layer = nodes.AddFinalLayer(config_lines, prev_layer_output, num_targets,
                                include_log_softmax = True, label_delay = label_delay,
                                name_affix = 'xent')

        config_files['{0}/layer{1}.config'.format(config_dir, i+1)] = config_lines
        config_lines = {'components':[], 'component-nodes':[]}

    for i in range(num_cwrnn_layers, num_hidden_layers):
        prev_layer = nodes.AddAffRelNormLayer(config_lines, "L{0}".format(i+1),
                                               prev_layer_output, hidden_dim,
                                               ng_affine_options, self_repair_scale = self_repair_scale_nonlinearity)
        prev_layer_output = prev_layer['output']
        num_learnable_params += prev_layer['num_learnable_params']

        # make the intermediate config file for layerwise discriminative
        # training
        num_learnable_params_final_layer = nodes.AddFinalLayer(config_lines, prev_layer_output, num_targets, ng_affine_options, label_delay = label_delay, include_log_softmax = include_log_softmax)

        if xent_regularize != 0.0:
            num_learnable_params_xent_final_layer = nodes.AddFinalLayer(config_lines, prev_layer_output, num_targets,
                                include_log_softmax = True, label_delay = label_delay,
                                name_affix = 'xent')

        config_files['{0}/layer{1}.config'.format(config_dir, i+1)] = config_lines
        config_lines = {'components':[], 'component-nodes':[]}


    # now add the parameters from the final layer
    num_learnable_params += num_learnable_params_final_layer
    num_learnable_params_xent += num_learnable_params_xent_final_layer

    # printing out the configs
    # init.config used to train lda-mllt train
    for key in config_files.keys():
        PrintConfig(key, config_files[key])



    left_context = left_context + extra_left_context

    # write the files used by other scripts like steps/nnet3/get_egs.sh
    f = open(config_dir + "/vars", "w")
    print('model_left_context=' + str(int(left_context)), file=f)
    print('model_right_context=' + str(int(right_context)), file=f)
    print('num_hidden_layers=' + str(int(num_hidden_layers)), file=f)
    print('initial_right_context=' + str(splice_indexes[0][-1]), file=f)
    print('num_learable_params=' + str(num_learnable_params), file=f)
    print('num_learable_params_xent=' + str(num_learnable_params_xent), file=f)

    f.close()

    print('This model has num_learnable_params={0:,} and num_learnable_params_xent={1:,}'.format(num_learnable_params, num_learnable_params_xent))

def Main():
    args = GetArgs()

    MakeConfigs(config_dir = args.config_dir,
                feat_dim = args.feat_dim, ivector_dim = args.ivector_dim,
                num_targets = args.num_targets,
                splice_indexes_string = args.splice_indexes,
                ratewise_params = args.ratewise_params,
                operating_time_period = args.operating_time_period,
                hidden_dim = args.hidden_dim,
                projection_dim = args.projection_dim,
                num_cwrnn_layers = args.num_cwrnn_layers,
                norm_based_clipping = args.norm_based_clipping,
                clipping_threshold = args.clipping_threshold,
                ng_affine_options = args.ng_affine_options,
                nonlinearity = args.nonlinearity,
                diag_init_scaling_factor = args.diag_init_scaling_factor,
                input_type = args.input_type,
                label_delay = args.label_delay,
                include_log_softmax = args.include_log_softmax,
                xent_regularize = args.xent_regularize,
                self_repair_scale_nonlinearity = args.self_repair_scale_nonlinearity,
                self_repair_scale_clipgradient = args.self_repair_scale_clipgradient)

if __name__ == "__main__":
    Main()
