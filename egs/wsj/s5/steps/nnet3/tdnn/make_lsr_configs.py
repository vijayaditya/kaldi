#!/usr/bin/env python

# we're using python 3.x style print but want it to work in python 2.x,
from __future__ import print_function
import os
import argparse
import shlex
import sys
import warnings
import copy
import imp
import ast
import pprint

nodes = imp.load_source('', 'steps/nnet3/components.py')
nnet3_train_lib = imp.load_source('ntl', 'steps/nnet3/nnet3_train_lib.py')
chain_lib = imp.load_source('ncl', 'steps/nnet3/chain/nnet3_chain_lib.py')

def GetArgs():
    # we add compulsary arguments as named arguments for readability
    parser = argparse.ArgumentParser(description="Writes config files and variables "
                                                 "for TDNNs creation and training",
                                     epilog="See steps/nnet3/tdnn/train.sh for example.")

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

    parser.add_argument("--num-streams", type=int, default=1, dest="num_streams")
    parser.add_argument("--main-stream", type=int, default=0,
                        help = "feature stream for which xent cost is computed")

    # General neural network options
    parser.add_argument("--splice-indexes", type=str, required = True,
                        help="Splice indexes at each layer, e.g. '-3,-2,-1,0,1,2,3 -3,0,3 0' ")
    parser.add_argument("--final-layer-normalize-target", type=float,
                        help="RMS target for final layer (set to <1 if final layer learns too fast",
                        default=1.0)
    parser.add_argument("--relu-dim", type=int,
                        help="dimension of ReLU nonlinearities")
    parser.add_argument("--self-repair-scale-nonlinearity", type=float,
                        help="A non-zero value activates the self-repair mechanism in the non-linearities",
                        default=None)
    parser.add_argument("--use-presoftmax-prior-scale", type=str, action=nnet3_train_lib.StrToBoolAction,
                        help="if true, a presoftmax-prior-scale is added",
                        choices=['true', 'false'], default = True)
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

    if not args.relu_dim > 0:
        raise Exception("--relu-dim has to be positive")

    return args

def PrintConfig(file_name, config_lines):
    f = open(file_name, 'w')
    f.write("\n".join(config_lines['components'])+"\n")
    f.write("\n#Component nodes\n")
    f.write("\n".join(config_lines['component-nodes']))
    f.close()

def ParseSpliceString(splice_indexes):
    splice_array = []
    left_context = 0
    right_context = 0
    split1 = splice_indexes.split();  # we already checked the string is nonempty.
    if len(split1) < 1:
        raise Exception("invalid splice-indexes argument, too short: "
                 + splice_indexes)
    try:
        for string in split1:
            split2 = string.split(",")
            if len(split2) < 1:
                raise Exception("invalid splice-indexes argument, too-short element: "
                         + splice_indexes)
            int_list = []
            for int_str in split2:
                int_list.append(int(int_str))
            if not int_list == sorted(int_list):
                raise Exception("elements of splice-indexes must be sorted: "
                         + splice_indexes)
            left_context += -int_list[0]
            right_context += int_list[-1]
            splice_array.append(int_list)
    except ValueError as e:
        raise Exception("invalid splice-indexes argument " + splice_indexes + str(e))
    left_context = max(0, left_context)
    right_context = max(0, right_context)

    return {'left_context':left_context,
            'right_context':right_context,
            'splice_indexes':splice_array,
            'num_hidden_layers':len(splice_array)
            }


def SpliceInputs(inputs, splice_indexes):
    outputs = []
    for input in inputs:
        descriptor = input['descriptor']
        list = [('Offset({0}, {1})'.format(descriptor, n) if n != 0 else descriptor) for n in splice_indexes]
        outputs.append({'descriptor' : "Append({0})".format(" , ".join(list)),
                        'dimension'  : len(list) * input['dimension']})
    return outputs

# The function signature of MakeConfigs is changed frequently as it is intended for local use in this script.
def MakeConfigs(config_dir, num_streams, main_stream, splice_indexes_string,
                feat_dim, ivector_dim, num_targets, relu_dim,
                use_presoftmax_prior_scale,
                final_layer_normalize_target,
                self_repair_scale_nonlinearity):

    parsed_splice_output = ParseSpliceString(splice_indexes_string.strip())

    left_context = parsed_splice_output['left_context']
    right_context = parsed_splice_output['right_context']
    num_hidden_layers = parsed_splice_output['num_hidden_layers']
    splice_indexes = parsed_splice_output['splice_indexes']

    prior_scale_file = '{0}/presoftmax_prior_scale.vec'.format(config_dir)

    config_lines = {'components':[], 'component-nodes':[]}

    config_files={}
    lsr_node_files={}

    inputs = []
    if num_streams > 1:
        # the input has num_streams different feature streams
        # concatenated. The ivector is also assumed to be a concatenation
        # of num_streams different ivectors
        inputs = nodes.AddSeparatedInputLayers(config_lines, num_streams, feat_dim, splice_indexes[0], ivector_dim)
    else:
        inputs.append(nodes.AddInputLayer(config_lines, feat_dim, splice_indexes[0], ivector_dim))
        init_input = inputs[0]

    # Add the init config lines for estimating the preconditioning matrices
    # This preconditioning matrix will be shared across all the streams
    # Further it will be estimated using data sampled from all the streams
    # so it will be used from a separate directory
    init_config_lines = {'components':[], 'component-nodes':[]}
    init_config_lines['components'].insert(0, '# Config file for initializing neural network prior to')
    init_config_lines['components'].insert(0, '# preconditioning matrix computation')
    init_input = nodes.AddInputLayer(init_config_lines, feat_dim / num_streams, splice_indexes[0], ivector_dim / num_streams)
    nodes.AddOutputLayer(init_config_lines, init_input)
    config_files[config_dir + '/init.config'] = init_config_lines

    prev_layer_outputs = nodes.AddLdaLayer(config_lines, "L0",
                                          inputs, config_dir + '/lda.mat')

    left_context = 0
    right_context = 0
    # we moved the first splice layer to before the LDA..
    # so the input to the first affine layer is going to [0] index
    splice_indexes[0] = [0]

    lsr_output_nodes = []

    for i in range(0, num_hidden_layers):
        # make the intermediate config file for layerwise discriminative training
        # prepare the spliced input
        if not (len(splice_indexes[i]) == 1 and splice_indexes[i][0] == 0):
            prev_layer_outputs = SpliceInputs(prev_layer_outputs,
                                              splice_indexes[i])
        else:
            # this is a normal affine node
            pass

        prev_layer_outputs = nodes.AddAffRelNormLayer(config_lines, "Tdnn_{0}".format(i),
                                                     prev_layer_outputs, relu_dim,
                                                     self_repair_scale = self_repair_scale_nonlinearity,
                                                     norm_target_rms = 1.0 if i < num_hidden_layers -1 else final_layer_normalize_target)

        if len(prev_layer_outputs) > 1:
            #LSR can be computed
            lsr_output_nodes += nodes.AddLsrLayer(config_lines, "LSR_{0}".format(i),
                                                  prev_layer_outputs, main_stream,
                                                  '{0}/lsr_{1}'.format(config_dir,i))

        # a final layer is added after each new layer as we are generating
        # configs for layer-wise discriminative training
        nodes.AddFinalLayer(config_lines, prev_layer_outputs[main_stream], num_targets,
                           use_presoftmax_prior_scale = use_presoftmax_prior_scale,
                           prior_scale_file = prior_scale_file)

        config_files['{0}/layer{1}.config'.format(config_dir, i+1)] = config_lines
        lsr_node_files['{0}/pseudo_sup_nodes_{1}.txt'.format(config_dir, i+1)] = copy.deepcopy(lsr_output_nodes)
        config_lines = {'components':[], 'component-nodes':[]}

    lsr_node_files['{0}/pseudo_sup_nodes.txt'.format(config_dir, i+1)] = lsr_output_nodes

    left_context += int(parsed_splice_output['left_context'])
    right_context += int(parsed_splice_output['right_context'])

    # write the files used by other scripts like steps/nnet3/get_egs.sh
    f = open(config_dir + "/vars", "w")
    print('model_left_context=' + str(left_context), file=f)
    print('model_right_context=' + str(right_context), file=f)
    print('num_hidden_layers=' + str(num_hidden_layers), file=f)
    print('num_targets=' + str(num_targets), file=f)
    print('add_lda=true', file=f)
    print('include_log_softmax=true', file=f)
    print('objective_type=linear', file=f)
    f.close()

    # printing out the configs
    # init.config used to train lda-mllt train
    for key in config_files.keys():
        PrintConfig(key, config_files[key])

    # write down the names of the pseudo sup nodes
    for key in lsr_node_files.keys():
        f = open(key, "w")
        f.write(",".join(lsr_node_files[key]))
        f.close()

def Main():
    args = GetArgs()

    pprint.pprint(vars(args))
    MakeConfigs(config_dir = args.config_dir,
                num_streams = args.num_streams,
                main_stream = args.main_stream,
                splice_indexes_string = args.splice_indexes,
                feat_dim = args.feat_dim, ivector_dim = args.ivector_dim,
                num_targets = args.num_targets,
                relu_dim = args.relu_dim,
                use_presoftmax_prior_scale = args.use_presoftmax_prior_scale,
                final_layer_normalize_target = args.final_layer_normalize_target,
                self_repair_scale_nonlinearity = args.self_repair_scale_nonlinearity)

if __name__ == "__main__":
    Main()

