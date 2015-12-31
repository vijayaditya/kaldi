#!/usr/bin/env python

from __future__ import print_function
import os
import argparse
import sys
import warnings
import copy
from operator import itemgetter
import numpy as np
try:
    import scipy.signal as signal
    has_scipy_signal = True
except ImportError:
    has_scipy_signal = False

def WriteKaldiMatrix(matrix, matrix_file_name):
    assert(len(matrix.shape) == 2)
    # matrix is a numpy array
    matrix_file = open(matrix_file_name, "w")
    [rows, cols ] = matrix.shape
    matrix_file.write('[\n')
    for row in range(rows):
        matrix_file.write(' '.join( map(lambda x: '{0:f}'.format(x), matrix[row, : ])))
        if row == rows - 1:
            matrix_file.write("]")
        else:
            matrix_file.write('\n')
    matrix_file.close()
def GetSumDescriptor(inputs):
    sum_descriptors = inputs
    while len(sum_descriptors) != 1:
        cur_sum_descriptors = []
        pair = []
        while len(sum_descriptors) > 0:
            value = sum_descriptors.pop()
            if value.strip() != '':
                pair.append(value)
            if len(pair) == 2:
                cur_sum_descriptors.append("Sum({0}, {1})".format(pair[0], pair[1]))
                pair = []
        if pair:
            cur_sum_descriptors.append(pair[0])
        sum_descriptors = cur_sum_descriptors
    return sum_descriptors



# adds the input nodes and returns the descriptor
def AddInputLayer(config_lines, feat_dim, splice_indexes=[0], ivector_dim=0):
    components = config_lines['components']
    component_nodes = config_lines['component-nodes']
    output_dim = 0
    components.append('input-node name=input dim=' + str(feat_dim))
    list = [('Offset(input, {0})'.format(n) if n != 0 else 'input') for n in splice_indexes]
    output_dim += len(splice_indexes) * feat_dim
    if ivector_dim > 0:
        components.append('input-node name=ivector dim=' + str(ivector_dim))
        list.append('ReplaceIndex(ivector, t, 0)')
        output_dim += ivector_dim
    if len(list) > 1:
        splice_descriptor = "Append({0})".format(", ".join(list))
    else:
        splice_descriptor = list[0]
    print(splice_descriptor)
    return {'descriptor': splice_descriptor,
            'dimension': output_dim}

def AddNoOpLayer(config_lines, name, input):
    components = config_lines['components']
    component_nodes = config_lines['component-nodes']

    components.append('component name={0}_noop type=NoOpComponent dim={1}'.format(name, input['dimension']))
    component_nodes.append('component-node name={0}_noop component={0}_noop input={1}'.format(name, input['descriptor']))

    return {'descriptor':  '{0}_noop'.format(name),
            'dimension': input['dimension']}



def AddLdaLayer(config_lines, name, input, lda_file):
    components = config_lines['components']
    component_nodes = config_lines['component-nodes']

    components.append('component name={0}_lda type=FixedAffineComponent matrix={1}'.format(name, lda_file))
    component_nodes.append('component-node name={0}_lda component={0}_lda input={1}'.format(name, input['descriptor']))

    return {'descriptor':  '{0}_lda'.format(name),
            'dimension': input['dimension']}

def AddBlockAffineLayer(config_lines, name, input, output_dim, num_blocks):
    components = config_lines['components']
    component_nodes = config_lines['component-nodes']
    assert((input['dimension'] % num_blocks == 0) and
            (output_dim % num_blocks == 0))
    components.append('component name={0}_block_affine type=BlockAffineComponent input-dim={1} output-dim={2} num-blocks={3}'.format(name, input['dimension'], output_dim, num_blocks))
    component_nodes.append('component-node name={0}_block_affine component={0}_block_affine input={1}'.format(name, input['descriptor']))

    return {'descriptor' : '{0}_block_affine'.format(name),
                           'dimension' : output_dim}


def AddPermuteLayer(config_lines, name, input, column_map):
    components = config_lines['components']
    component_nodes = config_lines['component-nodes']
    permute_indexes = ",".join(map(lambda x: str(x), column_map))
    components.append('component name={0}_permute type=PermuteComponent column-map={1}'.format(name, permute_indexes))
    component_nodes.append('component-node name={0}_permute component={0}_permute input={1}'.format(name, input['descriptor']))

    return {'descriptor': '{0}_permute'.format(name),
            'dimension': input['dimension']}



def AddAffineLayer(config_lines, name, input, output_dim, ng_affine_options = ""):
    components = config_lines['components']
    component_nodes = config_lines['component-nodes']

    components.append("component name={0}_affine type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, input['dimension'], output_dim, ng_affine_options))
    component_nodes.append("component-node name={0}_affine component={0}_affine input={1}".format(name, input['descriptor']))

    return {'descriptor':  '{0}_affine'.format(name),
            'dimension': output_dim}

def AddAffRelNormLayer(config_lines, name, input, output_dim, ng_affine_options = " bias-stddev=0 ", norm_target_rms = 1.0):
    components = config_lines['components']
    component_nodes = config_lines['component-nodes']

    components.append("component name={0}_affine type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, input['dimension'], output_dim, ng_affine_options))
    components.append("component name={0}_relu type=RectifiedLinearComponent dim={1}".format(name, output_dim))
    components.append("component name={0}_renorm type=NormalizeComponent dim={1} target-rms={2}".format(name, output_dim, norm_target_rms))

    component_nodes.append("component-node name={0}_affine component={0}_affine input={1}".format(name, input['descriptor']))
    component_nodes.append("component-node name={0}_relu component={0}_relu input={0}_affine".format(name))
    component_nodes.append("component-node name={0}_renorm component={0}_renorm input={0}_relu".format(name))

    return {'descriptor':  '{0}_renorm'.format(name),
            'dimension': output_dim}


def AddConvolutionLayer(config_lines, name, input,
                       input_x_dim, input_y_dim, input_z_dim,
                       filt_x_dim, filt_y_dim,
                       filt_x_step, filt_y_step,
                       num_filters, input_vectorization,
                       param_stddev = None, bias_stddev = None,
                       filter_bias_file = None,
                       is_updatable = True):
    assert(input['dimension'] == input_x_dim * input_y_dim * input_z_dim)
    components = config_lines['components']
    component_nodes = config_lines['component-nodes']

    conv_init_string = "component name={0}_conv type=ConvolutionComponent input-x-dim={1} input-y-dim={2} input-z-dim={3} filt-x-dim={4} filt-y-dim={5} filt-x-step={6} filt-y-step={7} input-vectorization-order={8}".format(name, input_x_dim, input_y_dim, input_z_dim, filt_x_dim, filt_y_dim, filt_x_step, filt_y_step, input_vectorization)
    if filter_bias_file is not None:
        conv_init_string += " matrix={0}".format(filter_bias_file)
    if is_updatable:
        conv_init_string += " is-updatable=true"
    else:
        conv_init_string += " is-updatable=false"

    components.append(conv_init_string)
    component_nodes.append("component-node name={0}_conv_t component={0}_conv input={1}".format(name, input['descriptor']))

    num_x_steps = (1 + (input_x_dim - filt_x_dim) / filt_x_step)
    num_y_steps = (1 + (input_y_dim - filt_y_dim) / filt_y_step)
    output_dim = num_x_steps * num_y_steps * num_filters;
    return {'descriptor':  '{0}_conv_t'.format(name),
            'dimension': output_dim}



def AddSoftmaxLayer(config_lines, name, input):
    components = config_lines['components']
    component_nodes = config_lines['component-nodes']

    components.append("component name={0}_log_softmax type=LogSoftmaxComponent dim={1}".format(name, input['dimension']))
    component_nodes.append("component-node name={0}_log_softmax component={0}_log_softmax input={1}".format(name, input['descriptor']))

    return {'descriptor':  '{0}_log_softmax'.format(name),
            'dimension': input['dimension']}


def AddOutputLayer(config_lines, input, label_delay=None, suffix=None):
    components = config_lines['components']
    component_nodes = config_lines['component-nodes']
    name = 'output'
    if suffix is not None:
        name = '{0}-{1}'.format(name, suffix)

    if label_delay is None:
        component_nodes.append('output-node name={0} input={1}'.format(name, input['descriptor']))
    else:
        component_nodes.append('output-node name={0} input=Offset({1},{2})'.format(name, input['descriptor'], label_delay))

def AddFinalLayer(config_lines, input, output_dim,
        ng_affine_options = " param-stddev=0 bias-stddev=0 ",
        label_delay=None,
        use_presoftmax_prior_scale = False,
        prior_scale_file = None,
        include_log_softmax = True,
        name_affix = None):

    components = config_lines['components']
    component_nodes = config_lines['component-nodes']
    
    if name_affix is not None:
        final_node_prefix = 'Final-' + str(name_affix)
    else:
        final_node_prefix = 'Final'

    prev_layer_output = AddAffineLayer(config_lines,
            final_node_prefix , input, output_dim,
            ng_affine_options)
    if include_log_softmax:
        if use_presoftmax_prior_scale :
            components.append('component name={0}-fixed-scale type=FixedScaleComponent scales={1}'.format(final_node_prefix, prior_scale_file))
            component_nodes.append('component-node name=Final-fixed-scale component={0}-fixed-scale input={1}'.format(finale_node_prefix,
                prev_layer_output['descriptor']))
            prev_layer_output['descriptor'] = "{0}-fixed-scale".format(final_node_prefix)
        prev_layer_output = AddSoftmaxLayer(config_lines, final_node_prefix, prev_layer_output)
    # we use the same name_affix as a prefix in for affine/scale nodes but as a
    # suffix for output node
    AddOutputLayer(config_lines, prev_layer_output, label_delay, suffix = name_affix)

def AddLstmLayer(config_lines,
                 name, input, cell_dim,
                 recurrent_projection_dim = 0,
                 non_recurrent_projection_dim = 0,
                 clipping_threshold = 1.0,
                 norm_based_clipping = "false",
                 ng_per_element_scale_options = "",
                 ng_affine_options = "",
                 lstm_delay = -1):
    assert(recurrent_projection_dim >= 0 and non_recurrent_projection_dim >= 0)
    components = config_lines['components']
    component_nodes = config_lines['component-nodes']

    input_descriptor = input['descriptor']
    input_dim = input['dimension']
    name = name.strip()

    if (recurrent_projection_dim == 0):
        add_recurrent_projection = False
        recurrent_projection_dim = cell_dim
        recurrent_connection = "m_t"
    else:
        add_recurrent_projection = True
        recurrent_connection = "r_t"
    if (non_recurrent_projection_dim == 0):
        add_non_recurrent_projection = False
    else:
        add_non_recurrent_projection = True

    # Natural gradient per element scale parameters
    ng_per_element_scale_options += " param-mean=0.0 param-stddev=1.0 "
    # Parameter Definitions W*(* replaced by - to have valid names)
    components.append("# Input gate control : W_i* matrices")
    components.append("component name={0}_W_i-xr type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, input_dim + recurrent_projection_dim, cell_dim, ng_affine_options))
    components.append("# note : the cell outputs pass through a diagonal matrix")
    components.append("component name={0}_w_ic type=NaturalGradientPerElementScaleComponent  dim={1} {2}".format(name, cell_dim, ng_per_element_scale_options))

    components.append("# Forget gate control : W_f* matrices")
    components.append("component name={0}_W_f-xr type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, input_dim + recurrent_projection_dim, cell_dim, ng_affine_options))
    components.append("# note : the cell outputs pass through a diagonal matrix")
    components.append("component name={0}_w_fc type=NaturalGradientPerElementScaleComponent  dim={1} {2}".format(name, cell_dim, ng_per_element_scale_options))

    components.append("#  Output gate control : W_o* matrices")
    components.append("component name={0}_W_o-xr type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, input_dim + recurrent_projection_dim, cell_dim, ng_affine_options))
    components.append("# note : the cell outputs pass through a diagonal matrix")
    components.append("component name={0}_w_oc type=NaturalGradientPerElementScaleComponent  dim={1} {2}".format(name, cell_dim, ng_per_element_scale_options))

    components.append("# Cell input matrices : W_c* matrices")
    components.append("component name={0}_W_c-xr type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, input_dim + recurrent_projection_dim, cell_dim, ng_affine_options))


    components.append("# Defining the non-linearities")
    components.append("component name={0}_i type=SigmoidComponent dim={1}".format(name, cell_dim))
    components.append("component name={0}_f type=SigmoidComponent dim={1}".format(name, cell_dim))
    components.append("component name={0}_o type=SigmoidComponent dim={1}".format(name, cell_dim))
    components.append("component name={0}_g type=TanhComponent dim={1}".format(name, cell_dim))
    components.append("component name={0}_h type=TanhComponent dim={1}".format(name, cell_dim))

    components.append("# Defining the cell computations")
    components.append("component name={0}_c1 type=ElementwiseProductComponent input-dim={1} output-dim={2}".format(name, 2 * cell_dim, cell_dim))
    components.append("component name={0}_c2 type=ElementwiseProductComponent input-dim={1} output-dim={2}".format(name, 2 * cell_dim, cell_dim))
    components.append("component name={0}_m type=ElementwiseProductComponent input-dim={1} output-dim={2}".format(name, 2 * cell_dim, cell_dim))
    components.append("component name={0}_c type=ClipGradientComponent dim={1} clipping-threshold={2} norm-based-clipping={3} ".format(name, cell_dim, clipping_threshold, norm_based_clipping))

    # c1_t and c2_t defined below
    component_nodes.append("component-node name={0}_c_t component={0}_c input=Sum({0}_c1_t, {0}_c2_t)".format(name))
    c_tminus1_descriptor = "IfDefined(Offset({0}_c_t, {1}))".format(name, lstm_delay)

    component_nodes.append("# i_t")
    component_nodes.append("component-node name={0}_i1 component={0}_W_i-xr input=Append({1}, IfDefined(Offset({0}_{2}, {3})))".format(name, input_descriptor, recurrent_connection, lstm_delay))
    component_nodes.append("component-node name={0}_i2 component={0}_w_ic  input={1}".format(name, c_tminus1_descriptor))
    component_nodes.append("component-node name={0}_i_t component={0}_i input=Sum({0}_i1, {0}_i2)".format(name))

    component_nodes.append("# f_t")
    component_nodes.append("component-node name={0}_f1 component={0}_W_f-xr input=Append({1}, IfDefined(Offset({0}_{2}, {3})))".format(name, input_descriptor, recurrent_connection, lstm_delay))
    component_nodes.append("component-node name={0}_f2 component={0}_w_fc  input={1}".format(name, c_tminus1_descriptor))
    component_nodes.append("component-node name={0}_f_t component={0}_f input=Sum({0}_f1,{0}_f2)".format(name))

    component_nodes.append("# o_t")
    component_nodes.append("component-node name={0}_o1 component={0}_W_o-xr input=Append({1}, IfDefined(Offset({0}_{2}, {3})))".format(name, input_descriptor, recurrent_connection, lstm_delay))
    component_nodes.append("component-node name={0}_o2 component={0}_w_oc input={0}_c_t".format(name))
    component_nodes.append("component-node name={0}_o_t component={0}_o input=Sum({0}_o1, {0}_o2)".format(name))

    component_nodes.append("# h_t")
    component_nodes.append("component-node name={0}_h_t component={0}_h input={0}_c_t".format(name))

    component_nodes.append("# g_t")
    component_nodes.append("component-node name={0}_g1 component={0}_W_c-xr input=Append({1}, IfDefined(Offset({0}_{2}, {3})))".format(name, input_descriptor, recurrent_connection, lstm_delay))
    component_nodes.append("component-node name={0}_g_t component={0}_g input={0}_g1".format(name))

    component_nodes.append("# parts of c_t")
    component_nodes.append("component-node name={0}_c1_t component={0}_c1  input=Append({0}_f_t, {1})".format(name, c_tminus1_descriptor))
    component_nodes.append("component-node name={0}_c2_t component={0}_c2 input=Append({0}_i_t, {0}_g_t)".format(name))

    component_nodes.append("# m_t")
    component_nodes.append("component-node name={0}_m_t component={0}_m input=Append({0}_o_t, {0}_h_t)".format(name))

    # add the recurrent connections
    if (add_recurrent_projection and add_non_recurrent_projection):
        components.append("# projection matrices : Wrm and Wpm")
        components.append("component name={0}_W-m type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, cell_dim, recurrent_projection_dim + non_recurrent_projection_dim, ng_affine_options))
        components.append("component name={0}_r type=ClipGradientComponent dim={1} clipping-threshold={2} norm-based-clipping={3} ".format(name, recurrent_projection_dim, clipping_threshold, norm_based_clipping))
        component_nodes.append("# r_t and p_t")
        component_nodes.append("component-node name={0}_rp_t component={0}_W-m input={0}_m_t".format(name))
        component_nodes.append("dim-range-node name={0}_r_t_preclip input-node={0}_rp_t dim-offset=0 dim={1}".format(name, recurrent_projection_dim))
        component_nodes.append("component-node name={0}_r_t component={0}_r input={0}_r_t_preclip".format(name))
        output_descriptor = '{0}_rp_t'.format(name)
        output_dim = recurrent_projection_dim + non_recurrent_projection_dim

    elif add_recurrent_projection:
        components.append("# projection matrices : Wrm")
        components.append("component name={0}_Wrm type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, cell_dim, recurrent_projection_dim, ng_affine_options))
        components.append("component name={0}_r type=ClipGradientComponent dim={1} clipping-threshold={2} norm-based-clipping={3} ".format(name, recurrent_projection_dim, clipping_threshold, norm_based_clipping))
        component_nodes.append("# r_t")
        component_nodes.append("component-node name={0}_r_t_preclip component={0}_Wrm input={0}_m_t".format(name))
        component_nodes.append("component-node name={0}_r_t component={0}_r input={0}_r_t_preclip".format(name))
        output_descriptor = '{0}_r_t'.format(name)
        output_dim = recurrent_projection_dim

    else:
        components.append("component name={0}_r type=ClipGradientComponent dim={1} clipping-threshold={2} norm-based-clipping={3} ".format(name, cell_dim, clipping_threshold, norm_based_clipping))
        component_nodes.append("component-node name={0}_r_t component={0}_r input={0}_m_t".format(name))
        output_descriptor = '{0}_r_t'.format(name)
        output_dim = cell_dim

    return {
            'descriptor': output_descriptor,
            'dimension':output_dim
            }

# lowpass filters used to reduce the rate of the signal to specified rates
filter_cache = {}

# function assumes that most of the input/parameter checks have already been performed
# using a commong convolution component for a all lp filters followed by
# dim-range node is very costly !!
def AddCwrnnLayer(config_lines,
                 name, input,
                 lpfilt_filename,
                 num_lpfilter_taps = None,
                 input_vectorization = 'zyx',
                 clipping_threshold = 1.0,
                 norm_based_clipping = "false",
                 ng_per_element_scale_options = "",
                 ratewise_params = {'T1': {'rate':1, 'dim':128},
                                    'T2': {'rate':1.0/2, 'dim':128},
                                    'T3': {'rate':1.0/4, 'dim':128},
                                    'T4': {'rate':1.0/8, 'dim':128}
                                    },
                 nonlinearity = "SigmoidComponent",
                 subsample = True,
                 filter_input_step = 1,
                 diag_init_scaling_factor = 0,
                 input_type = "smooth",
                 use_lstm = False,
                 projection_dim = 0# not used anymore
                 ):
    assert(filter_input_step > 0)
    components = config_lines['components']
    component_nodes = config_lines['component-nodes']

    key_rate_pairs = map(lambda key: (key, ratewise_params[key]['rate']), ratewise_params)
    key_rate_pairs.sort(key=itemgetter(1))
    keys = map(lambda x: x[0], key_rate_pairs)
    largest_time_period = int(1.0/ratewise_params[keys[0]]['rate'])

    max_num_lpfilter_taps = 0 # this will used by calling function to determine
    # input context of the CWRNN node
    # since we want to subsample the input feature representation
    # we will pass it through a low pass filters one for each rate
    slow_to_fast_descriptors = {}
    for key in keys:
        slow_to_fast_descriptors[key] = {}

    for key_index in xrange(len(keys)):
        key = keys[key_index]
        params = ratewise_params[key]
        rate = params['rate']
        time_period = int(round(1.0/params['rate']))
        cw_unit_input_descriptor = None
        if rate != 1 and input_type == "smooth":
            # we will downsample the input to this rate unit
            # so we will low pass filter the input to avoid aliasing
            num_lpfilter_taps = 2 * time_period + 1
            max_num_lpfilter_taps = max(num_lpfilter_taps, max_num_lpfilter_taps)
            if has_scipy_signal:
                # python implementation says nyq is nyquist frequency, but I think
                # they are using it as half the sampling rate
                # [partially confirmed based on function output comparison with matlab]
                lp_filter = signal.firwin(num_lpfilter_taps, rate, width=None, window='hamming', pass_zero=True, scale=True, nyq=1.0)
                # add a zero for the bias element expected by the
                # Convolution1dComponent
            elif filter_cache.has_key(rate):
                # as all users might not have scipy installed we are providing a
                # set of filters at the specified rates and tap lengths
                raise NotImplementedError("Low priority as this code block is just for compatibility with python installations without scipy")
            cur_lpfilt_filename = '{0}_{1}'.format(lpfilt_filename, key)
            filter_context = int((num_lpfilter_taps - 1) / 2)
            filter_input_splice_indexes = range(-1 * filter_context, filter_context + 1, filter_input_step)
            filter_vector_indexes = map(lambda x: x - min(filter_input_splice_indexes), filter_input_splice_indexes)
            lp_filter = lp_filter[filter_vector_indexes]
            lp_filter = np.append(lp_filter, 0)
            WriteKaldiMatrix(np.array([lp_filter]), cur_lpfilt_filename)
            input_dim = input['dimension']
            #TODO : perform a check to see if the input descriptors allow use of offset and append
            list = [('Offset({0}, {1})'.format(input['descriptor'], n) if n != 0 else input['descriptor']) for n in filter_input_splice_indexes]
            filter_input_descriptor = 'Append({0})'.format(' , '.join(list))
            filter_input_descriptor = {'descriptor':filter_input_descriptor,
                                    'dimension':len(filter_input_splice_indexes) * input_dim}

            input_x_dim = len(filter_input_splice_indexes)
            input_y_dim = input['dimension']
            input_z_dim = 1
            filt_x_dim = len(filter_input_splice_indexes)
            filt_y_dim = 1
            filt_x_step = 1
            filt_y_step = 1

            cw_unit_input_descriptor = AddConvolutionLayer(config_lines, '{0}_{1}'.format(name, key), filter_input_descriptor,
                                                          input_x_dim, input_y_dim, input_z_dim,
                                                          filt_x_dim, filt_y_dim,
                                                          filt_x_step, filt_y_step,
                                                          1, input_vectorization,
                                                          filter_bias_file = cur_lpfilt_filename,
                                                          is_updatable = False)

        else:
            cw_unit_input_descriptor = input

        fastrate_params = {}
        for fast_key_index in range(key_index+1, len(keys)):
            fast_key = keys[fast_key_index]
            fastrate_params[fast_key] = ratewise_params[fast_key]
        if use_lstm:
            [output_descriptor, fastrate_output_descriptors] = AddCwlstmRateUnit(config_lines, '{0}_{1}'.format(name, key),
                                                                                 cw_unit_input_descriptor, params['cell-dim'],
                                                                                 params['rate'], params['recurrent-projection'],
                                                                                 params['non-recurrent-projection'],
                                                                                 clipping_threshold = clipping_threshold,
                                                                                 norm_based_clipping = norm_based_clipping,
                                                                                 ng_per_element_scale_options = ng_per_element_scale_options,
                                                                                 lstm_delay = -1 * time_period,
                                                                                 slowrate_descriptors = slow_to_fast_descriptors[key],
                                                                                 fastrate_params = fastrate_params,
                                                                                 subsample = subsample,
                                                                                 input_type = input_type)
            for fastrate in fastrate_output_descriptors.keys():
                for dest_name in fastrate_output_descriptors[fastrate].keys():
                    try:
                        slow_to_fast_descriptors[fastrate][dest_name][key] = fastrate_output_descriptors[fastrate][dest_name]
                    except KeyError:
                        slow_to_fast_descriptors[fastrate][dest_name] = {}
                        slow_to_fast_descriptors[fastrate][dest_name][key] = fastrate_output_descriptors[fastrate][dest_name]

        else:
            [output_descriptor, fastrate_output_descriptors] = AddCwrnnRateUnit(config_lines, '{0}_{1}'.format(name, key),
                                                                                cw_unit_input_descriptor, params['dim'],
                                                                                params['rate'],
                                                                                clipping_threshold = clipping_threshold,
                                                                                norm_based_clipping = norm_based_clipping,
                                                                                delay = -1 * time_period,
                                                                                slowrate_descriptors = slow_to_fast_descriptors[key],
                                                                                fastrate_params = fastrate_params,
                                                                                nonlinearity = nonlinearity,
                                                                                subsample = subsample,
                                                                                diag_init_scaling_factor = diag_init_scaling_factor,
                                                                                input_type = input_type)
            for fastrate in fastrate_output_descriptors.keys():
                slow_to_fast_descriptors[fastrate][key] = fastrate_output_descriptors[fastrate]
    return [output_descriptor, max_num_lpfilter_taps, largest_time_period]



def AddCwrnnRateUnit(config_lines,
                    name, input, output_dim,
                    unit_rate,
                    clipping_threshold = 1.0,
                    norm_based_clipping = "false",
                    ng_affine_options = " bias-stddev=0 ",
                    delay = -1,
                    slowrate_descriptors = {},
                    fastrate_params = {},
                    nonlinearity = "SigmoidComponent",
                    subsample = True,
                    diag_init_scaling_factor = 0,
                    input_type = "smooth"):

    # if true, use subsampling for the slower rates
    # else always use delay of -1.
    if not subsample:
        delay = -1

    if nonlinearity == "RectifiedLinearComponent+NormalizeComponent":
        raise Exception("{0} is not yet supported".format(nonlinearity))
    if diag_init_scaling_factor != 0:
        recurrent_ng_affine_options = " diag-init-scaling-factor={0}".format(diag_init_scaling_factor)
        nonrecurrent_ng_affine_options = "{0} param-stddev={1}".format(ng_affine_options, diag_init_scaling_factor)
    else:
        recurrent_ng_affine_options = ng_affine_options
        nonrecurrent_ng_affine_options = ng_affine_options

    components = config_lines['components']
    component_nodes = config_lines['component-nodes']
    print(input)
    input_descriptor = input['descriptor']
    input_dim = input['dimension']
    name = name.strip()

    recurrent_connection = "r_t"
    if len(slowrate_descriptors.keys()) > 0:
        slowrate_sum_descriptor = GetSumDescriptor(map(lambda x: x['descriptor'], slowrate_descriptors.values()))[0]
    else:
        slowrate_sum_descriptor = ''

    unit_time_period = int(1.0 / unit_rate)

    if input_type in set(["stack", "sum"]):
        splice_indexes = range(-1*(unit_time_period-1), 1)

        list = [('Offset({0}, {1})'.format(input['descriptor'], n) if n != 0 else input['descriptor']) for n in splice_indexes]
        if input_type == "stack":
            input_descriptor = 'Append({0})'.format(' , '.join(list))
            input_dim = len(splice_indexes) * input_dim
        elif input_type == "sum":
            input_descriptor = GetSumDescriptor(list)[0]
    else:
        input_descriptor = input['descriptor']

    # Parameter Definitions W*(* replaced by - to have valid names)
    components.append("# Current rate components : Wxr")
    total_fastrate_unit_dim = sum(map(lambda x: x['dim'], fastrate_params.values()))
    components.append("component name={0}_W_r type=NaturalGradientAffineComponent input-dim={1} output-dim={2} rank-in={3} rank-out={4} {5}".format(name, output_dim, output_dim + total_fastrate_unit_dim, int(output_dim * 0.1), int((output_dim + total_fastrate_unit_dim) * 0.1), recurrent_ng_affine_options))
    components.append("component name={0}_W_x type=NaturalGradientAffineComponent input-dim={1} output-dim={2} rank-in={3} rank-out={4} {5}".format(name, input_dim, output_dim, int(input_dim * 0.1), int(output_dim * 0.1), nonrecurrent_ng_affine_options))
    component_nodes.append("component-node name={0}_W_x_t component={0}_W_x input={1}".format(name, input_descriptor))

    components.append("component name={0}_nonlin type={1} dim={2}".format(name, nonlinearity, output_dim))
    components.append("component name={0}_clip type=ClipGradientComponent dim={1} clipping-threshold={2} norm-based-clipping={3} ".format(name, output_dim, clipping_threshold, norm_based_clipping))
    if slowrate_sum_descriptor == '':
        component_nodes.append("component-node name={0}_{1}_preclip component={0}_nonlin input=Sum({0}_W_r_t, {0}_W_x_t)".format(name, recurrent_connection))
    else:
        component_nodes.append("component-node name={0}_{1}_preclip component={0}_nonlin input=Sum(Sum({0}_W_r_t, {0}_W_x_t), {2})".format(name, recurrent_connection, slowrate_sum_descriptor))
    component_nodes.append("component-node name={0}_{1} component={0}_clip input={0}_{1}_preclip".format(name, recurrent_connection))


    fastrate_output_descriptors = {}
    keys = fastrate_params.keys()
    if total_fastrate_unit_dim == 0:
        component_nodes.append("component-node name={0}_W_r_t component={0}_W_r input=IfDefined(Offset({0}_{1}, {2}))".format(name, recurrent_connection, delay))
    else:
        component_nodes.append("component-node name={0}_W_r_all_t component={0}_W_r input=IfDefined(Offset({0}_{1}, {2}))".format(name, recurrent_connection, delay))
        component_nodes.append("dim-range-node name={0}_W_r_t input-node={0}_W_r_all_t dim-offset=0 dim={1}".format(name, output_dim))
        offset = output_dim
        for key in fastrate_params.keys():
            params = fastrate_params[key]
            fastrate_time_period = int(1.0/params['rate'])
            output_name = '{0}_W_rfast-tmod{1}_t'.format(name, fastrate_time_period)
            component_nodes.append("dim-range-node name={1} input-node={0}_W_r_all_t dim-offset={2} dim={3}".format(name, output_name, offset, params['dim']))
            offset += params['dim']
            if subsample:
                fastrate_descriptor = 'Round({0}, {1})'.format(output_name, unit_time_period)
            else:
                fastrate_descriptor = output_name
            fastrate_output_descriptors[key] = {'descriptor': fastrate_descriptor,
                                                'dimension' : params['dim']}

    output_descriptor = "{0}_{1}".format(name, recurrent_connection)
    return [{'descriptor': output_descriptor, 'dimension':output_dim}, fastrate_output_descriptors]




def AddClstmLayer(config_lines,
                 name, input, cell_dim,
                 recurrent_projection_dim = 0,
                 non_recurrent_projection_dim = 0,
                 clipping_threshold = 1.0,
                 norm_based_clipping = "false",
                 ng_per_element_scale_options = "",
                 ng_affine_options = "",
                 lstm_delay = -1,
                 rates = [1]):
    assert(recurrent_projection_dim >= 0 and non_recurrent_projection_dim >= 0)
    components = config_lines['components']
    component_nodes = config_lines['component-nodes']

    input_descriptor = input['descriptor']
    input_dim = input['dimension']
    name = name.strip()

    if (recurrent_projection_dim == 0):
        add_recurrent_projection = False
        recurrent_projection_dim = cell_dim
        recurrent_connection = "m_t"
    else:
        add_recurrent_projection = True
        recurrent_connection = "r_t"
    if (non_recurrent_projection_dim == 0):
        add_non_recurrent_projection = False
    else:
        add_non_recurrent_projection = True

    # Natural gradient per element scale parameters
    ng_per_element_scale_options += " param-mean=0.0 param-stddev=1.0 "
    # Parameter Definitions W*(* replaced by - to have valid names)
    components.append("# Input gate control : W_i* matrices")
    components.append("component name={0}_W_i-xr type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, input_dim + recurrent_projection_dim, cell_dim, ng_affine_options))
    components.append("# note : the cell outputs pass through a diagonal matrix")
    components.append("component name={0}_w_ic type=NaturalGradientPerElementScaleComponent  dim={1} {2}".format(name, cell_dim, ng_per_element_scale_options))

    components.append("# Forget gate control : W_f* matrices")
    components.append("component name={0}_W_f-xr type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, input_dim + recurrent_projection_dim, cell_dim, ng_affine_options))
    components.append("# note : the cell outputs pass through a diagonal matrix")
    components.append("component name={0}_w_fc type=NaturalGradientPerElementScaleComponent  dim={1} {2}".format(name, cell_dim, ng_per_element_scale_options))

    components.append("#  Output gate control : W_o* matrices")
    components.append("component name={0}_W_o-xr type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, input_dim + recurrent_projection_dim, cell_dim, ng_affine_options))
    components.append("# note : the cell outputs pass through a diagonal matrix")
    components.append("component name={0}_w_oc type=NaturalGradientPerElementScaleComponent  dim={1} {2}".format(name, cell_dim, ng_per_element_scale_options))

    components.append("# Cell input matrices : W_c* matrices")
    components.append("component name={0}_W_c-xr type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, input_dim + recurrent_projection_dim, cell_dim, ng_affine_options))

    components.append("# Defining the non-linearities")
    components.append("component name={0}_i type=SigmoidComponent dim={1}".format(name, cell_dim))
    components.append("component name={0}_f type=SigmoidComponent dim={1}".format(name, cell_dim))
    components.append("component name={0}_o type=SigmoidComponent dim={1}".format(name, cell_dim))
    components.append("component name={0}_g type=TanhComponent dim={1}".format(name, cell_dim))
    components.append("component name={0}_h type=TanhComponent dim={1}".format(name, cell_dim))

    components.append("# Defining the cell computations")
    components.append("component name={0}_c1 type=ElementwiseProductComponent input-dim={1} output-dim={2}".format(name, 2 * cell_dim, cell_dim))
    components.append("component name={0}_c2 type=ElementwiseProductComponent input-dim={1} output-dim={2}".format(name, 2 * cell_dim, cell_dim))
    components.append("component name={0}_m type=ElementwiseProductComponent input-dim={1} output-dim={2}".format(name, 2 * cell_dim, cell_dim))
    components.append("component name={0}_c type=ClipGradientComponent dim={1} clipping-threshold={2} norm-based-clipping={3} ".format(name, cell_dim, clipping_threshold, norm_based_clipping))

    # c1_t and c2_t defined below
    component_nodes.append("component-node name={0}_c_t component={0}_c input=Sum({0}_c1_t, {0}_c2_t)".format(name))
    c_tminus1_descriptor = "IfDefined(Offset({0}_c_t, {1}))".format(name, lstm_delay)

    component_nodes.append("# i_t")
    component_nodes.append("component-node name={0}_i1 component={0}_W_i-xr input=Append({1}, IfDefined(Offset({0}_{2}, {3})))".format(name, input_descriptor, recurrent_connection, lstm_delay))
    component_nodes.append("component-node name={0}_i2 component={0}_w_ic  input={1}".format(name, c_tminus1_descriptor))
    component_nodes.append("component-node name={0}_i_t component={0}_i input=Sum({0}_i1, {0}_i2)".format(name))

    component_nodes.append("# f_t")
    component_nodes.append("component-node name={0}_f1 component={0}_W_f-xr input=Append({1}, IfDefined(Offset({0}_{2}, {3})))".format(name, input_descriptor, recurrent_connection, lstm_delay))
    component_nodes.append("component-node name={0}_f2 component={0}_w_fc  input={1}".format(name, c_tminus1_descriptor))
    component_nodes.append("component-node name={0}_f_t component={0}_f input=Sum({0}_f1,{0}_f2)".format(name))

    component_nodes.append("# o_t")
    component_nodes.append("component-node name={0}_o1 component={0}_W_o-xr input=Append({1}, IfDefined(Offset({0}_{2}, {3})))".format(name, input_descriptor, recurrent_connection, lstm_delay))
    component_nodes.append("component-node name={0}_o2 component={0}_w_oc input={0}_c_t".format(name))
    component_nodes.append("component-node name={0}_o_t component={0}_o input=Sum({0}_o1, {0}_o2)".format(name))

    component_nodes.append("# h_t")
    component_nodes.append("component-node name={0}_h_t component={0}_h input={0}_c_t".format(name))

    component_nodes.append("# g_t")
    component_nodes.append("component-node name={0}_g1 component={0}_W_c-xr input=Append({1}, IfDefined(Offset({0}_{2}, {3})))".format(name, input_descriptor, recurrent_connection, lstm_delay))
    component_nodes.append("component-node name={0}_g_t component={0}_g input={0}_g1".format(name))

    component_nodes.append("# parts of c_t")
    component_nodes.append("component-node name={0}_c1_t component={0}_c1  input=Append({0}_f_t, {1})".format(name, c_tminus1_descriptor))
    component_nodes.append("component-node name={0}_c2_t component={0}_c2 input=Append({0}_i_t, {0}_g_t)".format(name))

    component_nodes.append("# m_t")
    component_nodes.append("component-node name={0}_m_t component={0}_m input=Append({0}_o_t, {0}_h_t)".format(name))

    # add the recurrent connections
    if (add_recurrent_projection and add_non_recurrent_projection):
        components.append("# projection matrices : Wrm and Wpm")
        components.append("component name={0}_W-m type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, cell_dim, recurrent_projection_dim + non_recurrent_projection_dim, ng_affine_options))
        components.append("component name={0}_r type=ClipGradientComponent dim={1} clipping-threshold={2} norm-based-clipping={3} ".format(name, recurrent_projection_dim, clipping_threshold, norm_based_clipping))
        component_nodes.append("# r_t and p_t")
        component_nodes.append("component-node name={0}_rp_t component={0}_W-m input={0}_m_t".format(name))
        component_nodes.append("dim-range-node name={0}_r_t_preclip input-node={0}_rp_t dim-offset=0 dim={1}".format(name, recurrent_projection_dim))
        component_nodes.append("component-node name={0}_r_t component={0}_r input={0}_r_t_preclip".format(name))
        output_descriptor = '{0}_rp_t'.format(name)
        output_dim = recurrent_projection_dim + non_recurrent_projection_dim

    elif add_recurrent_projection:
        components.append("# projection matrices : Wrm")
        components.append("component name={0}_Wrm type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, cell_dim, recurrent_projection_dim, ng_affine_options))
        components.append("component name={0}_r type=ClipGradientComponent dim={1} clipping-threshold={2} norm-based-clipping={3} ".format(name, recurrent_projection_dim, clipping_threshold, norm_based_clipping))
        component_nodes.append("# r_t")
        component_nodes.append("component-node name={0}_r_t_preclip component={0}_Wrm input={0}_m_t".format(name))
        component_nodes.append("component-node name={0}_r_t component={0}_r input={0}_r_t_preclip".format(name))
        output_descriptor = '{0}_r_t'.format(name)
        output_dim = recurrent_projection_dim

    else:
        components.append("component name={0}_r type=ClipGradientComponent dim={1} clipping-threshold={2} norm-based-clipping={3} ".format(name, cell_dim, clipping_threshold, norm_based_clipping))
        component_nodes.append("component-node name={0}_r_t component={0}_r input={0}_m_t".format(name))
        output_descriptor = '{0}_r_t'.format(name)
        output_dim = cell_dim

    return {
            'descriptor': output_descriptor,
            'dimension':output_dim
            }



