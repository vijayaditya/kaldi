#!/usr/bin/env python

from __future__ import print_function
import os
import argparse
import sys
import warnings
import copy
from operator import itemgetter

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
    return {'output': {'descriptor': splice_descriptor,
                       'dimension': output_dim},
            'num_learnable_params' : 0}

def AddNoOpLayer(config_lines, name, input):
    components = config_lines['components']
    component_nodes = config_lines['component-nodes']

    components.append('component name={0}_noop type=NoOpComponent dim={1}'.format(name, input['dimension']))
    component_nodes.append('component-node name={0}_noop component={0}_noop input={1}'.format(name, input['descriptor']))

    return {'output' : {'descriptor':  '{0}_noop'.format(name),
                        'dimension': input['dimension']},
            'num_learnable_params' : 0}

def AddLdaLayer(config_lines, name, input, lda_file):
    return AddFixedAffineLayer(config_lines, name, input, lda_file)

def AddFixedAffineLayer(config_lines, name, input, matrix_file):
    components = config_lines['components']
    component_nodes = config_lines['component-nodes']

    components.append('component name={0}_fixaffine type=FixedAffineComponent matrix={1}'.format(name, matrix_file))
    component_nodes.append('component-node name={0}_fixaffine component={0}_fixaffine input={1}'.format(name, input['descriptor']))

    return {'output' : {'descriptor':  '{0}_fixaffine'.format(name),
                        'dimension': input['dimension']},
            'num_learnable_params' : 0}

def AddPerDimAffineLayerMulti(config_lines, name, input, num_block_affine, input_window, time_period = 1, return_if_single=True, inputs_optional = False):
    components = config_lines['components']
    component_nodes = config_lines['component-nodes']

    filter_context = int((input_window - 1) / 2)
    filter_input_splice_indexes = range(-1 * filter_context, filter_context + 1, time_period)
    if len(filter_input_splice_indexes) == 1 and filter_input_splice_indexes[0] == 0:
        return {'output':input,
                'filter_left_context' : 0,
                'filter_right_context' : 0,
                'num_learnable_params' : 0}
    if inputs_optional:
        list = [('IfDefined(Offset({0}, {1}))'.format(input['descriptor'], n) if n != 0 else 'IfDefined({0})'.format(input['descriptor'])) for n in filter_input_splice_indexes]
    else:
        list = [('Offset({0}, {1})'.format(input['descriptor'], n) if n != 0 else input['descriptor']) for n in filter_input_splice_indexes]
    filter_input_descriptor = 'Append({0})'.format(' , '.join(list))
    filter_input_descriptor = {'descriptor':filter_input_descriptor,
                               'dimension':len(filter_input_splice_indexes) * input['dimension']}
    num_learnable_params = 0
    # add permute component to shuffle the feature columns of the Append
    # descriptor output so that columns corresponding to the same feature index
    # are contiguous add a block-affine component to collapse all the feature
    # indexes across time steps into a single value
    num_feats = input['dimension']
    num_times = len(filter_input_splice_indexes)
    column_map = []
    for i in range(num_feats):
        for j in range(num_times):
            column_map.append(j * num_feats + i)

    permute_layer = AddPermuteLayer(config_lines,
            name, filter_input_descriptor, column_map)
    permuted_output_descriptor = permute_layer['output']
    num_learnable_params = permute_layer['num_learnable_params']

    outputs = []
    for i in range(1, num_block_affine + 1):
        # add a block-affine component
        prev_layer = AddBlockAffineLayer(config_lines, '{0}_PDA{1}'.format(name, i),
                                         permuted_output_descriptor,
                                         num_feats, num_feats)
        output_descriptor = prev_layer['output']
        num_learnable_params += prev_layer['num_learnable_params']
        outputs.append(output_descriptor)

    return {'outputs' : outputs,
            'num_learnable_params' : num_learnable_params,
            'filter_left_context' : filter_context,
            'filter_right_context' : filter_context }


def AddPerDimAffineLayer(config_lines, name, input, input_window, time_period = 1, return_if_single=True):
    components = config_lines['components']
    component_nodes = config_lines['component-nodes']

    filter_context = int((input_window - 1) / 2)
    filter_input_splice_indexes = range(-1 * filter_context, filter_context + 1, time_period)
    if len(filter_input_splice_indexes) == 1 and filter_input_splice_indexes[0] == 0:
        return {'output':input,
                'filter_left_context' : 0,
                'filter_right_context' : 0,
                'num_learnable_params' : 0}
    list = [('Offset({0}, {1})'.format(input['descriptor'], n) if n != 0 else input['descriptor']) for n in filter_input_splice_indexes]
    filter_input_descriptor = 'Append({0})'.format(' , '.join(list))
    filter_input_descriptor = {'descriptor':filter_input_descriptor,
                               'dimension':len(filter_input_splice_indexes) * input['dimension']}

    num_learnable_params = 0
    # add permute component to shuffle the feature columns of the Append
    # descriptor output so that columns corresponding to the same feature index
    # are contiguous add a block-affine component to collapse all the feature
    # indexes across time steps into a single value
    num_feats = input['dimension']
    num_times = len(filter_input_splice_indexes)
    column_map = []
    for i in range(num_feats):
        for j in range(num_times):
            column_map.append(j * num_feats + i)

    composite_config_lines = {'components':[], 'component-nodes':[]}

    permute_layer = AddPermuteLayer(composite_config_lines,
            name, filter_input_descriptor, column_map)
    permuted_output_descriptor = permute_layer['output']
    num_learnable_params = permute_layer['num_learnable_params']

    # add a block-affine component
    prev_layer = AddBlockAffineLayer(composite_config_lines, name,
                                     permuted_output_descriptor,
                                     num_feats, num_feats)
    output_descriptor = prev_layer['output']
    num_learnable_params += prev_layer['num_learnable_params']
    # strip names
    ccl = composite_config_lines['components']
    composite_config_line = ''
    for index in range(len(ccl)):
        parts = ccl[index].split()
        assert(parts[0] == "component" and parts[1].split('=')[0] == "name")
        composite_config_line += " component{0}='{1}'".format(index+1, " ".join(parts[2:]))

    components.append("component name={name} type=CompositeComponent num-components={nc} {rest}".format(name = '{0}_PDA'.format(name),
                   nc = len(ccl),
                   rest = composite_config_line))
    component_nodes.append("component-node name={0}_PDA component={0}_PDA input={1}".format(name, filter_input_descriptor['descriptor']))
    return {'output':{'descriptor': '{0}_PDA'.format(name),
                       'dimension': output_descriptor['dimension']},
            'num_learnable_params' : 0,
            'filter_left_context' : filter_context,
            'filter_right_context' : filter_context }

def AddBlockAffineLayer(config_lines, name, input, output_dim, num_blocks):
    components = config_lines['components']
    component_nodes = config_lines['component-nodes']
    assert((input['dimension'] % num_blocks == 0) and
            (output_dim % num_blocks == 0))
    components.append('component name={0}_block_affine type=BlockAffineComponent input-dim={1} output-dim={2} num-blocks={3}'.format(name, input['dimension'], output_dim, num_blocks))
    component_nodes.append('component-node name={0}_block_affine component={0}_block_affine input={1}'.format(name, input['descriptor']))

    return {'output' : {'descriptor' : '{0}_block_affine'.format(name),
                        'dimension' : output_dim},
            'num_learnable_params' : input['dimension'] * output_dim}


def AddPermuteLayer(config_lines, name, input, column_map):
    components = config_lines['components']
    component_nodes = config_lines['component-nodes']
    permute_indexes = ",".join(map(lambda x: str(x), column_map))
    components.append('component name={0}_permute type=PermuteComponent column-map={1}'.format(name, permute_indexes))
    component_nodes.append('component-node name={0}_permute component={0}_permute input={1}'.format(name, input['descriptor']))

    return {'output' : {'descriptor': '{0}_permute'.format(name),
                        'dimension': input['dimension']},
            'num_learnable_params' : 0 }

def AddAffineLayer(config_lines, name, input, output_dim, ng_affine_options = ""):
    components = config_lines['components']
    component_nodes = config_lines['component-nodes']

    components.append("component name={0}_affine type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, input['dimension'], output_dim, ng_affine_options))
    component_nodes.append("component-node name={0}_affine component={0}_affine input={1}".format(name, input['descriptor']))

    return {'output' : {'descriptor':  '{0}_affine'.format(name),
                        'dimension': output_dim},
            'num_learnable_params' : input['dimension'] * output_dim }

def AddAffRelNormLayer(config_lines, name, input, output_dim, ng_affine_options = " bias-stddev=0 ", norm_target_rms = 1.0, self_repair_scale = None):
    components = config_lines['components']
    component_nodes = config_lines['component-nodes']

    # self_repair_scale is a constant scaling the self-repair vector computed in RectifiedLinearComponent
    self_repair_string = "self-repair-scale={0:.10f}".format(self_repair_scale) if self_repair_scale is not None else ''
    components.append("component name={0}_affine type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, input['dimension'], output_dim, ng_affine_options))
    components.append("component name={0}_relu type=RectifiedLinearComponent dim={1} {2}".format(name, output_dim, self_repair_string))
    components.append("component name={0}_renorm type=NormalizeComponent dim={1} target-rms={2}".format(name, output_dim, norm_target_rms))

    component_nodes.append("component-node name={0}_affine component={0}_affine input={1}".format(name, input['descriptor']))
    component_nodes.append("component-node name={0}_relu component={0}_relu input={0}_affine".format(name))
    component_nodes.append("component-node name={0}_renorm component={0}_renorm input={0}_relu".format(name))
    return {'output' : {'descriptor':  '{0}_renorm'.format(name),
                        'dimension': output_dim},
            'num_learnable_params' : input['dimension'] * output_dim }

def AddAffPnormLayer(config_lines, name, input, pnorm_input_dim, pnorm_output_dim, ng_affine_options = " bias-stddev=0 ", norm_target_rms = 1.0):
    components = config_lines['components']
    component_nodes = config_lines['component-nodes']

    components.append("component name={0}_affine type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, input['dimension'], pnorm_input_dim, ng_affine_options))
    components.append("component name={0}_pnorm type=PnormComponent input-dim={1} output-dim={2}".format(name, pnorm_input_dim, pnorm_output_dim))
    components.append("component name={0}_renorm type=NormalizeComponent dim={1} target-rms={2}".format(name, pnorm_output_dim, norm_target_rms))

    component_nodes.append("component-node name={0}_affine component={0}_affine input={1}".format(name, input['descriptor']))
    component_nodes.append("component-node name={0}_pnorm component={0}_pnorm input={0}_affine".format(name))
    component_nodes.append("component-node name={0}_renorm component={0}_renorm input={0}_pnorm".format(name))

    return {'output' : {'descriptor':  '{0}_renorm'.format(name),
                        'dimension': pnorm_output_dim},
            'num_learnable_params' : input['dimension'] * pnorm_input_dim }

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

    conv_init_string = ("component name={name}_conv type=ConvolutionComponent "
                       "input-x-dim={input_x_dim} input-y-dim={input_y_dim} input-z-dim={input_z_dim} "
                       "filt-x-dim={filt_x_dim} filt-y-dim={filt_y_dim} "
                       "filt-x-step={filt_x_step} filt-y-step={filt_y_step} "
                       "input-vectorization-order={vector_order}".format(name = name,
                       input_x_dim = input_x_dim, input_y_dim = input_y_dim, input_z_dim = input_z_dim,
                       filt_x_dim = filt_x_dim, filt_y_dim = filt_y_dim,
                       filt_x_step = filt_x_step, filt_y_step = filt_y_step,
                       vector_order = input_vectorization))
    if filter_bias_file is not None:
        conv_init_string += " matrix={0}".format(filter_bias_file)
    else:
        conv_init_string += " num-filters={0}".format(num_filters)

    components.append(conv_init_string)
    component_nodes.append("component-node name={0}_conv_t component={0}_conv input={1}".format(name, input['descriptor']))

    num_x_steps = (1 + (input_x_dim - filt_x_dim) / filt_x_step)
    num_y_steps = (1 + (input_y_dim - filt_y_dim) / filt_y_step)
    output_dim = num_x_steps * num_y_steps * num_filters;
    return {'output' : {'descriptor':  '{0}_conv_t'.format(name),
                        'dimension': output_dim,
                        '3d-dim': [num_x_steps, num_y_steps, num_filters],
                        'vectorization': 'zyx'},
            'num_learnable_params' : filt_x_dim * filt_y_dim * input_z_dim }

# The Maxpooling component assumes input vectorizations of type zyx
def AddMaxpoolingLayer(config_lines, name, input,
                      input_x_dim, input_y_dim, input_z_dim,
                      pool_x_size, pool_y_size, pool_z_size,
                      pool_x_step, pool_y_step, pool_z_step):
    if input_x_dim < 1 or input_y_dim < 1 or input_z_dim < 1:
        raise Exception("non-positive maxpooling input size ({0}, {1}, {2})".
                 format(input_x_dim, input_y_dim, input_z_dim))
    if pool_x_size > input_x_dim or pool_y_size > input_y_dim or pool_z_size > input_z_dim:
        raise Exception("invalid maxpooling pool size vs. input size")
    if pool_x_step > pool_x_size or pool_y_step > pool_y_size or pool_z_step > pool_z_size:
        raise Exception("invalid maxpooling pool step vs. pool size")

    assert(input['dimension'] == input_x_dim * input_y_dim * input_z_dim)
    components = config_lines['components']
    component_nodes = config_lines['component-nodes']

    components.append('component name={name}_maxp type=MaxpoolingComponent '
                      'input-x-dim={input_x_dim} input-y-dim={input_y_dim} input-z-dim={input_z_dim} '
                      'pool-x-size={pool_x_size} pool-y-size={pool_y_size} pool-z-size={pool_z_size} '
                      'pool-x-step={pool_x_step} pool-y-step={pool_y_step} pool-z-step={pool_z_step} '.
                      format(name = name,
                      input_x_dim = input_x_dim, input_y_dim = input_y_dim, input_z_dim = input_z_dim,
                      pool_x_size = pool_x_size, pool_y_size = pool_y_size, pool_z_size = pool_z_size,
                      pool_x_step = pool_x_step, pool_y_step = pool_y_step, pool_z_step = pool_z_step))

    component_nodes.append('component-node name={0}_maxp_t component={0}_maxp input={1}'.format(name, input['descriptor']))

    num_pools_x = 1 + (input_x_dim - pool_x_size) / pool_x_step;
    num_pools_y = 1 + (input_y_dim - pool_y_size) / pool_y_step;
    num_pools_z = 1 + (input_z_dim - pool_z_size) / pool_z_step;
    output_dim = num_pools_x * num_pools_y * num_pools_z;

    return {'output' : {'descriptor':  '{0}_maxp_t'.format(name),
                        'dimension': output_dim,
                        '3d-dim': [num_pools_x, num_pools_y, num_pools_z],
                        'vectorization': 'zyx'},
            'num_learnable_params' : 0 }


def AddSoftmaxLayer(config_lines, name, input):
    components = config_lines['components']
    component_nodes = config_lines['component-nodes']

    components.append("component name={0}_log_softmax type=LogSoftmaxComponent dim={1}".format(name, input['dimension']))
    component_nodes.append("component-node name={0}_log_softmax component={0}_log_softmax input={1}".format(name, input['descriptor']))

    return {'output' : {'descriptor':  '{0}_log_softmax'.format(name),
                        'dimension': input['dimension']},
            'num_learnable_params' : 0 }


def AddSigmoidLayer(config_lines, name, input, self_repair_scale = None):
    components = config_lines['components']
    component_nodes = config_lines['component-nodes']

    # self_repair_scale is a constant scaling the self-repair vector computed in SigmoidComponent
    self_repair_string = "self-repair-scale={0:.10f}".format(self_repair_scale) if self_repair_scale is not None else ''
    components.append("component name={0}_sigmoid type=SigmoidComponent dim={1}".format(name, input['dimension'], self_repair_string))
    component_nodes.append("component-node name={0}_sigmoid component={0}_sigmoid input={1}".format(name, input['descriptor']))
    return {'output' : {'descriptor':  '{0}_sigmoid'.format(name),
                        'dimension': input['dimension']},
            'num_learnable_params' : 0}

def AddOutputLayer(config_lines, input, label_delay = None, suffix=None, objective_type = "linear"):
    components = config_lines['components']
    component_nodes = config_lines['component-nodes']
    name = 'output'
    if suffix is not None:
        name = '{0}-{1}'.format(name, suffix)

    if label_delay is None:
        component_nodes.append('output-node name={0} input={1} objective={2}'.format(name, input['descriptor'], objective_type))
    else:
        component_nodes.append('output-node name={0} input=Offset({1},{2}) objective={3}'.format(name, input['descriptor'], label_delay, objective_type))

def AddFinalLayer(config_lines, input, output_dim,
        ng_affine_options = " param-stddev=0 bias-stddev=0 ",
        label_delay=None,
        use_presoftmax_prior_scale = False,
        prior_scale_file = None,
        include_log_softmax = True,
        add_final_sigmoid = False,
        name_affix = None,
        objective_type = "linear"):
    components = config_lines['components']
    component_nodes = config_lines['component-nodes']

    num_learnable_params = 0
    if name_affix is not None:
        final_node_prefix = 'Final-' + str(name_affix)
    else:
        final_node_prefix = 'Final'

    prev_layer = AddAffineLayer(config_lines,
            final_node_prefix , input, output_dim,
            ng_affine_options)
    prev_layer_output = prev_layer['output']
    num_learnable_params += prev_layer['num_learnable_params']

    if include_log_softmax:
        if use_presoftmax_prior_scale :
            components.append('component name={0}-fixed-scale type=FixedScaleComponent scales={1}'.format(final_node_prefix, prior_scale_file))
            component_nodes.append('component-node name={0}-fixed-scale component={0}-fixed-scale input={1}'.format(final_node_prefix,
                prev_layer_output['descriptor']))
            prev_layer_output['descriptor'] = "{0}-fixed-scale".format(final_node_prefix)
        prev_layer = AddSoftmaxLayer(config_lines, final_node_prefix, prev_layer_output)
        prev_layer_output = prev_layer['output']
        num_learnable_params += prev_layer['num_learnable_params']
    elif add_final_sigmoid:
        # Useful when you need the final outputs to be probabilities
        # between 0 and 1.
        # Usually used with an objective-type such as "quadratic"
        prev_layer = AddSigmoidLayer(config_lines, final_node_prefix, prev_layer_output)
        prev_layer_output = prev_layer['output']
        num_learnable_params += prev_layer['num_learnable_params']
    # we use the same name_affix as a prefix in for affine/scale nodes but as a
    # suffix for output node
    AddOutputLayer(config_lines, prev_layer_output, label_delay, suffix = name_affix, objective_type = objective_type)
    return num_learnable_params

def AddLstmLayer(config_lines,
                 name, input, cell_dim,
                 recurrent_projection_dim = 0,
                 non_recurrent_projection_dim = 0,
                 clipping_threshold = 1.0,
                 norm_based_clipping = "false",
                 ng_per_element_scale_options = "",
                 ng_affine_options = "",
                 lstm_delay = -1,
                 self_repair_scale_nonlinearity = None,
                 self_repair_scale_clipgradient = None):

    assert(recurrent_projection_dim >= 0 and non_recurrent_projection_dim >= 0)
    num_learnable_params = 0
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

    # self_repair_scale_nonlinearity is a constant scaling the self-repair vector computed in derived classes of NonlinearComponent,
    # i.e.,  SigmoidComponent, TanhComponent and RectifiedLinearComponent
    self_repair_nonlinearity_string = "self-repair-scale={0:.10f}".format(self_repair_scale_nonlinearity) if self_repair_scale_nonlinearity is not None else ''
    # self_repair_scale_clipgradient is a constant scaling the self-repair vector computed in ClipGradientComponent
    self_repair_clipgradient_string = "self-repair-scale={0:.2f}".format(self_repair_scale_clipgradient) if self_repair_scale_clipgradient is not None else ''
    # Natural gradient per element scale parameters
    ng_per_element_scale_options += " param-mean=0.0 param-stddev=1.0 "
    # Paramete Definitions W*(* replaced by - to have valid names)
    components.append("# Input gate control : W_i* matrices")
    components.append("component name={0}_W_i-xr type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, input_dim + recurrent_projection_dim, cell_dim, ng_affine_options))
    num_learnable_params += (input_dim + recurrent_projection_dim) * cell_dim
    components.append("# note : the cell outputs pass through a diagonal matrix")
    components.append("component name={0}_w_ic type=NaturalGradientPerElementScaleComponent  dim={1} {2}".format(name, cell_dim, ng_per_element_scale_options))
    num_learnable_params += cell_dim

    components.append("# Forget gate control : W_f* matrices")
    components.append("component name={0}_W_f-xr type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, input_dim + recurrent_projection_dim, cell_dim, ng_affine_options))
    num_learnable_params += (input_dim + recurrent_projection_dim) * cell_dim

    components.append("# note : the cell outputs pass through a diagonal matrix")
    components.append("component name={0}_w_fc type=NaturalGradientPerElementScaleComponent  dim={1} {2}".format(name, cell_dim, ng_per_element_scale_options))
    num_learnable_params += cell_dim

    components.append("#  Output gate control : W_o* matrices")
    components.append("component name={0}_W_o-xr type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, input_dim + recurrent_projection_dim, cell_dim, ng_affine_options))
    num_learnable_params += (input_dim + recurrent_projection_dim) * cell_dim
    components.append("# note : the cell outputs pass through a diagonal matrix")
    components.append("component name={0}_w_oc type=NaturalGradientPerElementScaleComponent  dim={1} {2}".format(name, cell_dim, ng_per_element_scale_options))
    num_learnable_params += cell_dim

    components.append("# Cell input matrices : W_c* matrices")
    components.append("component name={0}_W_c-xr type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, input_dim + recurrent_projection_dim, cell_dim, ng_affine_options))
    num_learnable_params += (input_dim + recurrent_projection_dim) * cell_dim


    components.append("# Defining the non-linearities")
    components.append("component name={0}_i type=SigmoidComponent dim={1} {2}".format(name, cell_dim, self_repair_nonlinearity_string))
    components.append("component name={0}_f type=SigmoidComponent dim={1} {2}".format(name, cell_dim, self_repair_nonlinearity_string))
    components.append("component name={0}_o type=SigmoidComponent dim={1} {2}".format(name, cell_dim, self_repair_nonlinearity_string))
    components.append("component name={0}_g type=TanhComponent dim={1} {2}".format(name, cell_dim, self_repair_nonlinearity_string))
    components.append("component name={0}_h type=TanhComponent dim={1} {2}".format(name, cell_dim, self_repair_nonlinearity_string))

    components.append("# Defining the cell computations")
    components.append("component name={0}_c1 type=ElementwiseProductComponent input-dim={1} output-dim={2}".format(name, 2 * cell_dim, cell_dim))
    components.append("component name={0}_c2 type=ElementwiseProductComponent input-dim={1} output-dim={2}".format(name, 2 * cell_dim, cell_dim))
    components.append("component name={0}_m type=ElementwiseProductComponent input-dim={1} output-dim={2}".format(name, 2 * cell_dim, cell_dim))
    components.append("component name={0}_c type=ClipGradientComponent dim={1} clipping-threshold={2} norm-based-clipping={3} {4}".format(name, cell_dim, clipping_threshold, norm_based_clipping, self_repair_clipgradient_string))

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
        num_learnable_params += cell_dim * (non_recurrent_projection_dim + recurrent_projection_dim)
        components.append("component name={0}_r type=ClipGradientComponent dim={1} clipping-threshold={2} norm-based-clipping={3} {4}".format(name, recurrent_projection_dim, clipping_threshold, norm_based_clipping, self_repair_clipgradient_string))
        component_nodes.append("# r_t and p_t")
        component_nodes.append("component-node name={0}_rp_t component={0}_W-m input={0}_m_t".format(name))
        component_nodes.append("dim-range-node name={0}_r_t_preclip input-node={0}_rp_t dim-offset=0 dim={1}".format(name, recurrent_projection_dim))
        component_nodes.append("component-node name={0}_r_t component={0}_r input={0}_r_t_preclip".format(name))
        output_descriptor = '{0}_rp_t'.format(name)
        output_dim = recurrent_projection_dim + non_recurrent_projection_dim

    elif add_recurrent_projection:
        components.append("# projection matrices : Wrm")
        components.append("component name={0}_Wrm type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, cell_dim, recurrent_projection_dim, ng_affine_options))
        num_learnable_params += cell_dim * recurrent_projection_dim
        components.append("component name={0}_r type=ClipGradientComponent dim={1} clipping-threshold={2} norm-based-clipping={3} {4}".format(name, recurrent_projection_dim, clipping_threshold, norm_based_clipping, self_repair_clipgradient_string))
        component_nodes.append("# r_t")
        component_nodes.append("component-node name={0}_r_t_preclip component={0}_Wrm input={0}_m_t".format(name))
        component_nodes.append("component-node name={0}_r_t component={0}_r input={0}_r_t_preclip".format(name))
        output_descriptor = '{0}_r_t'.format(name)
        output_dim = recurrent_projection_dim

    else:
        components.append("component name={0}_r type=ClipGradientComponent dim={1} clipping-threshold={2} norm-based-clipping={3} {4}".format(name, cell_dim, clipping_threshold, norm_based_clipping, self_repair_clipgradient_string))
        component_nodes.append("component-node name={0}_r_t component={0}_r input={0}_m_t".format(name))
        output_descriptor = '{0}_r_t'.format(name)
        output_dim = cell_dim

    return {'output' : {'descriptor': output_descriptor,
                        'dimension':output_dim},
            'num_learnable_params' : num_learnable_params}


def AddBLstmLayer(config_lines,
                  name, input, cell_dim,
                  recurrent_projection_dim = 0,
                  non_recurrent_projection_dim = 0,
                  clipping_threshold = 1.0,
                  norm_based_clipping = "false",
                  ng_per_element_scale_options = "",
                  ng_affine_options = "",
                  lstm_delay = [-1,1],
                  self_repair_scale_nonlinearity = None,
                  self_repair_scale_clipgradient = None):
    assert(len(lstm_delay) == 2 and lstm_delay[0] < 0 and lstm_delay[1] > 0)
    num_learnable_params = 0
    prev_layer = AddLstmLayer(config_lines, "{0}_forward".format(name), input, cell_dim,
                                  recurrent_projection_dim, non_recurrent_projection_dim,
                                  clipping_threshold, norm_based_clipping,
                                  ng_per_element_scale_options, ng_affine_options,
                                  lstm_delay = lstm_delay[0],
                                  self_repair_scale_nonlinearity = self_repair_scale_nonlinearity,
                                  self_repair_scale_clipgradient = self_repair_scale_clipgradient)
    output_forward = prev_layer['output']
    num_learnable_params += prev_layer['num_learnable_params']

    prev_layer  = AddLstmLayer(config_lines, "{0}_backward".format(name), input, cell_dim,
                                   recurrent_projection_dim, non_recurrent_projection_dim,
                                   clipping_threshold, norm_based_clipping,
                                   ng_per_element_scale_options, ng_affine_options,
                                   lstm_delay = lstm_delay[1],
                                   self_repair_scale_nonlinearity = self_repair_scale_nonlinearity,
                                   self_repair_scale_clipgradient = self_repair_scale_clipgradient)
    output_backward = prev_layer['output']
    num_learnable_params += prev_layer['num_learnable_params']

    output_descriptor = 'Append({0}, {1})'.format(output_forward['descriptor'], output_backward['descriptor'])
    output_dim = output_forward['dimension'] + output_backward['dimension']

    return {'output' : {'descriptor': output_descriptor,
                        'dimension':output_dim},
            'num_learnable_params' : num_learnable_params}


#########################################################################################
# Clockwork RNN
#########################################################################################

def AddCwrnnLayer(config_lines,
                  name, input,
                  operating_time_period,
                  clipping_threshold = 1.0,
                  norm_based_clipping = "false",
                  ng_per_element_scale_options = "",
                  ratewise_params = {'T1': {'rate':1, 'dim':128},
                                     'T2': {'rate':1.0/2, 'dim':128},
                                     'T3': {'rate':1.0/4, 'dim':128},
                                     'T4': {'rate':1.0/8, 'dim':128}
                                     },
                  nonlinearity = "RectifiedLinearComponent+NormalizedComponent",
                  diag_init_scaling_factor = 0,
                  input_type = "none",
                  projection_dim = 0,
                  self_repair_scale_nonlinearity=None,
                  self_repair_scale_clipgradient=None):

    components = config_lines['components']
    component_nodes = config_lines['component-nodes']

    num_learnable_params = 0
    key_rate_pairs = map(lambda key: (key, ratewise_params[key]['rate']), ratewise_params)
    key_rate_pairs.sort(key=itemgetter(1))
    sorted_keys = map(lambda x: x[0], key_rate_pairs)
    largest_time_period = int(1.0/ratewise_params[sorted_keys[0]]['rate'])

    slow_to_fast_descriptors = {}
    for key in sorted_keys:
        slow_to_fast_descriptors[key] = {}

    output_descriptors_all_rates = {}
    for key_index in xrange(len(sorted_keys)):
        key = sorted_keys[key_index]
        params = ratewise_params[key]
        rate = params['rate']
        time_period = int(round(1.0/params['rate']))
        cw_unit_input_descriptor = input
        fastrate_params = {}
        for fast_key_index in range(key_index+1, len(sorted_keys)):
            fast_key = sorted_keys[fast_key_index]
            fastrate_params[fast_key] = ratewise_params[fast_key]

        [output_descriptor, fastrate_output_descriptors, cur_num_learnable_params] = AddCwrnnRateUnit(config_lines,
                                                                            name = '{0}_{1}'.format(name, key),
                                                                            input = cw_unit_input_descriptor,
                                                                            output_dim = params['dim'],
                                                                            unit_rate = params['rate'],
                                                                            operating_time_period = operating_time_period,
                                                                            clipping_threshold = clipping_threshold,
                                                                            norm_based_clipping = norm_based_clipping,
                                                                            delay = -1 * time_period,
                                                                            slowrate_descriptors = slow_to_fast_descriptors[key],
                                                                            fastrate_params = fastrate_params,
                                                                            nonlinearity = nonlinearity,
                                                                            diag_init_scaling_factor = diag_init_scaling_factor,
                                                                            input_type = input_type,
                                                                            self_repair_scale_nonlinearity = self_repair_scale_nonlinearity,
                                                                            self_repair_scale_clipgradient = self_repair_scale_clipgradient)
        num_learnable_params += cur_num_learnable_params
        output_descriptors_all_rates[key] = output_descriptor
        for fastrate in fastrate_output_descriptors.keys():
            slow_to_fast_descriptors[fastrate][key] = fastrate_output_descriptors[fastrate]

    # concatenate the output descriptors
    # add an if-defined for all descriptors leaving the one corresponding
    # to the fastest descriptor
    descriptor_strings = []
    dimension = 0
    for key in sorted_keys[:-1]:
        cur_descriptor = output_descriptors_all_rates[key]
        descriptor_strings.append('IfDefined({0})'.format(cur_descriptor['descriptor']))
        dimension += cur_descriptor['dimension']
    # this is the fastest rate so there is no if_defined
    descriptor_strings.append(output_descriptors_all_rates[sorted_keys[-1]]['descriptor'])
    dimension += output_descriptors_all_rates[sorted_keys[-1]]['dimension']


    output_descriptor = {}
    output_descriptor['descriptor'] = 'Append({0})'.format(', '.join(descriptor_strings))
    output_descriptor['dimension'] = dimension

    if projection_dim > 0:
        prev_layer = AddAffRelNormLayer(config_lines, '{0}_project'.format(name),
                                        output_descriptor, projection_dim)
        output_descriptor = prev_layer['output']
        num_learnable_params += prev_layer['num_learnable_params']

    return {'output' : output_descriptor,
            'num_learnable_params' : num_learnable_params,
            'largest_time_period' : largest_time_period}

def AddMultiRateTdnnLayer(config_lines, name, input,
                          rate_params, splice_indexes,
                          nonlin_type, nonlin_input_dim, nonlin_output_dim,
                          operating_time_period,
                          slow_rate_optional,
                          self_repair_scale, norm_target_rms):

    print(rate_params)
    if nonlin_type != 'relu':
        raise Exception('Multi-rate TDNN layer only supports ReLU nonlinearities')
    assert(nonlin_input_dim == nonlin_output_dim)
    nonlin_dim = nonlin_input_dim

    proj_outputs = []
    num_learnable_params = 0
    left_context = 0
    right_context = 0

    max_unit_time_period = 0
    num_slow_rates = 0
    for rate in rate_params.keys():
        rate_param = rate_params[rate]
        unit_time_period = int(1.0 / rate_param['rate'])
        max_unit_time_period = max(unit_time_period, max_unit_time_period)
        if unit_time_period > operating_time_period:
            num_slow_rates += 1
        if unit_time_period < operating_time_period:
            raise Exception("Invalid rate {rate} : Rate {rate} has unit time period {tp},"
                            " which is shorter than operating time period {otp}.".format(rate = rate_param['rate'],
                                                                                         tp = unit_time_period,
                                                                                         otp = operating_time_period))

    prev_layer = AddPerDimAffineLayerMulti(config_lines,
                                            '{0}_{1}'.format(name, max_unit_time_period),
                                            input, num_slow_rates,
                                            2 * (max_unit_time_period - operating_time_period) + 1,
                                            time_period = operating_time_period,
                                            inputs_optional = slow_rate_optional)

    prev_layer_outputs = prev_layer['outputs']
    num_learnable_params += prev_layer['num_learnable_params']
    if not slow_rate_optional:
        # if slow rate is optional then we don't need the context
        # I am assuming that only the slow rate units have the
        # PerDimAffine processed inputs
        left_context = prev_layer['filter_left_context']
        right_context = prev_layer['filter_right_context']

    rates = rate_params.keys()
    filtered_input_index = 0
    for i in xrange(len(rate_params.keys())):
        rate = rates[i]
        rate_param = rate_params[rate]
        unit_time_period = int(1.0 / rate_param['rate'])

        # filter input
        if unit_time_period == operating_time_period :
            unit_input = input
        else:
            unit_input = prev_layer_outputs[filtered_input_index]
            filtered_input_index += 1

        prev_layer_output = SpliceInput(unit_input, splice_indexes)

        prev_layer = AddAffRelNormLayer(config_lines,
                                        '{0}_{1}'.format(name, unit_time_period),
                                         prev_layer_output, nonlin_output_dim,
                                         norm_target_rms = norm_target_rms,
                                         self_repair_scale = self_repair_scale)
        prev_layer_output = prev_layer['output']
        num_learnable_params += prev_layer['num_learnable_params']

        if unit_time_period != operating_time_period:
            # this is the main thing here.
            # the use of round descriptor reduces the rate of different components
            #if slow_rate_optional:
            #    proj_outputs.append('IfDefined(Round({0},{1}))'.format(prev_layer_output['descriptor'], unit_time_period))
            #else:
            #    proj_outputs.append('Round({0},{1})'.format(prev_layer_output['descriptor'], unit_time_period))
            proj_outputs.append('Round({0},{1})'.format(prev_layer_output['descriptor'], unit_time_period))
        else:
            proj_outputs.append(prev_layer_output['descriptor'])

    output = GetSumDescriptor(proj_outputs)[0]

    return  {'output' : {'descriptor': output,
                         'dimension' : nonlin_output_dim},
             'num_learnable_params' : num_learnable_params,
             'left_context' : left_context,
             'right_context' : right_context}

def AddAffineNonlinLayer(config_lines, name, input,
                         nonlin_type, nonlin_input_dim, nonlin_output_dim,
                         self_repair_scale, norm_target_rms):
    if nonlin_type == "relu":
        prev_layer = AddAffRelNormLayer(config_lines, name,
                                               input, nonlin_output_dim,
                                               self_repair_scale = self_repair_scale,
                                               norm_target_rms = norm_target_rms)
    elif nonlin_type == "pnorm":
        prev_layer = AddAffPnormLayer(config_lines, name,
                                             input, nonlin_input_dim, nonlin_output_dim,
                                             norm_target_rms = norm_target_rms)
    else:
        raise Exception("Unknown nonlinearity type")
    return prev_layer


# Convenience functions

def SpliceInput(input, splice_indexes):

    appended_descriptors = []
    appended_dimension = 0

    try:
        zero_index = splice_indexes.index(0)
    except ValueError:
        zero_index = None

    for j in range(len(splice_indexes)):
        if j == zero_index:
            appended_descriptors.append(input['descriptor'])
            appended_dimension += input['dimension']
            continue
        appended_descriptors.append('Offset({0}, {1})'.format(input['descriptor'], splice_indexes[j]))
        appended_dimension += input['dimension']

    return {'descriptor' : "Append({0})".format(" , ".join(appended_descriptors)),
            'dimension'  : appended_dimension}

def AddFinalLayersWithXentSeperateForwardAffineRegularizer(config_lines,
                                     input, num_targets,
                                     nonlin_type, nonlin_input_dim, nonlin_output_dim,
                                     use_presoftmax_prior_scale,
                                     prior_scale_file,
                                     include_log_softmax,
                                     self_repair_scale,
                                     xent_regularize,
                                     final_layer_normalize_target):

    num_learnable_params = 0
    num_learnable_params_xent = 0
    if nonlin_type == "relu" :
        prev_layer_chain = AddAffRelNormLayer(config_lines, "Pre_final_chain",
                                                           input, nonlin_output_dim,
                                                           self_repair_scale = self_repair_scale,
                                                           norm_target_rms = final_layer_normalize_target)
        prev_layer_xent = AddAffRelNormLayer(config_lines, "Pre_final_xent",
                                                          input, nonlin_output_dim,
                                                          self_repair_scale = self_repair_scale,
                                                          norm_target_rms = final_layer_normalize_target)
    elif nonlin_type == "pnorm" :
        prev_layer_chain = AddAffPnormLayer(config_lines, "Pre_final_chain",
                                                         input, nonlin_input_dim, nonlin_output_dim,
                                                         norm_target_rms = final_layer_normalize_target)

        prev_layer_xent = AddAffPnormLayer(config_lines, "Pre_final_xent",
                                                        input, nonlin_input_dim, nonlin_output_dim,
                                                        norm_target_rms = final_layer_normalize_target)
    else:
        raise Exception("Unknown nonlinearity type")

    prev_layer_output_chain = prev_layer_chain['output']
    prev_layer_output_xent = prev_layer_xent['output']

    num_learnable_params += prev_layer_chain['num_learnable_params']
    num_learnable_params_xent += prev_layer_xent['num_learnable_params']

    num_learnable_params += AddFinalLayer(config_lines, prev_layer_output_chain, num_targets,
                                              use_presoftmax_prior_scale = use_presoftmax_prior_scale,
                                              prior_scale_file = prior_scale_file,
                                              include_log_softmax = include_log_softmax)

    # This block prints the configs for a separate output that will be
    # trained with a cross-entropy objective in the 'chain' models... this
    # has the effect of regularizing the hidden parts of the model.  we use
    # 0.5 / args.xent_regularize as the learning rate factor- the factor of
    # 1.0 / args.xent_regularize is suitable as it means the xent
    # final-layer learns at a rate independent of the regularization
    # constant; and the 0.5 was tuned so as to make the relative progress
    # similar in the xent and regular final layers.
    num_learnable_params_xent += AddFinalLayer(config_lines, prev_layer_output_xent, num_targets,
                                                   ng_affine_options = " param-stddev=0 bias-stddev=0 learning-rate-factor={0} ".format(
                                                   0.5 / xent_regularize),
                                                   use_presoftmax_prior_scale = use_presoftmax_prior_scale,
                                                   prior_scale_file = prior_scale_file,
                                                   include_log_softmax = True,
                                                   name_affix = 'xent')

    return [num_learnable_params, num_learnable_params_xent]

def AddFinalLayerWithXentRegularizer(config_lines, input, num_targets,
                                     nonlin_type, nonlin_input_dim, nonlin_output_dim,
                                     use_presoftmax_prior_scale,
                                     prior_scale_file,
                                     include_log_softmax,
                                     self_repair_scale,
                                     xent_regularize,
                                     add_final_sigmoid,
                                     objective_type):

    # add_final_sigmoid adds a sigmoid as a final layer as alternative
    # to log-softmax layer.
    # http://ufldl.stanford.edu/wiki/index.php/Softmax_Regression#Softmax_Regression_vs._k_Binary_Classifiers
    # This is useful when you need the final outputs to be probabilities between 0 and 1.
    # Usually used with an objective-type such as "quadratic".
    # Applications are k-binary classification such Ideal Ratio Mask prediction.
    num_learnable_params = AddFinalLayer(config_lines, input, num_targets,
                       use_presoftmax_prior_scale = use_presoftmax_prior_scale,
                       prior_scale_file = prior_scale_file,
                       include_log_softmax = include_log_softmax,
                       add_final_sigmoid = add_final_sigmoid,
                       objective_type = objective_type)

    if xent_regularize != 0.0:
        # This block prints the configs for a separate output that will be
        # trained with a cross-entropy objective in the 'chain' models... this
        # has the effect of regularizing the hidden parts of the model.  we use
        # 0.5 / args.xent_regularize as the learning rate factor- the factor of
        # 1.0 / args.xent_regularize is suitable as it means the xent
        # final-layer learns at a rate independent of the regularization
        # constant; and the 0.5 was tuned so as to make the relative progress
        # similar in the xent and regular final layers.
        num_learnable_params_xent = AddFinalLayer(config_lines, input, num_targets,
                            ng_affine_options = " param-stddev=0 bias-stddev=0 learning-rate-factor={0} ".format(
                                  0.5 / xent_regularize),
                            use_presoftmax_prior_scale = use_presoftmax_prior_scale,
                            prior_scale_file = prior_scale_file,
                            include_log_softmax = True,
                            name_affix = 'xent')

    return [num_learnable_params, num_learnable_params_xent]

# Note : this method is not a Add*Layer method we don't intend it to be directly called
# by the top level scripts. The output of this Unit does not follow the expected
# convention
def AddCwrnnRateUnit(config_lines,
                    name, input, output_dim,
                    unit_rate,
                    operating_time_period,
                    clipping_threshold = 1.0,
                    norm_based_clipping = "false",
                    ng_affine_options = " bias-stddev=0 ",
                    delay = -1,
                    slowrate_descriptors = {},
                    fastrate_params = {},
                    nonlinearity = "SigmoidComponent",
                    diag_init_scaling_factor = 0,
                    input_type = "none",
                    self_repair_scale_nonlinearity=None,
                    self_repair_scale_clipgradient=None):

    num_learnable_params = 0
    # self_repair_scale_nonlinearity is a constant scaling the self-repair vector computed in derived classes of NonlinearComponent,
    # i.e.,  SigmoidComponent, TanhComponent and RectifiedLinearComponent
    self_repair_nonlinearity_string = "self-repair-scale={0:.10f}".format(self_repair_scale_nonlinearity) if self_repair_scale_nonlinearity is not None else ''
    # self_repair_scale_clipgradient is a constant scaling the self-repair vector computed in ClipGradientComponent
    self_repair_clipgradient_string = "self-repair-scale={0:.2f}".format(self_repair_scale_clipgradient) if self_repair_scale_clipgradient is not None else ''

    if diag_init_scaling_factor != 0:
        raise NotImplementedError
        # this is tricky to implement and not so helpful so I will not implement unless absolutely necessary
        warnings.warn("Ignoring ng_affine_options, diag_init_scaling_factor is implemented by specifying the parameter matrix as a file",
                      stacklevel=3)
        # we use a single matrix from each input rate to all output rates,
        # so diag-init-scaling factor no longer works
        # we would have to load a matrix which follows this structure

    # assign some variables to reduce verbosity of the code
    components = config_lines['components']
    component_nodes = config_lines['component-nodes']
    input_descriptor = input['descriptor']
    input_dim = input['dimension']
    name = name.strip()

    recurrent_connection = "r_t"
    if len(slowrate_descriptors.keys()) > 0:
        slowrate_sum_descriptor = GetSumDescriptor(map(lambda x: x['descriptor'], slowrate_descriptors.values()))[0]
    else:
        slowrate_sum_descriptor = ''

    unit_time_period = int(1.0 / unit_rate)

    # process the input & update the input variables
    if input_type in set(["stack", "sum"]):
        splice_indexes = range(-1*(unit_time_period-1), 1, operating_time_period)
        list = [('Offset({0}, {1})'.format(input['descriptor'], n) if n != 0 else input['descriptor']) for n in splice_indexes]
        if input_type == "stack":
            input_descriptor = 'Append({0})'.format(' , '.join(list))
            input_dim = len(splice_indexes) * input_dim
        elif input_type == "sum":
            input_descriptor = GetSumDescriptor(list)[0]
    elif input_type == "per-dim-weighted-average":
        prev_layer = AddPerDimAffineLayer(config_lines, name, input, 2 * (unit_time_period - operating_time_period) + 1, time_period = operating_time_period)
        input = prev_layer['output']
        num_learnable_params += prev_layer['num_learnable_params']

        input_dim = input['dimension']
        input_descriptor = input['descriptor']

    # component definitions W*(* replaced by - to have valid names)
    components.append("# Components for rate {0}: Wxr".format(unit_rate))
    total_fastrate_unit_dim = sum(map(lambda x: x['dim'], fastrate_params.values()))
    recurrent_output_dim = output_dim + total_fastrate_unit_dim
    components.append("component name={0}_W_r type=NaturalGradientAffineComponent"
                      " input-dim={1} output-dim={2} rank-in={3} rank-out={4} {5}".format(
                      name, output_dim, recurrent_output_dim,
                      int(output_dim * 0.1), int(recurrent_output_dim * 0.1),
                      ng_affine_options))

    num_learnable_params += output_dim * recurrent_output_dim

    components.append("component name={0}_W_x type=NaturalGradientAffineComponent"
                      " input-dim={1} output-dim={2} rank-in={3} rank-out={4} {5}".format(
                      name, input_dim, output_dim,
                      int(input_dim * 0.1), int(output_dim * 0.1),
                      ng_affine_options))
    num_learnable_params += input_dim * output_dim
    component_nodes.append("component-node name={0}_W_x_t component={0}_W_x input={1}".format(name, input_descriptor))
    descriptors = [ '{0}_W_r_t'.format(name), '{0}_W_x_t'.format(name)]
    if slowrate_sum_descriptor != '':
        descriptors.append(slowrate_sum_descriptor)
    nonlin_input_sum_descriptor = GetSumDescriptor(descriptors)[0]


    # Add the non-linearity component.
    if nonlinearity == "RectifiedLinearComponent+NormalizeComponent":
        components.append("component name={0}_relu type=RectifiedLinearComponent dim={1} {2}".format(name, output_dim, self_repair_nonlinearity_string))
        components.append("component name={0}_renorm type=NormalizeComponent dim={1} target-rms={2}".format(name, output_dim, 1.0))

        component_nodes.append("component-node name={0}_{1}_relu component={0}_relu input={2}".format(name, recurrent_connection, nonlin_input_sum_descriptor))
        component_nodes.append("component-node name={0}_{1}_preclip component={0}_renorm input={0}_{1}_relu".format(name, recurrent_connection))

    elif nonlinearity in ['SigmoidComponent', 'TanhComponent', 'RectifiedLinearComponent']:
        components.append("component name={0}_nonlin type={1} dim={2} {3}".format(name, nonlinearity, output_dim, self_repair_nonlinearity_string))
        component_nodes.append("component-node name={0}_{1}_preclip component={0}_nonlin input={3}".format(name, recurrent_connection, nonlin_input_sum_descriptor))
    else:
        raise Exception("Unknown non-linearity type {0}.".format(nonlinearity))

    # Adding the gradient clipper
    components.append("component name={0}_clip type=ClipGradientComponent dim={1} clipping-threshold={2} norm-based-clipping={3} {4} ".format(name, output_dim, clipping_threshold, norm_based_clipping, self_repair_clipgradient_string))
    component_nodes.append("component-node name={0}_{1} component={0}_clip input={0}_{1}_preclip".format(name, recurrent_connection))


    # take the recurrent outputs from the current unit and divide them among all the fast rate units
    fastrate_output_descriptors = {}
    keys = fastrate_params.keys()
    if total_fastrate_unit_dim == 0:
        # there are no recurrent connections from this rate unit to faster rate  units
        component_nodes.append("component-node name={0}_W_r_t component={0}_W_r input=IfDefined(Offset({0}_{1}, {2}))".format(name, recurrent_connection, delay))
    else:
        # there are recurrent connections from this rate unit to faster rate  units
        # we will compute all these in a single matrix multiplication
        component_nodes.append("component-node name={0}_W_r_all_t component={0}_W_r input=IfDefined(Offset({0}_{1}, {2}))".format(name, recurrent_connection, delay))

        # the first subset in the recurrent outputs corresponds to the current rate
        component_nodes.append("dim-range-node name={0}_W_r_t input-node={0}_W_r_all_t dim-offset=0 dim={1}".format(name, output_dim))

        # dividing rest of the recurrent outputs among the faster rates
        offset = output_dim
        for key in fastrate_params.keys():
            params = fastrate_params[key]
            fastrate_time_period = int(1.0/params['rate'])
            output_name = '{0}_W_rfast-tmod{1}_t'.format(name, fastrate_time_period)
            component_nodes.append("dim-range-node name={1} input-node={0}_W_r_all_t dim-offset={2} dim={3}".format(name, output_name, offset, params['dim']))
            offset += params['dim']
            fastrate_descriptor = 'Round({0}, {1})'.format(output_name, unit_time_period)
            fastrate_output_descriptors[key] = {'descriptor': fastrate_descriptor,
                                                'dimension' : params['dim']}

    output_descriptor = "{0}_{1}".format(name, recurrent_connection)
    return [{'descriptor': output_descriptor, 'dimension':output_dim}, fastrate_output_descriptors, num_learnable_params]
