# Copyright 2016    Johns Hopkins University (Dan Povey)
#           2016    Vijayaditya Peddinti
# Apache 2.0.


""" This module has the implementations of different MRTDNN layers.
"""
import re

from libs.nnet3.xconfig.basic_layers import XconfigLayerBase
from libs.nnet3.xconfig.utils import XconfigParserError as xparser_error
from libs.nnet3.xconfig.layers import XconfigTdnnLayer

class XconfigMtdnnLayer(XconfigTdnnLayer):
    """This class extends the TDNN class to add the multi-rate functionality.
    """


    def __init__(self, first_token, key_to_value, prev_names = None):
        assert first_token == "mtdnn-layer"
        XconfigLayerBase.__init__(self, first_token, key_to_value, prev_names)

    def set_default_configs(self):

        super(XconfigMtdnnLayer, self).set_default_configs()
        child_config = {'operating-time-period' : 3,
                        'time-periods' : '3,6,9',
                        'rate-dims' : '512,512,512',
                        'slow-rate-optional' : False}
        self.config.update(child_config)

    def check_configs(self):

        super(XconfigMtdnnLayer, self).check_configs()
        if self.config['operating-time-period'] < 1:
            raise xparser_error('Invalid value for operating-time-period')
        # get_rate_params performs all the checks
        self.get_rate_params()

    def output_name(self, auxiliary_output = None):

        # this component uses round descriptors and combination of these
        # descriptors with offset descriptors can negate any performance
        # gains we anticipate. So we might never expose the intermediate outputs
        assert auxiliary_output == None
        return '{0}.proj-renorm'.format(self.name)

    def get_rate_params(self):
        """This function returns the rate-wise params of the multi-rate TDNN.
        """

        params = {}
        for key in ['time-periods', 'rate-dims']:
            value = self.config[key]
            try:
                params[key] = map(lambda x: int(x), value.split(","))
            except ValueError:
                raise xparser_error("Invalid value for {0}: {1}"
                                    "".format(key, value))

        if len(params['time-periods']) != len(params['rate-dims']):
            raise xparser_error("time-periods and rate-dims should have the"
                                " same number of elements.")

        rate_params = {}
        for i in xrange(len(params['time-periods'])):
            tp = params['time-periods'][i]
            dim = params['rate-dims'][i]
            if dim <= 0:
                raise xparser_error("Invalid value {0} in rate-dims."
                                    "".format(dim), self(str))
            if tp <= 1:
                raise xparser_error("Invalid value {0} in time-periods."
                                    "".format(tp), self(str))
            rate_params[tp] = dim

        return rate_params


    def get_processed_input(self, time_period):
        """This method will process the input for the particular time-period.
        The most simple processing would be just sum the inputs. But we
        can try other things like applying one-dimensional filters, or
        appending a subset of the inputs from non-zero time frames. These
        will be done in child classes which will over-ride this method.
        """

        input_dim = self.descriptors['input']['dim']
        input_descriptor = self.descriptors['input']['final-string']
        operating_period = self.config['operating-time-period']

        if time_period == operating_period:
            return [input_descriptor, input_dim]
        else:
            steps = (time_period / operating_period + 1) / 2
            filter_input_splice_indexes = map(lambda x:
                    operating_period * x, range(-steps, steps + 1))
            inputs_to_sum = []
            for index in filter_input_splice_indexes:
                if index == 0:
                    inputs_to_sum.append(input_descriptor)
                else:
                    inputs_to_sum.append('Offset({0}, {1})'
                                         ''.format(input_descriptor, index))
            return [self.sum_the_descriptors(inputs_to_sum), input_dim]


    def _generate_config(self):

        # assign some variables to reduce verbosity
        name = self.name
        # in the below code we will just call descriptor_strings as descriptors for conciseness
        output_dim = self.output_dim()
        rate_params = self.get_rate_params()
        ng_opt_str = self.config['ng-affine-options']
        self_repair_scale = self.config['self-repair-scale']
        target_rms = self.config['target-rms']
        slow_rate_optional = self.config['slow-rate-optional']
        operating_period = self.config['operating-time-period']

        configs = []
        rate_descs = []
        for time_period in rate_params.keys():
            rate_desc, rate_configs = self._add_rate_unit(time_period)
            configs += rate_configs
            if slow_rate_optional and operating_period < time_period:
                rate_desc = 'IfDefined({0})'.format(rate_desc)
            rate_descs.append(rate_desc)

        sum_desc = self.sum_the_descriptors(rate_descs)

        # We have to pass the sum_descriptor through a noop-component
        # as we don't want to directly expose the summed and rounded descriptors
        # to the other components. This is because the other components can
        # apply offsets on these descriptors and the combination of offset
        # and round descriptors can ultimately end up negating any computational
        # benefits we hoped to get through the use of the round descriptors.
        # We might as well perform a relu operation in place of the noop.
        relu_name='{0}.proj-relu'.format(self.name)
        line = ('component name={0}'
                ' type=RectifiedLinearComponent dim={1}'
                ' self-repair-scale={2}'
                ''.format(relu_name, output_dim, self_repair_scale))
        configs.append(line)
        line = ('component-node name={0}'
                ' component={0} input={1}'
                ''.format(relu_name, sum_desc))
        configs.append(line)

        renorm_name='{0}.proj-renorm'.format(self.name)
        line = ('component name={0}'
                ' type=NormalizeComponent dim={1}'
                ' target-rms={2}'
                ''.format(renorm_name, output_dim, target_rms))
        configs.append(line)
        line = ('component-node name={0}'
                ' component={0} input={1}'
                ''.format(renorm_name, relu_name))
        configs.append(line)

        return configs


    def _add_rate_unit(self, time_period):

        configs = []
        output_dim = self.output_dim()
        rate_params = self.get_rate_params()
        dim = rate_params[time_period]
        input_desc, input_dim = self.get_processed_input(time_period)

        input_desc, input_dim, sp_configs = self.splice_input(input_desc,
                input_dim, self.get_splice_indexes(),
                self.config['subset-dim'],
                dim_range_node_name = '{0}.tp{1}.input_subset'
                                      ''.format(self.name,
                                          time_period))
        configs += sp_configs

        ng_opt_str = self.config['ng-affine-options']
        affine_name='{0}.tp{1}.affine'.format(self.name, time_period)
        line = ('component name={0}'
                ' type=NaturalGradientAffineComponent'
                ' input-dim={1}'
                ' output-dim={2}'
                ' {3}'.format(affine_name, input_dim, dim, ng_opt_str))
        configs.append(line)

        line = ('component-node name={0}'
                ' component={0} input={1}'
                ''.format(affine_name, input_desc))
        configs.append(line)

        self_repair_scale = self.config['self-repair-scale']
        relu_name='{0}.tp{1}.relu'.format(self.name, time_period)
        line = ('component name={0}'
                ' type=RectifiedLinearComponent dim={1}'
                ' self-repair-scale={2}'
                ''.format(relu_name, dim, self_repair_scale))
        configs.append(line)
        line = ('component-node name={0}'
                ' component={0} input={1}'
                ''.format(relu_name, affine_name))
        configs.append(line)

        target_rms = self.config['target-rms']
        renorm_name='{0}.tp{1}.renorm'.format(self.name, time_period)
        line = ('component name={0}'
                ' type=NormalizeComponent dim={1}'
                ' target-rms={2}'
                ''.format(renorm_name, dim, target_rms))
        configs.append(line)
        line = ('component-node name={0}'
                ' component={0} input={1}'
                ''.format(renorm_name, relu_name))
        configs.append(line)

        # add the output projection
        affine_name='{0}.tp{1}.proj-affine'.format(self.name, time_period)
        line = ('component name={0}'
                ' type=NaturalGradientAffineComponent'
                ' input-dim={1}'
                ' output-dim={2}'
                ' {3}'.format(affine_name, dim,
                              output_dim, ng_opt_str))
        configs.append(line)
        line = ('component-node name={0}'
                ' component={0} input={1}'
                ''.format(affine_name, renorm_name))
        configs.append(line)

        operating_period = self.config['operating-time-period']
        if time_period > operating_period:
            return 'Round({0}, {1})'.format(affine_name, time_period), configs
        else:
            return affine_name, configs


    @staticmethod
    def sum_the_descriptors(descriptors):
        """This is a convenience function to create a single sum descriptor
        which sums all the input descriptors using the binary C++ Sum()
        descriptor.
        """

        # checking the descriptors are not empty
        assert(descriptors)
        sum_descriptors = descriptors
        if len(descriptors) == 1:
            return descriptors[0]
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
        return sum_descriptors[0]




class XconfigMtdnnlLayer(XconfigMtdnnLayer):


    def __init__(self, first_token, key_to_value, prev_names = None):
        assert first_token == "mtdnnl-layer"
        XconfigLayerBase.__init__(self, first_token, key_to_value, prev_names)

    def get_processed_input(self, time_period):
        """This method will process the input for the particular time-period.
        This is same as the one in MtdnnLayer except that it uses only the
        left contexts.
        """

        input_dim = self.descriptors['input']['dim']
        input_descriptor = self.descriptors['input']['final-string']
        operating_period = self.config['operating-time-period']

        if time_period == operating_period:
            return [input_descriptor, input_dim]
        else:
            steps = int(time_period / operating_period)
            filter_input_splice_indexes = map(lambda x:
                    operating_period * x, range(-(steps-1), 1))
            inputs_to_sum = []
            for index in filter_input_splice_indexes:
                if index == 0:
                    inputs_to_sum.append(input_descriptor)
                else:
                    inputs_to_sum.append('Offset({0}, {1})'
                                         ''.format(input_descriptor, index))
            return [self.sum_the_descriptors(inputs_to_sum), input_dim]

class XconfigMtdnnncLayer(XconfigMtdnnLayer):


    def __init__(self, first_token, key_to_value, prev_names = None):
        assert first_token == "mtdnnnc-layer"
        XconfigLayerBase.__init__(self, first_token, key_to_value, prev_names)

    def get_processed_input(self, time_period):
        """This method will process the input for the particular time-period.
        This is same as the one in MtdnnLayer except that it uses only the
        left contexts.
        """

        input_dim = self.descriptors['input']['dim']
        input_descriptor = self.descriptors['input']['final-string']

        return [input_descriptor, input_dim]
