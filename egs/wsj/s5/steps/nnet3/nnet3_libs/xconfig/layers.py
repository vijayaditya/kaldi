from __future__ import print_function
import subprocess
import logging
import math
import re
import sys
import traceback
import time
import argparse
import utils



# A base-class for classes representing layers of xconfig files.
# This mainly just sets self.layer_type, self.name and self.config,
class XconfigLayerBase(object):
    # Constructor.
    # first_token is the first token on the xconfig line, e.g. 'affine-layer'.f
    # key_to_value is a dict like:
    # { 'name':'affine1', 'input':'Append(0, 1, 2, ReplaceIndex(ivector, t, 0))', 'dim=1024' }.
    # The only required and 'special' values that are dealt with directly at this level, are
    # 'name' and 'input'.
    # The rest are put in self.config and are dealt with by the child classes' init functions.
    # all_layers is an array of objects inheriting XconfigLayerBase for all previously
    # parsed layers.

    def __init__(self, first_token, key_to_value, all_layers):
        self.layer_type = first_token
        if not 'name' in key_to_value:
            raise RuntimeError("Expected 'name' to be specified.")
        self.name = key_to_value['name']
        if not xconfig_utils.IsValidLineName(self.name):
            raise RuntimeError("Invalid value: name={0}".format(key_to_value['name']))

        # the following, which should be overridden in the child class, sets
        # default config parameters in self.config.
        self.SetDefaultConfigs()
        # The following is not to be reimplemented in child classes;
        # it sets the config values to those specified by the user, and
        # parses any Descriptors.
        self.SetConfigs(key_to_value, all_layers)
        # This method, sets the derived default config values i.e., config values
        # when not specified can be derived from other values.
        # It can be overridden in the child class.
        self.SetDerivedConfigs()
        # the following, which should be overridden in the child class, checks
        # that the config parameters that have been set are reasonable.
        self.CheckConfigs()


    # We broke this code out of __init__ for clarity.
    def SetConfigs(self, key_to_value, all_layers):
        # the child-class constructor will deal with the configuration values
        # in a more specific way.
        for key,value in key_to_value.items():
            if key != 'name':
                if not key in self.config:
                    raise RuntimeError("Configuration value {0}={1} was not expected in "
                                    "layer of type {2}".format(key, value, self.layer_type))
                self.config[key] = xconfig_utils.ConvertValueToType(key, type(self.config[key]), value)


        self.descriptors = dict()
        self.descriptor_dims = dict()
        # Parse Descriptors and get their dims and their 'final' string form.
        # Put them as 4-tuples (descriptor, string, normalized-string, final-string)
        # in self.descriptors[key]
        for key in self.GetInputDescriptorNames():
            if not key in self.config:
                raise RuntimeError("{0}: object of type {1} needs to override "
                                   "GetInputDescriptorNames()".format(sys.argv[0],
                                                                   str(type(self))))
            descriptor_string = self.config[key]  # input string.
            assert isinstance(descriptor_string, str)
            desc = self.ConvertToDescriptor(descriptor_string, all_layers)
            desc_dim = self.GetDimForDescriptor(desc, all_layers)
            desc_norm_str = desc.str()
            # desc_output_str contains the "final" component names, those that
            # appear in the actual config file (i.e. not names like
            # 'layer.auxiliary_output'); that's how it differs from desc_norm_str.
            # Note: it's possible that the two strings might be the same in
            # many, even most, cases-- it depends whether OutputName(self, auxiliary_output)
            # returns self.Name() + '.' + auxiliary_output when auxiliary_output is not None.
            # That's up to the designer of the layer type.
            desc_output_str = self.GetStringForDescriptor(desc, all_layers)
            self.descriptors[key] = {'string':desc,
                                     'normalized-string':desc_norm_str,
                                     'final-string':desc_output_str,
                                     'dim':desc_dim}
            # the following helps to check the code by parsing it again.
            desc2 = self.ConvertToDescriptor(desc_norm_str, all_layers)
            desc_norm_str2 = desc2.str()
            # if the following ever fails we'll have to do some debugging.
            if desc_norm_str != desc_norm_str2:
                raise RuntimeError("Likely code error: '{0}' != '{1}'".format(
                        desc_norm_str, desc_norm_str2))

    # This function converts 'this' to a string which could be printed to an
    # xconfig file; in xconfig_to_configs.py we actually expand all the lines to
    # strings and write it as xconfig.expanded as a reference (so users can
    # see any defaults).
    def str(self):
        ans = '{0} name={1}'.format(self.layer_type, self.name)
        ans += ' ' + ' '.join([ '{0}={1}'.format(key, self.config[key])
                                for key in sorted(self.config.keys())])
        return ans

    def __str__(self):
        return self.str()


    # This function converts any config variables in self.config which
    # correspond to Descriptors, into a 'normalized form' derived from parsing
    # them as Descriptors, replacing things like [-1] with the actual layer
    # names, and regenerating them as strings.  We stored this when the
    # object was initialized, in self.descriptors; this function just copies them
    # back to the config.
    def NormalizeDescriptors(self):
        for key, desc_str_dict in self.descriptors.items():
            self.config[key] = desc_str_dict['normalized-string']  # desc_norm_str

    # This function, which is a convenience function intended to be called from
    # child classes, converts a string representing a descriptor
    # ('descriptor_string') into an object of type Descriptor, and returns it.
    # It needs 'self' and 'all_layers' (where 'all_layers' is a list of objects
    # of type XconfigLayerBase) so that it can work out a list of the names of
    # other layers, and get dimensions from them.
    def ConvertToDescriptor(self, descriptor_string, all_layers):
        prev_names = xconfig_utils.GetPrevNames(all_layers, self)
        tokens = xconfig_utils.TokenizeDescriptor(descriptor_string, prev_names)
        pos = 0
        (descriptor, pos) = xconfig_utils.ParseNewDescriptor(tokens, pos, prev_names)
        # note: 'pos' should point to the 'end of string' marker
        # that terminates 'tokens'.
        if pos != len(tokens) - 1:
            raise RuntimeError("Parsing Descriptor, saw junk at end: " +
                            ' '.join(tokens[pos:-1]))
        return descriptor

    # Returns the dimension of a Descriptor object.
    # This is a convenience function used in SetConfigs.
    def GetDimForDescriptor(self, descriptor, all_layers):
        layer_to_dim_func = lambda name: xconfig_utils.GetDimFromLayerName(all_layers, self, name)
        return descriptor.Dim(layer_to_dim_func)

    # Returns the 'final' string form of a Descriptor object, as could be used
    # in config files.
    # This is a convenience function provided for use in child classes;
    def GetStringForDescriptor(self, descriptor, all_layers):
        layer_to_string_func = lambda name: xconfig_utils.GetStringFromLayerName(all_layers, self, name)
        return descriptor.ConfigString(layer_to_string_func)

    # Name() returns the name of this layer, e.g. 'affine1'.  It does not
    # necessarily correspond to a component name.
    def Name(self):
        return self.name

    ######  Functions that might be overridden by the child class: #####

    # child classes should override this.
    def SetDefaultConfigs(self):
        raise RuntimeError("Child classes must override SetDefaultConfigs().")

    # this is expected to be called after SetConfigs and before CheckConfigs()
    def SetDerivedConfigs(self):
        pass

    # child classes should override this.
    def CheckConfigs(self):
        pass

    # This function, which may be (but usually will not have to be) overridden
    # by child classes, returns a list of names of the input descriptors
    # expected by this component. Typically this would just return ['input'] as
    # most layers just have one 'input'. However some layers might require more
    # inputs (e.g. cell state of previous LSTM layer in Highway LSTMs).
    # It is used in the function 'NormalizeDescriptors()'.
    # This implementation will work for layer types whose only
    # Descriptor-valued config is 'input'.

    # If a child class adds more inputs, or does not have an input
    # (e.g. the XconfigInputLayer), it should override this function's
    # implementation to something like: `return ['input', 'input2']`
    def GetInputDescriptorNames(self):
        return [ 'input' ]

    # Returns a list of all auxiliary outputs that this
    # layer supports.  These are either 'None' for the regular output, or a
    # string (e.g. 'projection' or 'memory_cell') for any auxiliary outputs that
    # the layer might provide.  Most layer types will not need to override this.
    def AuxiliaryOutputs(self):
        return [ None ]

    # Called with auxiliary_output == None, this returns the component-node name of the
    # principal output of the layer (or if you prefer, the text form of a
    # descriptor that gives you such an output; such as Append(some_node,
    # some_other_node)).
    # The 'auxiliary_output' argument is a text value that is designed for extensions
    # to layers that have additional auxiliary outputs.  For example, to implement
    # a highway LSTM you need the memory-cell of a layer, so you might allow
    # qualifier='memory_cell' for such a layer type, and it would return the
    # component node or a suitable Descriptor: something like 'lstm3.c_t'
    def OutputName(self, auxiliary_output = None):
        raise RuntimeError("Child classes must override OutputName()")

    # The dimension that this layer outputs.  The 'auxiliary_output' parameter is for
    # layer types which support auxiliary outputs.
    def OutputDim(self, auxiliary_output = None):
        raise RuntimeError("Child classes must override OutputDim()")

    # This function returns lines destined for the 'full' config format, as
    # would be read by the C++ programs.
    # Since the program xconfig_to_configs.py writes several config files, this
    # function returns a list of pairs of the form (config_file_basename, line),
    # e.g. something like
    # [ ('init', 'input-node name=input dim=40'),
    #   ('ref', 'input-node name=input dim=40') ]
    # which would be written to config_dir/init.config and config_dir/ref.config.
    def GetFullConfig(self):
        raise RuntimeError("Child classes must override GetFullConfig()")


# This class is for lines like
# 'input name=input dim=40'
# or
# 'input name=ivector dim=100'
# in the config file.
class XconfigInputLayer(XconfigLayerBase):
    def __init__(self, first_token, key_to_value, prev_names = None):
        assert first_token == 'input'
        XconfigLayerBase.__init__(self, first_token, key_to_value, prev_names)


    def SetDefaultConfigs(self):
        self.config = { 'dim':None }

    def CheckConfigs(self):
        if self.config['dim'] is None:
            raise RuntimeError("Dimension of input-layer '{0}' is not set".format(self.name))
        if self.config['dim'] <= 0:
            raise RuntimeError("Dimension of input-layer '{0}' should be positive.".format(self.name))

    def GetInputDescriptorNames(self):
        return []  # there is no 'input' field in self.config.

    def OutputName(self, auxiliary_outputs = None):
        # there are no auxiliary outputs as this layer will just pass the input
        assert auxiliary_outputs is None
        return self.name

    def OutputDim(self, auxiliary_outputs = None):
        # there are no auxiliary outputs as this layer will just pass the input
        assert auxiliary_outputs is None
        return self.config['dim']

    def GetFullConfig(self):
        # unlike other layers the input layers need to be printed in 'init.config'
        # (which initializes the neural network prior to the LDA)
        ans = []
        for config_name in [ 'init', 'ref', 'final' ]:
            ans.append( (config_name,
                         'input-node name={0} dim={1}'.format(self.name,
                                                              self.config['dim'])))
        return ans



# This class is for lines like
# 'output name=output input=Append(input@-1, input@0, input@1, ReplaceIndex(ivector, t, 0))'
# This is for outputs that are not really output "layers" (there is no affine transform or
# nonlinearity), they just directly map to an output-node in nnet3.
class XconfigTrivialOutputLayer(XconfigLayerBase):
    def __init__(self, first_token, key_to_value, prev_names = None):
        assert first_token == 'output'
        XconfigLayerBase.__init__(self, first_token, key_to_value, prev_names)

    def SetDefaultConfigs(self):
        # note: self.config['input'] is a descriptor, '[-1]' means output
        # the most recent layer.
        self.config = { 'input':'[-1]' }

    def CheckConfigs(self):
        pass  # nothing to check; descriptor-parsing can't happen in this function.

    def OutputName(self, auxiliary_outputs = None):
        # there are no auxiliary outputs as this layer will just pass the output
        # of the previous layer
        assert auxiliary_outputs is None
        return self.name

    def OutputDim(self, auxiliary_outputs = None):
        assert auxiliary_outputs is None
        # note: each value of self.descriptors is (descriptor, dim, normalized-string, output-string).
        return self.descriptors['input']['dim']

    def GetFullConfig(self):
        # the input layers need to be printed in 'init.config' (which
        # initializes the neural network prior to the LDA), in 'ref.config',
        # which is a version of the config file used for getting left and right
        # context (it doesn't read anything for the LDA-like transform and/or
        # presoftmax-prior-scale components)
        # In 'full.config' we write everything, this is just for reference,
        # and also for cases where we don't use the LDA-like transform.
        ans = []

        # note: each value of self.descriptors is (descriptor, dim,
        # normalized-string, output-string).
        # by 'output-string' we mean a string that can appear in
        # config-files, i.e. it contains the 'final' names of nodes.
        descriptor_final_str = self.descriptors['input'][3]

        for config_name in ['init', 'ref', 'final' ]:
            ans.append( (config_name,
                         'output-node name={0} input={1}'.format(
                        self.name, descriptor_final_str)))
        return ans


# This class is for lines like
#  'output-layer name=output dim=4257 input=Append(input@-1, input@0, input@1, ReplaceIndex(ivector, t, 0))'
# By default this includes a log-softmax component.  The parameters are initialized to zero, as
# this is best for output layers.
# Parameters of the class, and their defaults:
#   input='[-1]'             [Descriptor giving the input of the layer.]
#   dim=None                   [Output dimension of layer, will normally equal the number of pdfs.]
#   include-log-softmax=true [setting it to false will omit the log-softmax component- useful for chain
#                              models.]
#   objective-type=linear    [the only other choice currently is 'quadratic', for use in regression
#                             problems]

#   learning-rate-factor=1.0 [Learning rate factor for the final affine component, multiplies the
#                              standard learning rate. normally you'll leave this as-is, but for
#                              xent regularization output layers for chain models you'll want to set
#                              learning-rate-factor=(0.5/xent_regularize), normally
#                              learning-rate-factor=5.0 since xent_regularize is normally 0.1.
#   presoftmax-scale-file=None [If set, a filename for a vector that will be used to scale the output
#                              of the affine component before the log-softmax (if
#                              include-log-softmax=true), or before the output (if not).  This is
#                              helpful to avoid instability in training due to some classes having
#                              much more data than others.  The way we normally create this vector
#                              is to take the priors of the classes to the power -0.25 and rescale
#                              them so the average is 1.0.  This factor -0.25 is referred to
#                              as presoftmax_prior_scale_power in scripts.]
#                              In the scripts this would normally be set to config_dir/presoftmax_prior_scale.vec
class XconfigOutputLayer(XconfigLayerBase):
    def __init__(self, first_token, key_to_value, prev_names = None):
        assert first_token == 'output-layer'
        XconfigLayerBase.__init__(self, first_token, key_to_value, prev_names)

    def SetDefaultConfigs(self):
        # note: self.config['input'] is a descriptor, '[-1]' means output
        # the most recent layer.
        self.config = { 'input':'[-1]',
                        'dim':None,
                        'include-log-softmax':True, # this would be false for chain models
                        'objective-type':'linear', # see Nnet::ProcessOutputNodeConfigLine in nnet-nnet.cc for other options
                        'learning-rate-factor':1.0,
                        'presoftmax-scale-file':None # used in DNN (not RNN) training using frame-level objfns
                        }

    def CheckConfigs(self):
        if self.config['dim'] is None:
            raise RuntimeError("In output-layer, dim has to be set.")
        elif self.config['dim'] <= 0:
            raise RuntimeError("In output-layer, dim has invalid value {0}".format(self.config['dim']))
        if self.config['objective-type'] != 'linear' and self.config['objective_type'] != 'quadratic':
            raise RuntimeError("In output-layer, objective-type has invalid value {0}".format(
                    self.config['objective-type']))
        if self.config['learning-rate-factor'] <= 0.0:
            raise RuntimeError("In output-layer, learning-rate-factor has invalid value {0}".format(
                    self.config['learning-rate-factor']))


    # you cannot access the output of this layer from other layers... see
    # comment in OutputName for the reason why.
    def AuxiliaryOutputs(self):
        return []

    def OutputName(self, auxiliary_outputs = None):
        # Note: nodes of type output-node in nnet3 may not be accessed in Descriptors,
        # so calling this with auxiliary_outputs=None doesn't make sense.  But it might make
        # sense to make the output of the softmax layer and/or the output of the
        # affine layer available as inputs to other layers, in some circumstances.
        # we'll implement that when it's needed.
        raise RuntimeError("Outputs of output-layer may not be used by other layers")

    def OutputDim(self, qualifier = None):
        # see comment in OutputName().
        raise RuntimeError("Outputs of output-layer may not be used by other layers")

    def GetFullConfig(self):
        ans = []

        # note: each value of self.descriptors is (descriptor, dim,
        # normalized-string, output-string).
        # by 'descriptor_final_string' we mean a string that can appear in
        # config-files, i.e. it contains the 'final' names of nodes.
        descriptor_final_string = self.descriptors['input']['final-string']
        input_dim = self.descriptors['input']['dim']
        output_dim = self.config['dim']
        objective_type = self.config['objective-type']
        learning_rate_factor = self.config['learning-rate-factor']
        include_log_softmax = self.config['include-log-softmax']
        presoftmax_scale_file = self.config['presoftmax-scale-file']


        # note: ref.config is used only for getting the left-context and right-context
        # of the network; all.config is where we put the actual network definition.
        for config_name in [ 'ref', 'final' ]:
            # First the affine node.
            line = ('component name={0}.affine type=NaturalGradientAffineComponent input-dim={1} '
                    'output-dim={2} param-stddev=0 bias-stddev=0 '.format(
                    self.name, input_dim, output_dim) +
                    ('learning-rate-factor={0} '.format(learning_rate_factor)
                     if learning_rate_factor != 1.0 else ''))
            ans.append((config_name, line))
            line = ('component-node name={0}.affine component={0}.affine input={1}'.format(
                    self.name, descriptor_final_string))
            ans.append((config_name, line))
            cur_node = '{0}.affine'.format(self.name)
            if presoftmax_scale_file is not None and config_name == 'final':
                # don't use the presoftmax-scale in 'ref.config' since that file won't exist at the
                # time we evaluate it.  (ref.config is used to find the left/right context).
                line = ('component name={0}.fixed-scale type=FixedScaleComponent scales={1}'.format(
                        self.name, presoftmax_scale_file))
                ans.append((config_name, line))
                line = ('component-node name={0}.fixed-scale component={0}.fixed-scale input={1}'.format(
                        self.name, cur_node))
                ans.append((config_name, line))
                cur_node = '{0}.fixed-scale'.format(self.name)
            if include_log_softmax:
                line = ('component name={0}.log-softmax type=LogSoftmaxComponent dim={1}'.format(
                        self.name, output_dim))
                ans.append((config_name, line))
                line = ('component-node name={0}.log-softmax component={0}.log-softmax input={1}'.format(
                        self.name, cur_node))
                ans.append((config_name, line))
                cur_node = '{0}.log-softmax'.format(self.name)
            line = ('output-node name={0} input={0}.log-softmax'.format(self.name, cur_node))
            ans.append((config_name, line))
        return ans


# This class is for parsing lines like
#  'relu-renorm-layer name=layer1 dim=1024 input=Append(-3,0,3)'
# or:
#  'sigmoid-layer name=layer1 dim=1024 input=Append(-3,0,3)'
# which specify addition of an affine component and a sequence of non-linearities.
# Here, the name of the layer itself dictates the sequence of nonlinearities
# that are applied after the affine component; the name should contain some
# combination of 'relu', 'renorm', 'sigmoid' and 'tanh',
# and these nonlinearities will be added along with the affine component.
#
# The dimension specified is the output dim; the input dim is worked out from the input descriptor.
# This class supports only nonlinearity types that do not change the dimension; we can create
# another layer type to enable the use p-norm and similar dimension-reducing nonlinearities.
#
# See other configuration values below.
#
# Parameters of the class, and their defaults:
#   input='[-1]'             [Descriptor giving the input of the layer.]
#   dim=None                   [Output dimension of layer, e.g. 1024]
#   self-repair-scale=1.0e-05  [Affects relu, sigmoid and tanh layers.]
#
class XconfigBasicLayer(XconfigLayerBase):
    def __init__(self, first_token, key_to_value, prev_names = None):
        # Here we just list some likely combinations.. you can just add any
        # combinations you want to use, to this list.
        assert first_token in [ 'relu-layer', 'relu-renorm-layer', 'sigmoid-layer',
                                'tanh-layer' ]
        XconfigLayerBase.__init__(self, first_token, key_to_value, prev_names)

    def SetDefaultConfigs(self):
        # note: self.config['input'] is a descriptor, '[-1]' means output
        # the most recent layer.
        self.config = { 'input':'[-1]',
                        'dim':None,
                        'self-repair-scale':1.0e-05,
                        'norm-target-rms':1.0}

    def CheckConfigs(self):
        if self.config['dim'] is None:
            raise RuntimeError("In {0}, dim has to be set.")
        elif self.config['dim'] < 0:
            raise RuntimeError("In {0}, dim has invalid value {1}".format(self.layer_type,
                                                                          self.config['dim']))
        if self.config['self-repair-scale'] < 0.0 or self.config['self-repair-scale'] > 1.0:
            raise RuntimeError("In {0}, self-repair-scale has invalid value {0}".format(
                    self.layer_type, self.config['self-repair-scale']))
        if self.config['norm-target-rms'] < 0.0:
            raise RuntimeError("In {0}, norm-target-rms has invalid value {0}".format(
                    self.layer_type, self.config['norm-target-rms']))

    def OutputName(self, auxiliary_output=None):
        # at a later stage we might want to expose even the pre-nonlinearity
        # vectors
        assert auxiliary_output == None

        split_layer_name = self.layer_type.split('-')
        assert split_layer_name[-1] == 'layer'
        last_nonlinearity = split_layer_name[-2]
        # return something like: layer3.renorm
        return '{0}.{1}'.format(self.name, last_nonlinearity)

    def OutputDim(self, qualifier = None):
        return self.config['dim']

    def GetFullConfig(self):

        ans = []

        split_layer_name = self.layer_type.split('-')
        assert split_layer_name[-1] == 'layer'
        nonlinearities = split_layer_name[:-1]

        # by 'descriptor_final_string' we mean a string that can appear in
        # config-files, i.e. it contains the 'final' names of nodes.
        descriptor_final_string = self.descriptors['input']['final-string']
        input_dim = self.descriptors['input'][1]
        output_dim = self.config['dim']
        self_repair_scale = self.config['self-repair-scale']

        for config_name in [ 'ref', 'final' ]:
            # First the affine node.
            line = ('component name={0}.affine type=NaturalGradientAffineComponent input-dim={1} '
                    'output-dim={2} '.format(self.name, input_dim, output_dim))
            ans.append((config_name, line))
            line = ('component-node name={0}.affine component={0}.affine input={1}'.format(
                    self.name, descriptor_final_string))
            ans.append((config_name, line))
            cur_node = '{0}.affine'.format(self.name)

            for nonlinearity in nonlinearities:
                if nonlinearity == 'relu':
                    line = ('component name={0}.{1} type=RectifiedLinearComponent dim={2} '
                            'self-repair-scale={3}'.format(self.name, nonlinearity, output_dim,
                                                           self_repair_scale))
                elif nonlinearity == 'sigmoid':
                    line = ('component name={0}.{1} type=SigmoidComponent dim={2} '
                            'self-repair-scale={3}'.format(self.name, nonlinearity, output_dim,
                                                           self_repair_scale))
                elif nonlinearity == 'tanh':
                    line = ('component name={0}.{1} type=TanhComponent dim={2} '
                            'self-repair-scale={3}'.format(self.name, nonlinearity, output_dim,
                                                           self_repair_scale))
                elif nonlinearity == 'renorm':
                    line = ('component name={0}.{1} type=NormalizeComponent dim={2} '.format(
                            self.name, nonlinearity, output_dim))
                else:
                    raise RuntimeError("Unknown nonlinearity type: {0}".format(nonlinearity))
                ans.append((config_name, line))
                line = 'component-node name={0}.{1} component={0}.{1} input={2}'.format(
                    self.name, nonlinearity, cur_node)
                ans.append((config_name, line))
                cur_node = '{0}.{1}'.format(self.name, nonlinearity)
        return ans


# This class is for lines like
#  'fixed-affine-layer name=lda input=Append(-2,-1,0,1,2,ReplaceIndex(ivector, t, 0)) affine-transform-file=foo/bar/lda.mat'
#
# The output dimension of the layer may be specified via 'dim=xxx', but if not specified,
# the dimension defaults to the same as the input.  Note: we don't attempt to read that
# file at the time the config is created, because in the recipes, that file is created
# after the config files.
#
# See other configuration values below.
#
# Parameters of the class, and their defaults:
#   input='[-1]'             [Descriptor giving the input of the layer.]
#   dim=None                   [Output dimension of layer; defaults to the same as the input dim.]
#   affine-transform-file='' [Must be specified.]
#
class XconfigFixedAffineLayer(XconfigLayerBase):
    def __init__(self, first_token, key_to_value, prev_names = None):
        assert first_token == 'fixed-affine-layer'
        XconfigLayerBase.__init__(self, first_token, key_to_value, prev_names)

    def SetDefaultConfigs(self):
        # note: self.config['input'] is a descriptor, '[-1]' means output
        # the most recent layer.
        self.config = { 'input':'[-1]', 'dim':None, 'affine-transform-file':None }

    def CheckConfigs(self):
        if self.config['affine-transform-file'] is None:
            raise RuntimeError("In fixed-affine-layer, affine-transform-file must be set.")

    def OutputName(self, auxiliary_output = None):
        # Fixed affine layer computes only one vector, there are no intermediate
        # vectors.
        assert auxiliary_output == None
        return self.name

    def OutputDim(self, qualifier = None):
        output_dim = self.config['dim']
        # If not set, the output-dim defaults to the input-dim.
        if output_dim is None:
            output_dim = self.descriptors['input']['dim']
        return output_dim

    def GetFullConfig(self):
        ans = []

        # note: each value of self.descriptors is (descriptor, dim,
        # normalized-string, output-string).
        # by 'descriptor_final_string' we mean a string that can appear in
        # config-files, i.e. it contains the 'final' names of nodes.
        descriptor_final_string = self.descriptors['input']['final-string']
        input_dim = self.descriptors['input']['dim']
        output_dim = self.OutputDim()
        transform_file = self.config['affine-transform-file']


        # to init.config we write an output-node with the name 'output' and
        # with a Descriptor equal to the descriptor that's the input to this
        # layer.  This will be used to accumulate stats to learn the LDA transform.
        line = 'output-node name=output input={0}'.format(descriptor_final_string)
        ans.append(('init', line))

        # write the 'real' component to final.config
        line = 'component name={0} type=FixedAffineComponent matrix={1}'.format(
            self.name, transform_file)
        ans.append(('final', line))
        # write a random version of the component, with the same dims, to ref.config
        line = 'component name={0} type=FixedAffineComponent input-dim={1} output-dim={2}'.format(
            self.name, input_dim, output_dim)
        ans.append(('ref', line))
        # the component-node gets written to final.config and ref.config.
        line = 'component-node name={0} component={0} input={1}'.format(
            self.name, descriptor_final_string)
        ans.append(('final', line))
        ans.append(('ref', line))
        return ans


# This class is for lines like
#   'lstmp-layer name=lstm1 input=[-1] delay=-3'
# It generates an LSTM sub-graph with output projections. It can also generate
# outputs without projection, but you could use the XconfigLstmLayer for this
# simple LSTM.
# The output dimension of the layer may be specified via 'cell-dim=xxx', but if not specified,
# the dimension defaults to the same as the input.
# See other configuration values below.
#
# Parameters of the class, and their defaults:
#   input='[-1]'             [Descriptor giving the input of the layer.]
#   cell-dim=None            [Dimension of the cell; defaults to the same as the input dim]
#   recurrent_projection_dim [Dimension of the projection used in recurrent connections]
#   non_recurrent_projection_dim        [Dimension of the projection in non-recurrent connections]
#   delay=-1                 [Delay in the recurrent connections of the LSTM ]
#   clipping-threshold=30    [nnet3 LSTMs use a gradient clipping component at the recurrent connections. This is the threshold used to decide if clipping has to be activated ]
#   norm-based-clipping=True [specifies if the gradient clipping has to activated based on total norm or based on per-element magnitude]
#   self_repair_scale_nonlinearity=1e-5      [It is a constant scaling the self-repair vector computed in derived classes of NonlinearComponent]
#                                       i.e.,  SigmoidComponent, TanhComponent and RectifiedLinearComponent ]
#   self_repair_scale_clipgradient=1.0  [It is a constant scaling the self-repair vector computed in ClipGradientComponent ]
#   ng-per-element-scale-options=None   [Additional options used for the diagonal matrices in the LSTM ]
#   ng-affine-options=None              [Additional options used for the full matrices in the LSTM, can be used to do things like set biases to initialize to 1]
class XconfigLstmpLayer(XconfigLayerBase):
    def __init__(self, first_token, key_to_value, prev_names = None):
        assert first_token == "lstm"
        XconfigLayerBase.__init__(self, first_token, key_to_value, prev_names)

    def SetDefaultConfigs(self):
        self.config = {'input':'[-1]',
                        'cell-dim':None, # this is a compulsary argument
                        'recurrent-projection-dim':None,
                        'non-recurrent-projection-dim':None,
                        'clipping-threshold':30.0,
                        'norm-based-clipping':True,
                        'delay':-1,
                        'ng-per-element-scale-options':None,
                        'ng-affine-options':None,
                        'self-repair-scale-nonlinearity':0.00001,
                        'self-repair-scale-clipgradient':1.0 }

    def SetDerivedConfigs(self):
        assert self.config['cell-dim'] is not None
        for key in ['recurrent-projection-dim', 'non-recurrent-projection-dim']:
            if self.config[key] is None:
                self.config[key] = self.config['cell-dim'] / 2

    def CheckConfigs(self):
        for key in ['cell-dim', 'recurrent-projection-dim', 'non-recurrent-projection-dim']:
            if self.config[key] is None:
                raise RuntimeError("In {0} of type {1}, {2} has to be set.".format(self.name, self.layer_type, key))
            if self.config[key] < 0:
                raise RuntimeError("In {0} of type {1}, {2} has invalid value {3}.".format(self.name, self.layer_type,
                                     key, self.config[key]))

        for key in ['self-repair-scale-nonlinearity', 'self-repair-scale-clipgradient']:
            if self.config[key] < 0.0 or self.config[key] > 1.0:
                raise RuntimeError("In {0}, {1} has invalid value {2}.".format(self.layer_type,
                                                                               key,
                                                                               self.config[key]))
    def AuxiliaryOutputs(self):
        return ['c_t']

    def OutputName(self, auxiliary_output = None):
        node_name = 'rp_t'
        if auxiliary_output is not None:
            if auxiliary_output in self.AuxiliaryOutputs():
                node_name = auxiliary_output
            else:
                raise Exception("In {0} of type {1}, unknown auxiliary output name {1}".format(self.layer_type, auxiliary_output))

        return '{0}.{1}'.format(self.name, node_name)

    def OutputDim(self, auxiliary_outputs = None):

        if auxiliary_output is not None:
            if auxiliary_output in self.AuxiliaryOutputs():
                if node_name == 'c_t':
                    return self.config['cell-dim']
                # add code for other auxiliary_outputs here when we decide to expose them
            else:
                raise Exception("In {0} of type {1}, unknown auxiliary output name {1}".format(self.layer_type, auxiliary_output))

        return self.config['recurrent-projection-dim'] + self.config['non-recurrent-projection-dim']

    def GetFullConfig(self):
        ans = []
        config_lines = self.GenerateLstmConfig()

        for line in config_lines:
            for config_name in ['ref', 'final']:
                # we do not support user specified matrices in LSTM initialization
                # so 'ref' and 'final' configs are the same.
                ans.append((config_name, line))
        return ans

    # convenience function to generate the LSTM config
    def GenerateLstmConfig(self):
        # assign some variables to reduce verbosity
        name = self.name
        repair_nonlin = self.config['self-repair-scale-nonlinearity']
        repair_clipgrad = self.config['self-repair-scale-clipgradient']
        repair_nonlin_str = "self-repair-scale={0:.10f}".format(repair_nonlin) if repair_nonlin is not None else ''
        repair_clipgrad_str = "self-repair-scale={0:.2f}".format(repair_clipgrad) if repair_clipgrad is not None else ''
        clipgrad_str = "clipping-threshold={0} norm-based-clipping={1} {2}".format(self.config['clipping_threshold'], self.config['norm_based_clipping'], repair_clipgrad_str)
        affine_str = self.config['ng-affine-options']
        # Natural gradient per element scale parameters
        # TODO: decide if we want to keep exposing these options
        if re.search('param-mean', ng_per_element_scale_options) is None and \
           re.search('param-stddev', ng_per_element_scale_options) is None:
           ng_per_element_scale_options += " param-mean=0.0 param-stddev=1.0 "
        pes_str = ng_per_element_scale_options


        # in the below code we will just call descriptor_strings as descriptors for conciseness
        input_dim = self.descriptors['input']['dim']
        input_descriptor = self.descriptors['input']['final-string']
        cell_dim = self.config['cell-dim']
        rec_proj_dim = self.config['recurrent-projection-dim']
        nonrec_proj_dim = self.config['non-recurrent-projection-dim']
        delay = self.config['delay']


        configs = []

        # the equations implemented here are
        # TODO: write these
        # naming convention
        # <layer-name>.W_<outputname>.<input_name> e.g. Lstm1.W_i.xr for matrix providing output to gate i and operating on an appended vector [x,r]
        configs.append("# Input gate control : W_i* matrices")
        configs.append("component name={0}.W_i.xr type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, input_dim + rec_proj_dim, cell_dim, affine_str))
        configs.append("# note : the cell outputs pass through a diagonal matrix")
        configs.append("component name={0}.w_i.c type=NaturalGradientPerElementScaleComponent  dim={1} {2}".format(name, cell_dim, pes_str))

        configs.append("# Forget gate control : W_f* matrices")
        configs.append("component name={0}.W_f.xr type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, input_dim + rec_proj_dim, cell_dim, affine_str))
        configs.append("# note : the cell outputs pass through a diagonal matrix")
        configs.append("component name={0}.w_f.c type=NaturalGradientPerElementScaleComponent  dim={1} {2}".format(name, cell_dim, pes_str))

        configs.append("#  Output gate control : W_o* matrices")
        configs.append("component name={0}.W_o.xr type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, input_dim + rec_proj_dim, cell_dim, affine_str))
        configs.append("# note : the cell outputs pass through a diagonal matrix")
        configs.append("component name={0}.w_o.c type=NaturalGradientPerElementScaleComponent  dim={1} {2}".format(name, cell_dim, pes_str))

        configs.append("# Cell input matrices : W_c* matrices")
        configs.append("component name={0}_W_c-xr type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, input_dim + rec_proj_dim, cell_dim, affine_str))


        configs.append("# Defining the non-linearities")
        configs.append("component name={0}.i type=SigmoidComponent dim={1} {2}".format(name, cell_dim, repair_nonlin_str))
        configs.append("component name={0}.f type=SigmoidComponent dim={1} {2}".format(name, cell_dim, repair_nonlin_str))
        configs.append("component name={0}.o type=SigmoidComponent dim={1} {2}".format(name, cell_dim, repair_nonlin_str))
        configs.append("component name={0}.g type=TanhComponent dim={1} {2}".format(name, cell_dim, repair_nonlin_str))
        configs.append("component name={0}.h type=TanhComponent dim={1} {2}".format(name, cell_dim, repair_nonlin_str))

        configs.append("# Defining the components for other cell computations")
        configs.append("component name={0}.c1 type=ElementwiseProductComponent input-dim={1} output-dim={2}".format(name, 2 * cell_dim, cell_dim))
        configs.append("component name={0}.c2 type=ElementwiseProductComponent input-dim={1} output-dim={2}".format(name, 2 * cell_dim, cell_dim))
        configs.append("component name={0}.m type=ElementwiseProductComponent input-dim={1} output-dim={2}".format(name, 2 * cell_dim, cell_dim))
        configs.append("component name={0}.c type=ClipGradientComponent dim={1} {2}".format(name, cell_dim, clipgrad_str))

        # c1_t and c2_t defined below
        configs.append("component-node name={0}.c_t component={0}.c input=Sum({0}.c1_t, {0}.c2_t)".format(name))
        delayed_c_t_descriptor = "IfDefined(Offset({0}.c_t, {1}))".format(name, delay)

        rec_connection = '{0}.rp_t'.format(name)
        configs.append("# i_t")
        configs.append("component-node name={0}.i1_t component={0}.W_i.xr input=Append({1}, IfDefined(Offset({2}, {3})))".format(name, input_descriptor, recurrent_connection, delay))
        configs.append("component-node name={0}.i2_t component={0}.w_i.c  input={1}".format(name, delayed_c_t_descriptor))
        configs.append("component-node name={0}.i_t component={0}.i input=Sum({0}.i1_t, {0}.i2_t)".format(name))

        configs.append("# f_t")
        configs.append("component-node name={0}.f1_t component={0}.W_f.xr input=Append({1}, IfDefined(Offset({2}, {3})))".format(name, input_descriptor, recurrent_connection, delay))
        configs.append("component-node name={0}.f2_t component={0}.w_f.c  input={1}".format(name, delayed_c_t_descriptor))
        configs.append("component-node name={0}.f_t component={0}.f input=Sum({0}.f1_t, {0}.f2_t)".format(name))

        configs.append("# o_t")
        configs.append("component-node name={0}.o1_t component={0}.W_o.xr input=Append({1}, IfDefined(Offset({2}, {3})))".format(name, input_descriptor, recurrent_connection, delay))
        configs.append("component-node name={0}.o2_t component={0}.w_o.c input={0}.c_t".format(name))
        configs.append("component-node name={0}.o_t component={0}_o input=Sum({0}.o1_t, {0}.o2_t)".format(name))

        configs.append("# h_t")
        configs.append("component-node name={0}.h_t component={0}.h input={0}.c_t".format(name))

        configs.append("# g_t")
        configs.append("component-node name={0}.g1_t component={0}.W_c.xr input=Append({1}, IfDefined(Offset({2}, {3})))".format(name, input_descriptor, recurrent_connection, delay))
        configs.append("component-node name={0}.g_t component={0}.g input={0}.g1_t".format(name))

        configs.append("# parts of c_t")
        configs.append("component-node name={0}.c1_t component={0}.c1  input=Append({0}.f_t, {1})".format(name, delayed_c_t_descriptor))
        configs.append("component-node name={0}.c2_t component={0}.c2 input=Append({0}.i_t, {0}.g_t)".format(name))

        configs.append("# m_t")
        configs.append("component-node name={0}.m_t component={0}.m input=Append({0}.o_t, {0}.h_t)".format(name))

        # add the recurrent connections
        configs.append("# projection matrices : Wrm and Wpm")
        configs.append("component name={0}.W_rp.m type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, cell_dim, recurrent_projection_dim + non_recurrent_projection_dim, affine_str))
        configs.append("component name={0}.r type=ClipGradientComponent dim={1} {2}".format(name, recurrent_projection_dim, clipgrad_str))

        configs.append("# r_t and p_t : rp_t will be the output")
        configs.append("component-node name={0}.rp_t component={0}.W_rp.m input={0}.m_t".format(name))
        configs.append("dim-range-node name={0}.r_t_preclip input-node={0}.rp_t dim-offset=0 dim={1}".format(name, recurrent_projection_dim))
        configs.append("component-node name={0}.r_t component={0}.r input={0}.r_t_preclip".format(name))

        return configs


# This class is for lines like
#   'lstm-layer name=lstm1 input=[-1] delay=-3'
# It generates an LSTM sub-graph without output projections.
# The output dimension of the layer may be specified via 'cell-dim=xxx', but if not specified,
# the dimension defaults to the same as the input.
# See other configuration values below.
#
# Parameters of the class, and their defaults:
#   input='[-1]'             [Descriptor giving the input of the layer.]
#   cell-dim=None            [Dimension of the cell; defaults to the same as the input dim]
#   delay=-1                 [Delay in the recurrent connections of the LSTM ]
#   clipping-threshold=30    [nnet3 LSTMs use a gradient clipping component at the recurrent connections. This is the threshold used to decide if clipping has to be activated ]
#   norm-based-clipping=True [specifies if the gradient clipping has to activated based on total norm or based on per-element magnitude]
#   self_repair_scale_nonlinearity=1e-5      [It is a constant scaling the self-repair vector computed in derived classes of NonlinearComponent]
#                                       i.e.,  SigmoidComponent, TanhComponent and RectifiedLinearComponent ]
#   self_repair_scale_clipgradient=1.0  [It is a constant scaling the self-repair vector computed in ClipGradientComponent ]
#   ng-per-element-scale-options=None   [Additional options used for the diagonal matrices in the LSTM ]
#   ng-affine-options=None              [Additional options used for the full matrices in the LSTM, can be used to do things like set biases to initialize to 1]
class XconfigLstmLayer(XconfigLayerBase):
    def __init__(self, first_token, key_to_value, prev_names = None):
        assert first_token == "lstm"
        XconfigLayerBase.__init__(self, first_token, key_to_value, prev_names)

    def SetDefaultConfigs(self):
        self.config = {'input':'[-1]',
                        'cell-dim':None, # this is a compulsary argument
                        'clipping-threshold':30.0,
                        'norm-based-clipping':True,
                        'delay':-1,
                        'ng-per-element-scale-options':None,
                        'ng-affine-options':None,
                        'self-repair-scale-nonlinearity':0.00001,
                        'self-repair-scale-clipgradient':1.0 }

    def CheckConfigs(self):
        key = 'cell-dim'
        if self.config[key] is None:
            raise RuntimeError("In {0} of type {1}, {2} has to be set.".format(self.name, self.layer_type, key))
        if self.config[key] <= 0:
            raise RuntimeError("In {0} of type {1}, {2} has invalid value {3}.".format(self.name, self.layer_type,
                               key, self.config[key]))

        for key in ['self-repair-scale-nonlinearity', 'self-repair-scale-clipgradient']:
            if self.config[key] < 0.0 or self.config[key] > 1.0:
                raise RuntimeError("In {0}, {1} has invalid value {2}.".format(self.layer_type,
                                                                               key,
                                                                               self.config[key]))
    def AuxiliaryOutputs(self):
        return ['c_t']

    def OutputName(self, auxiliary_output = None):
        node_name = 'm_t'
        if auxiliary_output is not None:
            if auxiliary_output in self.AuxiliaryOutputs():
                node_name = auxiliary_output
            else:
                raise Exception("In {0} of type {1}, unknown auxiliary output name {1}".format(self.layer_type, auxiliary_output))

        return '{0}.{1}'.format(self.name, node_name)

    def OutputDim(self, auxiliary_output = None):

        if auxiliary_output is not None:
            if auxiliary_output in self.AuxiliaryOutputs():
                if node_name == 'c_t':
                    return self.config['cell-dim']
                # add code for other auxiliary_outputs here when we decide to expose them
            else:
                raise Exception("In {0} of type {1}, unknown auxiliary output name {1}".format(self.layer_type, auxiliary_output))

        return self.config['cell-dim']

    def GetFullConfig(self):
        ans = []
        config_lines = self.GenerateLstmConfig()

        for line in config_lines:
            for config_name in ['ref', 'final']:
                # we do not support user specified matrices in LSTM initialization
                # so 'ref' and 'final' configs are the same.
                ans.append((config_name, line))
        return ans

    # convenience function to generate the LSTM config
    def GenerateLstmConfig(self):
        # assign some variables to reduce verbosity
        name = self.name
        repair_nonlin = self.config['self-repair-scale-nonlinearity']
        repair_clipgrad = self.config['self-repair-scale-clipgradient']
        repair_nonlin_str = "self-repair-scale={0:.10f}".format(repair_nonlin) if repair_nonlin is not None else ''
        repair_clipgrad_str = "self-repair-scale={0:.2f}".format(repair_clipgrad) if repair_clipgrad is not None else ''
        clipgrad_str = "clipping-threshold={0} norm-based-clipping={1} {2}".format(self.config['clipping_threshold'], self.config['norm_based_clipping'], repair_clipgrad_str)
        affine_str = self.config['ng-affine-options']
        # Natural gradient per element scale parameters
        # TODO: decide if we want to keep exposing these options
        if re.search('param-mean', ng_per_element_scale_options) is None and \
           re.search('param-stddev', ng_per_element_scale_options) is None:
           ng_per_element_scale_options += " param-mean=0.0 param-stddev=1.0 "
        pes_str = ng_per_element_scale_options


        # in the below code we will just call descriptor_strings as descriptors for conciseness
        input_dim = self.descriptors['input']['dim']
        input_descriptor = self.descriptors['input']['final-string']
        cell_dim = self.config['cell-dim']
        delay = self.config['delay']


        configs = []

        # the equations implemented here are
        # TODO: write these
        # naming convention
        # <layer-name>.W_<outputname>.<input_name> e.g. Lstm1.W_i.xr for matrix providing output to gate i and operating on an appended vector [x,r]
        configs.append("# Input gate control : W_i* matrices")
        configs.append("component name={0}.W_i.xr type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, input_dim + cell_dim, cell_dim, affine_str))
        configs.append("# note : the cell outputs pass through a diagonal matrix")
        configs.append("component name={0}.w_i.c type=NaturalGradientPerElementScaleComponent  dim={1} {2}".format(name, cell_dim, pes_str))

        configs.append("# Forget gate control : W_f* matrices")
        configs.append("component name={0}.W_f.xr type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, input_dim + cell_dim, cell_dim, affine_str))
        configs.append("# note : the cell outputs pass through a diagonal matrix")
        configs.append("component name={0}.w_f.c type=NaturalGradientPerElementScaleComponent  dim={1} {2}".format(name, cell_dim, pes_str))

        configs.append("#  Output gate control : W_o* matrices")
        configs.append("component name={0}.W_o.xr type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, input_dim + cell_dim, cell_dim, affine_str))
        configs.append("# note : the cell outputs pass through a diagonal matrix")
        configs.append("component name={0}.w_o.c type=NaturalGradientPerElementScaleComponent  dim={1} {2}".format(name, cell_dim, pes_str))

        configs.append("# Cell input matrices : W_c* matrices")
        configs.append("component name={0}_W_c-xr type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, input_dim + cell_dim, cell_dim, affine_str))


        configs.append("# Defining the non-linearities")
        configs.append("component name={0}.i type=SigmoidComponent dim={1} {2}".format(name, cell_dim, repair_nonlin_str))
        configs.append("component name={0}.f type=SigmoidComponent dim={1} {2}".format(name, cell_dim, repair_nonlin_str))
        configs.append("component name={0}.o type=SigmoidComponent dim={1} {2}".format(name, cell_dim, repair_nonlin_str))
        configs.append("component name={0}.g type=TanhComponent dim={1} {2}".format(name, cell_dim, repair_nonlin_str))
        configs.append("component name={0}.h type=TanhComponent dim={1} {2}".format(name, cell_dim, repair_nonlin_str))

        configs.append("# Defining the components for other cell computations")
        configs.append("component name={0}.c1 type=ElementwiseProductComponent input-dim={1} output-dim={2}".format(name, 2 * cell_dim, cell_dim))
        configs.append("component name={0}.c2 type=ElementwiseProductComponent input-dim={1} output-dim={2}".format(name, 2 * cell_dim, cell_dim))
        configs.append("component name={0}.m type=ElementwiseProductComponent input-dim={1} output-dim={2}".format(name, 2 * cell_dim, cell_dim))
        configs.append("component name={0}.c type=ClipGradientComponent dim={1} {2}".format(name, cell_dim, clipgrad_str))

        # c1_t and c2_t defined below
        configs.append("component-node name={0}.c_t component={0}.c input=Sum({0}.c1_t, {0}.c2_t)".format(name))
        delayed_c_t_descriptor = "IfDefined(Offset({0}.c_t, {1}))".format(name, delay)

        configs.append("# i_t")
        configs.append("component-node name={0}.i1_t component={0}.W_i.xr input=Append({1}, IfDefined(Offset({0}.r_t, {2})))".format(name, input_descriptor, delay))
        configs.append("component-node name={0}.i2_t component={0}.w_i.c  input={1}".format(name, delayed_c_t_descriptor))
        configs.append("component-node name={0}.i_t component={0}.i input=Sum({0}.i1_t, {0}.i2_t)".format(name))

        configs.append("# f_t")
        configs.append("component-node name={0}.f1_t component={0}.W_f.xr input=Append({1}, IfDefined(Offset({0}.r_t, {2})))".format(name, input_descriptor, delay))
        configs.append("component-node name={0}.f2_t component={0}.w_f.c  input={1}".format(name, delayed_c_t_descriptor))
        configs.append("component-node name={0}.f_t component={0}.f input=Sum({0}.f1_t, {0}.f2_t)".format(name))

        configs.append("# o_t")
        configs.append("component-node name={0}.o1_t component={0}.W_o.xr input=Append({1}, IfDefined(Offset({0}.r_t, {2})))".format(name, input_descriptor, delay))
        configs.append("component-node name={0}.o2_t component={0}.w_o.c input={0}.c_t".format(name))
        configs.append("component-node name={0}.o_t component={0}_o input=Sum({0}.o1_t, {0}.o2_t)".format(name))

        configs.append("# h_t")
        configs.append("component-node name={0}.h_t component={0}.h input={0}.c_t".format(name))

        configs.append("# g_t")
        configs.append("component-node name={0}.g1_t component={0}.W_c.xr input=Append({1}, IfDefined(Offset({0}.r_t, {2})))".format(name, input_descriptor, delay))
        configs.append("component-node name={0}.g_t component={0}.g input={0}.g1_t".format(name))

        configs.append("# parts of c_t")
        configs.append("component-node name={0}.c1_t component={0}.c1  input=Append({0}.f_t, {1})".format(name, delayed_c_t_descriptor))
        configs.append("component-node name={0}.c2_t component={0}.c2 input=Append({0}.i_t, {0}.g_t)".format(name))

        configs.append("# m_t")
        configs.append("component-node name={0}.m_t component={0}.m input=Append({0}.o_t, {0}.h_t)".format(name))

        # add the recurrent connections
        configs.append("component name={0}.r type=ClipGradientComponent dim={1} {2}".format(name, cell_dim, clipgrad_str))
        configs.append("component-node name={0}.r_t component={0}.r input={0}.m_t".format(name))

        return configs


# Converts a line as parsed by ParseConfigLine() into a first
# token e.g. 'input-layer' and a key->value map, into
# an objet inherited from XconfigLayerBase.
# 'prev_names' is a list of previous layer names, it's needed
# to parse things like '[-1]' (meaning: the previous layer)
# when they appear in Desriptors.
def ParsedLineToXconfigLayer(first_token, key_to_value, prev_names):
    if first_token == 'input':
        return XconfigInputLayer(first_token, key_to_value, prev_names)
    elif first_token == 'output':
        return XconfigTrivialOutputLayer(first_token, key_to_value, prev_names)
    elif first_token == 'output-layer':
        return XconfigOutputLayer(first_token, key_to_value, prev_names)
    elif first_token in [ 'relu-layer', 'relu-renorm-layer', 'sigmoid-layer', 'tanh-layer' ]:
        return XconfigBasicLayer(first_token, key_to_value, prev_names)
    elif first_token == 'fixed-affine-layer':
        return XconfigFixedAffineLayer(first_token, key_to_value, prev_names)
    else:
        raise RuntimeError("Error parsing xconfig line (no such layer type): " +
                        first_token + ' ' +
                        ' '.join(['{0}={1}'.format(x,y) for x,y in key_to_value.items()]))


# Uses ParseConfigLine() to turn a config line that has been parsed into
# a first token e.g. 'affine-layer' and a key->value map like { 'dim':'1024', 'name':'affine1' },
# and then turns this into an object representing that line of the config file.
# 'prev_names' is a list of the names of preceding lines of the
# config file.
def ConfigLineToObject(config_line, prev_names = None):
    (first_token, key_to_value) = xconfig_utils.ParseConfigLine(config_line)
    return ParsedLineToXconfigLayer(first_token, key_to_value, prev_names)



# This function reads an xconfig file and returns it as a list of layers
# (usually we use the variable name 'all_layers' elsewhere for this).
# It will die if the xconfig file is empty or if there was
# some error parsing it.
def ReadXconfigFile(xconfig_filename):
    try:
        f = open(xconfig_filename, 'r')
    except Exception as e:
        sys.exit("{0}: error reading xconfig file '{1}'; error was {2}".format(
            sys.argv[0], xconfig_filename, repr(e)))
    all_layers = []
    while True:
        line = f.readline()
        if line == '':
            break
        x = xconfig_utils.ParseConfigLine(line)
        if x is None:
            continue   # line was blank or only comments.
        (first_token, key_to_value) = x
        # the next call will raise an easy-to-understand exception if
        # it fails.
        this_layer = ParsedLineToXconfigLayer(first_token,
                                              key_to_value,
                                              all_layers)
        all_layers.append(this_layer)
    if len(all_layers) == 0:
        raise RuntimeError("{0}: xconfig file '{1}' is empty".format(
            sys.argv[0], xconfig_filename))
    f.close()
    return all_layers


def TestLayers():
    # for some config lines that should be printed the same way as they
    # are read, check that this is the case.
    for x in [ 'input name=input dim=30' ]:
        assert str(ConfigLineToObject(x, [])) == x
