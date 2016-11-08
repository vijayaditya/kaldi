# Copyright 2016    Johns Hopkins University (Dan Povey)
#           2016    Vijayaditya Peddinti
#           2016    Yiming Wang
# Apache 2.0.


""" This module has the implementations of different LSTM layers.
"""

from libs.nnet3.xconfig.basic_layers import XconfigLayerBase


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
        assert first_token == "lstmp"
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

        # the equations implemented here are from Sak et. al. "Long Short-Term Memory Recurrent Neural Network Architectures for Large Scale Acoustic Modeling"
        # http://static.googleusercontent.com/media/research.google.com/en//pubs/archive/43905.pdf
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

# Same as the LSTMP layer except that the matrix multiplications are combined
# we probably keep only version after experimentation. One year old experiments
# show that this version is slightly worse and might require some tuning
class XconfigLstmpcLayer(XconfigLstmpLayer):
    def __init__(self, first_token, key_to_value, prev_names = None):
        assert first_token == "lstmpc"
        XconfigLayerBase.__init__(self, first_token, key_to_value, prev_names)

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

        # naming convention
        # <layer-name>.W_<outputname>.<input_name> e.g. Lstm1.W_i.xr for matrix providing output to gate i and operating on an appended vector [x,r]
        configs.append("# Full W_ifoc* matrix")
        configs.append("component name={0}.W_ifoc.xr type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, input_dim + rec_proj_dim, 4*cell_dim, affine_str))
        configs.append("# note : the cell outputs pass through a diagonal matrix")

        # we will not combine the diagonal matrix operations as one of these has a different delay
        configs.append("# note : the cell outputs pass through a diagonal matrix")
        configs.append("component name={0}.w_i.c type=NaturalGradientPerElementScaleComponent  dim={1} {2}".format(name, cell_dim, pes_str))
        configs.append("component name={0}.w_f.c type=NaturalGradientPerElementScaleComponent  dim={1} {2}".format(name, cell_dim, pes_str))
        configs.append("component name={0}.w_o.c type=NaturalGradientPerElementScaleComponent  dim={1} {2}".format(name, cell_dim, pes_str))

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

        component_nodes.append("component-node name={0}.ifoc_t component={0}.W_ifoc.xr input=Append({1}, IfDefined(Offset({0}_{2}, {3})))".format(name, input_descriptor, recurrent_connection, lstm_delay))


        offset = 0
        component_nodes.append("# i_t")
        component_nodes.append("dim-range-node name={0}.i1_t input-node={0}.ifoc_t dim-offset={1} dim={2}".format(name, offset, cell_dim))
        offset += cell_dim
        component_nodes.append("component-node name={0}.i2_t component={0}.w_i.cinput={1}".format(name, delayed_c_t_descriptor))
        component_nodes.append("component-node name={0}.i_t component={0}.i input=Sum({0}.i1_t, {0}.i2_t)".format(name))

        component_nodes.append("# f_t")
        component_nodes.append("dim-range-node name={0}.f1_t input-node={0}.ifoc_t dim-offset={1} dim={2}".format(name, offset, cell_dim))
        offset += cell_dim
        component_nodes.append("component-node name={0}.f2_t component={0}.w_f.c  input={1}".format(name, delayed_c_t_descriptor))
        component_nodes.append("component-node name={0}.f_t component={0}.f input=Sum({0}.f1_t, {0}.f2_t)".format(name))

        component_nodes.append("# o_t")
        component_nodes.append("dim-range-node name={0}.o1_t input-node={0}.ifoc_t dim-offset={1} dim={2}".format(name, offset, cell_dim))
        offset += cell_dim
        component_nodes.append("component-node name={0}.o2_t component={0}.w_o.c input={0}.c_t".format(name))
        component_nodes.append("component-node name={0}.o_t component={0}.o input=Sum({0}.o1_t, {0}.o2_t)".format(name))

        component_nodes.append("# h_t")
        component_nodes.append("component-node name={0}.h_t component={0}.h input={0}.c_t".format(name))

        component_nodes.append("# g_t")
        component_nodes.append("dim-range-node name={0}.g1_t input-node={0}.ifoc_t dim-offset={1} dim={2}".format(name, offset, cell_dim))
        offset += cell_dim
        component_nodes.append("component-node name={0}.g_t component={0}.g input={0}.g1_t".format(name))


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
#   'lstmp-layer name=lstm1 input=[-1] delay=-3'



