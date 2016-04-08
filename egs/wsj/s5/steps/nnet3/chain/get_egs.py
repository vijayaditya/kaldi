#!/usr/bin/env python


# Copyright 2016 Vijayaditya Peddinti.
# Apache 2.0.


# this script is based on steps/nnet3/lstm/train.sh

import os
import subprocess
import argparse
import sys
import pprint
import logging
import imp
import traceback
import shutil
import math

train_lib = imp.load_source('ntl', 'steps/nnet3/nnet3_train_lib.py')
data_lib = imp.load_source('dtl', 'utils/data/data_lib.py')
chain_lib = imp.load_source('ncl', 'steps/nnet3/chain/nnet3_chain_lib.py')
nnet3_log_parse = imp.load_source('nlp', 'steps/nnet3/report/nnet3_log_parse_lib.py')

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s [%(filename)s:%(lineno)s - %(funcName)s - %(levelname)s ] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.info('Starting chain model trainer (train.py)')


def GetArgs():
    # we add compulsary arguments as named arguments for readability
    parser = argparse.ArgumentParser(description="""
Generates training examples used to train the 'chain' system (and also the"""
" validation examples used for diagnostics), and puts them in separate archives.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # feat options
    parser.add_argument("--feat.dir", type=str, dest='feat_dir', required = True,
                        help="Directory with features used for training the neural network.")
    parser.add_argument("--feat.online-ivector-dir", type=str, dest='online_ivector_dir',
                        default = None, action = train_lib.NullstrToNoneAction,
                        help="directory with the ivectors extracted in an online fashion.")
    parser.add_argument("--feat.cmvn-opts", type=str, dest='cmvn_opts',
                        default = None, action = train_lib.NullstrToNoneAction,
                        help="A string specifying '--norm-means' and '--norm-vars' values")

    # egs extraction options

    parser.add_argument("--cut-zero-frames", type=int, default=-1,
                        help="Number of frames (measured before subsampling)"
                        " to zero the derivative on each side of a cut point"
                        " (if set, activates new-style derivative weights)")
    parser.add_argument("--frame-subsampling-factor", type=int, default=3,
                        help="Frames-per-second of features we train on."
                        " Divided by frames-per-second at output of the chain model.")
    parser.add_argument("--alignment-subsampling-factor", type=int, default=3,
                        help="Frames-per-second of input alignments."
                        " Divided by frames-per-second at output of the chain model.")


    parser.add_argument("--chunk-width", type=int, default = 150,
                        help="Number of output labels in each example.")
    parser.add_argument("--chunk-overlap-per-eg", type=int, default = 0,
                        help="Number of supervised frames of overlap that we"
                        " aim per eg. It can be used to avoid data wastage when"
                        " using --left-deriv-truncate and --right-deriv-truncat"
                        " options in the training script")

    parser.add_argument("--chunk-left-context", type=int, default = 4,
                        help="Number of additional frames of input to the left"
                        " of the input chunk. This extra context will be used"
                        " in the estimation of RNN state before prediction of"
                        " the first label. In the case of FF-DNN this extra"
                        " context will be used to allow for frame-shifts")
    parser.add_argument("--chunk-right-context", type=int, default = 4,
                        help="Number of additional frames of input to the right"
                        " of the input chunk. This extra context will be used"
                        " in the estimation of bidirectional RNN state before"
                        " prediction of the first label.")
    parser.add_argument("--valid-left-context", type=int, default = None,
                        help=" Amount of left-context for validation egs, typically"
                        " used in recurrent architectures to ensure matched"
                        " condition with training egs")
    parser.add_argument("--valid-right-context", type=int, default = None,
                        help=" Amount of right-context for validation egs, typically"
                        " used in recurrent architectures to ensure matched"
                        " condition with training egs")
    parser.add_argument("--compress", type=str, default = True,
                        action = train_lib.StrToBoolAction,
                        choices = ["true", "false"],
                        help="If false, disables compression. Might be necessary"
                        " to check if results will be affected.")
    parser.add_argument("--num-utts-subset", type=int, default = 300,
                        help="Number of utterances in valudation and training"
                        " subsets used for shrinkage and diagnostics")
    parser.add_argument("--num-train-egs-combine", type=int, default=1000,
                        help="Training examples for combination weights at the"
                        " very end.")
    parser.add_argument("--num-valid-egs-combine", type=int, default=0,
                        help="Validation examples for combination weights at the"
                        " very end.")
    parser.add_argument("--frames-per-iter", type=int, default=400000,
                        help="Number of of supervised-frames seen per job of"
                        " training iteration. Measured at the sampling rate of"
                        " the features used. This is just a guideline; the script"
                        " will pick a number that divides the number of samples"
                        " in the entire data")
    parser.add_argument("--right-tolerance", type=int, default=None, help="")
    parser.add_argument("--left-tolerance", type=int, default=None, help="")


    parser.add_argument("--max-shuffle-jobs-run", type=int, default=50,
                        help="Limits the number of shuffle jobs which are"
                        " simultaneously run. Data shuffling jobs are fairly CPU"
                        " intensive as they include the nnet3-chain-normalize-egs"
                        " command; so we can run significant number of jobs"
                        " without overloading the disks.")
    parser.add_argument("--num-jobs", type=int, default=15,
                        help="Number of jobs to be run")

    parser.add_argument("--lat-dir", type=str, required = True,
                        help="Directory with alignments used for training the neural network.")
    parser.add_argument("--chain-dir", type=str, required = True,
                        help="Directory with trans_mdl, tree, normalization.fst")
    parser.add_argument("--dir", type=str, required = True,
                        help="Directory to store the examples")

    print(' '.join(sys.argv))
    print(sys.argv)

    args = parser.parse_args()

    [args, run_opts] = ProcessArgs(args)

    return [args, run_opts]

def ProcessArgs(args):
    # process the options
    if args.chunk_width < 1:
        raise Exception("--egs.chunk-width should have a minimum value of 1")

    if args.chunk_left_context < 0:
        raise Exception("--egs.chunk-left-context should be non-negative")

    if args.chunk_right_context < 0:
        raise Exception("--egs.chunk-right-context should be non-negative")

    if (not os.path.exists(args.dir)) or (not os.path.exists(args.dir+"/configs")):
        raise Exception("""This scripts expects {0} to exist and have a configs
        directory which is the output of make_configs.py script""")

    return args

def CheckForRequiredFiles(feat_dir, chain_dir, lat_dir, online_ivector_dir = None):
    required_files = ['{0}/feats.scp'.format(feat_dir), '{0}/lat.1.gz'.format(lat_dir),
                      '{0}/final.mdl'.format(lat_dir), '{0}/0.trans_mdl'.format(chain_dir),
                       '{0}/tree'.format(chain_dir), '{0}/normalization.fst'.format(chain_dir)]
    if online_ivector_dir is not None:
        required_files.append('{0}/ivector_online.scp'.format(online_ivector_dir))
        required_files.append('{0}/ivector_period'.format(online_ivector_dir))

    for file in required_files:
        if not os.path.isfile(file):
            raise Exception('Expected {0} to exist.'.format(file))


def SampleUtts(feat_dir, num_utts_subset, min_duration, exclude_list=None):
    utt2durs = data_lib.GetUtt2Dur(feat_dir).items()
    utt2uniq, uniq2utt = data_lib.GetUtt2Uniq(feat_dir)

    random.shuffle(utt2durs)
    valid_utts = []

    index = 0
    while len(valid_utts) < num_utts_subset:
        if utt2durs[index][-1] >= min_duration:
            if utt2uniq is not None:
                uniq_id = utt2uniq[utt2durs[index][0]]
                utts2add = uniq2utt[uniq_id]
            else:
                utts2add = [utt2durs[index][0]]
            for utt in utts2add:
                if exclude_list is not None and utt in exclude_list:
                    continue
            valid_utts = valid_utts + utts2add
        index = index + 1

    if len(valid_utts) < num_utts_subset:
        raise Exception("Number of utterances which have length at least "
                "{cw} is really low. Please check your data.".format(cw = chunk_width))
    return valid_utts

def WriteList(listd, file_name):
    file_handle = open(file_name, 'w')
    for item in listd:
        file_handle.write(str(item))
    file_handle.close()

def GetMaxOpenHandles():
    stdout, stderr = RunKaldiCommand("ulimit -n")
    return int(stdout)

def CopyTrainingLattices(lat_dir, dir, cmd, num_lat_jobs):
  RunKaldiCommand("""
  {cmd} --max-jobs-run 6 JOB=1:{nj} {dir}/log/lattice_copy.JOB.log \
    lattice-copy "ark:gunzip -c {latdir}/lat.JOB.gz|" ark,scp:{dir}/lat.JOB.ark,{dir}/lat.JOB.scp""".format(cmd = cmd, nj = num_lat_jobs, dir = dir,
                       latdir = lat_dir))

    total_lat_file = open('{0}/lat.scp'.format(dir), 'w')
    for id in range(1, num_lat_jobs+1):
        lat_file_name = '{0}/lat.{1}.scp'.format(dir, id)
        lat_lines = ''.join(open(lat_file_name, 'r').readlines())
        total_lat_file.write(lat_lines)
    total_lat_file.close()

def GetFeatIvectorStrings(dir, feat_dir, split_feat_dir, cmvn_opt_string, ivector_dir = None):
    feats="ark,s,cs:utils/filter_scp.pl --exclude {dir}/valid_uttlist {sdir}/JOB/feats.scp | apply-cmvn {cmvn} --utt2spk=ark:{sdir}/JOB/utt2spk scp:{sdir}/JOB/cmvn.scp scp:- ark:- |".format(dir = dir, sdir = split_feat_dir, cmvn = cmvn_opt_string)
    valid_feats="ark,s,cs:utils/filter_scp.pl {dir}/valid_uttlist {fdir}/feats.scp | apply-cmvn {cmvn} --utt2spk=ark:{fdir}/utt2spk scp:{fdir}/cmvn.scp scp:- ark:- |".format(dir = dir, fdir = feat_dir, cmvn = cmvn_opt_string)
    train_subset_feats="ark,s,cs:utils/filter_scp.pl {dir}/train_subset_uttlist  {fdir}/feats.scp | apply-cmvn {cmvn} --utt2spk=ark:{fdir}/utt2spk scp:{fdir}/cmvn.scp scp:- ark:- |".format(dir = dir, fdir = feat_dir, cmvn = cmvn_opt_string)

    if ivector_dir is not None:
        ivector_period = train_lib.GetIvectorPeriod(ivector_dir)
        ivector_opt="--ivectors='ark,s,cs:utils/filter_scp.pl {sdir}/JOB/utt2spk {idir}/ivector_online.scp | subsample-feats --n=-{period} scp:- ark:- |'".format(sdir = split_feat_dir, idir = ivector_dir, period = ivector_period)
        valid_ivector_opt="--ivectors='ark,s,cs:utils/filter_scp.pl {dir}/valid_uttlist {idir}/ivector_online.scp | subsample-feats --n=-{period} scp:- ark:- |'".format(dir = dir, idir = ivector_dir, period = ivector_period)
        train_subset_ivector_opt="--ivectors='ark,s,cs:utils/filter_scp.pl {dir}/train_subset_uttlist {idir}/ivector_online.scp | subsample-feats --n=-{period} scp:- ark:- |'".format(dir = dir, idir = ivector_dir, period = ivector_period)
    else:
        ivector_opt = None
        valid_ivector_opt = None
        train_subset_ivector_opt = None

    return {'feats':feats, 'valid_feats':valid_feats, 'train_subset_feats':train_subset_feats,
            'ivector_opt':ivector_opt, 'valid_ivector_opt':valid_ivector_opt, 'train_subset_ivector_opt':train_subset_ivector_opt}

def GetEgsOptions(left_context, right_context,
                     valid_left_context, valid_right_context,
                     chunk_width,
                     frames_overlap_per_eg, frame_subsampling_factor,
                     alignment_subsampling_factor, left_tolerance,
                     right_tolerance, compress, cut_zero_frames):

    egs_opts_func = lambda chunk_width, left_context, right_context :  "--left-context={lc} --right-context={rc} --num-frames={cw} --num-frames-overlap={fope} --frame-subsampling-factor={fsf} --compress={comp} --cut-zero-frames={czf}".format(lc = left_context, rc = right_context,
              cw = chunk_width, fopg = frames_overlap_per_eg,
              fsf = frame_subsampling_factor, comp = compress,
              czf = cut_zero_frames)

    if valid_left_context is None:
        valid_left_context = left_context
    if valid_right_context is None:
        valid_right_context = right_context

    # don't do the overlap thing for the validation data.
    valid_egs_opts="--left-context={vlc} --right-context={vrc} --num-frames={cw} --frame-subsampling-factor={fsf} --compress={comp}".format(vlc = valid_left_context,
            vrc = valid_right_context, cw = chunk_width,
            fsf = frame_subsampling_factor, comp = compress)

    ctc_supervision_all_opts="--lattice-input=true --frame-subsampling-factor={asf}".format(asf = alignment_subsampling_factor)
    if right_tolerance is not None:
        ctc_supervision_all_opts="{ctc} --right-tolerance={rt}".format(ctc = ctc_supervision_all_opts, rt = right_tolerance)

    if left_tolerance is not None:
        ctc_supervision_all_opts="{ctc} --left-tolerance={lt}".format(ctc = ctc_supervision_all_opts, lt = left_tolerance)

    return {'egs_opts_function' : egs_opts_func,
            'valid_egs_opts' : valid_egs_opts,
            'ctc_supervision_all_opts' : ctc_supervision_all_opts}

def GenerateValidTrainSubsetEgs(dir, lat_dir, chain_dir,
                                feat_ivector_strings, egs_opts,
                                num_train_egs_combine,
                                num_valid_egs_combine,
                                num_egs_diagnostic, cmd):
    valid_utts = map(lambda x: x.strip(), open('{0}/valid_uttlist', 'r').readlines())
    train_utts = map(lambda x: x.strip(), open('{0}/train_subset_uttlist', 'r').readlines())
    lat_scp = map(lambda x: x.strip(), open('{0}/lat.scp', 'r').readlines())
    utt_set = set([valid_utts + train_utts])

    lat_scp_special = []
    for line in lat_scp:
        if line.split()[0] in utt_set:
            lat_scp_special.append(line)
    file_handle = open('{0}/lat_special.scp'.format(dir), 'w')
    file_handle.write('\n'.join(lat_scp_special))
    file_handle.close()

    RunKaldiCommand("""
  {cmd} {dir}/log/create_valid_subset.log \
    lattice-align-phones --replace-output-symbols=true {ldir}/final.mdl scp:{dir}/lat_special.scp ark:- \| \
    chain-get-supervision {ctc_opt} {cdir}/tree {cdir}/0.trans_mdl \
      ark:- ark:- \| \
    nnet3-chain-get-egs {v_iv_opt} {v_egs_opt} {cdir}/normalization.fst \
      "{v_feats}" ark,s,cs:- "ark:{dir}/valid_all.cegs" """.format(
          cmd = cmd, dir = dir, ldir = lat_dir, cdir = chain_dir,
          ctc_opt = egs_opts['ctc_supervision_all_opts'],
          v_egs_opt = egs_opts['valid_egs_opts'],
          v_iv_opt = feat_ivector_strings['valid_ivector_opt'],
          v_feats = feat_ivector_strings['valid_feats']))

    RunKaldiCommand("""
  {cmd} {dir}/log/create_train_subset.log \
    lattice-align-phones --replace-output-symbols=true {ldir}/final.mdl scp:{dir}/lat_special.scp ark:- \| \
    chain-get-supervision {ctc_opt} \
    {cdir}/tree {cdir}/0.trans_mdl ark:- ark:- \| \
    nnet3-chain-get-egs {t_iv_opt} {v_egs_opt} {cdir}/normalization.fst \
       "{t_feats}" ark,s,cs:- "ark:{dir}/train_subset_all.cegs" """.format(
          cmd = cmd, dir = dir, ldir = lat_dir, cdir = chain_dir,
          ctc_opt = egs_opts['ctc_supervision_all_opts'],
          v_egs_opt = egs_opts['valid_egs_opts'],
          t_iv_opt = feat_ivector_strings['train_subset_ivector_opt'],
          v_feats = feat_ivector_strings['train_subset_feats']))

    logger.info("... Getting subsets of validation examples for diagnostics and combination.")

    RunKaldiCommand("""
  {cmd} {dir}/log/create_valid_subset_combine.log \
    nnet3-chain-subset-egs --n={nve_combine} ark:{dir}/valid_all.cegs \
    ark:{dir}/valid_combine.cegs""".format(
        cmd = cmd, dir = dir, nve_combine = num_valid_egs_combine))


    RunKaldiCommand("""
  {cmd} {dir}/log/create_valid_subset_diagnostic.log \
    nnet3-chain-subset-egs --n={ne_diagnostic} ark:{dir}/valid_all.cegs \
    ark:{dir}/valid_diagnostic.cegs""".format(
        cmd = cmd, dir = dir, ne_diagnostic = num_egs_diagnostic))

    RunKaldiCommand("""
  {cmd} {dir}/log/create_train_subset_combine.log \
    nnet3-chain-subset-egs --n={nte_combine} ark:{dir}/train_subset_all.cegs \
    ark:{dir}/train_combine.cegs""".format(
        cmd = cmd, dir = dir, nte_combine = num_train_egs_combine))

    RunKaldiCommand("""
  {cmd} {dir}/log/create_train_subset_diagnostic.log \
    nnet3-chain-subset-egs --n={ne_diagnostic} ark:{dir}/train_subset_all.cegs \
    ark:{dir}/train_diagnostic.cegs""".format(
        cmd = cmd, dir = dir, ne_diagnostic = num_egs_diagnostic))

    RunKaldiCommand(""" cat {dir}/valid_combine.cegs {dir}/train_combine.cegs > {dir}/combine.cegs""".format(dir = dir))

    # perform checks
    for file_name in '{0}/combine.cegs {0}/train_diagnostic.cegs {0}/valid_diagnostic.cegs'.format(dir).split():
        if os.path.getsize(file_name) == 0:
            raise Exception("No examples in {0}".format(file_name))

    # clean-up
    for file_name in '{0}/valid_all.cegs {0}/train_subset_all.cegs {0}/train_combine.cegs {0}/valid_combine.cegs'.format(dir).split():
        os.path.remove(file_name)

# args is a Namespace with the required parameters
def GenerateEgs(args):
    arg_string = pprint.pformat(vars(args))
    logger.info("Arguments for the data generation\n{0}".format(arg_string))

    # Check files
    CheckForRequiredFiles(args.feat_dir, args.chain_dir, args.lat_dir, args.online_ivector_dir)

    # split the training data into parts for individual jobs
    # we will use the same number of jobs as that used for alignment
    split_feat_dir = train_lib.SplitData(args.feat_dir, args.num_jobs)
    for directory in '{0}/log {0}/info'.format(args.dir).split():
        if not os.path.exists(directory):
            os.makedirs(directory)

    frame_shift = data_lib.GetFrameShift(args.feat_dir)
    min_duration = float(args.chunk_width)/frame_shift
    valid_utts = SampleUtts(args.feat_dir, arg.num_utts_subset, min_duration)
    train_utts = SampleUtts(args.feat_dir, arg.num_utts_subset, min_duration, exclude_list = valid_utts)
    WriteList(valid_utts, '{0}/valid_uttlist'.format(args.dir))
    WriteList(train_utts, '{0}/train_subset_uttlist'.format(args.dir))

    feat_ivector_strings = GetFeatIvectorStrings(args.dir, args.feat_dir, split_feat_dir, args.cmvn_opts, ivector_dir = args.online_ivector_dir)

    num_lat_jobs = train_lib.GetNumberOfJobs(args.lat_dir)
    if args.stage <= 1:
        logger.info("Copying training lattices.")
        CopyTrainingLattices(args.lat_dir, args.dir, args.cmd, num_lat_jobs)

    egs_opts = GetEgsOptions(args.left_context, args.right_context,
                            args.chunk_width, args.frames_overlap_per_eg,
                            args.frame_subsampling_factor,
                            args.alignment_subsampling_factor,
                            args.left_tolerance, args.right_tolerance,
                            args.compress, args.cut_zero_frames)
    if args.stage <= 2:
        logger.info("Getting validation and training subset examples")
        GenerateValidTrainSubsetEgs(dir, lat_dir, chain_dir,
                                    feat_ivector_strings, egs_opts,
                                    args.num_train_egs_combine,
                                    args.num_valid_egs_combine,
                                    args.num_egs_diagnostic, args.cmd)


    num_frames = data_lib.GetNumFrames(args.feat_dir)
    num_archives = int(num_frames/args.frames_per_iter+1)
    max_open_handles = GetMaxOpenHandles()

    if args.stage <= 3:
        logger.info("Generating training examples on disk.")
        GenerateTrainingExamples(args.dir, args.lat_dir,
                                 feat_ivector_strings, egs_opts,
                                 args.chunk_width, args.left_context, args.right_context,
                                 num_archives_intermediate)


    if args.stage <= 4:
        logger.info("Recombining and shuffling order of archives on disk")

    # Set some variables.
    feat_dim = train_lib.GetFeatDim(args.feat_dir)
    ivector_dim = train_lib.GetIvectorDim(ivector_dir)


    config_dir = '{0}/configs'.format(args.dir)
    var_file = '{0}/vars'.format(config_dir)

    [model_left_context, model_right_context, num_hidden_layers] = train_lib.ParseModelConfigVarsFile(var_file)
    # Initialize as "raw" nnet, prior to training the LDA-like preconditioning
    # matrix.  This first config just does any initial splicing that we do;
    # we do this as it's a convenient way to get the stats for the 'lda-like'
    # transform.
    if (args.stage <= -6):
        logger.info("Creating phone language-model")
        chain_lib.CreatePhoneLm(args.dir, args.tree_dir, run_opts, lm_opts = args.lm_opts)

    if (args.stage <= -5):
        logger.info("Creating denominator FST")
        chain_lib.CreateDenominatorFst(args.dir, args.tree_dir, run_opts)

    if (args.stage <= -4):
        logger.info("Initializing a basic network for estimating preconditioning matrix")
        train_lib.RunKaldiCommand("""
{command} {dir}/log/nnet_init.log \
    nnet3-init --srand=-2 {dir}/configs/init.config {dir}/init.raw
    """.format(command = run_opts.command,
               dir = args.dir))

    left_context = args.chunk_left_context + model_left_context
    right_context = args.chunk_right_context + model_right_context

    default_egs_dir = '{0}/egs'.format(args.dir)
    if (args.stage <= -3) and args.egs_dir is None:
        logger.info("Generating egs")
        # this is where get_egs.sh is called.
        chain_lib.GenerateChainEgs(args.dir, args.feat_dir, args.lat_dir, default_egs_dir,
                                    left_context + args.frame_subsampling_factor/2,
                                    right_context + args.frame_subsampling_factor/2,
                                    run_opts,
                                    left_tolerance = args.left_tolerance,
                                    right_tolerance = args.right_tolerance,
                                    frame_subsampling_factor = args.frame_subsampling_factor,
                                    alignment_subsampling_factor = args.alignment_subsampling_factor,
                                    frames_per_eg = args.chunk_width,
                                    egs_opts = args.egs_opts,
                                    cmvn_opts = args.cmvn_opts,
                                    online_ivector_dir = args.online_ivector_dir,
                                    frames_per_iter = args.frames_per_iter,
                                    transform_dir = args.transform_dir,
                                    stage = args.egs_stage)

    if args.egs_dir is None:
        egs_dir = default_egs_dir
    else:
        egs_dir = args.egs_dir

    [egs_left_context, egs_right_context, frames_per_eg, num_archives] = train_lib.VerifyEgsDir(egs_dir, feat_dim, ivector_dim, left_context, right_context)
    assert(args.chunk_width == frames_per_eg)
    num_archives_expanded = num_archives * args.frame_subsampling_factor

    if (args.num_jobs_final > num_archives_expanded):
        raise Exception('num_jobs_final cannot exceed the expanded number of archives')

    # copy the properties of the egs to dir for
    # use during decoding
    train_lib.CopyEgsPropertiesToExpDir(egs_dir, args.dir)

    if (args.stage <= -2):
        logger.info('Computing the preconditioning matrix for input features')

        chain_lib.ComputePreconditioningMatrix(args.dir, egs_dir, num_archives, run_opts,
                                               max_lda_jobs = args.max_lda_jobs,
                                               rand_prune = args.rand_prune)

    if (args.stage <= -1):
        logger.info("Preparing the initial acoustic model.")
        chain_lib.PrepareInitialAcousticModel(args.dir, run_opts)

    file_handle = open("{0}/frame_subsampling_factor".format(args.dir),"w")
    file_handle.write(str(args.frame_subsampling_factor))
    file_handle.close()

    # set num_iters so that as close as possible, we process the data $num_epochs
    # times, i.e. $num_iters*$avg_num_jobs) == $num_epochs*$num_archives,
    # where avg_num_jobs=(num_jobs_initial+num_jobs_final)/2.
    num_archives_to_process = args.num_epochs * num_archives_expanded
    num_archives_processed = 0
    num_iters=(num_archives_to_process * 2) / (args.num_jobs_initial + args.num_jobs_final)

    num_iters_combine = train_lib.VerifyIterations(num_iters, args.num_epochs,
                                                   num_hidden_layers, num_archives_expanded,
                                                   args.max_models_combine, args.add_layers_period,
                                                   args.num_jobs_final)

    learning_rate = lambda iter, current_num_jobs, num_archives_processed: train_lib.GetLearningRate(iter, current_num_jobs, num_iters,
                                                                                           num_archives_processed,
                                                                                           num_archives_to_process,
                                                                                           args.initial_effective_lrate,
                                                                                           args.final_effective_lrate)

    logger.info("Training will run for {0} epochs = {1} iterations".format(args.num_epochs, num_iters))
    for iter in range(num_iters):
        if (args.exit_stage is not None) and (iter == args.exit_stage):
            logger.info("Exiting early due to --exit-stage {0}".format(iter))
            return
        current_num_jobs = int(0.5 + args.num_jobs_initial + (args.num_jobs_final - args.num_jobs_initial) * float(iter) / num_iters)

        if args.stage <= iter:
            if args.shrink_value != 1.0:
                model_file = "{dir}/{iter}.mdl".format(dir = args.dir, iter = iter)
                shrinkage_value = args.shrink_value if train_lib.DoShrinkage(iter, model_file, args.shrink_nonlinearity, args.shrink_threshold) else 1
            else:
                shrinkage_value = args.shrink_value
            logger.info("On iteration {0}, learning rate is {1} and shrink value is {2}.".format(iter, learning_rate(iter, current_num_jobs, num_archives_processed), shrinkage_value))

            TrainOneIteration(args.dir, iter, egs_dir, current_num_jobs,
                              num_archives_processed, num_archives,
                              learning_rate(iter, current_num_jobs, num_archives_processed),
                              shrinkage_value,
                              args.num_chunk_per_minibatch,
                              num_hidden_layers, args.add_layers_period,
                              args.apply_deriv_weights, args.left_deriv_truncate, args.right_deriv_truncate,
                              args.l2_regularize, args.xent_regularize, args.leaky_hmm_coefficient,
                              args.momentum, args.max_param_change,
                              args.shuffle_buffer_size,
                              args.frame_subsampling_factor,
                              args.truncate_deriv_weights, run_opts)
            if args.cleanup:
                # do a clean up everythin but the last 2 models, under certain conditions
                train_lib.RemoveModel(args.dir, iter-2, num_iters, num_iters_combine,
                            args.preserve_model_interval)

            if args.email is not None:
                reporting_iter_interval = num_iters * args.reporting_interval
                if iter % reporting_iter_interval == 0:
                # lets do some reporting
                    [report, times, data] = nnet3_log_parse.GenerateAccuracyReport(args.dir, key="log-probability")
                    message = report
                    subject = "Update : Expt {dir} : Iter {iter}".format(dir = args.dir, iter = iter)
                    train_lib.SendMail(message, subject, args.email)

        num_archives_processed = num_archives_processed + current_num_jobs

    if args.stage <= num_iters:
        logger.info("Doing final combination to produce final.mdl")
        chain_lib.CombineModels(args.dir, num_iters, num_iters_combine,
                args.num_chunk_per_minibatch, egs_dir,
                args.leaky_hmm_coefficient, args.l2_regularize,
                args.xent_regularize, run_opts)

    if args.cleanup:
        logger.info("Cleaning up the experiment directory {0}".format(args.dir))
        remove_egs = args.remove_egs
        if args.egs_dir is not None:
            # this egs_dir was not created by this experiment so we will not
            # delete it
            remove_egs = False

        train_lib.CleanNnetDir(args.dir, num_iters, egs_dir,
                               preserve_model_interval = args.preserve_model_interval,
                               remove_egs = remove_egs)

    # do some reporting
    [report, times, data] = nnet3_log_parse.GenerateAccuracyReport(args.dir, "log-probability")
    if args.email is not None:
        train_lib.SendMail(report, "Update : Expt {0} : complete".format(args.dir), args.email)

    report_handle = open("{dir}/accuracy.report".format(dir = args.dir), "w")
    report_handle.write(report)
    report_handle.close()

def Main():
    [args, run_opts] = GetArgs()
    try:
        Train(args, run_opts)
    except Exception as e:
        if args.email is not None:
            message = "Training session for experiment {dir} died due to an error.".format(dir = args.dir)
            sendMail(message, message, args.email)
        traceback.print_exc()
        raise e

if __name__ == "__main__":
    Main()
