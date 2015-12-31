#!/bin/bash

# Copyright 2015  Johns Hopkins University (Author: Daniel Povey).
#           2015  Vijayaditya Peddinti
# Apache 2.0.


# this is a basic cwrnn script
# CWRNN script runs for more epochs than the TDNN script
# and each epoch takes twice the time

# At this script level we don't support not running on GPU, as it would be painfully slow.
# If you want to run without GPU you'd have to call lstm/train.sh with --gpu false

stage=0
train_stage=-10
get_egs_stage=-10
has_fisher=true
affix=
speed_perturb=true
common_egs_dir=
decode_iter=
l2_regularize=0.00
xent_regularize=0.00

# LSTM options
input_type="smooth"
splice_indexes="-2,-1,0,1,2 0 0"
label_delay=5
num_cwrnn_layers=3
hidden_dim=1024
chunk_left_context=80
clipping_threshold=30.0
norm_based_clipping=true
ratewise_params="{'T1': {'rate':1, 'dim':768},
                  'T2': {'rate':1.0/2, 'dim':512},
                  'T3': {'rate':1.0/4, 'dim':512}}"
nonlinearity="RectifiedLinearComponent"
diag_init_scaling_factor=0.1
ng_affine_options=
projection_dim=0
subsample=true
filter_input_step=1
include_log_softmax=true

# training options
num_epochs=4
initial_effective_lrate=0.0003
final_effective_lrate=0.00003
momentum=0.5
shrink=0.99
shrink_threshold=0.125
leftmost_questions_truncate=-1
max_param_change=2.0
final_layer_normalize_target=0.5
num_jobs_initial=3
num_jobs_final=16
chunk_width=150

num_chunk_per_minibatch=100
remove_egs=true

# feature options
use_ivectors=true

#decode options
extra_left_context=
frames_per_chunk=

# End configuration section.

echo "$0 $@"  # Print the command line for logging

. cmd.sh
. ./path.sh
. ./utils/parse_options.sh

if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

# The iVector-extraction and feature-dumping parts are the same as the standard
# nnet3 setup, and you can skip them by setting "--stage 8" if you have already
# run those things.
suffix=
if [ "$speed_perturb" == "true" ]; then
  suffix=_sp
fi
dir=exp/chain/cwrnn
dir=$dir${affix:+_$affix}
if [ $label_delay -gt 0 ]; then dir=${dir}_ld$label_delay; fi
dir=${dir}$suffix
train_set=train_nodup$suffix
ali_dir=exp/tri4_ali_nodup$suffix
treedir=exp/chain/tri5_2o_tree$suffix
lang=data/lang_chain_2o

# if we are using the speed-perturbed data we need to generate
# alignments for it.
local/nnet3/run_ivector_common.sh --stage $stage \
  --speed-perturb $speed_perturb \
  --generate-alignments $speed_perturb || exit 1;

if [ $stage -le 9 ]; then
  # Get the alignments as lattices (gives the CTC training more freedom).
  # use the same num-jobs as the alignments
  nj=$(cat exp/tri4_ali_nodup$suffix/num_jobs) || exit 1;
  steps/align_fmllr_lats.sh --nj $nj --cmd "$train_cmd" data/$train_set \
    data/lang exp/tri4 exp/tri4_lats_nodup$suffix
  rm exp/tri4_lats_nodup$suffix/fsts.*.gz # save space
fi

if [ $stage -le 10 ]; then
  # Create a version of the lang/ directory that has one state per phone in the
  # topo file. [note, it really has two states.. the first one is only repeated
  # once, the second one has zero or more repeats.]
  rm -rf $lang
  cp -r data/lang $lang
  silphonelist=$(cat $lang/phones/silence.csl) || exit 1;
  nonsilphonelist=$(cat $lang/phones/nonsilence.csl) || exit 1;
  # Use our special topology... note that later on may have to tune this
  # topology.
  steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist >$lang/topo
fi

if [ $stage -le 11 ]; then
  # Build a tree using our new topology.
  steps/nnet3/chain/build_tree.sh --frame-subsampling-factor 3 \
      --leftmost-questions-truncate $leftmost_questions_truncate \
      --cmd "$train_cmd" 9000 data/$train_set $lang $ali_dir $treedir
fi

if [ $stage -le 12 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{3,4,5,6}/$USER/kaldi-data/egs/swbd-cwrnn-$(date +'%m_%d_%H_%M')/s5/$dir/egs/storage $dir/egs/storage
  fi
  if [ "$use_ivectors" == "true" ]; then
    ivector_opts=" --online-ivector-dir exp/nnet3/ivectors_${train_set} "
    cmvn_opts="--norm-means=false --norm-vars=false"
  else
    ivector_opts=
    cmvn_opts="--norm-means=true --norm-vars=true"
  fi

 touch $dir/egs/.nodelete # keep egs around when that run dies.

  steps/nnet3/chain/train_cwrnn.sh $ivector_opts \
    --l2-regularize $l2_regularize \
    --xent-regularize $xent_regularize \
    --include-log-softmax $include_log_softmax \
    --stage $train_stage \
    --apply-deriv-weights false \
    --lm-opts "--num-extra-lm-states=2000" \
    --get-egs-stage $get_egs_stage \
    --num-chunk-per-minibatch $num_chunk_per_minibatch \
    --egs-opts "--frames-overlap-per-eg 0" \
    --label-delay $label_delay \
    --projection-dim $projection_dim \
    --subsample "$subsample" \
    --filter-input-step $filter_input_step \
    --chunk-width $chunk_width \
    --chunk-left-context $chunk_left_context \
    --num-epochs $num_epochs --num-jobs-initial $num_jobs_initial --num-jobs-final $num_jobs_final \
    --num-chunk-per-minibatch $num_chunk_per_minibatch \
    --splice-indexes "$splice_indexes" \
    --feat-type raw \
    --cmvn-opts "$cmvn_opts" \
    --initial-effective-lrate $initial_effective_lrate --final-effective-lrate $final_effective_lrate \
    --momentum $momentum \
    --shrink $shrink \
    --shrink-threshold $shrink_threshold \
    --input-type "$input_type" \
    --cmd "$decode_cmd" \
    --max-param-change $max_param_change \
    --ratewise-params "$ratewise_params" \
    --nonlinearity "$nonlinearity" \
    --diag-init-scaling-factor $diag_init_scaling_factor \
    --num-cwrnn-layers $num_cwrnn_layers \
    --hidden-dim $hidden_dim \
    --clipping-threshold $clipping_threshold \
    --norm-based-clipping $norm_based_clipping \
    --egs-dir "$common_egs_dir" \
    --remove-egs $remove_egs \
    data/${train_set}_hires $treedir exp/tri4_lats_nodup$suffix $dir  || exit 1;
fi

if [ $stage -le 13 ]; then
  # Note: it might appear that this $lang directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  utils/mkgraph.sh --self-loop-scale 1.0 data/lang_sw1_tg $dir $dir/graph_sw1_tg
fi

decode_suff=sw1_tg
graph_dir=$dir/graph_sw1_tg
if [ $stage -le 14 ]; then
  if [ -z $extra_left_context ]; then
    extra_left_context=$chunk_left_context
  fi
  if [ -z $frames_per_chunk ]; then
    frames_per_chunk=$chunk_width
  fi
  iter_opts=
  if [ ! -z $decode_iter ]; then
    iter_opts=" --iter $decode_iter "
  fi
  for decode_set in train_dev eval2000; do
      (
      steps/nnet3/lstm/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
          --nj 250 --cmd "$decode_cmd" $iter_opts \
          --extra-left-context $extra_left_context  \
          --frames-per-chunk "$frames_per_chunk" \
          --online-ivector-dir exp/nnet3/ivectors_${decode_set} \
         $graph_dir data/${decode_set}_hires $dir/decode_${decode_set}_${decode_suff} || exit 1;
      
      if $has_fisher; then
          steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
            data/lang_sw1_{tg,fsh_fg} data/${decode_set}_hires \
            $dir/decode_${decode_set}_sw1_{tg,fsh_fg} || exit 1;
      fi
      ) &
  done
fi
wait;
exit 0;
