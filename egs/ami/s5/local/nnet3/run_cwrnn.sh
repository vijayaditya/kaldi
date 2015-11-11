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
has_fisher=true
mic=ihm
use_sat_alignments=true
affix=
speed_perturb=true
common_egs_dir=

# LSTM options
input_type="smooth"
splice_indexes="-2,-1,0,1,2 0 0"
label_delay=5
num_cwrnn_layers=3
hidden_dim=1024
chunk_width=20
chunk_left_context=40
clipping_threshold=30.0
norm_based_clipping=true
ratewise_params=" {'T1': {'rate':1, 'dim':512},
                   'T2': {'rate':1.0/2, 'dim':256},
                   'T3': {'rate':1.0/4, 'dim':256},
                   'T4': {'rate':1.0/8, 'dim':256}
                   }"
nonlinearity="SigmoidComponent"
diag_init_scaling_factor=0
ng_affine_options=
projection_dim=0
subsample=true

# training options
num_epochs=10
initial_effective_lrate=0.0003
final_effective_lrate=0.00003
num_jobs_initial=2
num_jobs_final=12
momentum=0.5
shrink=0.99
shrink_threshold=0.125
num_chunk_per_minibatch=100
frames_per_iter=400000
remove_egs=true
max_param_change=1.0

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

use_delay=false
if [ $label_delay -gt 0 ]; then use_delay=true; fi

dir=exp/$mic/nnet3/cwrnn${speed_perturb:+_sp}${affix:+_$affix}${use_delay:+_ld$label_delay}
if [ "$use_sat_alignments" == "true" ] ; then
  gmm_dir=exp/$mic/tri4a
else
  gmm_dir=exp/$mic/tri3a
fi

if [ "$speed_perturb" == "true" ]; then
  train_set=train_sp
  ali_dir=${gmm_dir}_sp_ali
else
  train_set=train
  ali_dir=${gmm_dir}_ali
fi

final_lm=`cat data/local/lm/final_lm`
LM=$final_lm.pr1-7
graph_dir=$gmm_dir/graph_${LM}

local/nnet3/run_ivector_common.sh --stage $stage \
  --mic $mic \
  --use-sat-alignments $use_sat_alignments \
  --speed-perturb $speed_perturb || exit 1;

if [ $stage -le 8 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{3,4,5,6}/$USER/kaldi-data/egs/ami-$(date +'%m_%d_%H_%M')/s5/$dir/egs/storage $dir/egs/storage
  fi
  if [ "$use_ivectors" == "true" ]; then
    ivector_opts=" --online-ivector-dir exp/$mic/nnet3/ivectors_${train_set}_hires "
    cmvn_opts="--norm-means=false --norm-vars=false"
  else
    ivector_opts=
    cmvn_opts="--norm-means=true --norm-vars=true"
  fi

  samples_per_iter=$(python -c "print int(float($frames_per_iter)/($chunk_width))")
  steps/nnet3/cwrnn/train.sh $ivector_opts \
    --stage $train_stage \
    --label-delay $label_delay \
    --projection-dim $projection_dim \
    --subsample "$subsample" \
    --num-epochs $num_epochs --num-jobs-initial $num_jobs_initial --num-jobs-final $num_jobs_final \
    --num-chunk-per-minibatch $num_chunk_per_minibatch \
    --samples-per-iter $samples_per_iter \
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
    --chunk-width $chunk_width \
    --chunk-left-context $chunk_left_context \
    --norm-based-clipping $norm_based_clipping \
    --egs-dir "$common_egs_dir" \
    --remove-egs $remove_egs \
    data/$mic/${train_set}_hires data/lang $ali_dir $dir  || exit 1;
fi

if [ $stage -le 9 ]; then
  if [ -z $extra_left_context ]; then
    extra_left_context=$chunk_left_context
  fi
  if [ -z $frames_per_chunk ]; then
    frames_per_chunk=$chunk_width
  fi
  for decode_set in dev eval; do
      (
      num_jobs=`cat data/$mic/${decode_set}_hires/utt2spk|cut -d' ' -f2|sort -u|wc -l`
      decode_dir=${dir}/decode_${decode_set}
      if [ "$use_ivectors" == "true" ]; then
        ivector_opts=" --online-ivector-dir exp/$mic/nnet3/ivectors_${decode_set} "
      else
        ivector_opts=
      fi
      steps/nnet3/lstm/decode.sh --nj 250 --cmd "$decode_cmd" $ivector_opts \
          --extra-left-context $extra_left_context  \
          --frames-per-chunk "$frames_per_chunk" \
         $graph_dir data/$mic/${decode_set}_hires $decode_dir || exit 1;
      ) &
  done
fi
wait;
exit 0;
