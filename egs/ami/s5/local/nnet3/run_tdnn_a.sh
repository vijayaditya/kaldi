#!/bin/bash

# this is the standard "tdnn" system, built in nnet3; it's what we use to
# call multi-splice.

# At this script level we don't support not running on GPU, as it would be painfully slow.
# If you want to run without GPU you'd have to call train_tdnn.sh with --gpu false,
# --num-threads 16 and --minibatch-size 128.
set -e

stage=0
train_stage=-10
mic=ihm
use_sat_alignments=true
exp_name=tdnn_a
affix=
speed_perturb=true
common_egs_dir=
splice_indexes="-1,0,1 -1,0,1,2 -3,0,3 -3,0,3 -3,0,3 -6,-3,0 0"
subset_dim=0
remove_egs=true
relu_dim=700 # to match params with tdnn_sp
use_ihm_ali=false
max_wer=

# TDNN options
# this script uses the new tdnn config generator so it needs a final 0 to reflect that the final layer input has no splicing
# smoothing options
pool_window=
pool_type='none'
pool_lpfilter_width=

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

local/nnet3/run_ivector_common.sh --stage $stage \
                                  --mic $mic \
                                  --use-ihm-ali $use_ihm_ali \
                                  --use-sat-alignments $use_sat_alignments || exit 1;

# we still support this option as all the TDNN, LSTM, BLSTM systems were built
# using tri3a alignments
if [ $use_sat_alignments == "true" ]; then
  gmm=tri4a
else
  gmm=tri3a
fi

if [ $use_ihm_ali == "true" ]; then
  gmm_dir=exp/ihm/$gmm
  mic=${mic}_cleanali
  ali_dir=${gmm_dir}_${mic}_train_parallel_sp_ali
else
  gmm_dir=exp/$mic/$gmm
  ali_dir=${gmm_dir}_${mic}_train_sp_ali
fi

final_lm=`cat data/local/lm/final_lm`
LM=$final_lm.pr1-7
graph_dir=$gmm_dir/graph_${LM}
dir=exp/$mic/nnet3/${exp_name}${speed_perturb:+_sp}${affix:+_$affix}
mkdir -p $dir
train_data_dir=data/$mic/train_sp_hires
if [ ! -z $max_wer ]; then
  if [ $stage -le 10 ]; then
    #steps/cleanup/find_bad_utts.sh --cmd "$decode_cmd" --nj 100 data/$mic/train_sp/ data/lang $ali_dir ${gmm_dir}_bad_utts
    python local/sort_bad_utts.py --bad-utt-info-file ${gmm_dir}_bad_utts/all_info.sorted.txt --max-wer $max_wer --output-file $dir/wer_sorted_utts_${max_wer}wer
    utils/copy_data_dir.sh data/$mic/train_sp_hires data/$mic/train_${max_wer}wer_sp_hires
    utils/filter_scp.pl $dir/wer_sorted_utts_${max_wer}wer data/sdm1/train_sp_hires/feats.scp  > data/$mic/train_${max_wer}wer_sp_hires/feats.scp
    utils/fix_data_dir.sh data/$mic/train_${max_wer}wer_sp_hires
  fi
  train_data_dir=data/$mic/train_${max_wer}wer_sp_hires
fi

if [ $stage -le 11 ]; then
  echo "$0: creating neural net configs";
  if [ ! -z "$relu_dim" ]; then
    dim_opts="--relu-dim $relu_dim"
  else
    dim_opts="--pnorm-input-dim $pnorm_input_dim --pnorm-output-dim  $pnorm_output_dim"
  fi
  repair_opts=${self_repair_scale:+" --self-repair-scale $self_repair_scale "}

  # create the config files for nnet initialization
  pool_opts=
  pool_opts=$pool_opts${pool_type:+" --pool-type $pool_type "}
  pool_opts=$pool_opts${pool_window:+" --pool-window $pool_window "}
  pool_opts=$pool_opts${pool_lpfilter_width:+" --pool-lpfilter-width $pool_lpfilter_width "}

  # create the config files for nnet initialization
  python steps/nnet3/tdnn/make_configs.py  $pool_opts \
    $repair_opts \
    --feat-dir $train_data_dir \
    --ivector-dir exp/$mic/nnet3/ivectors_train_sp_hires \
    --ali-dir $ali_dir \
    $dim_opts \
    --splice-indexes "$splice_indexes"  \
    --use-presoftmax-prior-scale true \
   $dir/configs || exit 1;
fi

if [ $stage -le 12 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b{09,10,11,12}/$USER/kaldi-data/egs/ami-$(date +'%m_%d_%H_%M')/s5/$dir/egs/storage $dir/egs/storage
  fi

  steps/nnet3/train_dnn.py --stage=$train_stage \
    --cmd="$decode_cmd" \
    --feat.online-ivector-dir exp/$mic/nnet3/ivectors_train_sp_hires \
    --feat.cmvn-opts="--norm-means=false --norm-vars=false" \
    --trainer.num-epochs 3 \
    --trainer.optimization.num-jobs-initial 2 \
    --trainer.optimization.num-jobs-final 12 \
    --trainer.optimization.initial-effective-lrate 0.0015 \
    --trainer.optimization.final-effective-lrate 0.00015 \
    --egs.dir "$common_egs_dir" \
    --cleanup.remove-egs $remove_egs \
    --cleanup.preserve-model-interval 100 \
    --use-gpu true \
    --feat-dir=$train_data_dir \
    --ali-dir $ali_dir \
    --lang data/lang \
    --reporting.email="$reporting_email" \
    --dir=$dir  || exit 1;

fi

if [ $stage -le 13 ]; then
  # this version of the decoding treats each utterance separately
  # without carrying forward speaker information.
  for decode_set in dev eval; do
      (
      num_jobs=`cat data/$mic/${decode_set}_hires/utt2spk|cut -d' ' -f2|sort -u|wc -l`
      decode_dir=${dir}/decode_${decode_set}

      steps/nnet3/decode.sh --nj $num_jobs --cmd "$decode_cmd" \
          --online-ivector-dir exp/$mic/nnet3/ivectors_${decode_set} \
         $graph_dir data/$mic/${decode_set}_hires $decode_dir || exit 1;
      ) &
  done
fi
wait;
exit 0;
