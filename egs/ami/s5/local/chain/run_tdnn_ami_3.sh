#!/bin/bash

#adapted from swbd's local/chain/6z.sh script. We change the TDNN config
# but with larger tolerance(10 vs 5) for chain
set -e

# configs for 'chain'
stage=10
train_stage=-10
get_egs_stage=-10
mic=ihm
use_ihm_ali=false
affix=
speed_perturb=true
common_egs_dir=
splice_indexes="-1,0,1 -1,0,1,2 -3,0,3 -3,0,3 -3,0,3 -6,-3,0 0"
subset_dim=0
relu_dim=450

# TDNN options
# this script uses the new tdnn config generator so it needs a final 0 to reflect that the final layer input has no splicing
# smoothing options
pool_window=
pool_type='none'
pool_lpfilter_width=
self_repair_scale=0.00001

# training options
num_epochs=4
initial_effective_lrate=0.001
final_effective_lrate=0.0001
leftmost_questions_truncate=-1
max_param_change=2.0
final_layer_normalize_target=1.0
num_jobs_initial=2
num_jobs_final=12
minibatch_size=128
frames_per_eg=150
remove_egs=true
xent_regularize=0.1
max_wer=
min_seg_len=

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

# if we are using the speed-perturbed data we need to generate
# alignments for it.
local/nnet3/run_ivector_common.sh --stage $stage \
                                  --mic $mic \
                                  --use-ihm-ali $use_ihm_ali \
                                  --use-sat-alignments true || exit 1;


gmm=tri4a
if [ $use_ihm_ali == "true" ]; then
  gmm_dir=exp/ihm/$gmm
  mic=${mic}_cleanali
  ali_dir=${gmm_dir}_${mic}_train_parallel_sp_ali
  lat_dir=${gmm_dir}_${mic}_train_parallel_sp_lats
else
  gmm_dir=exp/$mic/$gmm
  ali_dir=${gmm_dir}_${mic}_train_sp_ali
  lat_dir=${gmm_dir}_${mic}_train_sp_lats
fi




dir=exp/$mic/chain/tdnn${affix:+_$affix} # Note: _sp will get added to this if $speed_perturb == true.
dir=${dir}_sp


treedir=exp/$mic/chain/tri5_2y_tree_sp${affix:+_$affix}
lang=data/$mic/lang_chain_2y${affix:+_$affix}

final_lm=`cat data/local/lm/final_lm`
LM=$final_lm.pr1-7
graph_dir=$dir/graph_${LM}

train_set=train_sp
latgen_train_set=train_sp
if [ $use_ihm_ali == "true" ]; then
  latgen_train_set=train_parallel_sp
fi

if [ ! -z $min_seg_len ]; then
  # combining the segments in training data to have a minimum length
  if [ $stage -le 10 ]; then
    steps/cleanup/combine_short_segments.sh $min_seg_len data/$mic/${train_set}_hires data/$mic/${train_set}_min${min_seg_len}_hires
    #extract ivectors for the new data
    steps/online/nnet2/copy_data_dir.sh --utts-per-spk-max 2 \
      data/$mic/${train_set}_min${min_seg_len}_hires data/$mic/${train_set}_min${min_seg_len}_hires_max2
    steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 30 \
      data/$mic/${train_set}_min${min_seg_len}_hires_max2 \
      exp/$mic/nnet3/extractor \
      exp/$mic/nnet3/ivectors_${train_set}_min${min_seg_len}_hires || exit 1;
   # combine the non-hires features for alignments/lattices
   steps/cleanup/combine_short_segments.sh $min_seg_len data/$mic/${latgen_train_set} data/$mic/${latgen_train_set}_min${min_seg_len}
  fi
  train_set=${train_set}_min${min_seg_len}
  latgen_train_set=${latgen_train_set}_min${min_seg_len}

  if [ $stage -le 11 ]; then
    # realigning data as the segments would have changed
    steps/align_fmllr.sh --nj 100 --cmd "$train_cmd" data/$mic/$latgen_train_set data/lang $gmm_dir ${ali_dir}_min${min_seg_len} || exit 1;
  fi
  ali_dir=${ali_dir}_min${min_seg_len}
  lat_dir=${lat_dir}_min${min_seg_len}
fi

if [ $stage -le 12 ]; then
  # Get the alignments as lattices (gives the chain training more freedom).
  # use the same num-jobs as the alignments
  steps/align_fmllr_lats.sh --nj 100 --cmd "$train_cmd" data/$mic/$latgen_train_set \
    data/lang $gmm_dir $lat_dir
  rm $lat_dir/fsts.*.gz # save space
fi


if [ $stage -le 13 ]; then
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

if [ $stage -le 14 ]; then
  # Build a tree using our new topology.
  steps/nnet3/chain/build_tree.sh --frame-subsampling-factor 3 \
      --leftmost-questions-truncate $leftmost_questions_truncate \
      --cmd "$train_cmd" 4200 data/$mic/$latgen_train_set $lang $ali_dir $treedir
fi

mkdir -p $dir
train_data_dir=data/$mic/${train_set}_hires
if [ ! -z $max_wer ]; then
  if [ $stage -le 15 ]; then
    bad_utts_dir=${gmm_dir}_${train_set}_bad_utts
    steps/cleanup/find_bad_utts.sh --cmd "$decode_cmd" --nj 300 data/$mic/${train_set} data/lang $ali_dir $bad_utts_dir
    python local/sort_bad_utts.py --bad-utt-info-file $bad_utts_dir/all_info.sorted.txt --max-wer $max_wer --output-file $dir/wer_sorted_utts_${max_wer}wer
    utils/copy_data_dir.sh data/$mic/${train_set}_hires data/$mic/${train_set}_${max_wer}wer_hires
    utils/filter_scp.pl $dir/wer_sorted_utts_${max_wer}wer data/sdm1/${train_set}_hires/feats.scp  > data/$mic/${train_set}_${max_wer}wer_hires/feats.scp
    utils/fix_data_dir.sh data/$mic/${train_set}_${max_wer}wer_hires
  fi
  train_data_dir=data/$mic/${train_set}_${max_wer}wer_hires
fi

if [ $stage -le 16 ]; then
  echo "$0: creating neural net configs";

  if [ ! -z "$relu_dim" ]; then
    dim_opts="--relu-dim $relu_dim"
  else
    dim_opts="--pnorm-input-dim $pnorm_input_dim --pnorm-output-dim  $pnorm_output_dim"
  fi

  # create the config files for nnet initialization
  pool_opts=
  pool_opts=$pool_opts${pool_type:+" --pool-type $pool_type "}
  pool_opts=$pool_opts${pool_window:+" --pool-window $pool_window "}
  pool_opts=$pool_opts${pool_lpfilter_width:+" --pool-lpfilter-width $pool_lpfilter_width "}
  repair_opts=${self_repair_scale:+" --self-repair-scale $self_repair_scale "}

  steps/nnet3/tdnn/make_configs.py $pool_opts \
    $repair_opts \
    --subset-dim "$subset_dim" \
    --feat-dir $train_data_dir \
    --ivector-dir exp/$mic/nnet3/ivectors_train_sp_hires \
    --tree-dir $treedir \
    $dim_opts \
    --splice-indexes "$splice_indexes"  \
    --use-presoftmax-prior-scale false \
    --xent-regularize $xent_regularize \
    --xent-separate-forward-affine true \
    --include-log-softmax false \
    --final-layer-normalize-target $final_layer_normalize_target \
   $dir/configs || exit 1;
fi

if [ $stage -le 17 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{5,6,7,8}/$USER/kaldi-data/egs/ami-$(date +'%m_%d_%H_%M')/s5/$dir/egs/storage $dir/egs/storage
  fi

 touch $dir/egs/.nodelete # keep egs around when that run dies.

 steps/nnet3/chain/train.py --stage $train_stage \
    --cmd "$decode_cmd" \
    --feat.online-ivector-dir exp/$mic/nnet3/ivectors_${train_set}_hires \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --chain.left-tolerance 10 \
    --chain.right-tolerance 10 \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.00005 \
    --chain.apply-deriv-weights false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --egs.dir "$common_egs_dir" \
    --egs.stage $get_egs_stage \
    --egs.opts "--frames-overlap-per-eg 0" \
    --egs.chunk-width $frames_per_eg \
    --trainer.num-chunk-per-minibatch $minibatch_size \
    --trainer.frames-per-iter 1500000 \
    --trainer.num-epochs $num_epochs \
    --trainer.optimization.num-jobs-initial $num_jobs_initial \
    --trainer.optimization.num-jobs-final $num_jobs_final \
    --trainer.optimization.initial-effective-lrate $initial_effective_lrate \
    --trainer.optimization.final-effective-lrate $final_effective_lrate \
    --trainer.max-param-change $max_param_change \
    --cleanup.remove-egs $remove_egs \
    --feat-dir $train_data_dir \
    --tree-dir $treedir \
    --lat-dir $lat_dir \
    --dir $dir  || exit 1;
fi

if [ $stage -le 18 ]; then
  # Note: it might appear that this $lang directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  utils/mkgraph.sh --self-loop-scale 1.0 data/lang_${LM} $dir $graph_dir
fi

if [ $stage -le 19 ]; then
  for decode_set in dev eval; do
      (
      num_jobs=`cat data/$mic/${decode_set}_hires/utt2spk|cut -d' ' -f2|sort -u|wc -l`

      steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
         --extra-left-context 20 \
          --nj $num_jobs --cmd "$decode_cmd" \
          --online-ivector-dir exp/$mic/nnet3/ivectors_${decode_set} \
         $graph_dir data/$mic/${decode_set}_hires $dir/decode_${decode_set} || exit 1;
      ) &
  done
fi
wait;
exit 0;
