#!/bin/bash

# this is a script to train the nnet3 TDNN acoustic model


stage=1
affix=
train_stage=-10
reporting_email=
common_egs_dir=
remove_egs=true
egs_stage=0

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh


if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.  Otherwise, call this script with --use-gpu false
EOF
fi

# do the common parts of the script.
local/lsr/run_ivector_common.sh --stage $stage || exit 1;

ali_dir=exp/tri5a_rvb_ali
dir=exp/nnet3/tdnn
dir=$dir${affix:+_$affix}

if [ $stage -le 7 ]; then
  echo "$0: creating neural net configs";

  # create the config files for nnet initialization
  python steps/nnet3/tdnn/make_configs.py  \
    --feat-dir data/train_rvb_hires \
    --ivector-dir exp/nnet3/ivectors_train \
    --ali-dir $ali_dir \
    --relu-dim 1248 \
    --splice-indexes "-2,-1,0,1,2 -1,2 -3,3 -3,3 -7,2 0"  \
    --use-presoftmax-prior-scale true \
   $dir/configs || exit 1;
fi

if [ $stage -le 8 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{3,4,5,6}/$USER/kaldi-data/egs/aspire-$(date +'%m_%d_%H_%M')/s5/$dir/egs/storage $dir/egs/storage
  fi

  steps/nnet3/train_dnn.py --stage=$train_stage \
    --cmd="$decode_cmd" \
    --feat.online-ivector-dir exp/nnet3/ivectors_train \
    --feat.cmvn-opts="--norm-means=false --norm-vars=false" \
    --trainer.num-epochs 3 \
    --trainer.optimization.num-jobs-initial 4 \
    --trainer.optimization.num-jobs-final 22 \
    --trainer.optimization.initial-effective-lrate 0.0017 \
    --trainer.optimization.final-effective-lrate 0.00017 \
    --egs.dir "$common_egs_dir" \
    --egs.stage "$egs_stage" \
    --cleanup.remove-egs $remove_egs \
    --cleanup.preserve-model-interval 50 \
    --feat-dir=data/train_rvb_hires \
    --ali-dir $ali_dir \
    --lang data/lang \
    --reporting.email="$reporting_email" \
    --dir=$dir  || exit 1;

fi

if [ $stage -le 9 ]; then
  # dump iVectors for the testing data.
  for data_dir in dev_rvb test_rvb dev_aspire dev test; do
    steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 20 \
      data/${data_dir}_hires exp/nnet3/extractor exp/nnet3/ivectors_${data_dir} || exit 1;
  done
fi

# this is just a sample decode of data dirs with segments files
# the actual decodes are done using local/multi_condition/prep_test_aspire.sh script
# see run.sh for these details
graph_dir=exp/tri5a/graph
if [ $stage -le 9 ]; then
  for data_dir in dev_rvb test_rvb dev_aspire dev test; do
    (
    num_jobs=`cat data/${decode_set}_hires/utt2spk|cut -d' ' -f2|sort -u|wc -l`
    steps/nnet3/decode.sh --nj $num_jobs --cmd "$decode_cmd" \
        --online-ivector-dir exp/nnet3/ivectors_${decode_set} \
       $graph_dir data/${decode_set}_hires $dir/decode_${decode_set} || exit 1;
    ) &
  done
fi
wait;


#ASpIRE decodes

# final result
local/nnet3/prep_test_aspire.sh --stage 4 --decode-num-jobs 200  \
 --sub-speaker-frames 6000 --window 10 --overlap 5 --max-count 75 --pass2-decode-opts "--min-active 1000" \
 --ivector-scale 0.75  --tune-hyper true dev_aspire data/lang exp/nnet3/tdnn
# %WER 31.0 | 2120 27217 | 74.8 16.1 9.1 5.9 31.0 77.9 | -0.707 | exp/nnet3/tdnn/decode_dev_aspire_whole_uniformsegmented_win10_over5_v6_200jobs_iterfinal_pp_fg/score_14/penalty_0.0/ctm.filt.filt.sys

exit 0;

# intermediate results
local/nnet3/prep_test_aspire.sh --stage 4 --decode-num-jobs 200 --iter 100  \
   --sub-speaker-frames 6000 --window 10 --overlap 5 --max-count 75 --pass2-decode-opts "--min-active 1000" \
   --ivector-scale 0.75  --tune-hyper true dev_aspire data/lang exp/nnet3/tdnn
#%WER 34.2 | 2120 27212 | 71.6 18.3 10.2 5.8 34.2 80.2 | -0.613 | exp/nnet3/tdnn/decode_dev_aspire_whole_uniformsegmented_win10_over5_v6_200jobs_iter100_pp_fg/score_14/penalty_0.0/ctm.filt.filt.sys

  local/nnet3/prep_test_aspire.sh --stage 4 --decode-num-jobs 200 --iter 200  \
   --sub-speaker-frames 6000 --window 10 --overlap 5 --max-count 75 --pass2-decode-opts "--min-active 1000" \
   --ivector-scale 0.75  --tune-hyper true dev_aspire data/lang exp/nnet3/tdnn
#%WER 32.8 | 2120 27212 | 73.2 17.3 9.4 6.0 32.8 79.3 | -0.657 | exp/nnet3/tdnn/decode_dev_aspire_whole_uniformsegmented_win10_over5_v6_200jobs_iter200_pp_fg/score_13/penalty_0.0/ctm.filt.filt.sys

  local/nnet3/prep_test_aspire.sh --stage 4 --decode-num-jobs 200 --iter 300  \
   --sub-speaker-frames 6000 --window 10 --overlap 5 --max-count 75 --pass2-decode-opts "--min-active 1000" \
   --ivector-scale 0.75  --tune-hyper true dev_aspire data/lang exp/nnet3/tdnn
#%WER 32.3 | 2120 27215 | 73.7 17.1 9.2 6.0 32.3 79.7 | -0.676 | exp/nnet3/tdnn/decode_dev_aspire_whole_uniformsegmented_win10_over5_v6_200jobs_iter300_pp_fg/score_13/penalty_0.0/ctm.filt.filt.sys

  local/nnet3/prep_test_aspire.sh --stage 4 --decode-num-jobs 200 --iter 400  \
   --sub-speaker-frames 6000 --window 10 --overlap 5 --max-count 75 --pass2-decode-opts "--min-active 1000" \
   --ivector-scale 0.75  --tune-hyper true dev_aspire data/lang exp/nnet3/tdnn
#%WER 31.7 | 2120 27215 | 74.3 16.8 8.9 6.0 31.7 78.9 | -0.690 | exp/nnet3/tdnn/decode_dev_aspire_whole_uniformsegmented_win10_over5_v6_200jobs_iter400_pp_fg/score_13/penalty_0.0/ctm.filt.filt.sys

  local/nnet3/prep_test_aspire.sh --stage 4 --decode-num-jobs 200 --iter 500  \
   --sub-speaker-frames 6000 --window 10 --overlap 5 --max-count 75 --pass2-decode-opts "--min-active 1000" \
   --ivector-scale 0.75  --tune-hyper true dev_aspire data/lang exp/nnet3/tdnn
#%WER 31.6 | 2120 27216 | 74.5 16.6 8.8 6.1 31.6 79.7 | -0.723 | exp/nnet3/tdnn/decode_dev_aspire_whole_uniformsegmented_win10_over5_v6_200jobs_iter500_pp_fg/score_13/penalty_0.0/ctm.filt.filt.sys

  local/nnet3/prep_test_aspire.sh --stage 4 --decode-num-jobs 200 --iter 600  \
   --sub-speaker-frames 6000 --window 10 --overlap 5 --max-count 75 --pass2-decode-opts "--min-active 1000" \
   --ivector-scale 0.75  --tune-hyper true dev_aspire data/lang exp/nnet3/tdnn
#%WER 31.3 | 2120 27216 | 74.9 16.6 8.5 6.2 31.3 78.4 | -0.737 | exp/nnet3/tdnn/decode_dev_aspire_whole_uniformsegmented_win10_over5_v6_200jobs_iter600_pp_fg/score_12/penalty_0.0/ctm.filt.filt.sys

  local/nnet3/prep_test_aspire.sh --stage 4 --decode-num-jobs 200 --iter 700  \
   --sub-speaker-frames 6000 --window 10 --overlap 5 --max-count 75 --pass2-decode-opts "--min-active 1000" \
   --ivector-scale 0.75  --tune-hyper true dev_aspire data/lang exp/nnet3/tdnn
#%WER 31.2 | 2120 27216 | 74.7 16.2 9.1 5.9 31.2 79.0 | -0.708 | exp/nnet3/tdnn/decode_dev_aspire_whole_uniformsegmented_win10_over5_v6_200jobs_iter700_pp_fg/score_14/penalty_0.0/ctm.filt.filt.sys

  local/nnet3/prep_test_aspire.sh --stage 4 --decode-num-jobs 200 --iter 800  \
   --sub-speaker-frames 6000 --window 10 --overlap 5 --max-count 75 --pass2-decode-opts "--min-active 1000" \
   --ivector-scale 0.75  --tune-hyper true dev_aspire data/lang exp/nnet3/tdnn
#%WER 31.1 | 2120 27219 | 74.7 16.4 8.9 5.9 31.1 78.4 | -0.732 | exp/nnet3/tdnn/decode_dev_aspire_whole_uniformsegmented_win10_over5_v6_200jobs_iter800_pp_fg/score_12/penalty_0.25/ctm.filt.filt.sys

  local/nnet3/prep_test_aspire.sh --stage 4 --decode-num-jobs 200 --iter 1000  \
   --sub-speaker-frames 6000 --window 10 --overlap 5 --max-count 75 --pass2-decode-opts "--min-active 1000" \
   --ivector-scale 0.75  --tune-hyper true dev_aspire data/lang exp/nnet3/tdnn
#%WER 31.1 | 2120 27220 | 74.9 16.3 8.8 6.0 31.1 78.1 | -0.719 | exp/nnet3/tdnn/decode_dev_aspire_whole_uniformsegmented_win10_over5_v6_200jobs_iter1000_pp_fg/score_13/penalty_0.0/ctm.filt.filt.sys

exit 0;
