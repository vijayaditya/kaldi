exp=$1

if [ $exp -eq 1 ]; then
dir_name=exp/chain/tdnn_v1_trial1_sp/
mkdir -p $dir_name
for f in 0.trans_mdl cmvn_opts frame_subsampling_factor normalization.fst phone_lm.fst den.fst tree; do
  cp exp/chain/tdnn_2o_sp/$f $dir_name
done

 local/chain/tdnn/run_tdnn_v1.sh --affix trial1 \
                                 --stage 12 \
                                 --train-stage -5 \
                                 --common-egs-dir exp/chain/tdnn_2o_sp/egs
fi

if [ $exp -eq 2 ]; then
#  had to reduce the batch size as there were memory issues
# models up to iteration 216 cannot be read anymore as the Read.WRite methods changed
# there are more issues, I am just restarting the experiment
dir_name=exp/chain/tdnn_v1_trial2_sp/
mkdir -p $dir_name
for f in 0.trans_mdl cmvn_opts frame_subsampling_factor normalization.fst phone_lm.fst den.fst tree; do
  cp exp/chain/tdnn_2o_sp/$f $dir_name
done

 local/chain/tdnn/run_tdnn_v1.sh --affix trial2 \
                                 --stage 12 \
                                 --train-stage 216 \
                                 --minibatch-size 64 \
                                 --pool-type 'low-pass' \
                                 --pool-lpfilter-width "0.333" \
                                 --pool-window 7
fi


if [ $exp -eq 3 ]; then
  # same as trial1 but with smaller mini-batch size to be used as a control for trial2
dir_name=exp/chain/tdnn_v1_trial3_sp/
mkdir -p $dir_name
for f in 0.trans_mdl cmvn_opts frame_subsampling_factor normalization.fst phone_lm.fst den.fst tree; do
  cp exp/chain/tdnn_v1_trial1_sp/$f $dir_name
done

 local/chain/tdnn/run_tdnn_v1.sh --affix trial3 \
                                 --stage 12 \
                                 --train-stage 298 \
                                 --minibatch-size 64 \
                                 --common-egs-dir exp/chain/tdnn_v1_trial2_sp/egs
fi

if [ $exp -eq 4 ]; then
  # same as trial2 but with updatable convolution layers
dir_name=exp/chain/tdnn_v1_trial4_sp/
mkdir -p $dir_name
for f in 0.trans_mdl cmvn_opts frame_subsampling_factor normalization.fst phone_lm.fst den.fst tree; do
  cp exp/chain/tdnn_v1_trial1_sp/$f $dir_name
done

 local/chain/tdnn/run_tdnn_v1.sh --affix trial4 \
                                 --stage 12 \
                                 --train-stage 469 \
                                 --minibatch-size 64 \
                                 --pool-type 'weighted-average' \
                                 --pool-lpfilter-width "0.333" \
                                 --pool-window 7 \
                                 --common-egs-dir exp/chain/tdnn_v1_trial2_sp/egs
fi


if [ $exp -eq 5 ]; then
  # this is trial2 just restarted
dir_name=exp/chain/tdnn_v1_trial5_sp/
mkdir -p $dir_name
for f in 0.trans_mdl cmvn_opts frame_subsampling_factor normalization.fst phone_lm.fst den.fst tree; do
  cp exp/chain/tdnn_v1_trial1_sp/$f $dir_name
done

 local/chain/tdnn/run_tdnn_v1.sh --affix trial5 \
                                 --stage 12 \
                                 --train-stage 182 \
                                 --minibatch-size 64 \
                                 --pool-type 'low-pass' \
                                 --pool-lpfilter-width "0.333" \
                                 --pool-window 7 \
                                 --common-egs-dir exp/chain/tdnn_v1_trial2_sp/egs
fi


if [ $exp -eq 6 ]; then
  # same as trial2 but with per-dim affine component
dir_name=exp/chain/tdnn_v1_trial6_sp/
mkdir -p $dir_name
for f in 0.trans_mdl cmvn_opts frame_subsampling_factor normalization.fst phone_lm.fst den.fst tree; do
  cp exp/chain/tdnn_v1_trial1_sp/$f $dir_name
done

 local/chain/tdnn/run_tdnn_v1.sh --affix trial6 \
                                 --stage 12 \
                                 --train-stage 323 \
                                 --minibatch-size 64 \
                                 --pool-type 'per-dim-weighted-average' \
                                 --pool-window 7 \
                                 --common-egs-dir exp/chain/tdnn_v1_trial2_sp/egs
fi


if [ $exp -eq 7 ]; then
  # same as trial2 but with per-dim affine component
dir_name=exp/chain/tdnn_v1_trial7_sp/
mkdir -p $dir_name
for f in 0.trans_mdl cmvn_opts frame_subsampling_factor normalization.fst phone_lm.fst den.fst tree; do
  cp exp/chain/tdnn_v1_trial1_sp/$f $dir_name
done

 local/chain/tdnn/run_tdnn_v1.sh --affix trial7 \
                                 --stage 12 \
                                 --train-stage -5 \
                                 --splice-indexes "-2,-1,0,1,2 -1,0,1 -1,0,1 -1,0,1 -1,0,1 -1,0,1 -1,0,1 -1,0,1 -1,0,1 -1,0,1 -1,0,1 -1,0,1 -1,0 -1,0 -1,0 -1,0 -1,0" \
                                 --relu-dim 450 \
                                 --minibatch-size 64 \
                                 --common-egs-dir exp/chain/tdnn_v1_trial2_sp/egs
fi


if [ $exp -eq 8 ]; then
  # same as trial2 but with updatable convolution layers
dir_name=exp/chain/tdnn_v1_trial8_sp/
mkdir -p $dir_name
for f in 0.trans_mdl cmvn_opts frame_subsampling_factor normalization.fst phone_lm.fst den.fst tree; do
  cp exp/chain/tdnn_v1_trial1_sp/$f $dir_name
done

 local/chain/tdnn/run_tdnn_v2.sh --affix trial8 \
                                 --stage 12 \
                                 --train-stage -5 \
                                 --minibatch-size 64 \
                                 --pool-type 'weighted-average' \
                                 --pool-lpfilter-width "0.333" \
                                 --pool-window 7 \
                                 --common-egs-dir exp/chain/tdnn_v1_trial2_sp/egs
fi

if [ $exp -eq 9 ]; then
  # same as trial2 but with updatable convolution layers
dir_name=exp/chain/tdnn_v2_trial1_sp/
mkdir -p $dir_name
for f in 0.trans_mdl cmvn_opts frame_subsampling_factor normalization.fst phone_lm.fst den.fst tree; do
  cp exp/chain/tdnn_v1_trial1_sp/$f $dir_name
done

 local/chain/tdnn/run_tdnn_v2.sh --affix trial1 \
                                 --stage 12 \
                                 --train-stage -5 \
                                 --minibatch-size 64 \
                                 --pool-type 'low-pass' \
                                 --pool-lpfilter-width "0.333" \
                                 --pool-window 7 \
                                 --common-egs-dir exp/chain/tdnn_v1_trial2_sp/egs
fi

if [ $exp -eq 10 ]; then
dir_name=exp/chain/tdnn_v2_trial2_sp/
mkdir -p $dir_name
for f in 0.trans_mdl cmvn_opts frame_subsampling_factor normalization.fst phone_lm.fst den.fst tree; do
  cp exp/chain/tdnn_v1_trial1_sp/$f $dir_name
done

 local/chain/tdnn/run_tdnn_v2.sh --affix trial2 \
                                 --stage 12 \
                                 --train-stage -5 \
                                 --minibatch-size 64 \
                                 --pool-type 'per-dim-weighted-average' \
                                 --pool-window 7 \
                                 --common-egs-dir exp/chain/tdnn_v1_trial2_sp/egs
fi

if [ $exp -eq 11 ]; then
dir_name=exp/chain/tdnn_v2_trial3_sp/
mkdir -p $dir_name
for f in 0.trans_mdl cmvn_opts frame_subsampling_factor normalization.fst phone_lm.fst den.fst tree; do
  cp exp/chain/tdnn_v1_trial1_sp/$f $dir_name
done

 local/chain/tdnn/run_tdnn_v2.sh --affix trial3 \
                                 --stage 12 \
                                 --train-stage -5 \
                                 --relu-dim 500 \
                                 --minibatch-size 64 \
                                 --pool-type 'weighted-average' \
                                 --pool-lpfilter-width "0.333" \
                                 --pool-window 7 \
                                 --common-egs-dir exp/chain/tdnn_v1_trial2_sp/egs
fi


if [ $exp -eq 12 ]; then
dir_name=exp/chain/tdnn_v2_trial4_sp/
mkdir -p $dir_name
for f in 0.trans_mdl cmvn_opts frame_subsampling_factor normalization.fst phone_lm.fst den.fst tree; do
  cp exp/chain/tdnn_v1_trial1_sp/$f $dir_name
done

 local/chain/tdnn/run_tdnn_v2.sh --affix trial4 \
                                 --stage 12 \
                                 --train-stage -5 \
                                 --relu-dim 500 \
                                 --minibatch-size 64 \
                                 --pool-type 'low-pass' \
                                 --pool-lpfilter-width "0.333" \
                                 --pool-window 7 \
                                 --common-egs-dir exp/chain/tdnn_v1_trial2_sp/egs
fi

if [ $exp -eq 13 ]; then
dir_name=exp/chain/tdnn_v2_trial5_sp/
mkdir -p $dir_name
for f in 0.trans_mdl cmvn_opts frame_subsampling_factor normalization.fst phone_lm.fst den.fst tree; do
  cp exp/chain/tdnn_v1_trial1_sp/$f $dir_name
done

 local/chain/tdnn/run_tdnn_v2.sh --affix trial5 \
                                 --stage 12 \
                                 --train-stage -5 \
                                 --relu-dim 500 \
                                 --minibatch-size 64 \
                                 --pool-type 'per-dim-weighted-average' \
                                 --pool-window 7 \
                                 --common-egs-dir exp/chain/tdnn_v1_trial2_sp/egs
fi

if [ $exp -eq 14 ]; then
dir_name=exp/chain/tdnn_v3_trial1_sp/
mkdir -p $dir_name
for f in 0.trans_mdl cmvn_opts frame_subsampling_factor normalization.fst phone_lm.fst den.fst tree; do
  cp exp/chain/tdnn_v1_trial1_sp/$f $dir_name
done

 local/chain/tdnn/run_tdnn_v3.sh --affix trial1 \
                                 --stage 12 \
                                 --train-stage -1 \
                                 --minibatch-size 64 \
                                 --pool-type 'per-dim-weighted-average' \
                                 --pool-window 7 \
                                 --common-egs-dir exp/chain/tdnn_v1_trial2_sp/egs
fi

if [ $exp -eq 15 ]; then
dir_name=exp/chain/tdnn_v4_trial1_sp/
mkdir -p $dir_name
for f in 0.trans_mdl cmvn_opts frame_subsampling_factor normalization.fst phone_lm.fst den.fst tree; do
  cp exp/chain/tdnn_v1_trial1_sp/$f $dir_name
done

 local/chain/tdnn/run_tdnn_v4.sh --affix trial1 \
                                 --stage 12 \
                                 --train-stage 116 \
                                 --minibatch-size 64 \
                                 --pool-type 'per-dim-weighted-average' \
                                 --pool-window 7 \
                                 --common-egs-dir exp/chain/tdnn_v1_trial2_sp/egs
fi

if [ $exp -eq 16 ]; then
dir_name=exp/chain/tdnn_v4_trial2_sp/
mkdir -p $dir_name
for f in 0.trans_mdl cmvn_opts frame_subsampling_factor normalization.fst phone_lm.fst den.fst tree; do
  cp exp/chain/tdnn_v1_trial1_sp/$f $dir_name
done

 local/chain/tdnn/run_tdnn_v4.sh --affix trial2 \
                                 --stage 12 \
                                 --train-stage -5 \
                                 --minibatch-size 64 \
                                 --pool-type 'per-dim-weighted-average' \
                                 --pool-window 7 \
                                 --self-repair-scale "" \
                                 --common-egs-dir exp/chain/tdnn_v1_trial2_sp/egs
fi


if [ $exp -eq 17 ]; then
  # this is very similar to v3_trial1 as expected, so discontinuing this was
  # similar to v4, except for HMM leaky coefficient reducing hmm leaky
  # coefficient to 1e-5, brings the training progress back to before which
  # causes a lot of undertraining

dir_name=exp/chain/tdnn_v5_trial1_sp/
mkdir -p $dir_name
for f in 0.trans_mdl cmvn_opts frame_subsampling_factor normalization.fst phone_lm.fst den.fst tree; do
  cp exp/chain/tdnn_v1_trial1_sp/$f $dir_name
done

 local/chain/tdnn/run_tdnn_v5.sh --affix trial1 \
                                 --stage 12 \
                                 --train-stage -15 \
                                 --minibatch-size 64 \
                                 --pool-type 'per-dim-weighted-average' \
                                 --pool-window 7 \
                                 --common-egs-dir exp/chain/tdnn_v1_trial2_sp/egs
fi


if [ $exp -eq 18 ]; then
dir_name=exp/chain/tdnn_v5_mdwa_sp/
mkdir -p $dir_name
for f in 0.trans_mdl cmvn_opts frame_subsampling_factor normalization.fst phone_lm.fst den.fst tree; do
  cp exp/chain/tdnn_v1_trial1_sp/$f $dir_name
done

 local/chain/tdnn/run_tdnn_v5.sh --affix mdwa \
                                 --stage 12 \
                                 --train-stage 0 \
                                 --minibatch-size 64 \
                                 --pool-type 'multi-dim-weighted-average' \
                                 --pool-window 7 \
                                 --common-egs-dir exp/chain/tdnn_v1_trial2_sp/egs
fi


if [ $exp -eq 19 ]; then
dir_name=exp/chain/tdnn_v5_mdwa_sp/
mkdir -p $dir_name
for f in 0.trans_mdl cmvn_opts frame_subsampling_factor normalization.fst phone_lm.fst den.fst tree; do
  cp exp/chain/tdnn_v1_trial1_sp/$f $dir_name
done

 local/chain/tdnn/run_tdnn_v5.sh --affix mdwa \
                                 --stage 12 \
                                 --train-stage -15 \
                                 --minibatch-size 64 \
                                 --pool-type 'none' \
                                 --pool-window 7 \
                                 --common-egs-dir exp/chain/tdnn_v1_trial2_sp/egs
fi
