#!/bin/bash
# Copyright Johns Hopkins University (Author: Daniel Povey, Vijayaditya Peddinti) 2016.  Apache 2.0.
# This script generates the ctm files for dev_aspire, test_aspire and eval_aspire
# for scoring with ASpIRE scoring server.
# It also provides the WER for dev_aspire data.

iter=final
mfccdir=mfcc_reverb_submission
stage=0
decode_num_jobs=200
num_jobs=30
LMWT=12
word_ins_penalty=0.0
min_lmwt=1 # for chain models we will span lmwts on either side of 10
max_lmwt=20
word_ins_penalties=0.0,0.25,0.5,0.75,1.0
decode_mbr=true
lattice_beam=8
ctm_beam=6
max_count=100 # parameter for extract_ivectors.sh
sub_speaker_frames=1500
overlap=5
window=30
affix=
ivector_scale=1.0
pad_frames=0  # this did not seem to be helpful but leaving it as an option.
tune_hyper=true
pass2_decode_opts=
filter_ctm=true
weights_file=
silence_weight=0.00001

extra_left_context=0
extra_right_context=0
frames_per_chunk=50


. ./cmd.sh
[ -f ./path.sh ] && . ./path.sh
. utils/parse_options.sh || exit 1;

if [ $# -ne 3 ]; then
  echo "Usage: $0 [options] <data-dir> <lang-dir> <model-dir>"
  echo " Options:"
  echo "    --stage (0|1|2)                 # start scoring script from part-way through."
  echo "e.g.:"
  echo "$0 data/train data/lang exp/nnet3/tdnn"
  exit 1;
fi

data_dir=$1 #select from {dev_aspire, test_aspire, eval_aspire}
lang=$2 # data/lang
dir=$3 # exp/nnet3/tdnn

model_affix=`basename $dir`
ivector_dir=exp/nnet3
ivector_affix=${affix:+_$affix}_chain_${model_affix}_iter$iter
affix=_${affix}_iter${iter}
act_data_dir=${data_dir}


if [ "$data_dir" == "test_aspire" ]; then
  out_file=single_dev_test${affix}_$model_affix.ctm
elif [ "$data_dir" == "eval_aspire" ]; then
  out_file=single_eval${affix}_$model_affix.ctm
else
  data_dir=${data_dir}_whole
  out_file=single_dev${affix}_${model_affix}.ctm
fi




if [ $stage -le 4 ]; then
  echo "Extracting i-vectors, stage 1"
  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 20 \
    --max-count $max_count \
    data/${segmented_data_dir}_hires $ivector_dir/extractor \
    $ivector_dir/ivectors_${segmented_data_dir}${ivector_affix}_stage1 || exit 1;
fi
if [ $ivector_scale != 1.0 ] && [ $ivector_scale != 1 ]; then
  ivector_scale_affix=_scale$ivector_scale
else
  ivector_scale_affix=
fi

if [ $stage -le 5 ]; then
  if [ "$ivector_scale_affix" != "" ]; then
    echo "$0: Scaling iVectors, stage 1"
    srcdir=$ivector_dir/ivectors_${segmented_data_dir}${ivector_affix}_stage1
    outdir=$ivector_dir/ivectors_${segmented_data_dir}${ivector_affix}${ivector_scale_affix}_stage1
    mkdir -p $outdir
    copy-matrix --scale=$ivector_scale scp:$srcdir/ivector_online.scp ark:- | \
      copy-feats --compress=true ark:-  ark,scp:$outdir/ivector_online.ark,$outdir/ivector_online.scp || exit 1;
    cp $srcdir/ivector_period $outdir/ivector_period
  fi
fi

decode_dir=$dir/decode_${segmented_data_dir}${affix}_pp
# generate the lattices
if [ $stage -le 6 ]; then
  echo "Generating lattices, stage 1"
  steps/nnet3/decode.sh --nj $decode_num_jobs --cmd "$decode_cmd" --config conf/decode.config \
    --acwt 1.0 --post-decode-acwt 10.0 \
    --extra-left-context $extra_left_context  \
    --extra-right-context $extra_right_context  \
    --frames-per-chunk "$frames_per_chunk" \
    --online-ivector-dir $ivector_dir/ivectors_${segmented_data_dir}${ivector_affix}${ivector_scale_affix}_stage1 \
    --skip-scoring true --iter $iter \
    $dir/graph_pp data/${segmented_data_dir}_hires ${decode_dir}_stage1 || exit 1;
fi

if [ $stage -le 7 ]; then
  echo "$0: generating CTM from stage-1 lattices"
  local/multi_condition/get_ctm_conf.sh --cmd "$decode_cmd" \
    --use-segments false --iter $iter \
    data/${segmented_data_dir}_hires \
    ${lang} \
    ${decode_dir}_stage1 || exit 1;
fi

if [ $stage -le 8 ]; then
  if $filter_ctm; then
    if [ ! -z $weights_file ]; then
      echo "$0: Using provided weights file $weights_file"
      ivector_extractor_input=$weights_file
    else
      ctm=${decode_dir}_stage1/score_10/${segmented_data_dir}_hires.ctm
      echo "$0: generating weights file from stage-1 ctm $ctm"

      feat-to-len scp:data/${segmented_data_dir}_hires/feats.scp ark,t:- >${decode_dir}_stage1/utt.lengths.$affix
      if [ ! -f $ctm ]; then  echo "$0: stage 8: expected ctm to exist: $ctm"; exit 1; fi
      cat $ctm | awk '$6 == 1.0 && $4 < 1.0' | \
      grep -v -w mm | grep -v -w mhm | grep -v -F '[noise]' | \
      grep -v -F '[laughter]' | grep -v -F '<unk>' | \
      perl -e ' $lengths=shift @ARGV;  $pad_frames=shift @ARGV; $silence_weight=shift @ARGV;
       $pad_frames >= 0 || die "bad pad-frames value $pad_frames";
       open(L, "<$lengths") || die "opening lengths file";
       @all_utts = ();
       $utt2ref = { };
       while (<L>) {
         ($utt, $len) = split(" ", $_);
         push @all_utts, $utt;
         $array_ref = [ ];
         for ($n = 0; $n < $len; $n++) { ${$array_ref}[$n] = $silence_weight; }
         $utt2ref{$utt} = $array_ref;
       }
       while (<STDIN>) {
         @A = split(" ", $_);
         @A == 6 || die "bad ctm line $_";
         $utt = $A[0]; $beg = $A[2]; $len = $A[3];
         $beg_int = int($beg * 100) - $pad_frames;
         $len_int = int($len * 100) + 2*$pad_frames;
         $array_ref = $utt2ref{$utt};
         !defined $array_ref  && die "No length info for utterance $utt";
         for ($t = $beg_int; $t < $beg_int + $len_int; $t++) {
           if ($t >= 0 && $t < @$array_ref) {
             ${$array_ref}[$t] = 1;
            }
          }
        }
        foreach $utt (@all_utts) {  $array_ref = $utt2ref{$utt};
          print $utt, " [ ", join(" ", @$array_ref), " ]\n";
          } ' ${decode_dir}_stage1/utt.lengths.$affix $pad_frames $silence_weight   | gzip -c >${decode_dir}_stage1/weights${affix}.gz
          ivector_extractor_input=${decode_dir}_stage1/weights${affix}.gz
        fi
      else
        ivector_extractor_input=${decode_dir}_stage1
      fi
fi

if [ $stage -le 8 ]; then
  echo "Extracting i-vectors, stage 2 with input $ivector_extractor_input"
  # this does offline decoding, except we estimate the iVectors per
  # speaker, excluding silence (based on alignments from a GMM decoding), with a
  # different script.  This is just to demonstrate that script.
  # the --sub-speaker-frames is optional; if provided, it will divide each speaker
  # up into "sub-speakers" of at least that many frames... can be useful if
  # acoustic conditions drift over time within the speaker's data.
  steps/online/nnet2/extract_ivectors.sh --cmd "$train_cmd" --nj 20 \
    --silence-weight $silence_weight \
    --sub-speaker-frames $sub_speaker_frames --max-count $max_count \
    data/${segmented_data_dir}_hires $lang $ivector_dir/extractor \
    $ivector_extractor_input $ivector_dir/ivectors_${segmented_data_dir}${ivector_affix} || exit 1;
fi

if [ $stage -le 9 ]; then
  echo "Generating lattices, stage 2 with --acwt $acwt"
  rm -f ${decode_dir}_tg/.error
  steps/nnet3/decode.sh --nj $decode_num_jobs --cmd "$decode_cmd" --config conf/decode.config $pass2_decode_opts \
      --acwt 1.0 --post-decode-acwt 10.0 \
      --extra-left-context $extra_left_context  \
      --extra-right-context $extra_right_context  \
      --frames-per-chunk "$frames_per_chunk" \
      --skip-scoring true --iter $iter --lattice-beam $lattice_beam \
      --online-ivector-dir $ivector_dir/ivectors_${segmented_data_dir}${ivector_affix} \
     $dir/graph_pp data/${segmented_data_dir}_hires ${decode_dir}_tg || touch ${decode_dir}_tg/.error
  [ -f ${decode_dir}_tg/.error ] && echo "$0: Error decoding" && exit 1;
fi

if [ $stage -le 10 ]; then
  echo "Rescoring lattices"
  steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
    --skip-scoring true \
    ${lang}_pp_test{,_fg} data/${segmented_data_dir}_hires \
    ${decode_dir}_{tg,fg} || exit 1;
fi

decode_dir=${decode_dir}_fg
if [ -z $iter ]; then
  model=$decode_dir/../final.mdl # assume model one level up from decoding dir.
else
  model=$decode_dir/../$iter.mdl
fi

if [ $stage -le 11 ]; then
  # tune the LMWT and WIP
  # make command for filtering the ctms
  local/score_aspire.sh --cmd "$decode_cmd" \
    --min-lmwt $min_lmwt --max-lmwt $max_lmwt \
    --word-ins-penalty $word_ins_penalty \
    --word-ins-penalties "$word_ins_penalties" \
    --ctm-beam $ctm_beam \
    --decode-mbr $decode_mbr \
    --window $window \
    --overlap $overlap \
    $decode_dir $act_data_dir $segmented_data_dir
fi
