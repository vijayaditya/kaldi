#!/bin/bash

set -e

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

dir=$1
mic=$2

final_lm=`cat data/local/lm/final_lm`
LM=$final_lm.pr1-7
graph_dir=$dir/graph_${LM}

#for decode_set in dev eval; do

for decode_set in dev; do
    (
    num_jobs=`cat data/$mic/${decode_set}_hires/utt2spk|cut -d' ' -f2|sort -u|wc -l`
    decode_dir=$dir/decode_$decode_set

    cp -rv $decode_dir ${decode_dir}_nospkovlp
    local/score_asclite.sh --overlap-spk 1 \
        --cmd "$decode_cmd" \
        --asclite true \
       data/$mic/${decode_set}_hires $graph_dir ${decode_dir}_nospkovlp || exit 1;
    ) &
done
wait;
exit 0;
