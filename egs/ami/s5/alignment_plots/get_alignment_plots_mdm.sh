ali_dirs=("../exp/mdm8/tri4a_mdm8_train_sp_ali/" "../exp/ihm/tri4a_mdm8_cleanali_train_parallel_sp_ali/"  "../exp/mdm8/nnet3/tdnn_sp/train_sp_ali/");
ali_names=('gmm' 'gmm_clean' 'nnet')


num_alis=${#ali_dirs[@]}
false && {
for i in `seq 1 $num_alis`; do
  i=`echo $i-1|bc`
  ali_dir=${ali_dirs[$i]}
  ali_name=${ali_names[$i]}
  echo $i $ali_dir $ali_name

  ali-to-phones --write-lengths $ali_dir/final.mdl ark:"gunzip -c $ali_dir/ali.*.gz |" ark,t:$ali_name.txt
done
}

args=
for i in `seq 1 $num_alis`; do
  i=`echo $i-1|bc`
  ali_name=${ali_names[$i]}
  args="$args --key $ali_name --ali-file $ali_name.txt "
done
echo $args
./generate_plots.py $args report
