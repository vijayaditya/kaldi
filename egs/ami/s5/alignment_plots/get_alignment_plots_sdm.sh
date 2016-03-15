report_dir=$1
ali_dirs=("../exp/sdm1/tri4a_sdm1_train_sp_ali/" "../exp/ihm/tri4a_sdm1_cleanali_train_parallel_sp_ali/");
ali_names=('SDM' 'IHM')
wav_dirs=('../data/sdm1/train_sp/' '../data/sdm1_cleanali/train_parallel_sp/')


num_alis=${#ali_dirs[@]}

false && {
rm -rf $report_dir
mkdir -p $report_dir
for i in `seq 1 $num_alis`; do
  i=`echo $i-1|bc`
  ali_dir=${ali_dirs[$i]}
  ali_name=${ali_names[$i]}
  echo $i $ali_dir $ali_name

  ali-to-phones --write-lengths $ali_dir/final.mdl ark:"gunzip -c $ali_dir/ali.*.gz |" ark,t:$report_dir/$ali_name.txt
done
}
args=
for i in `seq 1 $num_alis`; do
  i=`echo $i-1|bc`
  ali_name=${ali_names[$i]}
  wav_dir=${wav_dirs[$i]}
  args="$args --key $ali_name --ali-file $report_dir/$ali_name.txt --wav-dir $wav_dir "
done
echo $args
./generate_sound_plots.py  --utt-name-file 3plot_utt  --text-file ${wav_dirs[0]}/text  $args $report_dir
#./generate_sound_plots.py --max-plots 30 --text-file ${wav_dirs[0]}/text  $args $report_dir
