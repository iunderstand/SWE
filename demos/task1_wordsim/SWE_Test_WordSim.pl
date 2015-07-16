
use File::Path;
#=============================================#
#     Scripts for Semantic Word Embedding     #
#             ACL-2015, Beijing               #
#---------------------------------------------#
#     Quan Liu, University of Science and     #
#               Technology of China. 2015     #
#       http://home.ustc.edu.cn/~quanliu/     #
#=============================================#

# SWE_Test_WordSim.pl
# Word embedding for word similarity task.
# Task: WordSim-353 task.
# Criterion: Spearman correlation

my $calsim_tool = "../../bin/SWE_Test_WordSim";

my $test_set    = "WordSim353";
my $word_pair   = "testset/SWE.EN.TestSet.$test_set.pair";
my $answer_file = "testset/SWE.EN.TestSet.$test_set.sort";

my $out_model_path = "EmbedVector_TEXT8"; # $ARGV[0];
my @semantic_flag  = ("SA1", "HH1", "COM1");

my $test_result = "result/$out_model_path.$test_set.result";

mkdir "result";

## (ADD)
open SIMRES, ">>$test_result";
my $cur_time = GetCurTime();
print SIMRES "\n====== $test_set\n";
print SIMRES "--- $cur_time\n";

##-----------------------------
my $iter_times = 1;

foreach my $semantic_flag (@semantic_flag)
{
		
	my $sample_num   = 1e-4;
	
	my $out_model    = "$out_model_path/sem$semantic_flag.Inter_run$iter_times.NEG$sample_num";
	if (!-s $out_model) {
		next;
	}
	#mkpath $out_model if !-s $out_model;
	
	## (ADD)
	my $result_dir   = "$out_model/WordSim";
	mkpath $result_dir if !-s $result_dir;
	
	## Network
	my @layer1_size  = (100);
	my $window_size  = 5;
	my $learn_rate   = 0.025;
	my @run_negative = (5);
	
	my @inter_param = (0, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5);	
	my @hinge_margin= (0.0);
	my @delta_right  = (1); # use right part of inequation
	my $delta_left = 1; # use left part of inequation
	my $sem_addtime = 0;
	my $weight_decay= 0.0;
		
	print SIMRES ">>> Semantic: $semantic_flag  Inter_run: $iter_times times\n";
	
	foreach my $layer1_size (@layer1_size) 
	{
	foreach my $run_negative (@run_negative) 
	{	
	foreach my $inter_param (@inter_param) {
		foreach my $hinge_margin (@hinge_margin) {
			foreach my $delta_right (@delta_right) {

			my $save_embeded = "$out_model/wordembed.sem$semantic_flag.dim$layer1_size.win$window_size.neg$run_negative.samp$sample_num.inter$inter_param.hinge$hinge_margin.add$sem_addtime.decay$weight_decay.l$delta_left.r$delta_right.embeded.txt";	
			my $save_runlog  = "$out_model/wordembed.sem$semantic_flag.dim$layer1_size.win$window_size.neg$run_negative.samp$sample_num.inter$inter_param.hinge$hinge_margin.add$sem_addtime.decay$weight_decay.l$delta_left.r$delta_right.logfile.txt";
			
			print "\n>> $save_embeded\n";
			my $model_flag  = "wordembed.sem$semantic_flag.dim$layer1_size.win$window_size.neg$run_negative.samp$sample_num.inter$inter_param.hinge$hinge_margin.add$sem_addtime.decay$weight_decay.l$delta_left.r$delta_right";
			my $calsim_res  = "$result_dir/$model_flag.$test_set.res";
			my $sim_compare = "$result_dir/$model_flag.$test_set.sim";
			
			my ($calcu_num, $total_num, $model_spearman) = CalCuModel($word_pair, $save_embeded, $calsim_res, $sim_compare, $answer_file);
			print "---calcu: ($calcu_num/$total_num), spearman: $model_spearman\n\n";
			printf SIMRES "---model: $model_flag\n";
			printf SIMRES "---calcu: ($calcu_num/$total_num), spearman: $model_spearman\n\n";
	}}}}}
}
close SIMRES;


### sub
sub GetCurTime()
{
	my ($sec, $min, $hour, $mday, $mon, $year, $wday, $yday, $isdst) = localtime(time);
	my $curTime = sprintf("%04d-%02d-%02d--%02d:%02d:%02d", 
					$year+1900, $mon+1, $mday, $hour, $min, $sec);
	return $curTime;
}

### sub
sub CalCuModel()
{
	my ($pair_file, $embed_file, $calsim_res, $sim_compare, $answer_file) = @_;
	
	if (!-s $calsim_res) 
	{
		print "--- $calsim_tool $pair_file $embed_file $calsim_res cosine\n";
		system("$calsim_tool $pair_file $embed_file $calsim_res cosine");
	}
	else{
		print "--- $calsim_res exists\n";
	}
	
	my %sim_label = ();
	my %sim_model = ();
	my %oov_item  = ();
	
	my $calcu_num = 0;
	open RES, "<$calsim_res";
	while (<RES>) {
		if (/^(\S+)\s+(\S+)\s+(\S+)\n/) {
			$sim_model{"$1 $2"} = $3;
			#print "$1 $2 $3\n";
			$calcu_num++;
		}
	}
	close RES; 
	
	my $total_num = 0;
	print "--- answer file: $answer_file\n";
	open ANS, "<$answer_file";
	while (<ANS>) {		
		if (/^(\S+)\s+(\S+)\s+(\S+)/) {
			my $tmp_pair = "$1 $2";
			my $value = $3;
			if (exists $sim_model{$tmp_pair}) {
				$sim_label{$tmp_pair} = $value;
			}
			else{
				$oov_item{$tmp_pair}++;
			}
			$total_num++;
		}
	}
	close ANS; 
	
	my $spearman = SpearmanRank(\%sim_label, \%sim_model);
	
	open CMP, ">$sim_compare";
	foreach my $pair (sort {$sim_model{$b}<=>$sim_model{$a}} keys %sim_model) {
		print CMP "$sim_model{$pair} $sim_label{$pair}\n";
	}
	print "--- not in model:\n";
	foreach my $pair (sort keys %oov_item) {
		print CMP "$pair\n";
	}
	close CMP;
	
	return ($calcu_num, $total_num, $spearman);
}

## 
sub SpearmanRank()
{
	my ($seq_a, $seq_b) = @_;
	my %seq_a = %$seq_a;
	my %seq_b = %$seq_b;
	
	## malloc
	my %rep_num = ();
	my %src_rank= ();
	foreach my $key (sort {$seq_a{$b}<=>$seq_a{$a}} keys %seq_a) {
		my $value = $seq_a{$key};
		$rep_num{$value}++; 
	}
	my $rank_ID = 1;
	foreach my $key (sort {$b<=>$a} keys %rep_num) {
		my $rep_num = $rep_num{$key};
		my $sum_rep = 0;		
		for (my $i = $rank_ID; $i < ($rank_ID+$rep_num); $i++) {
			$sum_rep += $i;
		}
		$sum_rep /= $rep_num; 
		$rank_ID += $rep_num;
		$src_rank{$key} = $sum_rep;
	}
	
	## calculation
	my %org_rank = ();
	my %key_rank = ();
	my $new_rank = 1;
	my $sumDiff = 0.0;
	my $sampNum = 0;
	foreach my $key (sort {$seq_b{$b}<=>$seq_b{$a}} keys %seq_b) {
		my $new_value = $seq_b{$key};
		my $src_value = $seq_a{$key};
		my $src_rank  = $src_rank{$src_value};
		$key_rank{$key} = $new_rank;
		$org_rank{$key} = $src_rank;
		my $diff_squa = ($src_rank - $new_rank)**2;
		#print SPM "$doc $rawRank $asrRank $diff_squa\n";
		$new_rank++;
		$sumDiff+= $diff_squa;
		$sampNum++;
	}
	
	##
	my $spearman_corr = PearsonCorr(\%org_rank, \%key_rank);
	my $spearman_simp = 1-(6*$sumDiff/($sampNum*($sampNum**2-1)));
	print "spearman: $spearman_corr, approximate: $spearman_simp\n";
	return $spearman_corr;
}


## Pearson Coefficient
sub PearsonCorr()
{
	my ($seq_a, $seq_b) = @_;
	my %seq_a = %$seq_a;
	my %seq_b = %$seq_b;
	
	my $sum_a = 0;
	my $sum_a2= 0;
	my $sum_b = 0;
	my $sum_b2= 0;
	my $sum_ab= 0;
	my $ele_Na= 0;
	my $ele_Nb= 0;
	my $ele_N = 0;
	
	##
	foreach my $key (keys %seq_a) 
	{
		$ele_Na++;
		$sum_a += $seq_a{$key};
		$sum_a2+= ($seq_a{$key}**2);
	}
	foreach my $key (keys %seq_b) 
	{
		$ele_Nb++;
		$sum_b += $seq_b{$key};
		$sum_b2+= ($seq_b{$key}**2);
	}	

	if ($ele_Na ne $ele_Nb) {
		print "--- error, the element num is not the same\n";
		return "-999.9";
	}
	else{
		$ele_N = $ele_Na;
		foreach my $key (keys %seq_b) {
			$sum_ab += ($seq_a{$key} * $seq_b{$key});
		}
		my $pearson_fenzi = ($sum_ab-($sum_a*$sum_b/$ele_N));
		my $pearson_fenmu = ( ($sum_a2-($sum_a*$sum_a/$ele_N)) * ($sum_b2-($sum_b*$sum_b/$ele_N)) ) ** 0.5;
		my $pearson_corr  = $pearson_fenzi / $pearson_fenmu;
		return $pearson_corr;
	}
	##
}

## OLD 
sub SpearmanRank_SimpleType()
{
	my ($seq_a, $seq_b) = @_;
	my %seq_a = %$seq_a;
	my %seq_b = %$seq_b;
	
	##
	my %rep_num = ();
	my %src_rank= ();
	foreach my $key (sort {$seq_a{$b}<=>$seq_a{$a}} keys %seq_a) {
		my $value = $seq_a{$key};
		$rep_num{$value}++; 
	}
	my $rank_ID = 1;
	foreach my $key (sort {$b<=>$a} keys %rep_num) {
		my $rep_num = $rep_num{$key};
		my $sum_rep = 0;		
		for (my $i = $rank_ID; $i < ($rank_ID+$rep_num); $i++) {
			$sum_rep += $i;
		}
		$sum_rep /= $rep_num; 
		$rank_ID += $rep_num;
		$src_rank{$key} = $sum_rep;
	}
	
	## calculation
	my $new_rank = 1;
	my $sumDiff = 0.0;
	my $sampNum = 0;
	foreach my $key (sort {$seq_b{$b}<=>$seq_b{$a}} keys %seq_b) {
		my $new_value = $seq_b{$key};
		my $src_value = $seq_a{$key};
		my $src_rank  = $src_rank{$src_value};
		my $diff_squa = ($src_rank - $new_rank)**2;
		#print SPM "$doc $rawRank $asrRank $diff_squa\n";
		$new_rank++;
		$sumDiff+= $diff_squa;
		$sampNum++;
	}
	my $spearman = 1-(6*$sumDiff/($sampNum*($sampNum**2-1)));
	return $spearman;
}

