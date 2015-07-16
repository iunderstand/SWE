
use File::Path;
#=============================================#
#     Scripts for Semantic Word Embedding     #
#             ACL-2015, Beijing               #
#---------------------------------------------#
#     Quan Liu, University of Science and     #
#               Technology of China. 2015     #
#       http://home.ustc.edu.cn/~quanliu/     #
#=============================================#

# SWE_Test_SynSel.pl
# Word embedding for synonym selection task.
# Task: TOEFL 80 question synonym selection.
# Criterion: Selection accuracy (%)

my $calsim_tool = "../../bin/SWE_Test_SynSel";

my $test_set    = "TOEFL80";
my $word_pair   = "testset/SWE.EN.TestSet.$test_set.question";
my $answer_file = "testset/SWE.EN.TestSet.$test_set.answer";
my $cand_num    = 4;

my $out_model_path = $ARGV[0]; # "EmbedModel_ToolA_ENWIKI9";
my @semantic_flag  = ("SA1", "HH1", "COM1");

my $test_result = "result/$test_set.result";

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
	my $result_dir   = "$out_model/SynSel";
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
			
			my ($corr_num, $total_num, $model_accuracy) = CalCuModel($word_pair, $save_embeded, $calsim_res, $sim_compare, $answer_file);
			print "---correct: ($corr_num/$total_num), accuracy: $model_accuracy\n\n";
			printf SIMRES "---model: $model_flag\n";
			printf SIMRES "---corr: ($corr_num/$total_num), accuracy: $model_accuracy\n\n";
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
		print "--- $calsim_tool $pair_file $embed_file $calsim_res $cand_num\n";
		system("$calsim_tool $pair_file $embed_file $calsim_res $cand_num");
	}
	else
	{
		print "--- $calsim_res exists\n";
	}
	
	my @ans = ();
	my @res = ();

	open RES, "<$calsim_res";
	while (<RES>) {
		if (/best:\s+(\S+)/) {
			push @res, $1;
		}
	}
	close RES; 
	
	print "--- answer file: $answer_file\n";
	open ANS, "<$answer_file";
	while (<ANS>) {		
		if (/^(\S+)/) {
			push @ans, $1;		
		}
	}
	close ANS; 
	
	open OUT, ">$sim_compare";
	my $res_num = $#res+1;
	my $ans_num = $#ans+1;
	if ($res_num ne $ans_num) {
		print "\nERROR, Num mismatch\n";
	}
	else{
		my $total_num = $#res + 1;
		my $corr_num = 0;
		for (my $i = 0; $i < $total_num; $i++) 
		{
			my $res_word = $res[$i];
			my $ans_word = $ans[$i];
			if ($res_word eq $ans_word) {
				$corr_num++;
				print OUT "ANS:$ans_word, RES: $res_word.  CORR\n";
			}
			else{
				print OUT "ANS:$ans_word, RES: $res_word.  FA\n";
				#print "$res_word ne $ans_word\n";
			}
		}
		my $corr_rate = $corr_num / $total_num;
		return ($corr_num, $total_num, $corr_rate);
	}	
	close OUT;
}

