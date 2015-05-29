
use File::Path;

#=============================================#
#     Scripts for Semantic Word Embedding     #
#             ACL-2015, Beijing               #
#---------------------------------------------#
#     Quan Liu, University of Science and     #
#               Technology of China. 2015     #
#       http://home.ustc.edu.cn/~quanliu/     #
#=============================================#

# SWE_Test_SentCompletion.pl
# Using SWE word embeddings for sentence completion.
# Task: Microsoft sentence completion challenge.

my $model_tool   = "../bin/SemWE_Test_SentComplete";

## Train files
my $train_data   = "../corpora/Holmes";
my $train_file   = "$train_data/Holmes_Training_Data.txt";
my $vocab_file   = "$train_data/Holmes_Training_Data.txt.wordfreq.cut5";

## ADD
my $msrsc_input  = "testset/Holmes.questions";
my $msrsc_answer = "testset/Holmes.answers";
my $msrsc_score  = "testset/score.pl";

my $out_model_path = "EmbedVector_Holmes";
my @semantic_flag  = ("SA1", "HH1", "COM1");

my $test_result = "result/$out_model_path.Holmes.result";

mkdir "result" if !-s "result";

## (ADD)
open SIMRES, ">>$test_result";
my $cur_time = GetCurTime();
print SIMRES "\n====== $test_set\n";
print SIMRES "--- $cur_time\n";

my $iter_times = 1;

foreach my $semantic_flag (@semantic_flag) 
{
	my $semantic_train = "../semantics/Holmes/SemWE.EN.KnowDB.$semantic_flag.inHolmes.train";
	my $semantic_valid = "../semantics/Holmes/SemWE.EN.KnowDB.$semantic_flag.inHolmes.valid";
	
	my $sample_num   = 1e-5; 

	my $out_model    = "$out_model_path/sem$semantic_flag.Inter_run$iter_times.NEG$sample_num";
	if (!-s $out_model) {
		next;
	}
	
	## (ADD)
	my $result_dir   = "$out_model/Holmes";
	mkpath $result_dir if !-s $result_dir;
	
	## Network
	my @layer1_size  = (600);
	my $window_size  = 5;
	my $learn_rate   = 0.025;
	
	my @run_negative = (5);

	my @inter_param = (0, 0.01, 0.05, 0.1, 0.2, 0.3);	
	my @hinge_margin= (0.0);
	my @delta_right  = (1); # use right part of inequation
	my $delta_left = 1; # use left part of inequation
	my $sem_addtime = 0;
	my $weight_decay= 0.0;
	
	print SIMRES ">>> Semantic: $semantic_flag, RunTimes: $iter_times\n";

	foreach my $layer1_size (@layer1_size) 
	{
	foreach my $run_negative (@run_negative) 
	{	
		foreach my $inter_param (@inter_param) {
		foreach my $hinge_margin (@hinge_margin) {
			foreach my $delta_right (@delta_right) {
			
			my $save_embeded = "$out_model/wordembed.sem$semantic_flag.dim$layer1_size.win$window_size.neg$run_negative.samp$sample_num.inter$inter_param.hinge$hinge_margin.add$sem_addtime.decay$weight_decay.l$delta_left.r$delta_right.embeded.txt";	
			my $save_runlog  = "$out_model/wordembed.sem$semantic_flag.dim$layer1_size.win$window_size.neg$run_negative.samp$sample_num.inter$inter_param.hinge$hinge_margin.add$sem_addtime.decay$weight_decay.l$delta_left.r$delta_right.logfile.txt";
			
			my $model_flag = "wordembed.sem$semantic_flag.dim$layer1_size.win$window_size.neg$run_negative.samp$sample_num.inter$inter_param.hinge$hinge_margin.add$sem_addtime.decay$weight_decay.l$delta_left.r$delta_right";
			my $msrsc_output  = "$result_dir/$model_flag.SentCompletion.res";
			my $msrsc_best  = "$result_dir/$model_flag.SentCompletion.best";
			my $msrsc_correct  = "$result_dir/$model_flag.SentCompletion.corr";
			
			print "\n>>$save_embeded\n";
			
			my $train_cmd =
				 "$model_tool  -debug 2".
				 " -size       $layer1_size".
				 " -train      $train_file".
				 " -read-vocab $vocab_file".
				 " -cbow       0".
				 " -hs         0".
				 " -alpha      $learn_rate".
				 " -window     $window_size".
				 " -sample     $sample_num".		 
				 " -negative   $run_negative".
				 " -threads    $run_threads".
				 " -output $save_embeded".
				 " -sem-coeff $inter_param".
				" -sem-addtime $sem_addtime".
				" -sem-hinge $hinge_margin".
				" -weight-decay $weight_decay".
				" -sem-train $semantic_train".
				" -sem-valid $semantic_valid".
				" -iter $iter_times".
				" -sent-in $msrsc_input".
				" -sent-out $msrsc_output".
				" -load-embeded $save_embeded".
				" -load-predict $save_embeded.predict";
				
			if (!-s $msrsc_output) 
			{
				system("$train_cmd");
				#system("$train_cmd >$save_runlog");
			}
			
			print "--- Correct Rate testing...";
			FindBest($msrsc_output, $msrsc_best);
			system("perl $msrsc_score $msrsc_best $msrsc_answer >$msrsc_correct");
			
			my $corr_rate = 0.0;
			open IN, "<$msrsc_correct";
			while (<IN>) {
				if (/Overall average:\s+(\S+)/) {
					$corr_rate = $1;
				}
			}
			close IN;
			print "---accuracy: $corr_rate\n\n";
			printf SIMRES "---model: $model_flag\n";
			printf SIMRES "---accuracy: $corr_rate\n\n";
			
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

sub FindBest()
{
	my ($src_file, $out_file) = @_;
	open IN, "<$src_file";
	open OUT,">$out_file";
	my $count = 0;
	my $best = "";
	my $bestlp = -1000000;		
	while (<IN>) {
		if (/^(.*)\s+(\S+)$/) {
			my $score = $2;
			my $sent = $1;
			if ($score > $bestlp) {
				$bestlp = $score;
				$best = $sent;
			}
			if ((++$count % 5) == 0) {
				print OUT "$best\n";
				$bestlp = -1000000;
			}
		}			
	}
	close IN;
	close OUT;	
}