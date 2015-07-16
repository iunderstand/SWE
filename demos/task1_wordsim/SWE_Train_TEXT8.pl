
use File::Path;
#=============================================#
#     Scripts for Semantic Word Embedding     #
#             ACL-2015, Beijing               #
#---------------------------------------------#
#     Quan Liu, University of Science and     #
#               Technology of China. 2015     #
#       http://home.ustc.edu.cn/~quanliu/     #
#=============================================#

# SWE_Train_TEXT8.pl
# SWE model training on TEXT8 corpus.
# For demo on word similarity and synonym selection task.
# Larger corpora are encourged to employ for SWE training.

my $model_tool   = "../../bin/SWE_Train";

## Train files
my $train_data   = "../../corpora/TEXT8";
my $train_file   = "$train_data/text8.txt";
my $vocab_file   = "$train_data/text8.wordfreq.cut5";

my $out_model_path = "EmbedVector_TEXT8";
my @semantic_flag  = ("SA1", "HH1", "COM1");

my $iter_times = 1;

foreach my $semantic_flag (@semantic_flag) {
	
	my $semantic_train = "../../semantics/TEXT8/SemWE.EN.KnowDB.$semantic_flag.inTEXT8.train";
	my $semantic_valid = "../../semantics/TEXT8/SemWE.EN.KnowDB.$semantic_flag.inTEXT8.valid";
	
	my $sample_num   = 1e-4; # 
	
	my $out_model    = "$out_model_path/sem$semantic_flag.Inter_run$iter_times.NEG$sample_num";
	mkpath $out_model if !-s $out_model;
	
	## Network
	my @layer1_size  = (100);
	my $window_size  = 5;
	my $learn_rate   = 0.025;	
	my @run_negative = (5);
	
	## SWE parameters
	my @inter_param = (0, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5);	
	# Default setting
	my @hinge_margin= (0.0); # hinge function for SWE quantization
	my @delta_right  = (1); # use right part of inequation
	my $delta_left = 1; # use left part of inequation
	my $sem_addtime = 0;
	my $weight_decay= 0.0;
	
	my $run_threads  = 8;

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
				" -iter $iter_times";
				" -delta-right $delta_right";
					
			system("$train_cmd");
			#system("$train_cmd >$save_runlog");
	}}}}}
}
