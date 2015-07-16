
use File::Path;
use File::Copy;

my $embed_path = $ARGV[0];
my $work_dir   = $ARGV[1];
my $seman_flag = $ARGV[2];

my $bin_dir    = "/home/quanliu/SWE/task3_conll03ner/NerACL2010_Experiments/bin";
#my $data_rar   = "/home/quanliu/SWE/task3_conll03ner/Data.rar";

my $embed_dir  = "$embed_path/$seman_flag"; # "/home/quanliu/SWE/task3_conll03ner/Reuters_WordEmbedding/$seman_flag";

#my $work_dir   = "Reuters_ACL2015Camera";

my $config_file= "CoNLL2003.Sample.config";

my $format_trans= 1;
my $copy_file   = 1;
my $embed_dimen = 50;
my $norm_coeff  = 0.3;

my $embed_dir  = "$embed_path/$seman_flag"; # "/home/quanliu/SWE/task3_conll03ner/Reuters_WordEmbedding/$seman_flag";
my $config_dir = "/home/quanliu/SWE/task3_conll03ner/$work_dir/ConfigDoc/$seman_flag.GAZ";
my $save_model = "/home/quanliu/SWE/task3_conll03ner/$work_dir/NER-Model/$seman_flag.GAZ";
my $run_dir    = "/home/quanliu/SWE/task3_conll03ner/$work_dir/$seman_flag.GAZ";
my $run_bash   = "/home/quanliu/SWE/task3_conll03ner/$work_dir/$seman_flag.GAZ/AllRun.pl";

#my $embed_dir  = "/disk1/quanliu/WordEmbed_forNER/Reuters_WordEmbedding/$seman_flag";
#my $config_dir = "/disk1/quanliu/WordEmbed_forNER/Reuters_ACL2010Config/$seman_flag.GAZ";
#my $save_model = "/disk1/quanliu/WordEmbed_forNER/Reuters_ACL2010Model/$seman_flag.GAZ";
#my $run_dir    = "/disk1/quanliu/WordEmbed_forNER/Reuters_NERCoreDir/$seman_flag.GAZ";

my $config_file= "CoNLL2003.Sample.config";

my $format_trans= 1;
my $copy_file   = 1;
my $embed_dimen = 50;
my $norm_coeff  = 0.3;

print "\n===embedding path: $embed_path\n===work dir: $work_dir\n===semantic flag: $seman_flag\n\n";

mkpath $config_dir if !-s $config_dir;
mkpath $save_model if !-s $save_model;
mkpath $run_dir if !-s $run_dir;
opendir(DIR, "$embed_dir");
my $fileNum = 0;
open ALL, ">$run_bash";
foreach my $file (readdir DIR) 
{
	if ($file =~ /wordembed\.(\S+)\.embeded\.txt$/)
	{
		my $model_flag = $1;
		my $src_embed = "$embed_dir/wordembed.$model_flag.embeded.txt";
		my $out_embed = "$embed_dir/wordembed.$model_flag.embeded.new";
		
		print "\n>> $model_flag\n";
		
		## 1. Delete first line
		if ($format_trans) 
		{
			print "--- Word Embedding Format Transform\n";
			open EMB, "<$src_embed";
			open NEW, ">$out_embed";
			print "--- src: $src_embed\n";
			print "--- out: $out_embed\n";
			my $lineID = 0;			
			while (<EMB>) {
				#print "$_";
				my $line = $_;
				$lineID++;				
				if ($lineID > 1) {
					print NEW "$line";
				}
			}
			close EMB;	
			close NEW;
		}
		
		## 2. Configure File
		my $config_flag= "$model_flag.Norm$norm_coeff";
		my $out_config = "$config_dir/$config_flag.config";
		
		open IN,"<$config_file";				
		open OUT,">$out_config";
		while (<IN>) 
		{
			if (/configFilename\s+(\S+)/i) {
				print OUT "configFilename\t$config_flag\n";
			}
			elsif (/pathsToWordEmbeddings\s+(\S+)/i) {
				print OUT "pathsToWordEmbeddings\t$out_embed\n";
			}
			elsif (/embeddingDimensionalities\s+/i) {
				print OUT "embeddingDimensionalities\t$embed_dimen\n";
			}
			elsif (/normalizationConstantsForEmbeddings\s+/i) {
				print OUT "normalizationConstantsForEmbeddings\t$norm_coeff\n";
			}
			elsif (/pathToModelFile/i) {
				print OUT "pathToModelFile\t$save_model\n";
			}
			elsif (/GazetteersFeatures/i) {
				print OUT "GazetteersFeatures\t1\n";
			}
			else{
				print OUT "$_";
			}
		}
		close IN;
		close OUT;
		
		$fileNum++;
		my $out_script = "$run_dir/$fileNum.$config_flag.pl";
		my $out_result = "$run_dir/$fileNum.$config_flag.MainResult";
		open RUN, ">$out_script";
		print RUN "system(\"nice java -Xmx6g -classpath LBJ2.jar:LBJ2Library.jar:bin:stanford-ner.jar:stanford-ner.src.jar:lucene-core-2.4.1.jar ExperimentsACL2010.TrainExperimentsCoNLLDevTuningGivenConfig $out_config >$out_result\");\n";
		close RUN;
		
		print ALL "print \"perl $fileNum.$config_flag.pl\\n\\n\";\n";
		print ALL "system(\"perl $fileNum.$config_flag.pl\");\n";
		
		## Copy
		if ($copy_file) {
			CopyBinToDir($bin_dir, $run_dir);		
		}
	}
}
close ALL;
closedir(DIR);

#print "Copy data...\n";
#system("cp -r $data_dir /home/quanliu/SWE/task3_conll03ner/$work_dir");
#mkdir "DebugLog";
#copy($data_rar, "$rub_dir/Data.rar");


sub CopyBinToDir()
{
	my ($src_dir, $out_dir) = @_;
	opendir(DIR, $src_dir);
	mkpath $out_dir if !-s $out_dir;
	foreach my $file (readdir DIR) {
		if ($file eq "bin") {
			opendir(BIN, "$src_dir/bin");
			foreach my $sub (readdir BIN) {
				if ($sub =~ /[a-z]+/i) {
					if ($sub ne "Experiments") {
						##==
						my $sub_dir = "$src_dir/bin/$sub";
						my $sub_out = "$out_dir/bin/$sub";
						mkpath $sub_out if !-s $sub_out;
						opendir(SUB,$sub_dir);
						foreach my $sub_file (readdir SUB) {
							if ($sub_file =~ /[a-z]+/i) {
								copy("$src_dir/bin/$sub/$sub_file", "$sub_out/$sub_file");
							}
						}
						closedir(SUB);
						##==
					}
				}
			}
			close BIN;
		}
		elsif ($file eq "src") {
			;
		}
		elsif ($file ne "." and $file ne "..") {		
			copy("$src_dir/$file", "$out_dir/$file");
		}
	}
	closedir(DIR);
}
