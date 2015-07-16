
#=============================================#
#     Scripts for Semantic Word Embedding     #
#             ACL-2015, Beijing               #
#---------------------------------------------#
#     Quan Liu, University of Science and     #
#               Technology of China. 2015     #
#       http://home.ustc.edu.cn/~quanliu/     #
#=============================================#

# SWE_script_IneqFilterByVocab.pl
# Filter semantic inequalities by vocabulary

my $vocab_file    = "../../corpora/TEXT8/text8.wordfreq.cut5";

my %know_to_flag  = ();
$know_to_flag{"../SWE.EN.KnowDB.WordNet-Book.Synon-Anton"} = "SA1.inTEXT8";
$know_to_flag{"../SWE.EN.KnowDB.WordNet.Hyper-Hypon"} = "HH1.inTEXT8";
$know_to_flag{"../SWE.EN.KnowDB.WordNet-Book.AllSet"} = "COM1.inTEXT8";

foreach my $src_knowledge (keys %know_to_flag) 
{
	
	my $filter_flag = $know_to_flag{$src_knowledge};
	my $filter_ineq = "SemWE.EN.KnowDB.$filter_flag";
	my $filter_wset = "SemWE.EN.KnowDB.$filter_flag.wordset";
	
	print "\nSemantic Inequaltiy Filtering By Vocabulary...\n";
	print "== src knowledge: $src_knowledge\n";
	print "== filter flag: $filter_flag\n";
	InequalityDictFilter($src_knowledge, $vocab_file, $filter_ineq, $filter_wset);

	## Split train and Valid set
	print "== Split train and Valid set...\n";
	system("perl SWE_script_SplitTrainValid.pl $filter_ineq 3000");
}

###########################
sub InequalityDictFilter()
{
	my ($src_knowledge, $vocab_file, $filter_ineq, $filter_wset) = @_;
	
	my %vocab = ();
	open VOC, "<$vocab_file";
	while (<VOC>) {
		if (/^(\S+)\s+(\d+)\n/) {
			my $word = $1;
			my $freq = $2;
			$vocab{$word} = $freq;
		}
	}
	close VOC;

	my %wordset = ();
	open SRC, "<$src_knowledge";
	open OUT, ">$filter_ineq";
	while (<SRC>) {
		chomp;
		my $line = $_;
		my @word = split(/\s+/, $line);
		my $useflag = 1;
		foreach my $word (@word) {
			if (!exists $vocab{$word}) {
				#print "$word\n";
				$useflag = 0;
			}
		}
		if ($useflag == 1) {
			print OUT "$line\n"; 
			foreach my $word (@word) {
				$wordset{$word}++;
			}
		}
	}
	close SRC;
	close OUT;

	open WORD, ">$filter_wset";
	foreach my $word (sort {$wordset{$b}<=>$wordset{$a}} keys %wordset) {
		print WORD "$word $wordset{$word}\n";
	}
	close WORD;
}

