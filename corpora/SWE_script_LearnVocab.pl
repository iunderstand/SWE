
#=============================================#
#     Scripts for Semantic Word Embedding     #
#             ACL-2015, Beijing               #
#---------------------------------------------#
#     Quan Liu, University of Science and     #
#               Technology of China. 2015     #
#       http://home.ustc.edu.cn/~quanliu/     #
#=============================================#

# SWE_script_LearnVocab.pl
# Learn vocabulary from one training corpus.
# Out: words with sorted occurrence frequency.

my $src_corpus = "text8.txt";
my $out_vocab  = "text8.wordfreq";

print "--> src corpus: $src_corpus\n";
print "--> out vocab: $out_vocab\n";
my %vocab = ();
my $sent_num = 0;
open SRC, "<$src_corpus";
while (<SRC>) {
	chomp;
	my $sent = $_;
	my @word = split(/\s+/, $sent);
	foreach my $word (@word) {
		$vocab{$word}++;
	}
	$sent_num++;
	if ($sent_num % 500000 == 0) {
		print "$sent_num\n";
	}
}
close SRC;

open OUT, ">$out_vocab";
foreach my $word (sort {$vocab{$b}<=>$vocab{$a}} keys %vocab) {
	print OUT "$word $vocab{$word}\n";
}
close OUT;
