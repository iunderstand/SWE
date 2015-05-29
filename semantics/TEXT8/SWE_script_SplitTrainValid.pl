
#=============================================#
#     Scripts for Semantic Word Embedding     #
#             ACL-2015, Beijing               #
#---------------------------------------------#
#     Quan Liu, University of Science and     #
#               Technology of China. 2015     #
#       http://home.ustc.edu.cn/~quanliu/     #
#=============================================#

# SWE_script_SplitTrainValid.pl
# Split source file into train and valid set.

my $kdb_total = $ARGV[0];
my $valid_num = $ARGV[1];

my $kdb_train = "$kdb_total.train";
my $kdb_valid = "$kdb_total.valid";

open IN, "<$kdb_total";
my @IN = <IN>;
close IN;

my $ineq_num = $#IN+1;
my %valid = ();
while ((keys %valid) < $valid_num) 
{
	$valid{int(rand($ineq_num))} = 1;
}

open TRA, ">$kdb_train";
open VAL, ">$kdb_valid";
foreach (keys %valid) 
{
	print VAL "$IN[$_]";
}
for (my $i = 0; $i <= $#IN; $i++)
{
	if (!exists $valid{$i}) 
	{
		print TRA "$IN[$i]";
	}
}
close TRA;
close VAL;


