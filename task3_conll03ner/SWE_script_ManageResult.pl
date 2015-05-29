

my $data_dir  = $ARGV[0]; # "Reuters_ACL2015Camera";
my $stat_key  = $ARGV[1];
my $stat_file = "$data_dir/$stat_key.NERresults";


print "\nStatFlag: $stat_key\n\n";

open OUT, ">$stat_file";
opendir(DIR, $data_dir);
foreach my $subdir (readdir DIR) 
{
	if ($subdir eq $stat_key) {
		
		opendir(SUB, "$data_dir/$subdir");
		foreach my $file_name (readdir SUB) {
			if ($file_name =~ /MainResult/) {
				my $res_file = "$data_dir/$subdir/$file_name";
				print OUT ">> $res_file\n";	
				print ">> $res_file\n";
				my $ner_result = GetResult($res_file);
				print OUT "$ner_result\n";
				print "$ner_result\n";
			}
		}
		closedir(SUB);		
	}
}
closedir(DIR);
close OUT;


sub GetResult()
{
	my ($res_file) = @_;
	my $result_string = "";

	## Phrase Level
	#print OUT "\n";
	my $read_res = 0;
	my $res_flag = 0;		
	open RES, "<$res_file";	
	my @resbuff = ();
	while (<RES>) {
		if (/Phrase-level Acc Level1/i) {
			$read_res = 1;
		}
		elsif (/Token-level Acc Level1/i) {
			$read_res = 0;	
		}
		if (/Overall\s+\S+\s+\S+\s+(\S+)\s+/) {				
			my $f1_res = $1;
			if ($read_res eq 1){
				$res_flag++;
				push @resbuff, $f1_res;
#					if ($res_flag == 1) {
#						print OUT "-- test f1: $f1_res\n";
#					}
#					if ($res_flag == 2) {
#						print OUT "-- dev f1: $f1_res\n";
#					}
#					if ($res_flag == 3) {
#						print OUT "-- muc7 f1: $f1_res\n";
#					}
				if ($#resbuff == 2) {
					#print OUT "[Phrase Level] dev: $resbuff[1]; test: $resbuff[0]; muc7: $resbuff[2]\n";
					$result_string .= "[Phrase Level] dev: $resbuff[1]; test: $resbuff[0]; muc7: $resbuff[2]\n";
				}
			}
		}			
	}
	close RES;

	## Token Level
	#print OUT "[Token Level]:\n";
	my $read_res = 0;
	my $res_flag = 0;		
	open RES, "<$res_file";	
	my @resbuff = ();
	while (<RES>) {
		if (/Token-level Acc Level1/i) {
			$read_res = 1;
		}
		if (/Overall\s+\S+\s+\S+\s+(\S+)\s+/) {				
			my $f1_res = $1;
			if ($read_res eq 1){
				$res_flag++;
				$read_res = 0;
				push @resbuff, $f1_res;
				if ($#resbuff == 2) {
					#print OUT "[Tokens Level] dev: $resbuff[1]; test: $resbuff[0]; muc7: $resbuff[2]\n";
					$result_string .= "[Tokens Level] dev: $resbuff[1]; test: $resbuff[0]; muc7: $resbuff[2]\n";
				}
			}
		}			
	}
	close RES;
	return $result_string;
}
