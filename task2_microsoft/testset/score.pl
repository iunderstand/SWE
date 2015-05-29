#!/usr/bin/perl -w

# scoring script for Holmes dataset

die "usage: score file reference\n" unless ($#ARGV==1);
$infile = $ARGV[0];
$reffile = $ARGV[1];

open(FIN, $infile) or die "unable to open $infile\n";
open(RIN, $reffile) or die "unable to open $reffile\n";

@hyp = <FIN>;
@ref = <RIN>;

die "mismatched number of hypotheses and references\n" unless ($#hyp==$#ref);
$nlines = $#hyp+1;
$ndev = int($nlines/2);
$ntest = $nlines - $ndev;

sub normalize {
    my $str = shift;
    $str =~ s/[\r\n]//og;
    $str =~ s/^\s+//o;
    $str =~ s/\s+$//o;
    $str =~ s/\s+/ /og;
    return $str;
}

$correct = 0;
$correct_dev = 0;
$correct_test = 0;
$tot = 0;
for ($i=0; $i<=$#ref; $i++) {
    $ref[$i] = normalize($ref[$i]);
    $hyp[$i] = normalize($hyp[$i]);
    print "$ref[$i]\n\t=> $hyp[$i]\n********\n" unless ($ref[$i] eq $hyp[$i]);
    $correct++ if ($ref[$i] eq $hyp[$i]);
    $correct_dev++ if ($ref[$i] eq $hyp[$i] and $i<$ndev);
    $correct_test++ if ($ref[$i] eq $hyp[$i] and $i>=$ndev);
    $tot++;
}

print "$correct of $tot correct\n";

$ave = 100 * $correct / $tot;
$dave = 100 * $correct_dev / $ndev;
$tave = 100 * $correct_test / $ntest;

print "Overall average: $ave%\n";
print  "dev: $dave%\ntest: $tave%\n";
