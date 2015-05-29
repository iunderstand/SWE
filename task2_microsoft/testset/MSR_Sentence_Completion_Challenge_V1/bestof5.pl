#!/usr/bin/perl -w

$count = 0;
$best = "";
$bestlp = -1000000;

while (<>) {
    s/\s+[\r\n]+$//o;
    @fields = split(/\t/);
    if ($fields[1] > $bestlp) {
        $bestlp = $fields[1];
        $best = $fields[0];
    }
    
    if ((++$count % 5) == 0) {
        print "$best\n";
        $bestlp = -1000000;
    }
}
