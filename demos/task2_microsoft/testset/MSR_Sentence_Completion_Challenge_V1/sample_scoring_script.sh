
cat Data/sample.lm_output.txt | ./bestof5.pl > Data/sample.temp

./score.pl Data/sample.temp Data/Holmes.lm_format.answers.txt
