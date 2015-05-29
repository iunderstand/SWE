
This contains the MSR Sentence Completion dataset. It is derived from 
the works of Sir Arthur Conan Doyle, specifically sentences from
the following Sherlock Holmes collections available through Project Gutenberg:
* The Sign of the Four 
* The Hound of the Baskervilles 
* The Adventures of Sherlock Holmes
* The Memoirs of Sherlock Holmes 
* The Valley of Fear  
Under no circumstances should data from these sources be used as training data.
Please see http://research.microsoft.com/scc/ for a full description of the data.

Questions please contact {gzweig,cburges}@microsoft.com

Brief file description:
In the Data/ subdirectory, Holmes.human_format.questions and Holmes.human_format.answers contain the questions and answers in human-friendly format. For example,

From the question file:
1) The metal work was in the form of a double ring , but it had been bent and _____ out of its original shape.
        a) marched
        b) faded
        c) wriggled
        d) poured
        e) twisted

From the answer file:
1) [e] twisted

Holmes.machine_format.questions and Holmes.machine_format.answers have the same questions in a more machine-friendly format. In the question file, each question is fully expanded with the option places in square brackets []. The questions are numbered. 

Holmes.lm_format.questions and Holmes.lm_format.answers has the data in a format suitable for conventional language modeling, with the sentences prefixed by <s> and terminated in </s> rather than a period. Note that the questions are not numbered, but come in the same blocks of five as before.  The answer file has the correct answer for each block of five sentences, and thus has one fifth the lines.

Data/sample.lm_output has sample output from a program that assigns a log-probability to each sentence. To score it, make sure the perl scripts in this directory are executable, and type "./sample_scoring_script.sh" You should see output indicating 372 of 1040 correct.
 
The file list_of_training_texts shows the Project Gutenberg texts that were used in building the language model that proposed word alternates.

While we have not created separate files, if you wish to use separate development and test sets, we recommend using the first 520 sentences as the development set and last 520 sentences as the test set; the scoring script will print these numbers by default.
