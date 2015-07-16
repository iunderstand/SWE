

## Semantic Word Embedding (SWE) ##

## Quan Liu, University of Science and Technology of China

## ACL 2015 long paper: Learning Semantic Word Embeddings based on Ordinal Knowledge Constraints


Key: Our main code for training SWE models is: SWE_Train.c

This toolkit is modified from the open word2vec toolkit (https://code.google.com/p/word2vec/).

## word2vec license:
//  Copyright 2013 Google Inc. All Rights Reserved.
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//      http://www.apache.org/licenses/LICENSE-2.0
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.


## 1. Some Notices about the Skip-gram model

In our SWE_Train.c codes, for the Skip-gram model, we use the central word to predict the context words.

Tomas Mikolov declared that, for efficiency, the implemented Skip-gram model in the word2vec toolkit uses 
the context words to predict central word.
It can be shown that both implementations of Skip-gram are almost equal (just write on a paper what sequence of training pairs will you obtain for some sentence - it will be just shuffled differently if you predict the middle word given context versus predicting context words given the middle one).
More information could be found at:

https://groups.google.com/forum/#!searchin/word2vec-toolkit/skip-gram/word2vec-toolkit/2R3XVndDXi8/lcosnvYxepUJ


Some version of SWE codes

SWE_Train_v1.c: the same as SWE_Train.c, add semantic constraints for each trained central word.

SWE_Train_v2.c: use the context words to predict central word, add semantic constraints for each trained central word.

SWE_Train_v3.c: use the context words to predict central word, add semantic constraints for all trained context words.

For the default setting, we use the SWE_Train_v1 as our SWE model training tool.
We also recommend you to use the other two versions for experiemnts.


## 2. SWE Evaluation Tools

Beyond the training tool, we also supply some evaluation tools for testing SWE models.

(1) SWE_Test_WordSim.cpp: Word embedding for word similarity

(2) SWE_Test_SynSel.cpp: Word embedding for synonym selection

(3) SWE_Test_SentComplete.c: Word embedding for sentence completion


## 3. Prerequisites

You can choose to use the Fast Version of toolkit (the Intel Math Kernel Library is needed).

