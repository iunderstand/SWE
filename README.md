# Semantic Word Embeddings (SWE)

SWE represents a general framework to incorporate semantic knowledge into the popular data-driven learning process of word embeddings to improve the quality of them. Under the SWE framework, semantic knowledge could be quantized as many ordinal ranking inequalities and the learning of word vectors is formulated as a constrained optimization problem. In detail, the data-derived objective function is optimized subject to all ordinal knowledge inequality constraints extracted from available knowledge resources such as Thesaurus, WordNet, knowledge graphs, etc. We have demonstrated that this constrained optimization problem can be efficiently solved by the stochastic gradient descent (SGD) algorithm, even for a large number of inequality constraints. Experimental results on four standard NLP tasks, including word similarity measure, sentence completion, name entity recognition, and the TOEFL synonym selection, have all demonstrated that the quality of learned word vectors can be significantly improved after semantic knowledge is incorporated as inequality constraints during the learning process of word embeddings.

The main SWE functions include
* SWE_Train	main tool, support SWE model as well as Skip-gram training.
* SWE_Test_WordSim	tool for applying word embeddings for word similarity.
* SWE_Test_SentComplete	tool for applying word embeddings for sentence completion
* SWE_Test_SynSel	tool for applying word embeddings for synonym selection.
	
The SWE application include
* 
