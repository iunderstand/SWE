
//=============================================//
//      Codes for Semantic Word Embedding      //
//             ACL-2015, Beijing               //
//---------------------------------------------//
//     Quan Liu, University of Science and     //
//               Technology of China. 2015     //
//       http://home.ustc.edu.cn/~quanliu/     //
//=============================================//

// SemWE_Test_SynSel.cpp
// Word Embedding for Synonym Selection task.
// Using Intel MKL to make calculation faster

#define MKL_YES

#include <iostream>
#include <map>
#include <vector>
#include <string.h>
#include <fstream>
#include <math.h>
#include <time.h>
#include <algorithm>

#ifdef MKL_YES
#include <mkl.h>
#endif

using namespace std;

//typedef map<string, vector<float>> WordEmbed;
typedef map<string, int> WordMapper;
struct WordSim{
	int wordID;
	float simRES;
};

#define WORD_LEN 1024
typedef float real;

real SemWE_VectorDot(real *vec_a, real *vec_b, int vec_size)
{
	int  i = 0;
	int incx = 1;
	int incy = 1;
	real fastres = 0.0;
	#ifdef MKL_YES
	fastres = sdot(&vec_size, vec_a, &incx, vec_b, &incy);
	#else
	for (i = 0; i < vec_size; ++i){
		fastres += vec_a[i]*vec_b[i];
	}
	#endif	
	return fastres;
}
void SemWE_VectorCopy(real *vec_a, real *vec_b, int vec_size)
{
	int  i = 0;
	int incx = 1;
	int incy = 1;
	#ifdef MKL_YES
	scopy(&vec_size, vec_b, &incx, vec_a, &incy);
	#else
	for (i = 0; i < vec_size; ++i){
		vec_a[i] = vec_b[i];
	}
	#endif
}
void SemWE_VectorLinear(real *main_vector, real *scale_vector, real scale_coeff, int vec_size)
{
	int  i = 0;
	int incx = 1;
	int incy = 1;
	#ifdef MKL_YES
	saxpy(&vec_size, &scale_coeff, scale_vector, &incx, main_vector, &incy);
	#else
	for (i = 0; i < vec_size; ++i){
		main_vector[i] += scale_coeff * scale_vector[i];
	}
	#endif
}
void SemWE_VectorScale(real *input_vector, real scale_coeff, int vec_size)
{
	int i = 0;
	int incx = 1;
	#ifdef MKL_YES
	sscal(&vec_size, &scale_coeff, input_vector, &incx);
	#else
	for (i = 0; i < vec_size; ++i){
		input_vector[i] = scale_coeff * input_vector[i];
	}
	#endif
}
real SemWE_VectorNorm(real *input_vector, int vec_size)
{
	int i = 0;
	real norm_value = 0.0;
	int incx = 1;
	#ifdef MKL_YES
	norm_value = snrm2(&vec_size, input_vector, &incx);
	#else
	for (i = 0; i < vec_size; ++i){
		norm_value += (input_vector[i]*input_vector[i]);
	}
	norm_value = sqrt(norm_value);
	#endif
	return norm_value;
}

/////////////// Euclidean Distance Process ///
real SemWE_CalcEuclidean(real *vec_a, real *vec_b, int vec_size)
{
	real eulidean_value = 0.0;
	real *minus_vector = (real*)malloc(vec_size*sizeof(real));
	SemWE_VectorCopy(minus_vector, vec_a, vec_size);
	SemWE_VectorLinear(minus_vector, vec_b, -1.0, vec_size);
	eulidean_value = SemWE_VectorNorm(minus_vector, vec_size);
	free(minus_vector); minus_vector = NULL;
	return eulidean_value;
}

real SemWE_CalcCosine(real *vec_a, real *vec_b, int vec_size)
{
	real val_dot = 0.0;
	real val_nrmA= 0.0;
	real val_nrmB= 0.0;
	val_dot = SemWE_VectorDot(vec_a, vec_b, vec_size);
	val_nrmA= SemWE_VectorNorm(vec_a, vec_size);
	val_nrmB= SemWE_VectorNorm(vec_b, vec_size);
	return (val_dot/(val_nrmA*val_nrmB));
}	



bool CompareCosine(WordSim a, WordSim b)
{
	return a.simRES>b.simRES;
}
bool CompareEuclidean(WordSim a, WordSim b)
{
	return a.simRES<b.simRES;
}

real *wordEmbed;
WordMapper wordMapper;
map<string, int> wordSet;
char word_embed[2048];
char sim_result[2048];
char word_pair[2048];
int  candidate_num = 4; // euclidean
int threadNum = 1;
int word_num = 0;
int vect_dim = 0;
long long calcu_wordnum = 0;
long long calcuNum = 0;

//////////////////////////////////////////////////////////////////////////
int main (int argc, char *argv[])
{		
	if (argc < 4)
	{
		printf("SemWE_Test_SynSel synonym_question word_embed synonym_result candidate_num\n");
		printf("Primary Designed for the TOEFL Synonym Selection Task\n");
		exit(1);
	}
	
	strcpy(word_pair, argv[1]);
	strcpy(word_embed, argv[2]);
	strcpy(sim_result, argv[3]);
	candidate_num = atoi(argv[4]);
				
	char tmp_word[WORD_LEN];
	float tmp_value = 0.0;
	
	printf(">> Synonym Selection on Word Embedding Model\n");
	
	FILE *fTEST = fopen(word_pair, "r");
	if (fTEST == NULL)
	{
		printf(">> Error, can not open file %s\n", word_pair);
		exit(1);
	}
	FILE *fRES = fopen(sim_result, "w");
	if (fRES == NULL)
	{
		printf(">> Error, can not open file %s\n", sim_result);
		exit(1);
	}

	printf("--- Load Word Embedding from: %s\n", word_embed);
	FILE *fEMB = fopen(word_embed, "r");
	fscanf(fEMB, "%d%d", &word_num, &vect_dim);
	printf("--- word num: %d\n--- vec dimen: %d\n", word_num, vect_dim);
	wordEmbed = (real*)malloc(sizeof(real)*word_num*vect_dim);
	
	for (int i = 0; i < word_num; i++)
	{
		fscanf(fEMB, "%s", tmp_word);		
		for (int j = 0; j < vect_dim; j++)
		{
			fscanf(fEMB, "%f", &tmp_value);
			wordEmbed[i*vect_dim+j] = tmp_value;
			//tmp_vector.push_back(tmp_value);
		}
		wordMapper.insert(make_pair(tmp_word, i));
		wordSet.insert(make_pair(tmp_word, i));
	}
	/*if (wordMapper.size() != word_num)
	{
	}*/
	fclose(fEMB);
	printf("--- Load finish\n");
				
	//printf(">> Synonym Selection\n");
	printf("--- Synonym question : %s\n", word_pair);
	printf("--- Candidate num: %d\n", candidate_num);
	printf("--- Word Embed: %s\n", word_embed);	
	printf("--- Selection result: %s\n", sim_result);
	clock_t start = clock();
	WordSim tmp_sim;
	
	char word_A[1024];
	char word_B[1024];
	
	int all_num = 0;
	int use_num = 0;
	int cand_id = 0;
	while (fscanf(fTEST, "%s", word_A) != EOF)
	{
		//printf("--- %s %s: ", word_A, word_B);
		all_num++;
		
		fprintf(fRES, "%s\t", word_A);

		float max_sim = -10000.0;
		string sel_synword = "";
		// candidate
		for (cand_id = 0; cand_id < candidate_num; cand_id++)
		{
			fscanf(fTEST, "%s", word_B);
			if (cand_id == 0){
				sel_synword = word_B;
			}
			float sim_AB = -100.0;

			if (wordSet.find(word_A) == wordSet.end() || wordSet.find(word_B) == wordSet.end())
			{
				;
			}
			else{
				int index_A = wordSet[word_A];
				int index_B = wordSet[word_B];
				
				
				sim_AB = SemWE_CalcCosine(&wordEmbed[index_A*vect_dim], &wordEmbed[index_B*vect_dim], vect_dim);
				if (sim_AB > max_sim)
				{
					max_sim = sim_AB;
					sel_synword = word_B;
				}				
			}	
			fprintf(fRES, "%s(%.6f) ", word_B, sim_AB);
		}
		fprintf(fRES, "=>best: %s\n", sel_synword.c_str());
		//fprintf(fRES, "%s\t%s", sel_synword.c_str());
	}
	fclose(fRES);
	fclose(fTEST);
	//////////////////////////////////////////////////////////////////////////
	double timeCost = (clock()-start)/CLOCKS_PER_SEC;		
	//printf("--- calculate nums: %d (/%d)\n", use_num, all_num);
	printf("--- elapsed time: %f\n", timeCost);		
	//printf(">> Finish.\n");
	
	free(wordEmbed); wordEmbed = NULL;	
	return 0;
}



