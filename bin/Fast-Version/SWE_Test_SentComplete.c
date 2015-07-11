//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

//=============================================//
//      Codes for Semantic Word Embedding      //
//             ACL-2015, Beijing               //
//---------------------------------------------//
//     Quan Liu, University of Science and     //
//               Technology of China. 2015     //
//       http://home.ustc.edu.cn/~quanliu/     //
//=============================================//

// SWE for Sentence Completion task.
// Modified from the word2vec toolkit.
// SemWE_Test_SentComplete.c

//=== Here is the orginal word2vec license ===//
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


#define ON_LINUX

#define MKL_YES

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifdef MKL_YES
#include <mkl.h>
#endif

#ifdef ON_LINUX
#include <pthread.h>
#endif

#ifdef ON_WINDOWS
#include <process.h>
#include <windows.h>
#endif

#define FILE_LENGTH 1048

#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40

const int vocab_hash_size = 30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary

typedef float real;                    // Precision of float numbers

struct vocab_word {
  long long cn;
  int *point;
  char *word, *code, codelen;
};

char embeded_file[FILE_LENGTH], predict_file[FILE_LENGTH];

char train_file[FILE_LENGTH], output_file[FILE_LENGTH];
char save_vocab_file[FILE_LENGTH], read_vocab_file[FILE_LENGTH];
struct vocab_word *vocab;
int binary = 0, cbow = 1, debug_mode = 2, window = 5, min_count = 5, num_threads = 12, min_reduce = 1;
int *vocab_hash;
long long vocab_max_size = 1000, vocab_size = 0, layer1_size = 100;
long long train_words = 0, word_count_actual = 0, iter = 5, file_size = 0, classes = 0;
real alpha = 0.025, starting_alpha, sample = 1e-3;
real *syn0, *syn1, *syn1neg, *expTable;
clock_t start;

int hs = 0, negative = 5;
const int table_size = 1e8;
int *table;

// Microsoft file
char MSRSC_input[FILE_LENGTH];
char MSRSC_output[FILE_LENGTH];


// Quan Liu
#define SEM_KDBNUM 1
#define CVTEST_TIME 1000000
struct TypeInEquation {
	// sim(word_A, word_B) > sim(word_C, word_D)
	int index_A;
	int index_B;
	int index_C;
	int index_D;
	// distance(word_C, word_D) - distance(word_A, word_B)
};
struct TypeTermKDB {
	long long *KDB_IDset;
	int KDB_nums;
};
// Type Qsem track
struct TypeQsemVal
{
	real Qsem_total;
	long long SatisfyNum;
	real SatisfyRate;
	long long totalKDBNum;
	real Qsem_total_prev;	
	real Dgap_total;
	real Dgap_total_prev;
};
struct TypeQbaseVal
{	
	double log10_prob;
	long long token_num;
	real QbaseVal;
	//real perplexity; // ppl = exp10( -log10(p) / wordNum) ?
};

real *zero_vector;

time_t runTimer, endTimer;
struct tm *begRunTime;
struct tm *endRunTime;

// Semantic Constraint
char semwe_inequation_file[FILE_LENGTH];
char semwe_inequation_fileCV[FILE_LENGTH];
real semwe_add_time = 0.0;
real semwe_inter_coeff = 0.005; // 0.1*sema1c + 0.9*baseline
real semwe_weight_decay  = 0.0; // 2014/09/24 weight decay to prevent overfitting
real semwe_hinge_margin = 0.0; // Hinge function low bound, f(x) = max(low_bound, x)

int delta_left = 1;
int delta_right= 1;

struct TypeInEquation *KnowDB_InEquation;
struct TypeTermKDB *KnowDB_TermKDB;
struct TypeInEquation *KnowDB_InEquation_CV;
struct TypeTermKDB *KnowDB_TermKDB_CV;
struct TypeQsemVal KnowDB_QsemVal; // running curve trends
struct TypeQsemVal KnowDB_QsemVal_CV; // running curve trends
struct TypeQbaseVal SemWE_Qbase;

///////////////////////////////////////
// function definition
void ReadWord(char *word, FILE *fin);
int  GetWordHash(char *word);
int  SearchVocab(char *word);
int  ReadWordIndex(FILE *fin);
void SemWE_AddToKnowDB_InSet(int word_index, long long eleID);
void SemWE_AddToKnowDB_CVSet(int word_index, long long eleID);
void SemWE_LoadInEquation_InSet(char *inequation_file);
void SemWE_LoadInEquation_CVSet(char *inequation_file);

real SemWE_VectorDot(real *vec_a, real *vec_b);
void SemWE_VectorLinear(real *main_vector, real *scale_vector, real scale_coeff);
void SemWE_VectorCopy(real *vec_a, real *vec_b);
void SemWE_VectorScale(real *input_vector, real scale_coeff);
real SemWE_VectorNorm(real *input_vector);
void SemWE_VectorMinus(real *minus_res, real *vec_a, real *vec_b);
real SemWE_CalcCosine(int index_A, int index_B);

real SemWE_DeriveHinge(real key_value); // 2014/09/30
real SemWE_CalcuHinge(real key_value);
void SemWE_QsemDerive_Cosine(int word_index, real* derive_distEu);
struct TypeQsemVal SemWE_Qsem_Cosine_InSet();
struct TypeQsemVal SemWE_Qsem_Cosine_CVSet();

//////////////////////////////////////////////
// Partial derivative
void SemWE_QsemDerive_Cosine(int word_index, real* derive_distEu)
{		
	int i = 0;
	int k = 0;
	int kdb_num;
	long long eleID = -1;
	struct TypeInEquation tmp_InEquation;
	real dist_AB = 0;
	real dist_CD = 0;
	int  index_A = -1;
	int  index_B = -1;
	int  index_C = -1;
	int  index_D = -1;
	real norm_A = 0.0;
	real norm_B = 0.0;
	real norm_C = 0.0;
	real norm_D = 0.0;
	real CD_minus_AB;
	real ABCD_Sigmoid;
	real derive_Sigmoid;
			
	real* derive_wordCD = (real*)malloc(layer1_size*sizeof(real));
	real* derive_wordAB = (real*)malloc(layer1_size*sizeof(real));
	
	SemWE_VectorCopy(derive_distEu, zero_vector);
	SemWE_VectorCopy(derive_wordAB, zero_vector);
	SemWE_VectorCopy(derive_wordCD, zero_vector);
				
	kdb_num = KnowDB_TermKDB[word_index].KDB_nums;
		
	for (i = 0; i < kdb_num; ++i)
	{
		eleID = KnowDB_TermKDB[word_index].KDB_IDset[i];	
		tmp_InEquation = KnowDB_InEquation[eleID];
		index_A = tmp_InEquation.index_A;
		index_B = tmp_InEquation.index_B;
		index_C = tmp_InEquation.index_C;
		index_D = tmp_InEquation.index_D;
		
		///
		norm_A = SemWE_VectorNorm(&syn0[index_A*layer1_size]);
		norm_B = SemWE_VectorNorm(&syn0[index_B*layer1_size]);
		norm_C = SemWE_VectorNorm(&syn0[index_C*layer1_size]);
		norm_D = SemWE_VectorNorm(&syn0[index_D*layer1_size]);
						
		dist_AB = SemWE_VectorDot(&syn0[index_A*layer1_size], &syn0[index_B*layer1_size]);
		dist_CD = SemWE_VectorDot(&syn0[index_C*layer1_size], &syn0[index_D*layer1_size]);
		dist_AB /= (norm_A*norm_B);
		dist_CD /= (norm_C*norm_D);
		
		// Distance minus Sigmoid
		// distance(word_C, word_D) - distance(word_A, word_B)
		CD_minus_AB = semwe_hinge_margin - (dist_AB - dist_CD);

		ABCD_Sigmoid   = SemWE_CalcuHinge(CD_minus_AB);
		derive_Sigmoid = SemWE_DeriveHinge(ABCD_Sigmoid);
		
		//printf("gap=%f norm=%f derive=%f\n", CD_minus_AB, ABCD_Sigmoid, derive_Sigmoid);
		if (derive_Sigmoid != 0.0)
		{				
			// Quan Liu, September 28, 2014. must add this!!!
			SemWE_VectorCopy(derive_wordAB, zero_vector);
			SemWE_VectorCopy(derive_wordCD, zero_vector);		
			
			if (word_index == index_A){
				SemWE_VectorCopy(derive_wordAB, &syn0[index_B*layer1_size]);			
				SemWE_VectorScale(derive_wordAB, (1.0/(norm_A*norm_B)));					
				SemWE_VectorLinear(derive_wordAB, &syn0[index_A*layer1_size], (-1*dist_AB/(norm_A*norm_A)));
			}        
			else if (word_index == index_B){
				SemWE_VectorCopy(derive_wordAB, &syn0[index_A*layer1_size]);			
				SemWE_VectorScale(derive_wordAB, (1.0/(norm_A*norm_B)));					
				SemWE_VectorLinear(derive_wordAB, &syn0[index_B*layer1_size], (-1*dist_AB/(norm_B*norm_B)));  
			} 
			if (word_index == index_C){
				SemWE_VectorCopy(derive_wordCD, &syn0[index_D*layer1_size]);			
				SemWE_VectorScale(derive_wordCD, (1.0/(norm_C*norm_D)));					
				SemWE_VectorLinear(derive_wordCD, &syn0[index_C*layer1_size], (-1*dist_CD/(norm_C*norm_C)));
				
			} 
			else if (word_index == index_D){
				SemWE_VectorCopy(derive_wordCD, &syn0[index_C*layer1_size]);			
				SemWE_VectorScale(derive_wordCD, (1.0/(norm_C*norm_D)));					
				SemWE_VectorLinear(derive_wordCD, &syn0[index_D*layer1_size], (-1*dist_CD/(norm_D*norm_D)));
			}
			///
			SemWE_VectorLinear(derive_wordCD, derive_wordAB, -1.0);
			SemWE_VectorLinear(derive_distEu, derive_wordCD, derive_Sigmoid);		
		}
	} 
	
	free(derive_wordAB);  derive_wordAB = NULL;
	free(derive_wordCD);  derive_wordCD = NULL;
}
// All the Know DB Qsem
struct TypeQsemVal SemWE_Qsem_Cosine_InSet()
{
	int i = 0;
	int kdb_num = 0;
	struct TypeQsemVal Qsem_Value;
	struct TypeInEquation tmp_InEquation;
	long long elemID = 0;
	real dist_AB = 0;
	real dist_CD = 0;
	int  index_A = -1;
	int  index_B = -1;
	int  index_C = -1;
	int  index_D = -1;
	real CD_minus_AB;
	real ABCD_Sigmoid;
	
	Qsem_Value.SatisfyNum = 0;
	Qsem_Value.Qsem_total = 0.0;
	Qsem_Value.Dgap_total = 0.0;

	kdb_num = KnowDB_QsemVal.totalKDBNum;
	for (elemID = 0; elemID < kdb_num; ++elemID)
	{
		tmp_InEquation = KnowDB_InEquation[elemID];
		index_A = tmp_InEquation.index_A;
		index_B = tmp_InEquation.index_B;
		index_C = tmp_InEquation.index_C;
		index_D = tmp_InEquation.index_D;
		
		dist_AB = SemWE_CalcCosine(index_A, index_B);
		dist_CD = SemWE_CalcCosine(index_C, index_D);

		// Distance minus Sigmoid
		// distance(word_C, word_D) - distance(word_A, word_B)
		CD_minus_AB = semwe_hinge_margin-(dist_AB -dist_CD);

		ABCD_Sigmoid = SemWE_CalcuHinge(CD_minus_AB);	

		Qsem_Value.Qsem_total += ABCD_Sigmoid;			
		Qsem_Value.Dgap_total += CD_minus_AB;
		
		if (dist_AB > dist_CD)
		{
			Qsem_Value.SatisfyNum++;
		}		
	}
	Qsem_Value.SatisfyRate = 1.0*Qsem_Value.SatisfyNum / KnowDB_QsemVal.totalKDBNum;	
	//printf("Qsem_Value.Qsem_total=%f, Qsem_Value.SatisfyRate=%f\n", Qsem_Value.Qsem_total, Qsem_Value.SatisfyRate);
	return Qsem_Value;
}

struct TypeQsemVal SemWE_Qsem_Cosine_CVSet()
{
	int i = 0;
	int kdb_num = 0;	
	struct TypeQsemVal Qsem_Value;
	struct TypeInEquation tmp_InEquation;	
	long long elemID = 0;
	real dist_AB = 0;
	real dist_CD = 0;
	int  index_A = -1;
	int  index_B = -1;
	int  index_C = -1;
	int  index_D = -1;
	real CD_minus_AB;
	real ABCD_Sigmoid;
	
	Qsem_Value.SatisfyNum = 0;
	Qsem_Value.Qsem_total = 0.0;
	Qsem_Value.Dgap_total = 0.0;

	kdb_num = KnowDB_QsemVal_CV.totalKDBNum;
	for (elemID = 0; elemID < kdb_num; ++elemID)
	{
		tmp_InEquation = KnowDB_InEquation_CV[elemID];
		index_A = tmp_InEquation.index_A;
		index_B = tmp_InEquation.index_B;
		index_C = tmp_InEquation.index_C;
		index_D = tmp_InEquation.index_D;

		dist_AB = SemWE_CalcCosine(index_A, index_B);
		dist_CD = SemWE_CalcCosine(index_C, index_D);

		// Distance minus Sigmoid
		// distance(word_C, word_D) - distance(word_A, word_B)
		CD_minus_AB = semwe_hinge_margin-(dist_AB -dist_CD);

		ABCD_Sigmoid   = SemWE_CalcuHinge(CD_minus_AB);	

		Qsem_Value.Qsem_total += ABCD_Sigmoid;			
		Qsem_Value.Dgap_total += CD_minus_AB;
		
		if (dist_AB > dist_CD)
		{
			Qsem_Value.SatisfyNum++;
		}		
	}
	Qsem_Value.SatisfyRate = 1.0*Qsem_Value.SatisfyNum / KnowDB_QsemVal_CV.totalKDBNum;	
	//printf("VALID: Qsem_Value.Qsem_total=%f, Qsem_Value.SatisfyRate=%f\n", Qsem_Value.Qsem_total, Qsem_Value.SatisfyRate);
	return Qsem_Value;
}


///////////////////////////////////////////////////
/// Math function
real SemWE_CalcuHinge(real key_value)
{
	// October 31, 2014
	if (key_value > 0){
		return key_value;
	}
	else{
		return 0;
	}
}
real SemWE_DeriveHinge(real key_value)
{	
	real core_derive = 0.0;
	if (key_value > 0){
		core_derive = 1;
	}
	else{
		core_derive = 0;
	}
	return core_derive;
}
//
real SemWE_VectorDot(real *vec_a, real *vec_b)
{
	/*int incx = 1;
	int incy = 1;
	int size = layer1_size;
	real fastres = 0.0;
	fastres = sdot(&size, vec_a, &incx, vec_b, &incy);
	return fastres;
	*/
	int  i = 0;
	int incx = 1;
	int incy = 1;
	int vec_size = layer1_size;
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
void SemWE_VectorCopy(real *vec_a, real *vec_b)
{
	// vec_a = vec_b
	/*int incx = 1;
	int incy = 1;
	int size = layer1_size;
	//int i = 0;		
	scopy(&size, vec_b, &incx, vec_a, &incy);
	*/
	int  i = 0;
	int incx = 1;
	int incy = 1;
	int vec_size = layer1_size;
	#ifdef MKL_YES
	scopy(&vec_size, vec_b, &incx, vec_a, &incy);
	#else
	for (i = 0; i < vec_size; ++i){
		vec_a[i] = vec_b[i];
	}
	#endif
}
void SemWE_VectorMinus(real *minus_res, real *vec_a, real *vec_b)
{
	int i = 0;
	for (i = 0; i < layer1_size; ++i){
		minus_res[i] = vec_a[i] - vec_b[i];
	}
}
void SemWE_VectorLinear(real *main_vector, real *scale_vector, real scale_coeff)
{
	// main_vector = main_vector + scale_coeff * scale_vector
	/*int incx = 1;
	int incy = 1;
	int size = layer1_size;
	saxpy(&size, &scale_coeff, scale_vector, &incx, main_vector, &incy);
	*/
	int  i = 0;
	int incx = 1;
	int incy = 1;
	int vec_size = layer1_size;
	#ifdef MKL_YES
	saxpy(&vec_size, &scale_coeff, scale_vector, &incx, main_vector, &incy);
	#else
	for (i = 0; i < vec_size; ++i){
		main_vector[i] += scale_coeff * scale_vector[i];
	}
	#endif
}
void SemWE_VectorScale(real *input_vector, real scale_coeff)
{
	/*int i = 0;
	int incx = 1;
	int size = layer1_size;
	sscal(&size, &scale_coeff, input_vector, &incx);
	*/

	int i = 0;
	int incx = 1;
	int vec_size = layer1_size;
	#ifdef MKL_YES
	sscal(&vec_size, &scale_coeff, input_vector, &incx);
	#else
	for (i = 0; i < vec_size; ++i){
		input_vector[i] = scale_coeff * input_vector[i];
	}
	#endif
}
real SemWE_VectorNorm(real *input_vector)
{
	/*int  i = 0;
	real norm_value = 0.0;
	int  incx = 1;
	int size = layer1_size;
	norm_value = snrm2(&size, input_vector, &incx);
	return norm_value;
	*/

	int i = 0;
	real norm_value = 0.0;
	int incx = 1;
	int vec_size = layer1_size;
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
real SemWE_CalcCosine(int index_A, int index_B)
{
	real cosine_value = 0.0;
	real norm_vec_a = 0.0;
	real norm_vec_b = 0.0;
	cosine_value = SemWE_VectorDot(&syn0[index_A*layer1_size], &syn0[index_B*layer1_size]);	
	norm_vec_a = SemWE_VectorNorm(&syn0[index_A*layer1_size]);
	norm_vec_b = SemWE_VectorNorm(&syn0[index_B*layer1_size]);	
	cosine_value = cosine_value/(norm_vec_a*norm_vec_b);
	//printf("cosine<%d %d>: %f\n", index_A, index_B, cosine_value);
	return cosine_value;
}


//////////////////////////////////////////////
// Load Knowledge InEquation
void SemWE_LoadInEquation_InSet(char *inequation_file)
{    
  int i = 0; 
  long long eleNum = 0;
  long long eleID = 0; 
  FILE *fKNOW = NULL;

  KnowDB_TermKDB = (struct TypeTermKDB*)malloc(vocab_size*sizeof(struct TypeTermKDB));
  
  for (i = 0; i < vocab_size; ++i)
  {
	KnowDB_TermKDB[i].KDB_IDset = (long long*)calloc(SEM_KDBNUM, sizeof(long long));
	KnowDB_TermKDB[i].KDB_nums  = 0;	
  }
  fKNOW = fopen(inequation_file, "rb");
  if (fKNOW == NULL){
    printf("ERROR, can not open file %s\n", inequation_file);
    exit(1);
  }    
  /// read inequation num
  /// file format: sim(word_A, word_B) > sim(word_C, word_D)
  while (1) {
    char word_A[MAX_STRING];
    char word_B[MAX_STRING];
    char word_C[MAX_STRING];
    char word_D[MAX_STRING];
    char word_T[MAX_STRING];
    ReadWord(word_A, fKNOW);
    ReadWord(word_B, fKNOW);
    ReadWord(word_C, fKNOW);
    ReadWord(word_D, fKNOW);
    ReadWord(word_T, fKNOW);
    if (feof(fKNOW)) break;    
    if (strcmp(word_T, "</s>") != 0) 
    {
      printf("please ensure 4 words at every line\n");
      exit(1);
    }    
    eleNum++;
  }
  fclose(fKNOW);
  printf("--- InEquation Nums: %lld\n",  eleNum);
  
  /// main read
  KnowDB_QsemVal.totalKDBNum = eleNum;
  KnowDB_InEquation = (struct TypeInEquation *)malloc(eleNum * sizeof(struct TypeInEquation));
  
  fKNOW = fopen(inequation_file, "r");
  /// file format: sim(word_A, word_B) > sim(word_C, word_D)  
  while (1) {
    char word_A[MAX_STRING];
    char word_B[MAX_STRING];
    char word_C[MAX_STRING];
    char word_D[MAX_STRING];
    char word_T[MAX_STRING];
    int  index_A = -1;
    int  index_B = -1;
    int  index_C = -1;
    int  index_D = -1;
    int  index_T = -1;
    ReadWord(word_A, fKNOW);
    ReadWord(word_B, fKNOW);
    ReadWord(word_C, fKNOW);
    ReadWord(word_D, fKNOW);
    ReadWord(word_T, fKNOW);  
    if (feof(fKNOW)) break;  

    index_A = SearchVocab(word_A);
    index_B = SearchVocab(word_B);
    index_C = SearchVocab(word_C);
    index_D = SearchVocab(word_D);       
    KnowDB_InEquation[eleID].index_A = index_A;
    KnowDB_InEquation[eleID].index_B = index_B;
    KnowDB_InEquation[eleID].index_C = index_C;
    KnowDB_InEquation[eleID].index_D = index_D;
    //KnowDB_InEquation[eleID].AB_minus_CD = 0.0;
	
    if (delta_left == 1)
    {
		SemWE_AddToKnowDB_InSet(index_A, eleID);
	    if (index_B != index_A) SemWE_AddToKnowDB_InSet(index_B, eleID);
	}	
	if (delta_right == 1)
	{
		if (index_C != index_B && index_C != index_A) SemWE_AddToKnowDB_InSet(index_C, eleID);  
		if (index_D != index_C && (index_D != index_B && index_D != index_A)) SemWE_AddToKnowDB_InSet(index_D, eleID);	
	}
	
	eleID++;    	    
  }
  fclose(fKNOW);  
  printf("--- Finish reading the Knowledge Database\n");
  //SemWE_SaveReadKDB();
}
void SemWE_AddToKnowDB_InSet(int word_index, long long eleID)
{ 
	int kdb_num = 0;
	KnowDB_TermKDB[word_index].KDB_nums++;
	kdb_num = KnowDB_TermKDB[word_index].KDB_nums;
	if (kdb_num > SEM_KDBNUM)
	{
		KnowDB_TermKDB[word_index].KDB_IDset = (long long*)realloc(KnowDB_TermKDB[word_index].KDB_IDset, kdb_num*sizeof(long long));
	}    
	KnowDB_TermKDB[word_index].KDB_IDset[kdb_num-1] = eleID;
}

void SemWE_LoadInEquation_CVSet(char *inequation_file)
{    
	int i = 0; 
	long long eleNum = 0;
	long long eleID = 0; 
	FILE *fKNOW = NULL;

	KnowDB_TermKDB_CV = (struct TypeTermKDB*)malloc(vocab_size*sizeof(struct TypeTermKDB));
	for (i = 0; i < vocab_size; ++i)
	{
		KnowDB_TermKDB_CV[i].KDB_IDset = (long long*)calloc(SEM_KDBNUM, sizeof(long long));
		KnowDB_TermKDB_CV[i].KDB_nums  = 0;
	}
	fKNOW = fopen(inequation_file, "rb");
	if (fKNOW == NULL){
		printf("ERROR, can not open file %s\n", inequation_file);
		exit(1);
	}    
	/// read inequation num
	/// file format: sim(word_A, word_B) > sim(word_C, word_D)
	while (1) {
		char word_A[MAX_STRING];
		char word_B[MAX_STRING];
		char word_C[MAX_STRING];
		char word_D[MAX_STRING];
		char word_T[MAX_STRING];		
		ReadWord(word_A, fKNOW);
		ReadWord(word_B, fKNOW);
		ReadWord(word_C, fKNOW);
		ReadWord(word_D, fKNOW);
		ReadWord(word_T, fKNOW);
		if (feof(fKNOW)) break;    
		if (strcmp(word_T, "</s>") != 0) 
		{
			printf("please ensure 4 words at every line\n");
			exit(1);
		}    
		eleNum++;
	}
	fclose(fKNOW);
	printf("--- CV set InEquation Nums: %lld\n",  eleNum);

	/// main read
	KnowDB_QsemVal_CV.totalKDBNum = eleNum;
	KnowDB_InEquation_CV = (struct TypeInEquation *)malloc(eleNum * sizeof(struct TypeInEquation));

	fKNOW = fopen(inequation_file, "r");
	/// file format: sim(word_A, word_B) > sim(word_C, word_D)  
	while (1) {
		char word_A[MAX_STRING];
		char word_B[MAX_STRING];
		char word_C[MAX_STRING];
		char word_D[MAX_STRING];
		char word_T[MAX_STRING];
		int  index_A = -1;
		int  index_B = -1;
		int  index_C = -1;
		int  index_D = -1;
		int  index_T = -1;
		ReadWord(word_A, fKNOW);
		ReadWord(word_B, fKNOW);
		ReadWord(word_C, fKNOW);
		ReadWord(word_D, fKNOW);
		ReadWord(word_T, fKNOW);  
		if (feof(fKNOW)) break;  

		index_A = SearchVocab(word_A);
		index_B = SearchVocab(word_B);
		index_C = SearchVocab(word_C);
		index_D = SearchVocab(word_D);       
		KnowDB_InEquation_CV[eleID].index_A = index_A;
		KnowDB_InEquation_CV[eleID].index_B = index_B;
		KnowDB_InEquation_CV[eleID].index_C = index_C;
		KnowDB_InEquation_CV[eleID].index_D = index_D;
		//KnowDB_InEquation[eleID].AB_minus_CD = 0.0;

		SemWE_AddToKnowDB_CVSet(index_A, eleID);
		if (index_B != index_A) SemWE_AddToKnowDB_CVSet(index_B, eleID);
		if (index_C != index_B && index_C != index_A) SemWE_AddToKnowDB_CVSet(index_C, eleID);
		if (index_D != index_C && (index_D != index_B && index_D != index_A)) SemWE_AddToKnowDB_CVSet(index_D, eleID);
		eleID++;			    
	}
	fclose(fKNOW);  
	printf("--- Finish reading the CV Knowledge Database\n");
}
void SemWE_AddToKnowDB_CVSet(int word_index, long long eleID)
{ 
	int kdb_num = 0;
	KnowDB_TermKDB_CV[word_index].KDB_nums++;
	kdb_num = KnowDB_TermKDB_CV[word_index].KDB_nums;
	if (kdb_num > SEM_KDBNUM)
	{
		KnowDB_TermKDB_CV[word_index].KDB_IDset = (long long*)realloc(KnowDB_TermKDB_CV[word_index].KDB_IDset, kdb_num*sizeof(long long));
	}    
	KnowDB_TermKDB_CV[word_index].KDB_IDset[kdb_num-1] = eleID;
}

//////////////////////////////////////////////
void InitUnigramTable() {
  int a, i;
  long long train_words_pow = 0;
  real d1, power = 0.75;
  table = (int *)malloc(table_size * sizeof(int));
  for (a = 0; a < vocab_size; a++) train_words_pow += pow(vocab[a].cn, power);
  i = 0;
  d1 = pow(vocab[i].cn, power) / (real)train_words_pow;
  for (a = 0; a < table_size; a++) {
    table[a] = i;
    if (a / (real)table_size > d1) {
      i++;
      d1 += pow(vocab[i].cn, power) / (real)train_words_pow;
    }
    if (i >= vocab_size) i = vocab_size - 1;
  }
}

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
void ReadWord(char *word, FILE *fin) {
  int a = 0, ch;
  while (!feof(fin)) {
    ch = fgetc(fin);
    if (ch == 13) continue;
    if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
      if (a > 0) {
        if (ch == '\n') ungetc(ch, fin);
        break;
      }
      if (ch == '\n') {
        strcpy(word, (char *)"</s>");
        return;
      } else continue;
    }
    word[a] = ch;
    a++;
    if (a >= MAX_STRING - 1) a--;   // Truncate too long words
  }
  word[a] = 0;
}

// Returns hash value of a word
int GetWordHash(char *word) {
  unsigned long long a, hash = 0;
  for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
  hash = hash % vocab_hash_size;
  return hash;
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
int SearchVocab(char *word) {
  unsigned int hash = GetWordHash(word);
  while (1) {
    if (vocab_hash[hash] == -1) return -1;
    if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash];
    hash = (hash + 1) % vocab_hash_size;
  }
  return -1;
}

// Reads a word and returns its index in the vocabulary
int ReadWordIndex(FILE *fin) {
  char word[MAX_STRING];
  ReadWord(word, fin);
  if (feof(fin)) return -1;
  return SearchVocab(word);
}

// Adds a word to the vocabulary
int AddWordToVocab(char *word) {
  unsigned int hash, length = strlen(word) + 1;
  if (length > MAX_STRING) length = MAX_STRING;
  vocab[vocab_size].word = (char *)calloc(length, sizeof(char));
  strcpy(vocab[vocab_size].word, word);
  vocab[vocab_size].cn = 0;
  vocab_size++;
  // Reallocate memory if needed
  if (vocab_size + 2 >= vocab_max_size) {
    vocab_max_size += 1000;
    vocab = (struct vocab_word *)realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
  }
  hash = GetWordHash(word);
  while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
  vocab_hash[hash] = vocab_size - 1;
  return vocab_size - 1;
}

// Used later for sorting by word counts
int VocabCompare(const void *a, const void *b) {
    return ((struct vocab_word *)b)->cn - ((struct vocab_word *)a)->cn;
}

// Sorts the vocabulary by frequency using word counts
void SortVocab() {
  int a, size;
  unsigned int hash;
  // Sort the vocabulary and keep </s> at the first position
  qsort(&vocab[1], vocab_size - 1, sizeof(struct vocab_word), VocabCompare);
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  size = vocab_size;
  train_words = 0;
  for (a = 0; a < size; a++) {
    // Words occuring less than min_count times will be discarded from the vocab
    if ((vocab[a].cn < min_count) && (a != 0)) {
      vocab_size--;
      free(vocab[a].word);
    } else {
      // Hash will be re-computed, as after the sorting it is not actual
      hash=GetWordHash(vocab[a].word);
      while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
      vocab_hash[hash] = a;
      train_words += vocab[a].cn;
    }
  }
  vocab = (struct vocab_word *)realloc(vocab, (vocab_size + 1) * sizeof(struct vocab_word));
  // Allocate memory for the binary tree construction
  for (a = 0; a < vocab_size; a++) {
    vocab[a].code = (char *)calloc(MAX_CODE_LENGTH, sizeof(char));
    vocab[a].point = (int *)calloc(MAX_CODE_LENGTH, sizeof(int));
  }
}

// Reduces the vocabulary by removing infrequent tokens
void ReduceVocab() {
  int a, b = 0;
  unsigned int hash;
  for (a = 0; a < vocab_size; a++) if (vocab[a].cn > min_reduce) {
    vocab[b].cn = vocab[a].cn;
    vocab[b].word = vocab[a].word;
    b++;
  } else free(vocab[a].word);
  vocab_size = b;
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  for (a = 0; a < vocab_size; a++) {
    // Hash will be re-computed, as it is not actual
    hash = GetWordHash(vocab[a].word);
    while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
    vocab_hash[hash] = a;
  }
  fflush(stdout);
  min_reduce++;
}

// Create binary Huffman tree using the word counts
// Frequent words will have short uniqe binary codes
void CreateBinaryTree() {
  long long a, b, i, min1i, min2i, pos1, pos2, point[MAX_CODE_LENGTH];
  char code[MAX_CODE_LENGTH];
  long long *count = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  long long *binary = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  long long *parent_node = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  for (a = 0; a < vocab_size; a++) count[a] = vocab[a].cn;
  for (a = vocab_size; a < vocab_size * 2; a++) count[a] = 1e15;
  pos1 = vocab_size - 1;
  pos2 = vocab_size;
  // Following algorithm constructs the Huffman tree by adding one node at a time
  for (a = 0; a < vocab_size - 1; a++) {
    // First, find two smallest nodes 'min1, min2'
    if (pos1 >= 0) {
      if (count[pos1] < count[pos2]) {
        min1i = pos1;
        pos1--;
      } else {
        min1i = pos2;
        pos2++;
      }
    } else {
      min1i = pos2;
      pos2++;
    }
    if (pos1 >= 0) {
      if (count[pos1] < count[pos2]) {
        min2i = pos1;
        pos1--;
      } else {
        min2i = pos2;
        pos2++;
      }
    } else {
      min2i = pos2;
      pos2++;
    }
    count[vocab_size + a] = count[min1i] + count[min2i];
    parent_node[min1i] = vocab_size + a;
    parent_node[min2i] = vocab_size + a;
    binary[min2i] = 1;
  }
  // Now assign binary code to each vocabulary word
  for (a = 0; a < vocab_size; a++) {
    b = a;
    i = 0;
    while (1) {
      code[i] = binary[b];
      point[i] = b;
      i++;
      b = parent_node[b];
      if (b == vocab_size * 2 - 2) break;
    }
    vocab[a].codelen = i;
    vocab[a].point[0] = vocab_size - 2;
    for (b = 0; b < i; b++) {
      vocab[a].code[i - b - 1] = code[b];
      vocab[a].point[i - b] = point[b] - vocab_size;
    }
  }
  free(count);
  free(binary);
  free(parent_node);
}

void LearnVocabFromTrainFile() {
  char word[MAX_STRING];
  FILE *fin;
  long long a, i;
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  vocab_size = 0;
  AddWordToVocab((char *)"</s>");
  while (1) {
    ReadWord(word, fin);
    if (feof(fin)) break;
    train_words++;
    if ((debug_mode > 1) && (train_words % 100000 == 0)) {
      printf("%lldK%c", train_words / 1000, 13);
      fflush(stdout);
    }
    i = SearchVocab(word);
    if (i == -1) {
      a = AddWordToVocab(word);
      vocab[a].cn = 1;
    } else vocab[i].cn++;
    if (vocab_size > vocab_hash_size * 0.7) ReduceVocab();
  }
  SortVocab();
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
  file_size = ftell(fin);
  fclose(fin);
}

void SaveVocab() {
  long long i;
  FILE *fo = fopen(save_vocab_file, "wb");
  for (i = 0; i < vocab_size; i++) fprintf(fo, "%s %lld\n", vocab[i].word, vocab[i].cn);
  fclose(fo);
}

void ReadVocab() {
  long long a, i = 0;
  char c;
  char word[MAX_STRING];
  long long wordcn = 0;
  FILE *fin = fopen(read_vocab_file, "rb");
  if (fin == NULL) {
    printf("Vocabulary file not found\n");
    exit(1);
  }
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  vocab_size = 0;
  
  a = AddWordToVocab((char *)"</s>");
  vocab[a].cn = 100;

  while (fscanf(fin, "%s%lld", word, &wordcn) != EOF){
	  a = AddWordToVocab(word);
	  vocab[a].cn = wordcn;
	  //printf("%s %lld\n", word, vocab[a].cn);
  }
  /*while (1) {
    ReadWord(word, fin);
    if (feof(fin)) break;
    a = AddWordToVocab(word);
    fscanf(fin, "%lld%c", &vocab[a].cn, &c);
    i++;
  }*/
  
  SortVocab();
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  fseek(fin, 0, SEEK_END);
  file_size = ftell(fin);
  fclose(fin);
}

void InitNet() {
	long long a, b;
	unsigned long long next_random = 1;
	
	//a = posix_memalign((void **)&syn0, 128, (long long)vocab_size * layer1_size * sizeof(real));
#ifdef ON_LINUX
	a = posix_memalign((void **)&syn0, 128, (long long)vocab_size * layer1_size * sizeof(real));
#endif
#ifdef ON_WINDOWS
	syn0 = (real *)malloc((long long)vocab_size * layer1_size * sizeof(real));
#endif
	if (syn0 == NULL) {printf("Memory allocation failed\n"); exit(1);}
	
	if (hs) {
	
	//a = posix_memalign((void **)&syn1, 128, (long long)vocab_size * layer1_size * sizeof(real));
#ifdef ON_LINUX
	a = posix_memalign((void **)&syn1, 128, (long long)vocab_size * layer1_size * sizeof(real));
#endif
#ifdef ON_WINDOWS
	syn1 = (real *) malloc((long long)vocab_size * layer1_size * sizeof(real));
#endif
	
	if (syn1 == NULL) {printf("Memory allocation failed\n"); exit(1);}
	for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++)
	 syn1[a * layer1_size + b] = 0;
	}
	if (negative>0) {
	  //a = posix_memalign((void **)&syn1neg, 128, (long long)vocab_size * layer1_size * sizeof(real));
#ifdef ON_LINUX
	  a = posix_memalign((void **)&syn1neg, 128, (long long)vocab_size * layer1_size * sizeof(real));
#endif
#ifdef ON_WINDOWS
	 syn1neg = (real *)malloc((long long)vocab_size * layer1_size * sizeof(real));
#endif	  
	
	if (syn1neg == NULL) {printf("Memory allocation failed\n"); exit(1);}
	for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++)
		syn1neg[a * layer1_size + b] = 0;
	}
	for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++) {
		next_random = next_random * (unsigned long long)25214903917 + 11;
		syn0[a * layer1_size + b] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / layer1_size;
	}
	CreateBinaryTree();
}

#ifdef ON_LINUX
void *TrainModelThread(void *id) {
#endif
#ifdef ON_WINDOWS
	unsigned int __stdcall TrainModelThread(void *id){
#endif
  long long a, b, d, cw, word, last_word, sentence_length = 0, sentence_position = 0;
  long long word_count = 0, last_word_count = 0, sen[MAX_SENTENCE_LENGTH + 1];
  long long l1, l2, c, target, label, local_iter = iter;
  unsigned long long next_random = (long long)id;
  real f, g;
  clock_t now;
  
  long long threadID = (long long)id;
  real run_process;
  real run_speed;
  struct TypeQsemVal tmp_QsemValue;
  struct TypeQsemVal tmp_QsemValueCV;
  long long last_word_count2 = 0;
	
  real* derive_distEu = (real*)malloc(layer1_size*sizeof(real));
  real *vector_decay = (real*) malloc(layer1_size*sizeof(real));

  real *neu1 = (real *)calloc(layer1_size, sizeof(real));
  real *neu1e = (real *)calloc(layer1_size, sizeof(real));
  FILE *fi = fopen(train_file, "rb");
  fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
  while (1) {
    if (word_count - last_word_count > 10000) {
      word_count_actual += word_count - last_word_count;
      last_word_count = word_count;
	  ///
	  run_process = word_count_actual / (real)(train_words + 1) * 100;
	  		
      alpha = starting_alpha * (1 - word_count_actual / (real)(iter * train_words + 1));
      if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
    }
	
	//////////////////////////
	if (word_count - last_word_count2 > CVTEST_TIME) {	// 1 million			
			last_word_count2 = word_count;
						
			// InSet
			tmp_QsemValue = SemWE_Qsem_Cosine_InSet();
			KnowDB_QsemVal.Qsem_total = tmp_QsemValue.Qsem_total;			
			KnowDB_QsemVal.SatisfyNum = tmp_QsemValue.SatisfyNum;
			KnowDB_QsemVal.SatisfyRate = tmp_QsemValue.SatisfyRate;		
			// CVSet
			tmp_QsemValueCV = SemWE_Qsem_Cosine_CVSet();
			KnowDB_QsemVal_CV.Qsem_total = tmp_QsemValueCV.Qsem_total;			
			KnowDB_QsemVal_CV.SatisfyNum = tmp_QsemValueCV.SatisfyNum;
			KnowDB_QsemVal_CV.SatisfyRate = tmp_QsemValueCV.SatisfyRate;	
			////printf("--- WordRun: %lld  Qsem: %f\n", word_count_actual, KnowDB_QsemVal.Qsem_total);
			
			printf("--- Alpha: %f  Progress: %.4f%%  Thread: %lld  ThreadCount: %lld  Train_Qsem: %.4f  Train_SatisfyRate: %.4f  Valid_Qsem: %.4f  Valid_SatisfyRate: %.4f\n", 
			alpha, run_process, threadID,
			word_count, KnowDB_QsemVal.Qsem_total, KnowDB_QsemVal.SatisfyRate, KnowDB_QsemVal_CV.Qsem_total, KnowDB_QsemVal_CV.SatisfyRate);
			
		}

    if (sentence_length == 0) {
      while (1) {
        word = ReadWordIndex(fi);
        if (feof(fi)) break;
        if (word == -1) continue;
        word_count++;
        if (word == 0) break;
        // The subsampling randomly discards frequent words while keeping the ranking same
        if (sample > 0) {
          real ran = (sqrt(vocab[word].cn / (sample * train_words)) + 1) * (sample * train_words) / vocab[word].cn;
          next_random = next_random * (unsigned long long)25214903917 + 11;
          if (ran < (next_random & 0xFFFF) / (real)65536) continue;
        }
        sen[sentence_length] = word;
        sentence_length++;
        if (sentence_length >= MAX_SENTENCE_LENGTH) break;
      }
      sentence_position = 0;
    }
    if (feof(fi) || (word_count > train_words / num_threads)) {
      word_count_actual += word_count - last_word_count;
      local_iter--;
      if (local_iter == 0) break;
      word_count = 0;
      last_word_count = 0;
      sentence_length = 0;
      fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
      continue;
    }
    word = sen[sentence_position];
    if (word == -1) continue;
    for (c = 0; c < layer1_size; c++) neu1[c] = 0;
    for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
    next_random = next_random * (unsigned long long)25214903917 + 11;
    b = next_random % window;
    if (cbow) {  //train the cbow architecture
      // in -> hidden
      cw = 0;
      for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
        c = sentence_position - window + a;
        if (c < 0) continue;
        if (c >= sentence_length) continue;
        last_word = sen[c];
        if (last_word == -1) continue;
        for (c = 0; c < layer1_size; c++) neu1[c] += syn0[c + last_word * layer1_size];
        cw++;
      }
      if (cw) {
        for (c = 0; c < layer1_size; c++) neu1[c] /= cw;
        if (hs) for (d = 0; d < vocab[word].codelen; d++) {
          f = 0;
          l2 = vocab[word].point[d] * layer1_size;
          // Propagate hidden -> output
          for (c = 0; c < layer1_size; c++) f += neu1[c] * syn1[c + l2];
          if (f <= -MAX_EXP) continue;
          else if (f >= MAX_EXP) continue;
          else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
          // 'g' is the gradient multiplied by the learning rate
          g = (1 - vocab[word].code[d] - f) * alpha;
          // Propagate errors output -> hidden
          for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1[c + l2];
          // Learn weights hidden -> output
          for (c = 0; c < layer1_size; c++) syn1[c + l2] += g * neu1[c];
        }
        // NEGATIVE SAMPLING
        if (negative > 0) for (d = 0; d < negative + 1; d++) {
          if (d == 0) {
            target = word;
            label = 1;
          } else {
            next_random = next_random * (unsigned long long)25214903917 + 11;
            target = table[(next_random >> 16) % table_size];
            if (target == 0) target = next_random % (vocab_size - 1) + 1;
            if (target == word) continue;
            label = 0;
          }
          l2 = target * layer1_size;
          f = 0;
          for (c = 0; c < layer1_size; c++) f += neu1[c] * syn1neg[c + l2];
          if (f > MAX_EXP) g = (label - 1) * alpha;
          else if (f < -MAX_EXP) g = (label - 0) * alpha;
          else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
          for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];
          for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * neu1[c];
        }
        // hidden -> in
        for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
          c = sentence_position - window + a;
          if (c < 0) continue;
          if (c >= sentence_length) continue;
          last_word = sen[c];
          if (last_word == -1) continue;
          for (c = 0; c < layer1_size; c++) syn0[c + last_word * layer1_size] += neu1e[c];
        }
      }
    } 
	else 
	{ 
	  /////////////////////////////////
	  // Add semantic constraint !!//
	  /////////////////////////////////
	  if (semwe_inter_coeff > 0.0 && (run_process > semwe_add_time && KnowDB_TermKDB[word].KDB_nums >= 1))
		{
			SemWE_QsemDerive_Cosine(word, derive_distEu);												
			
			SemWE_VectorScale(derive_distEu, (-1.0*semwe_inter_coeff * alpha));
			
			// weight deacy
			if (semwe_weight_decay > 0.0) 
			{
				SemWE_VectorCopy(vector_decay, &syn0[word * layer1_size]);
				SemWE_VectorLinear(&syn0[word * layer1_size], vector_decay, (-1.0*alpha*semwe_weight_decay));
			}
			SemWE_VectorLinear(&syn0[word * layer1_size], derive_distEu, 1.0);
		}
	  
	  //////////////////////////////////
	  //train skip-gram	  
      for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
        c = sentence_position - window + a;
        if (c < 0) continue;
        if (c >= sentence_length) continue;
        last_word = sen[c];
        if (last_word == -1) continue;
        
		//l1 = last_word * layer1_size;
		l1 = word * layer1_size;
        
		for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
        				
        // NEGATIVE SAMPLING
        if (negative > 0) for (d = 0; d < negative + 1; d++) {
          if (d == 0) {
            target = last_word;
            label = 1;
          } else {
            next_random = next_random * (unsigned long long)25214903917 + 11;
            target = table[(next_random >> 16) % table_size];
            if (target == 0) target = next_random % (vocab_size - 1) + 1;
            if (target == last_word) continue;
            label = 0;
          }
          l2 = target * layer1_size;
          f = 0;
          for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1neg[c + l2];
          if (f > MAX_EXP) g = (label - 1) * alpha;
          else if (f < -MAX_EXP) g = (label - 0) * alpha;
          else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
          
		  for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];
          for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * syn0[c + l1];
        }
		
        // Learn weights input -> hidden
        for (c = 0; c < layer1_size; c++) syn0[c + l1] += neu1e[c];
      }
    }
    sentence_position++;
    if (sentence_position >= sentence_length) {
      sentence_length = 0;
      continue;
    }
  }
  fclose(fi);
  free(neu1);
  free(neu1e);
  free(vector_decay);
  free(derive_distEu);
#ifdef ON_LINUX
  pthread_exit(NULL);
#endif
#ifdef ON_WINDOWS
  return 0;
#endif
}


//// Core function////
void MicrosoftCompletion() 
{
  long long a, b, d, cw, word, last_word, sentence_length = 0, sentence_position = 0;
  long long word_count = 0, last_word_count = 0, sen[MAX_SENTENCE_LENGTH + 1];
  long long l1, l2, c, target, label, local_iter = iter;
  
  //////
  int vocab_idx = 0;
  int nRow = vocab_size;
  int nCol = layer1_size;
  int incx = 1;
  int incy = 1;
  int lda = nCol;
  float alpha = 1.0;
  float beta = 0.0;  
  float softmax_prob = 0.0;
  float softmax_sum = 0.0;
  float word_prior = 0.0; // prior probability, unigram occurrence.
  int test_num = 0;
	  
  real *softmax_fenmu = (real*)malloc(vocab_size*sizeof(real));
  
  ///
  real f;  
  real sent_sigmoid = 0.0;  
  real sent_softmax = 0.0; // word prob calculation
  real sent_cosine  = 0.0; // cosine similarity sum up!
  real veclen_A = 0.0;
  real veclen_B = 0.0;
  real cosine_sim = 0.0;
  char word_str[MAX_STRING];
  char outfile_softmax[FILE_LENGTH];
  char outfile_cosine[FILE_LENGTH];
  sprintf(outfile_softmax, "%s.softmax", MSRSC_output);
  sprintf(outfile_cosine, "%s.cosine", MSRSC_output);
  
  for (a = 0; a < vocab_size; a++) {
	  softmax_fenmu[a] = 0.0;
  }

  FILE *fi = fopen(MSRSC_input, "rb");
  FILE *fo = fopen(MSRSC_output, "wb");	
  FILE *fc = fopen(outfile_cosine, "wb");
  
  printf("--- Word Occurrence Prob Estimation By Word Embedding\n");
  printf("--- Application: a)N-Best-Rescore; b)Sentence-Completion\n");
  printf("--- source file: %s\n", MSRSC_input);
  printf("--- output res1: sigmoid, %s\n", MSRSC_output);
	//printf("--- output res2: softmax, %s\n", outfile_softmax);
  printf("--- output res3: cosine, %s\n", outfile_cosine);
  
  while (1) {    
    if (sentence_length == 0) {
	  
	  test_num++;
      if (test_num % 500 == 0)
      {
		  printf("--- current test question num: %d\n", test_num);
      }
      ///
      while (1) {
        		
		ReadWord(word_str, fi);
		if (feof(fi)) break;
		word = SearchVocab(word_str);
		
		if (word != 0){
			fprintf(fo, "%s ", word_str);
			//fprintf(fn, "%s ", word_str);
			fprintf(fc, "%s ", word_str);
		}
		
        if (feof(fi)) break;
        if (word == -1) continue;
        		
        if (word == 0) break;
        // The subsampling randomly discards frequent words while keeping the ranking same
        /*if (sample > 0) {
          real ran = (sqrt(vocab[word].cn / (sample * train_words)) + 1) * (sample * train_words) / vocab[word].cn;
          next_random = next_random * (unsigned long long)25214903917 + 11;
          if (ran < (next_random & 0xFFFF) / (real)65536) continue;
        }*/
        sen[sentence_length] = word;
        sentence_length++;
        if (sentence_length >= MAX_SENTENCE_LENGTH) break;
      }
      sentence_position = 0;
    }
    if (feof(fi))  break;
	
	sent_sigmoid = 0.0;
	sent_softmax = 0.0;
    sent_cosine  = 0.0;
	word = sen[sentence_position];
    if (word == -1) continue;
	
	// word prior information
	word_prior = 1.0*vocab[word].cn / train_words;
	//printf("-- word: %s, prior: %d, %f, %f\n", vocab[word].word, vocab[word].cn, word_prior, log(word_prior));
	veclen_A = 0.0;
	for (c = 0; c < layer1_size; c++){
		l1 = word * layer1_size;
		veclen_A += pow(syn0[c + l1], 2);
	}
	veclen_A = sqrt(veclen_A);
	
	for (a = 1; a < sentence_length; a++)
	{
		last_word = sen[a];
		if (last_word == -1) continue;
		
		l1 = word * layer1_size;	
		l2 = last_word * layer1_size;
				
		f = 0;
		veclen_B = 0.0;
		cosine_sim = 0.0;
		for (c = 0; c < layer1_size; c++)
		{
			f += syn0[c + l1] * syn1neg[c + l2];			
			cosine_sim += syn0[c + l1] * syn0[c + l2];
			veclen_B += pow(syn0[c + l2], 2);
		}
		
		// Co-ocurrence probability (association)
		sent_sigmoid += (1.0/(1.0+exp(-1.0*f)));
		
		veclen_B = sqrt(veclen_B);
		if (veclen_B > 0.0 && veclen_A > 0.0)
		{
			cosine_sim /= (veclen_A * veclen_B);
			sent_cosine += cosine_sim;
		}			
	}	
	
	// print output
	fprintf(fo, "%f\n", sent_sigmoid);
	fprintf(fc, "%f\n", sent_cosine);
	sentence_length = 0;
    continue;
  }
  fclose(fi);
  fclose(fo);
  //fclose(fn);
  fclose(fc);
  free(softmax_fenmu);
  printf("--- Completion Finished!\n");
}


void TrainModel() {
  long a, b, c, d;
  FILE *fIN;
	
  printf("--- step0: read vocabulary and init network...");
  ReadVocab();
  //if (read_vocab_file[0] != 0) ReadVocab(); else LearnVocabFromTrainFile();
  //if (save_vocab_file[0] != 0) SaveVocab();
  if (output_file[0] == 0) return;
  InitNet();
  if (negative > 0) InitUnigramTable();
  printf("done\n");
  
  /////////////////////////
  printf("--- step1: loading word embedding model...");
  long long size_1, size_2;
  char tmp_word[1024];
  real tmp_value = 0.0;
  
  fIN = fopen(embeded_file, "r");
  fscanf(fIN, "%lld%lld", &size_1, &size_2);
  printf("load embedding: vocab_size=%lld, embeded_dim=%lld ...", size_1, size_2);
	if (size_1 != vocab_size) {
		printf("You are loading a word embedding with different size (differ vocab: %lld %lld)\n", size_1, vocab_size);
		exit(1);
	}
	if (size_2 != layer1_size) {
		printf("You are loading a word embedding with different size (differ embed: %lld %lld)\n", size_2, layer1_size);
		exit(1);
	}
	for (a = 0; a < vocab_size; a++) {
		fscanf(fIN, "%s", tmp_word);	
		if (strcmp(tmp_word, vocab[a].word) != 0)
		{
			printf("!!!error, word mismatch\n");
			exit(1);
		}
		for (b = 0; b < layer1_size; b++) {
			fscanf(fIN, "%f", &tmp_value);				
			syn0[a * layer1_size + b] = tmp_value;
			//printf("%f ", syn0[a * layer1_size + b]);
		}
		//printf("\n");
	}
  fclose(fIN);
  printf("done\n");

  /////// predict
  //sprintf(predict_file, "%s.predict", output_file);
  fIN = fopen(predict_file, "r");
	fscanf(fIN, "%lld%lld", &size_1, &size_2);
	printf("load predict file: vocab_size=%lld, embeded_dim=%lld ...", size_1, size_2);
	if (size_1 != vocab_size) {
		printf("You are loading a word embedding with different size (differ vocab: %lld %lld)\n", size_1, vocab_size);
		exit(1);
	}
	if (size_2 != layer1_size) {
		printf("You are loading a word embedding with different size (differ embed: %lld %lld)\n", size_2, layer1_size);
		exit(1);
	}
	for (a = 0; a < vocab_size; a++) {
		fscanf(fIN, "%s", tmp_word);	
		if (strcmp(tmp_word, vocab[a].word) != 0)
		{
			printf("!!!error, word mismatch\n");
			exit(1);
		}
		for (b = 0; b < layer1_size; b++) {
			fscanf(fIN, "%f", &tmp_value);				
			syn1neg[a * layer1_size + b] = tmp_value;
			//printf("%f ", syn0[a * layer1_size + b]);
		}
		//printf("\n");
	}
	fclose(fIN);
  printf("done\n");
  
  /// Main work
  printf("--- step2: Microsoft Sentence Completion liked Evaluation...");
  MicrosoftCompletion();  
  printf("done\n");
}

int ArgPos(char *str, int argc, char **argv) {
  int a;
  for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
    if (a == argc - 1) {
      printf("Argument missing for %s\n", str);
      exit(1);
    }
    return a;
  }
  return -1;
}

int main(int argc, char **argv) {
	int i;
	FILE *fTest= NULL;

	if (argc == 1) {
	printf("SWE toolkit: Test ASR Nbest / MSR Sentence Completion\n\n");
	printf("Options:\n");
	printf("Parameters for training:\n");
	printf("\t-train <file>\n");
	printf("\t\tUse text data from <file> to train the model\n");
	printf("\t-output <file>\n");
	printf("\t\tUse <file> to save the resulting word vectors / word clusters\n");
	printf("\t-size <int>\n");
	printf("\t\tSet size of word vectors; default is 100\n");
	printf("\t-window <int>\n");
	printf("\t\tSet max skip length between words; default is 5\n");
	printf("\t-sample <float>\n");
	printf("\t\tSet threshold for occurrence of words. Those that appear with higher frequency in the training data\n");
	printf("\t\twill be randomly down-sampled; default is 1e-3, useful range is (0, 1e-5)\n");
	printf("\t-hs <int>\n");
	printf("\t\tUse Hierarchical Softmax; default is 0 (not used)\n");
	printf("\t-negative <int>\n");
	printf("\t\tNumber of negative examples; default is 5, common values are 3 - 10 (0 = not used)\n");
	printf("\t-threads <int>\n");
	printf("\t\tUse <int> threads (default 12)\n");
	printf("\t-iter <int>\n");
	printf("\t\tRun more training iterations (default 5)\n");
	printf("\t-min-count <int>\n");
	printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");
	printf("\t-alpha <float>\n");
	printf("\t\tSet the starting learning rate; default is 0.025 for skip-gram and 0.05 for CBOW\n");
	printf("\t-classes <int>\n");
	printf("\t\tOutput word classes rather than word vectors; default number of classes is 0 (vectors are written)\n");
	printf("\t-debug <int>\n");
	printf("\t\tSet the debug mode (default = 2 = more info during training)\n");
	printf("\t-binary <int>\n");
	printf("\t\tSave the resulting vectors in binary moded; default is 0 (off)\n");
	printf("\t-save-vocab <file>\n");
	printf("\t\tThe vocabulary will be saved to <file>\n");
	printf("\t-read-vocab <file>\n");
	printf("\t\tThe vocabulary will be read from <file>, not constructed from the training data\n");
	printf("\t-cbow <int>\n");
	printf("\t\tUse the continuous bag of words model; default is 1 (use 0 for skip-gram model)\n");
	
	/// SWE Settings
	printf("\nSWE parameter setting\n");
	printf("\t-sem-train <file>\n");
	printf("\t\tSet the semantic constraint training set to <file>\n");
	printf("\t-sem-valid <file>\n");
	printf("\t\tSet the semantic constraint validation set to <file>\n");
	printf("\t-sem-coeff <float>\n");
	printf("\t\tSet the SWE combination coeff value; default is 0.1\n");
	printf("\t-sem-hinge <float>\n");
	printf("\t\tSet the gate value of hinge function for SWE. default is 0.0\n");
	printf("\t-sem-addtime <float>\n");
	printf("\t\tSet the time (in process: %) for adding semantic constraint. default is 0 (add at the begining)\n");
	printf("\t-delta-left <int>\n");
	printf("\t\tUse the left word pair of each similarity inequality or not. default is 1\n");
	printf("\t-delta-right <int>\n");
	printf("\t\tUse the right word pair of each similarity inequality or not. default is 1\n");
	printf("\t-weight-decay <float>\n");
	printf("\t\tSet weight decay coeffcient. default is 0\n");
	
	///
	printf("\nExamples:\n");
	printf("./SWE_InEquation -train data.txt -size 200 -window 5 -sample 1e-4 -negative 5 -hs 0 -binary 0 -cbow 0 -iter 3 -sem-train sem.train.txt -sem-valid sem.valid.txt -sem-coeff 0.1 -sem-hinge 0.0 -sem-addtime 0 -weight-decay 0 -delta-left 1 -delta-right 1 -sent-in test.sent.src -sent-out test.sent.out -load-embeded word.vector.txt -load-predict word.predict.txt\n\n");
	return 0;
	}
	output_file[0] = 0;
	save_vocab_file[0] = 0;
	read_vocab_file[0] = 0;
	if ((i = ArgPos((char *)"-size", argc, argv)) > 0) layer1_size = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-save-vocab", argc, argv)) > 0) strcpy(save_vocab_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-read-vocab", argc, argv)) > 0) strcpy(read_vocab_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-cbow", argc, argv)) > 0) cbow = atoi(argv[i + 1]);
	if (cbow) alpha = 0.05;
	if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
	if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);
	if ((i = ArgPos((char *)"-hs", argc, argv)) > 0) hs = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-iter", argc, argv)) > 0) iter = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-classes", argc, argv)) > 0) classes = atoi(argv[i + 1]);
	
	//
	if ((i = ArgPos((char *)"-sem-train", argc, argv)) > 0) strcpy(semwe_inequation_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-sem-valid", argc, argv)) > 0) strcpy(semwe_inequation_fileCV, argv[i + 1]);	
	if ((i = ArgPos((char *)"-sem-coeff", argc, argv)) > 0) semwe_inter_coeff = atof(argv[i + 1]);
	if ((i = ArgPos((char *)"-sem-addtime", argc, argv)) > 0) semwe_add_time = atof(argv[i + 1]); // 1 -> 100
	if ((i = ArgPos((char *)"-weight-decay", argc, argv)) > 0) semwe_weight_decay = atof(argv[i + 1]); //
	if ((i = ArgPos((char *)"-sem-hinge", argc, argv)) > 0) semwe_hinge_margin = atof(argv[i + 1]); 
	
	if ((i = ArgPos((char *)"-delta-left", argc, argv)) > 0) delta_left = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-delta-right", argc, argv)) > 0) delta_right = atoi(argv[i + 1]);
		
	// Core
	if ((i = ArgPos((char *)"-sent-in", argc, argv)) > 0) strcpy(MSRSC_input, argv[i + 1]);
	if ((i = ArgPos((char *)"-sent-out", argc, argv)) > 0) strcpy(MSRSC_output, argv[i + 1]);
	
	if ((i = ArgPos((char *)"-load-embeded", argc, argv)) > 0) strcpy(embeded_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-load-predict", argc, argv)) > 0) strcpy(predict_file, argv[i + 1]);
	
	
    printf("SWE Tool For: Microsoft Sentence Completion OR ASR N-Best Ranking Score Calculation\n");
	printf("MODEL-Evaluation: Load Word Embedding Models and Testing\n");
	printf("Train Setting embedding size: %d\n", layer1_size);
	printf("Train Setting window size: %d\n", window);
	printf("Train Setting sample value: %f\n", sample);	
	printf("Train Setting negative num: %d\n", negative);
	printf("Running Threads: %d\n", num_threads);
	printf("Iteration times: %d\n", iter);
		
	printf("SemWE Add Time(/%%): %f\n", semwe_add_time);
	printf("SemWE Weight Decay: %f\n", semwe_weight_decay);
	printf("SemWE Inter Coeff: %f\n", semwe_inter_coeff);	
	printf("SemWE Norm Hinge Margin: %f\n", semwe_hinge_margin);
	printf("SemWE Inequation Delta Left: %d\n", delta_left);
	printf("SemWE Inequation Delta Right: %d\n", delta_right);
	
	printf("MSR Sentence Completion type file input: %s\n", MSRSC_input);
	printf("MSR Sentence Completion type file output: %s\n", MSRSC_output);
	
	fTest = fopen(embeded_file, "r");
	if (fTest == NULL) {
		printf("Error, can not find file %s\n", embeded_file);
		exit(1);
	}
	fclose(fTest);
	fTest = fopen(predict_file, "r");
	if (fTest == NULL) {
		printf("Error, can not find file %s\n", predict_file);
		exit(1);
	}
	fclose(fTest);
	
	zero_vector  = (real*)malloc(layer1_size*sizeof(real));
	for (i = 0; i < layer1_size; i++) zero_vector[i] = 0.0;
	
	//
	vocab = (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));
	vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));
	expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
	for (i = 0; i < EXP_TABLE_SIZE; i++) {
	expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
	expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
	}
	TrainModel();

	free(syn0);
	free(KnowDB_TermKDB);
	free(KnowDB_InEquation);
	free(zero_vector);
	return 0;
}

//-------------   SWE_Test_SentComplete.c  --------------//
