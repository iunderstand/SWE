
CC = g++
CFLAGS = -lm -pthread -O3 -march=native -Wall -funroll-loops -Wno-unused-result -I/opt/intel/mkl/include -L/opt/intel/mkl/lib/intel64 -L/opt/intel/lib/intel64 -lmkl_intel_lp64 -lmkl_intel_thread -liomp5 -lmkl_core

all: SWE_Train SWE_Test_SentComplete SWE_Test_WordSim SWE_Test_SynSel

SWE_Train : SWE_Train.c
	$(CC) SWE_Train.c -o SWE_Train $(CFLAGS)
SWE_Test_SentComplete : SWE_Test_SentComplete.c
	$(CC) SWE_Test_SentComplete.c -o SWE_Test_SentComplete $(CFLAGS)
SWE_Test_WordSim : SWE_Test_WordSim.cpp
	$(CC) SWE_Test_WordSim.cpp -o SWE_Test_WordSim $(CFLAGS)
SWE_Test_SynSel : SWE_Test_SynSel.cpp
	$(CC) SWE_Test_SynSel.cpp -o SWE_Test_SynSel $(CFLAGS)

clean:
	rm -rf SWE_Train SWE_Test_SentComplete SWE_Test_WordSim SWE_Test_SynSel
