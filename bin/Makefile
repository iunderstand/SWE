
CC = g++
CFLAGS = -lm -pthread -O3 -march=native -Wall -funroll-loops -Wno-unused-result 

all: SWE_Train SWE_Test_SentComplete SWE_Test_WordSim SWE_Test_SynSel SWE_Train_v1 SWE_Train_v2 SWE_Train_v3

SWE_Train : SWE_Train.c
	$(CC) SWE_Train.c -o SWE_Train $(CFLAGS)
SWE_Test_SentComplete : SWE_Test_SentComplete.c
	$(CC) SWE_Test_SentComplete.c -o SWE_Test_SentComplete $(CFLAGS)
SWE_Test_WordSim : SWE_Test_WordSim.cpp
	$(CC) SWE_Test_WordSim.cpp -o SWE_Test_WordSim $(CFLAGS)
SWE_Test_SynSel : SWE_Test_SynSel.cpp
	$(CC) SWE_Test_SynSel.cpp -o SWE_Test_SynSel $(CFLAGS)

SWE_Train_v1 : SWE_Train_v1.c
	$(CC) SWE_Train_v1.c -o SWE_Train_v1 $(CFLAGS)
SWE_Train_v2 : SWE_Train_v2.c
	$(CC) SWE_Train_v2.c -o SWE_Train_v2 $(CFLAGS)
SWE_Train_v3 : SWE_Train_v3.c
	$(CC) SWE_Train_v3.c -o SWE_Train_v3 $(CFLAGS)

clean:
	rm -rf SWE_Train SWE_Test_SentComplete SWE_Test_WordSim SWE_Test_SynSel SWE_Train_v1 SWE_Train_v2 SWE_Train_v3
