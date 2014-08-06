CC = g++
CFLAGS = -w -Wall -O3 -Wno-deprecated
OBJ  = ./main.o ./mrf.o ./bqp.o ./sap.o
LINKOBJ  = ./main.o ./mrf.o ./bqp.o ./sap.o
BIN  = ./mrf


LIBRARIES = -lm -lnsl -dl  -lstdc++

.PHONY: all all-before all-after clean clean-custom

all: ./mrf


clean: clean-custom
		rm -f $(OBJ) $(BIN)

$(BIN): $(LINKOBJ)
		$(CC) $(CFLAGS) $(LINKOBJ) -o $(BIN) $(LIBRARIES)

./main.o: main.cpp mrf.h
		$(CC) $(CFLAGS) -c ./main.cpp -o ./main.o

./sap.o: sap.cpp sap.h
		$(CC) $(CFLAGS) -c ./sap.cpp -o sap.o

./mrf.o: mrf.cpp mrf.h bqp.h
		$(CC) $(CFLAGS) -c mrf.cpp -o mrf.o

./bqp.o: bqp.cpp bqp.h sap.h
		$(CC) $(CFLAGS) -c bqp.cpp -o bqp.o
