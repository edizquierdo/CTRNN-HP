main: main.o CTRNN.o TSearch.o random.o
	g++ -o main main.o CTRNN.o TSearch.o random.o
random.o: random.cpp random.h VectorMatrix.h
	g++ -c -O3 random.cpp
CTRNN.o: CTRNN.cpp random.h CTRNN.h
	g++ -c -O3 CTRNN.cpp
TSearch.o: TSearch.cpp TSearch.h
	g++ -c -O3 TSearch.cpp
main.o: main.cpp CTRNN.h TSearch.h
	g++ -c -O3 main.cpp
clean:
	rm *.o main
