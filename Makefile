# Variáveis de configuração
CC = mpic++
CFLAGS = -fopenmp
OPENCV = `pkg-config --cflags --libs opencv4`
TARGET = main

# Comando principal de compilação (Atualizado com o process.cpp!)
all:
	$(CC) main.cpp filters.cpp process.cpp -o $(TARGET) $(CFLAGS) $(OPENCV)

clean:
	rm -f $(TARGET)