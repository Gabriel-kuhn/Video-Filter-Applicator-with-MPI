/*
// Para compilar:
make

// Para executar com 3 processos (1 Master - 2 Workers):
mpirun -np 3 ./main
*/

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "filters.h"
#include "process.h"
#include <mpi.h>
#include <map>

int main(int argc, char** argv) {
    int rank, size;

    // Inicialização do MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // inicializando o rank do processo atual
    MPI_Comm_size(MPI_COMM_WORLD, &size); // inicializando a quantidade de processos

    // Variáveis que todos precisam saber para alocar memória
    int width = 0, height = 0;
    double fps = 30.0;

    if (rank == 0) {
        cv::VideoCapture capture(0); // Abre a webcam
        if (!capture.isOpened()) {
            printf("[Coordenador] Erro ao abrir a câmera.\nParece que ela está indisponível"); // Algum outro processo pode estar usando a câmera quando tentarmos usá-la
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
    
        // Caso queira uma melhora de desemepnho, podemos forçar uma resolução de captura menor (640x480)
        // OBS: não senti melhora de desempenho
        // capture.set(cv::CAP_PROP_FRAME_WIDTH, 640);
        // capture.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

        // Medidas originais da câmera do dispositivo
        width  = (int) capture.get(cv::CAP_PROP_FRAME_WIDTH);
        height = (int) capture.get(cv::CAP_PROP_FRAME_HEIGHT);

        fps = capture.get(cv::CAP_PROP_FPS);
        if (fps <= 0) fps = 30.0;
        
        capture.release(); // Fecha só para garantir, vamos reabrir no loop real
    }

    // O Bcast vai conferir quem é o root (3º parâmetro)
    // Se o processo atual for o root, ele envia os dados
    // Se ele não por o root (ex: rank > 0) ele vai receber os dados
    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&fps, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Calcula o tamanho total do buffer em bytes (Largura x Altura x 3 canais BGR)
    int total_bytes = width * height * 3;

    printf("[Rank %d] Pronto. Resolução do sistema: %dx%d | FPS: %.2f\n", rank, width, height, fps);


    // Aqui temos um divisor, dependendo do rank do processo, ele será um Master ou um worker
    if (rank == 0) {
        start_master(size, width, height, fps, total_bytes);

    } else {
        start_worker(width, height, total_bytes);
    }

    MPI_Finalize();
    return 0;
}