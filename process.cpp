#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "filters.h"
#include "process.h"
#include <mpi.h>
#include <map>

#define END_TAG 9090

void start_master(int size, int width, int height, double fps, int total_bytes) {
    cv::VideoCapture capture(0);
    cv::Mat frame;
    cv::namedWindow("Processing image capture", cv::WINDOW_AUTOSIZE);

    int frame_id = 0;
    int next_frame_to_show = 0;
    bool has_user_exit = false; // variável de controle para saber quando o usuário finalizou o programa
    
    // Uma "sala de espera" para guardar os frames que voltarem fora de ordem
    std::map<int, cv::Mat> buffer;

    // Loop de Envio Inicial
    // Precisamos primeiro popular TODOS os trabalhadores com frames para eles processarem
    // O while principal vai ser reativo então é necessário que essa parte seja separada do loop principal
    for (int i = 1; i < size; i++) {
        if (capture.read(frame)) {
            if (frame.empty()) break;
            
            // A TAG é o próprio número do frame (frame_id)
            // A TAG (id do frame atual) parace inútil, pois o worker vai receber de qualquer TAG (MPI_ANY_TAG), 
            // PORÉM, ela será MUITO importante para a ordenação dos frames no while principal
            MPI_Send(frame.data, total_bytes, MPI_UNSIGNED_CHAR, i, frame_id, MPI_COMM_WORLD);
            frame_id++;
        }
    }


    // .:| === WHILE PRINCIPAL ============= |
    // ele será reativo, a cada vez que algum trabalhador enviar um frame processado (pronto)
    // o master receberá, adiciona-rá na lista de forma ordenada
    // e liberará mais um frame para aquele worker seguir trabalhando
    while (next_frame_to_show < frame_id) {
        cv::Mat received_frame = cv::Mat::zeros(height, width, CV_8UC3);
        MPI_Status status;

        // Aqui fica o recebimento dos frames processados, esperamos o resultado de qualquer worker 
        MPI_Recv(received_frame.data, total_bytes, MPI_UNSIGNED_CHAR, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

        // Vamos precisar usar o MPI_status, pois precisamos de algumas informações além do frame resultante
        // precisamos saber:
        // Worker que respondeu -> para enviar mais trabalho especificamente para ele
        // Frame que foi processado -> para adicionar de forma ordenada na lista
        int worker_source = status.MPI_SOURCE;
        int received_frame_id = status.MPI_TAG;

        // O id do frame processado serve como endereço para a lista ordenada
        buffer[received_frame_id] = received_frame;

        // CRITÉRIO DE RECOMPOSIÇÃO: Verifica se o frame que precisamos exibir agora já está na sala de espera
        while (buffer.find(next_frame_to_show) != buffer.end()) {
            // Exibe o frame correto na ordem cronológica literal
            cv::imshow("Processing image capture", buffer[next_frame_to_show]);
            
            // Remove da memória para não estourar a RAM
            buffer.erase(next_frame_to_show);
            next_frame_to_show++;

            int delay = 1000 / fps;
            if (cv::waitKey(delay) == 27) {  // 27 = esc
                has_user_exit = true;          
                break;                        
            }
        }

        if (has_user_exit) {
            break;
        }

        // Se a câmera ainda tiver frames a serem processados, 
        // O master continua o fluxo enviando para o próximo trabalhador que acabou de ficar livre
        if (capture.read(frame) && !frame.empty()) {
            MPI_Send(frame.data, total_bytes, MPI_UNSIGNED_CHAR, worker_source, frame_id, MPI_COMM_WORLD);
            frame_id++;
        }
    }


    // Envia sinal de término (Tag de encerramento), os trabalhadores precisam saber se eles devem para seu processamento
    for (int i = 1; i < size; i++) {
        MPI_Send(NULL, 0, MPI_UNSIGNED_CHAR, i, END_TAG, MPI_COMM_WORLD); // 999999 = Tag de encerramento
    }

    // Liberação de recursos
    capture.release();
    cv::destroyAllWindows();
}


void start_worker(int width, int height, int total_bytes) {
    cv::Mat receive_buffer = cv::Mat::zeros(height, width, CV_8UC3);
    cv::Mat send_buffer = cv::Mat::zeros(height, width, CV_8UC3);
    MPI_Status status;

    while (true) {
        // Recebe o frame que será trabalhado
        MPI_Recv(receive_buffer.data, total_bytes, MPI_UNSIGNED_CHAR, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

        // Porém, é avaliado se a TAG enviada não é uma END_TAG (tag de encerramento)
        // Caso seja, o trabalhador sabe que deve parar seu processamento
        if (status.MPI_TAG == END_TAG) {
            break;
        }

        int meu_frame_id = status.MPI_TAG;

        // Aplica o filtro
        // TODO ainda precisamos ver uma forma de mudar o filtro a ser processado
        applySharpen(receive_buffer, send_buffer);

        // Envia para o Master o frame já processado e mantendo a TAG recebida
        MPI_Send(send_buffer.data, total_bytes, MPI_UNSIGNED_CHAR, 0, meu_frame_id, MPI_COMM_WORLD);
    }
}