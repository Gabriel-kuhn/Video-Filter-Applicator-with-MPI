#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "filters.h"
#include "process.h"
#include <mpi.h>
#include <map>

// TODO tentar negativo
#define END_TAG 999999 // Não pode ser menor, se não irá conflitar com a lógica de troca de filtro
#define CICLO_ID 10000



// .:| ====== Funções Auxiliares ====================================================

// Precisamos reciclar o frame, isso é fazer com que ele resete para 1 quando chegar 
// no valor limite definido em CICLO_ID
// Após isso, precisamos formatar a tag que será enviada, para que essa contenha o frama_id e o filtro

// Formatação da TAG:
// FRAME_RECICLADO + FILTRO
int get_formated_tag(int current_filter, int frame_id) {
    int recycled_frame_id = frame_id % CICLO_ID;
    return (current_filter * CICLO_ID) + recycled_frame_id;
}


int get_recycled_frame(int next_frame_to_show, int received_tag) {
    return next_frame_to_show + ((received_tag - (next_frame_to_show % CICLO_ID) + CICLO_ID) % CICLO_ID);
}



// .:| ====== Funções Principais ====================================================

void start_master(int size, int width, int height, double fps, int total_bytes) {
    cv::VideoCapture capture(0);
    cv::Mat frame;
    cv::namedWindow("Processing image capture", cv::WINDOW_AUTOSIZE);

    int frame_id = 0;
    int next_frame_to_show = 0;
    int current_filter = 1;
    bool has_user_exit = false; // variável de controle para saber quando o usuário finalizou o programa

    // Matriz de resultado -> buffer
    // Ela serve para guardar os frames procesados
    // Caso algum trabalhador termine seu trabalho mas não é hora de exibir ele ainda, guardamos ele para quando for a hora de exibí-lo
    std::map<int, cv::Mat> buffer;

    // Loop de Envio Inicial
    // Precisamos primeiro popular TODOS os trabalhadores com frames para eles processarem
    // O while principal vai ser reativo então é necessário que essa parte seja separada do loop principal
    for (int i = 1; i < size; i++) {
        if (capture.read(frame)) {
            if (frame.empty()) break;

            int formated_tag = get_formated_tag(current_filter, frame_id);
            
            // A TAG é o próprio número do frame (frame_id)
            // A TAG (id do frame atual) parace inútil, pois o worker vai receber de qualquer TAG (MPI_ANY_TAG), 
            // PORÉM, ela será MUITO importante para a ordenação dos frames no while principal
            MPI_Send(frame.data, total_bytes, MPI_UNSIGNED_CHAR, i, formated_tag, MPI_COMM_WORLD);
            frame_id++;
        }
    }


    // .:| === WHILE PRINCIPAL ============= |
    // ele será reativo, a cada vez que algum trabalhador enviar um frame processado (pronto)
    // o master receberá, adiciona-rá na lista de forma ordenada
    // e liberará mais um frame para aquele worker seguir trabalhando

    // A codição do while funciona da seguinte forma:
    // Enquanto o id do frame que eu preciso exibir (next_frame_to_show) for menor 
    // Que o total de frames que eu já capturei da câmera e enviei para a rede (frame_id), o loop continua.
    // Em outra palavras, enquanto ainda tiver trabalho a ser feito, continue distribuido o trabalho
    while (next_frame_to_show < frame_id) {

        // Vamos iniciar uma matriz de 3 canais com todos os valores zerados
        // Basicamente é uma imagem toda preta
        // Serve para guaradar na memória a matriz que portará o resultado do frame processado
        cv::Mat received_frame = cv::Mat::zeros(height, width, CV_8UC3);
        MPI_Status status;

        // Aqui fica o recebimento dos frames processados, esperamos o resultado de qualquer worker 
        MPI_Recv(received_frame.data, total_bytes, MPI_UNSIGNED_CHAR, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

        // Vamos precisar usar o MPI_status, pois precisamos de algumas informações além do frame resultante
        // precisamos saber:
        // Worker que respondeu -> para enviar mais trabalho especificamente para ele
        // Frame que foi processado -> para adicionar de forma ordenada na lista
        int worker_source = status.MPI_SOURCE;
        int received_tag = status.MPI_TAG;

        // TODO trasnformar em uma função para facilitar
        int recycled_frame_id = get_recycled_frame(next_frame_to_show, received_tag);

        // O id do frame processado serve como endereço para a lista ordenada
        buffer[recycled_frame_id] = received_frame;

        // CRITÉRIO DE RECOMPOSIÇÃO: Verifica se o frame que precisamos exibir agora já está na sala de espera
        while (buffer.find(next_frame_to_show) != buffer.end()) {
            // Exibe o frame correto na ordem cronológica literal
            cv::imshow("Processing image capture", buffer[next_frame_to_show]);
            
            // Remove da memória para não estourar a RAM
            buffer.erase(next_frame_to_show);
            next_frame_to_show++;

            int delay = 1000 / fps;
            int key = cv::waitKey(delay);

            if (key == 27) {  // 27 = esc
                has_user_exit = true;          
                break;    

            } else if (key == '1') {
                current_filter = 1;
                printf("[Coordenador] Filtro alterado para: 1 (Imagem Original)\n");

            } else if (key == '2') {
                current_filter = 2;
                printf("[Coordenador] Filtro alterado para: 2 (GrayScale)\n");

            } else if (key == '3') {
                current_filter = 3;
                printf("[Coordenador] Filtro alterado para: 3 (Sharpen)\n");
            }
        }

        if (has_user_exit) {
            printf("[Coordenador] Encerrando o sistema...\n");
            
            // Libera os recursos locais do Master antes de fechar tudo
            capture.release();
            cv::destroyAllWindows();
            
            // Aborta todos os processos do COMM_WORLD de forma limpa e síncrona
            MPI_Abort(MPI_COMM_WORLD, 0);
            break;
        }

        // Se a câmera ainda tiver frames a serem processados, 
        // O master continua o fluxo enviando para o próximo trabalhador que acabou de ficar livre
        if (capture.read(frame) && !frame.empty()) {
            int formated_tag = get_formated_tag(current_filter, frame_id);
 
            MPI_Send(frame.data, total_bytes, MPI_UNSIGNED_CHAR, worker_source, formated_tag, MPI_COMM_WORLD);
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
        int tag = status.MPI_TAG;
        if (tag == END_TAG) {
            break;
        }

        int current_filter = tag / CICLO_ID;

        // Aplica o filtro
        if (current_filter == 1) {
            receive_buffer.copyTo(send_buffer); // Imagem original

        } else if (current_filter == 2) {
            applyGrayScale(receive_buffer, send_buffer);

        } else if (current_filter == 3) {
            applySharpen(receive_buffer, send_buffer);

        } else {
            receive_buffer.copyTo(send_buffer); // caso venha algo inválido mostra a imagem original
        }

        // Envia para o Master o frame já processado e mantendo a TAG recebida
        MPI_Send(send_buffer.data, total_bytes, MPI_UNSIGNED_CHAR, 0, tag, MPI_COMM_WORLD);
    }
}