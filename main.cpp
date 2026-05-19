/*
// Para compilar:

g++ main.cpp -o main -fopenmp `pkg-config --cflags --libs opencv4`

// Para executar:
./main
*/

#include <stdio.h>
#include <opencv2/opencv.hpp>

void applayGrayScale(const cv::Mat& entrada, cv::Mat& saida) {
    // Garante que a imagem de saída tenha o mesmo tamanho e tipo da entrada
    saida = cv::Mat::zeros(entrada.size(), entrada.type());

    int linhas = entrada.rows;
    int colunas = entrada.cols;


    // Vamos usar schedule dynamic assim sempre que tiver disponível vai liberar a thread para processar o próximo pixel
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < linhas; i++) {
        for (int j = 0; j < colunas; j++) {
            
            // Separamos cada cor de cada pixel
            cv::Vec3b pixelOriginal = entrada.at<cv::Vec3b>(i, j);
            uchar azul  = pixelOriginal[0];
            uchar verde = pixelOriginal[1];
            uchar vermelho   = pixelOriginal[2];

            
            // Essa é uma forma de converter para gray scale levando em consideração a limunancia
            // Algumas cores são mais influentes que outras
            uchar tomDeCinza = (uchar)(0.114 * azul + 0.587 * verde + 0.299 * vermelho);

            // 3. Como o arquivo de saída exige uma imagem BGR (3 canais),
            // nós jogamos o mesmo valor cinza nos 3 canais do pixel de saída.
            cv::Vec3b pixelCinza;
            pixelCinza[0] = tomDeCinza; // Canal Azul vira Cinza
            pixelCinza[1] = tomDeCinza; // Canal Verde vira Cinza
            pixelCinza[2] = tomDeCinza; // Canal Vermelho vira Cinza

            // 4. Salvamos o pixel modificado na matriz de saída
            saida.at<cv::Vec3b>(i, j) = pixelCinza;
        }
    }
}

int main(int argc, char** argv) {
    cv::VideoCapture capture;
    cv::VideoWriter writer;
    cv::Mat frame, frameProcessado; // Mudamos o nome de 'gray' para 'frameProcessado' para ficar mais claro

    int width, height;
    double fps;
    int fourcc;

    // So vamos usar se precisar passar algum argumento
    // if (argc < 3) {
    //     printf("Uso: %s <video_entrada> <video_saida>\n", argv[0]);
    //     return 1;
    // }

    // Capture.open(0) serve para abrir a camera do dispositivo
    // Se quiser ler algum arquivo passado no terminal precisa passar o argv[1]
    capture.open(0);
    if (!capture.isOpened()) {
        printf("Erro ao abrir a câmera ou arquivo de entrada.\n");
        return 1;
    }

    width  = (int) capture.get(cv::CAP_PROP_FRAME_WIDTH);
    height = (int) capture.get(cv::CAP_PROP_FRAME_HEIGHT);
    fps    = capture.get(cv::CAP_PROP_FPS);

    if (fps <= 0) fps = 30.0;

    fourcc = cv::VideoWriter::fourcc('X', 'V', 'I', 'D');


    // Se quiser criar um arquivo de saida para salvar o resultado
    // writer.open(argv[2], fourcc, fps, cv::Size(width, height), true);
    // if (!writer.isOpened()) {
    //     printf("Erro ao criar vídeo de saída: %s\n", argv[2]);
    //     capture.release();
    //     return 1;
    // }

    cv::namedWindow("Processing image capture", cv::WINDOW_AUTOSIZE);

    printf("Processando vídeo...\n");
    printf("Resolução: %dx%d | FPS: %.2f\n", width, height, fps);

    while (capture.read(frame)) {
        if (frame.empty()) break;

        // Chama a função para aplicar gray scale
        applayGrayScale(frame, frameProcessado);

        // Gravamos e exibimos o frame processado manualmente
        //writer.write(frameProcessado);
        cv::imshow("Processing image capture", frameProcessado);

        int delay = 1000 / fps;
        if (cv::waitKey(delay) == 27) {  // 27 tecla ESC
            break;
        }
    }

    // Liberação de memória
    writer.release();
    capture.release();
    cv::destroyAllWindows();

    // Não tamo mais salvando o vídeo mas podemos exibir se quiserem
    //printf("Vídeo salvo com sucesso em: %s\n", argv[2]);
    return 0;
}