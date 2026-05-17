/*
//precisa instalar o opencv e compilador c++ pois a versão mais recente só roda em c++

apt install libopencv-dev
apt install g++

//para compilar:
g++ video_copy.c -o video_copy `pkg-config --cflags --libs opencv4`

//para executar:
./video_copy entrada.avi saida.avi

*/

#include <stdio.h>
#include <opencv2/opencv.hpp>

int main(int argc, char** argv) {
    cv::VideoCapture capture;
    cv::VideoWriter writer;
    cv::Mat frame, gray;

    int width, height;
    double fps;
    int fourcc;

    if (argc < 3) {
        printf("Uso: %s <video_entrada> <video_saida>\n", argv[0]);
        return 1;
    }

    capture.open(0);
    if (!capture.isOpened()) {
        printf("Erro ao abrir o vídeo de entrada: %s\n", argv[1]);
        return 1;
    }

    width  = (int) capture.get(cv::CAP_PROP_FRAME_WIDTH);
    height = (int) capture.get(cv::CAP_PROP_FRAME_HEIGHT);
    fps    = capture.get(cv::CAP_PROP_FPS);

    if (fps <= 0) fps = 30.0;

    fourcc = cv::VideoWriter::fourcc('X', 'V', 'I', 'D');

    writer.open(argv[2], fourcc, fps, cv::Size(width, height), true);
    if (!writer.isOpened()) {
        printf("Erro ao criar vídeo de saída: %s\n", argv[2]);
        capture.release();
        return 1;
    }

    printf("Processando vídeo...\n");
    printf("Resolução: %dx%d | FPS: %.2f\n", width, height, fps);

    while (capture.read(frame)) {
		//aqui faz o processamento frame-a-frame
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY); // TEMOS que remover o uso da função do OPENCV
        cv::cvtColor(gray, frame, cv::COLOR_GRAY2BGR);

        writer.write(frame);
        cv::imshow("Processing image capture", frame);

        int delay = 1000 / fps;
        if (cv::waitKey(delay) == 27) {  // 27 tecla ESC, serve para fechar o vídeo
            break;
        }
    }

    writer.release();
    capture.release();

    printf("Vídeo salvo com sucesso em: %s\n", argv[2]);
    return 0;
}