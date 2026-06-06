#include "filters.h"
#include <omp.h>


void applyGrayScale(const cv::Mat& entrada, cv::Mat& saida) {
    omp_set_num_threads(2); // Assim não forçamos tanto a CPU

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

void applySharpen(const cv::Mat& entrada, cv::Mat& saida) {
    omp_set_num_threads(2); // Assim não forçamos tanto a CPU

    saida = cv::Mat::zeros(entrada.size(), entrada.type());
    int linhas = entrada.rows;
    int colunas = entrada.cols;

    int kernel[3][3] = {
        { 0, -1,  0 },
        {-1,  5, -1 },
        { 0, -1,  0 }
    };

    #pragma omp parallel for schedule(dynamic)
    for (int i = 1; i < linhas - 1; i++) {
        for (int j = 1; j < colunas - 1; j++) {
            int azulAcumulado = 0, verdeAcumulado = 0, vermelhoAcumulado = 0;

            for (int k_i = -1; k_i <= 1; k_i++) {
                for (int k_j = -1; k_j <= 1; k_j++) {
                    cv::Vec3b vizinho = entrada.at<cv::Vec3b>(i + k_i, j + k_j);
                    int peso = kernel[k_i + 1][k_j + 1];

                    azulAcumulado     += vizinho[0] * peso;
                    verdeAcumulado    += vizinho[1] * peso;
                    vermelhoAcumulado += vizinho[2] * peso;
                }
            }

            cv::Vec3b novoPixel;
            novoPixel[0] = cv::saturate_cast<uchar>(azulAcumulado);
            novoPixel[1] = cv::saturate_cast<uchar>(verdeAcumulado);
            novoPixel[2] = cv::saturate_cast<uchar>(vermelhoAcumulado);

            saida.at<cv::Vec3b>(i, j) = novoPixel;
        }
    }
}