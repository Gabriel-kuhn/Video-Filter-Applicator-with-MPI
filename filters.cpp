#include "filters.h"
#include <omp.h>
#include <cmath>


void applyGrayScale(const cv::Mat& entrada, cv::Mat& saida) {
    
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

void applyGaussian(const cv::Mat& entrada, cv::Mat& saida) {
    
    saida = cv::Mat::zeros(entrada.size(), entrada.type());
    int linhas = entrada.rows;
    int colunas = entrada.cols;

    // Kernel Gaussiano 5x5 (aproximação inteira da distribuição normal)
    // A soma dos pesos é 256, por isso dividimos o resultado acumulado por 256
    int kernel[5][5] = {
        { 1,  4,  6,  4, 1 },
        { 4, 16, 24, 16, 4 },
        { 6, 24, 36, 24, 6 },
        { 4, 16, 24, 16, 4 },
        { 1,  4,  6,  4, 1 }
    };
    const int pesoTotal = 256;

    // Como o kernel é 5x5, precisamos de uma borda de 2 pixels para não acessar
    // posições fora da matriz (por isso o laço começa em 2 e termina em -2)
    #pragma omp parallel for schedule(dynamic)
    for (int i = 2; i < linhas - 2; i++) {
        for (int j = 2; j < colunas - 2; j++) {
            int azulAcumulado = 0, verdeAcumulado = 0, vermelhoAcumulado = 0;

            for (int k_i = -2; k_i <= 2; k_i++) {
                for (int k_j = -2; k_j <= 2; k_j++) {
                    cv::Vec3b vizinho = entrada.at<cv::Vec3b>(i + k_i, j + k_j);
                    int peso = kernel[k_i + 2][k_j + 2];

                    azulAcumulado     += vizinho[0] * peso;
                    verdeAcumulado    += vizinho[1] * peso;
                    vermelhoAcumulado += vizinho[2] * peso;
                }
            }

            cv::Vec3b novoPixel;
            novoPixel[0] = cv::saturate_cast<uchar>(azulAcumulado / pesoTotal);
            novoPixel[1] = cv::saturate_cast<uchar>(verdeAcumulado / pesoTotal);
            novoPixel[2] = cv::saturate_cast<uchar>(vermelhoAcumulado / pesoTotal);

            saida.at<cv::Vec3b>(i, j) = novoPixel;
        }
    }
}

void applySobel(const cv::Mat& entrada, cv::Mat& saida) {
    
    saida = cv::Mat::zeros(entrada.size(), entrada.type());
    int linhas = entrada.rows;
    int colunas = entrada.cols;

    // Máscaras de Sobel para detecção de bordas horizontais (Gx) e verticais (Gy)
    int kernelX[3][3] = {
        { -1, 0, 1 },
        { -2, 0, 2 },
        { -1, 0, 1 }
    };

    int kernelY[3][3] = {
        { -1, -2, -1 },
        {  0,  0,  0 },
        {  1,  2,  1 }
    };

    #pragma omp parallel for schedule(dynamic)
    for (int i = 1; i < linhas - 1; i++) {
        for (int j = 1; j < colunas - 1; j++) {
            int gx = 0, gy = 0;

            for (int k_i = -1; k_i <= 1; k_i++) {
                for (int k_j = -1; k_j <= 1; k_j++) {
                    cv::Vec3b vizinho = entrada.at<cv::Vec3b>(i + k_i, j + k_j);

                    // Convertemos o vizinho para tom de cinza antes de aplicar o Sobel,
                    // pois a detecção de bordas é feita sobre a intensidade luminosa
                    int cinza = (int)(0.114 * vizinho[0] + 0.587 * vizinho[1] + 0.299 * vizinho[2]);

                    gx += cinza * kernelX[k_i + 1][k_j + 1];
                    gy += cinza * kernelY[k_i + 1][k_j + 1];
                }
            }

            // Magnitude do gradiente: indica a intensidade da borda naquele ponto
            int magnitude = (int) std::sqrt((double)(gx * gx + gy * gy));
            uchar bordaIntensidade = cv::saturate_cast<uchar>(magnitude);

            cv::Vec3b pixelBorda;
            pixelBorda[0] = bordaIntensidade;
            pixelBorda[1] = bordaIntensidade;
            pixelBorda[2] = bordaIntensidade;

            saida.at<cv::Vec3b>(i, j) = pixelBorda;
        }
    }
}