#ifndef FILTERS_H
#define FILTERS_H

#include <opencv2/opencv.hpp>

// Declaração das funções de filtro disponíveis
void applyGrayScale(const cv::Mat& entrada, cv::Mat& saida);
void applySharpen(const cv::Mat& entrada, cv::Mat& saida);
void applyGaussian(const cv::Mat& entrada, cv::Mat& saida);
void applySobel(const cv::Mat& entrada, cv::Mat& saida);

#endif