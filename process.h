#ifndef PROCESS_H
#define PROCESS_H

#include <opencv2/opencv.hpp>

// Declaração das funções de worker e master disponíveis
void start_master(int size, int width, int height, double fps, int total_bytes);
void start_worker(int width, int height, int total_bytes);

#endif