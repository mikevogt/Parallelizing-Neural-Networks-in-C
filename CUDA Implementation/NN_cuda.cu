#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#define ROW 773
#define COL 26
#define TRAIN_ROW 541
#define TEST_ROW 232
// define nodes
#define INPUT_NODES 26
#define HIDDEN_NODES 10
#define OUTPUT_NODES 1

#define ALPHA 0.1

// Activation Functions
__device__ float sigmoid_device(float x){
    return 1/(1 + exp(-x));
}
float sigmoid(float x){
    return 1/(1 + exp(-x));
}
__device__ float diff_Sigmoid(float x){
  return x * (1 - x);
}

__global__ void cuda_forward_1(float* input, float* weight1, float* layer1, float* bias_layer1){

  int ind_x = blockIdx.x * blockDim.x + threadIdx.x;
  if(ind_x < HIDDEN_NODES){
      float act = 0.0;
      for(int j = 0; j < INPUT_NODES; j++){
          act += input[j]*weight1[ind_x*INPUT_NODES + j] ;
      }
      layer1[ind_x] = sigmoid_device(act+ bias_layer1[ind_x]);
  }
}

__global__ void cuda_forward_2(float* weight2, float* layer1, float* layer2, float* bias_layer2){

  int ind_x = blockIdx.x * blockDim.x + threadIdx.x;
  if(ind_x < OUTPUT_NODES){
      float act = 0.0;
      for(int j = 0; j < HIDDEN_NODES; j++){
          act += layer1[j]*weight2[ind_x* HIDDEN_NODES + j] ;
      }
      layer2[ind_x] = sigmoid_device(act+ bias_layer2[ind_x]);
  }
}

__global__ void cuda_backprop_out(float* d3, float *layer2, float *label){

  int ind_x = blockIdx.x * blockDim.x + threadIdx.x;
  if(ind_x < OUTPUT_NODES){
    float err = layer2[ind_x] - label[ind_x];
    d3[ind_x] = err;
  }
  return;
}

__global__ void cuda_backprop_hidden(float* d2, float* layer1, float* weight2, float* d3){
  int ind_x = blockIdx.x * blockDim.x + threadIdx.x;
  if(ind_x < HIDDEN_NODES){
    float error_hidden = 0.0;
    for(int j = 0; j < OUTPUT_NODES; j++){
      error_hidden += d3[j]*weight2[j*HIDDEN_NODES + ind_x];
    }
      d2[ind_x] = error_hidden * (layer1[ind_x] * (1 - layer1[ind_x]));
  }
}

__global__ void update_weight2(float* weight2, float* layer1, float* d3){
  int ind_x = blockIdx.x * blockDim.x + threadIdx.x;
  if(ind_x < OUTPUT_NODES){
    for(int j = 0; j < HIDDEN_NODES; j++){
      weight2[ind_x*HIDDEN_NODES + j] -= layer1[j]*d3[ind_x]*ALPHA;
    }
  }
}

__global__ void update_weight1(float* weight1, float* input, float* d2){
  int ind_x = blockIdx.x * blockDim.x + threadIdx.x;
  if(ind_x < HIDDEN_NODES){
    for(int j = 0; j < INPUT_NODES; j++){
      weight1[ind_x*INPUT_NODES + j] -= input[j]*d2[ind_x]*ALPHA;
    }
  }
}

void predict(float *input_matrix,
            float *pred_arr,
            float *weight1,
            float *weight2,
            float layer1[HIDDEN_NODES],
            float layer2[OUTPUT_NODES])
            {

    //this will be each extracted input row
    float input[COL];
    //float output=0;

    // iterate through input matrix row by row, extracting each row for training
    for(int row = 0; row < TEST_ROW; row++){
        for(int col = 0; col < COL; col++){
            input[col] = input_matrix[row*COL + col];
        }

        // FORWARD PROPAGATION:
        for(int i = 0; i < HIDDEN_NODES; i++){
            float act = 0.0;
            for(int j = 0; j < INPUT_NODES; j++){
                act += input[j]*weight1[i * INPUT_NODES + j];
            }
            layer1[i] = sigmoid(act);
        }
        for(int i = 0; i < OUTPUT_NODES; i++){
            float act = 0.0;
            for(int j = 0; j < HIDDEN_NODES; j++){
                act += layer1[j]*weight2[i * HIDDEN_NODES + j];
            }
            layer2[i] = sigmoid(act);
        }

        //store predictions in an array
        for(int i = 0; i < OUTPUT_NODES; i++){
            if(layer2[i]>0.5){
                pred_arr[row] = 1;
            }
            else{
                pred_arr[row] = 0;
            }

        }
    }

    return;
}

float train_nn(float *input_matrix,
              float label[TRAIN_ROW],
              float *weight1,
              float *weight2,
              float layer1[HIDDEN_NODES],
              float layer2[OUTPUT_NODES],
              int p_epoch)
              {
    //this will be each extracted input row
    float input[COL];

    for(int epoch=0; epoch < p_epoch; epoch++){
      // iterate through input matrix row by row, extracting each row for training
      for(int row = 0; row < TRAIN_ROW; row++){
          for(int col = 0; col < COL; col++){
              input[col] = input_matrix[row*COL + col];
          }

          //this is for one row instance of forward and backprop
          // FORWARD PROPAGATION:
          for(int i = 0; i < HIDDEN_NODES; i++){
              float act = 0.0;
              for(int j = 0; j < INPUT_NODES; j++){
                  act += input[j]*weight1[i*INPUT_NODES + j];
              }
              layer1[i] = sigmoid(act);
          }
          for(int i = 0; i < OUTPUT_NODES; i++){
              float act = 0.0;
              for(int j = 0; j < HIDDEN_NODES; j++){
                  act += layer1[j]*weight2[i* HIDDEN_NODES + j];
              }
              layer2[i] = sigmoid(act);
          }
          // BACKPROPAGATION:

          // calculate errors
          float d3[OUTPUT_NODES];
          for(int i = 0; i < OUTPUT_NODES; i++){
              float error_output = layer2[i] - label[row];
              d3[i] = error_output;
          }
          float d2[HIDDEN_NODES];
          for(int i = 0; i < HIDDEN_NODES; i++){
              float error_hidden = 0.0;
              for(int j = 0; j < OUTPUT_NODES; j++){
                  error_hidden += d3[j]*weight2[j*HIDDEN_NODES + i];
              }
              d2[i] = error_hidden * (layer1[i] * (1 - layer1[i]));
          }

          // update weights
          for(int i = 0; i < OUTPUT_NODES; i++){
              for(int j = 0; j < HIDDEN_NODES; j++){
                  weight2[i*HIDDEN_NODES + j] -= layer1[j]*d3[i]*ALPHA;
              }
          }
          for(int i = 0; i < HIDDEN_NODES; i++){
              for(int j = 0; j < INPUT_NODES; j++){
                  weight1[i*INPUT_NODES + j] -= input[j]*d2[i]*ALPHA;
              }
          }
      }
    }

    return 0;
}

void import_data(float *train_arr, float *train_y_arr, float *test_arr , float *test_y_arr){
  FILE* str = fopen("Data/train_data.csv", "r");

  char line[1024];
  int count = 0;
  while (fgets(line, 1024, str))
  {
    char* tmp = strdup(line);
    char* c = strtok(tmp,",");
    //train_arr[count] = new float[1];
    while(c != NULL){
      train_arr[count] = (float)atof(c);
      count ++;
      c = strtok(NULL, ",");
    }
    free(tmp);
  }
  //IMPORT TRAINING LABELS

  FILE* str_y = fopen("Data/train_y.csv", "r");

  char line_y[1024];
  int count_y = 0;
  while (fgets(line_y, 1024, str_y))
  {
    char* tmp = strdup(line_y);
    char* c = strtok(tmp,",");

    while(c != NULL){
      train_y_arr[count_y] = (float)atof(c);
      count_y ++;
      c = strtok(NULL, ",");
    }
    free(tmp);
  }

  //IMPORT TESTING DATA

  FILE* str_t = fopen("Data/test_data.csv", "r");

  char line_t[1024];
  int count_t = 0;
  while (fgets(line_t, 1024, str_t))
  {
    char* tmp = strdup(line_t);
    char* c = strtok(tmp,",");

    while(c != NULL){
      test_arr[count_t] = (float)atof(c);
      count_t ++;
      c = strtok(NULL, ",");
    }
    free(tmp);
  }

  //IMPORT TEST LABELS

  FILE* str_ty = fopen("Data/test_y.csv", "r");

  char line_ty[1024];
  int count_ty = 0;
  while (fgets(line_ty, 1024, str_ty))
  {
    char* tmp = strdup(line_ty);
    char* c = strtok(tmp,",");

    while(c != NULL){
      test_y_arr[count_ty] = (float)atof(c);
      count_ty ++;
      c = strtok(NULL, ",");
    }
    free(tmp);
  }

}

int main(int argc, char *argv[]){

  float train_arr[TRAIN_ROW*COL];
  float train_y_arr[TRAIN_ROW*1];
  float test_arr[TEST_ROW*COL];
  float test_y_arr[TEST_ROW*1];
  float weight_layer1[HIDDEN_NODES*INPUT_NODES];
  float weight_layer2[OUTPUT_NODES*HIDDEN_NODES];
  float bias_layer1[HIDDEN_NODES];
  float bias_layer2[OUTPUT_NODES];
  float layer1[HIDDEN_NODES];
  float layer2[OUTPUT_NODES];
  float d3[OUTPUT_NODES];
  float d2[HIDDEN_NODES];
  float** train_arr_device = new float*[TRAIN_ROW];
  float** train_arr_y_device = new float*[TRAIN_ROW];
  float* weight1_device;
  float* weight2_device;
  float* layer1_device;
  float* layer2_device;
  float* d3_device;
  float* d2_device;
  float* bias_layer1_device;
  float* bias_layer2_device;
  cudaDeviceReset();
  float** train_final = new float* [TRAIN_ROW];
  float** train_y_final = new float* [TRAIN_ROW];
  float *output = (float *)malloc(sizeof(TRAIN_ROW*sizeof(float)));
  float *output_test = (float *)malloc(sizeof(TEST_ROW*sizeof(float)));

  //IMPORT TRAINING DATA
  import_data(train_arr, train_y_arr, test_arr, test_y_arr);

  for (size_t i = 0; i < TRAIN_ROW; i++) {
    train_final[i] = new float[COL];
    train_y_final[i] = new float[COL];
    for (size_t j = 0; j < COL; j++) {
      train_final[i][j] = train_arr[i*COL + j];
    }
    for (size_t k = 0; k < 1; k++) {
      train_y_final[i][k] = train_y_arr[i];
    }
  }

  // generate random weights and biases
  for(int i = 0; i < HIDDEN_NODES; i++){
      for(int j = 0; j < INPUT_NODES; j++){
          weight_layer1[i*INPUT_NODES + j] = ((double)rand())/((double)RAND_MAX);
      }
  }
  for(int i = 0; i < OUTPUT_NODES; i++){
      for(int j = 0; j < HIDDEN_NODES; j++){
          weight_layer2[i*HIDDEN_NODES + j] = ((double)rand())/((double)RAND_MAX);
      }
  }
  for(int i = 0; i < HIDDEN_NODES; i++){
    bias_layer1[i] = ((double)rand())/((double)RAND_MAX);
  }
  for(int i = 0; i < OUTPUT_NODES; i++){
    bias_layer2[i] = ((double)rand())/((double)RAND_MAX);
  }

  for (size_t i = 0; i < TRAIN_ROW; i++) {
    cudaMalloc(&train_arr_device[i], sizeof(float)*COL);
    cudaMemcpy(train_arr_device[i], train_final[i], sizeof(float)*COL, cudaMemcpyHostToDevice);

    cudaMalloc(&train_arr_y_device[i], sizeof(float)*1);
    cudaMemcpy(train_arr_y_device[i], train_y_final[i], sizeof(float)*1, cudaMemcpyHostToDevice);
  }

  cudaMalloc(&weight1_device, sizeof(float)*HIDDEN_NODES*INPUT_NODES);
  cudaMemcpy(weight1_device, weight_layer1, sizeof(float)*HIDDEN_NODES*INPUT_NODES, cudaMemcpyHostToDevice);

  cudaMalloc(&weight2_device, sizeof(float)*OUTPUT_NODES*HIDDEN_NODES);
  cudaMemcpy(weight2_device, weight_layer2, sizeof(float)*OUTPUT_NODES*HIDDEN_NODES, cudaMemcpyHostToDevice);

  cudaMalloc(&layer1_device, sizeof(float)*HIDDEN_NODES);
  cudaMemcpy(layer1_device, layer1, sizeof(float)*HIDDEN_NODES, cudaMemcpyHostToDevice);

  cudaMalloc(&layer2_device, sizeof(float)*OUTPUT_NODES);
  cudaMemcpy(layer2_device, layer2, sizeof(float)*OUTPUT_NODES, cudaMemcpyHostToDevice);

  cudaMalloc(&d3_device, sizeof(float)*OUTPUT_NODES);
  cudaMemcpy(d3_device, d3, sizeof(float)*OUTPUT_NODES, cudaMemcpyHostToDevice);

  cudaMalloc(&d2_device, sizeof(float)*HIDDEN_NODES);
  cudaMemcpy(d2_device, d2, sizeof(float)*HIDDEN_NODES, cudaMemcpyHostToDevice);

  cudaMalloc(&bias_layer1_device, sizeof(float)*HIDDEN_NODES);
  cudaMemcpy(bias_layer1_device, bias_layer1, sizeof(float)*HIDDEN_NODES, cudaMemcpyHostToDevice);

  cudaMalloc(&bias_layer2_device, sizeof(float)*HIDDEN_NODES);
  cudaMemcpy(bias_layer2_device, bias_layer2, sizeof(float)*HIDDEN_NODES, cudaMemcpyHostToDevice);

  // NEURAL NETWORK
  dim3 dimGrid(39,2,1);
	dim3 dimBlock(14,14,1);
  int epoch = 400;

  printf("                          TRAINING WITH %d EPOCHS:\n__________________________________________________________________________\n__________________________________________________________________________\n\n", epoch);

  cudaEvent_t beginLaunch, endLaunch;
  cudaEventCreate(&beginLaunch);
  cudaEventCreate(&endLaunch);
  cudaEventRecord(beginLaunch,0);
  float mse_total;
  float mse_old = 100000;
  float mse_difference = 100000;
  float mse_abs = 10000;
  int max_epoch = 0;
  while(mse_abs > 0.0001 && max_epoch < epoch){ //
    mse_total = 0.0;

    for (size_t j = 0; j < TRAIN_ROW; j++) { //TRAIN_ROW

      cuda_forward_1<<<dimBlock , dimGrid>>>(train_arr_device[j], weight1_device, layer1_device, bias_layer1_device);

      cuda_forward_2<<<dimBlock , dimGrid>>>(weight2_device, layer1_device, layer2_device, bias_layer2_device);

      cuda_backprop_out<<<dimBlock , dimGrid>>>(d3_device, layer2_device, train_arr_y_device[j]);
      cudaMemcpy(d3, d3_device, sizeof(float)*OUTPUT_NODES, cudaMemcpyDeviceToHost);
      cudaMemcpy(d3_device, d3, sizeof(float)*OUTPUT_NODES, cudaMemcpyHostToDevice);

      mse_total += abs(0.5*d3[0]*d3[0]);

      cuda_backprop_hidden<<<dimBlock , dimGrid>>>(d2_device, layer1_device, weight2_device, d3_device);


      update_weight2<<<dimBlock , dimGrid>>>(weight2_device, layer1_device, d3_device);

      update_weight1<<<dimBlock , dimGrid>>>(weight1_device, train_arr_device[j], d2_device);

    }
    printf("%f\n", mse_total);
    mse_difference = mse_old - mse_total;
    mse_abs = abs(mse_difference);
    mse_old = mse_total;
    max_epoch += 1;
    printf("MSE ABS DIFFERENCE FOR EPOCH: %f\n", mse_abs);

  }
  float mse_final = mse_total;
  cudaEventRecord(endLaunch,0);
  cudaEventSynchronize(endLaunch);
  cudaMemcpy(weight_layer1, weight1_device, sizeof(float)*HIDDEN_NODES*INPUT_NODES, cudaMemcpyDeviceToHost);
  cudaMemcpy(weight_layer2, weight2_device, sizeof(float)*OUTPUT_NODES*HIDDEN_NODES, cudaMemcpyDeviceToHost);

  float time_share = 0;
	cudaEventElapsedTime(&time_share, beginLaunch, endLaunch);
  printf("The time taken to train with %d epochs is: %fms\n", max_epoch, time_share);
  printf("MSE FINAL: %f\n", mse_final);

  predict(test_arr, output, weight_layer1, weight_layer2, layer1, layer2);

  int count_final=0;

  for(int i = 0; i < TEST_ROW; i++){
    if(output[i] == test_y_arr[i]){
        count_final +=1;
    }
  }
  float prediction = (float)count_final/TEST_ROW;
  printf("The final prediction accuracy is: %f \n", prediction);
  free(output);
  cudaFree(train_arr_device);
  cudaFree(train_arr_y_device);
  cudaFree(weight1_device);
  cudaFree(weight2_device);
  cudaFree(bias_layer1_device);
  cudaFree(bias_layer2_device);
  cudaFree(layer1_device);
  cudaFree(layer2_device);
  cudaFree(d3_device);
  cudaFree(d2_device);
  return 0;
}
