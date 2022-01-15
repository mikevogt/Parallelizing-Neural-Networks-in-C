#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

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
float sigmoid(float x){
    return 1/(1 + exp(-x));
}

void forward_prop(float *input,
                  float *weight1,
                  float *weight2,
                  float *layer1,
                  float *layer2,
                  float *bias1,
                  float *bias2)
                  {

      // FORWARD PROPAGATION:
      for(int i = 0; i < HIDDEN_NODES; i++){
          float act = 0.0;
          for(int j = 0; j < INPUT_NODES; j++){
              act += input[j]*weight1[i*INPUT_NODES + j];
          }
          layer1[i] = sigmoid(act + bias1[i]);
      }
      for(int i = 0; i < OUTPUT_NODES; i++){
          float act = 0.0;
          for(int j = 0; j < HIDDEN_NODES; j++){
              act += layer1[j]*weight2[i* HIDDEN_NODES + j];
          }
          layer2[i] = sigmoid(act + bias2[i]);
      }
}

void backprop(float *input,
              float label,
              float *weight1,
              float *weight2,
              float *layer1,
              float *layer2,
              float *d2,
              float *d3)
              {

      // BACKPROPAGATION:
      // calculate errors
      for(int i = 0; i < OUTPUT_NODES; i++){
          float error_output = layer2[i] - label;
          d3[i] = error_output;
      }
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


void import_data(float* train_arr, float* train_y_arr, float* test_arr , float* test_y_arr){
  FILE* str = fopen("Data/train_data.csv", "r");

  char line[1024];
  int count = 0;
  while (fgets(line, 1024, str))
  {
    char* tmp = strdup(line);
    char* c = strtok(tmp,",");

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
      train_y_arr[count_y] = atof(c);
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
      test_arr[count_t] = atof(c);
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
      test_y_arr[count_ty] = atof(c);
      count_ty ++;
      c = strtok(NULL, ",");
    }
    free(tmp);
  }

}

void main(int argc, char *argv[]){

  float* train_arr = malloc(TRAIN_ROW*COL*sizeof(float));
  float* train_y_arr = malloc(TRAIN_ROW*1*sizeof(float));
  float* test_arr = malloc(TEST_ROW*COL*sizeof(float));
  float* test_y_arr = malloc(TEST_ROW*1*sizeof(float));


  import_data(train_arr, train_y_arr, test_arr, test_y_arr);
  // NEURAL NETWORK
  // define weights and biases
  float weight_layer1[HIDDEN_NODES*INPUT_NODES];

  float weight_layer2[OUTPUT_NODES*HIDDEN_NODES];

  float input[COL];
  float *output = (float *)malloc(sizeof(TRAIN_ROW*sizeof(float)));

  float bias_layer1[HIDDEN_NODES];
  float bias_layer2[OUTPUT_NODES];

  // define layers of the NN to store the values
  float layer1[HIDDEN_NODES];
  float layer2[OUTPUT_NODES];

  float d3[OUTPUT_NODES];
  float d2[HIDDEN_NODES];

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
  for(int i=0; i < HIDDEN_NODES; i++){
    bias_layer1[i] = ((double)rand())/((double)RAND_MAX);
  }
  for(int i=0; i < OUTPUT_NODES; i++){
    bias_layer2[i] = ((double)rand())/((double)RAND_MAX);
  }

  clock_t t;
  t = clock();
  int p_epoch = 100000;

  float mse_total;
  float mse_old = 100000;
  float mse_difference = 100000;
  float mse_abs = 10000;
  int max_epoch = 0;

  for(int epoch=0; epoch < 10000; epoch++){
    // iterate through input matrix row by row, extracting each row for training
        mse_total = 0.0;
        for(int row = 0; row < TRAIN_ROW; row++){
            for(int col = 0; col < COL; col++){
                input[col] = train_arr[row*COL + col];
            }
            forward_prop(input, weight_layer1, weight_layer2, layer1, layer2, bias_layer1, bias_layer2);
            backprop(input, train_y_arr[row], weight_layer1, weight_layer2, layer1, layer2, d2, d3);
            mse_total += 0.5*d3[0]*d3[0];
        }
        //printf("%f\n", mse_total);
        mse_difference = mse_old - mse_total;
        mse_abs = sqrt(mse_difference*mse_difference);
        //printf("MSE ABS DIFFERENCE FOR EPOCH: %f\n", mse_abs);
  }

  //float a = train_nn(train_arr, train_y_arr, weight_layer1, weight_layer2, layer1, layer2, epoch);
  t = clock() - t;
  double time_serial = ((double)t)/CLOCKS_PER_SEC;
  printf("\nThe runtime for training a serial implementation of a NN, using %d epochs is: %fs\n", p_epoch, time_serial);


  //print weight matrix after training
  for(int i=0; i < HIDDEN_NODES; i++){
    for(int j=0; j < INPUT_NODES; j++){
        printf("%f ", weight_layer1[i*INPUT_NODES +j]);
    }
    printf("\n");
  }

  // predict on training dataset
  for(int row = 0; row < TRAIN_ROW; row++){
      for(int col = 0; col < COL; col++){
          input[col] = train_arr[row*COL + col];
      }
      forward_prop(input, weight_layer1, weight_layer2, layer1, layer2, bias_layer1, bias_layer2);
      for(int i = 0; i < OUTPUT_NODES; i++){
          if(layer2[i]>0.5){
              output[row] = 1;
          }
          else{
              output[row] = 0;
          }
        }
    }

    int count_final=0;

    for(int i = 0; i < TRAIN_ROW; i++){
        //printf("predicted %f\n", output[i]);
        //printf("actual %f\n", train_y_arr[i]);
        if(output[i] == train_y_arr[i]){
            count_final +=1;
        }
    }

    float* output_test = malloc(TEST_ROW*sizeof(float));
    for(int row = 0; row < TEST_ROW; row++){
        for(int col = 0; col < COL; col++){
            input[col] = test_arr[row*COL + col];
        }
        forward_prop(input, weight_layer1, weight_layer2, layer1, layer2, bias_layer1, bias_layer2);
        for(int i = 0; i < OUTPUT_NODES; i++){
            if(layer2[i]>0.5){
                output_test[row] = 1;
            }
            else{
                output_test[row] = 0;
            }
          }
      }

      int count_test=0;

      for(int i = 0; i < TEST_ROW; i++){
          //printf("predicted %f\n", output[i]);
          //printf("actual %f\n", train_y_arr[i]);
          if(output_test[i] == test_y_arr[i]){
              count_test +=1;
          }
      }

    printf("Final count on training dataset%d\n", count_final);
    printf("Final count on testing dataset%d\n", count_test);
    free(output);
    return;
}
