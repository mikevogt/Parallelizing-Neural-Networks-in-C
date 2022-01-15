
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <mpi.h>

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

float diff_Sigmoid(float x){
  return x * (1 - x);
}

float predict(float *input_matrix,
            float *pred_arr,
            float *weight1,
            float *weight2,
            float layer1[HIDDEN_NODES],
            float layer2[OUTPUT_NODES])
            {

    //this will be each extracted input row
    float input[COL];
    float output=0;

    // iterate through input matrix row by row, extracting each row for training
    for(int row = 0; row < TRAIN_ROW; row++){
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

    return 0;
}

float train_nn(float *input_matrix,
              float label[TRAIN_ROW],
              float *weight1,
              float *weight2,
              float layer1[HIDDEN_NODES],
              float layer2[OUTPUT_NODES],
              float float d2[HIDDEN_NODES];
              float d3[OUTPUT_NODES];
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
          for(int i = 0; i < OUTPUT_NODES; i++){
              float error_output = layer2[i] - label[row];
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
      MPI_Barrier(MPI_COMM_WORLD);
    }

    return 0;
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

  int nproc, procID;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  MPI_Comm_rank(MPI_COMM_WORLD, &procID);
  printf("number of processes %d\n", nproc);
  printf("the rank %d\n", procID);

  float train_arr[TRAIN_ROW*COL];
  float train_y_arr[TRAIN_ROW*1];
  float test_arr[TEST_ROW*COL];
  float test_y_arr[TEST_ROW*1];

  import_data(train_arr, train_y_arr, test_arr, test_y_arr);

  // print train_arr
  for(int i=0; i < TRAIN_ROW; i++){
    for(int c=0; c < COL; c++){
        //printf("%f ", train_arr[i*COL + c]);
    }
    //printf("\n");
  }

  // distribute data
  MPI_Bcast(train_arr, TRAIN_ROW*COL, MPI_FLOAT, 0, MPI_COMM_WORLD); //buffer, count, source, comm
  MPI_Bcast(train_y_arr, TRAIN_ROW*1, MPI_FLOAT, 0, MPI_COMM_WORLD);

  // NEURAL NETWORK
  // define weights and biases
  float weight_layer1[HIDDEN_NODES*INPUT_NODES];
  float weight_layer2[OUTPUT_NODES*HIDDEN_NODES];

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

  int epoch = 1000;

  float a = train_nn(train_arr, train_y_arr, weight_layer1, weight_layer2, layer1, layer2, d2, d3, epoch);

  float b = 0.0;

  b = predict(train_arr, output, weight_layer1, weight_layer2, layer1, layer2);

  int count_final=0;

  for(int i = 0; i < TRAIN_ROW; i++){
    //printf("predicted %f\n", output[i]);
    //printf("actual %f\n", train_y_arr[i]);
    if(output[i] == train_y_arr[i]){
        count_final +=1;
    }
  }

  printf("%d\n", count_final);
  free(output);
  return;
}
