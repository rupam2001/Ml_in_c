#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include "csv_reader.h"

//helper functions
int absolute(int a){
    return a > 0 ? a: a*(-1);
}

float factorial(int n){
    if(n == 0 || n == 1) return 1;
    else return n*factorial(n-1);
}
float power(float x, int p ){
    if(p == 1) return x;
    if(p == 0) return 1;
    else return x * power(x, p-1);
}

float expR(float x){  // e^(-x)
    int limit = 10;
    float result = 1;
    for(int i = 1; i < limit; i++){
        if((i+1)%2 == 0) 
            result -= power(x,i) / factorial(i);
        else
            result += power(x,i) / factorial(i);
    }
    return result;
}


float logE(float x){   //ln(x)
    float result = 0.0;
    int limit = 30;
    float term =  (x - 1 ) / (x + 1);

    for(int i = 1; i < limit; i++){
        result += power(term, 2*i - 1) / ( 2*i - 1); 
    }
    result *= 2;
    return result;
}

float logX(float x){  //log(x)
    int limit = 100;
    float result = 0;
    for(int i = 1; i < limit; i++){
        result -= power(x, i) / i;
    }
    return result;
}

// single layer neuron
typedef struct model
{
    float *W;   //weights
    float b;  //bias
    float **X;
    float *Y;
    float *a;
    int size_W;
    int size_X_per;  //size of the each input feature vector
    int num_X;  // number of training examples
    char *loss_fn_choice;
    int show_cost;
    float cost;  //after performing summation
    float learning_rate;
    char *feature_cross;   // none, [X^2], [X, X^2], [X1xX2..], [X^2, X1xX2..], [X, X^2, X1xX2..] (not in use)
    
} Model;


void initialize_weights(Model *M, int num_W){

    int x = time(NULL);  //getting the time in miliseconds
    int P1 = 263, P2 = 71;  //for generating random numbers

    M->W = (float *)malloc(num_W*sizeof(float));
    M->b = 0;
    if(M->W == NULL ){printf("Can not allocate memory...\n"); exit(-1);}

    for(int i = 0; i < num_W; i++){
        x = ((P1 * x) + P2 )%1000 ;
        M->W[i] = x * 0.001;
    }
    M->size_W = num_W;
    M->feature_cross = "none";
}


void compile(Model *M, int num_X, int size_X_per, char *loss_fn_choice, int show_cost){
    
   if(loss_fn_choice != "mse" && loss_fn_choice != "binary_crossentropy"){
       printf("\nUnknown loss function\n");
       exit(-1);
   } 
   
   M->loss_fn_choice = loss_fn_choice;
   M->show_cost = show_cost;
   M->num_X = num_X;
   M->a = (float*) calloc(num_X, sizeof(float));
   if(size_X_per != M->size_W){printf("input shape not matching\n"); exit(-1);}
   M->size_X_per = size_X_per;


}


static float sigmoid(float z){    //returns a
    float s = 1 / (1 + expR(z));
    return s;
}



static float loss_fn(float a, float y){   //for binary_crossentropy only

    float l  = - (y * logE(a)) - ((1 - y) * logE(1-a));
    
    return l;

}

static void back_propagate(Model *M, float **X, float *Y){
    //compute gradient
    float *dw = (float*) calloc(M->size_W, sizeof(float));
    float db = 0.0;

    for(int i = 0; i < M->num_X; i++){  //for each example vector
        for(int j = 0; j < M->size_W; j++ ){  //for each input feature element
            dw[j] += (X[i][j] * (M->a[i] - Y[i]));
        }
        db += (M->a[i] - Y[i]);
    }
    // printf("%f\n", db);
    for(int j = 0; j < M->size_W; j++ ){
        M->W[j] = M->W[j] - ((M->learning_rate * dw[j]) / M->num_X); 
        // M->b = M->b - (M->learning_rate * db);
    }
    db /= M->num_X;
    M->b = M->b - (M->learning_rate * db);
}



static void forwar_propagate(Model *M, float **X){
    float z ;
    M->cost = 0;
    for(int i = 0; i < M->num_X; i++){
        z = 0.0;
        for(int j = 0; j < M->size_W; j++ ){
          z +=  (M->W[j] * X[i][j]);
        }
        z += M->b;
        M->a[i] = sigmoid(z);
        M->cost += loss_fn( M->a[i], M->Y[i]);
    }
    M->cost = M->cost / M->num_X ;   //average of losts
    if(M->show_cost == 1){
        printf("cost: %f\n", M->cost);
    }
}


void classification_train(Model *M, float **X, float *Y, float learining_rate, int epochs){
    
    M->Y = Y;
    M->learning_rate = learining_rate;
    // printf("%f", X[0][0]);
    

    printf("training....\n");
    for(int ep = 0; ep < epochs; ep++){
        forwar_propagate(M, X);
        back_propagate(M, X, Y);
    }
    printf("Done!\n");
}

float predict(Model *M, float *X){
    float result ;
    for(int i = 0; i < M->size_W; i++){
        result += X[i] * M->W[i];
    }
    result += M->b;
    if(M->loss_fn_choice == "binary_crossentropy")
        result = sigmoid(result);
    else if (M->loss_fn_choice == "mse");

    return result;
}

//////linear_regressor
static float loss_fn_MSE(float a, float y){
    float mse = power( (y - a), 2);
    return mse;
}


static void forward_propagate_linearRegressor(Model *M, float X[][M->size_X_per]){
    float z ;
    M->cost = 0;
    for(int i = 0; i < M->num_X; i++){
        z = 0.0;
        for(int j = 0; j < M->size_W; j++ ){
          z +=  (M->W[j] * X[i][j]);
        }
        z += M->b;
        M->a[i] = z;
        M->cost += loss_fn_MSE( M->a[i], M->Y[i]);
    }
    M->cost = M->cost / M->num_X ;   //average of losts
    if(M->show_cost == 1){
        printf("cost: %f\n", M->cost);
    }
}
static void back_propagate_linearRegressor(Model *M, float X[][M->size_X_per], float *Y){
    //compute gradient
    float *dw = (float*) calloc(M->size_W, sizeof(float));
    float db = 0.0;

    for(int i = 0; i < M->num_X; i++){  //for each example vector
        for(int j = 0; j < M->size_W; j++ ){  //for each input feature element
            dw[j] += -2 * (X[i][j] * (Y[i] - M->a[i]));
        }
        db += -2 * (Y[i] - M->a[i]);
    }
    for(int j = 0; j < M->size_W; j++ ){
        M->W[j] = M->W[j] - ((M->learning_rate * dw[j]) / M->num_X); 
    }
    db /= M->num_X;
    M->b = M->b - (M->learning_rate * db);
    free(dw);////////////////////////////////////////////////freeing
}

void linearRegressor_train(Model *M, float X[][M->size_X_per], float *Y, float learining_rate, int epochs ){
    M->Y = Y;
    M->learning_rate = learining_rate;
    for (int i = 0; i < epochs; i++)
    {
        forward_propagate_linearRegressor(M, X);
        back_propagate_linearRegressor(M, X, Y);
    }   
}


// static void feature_cross(Model *M, float X[][M->size_X_per], float *Y, int epochs){

//     if(M->feature_cross == "X^2"){
//         float **X_crossed = (float **)malloc(M->num_X * sizeof(float *));
//         for (int i = 0; i < M->size_X_per; i++)
//         {
//             X_crossed[i] = (float *)malloc(M->size_X_per * sizeof(float));
//         }
        
//         for(int i = 0; i < M->num_X; i++){
//             for(int j = 0; j < M->size_X_per; j++){
//                 X_crossed[i][j] = power(X[i][j], 2);   
//             }
//         }
//         printf("training....\n");
//         for(int ep = 0; ep < epochs; ep++){
//             forwar_propagate(M, X_crossed);
//             back_propagate(M, X_crossed, Y);
//         }
//         printf("Done!\n");
//     }
    
    
// }




