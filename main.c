#include<stdio.h>
#include "ML.h"

int main(){
    
    int batch_size = -1;  // -1 reads the whole csv file
    int label_column = 3; //label column is at postion 3 (counting from 0)
    char *filePath = "/home/rupam/C/Project/float_table.csv";  //path to the csv file
    int features_len = 3;  // since there are 3 feature inputs in the csv
    int num_features = 8;  // there are 8 features to train
    float learning_rate = 0.1;  //for backpropagation
    int num_epochs = 1000;
    float test[] = {1.0, 1.0, 0.0}; //testing samples

    CSV csv;  
    read_csv(&csv, filePath);  //reading the csv file
    
    makeTableFloat(&csv, batch_size, label_column);      // making the table of features and labels  
    Model model;   
    initialize_weights(&model, features_len); 
    compile(&model, num_features, features_len, "binary_crossentropy", 1 );
    classification_train(&model, &csv.floatTable, csv.floatLabels, learning_rate, num_epochs);
    float pred = predict(&model, test);
    pritnf("prediction is: %f\n", pred);
}