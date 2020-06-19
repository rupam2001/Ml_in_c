#include<stdio.h>
#include<stdlib.h>

//helper functions
int len(char **arr){
    int i = 0;
    while(arr[i][0] != '\0'){
        i++;
    }
    return i;
}

typedef struct 
{
    FILE *file;
    char currentViewText[100];
    float **floatTable;
    float *floatLabels;
} CSV;

typedef struct 
{
    /* data */
    float **floatTable;
} Table;


void read_csv(CSV *csv, char *filePath){
    csv->file = fopen(filePath, "r");
    if(csv->file == NULL){ printf("Failed to open %s", filePath); exit(-1);}
}
char* getHeader(CSV *csv){
    fscanf(csv->file, "%s", csv->currentViewText);
    return csv->currentViewText;
}
char** toArray(char *head){
    //spliting the array at ',' s
    int idx = 0;
    int count_comma = 0;
    char **array = (char**)malloc(100*sizeof(char *));
    for (int i = 0; i < 100; i++)
    {
        array[i] = (char*)malloc(50*sizeof(char *));
    }
    
    int column_len = 0;
    while (head[idx] != '\0')
    {
        if(head[idx] == ','){
            count_comma++;
            column_len = 0;
            idx++;
            continue;
        }
        array[count_comma][column_len] = head[idx];
        idx++;   
        column_len++;
    }
    return array;
}

void makeTableFloat( CSV *csv, int batch_size, int label_column){
    char each_row[100];
    int i = 0;
    fgets(each_row, sizeof(each_row), csv->file); // skipping the header (first line)
    csv->floatTable = (float**)malloc(batch_size*sizeof(float*));
    csv->floatLabels = (float*)malloc(batch_size*sizeof(float));
    
    if(batch_size != -1){ 
        // fseek(csv->file,2,SEEK_SET);
        while(i < batch_size && fgets(each_row, sizeof(each_row), csv->file)){
            // printf("loop:%d %s\n", i,each_row);
            char** row_array = toArray(each_row);
            int length = len(row_array);
            float f[length];
            for(int j = 0; j < length; j++){  //traversing each column of ith row
                if(j == label_column){ 
                    csv->floatLabels[i] = strtof(row_array[j], NULL);   //the labels column
                    continue;
                }
                f[j] = strtof(row_array[j], NULL); 
                // printf("%f\n",f);
            }
            csv->floatTable[i] = f;
            i++;
        }
    }
    else{//read the whole csv
        while(fgets(each_row, sizeof(each_row), csv->file)){
            // printf("loop:%d %s\n", i,each_row);
            char** row_array = toArray(each_row);
            int length = len(row_array);
            float f[length];
            for(int j = 0; j < length; j++){
                if(j == label_column) continue;
                f[j] = strtof(row_array[j], NULL); 
                // printf("%f\n",f);
            }
            csv->floatTable[i] = f;
            i++;
        }
    }
}











