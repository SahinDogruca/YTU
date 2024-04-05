#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>

#define TRAINING_SIZE 160

void fillDict(FILE *file, char ***dict_ref, char ****texts_ref, int *textLine_ref, int *dictCap_ref, int *dictLen_ref)
{
    char line[1000];
    char *token = NULL;
    int lineCount = 0;

    while (fgets(line, 1000, file) != NULL)
    {
        lineCount = 0;
        line[strcspn(line, "\n")] = 0;
        token = strtok(line, " ");

        while (token != NULL)
        {
            bool alreadyExists = false;
            for (int i = 0; i < *dictLen_ref; i++)
            {
                if (strcmp(token, (*dict_ref)[i]) == 0)
                {
                    alreadyExists = true;
                    break;
                }
            }
            if (!alreadyExists)
            {
                (*dict_ref)[*dictLen_ref] = (char *)malloc(sizeof(char) * (strlen(token) + 1));
                strcpy((*dict_ref)[*dictLen_ref], token);
                (*dictLen_ref)++;
            }

            (*texts_ref)[*textLine_ref][lineCount] = (char *)malloc(sizeof(char) * (strlen(token) + 1));
            strcpy((*texts_ref)[*textLine_ref][lineCount], token);

            if (*dictLen_ref == (*dictCap_ref))
            {
                (*dictCap_ref) *= 2;
                (*dict_ref) = (char **)realloc((*dict_ref), sizeof(char *) * (*dictCap_ref));
            }

            token = strtok(NULL, " ");
            lineCount++;
        }

        (*texts_ref)[*textLine_ref] = (char **)realloc((*texts_ref)[*textLine_ref], sizeof(char *) * lineCount);
        (*textLine_ref)++;
    }
}

void createDict(char ***dict_ref, char ****texts_ref, int *dictCap_ref)
{
    (*dict_ref) = (char **)malloc(sizeof(char *) * (*dictCap_ref));
    int dictLen = 0;
    char *token = NULL;
    int textLine = 0;

    FILE *file = fopen("texts.txt", "r");

    fillDict(file, dict_ref, texts_ref, &textLine, dictCap_ref, &dictLen);

    (*dict_ref) = (char **)realloc((*dict_ref), sizeof(char *) * dictLen);
    (*dictCap_ref) = dictLen;

    fclose(file);
}

bool contains(char *word, char **text)
{
    for (int i = 0; i < 1000; i++)
    {
        if (text[i] == NULL)
            break;

        if (strcmp(word, text[i]) == 0)
        {
            // printf("Word: %s\n", word);
            return true;
        }
    }

    return false;
}

float compute_cost(bool **hotVectors, float *labels, int dictCap, float *weights)
{
    float cost = 0;
    for (int i = 0; i < TRAINING_SIZE; i++)
    {
        float prediction = 0;
        for (int j = 0; j < dictCap; j++)
        {
            prediction += tanh((float)hotVectors[i][j] * weights[j]);
        }
        cost += pow(prediction - labels[i], 2);
    }
    cost /= (2 * TRAINING_SIZE);
    return cost;
}

void compute_gradient(bool **hotVectors, float *labels, int dictCap, float *weights, float **grad_ref)
{
    for (int i = 0; i < dictCap; i++)
    {
        (*grad_ref)[i] = 0;
        for (int j = 0; j < TRAINING_SIZE; j++)
        {
            (*grad_ref)[i] += (tanh((float)hotVectors[j][i] * weights[i]) - labels[j]) * (float)hotVectors[j][i] * (1 / pow(cosh((float)hotVectors[j][i] * weights[i]), 2));
        }
        (*grad_ref)[i] /= TRAINING_SIZE;
    }
}

void gradient_descent(float **c_hist, float ***w_hist, bool **hotVectors, float *labels, int dictCap, float *weights, float alpha, int num_iters)
{
    float *grad = (float *)malloc(sizeof(float) * dictCap);
    for (int i = 0; i < num_iters; i++)
    {
        compute_gradient(hotVectors, labels, dictCap, weights, &grad);
        for (int j = 0; j < dictCap; j++)
        {
            weights[j] -= alpha * grad[j];
        }

        if (i < 100000)
        {
            for (int j = 0; j < dictCap; j++)
                (*w_hist)[i][j] = weights[j];
            (*c_hist)[i] = compute_cost(hotVectors, labels, dictCap, weights);
        }

        if (i % (int)ceil(num_iters / 10) == 0)
            printf("iter: %d, cost: %f\n", i, (*c_hist)[i]);
    }

    free(grad);
}

void gd_test(float *weights, bool **hotVectors, float *labels, int dictCap)
{
    int i, count = 0, isTrue;
    for (i = TRAINING_SIZE; i < 200; i++)
    {
        float prediction = 0;
        for (int j = 0; j < dictCap; j++)
            prediction += (float)hotVectors[i][j] * weights[j];

        prediction = tanh(prediction);
        isTrue = fabs(labels[i] - prediction) < 0.5 ? 1 : 0;
        count += isTrue;
        printf("iter: %d, Prediction: %f, Label: %.1f, isTrue: %d\n", i, prediction, labels[i], isTrue);
    }

    printf("True Count : %d\n", count);
}

int main()
{
    char **dict;
    int dictCap = 100;
    int i = 0, j = 0;

    char ***texts = (char ***)malloc(sizeof(char **) * 200);
    for (i = 0; i < 200; i++)
        texts[i] = (char **)malloc(sizeof(char *) * 100);

    createDict(&dict, &texts, &dictCap);

    bool **hotVectors;
    hotVectors = (bool **)malloc(sizeof(bool *) * 200);
    for (i = 0; i < 200; i++)
    {
        hotVectors[i] = (bool *)malloc(sizeof(bool) * dictCap);
        for (j = 0; j < dictCap; j++)
            hotVectors[i][j] = contains(dict[j], texts[i]);
    }

    float *labels = (float *)malloc(sizeof(float) * 200);
    for (i = 0; i < 200; i++)
    {
        if (i < 80 || (i >= 160 && i < 180))
            labels[i] = 1;
        else
            labels[i] = -1;
    }

    srand(time(NULL));

    float *weights = (float *)malloc(sizeof(float) * dictCap);
    for (i = 0; i < dictCap; i++)
        weights[i] = (float)rand() / RAND_MAX;

    int num_iters = 10000;
    int epocs = (num_iters < 100000 ? num_iters : 100000);

    float **w_hist = (float **)malloc(sizeof(float *) * epocs);
    for (i = 0; i < epocs; i++)
        w_hist[i] = (float *)malloc(sizeof(float) * dictCap);

    float *c_hist = (float *)malloc(sizeof(float) * epocs);

    gradient_descent(&c_hist, &w_hist, hotVectors, labels, dictCap, weights, 0.01, num_iters);

    printf("Final Cost: %f\n", compute_cost(hotVectors, labels, dictCap, weights));
    printf("Final Weights: ");
    for (i = 0; i < dictCap; i++)
        printf("%f ", weights[i]);

    gd_test(weights, hotVectors, labels, dictCap);

    // Free allocated memory
    for (i = 0; i < dictCap; i++)
        free(dict[i]);
    free(dict);

    for (i = 0; i < 200; i++)
    {
        j = 0;
        while (texts[i] != NULL && texts[i][j] != NULL)
            free(texts[i][j++]);
        free(texts[i]);
    }
    free(texts);

    for (i = 0; i < 200; i++)
        free(hotVectors[i]);
    free(hotVectors);

    free(labels);
    free(weights);

    for (i = 0; i < epocs; i++)
        free(w_hist[i]);
    free(w_hist);
    free(c_hist);

    return 0;
}