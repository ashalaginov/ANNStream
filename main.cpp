/* 
 * File:   main.cpp
 * \author Andrey Shalaginov <andrii.shalaginov@hig.no>
 * \brief Implementation of ANN online learning using Amazon Employee Access Challenge
 * Created on August 3, 2014, 3:56 PM
 */

#include <cstdlib>
#include <iostream>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <stdexcept>
#include <omp.h> //OpenMP
#include <climits>
#include <math.h> 
#include <time.h>  
#include <unistd.h>
#include <ctime>

//Include STL
#include <string>
#include <fstream>
#include <vector>
#include <map>


//Parallel
#define CHUNKSIZE 1 /*defines the chunk size as 1 contiguous iteration*/   
#define maxThreads 8

//ANN
#define dimension  9
#define perceptronSize 3
#define maxEpochs 1 //maximal number of epochs in ANN learning 
#define maxAlpha 0.3
#define minAlpha 0
#define useGAOptimization 0 //GA optimization in ANN OR
#define useGoldedSectionOptimization 0 //Golden Section Search optimization in ANN 

//GA
#define maxSteps 10 //maximal number of iterations in GA
#define popSize 5 //size of each generation in Genetic Algorithm
#define mutationNumber 2 //number of mutating individuals in each population of GA
#define crossoverNumber 4 //number of mutating individuals in each population of GA
#define precision  1e-3 //3-point derivative precision

using namespace std;

/**
 * Sigmoid activation function
 * @param argc 1
 * @param argv double argument x
 * @return double sigmoid value
 */
double sigmoid(double x)
{
    return 1 / (1 + exp(-x));
}

/**
 * Initialization operation based on seed argument's values.
 * @param arg IN - seed values.
 * @param individuals IN/OUT - population.
 */
void initialization(vector<double> &individuals)
{
    double tmp;

    for (unsigned int i = 0; i < popSize; i++)
    {
        tmp = maxAlpha - (maxAlpha - minAlpha) * (double) rand() / RAND_MAX;

        individuals.push_back(tmp);
    }
}

void crossover(vector<double> &individuals)
{
    double parent1, parent2, offspring1, offspring2, tmp;
    for (unsigned int i = 0; i < crossoverNumber; i++)
    {
        parent1 = individuals[rand() % individuals.size()];
        parent2 = individuals[rand() % individuals.size()];
        tmp = (double) rand() / RAND_MAX;
        offspring1 = tmp * parent1 + (1 - tmp) * parent2;
        individuals.push_back(offspring1);
        //printf(" off1 %f ", offspring1);
        offspring2 = tmp * parent2 + (1 - tmp) * parent1;
        //printf(" off2 %f ", offspring2);
        individuals.push_back(offspring2);
    }
}

void mutation(vector<double> &individuals)
{
    double mutated;
    int sign;
    for (unsigned int i = 0; i < crossoverNumber; i++)
    {
        mutated = individuals[rand() % individuals.size()];
        (rand() % 2) ? sign = 1 : sign = -1;
        mutated = mutated + sign * maxAlpha;
        individuals.push_back(mutated);
    }
}

double selectionOptimal(vector<double> &individuals, vector<double> &funValues)
{
    double errorOptimal = INT_MAX, optId;
    for (unsigned int i = 0; i < individuals.size(); i++)
    {
        if (funValues[i] < errorOptimal)
        {
            errorOptimal = funValues[i];
            optId = i;
        }
    }

    return individuals[optId];
}

void selection(vector<double> &individuals, vector<double> &funValues)
{
    map<double, double > individualsContainer;
    map<double, double >::iterator it;
    unsigned int i;
    //arrange according function values 
    for (i = 0; i < individuals.size(); i++)
        individualsContainer[funValues[i]] = individuals[i];

    individuals.clear();
    funValues.clear();
    i = 0;
    //selection
    for (it = individualsContainer.begin(); it != individualsContainer.end() && i < popSize; it++)
    {
        individuals.push_back((*it).second);
        funValues.push_back((*it).first);
        i++;
    }

}

/**
 * Calculate the Error function value for the given weight index and input pattern constant
 * @param argc 5
 * @param argv
 * @return 
 */
double errorFunction(const vector<int> &layerId, const vector<int> &neuronConnectionId, const vector<double> &weightValue, const vector<double> &inputData, double classId, double w[][dimension * dimension], double wOutput[dimension])
{
    unsigned int l, j, k, m;
    double tmpSum, yOutput;
    double y[perceptronSize][dimension],
            numWeightsToChange,
            wNew[perceptronSize][dimension * dimension],
            wOutputNew[dimension];

    numWeightsToChange = weightValue.size();

    //copy the weight vector
    //#pragma omp parallel for private(j)
    for (l = 0; l < perceptronSize; l++)
        for (j = 0; j < dimension * dimension; j++)
            wNew[l][j] = w[l][j];

    for (j = 0; j < dimension; j++)
        wOutputNew[j] = wOutput[j];


    //update new weights
    for (m = 0; m < numWeightsToChange; m++)
    {
        //check whether the  output weights or not 
        if (layerId[m] == perceptronSize)
            //update output weights
            wOutputNew[neuronConnectionId[m]] = weightValue[m];
        else
            //update network weights
            wNew[layerId[m]][neuronConnectionId[m]] = weightValue[m];
    }

    //calculate each output neuron signal
    for (l = 0; l < perceptronSize; l++)
    {
        for (j = 0; j < dimension; j++)
        {
            tmpSum = 0;
            // #pragma omp for schedule(static,1)
            // #pragma omp parallel for  reduction(+:tmpSum) num_threads(4)
            for (k = 0; k < dimension; k++)
            {
                if (l == 0)
                    tmpSum += (double) inputData[j] * wNew[l][j * dimension + k];
                else
                    tmpSum += y[l - 1][k] * wNew[l][j * dimension + k]; //check
            }
            //#pragma omp critical
            y[l][j] = sigmoid(tmpSum);
        }
    }

    //calculate output value
    tmpSum = 0;
    for (k = 0; k < dimension; k++)
        tmpSum += y[perceptronSize - 1][k] * wOutputNew[k];
    yOutput = sigmoid(tmpSum);

    return pow(yOutput - classId, 2) / 2;
}

/*
 * 
 */
int main(int argc, char** argv)
{
    vector< vector<double> > inputTrain, inputTest; //train and test datasets patterns
    unsigned int numTrainData = 0, numTestData = 0;
    vector<double> tmp; //temporary variable
    double tmp1; //temporary variables
    FILE *pFileTrain, *pFileTest; //pointer to train and test files
    vector<double> classIDTrain, classIDTest; //train and test classes
    double cl, cl1; //temporary variable

    clock_t begin = clock();
    /* initialize random seed: */
    srand(time(NULL));

    //Read the train data 
    if ((pFileTrain = fopen("train.txt", "rt")) == NULL)
        puts("Error while opening input train file!");

    //Read the test data
    if ((pFileTest = fopen("train.txt", "rt")) == NULL)
        puts("Error while opening input test file!");

    int ground = 0;
    //Parsing train file content into data structure
    while (!feof(pFileTrain))
    {
        tmp.clear();
        for (int i = 0; i < dimension; i++)
        {
            fscanf(pFileTrain, "%lf ", &tmp1);
            tmp.push_back(tmp1);
            //printf("%lf ", tmp1);
        }

        fscanf(pFileTrain, "%lf", &cl);
        fscanf(pFileTrain, "\n");
        //printf(":  %lf\n", cl);

        if (cl == 1)
            ground++;
        if (ground < 2000 || cl == 0)
        {
            classIDTrain.push_back(cl);
            inputTrain.push_back(tmp);
            numTrainData++;
        }

    }

    ground = 0;
    //Parsing test file content into data structure
    while (!feof(pFileTest))
    {

        tmp.clear();
        for (int i = 0; i < dimension; i++)
        {
            fscanf(pFileTest, "%lf ", &tmp1);
            tmp.push_back(tmp1);
            //printf("%lf ", tmp1);
        }
        // printf(":  %lf\n", cl1);
        fscanf(pFileTest, "%lf", &cl1);
        fscanf(pFileTest, "\n");

        if (cl1 > 0.5)
            ground++;
        if (ground < 2000 || cl1 <= 0.5)
        {
            classIDTest.push_back(cl1);

            inputTest.push_back(tmp);
            numTestData++;
        }
    }

    //Frees & Closes
    fclose(pFileTrain);
    fclose(pFileTest);


    //Amount of obtained data samples 
    printf("Train data: %d | Test data: %d \n", numTrainData, numTestData);
    clock_t end = clock();
    printf("\nElasped time is %.8lf seconds.", double(end - begin) / (CLOCKS_PER_SEC));
    //Initialization of the neural network: 
    double y[perceptronSize][dimension],
            yTmp[perceptronSize][dimension],
            delta[perceptronSize][dimension], // error signals 'delta'
            deltaOutput, //error signal of the output neuron
            w[perceptronSize][dimension * dimension], //weights
            yOutput, //output signal
            wOutput[dimension],
            alphaOptimal = maxAlpha,
            tmpSum; //global output signal
    unsigned int i, j, l, k;
    signed int minMaxNormalization[dimension][2]; //min and max for each feature

    //Train data
    //statistics collection for data normalization
    for (i = 0; i < dimension; i++)
    {
        int max = INT_MIN, min = INT_MAX;
        for (j = 0; j < numTrainData; j++)
        {
            if (inputTrain[j][i] > max)
                max = inputTrain[j][i];
            if (inputTrain[j][i] < min)
                min = inputTrain[j][i];
        }
        minMaxNormalization[i][0] = min;
        minMaxNormalization[i][1] = max;
    }

    //data normalization
    for (i = 0; i < dimension; i++)
        for (j = 0; j < numTrainData; j++)

            inputTrain[j][i] = (inputTrain[j][i] - minMaxNormalization[i][0]) / (minMaxNormalization[i][1] - minMaxNormalization[i][0]);

    //Test data
    for (i = 0; i < dimension; i++)
    {
        int max = INT_MIN, min = INT_MAX;
        for (j = 0; j < numTestData; j++)
        {
            if (inputTest[j][i] > max)
                max = inputTest[j][i];
            if (inputTest[j][i] < min)
                min = inputTest[j][i];
        }
        minMaxNormalization[i][0] = min;
        minMaxNormalization[i][1] = max;
    }

    //data normalization
    for (i = 0; i < dimension; i++)
        for (j = 0; j < numTestData; j++)

            inputTest[j][i] = (inputTest[j][i] - minMaxNormalization[i][0]) / (minMaxNormalization[i][1] - minMaxNormalization[i][0]);


    //weights initialization
    for (i = 0; i < perceptronSize; i++)
        for (j = 0; j < dimension * dimension; j++)
            //w[i][j] = (double) 0.8 - 1.2 * (double) rand() / (double) RAND_MAX;
            w[i][j] = 0.5;


    for (j = 0; j < dimension; j++)
        //Output[j] = (double) 0.8 - 1.2 * (double) rand() / (double) RAND_MAX;
        wOutput[j] = 0.5;


    //Experiment
    int numberTraining = numTrainData,
            initialTestStep = 1000,
            numberTesting = numberTraining - initialTestStep;

    //Evaluation of the results based on test data
    double MAE = 0, RRSENumerator = 0, RRSEdenominator = 0, targetedMean = 0, RMSE = 0, MAPE = 0;
    for (i = initialTestStep; i < numberTraining; i++)
        targetedMean += (double) classIDTest[i];
    targetedMean = targetedMean / numberTesting;
    int count = 0, count1 = 0;

    printf("\nData normalization finished");
    printf("\nElasped time is %.8lf seconds.", double(clock() - end) / (CLOCKS_PER_SEC));
    end = clock();

    //For each sample in train data
    for (int g = 0; g < maxEpochs; g++)
    {


        //For each of the samples
        for (i = 0; i < numberTraining; i++)
        {
            printf("Sample id %d \n", i);
            //calculate each output neuron signal
            for (l = 0; l < perceptronSize; l++)
            {
                for (j = 0; j < dimension; j++)
                {
                    tmpSum = 0;
                    for (k = 0; k < dimension; k++)
                    {
                        if (l == 0)
                            tmpSum += (double) inputTrain[i][j] * w[l][j * dimension + k];
                        else
                            tmpSum += y[l - 1][k] * w[l][j * dimension + k]; //check
                    }
                    y[l][j] = sigmoid(tmpSum);
                }
            }
            //calculate output value
            tmpSum = 0;
            for (k = 0; k < dimension; k++)
                tmpSum += y[perceptronSize - 1][k] * wOutput[k];
            yOutput = sigmoid(tmpSum);

            //calculate error signals for each neuron (backward step))
            //output node
            deltaOutput = (classIDTrain[i] - yOutput) * yOutput * (1 - yOutput);
            //deltaOutput = (classIDTrain[i] - yOutput);

            //deltas for other neurons
            l = perceptronSize;
            while (l > 0)
            {
                l--;
                for (j = 0; j < dimension; j++)
                {
                    //last layer
                    if (l == perceptronSize - 1)
                        //delta[perceptronSize - 1][j] = y[l][j] * (1 - y[l][j]) * wOutput[j] * deltaOutput;
                        delta[perceptronSize - 1][j] = deltaOutput * y[l][j] * (1 - y[l][j]) * wOutput[j];
                    else
                    {
                        //other layers
                        tmpSum = 0;
                        for (k = 0; k < dimension; k++)
                            tmpSum += w[l][j * dimension + k] * delta[l + 1][k];
                        delta[l][j] = tmpSum * y[l][j] * (1 - y[l][j]);
                    }
                }
            }

            //calculate weights updates for each neuron (forward step)
            vector<double> alphas;
            vector<double> funValues;
            vector<int> layerId;
            vector<int> neuronConnectionId;
            vector<double> weightValue, weightValue2;
            double derivativeValue, tmpW;
            for (l = 0; l < perceptronSize; l++)
            {
                for (j = 0; j < dimension; j++)
                {
                    for (k = 0; k < dimension; k++)
                    {
                        //optimal alpha estimation for current neuron weight
                        //optimization for all weigths
                        if (useGAOptimization == 1)
                        {
                            alphas.clear();
                            funValues.clear();
                            layerId.clear();
                            neuronConnectionId.clear();
                            weightValue.clear(); //first changed value of the neuron's weight
                            weightValue2.clear(); //second changed value of the neuron weight

                            //nice plot - 1,0  1,1+dimension, 100-100 range
                            //push new values
                            initialization(alphas);
                            layerId.push_back(l);
                            neuronConnectionId.push_back(j * dimension + k);
                            weightValue.push_back(w[l][j * dimension + k] + precision);
                            weightValue2.push_back(w[l][j * dimension + k] - precision);
                            derivativeValue = (errorFunction(layerId, neuronConnectionId, weightValue, inputTrain[i], classIDTrain[i], w, wOutput)
                                    - errorFunction(layerId, neuronConnectionId, weightValue2, inputTrain[i], classIDTrain[i], w, wOutput)) / (2 * precision);
                            tmpW = weightValue[0];

                            //  #pragma omp parallel
                            for (unsigned int t = 0; t < maxSteps; t++)
                            {
                                crossover(alphas);
                                mutation(alphas);
                                //#pragma omp for nowait //fill vector in parallel
                                for (unsigned int y = 0; y < alphas.size(); y++)
                                {

                                    weightValue.clear(); //delete just the weights value
                                    weightValue.push_back(tmpW - alphas[y] * derivativeValue);
                                    //#pragma omp critical
                                    funValues.push_back(errorFunction(layerId, neuronConnectionId, weightValue, inputTrain[i], classIDTrain[i], w, wOutput));
                                }
                                selection(alphas, funValues);
                            }

                            alphaOptimal = selectionOptimal(alphas, funValues);
                        }

                        if (useGoldedSectionOptimization == 1)
                        {
                            double a = minAlpha, b = maxAlpha, x1 = 0, x2 = 0, f1 = 0, f2 = 0;
                            double phi = (1 + sqrt(5)) / 2;

                            layerId.clear();
                            neuronConnectionId.clear();
                            weightValue.clear(); //first changed value of the neuron's weight
                            weightValue2.clear(); //second changed value of the neuron weight

                            //nice plot - 1,0  1,1+dimension, 100-100 range
                            //push new values
                            layerId.push_back(l);
                            neuronConnectionId.push_back(j * dimension + k);
                            weightValue.push_back(w[l][j * dimension + k] + precision);
                            weightValue2.push_back(w[l][j * dimension + k] - precision);
                            derivativeValue = (errorFunction(layerId, neuronConnectionId, weightValue, inputTrain[i], classIDTrain[i], w, wOutput)
                                    - errorFunction(layerId, neuronConnectionId, weightValue2, inputTrain[i], classIDTrain[i], w, wOutput)) / (2 * precision);
                            tmpW = weightValue[0];

                            for (unsigned int t = 0; t < maxSteps; t++)
                            {

                                x1 = b - (b - a) / phi;
                                weightValue.clear(); //delete just the weights value
                                weightValue.push_back(tmpW - x1 * derivativeValue);
                                f1 = errorFunction(layerId, neuronConnectionId, weightValue, inputTrain[i], classIDTrain[i], w, wOutput);

                                x2 = a + (b - a) / phi;
                                weightValue.clear(); //delete just the weights value
                                weightValue.push_back(tmpW - x2 * derivativeValue);
                                f2 = errorFunction(layerId, neuronConnectionId, weightValue, inputTrain[i], classIDTrain[i], w, wOutput);
                                if (f1 >= f2)
                                    a = x1;
                                else
                                    b = x2;
                            }

                            alphaOptimal = (a + b) / 2;

                        }


                        if (l == 0)
                            w[l][j * dimension + k] += alphaOptimal * delta[l][j * dimension + k] * (double) inputTrain[i][j];
                        else
                            w[l][j * dimension + k] += alphaOptimal * delta[l][j * dimension + k] * y[l - 1][k];
                    }
                }
            }

            //output signals
            for (j = 0; j < dimension; j++)
            {
                //optimal alpha estimation
                //optimization for last layer 
                //optimal alpha estimation
                if (useGAOptimization == 1)
                {


                    alphas.clear();
                    funValues.clear();
                    initialization(alphas);
                    layerId.clear();
                    neuronConnectionId.clear();
                    weightValue.clear(); //first changed value of the neuron's weight
                    weightValue2.clear(); //second changed value of the neuron weight
                    //nice plot - 1,0  1,1+dimension, 100-100 range
                    //push new values
                    layerId.push_back(perceptronSize);
                    neuronConnectionId.push_back(j);
                    weightValue.push_back(wOutput[j] + precision);
                    weightValue2.push_back(wOutput[j] - precision);
                    derivativeValue = (errorFunction(layerId, neuronConnectionId, weightValue, inputTrain[i], classIDTrain[i], w, wOutput)
                            - errorFunction(layerId, neuronConnectionId, weightValue2, inputTrain[i], classIDTrain[i], w, wOutput)) / (2 * precision);

                    tmpW = weightValue[0];

                    for (unsigned int t = 0; t < maxSteps; t++)
                    {
                        crossover(alphas);
                        mutation(alphas);
                        for (unsigned int y = 0; y < alphas.size(); y++)
                        {
                            weightValue.clear(); //delete just the weights value
                            weightValue.push_back(tmpW - alphas[y] * derivativeValue);
                            funValues.push_back(errorFunction(layerId, neuronConnectionId, weightValue, inputTrain[i], classIDTrain[i], w, wOutput));
                        }
                        selection(alphas, funValues);
                    }
                    alphaOptimal = selectionOptimal(alphas, funValues);
                }

                if (useGoldedSectionOptimization == 1)
                {
                    double a = minAlpha, b = maxAlpha, x1 = 0, x2 = 0, f1 = 0, f2 = 0;
                    double phi = (1 + sqrt(5)) / 2;

                    layerId.clear();
                    neuronConnectionId.clear();
                    weightValue.clear(); //first changed value of the neuron's weight
                    weightValue2.clear(); //second changed value of the neuron weight

                    //nice plot - 1,0  1,1+dimension, 100-100 range
                    //push new values
                    //push new values
                    layerId.push_back(perceptronSize);
                    neuronConnectionId.push_back(j);
                    weightValue.push_back(wOutput[j] + precision);
                    weightValue2.push_back(wOutput[j] - precision);
                    derivativeValue = (errorFunction(layerId, neuronConnectionId, weightValue, inputTrain[i], classIDTrain[i], w, wOutput)
                            - errorFunction(layerId, neuronConnectionId, weightValue2, inputTrain[i], classIDTrain[i], w, wOutput)) / (2 * precision);

                    tmpW = weightValue[0];

                    for (unsigned int t = 0; t < maxSteps; t++)
                    {

                        x1 = b - (b - a) / phi;
                        weightValue.clear(); //delete just the weights value
                        weightValue.push_back(tmpW - x1 * derivativeValue);
                        f1 = errorFunction(layerId, neuronConnectionId, weightValue, inputTrain[i], classIDTrain[i], w, wOutput);

                        x2 = a + (b - a) / phi;
                        weightValue.clear(); //delete just the weights value
                        weightValue.push_back(tmpW - x2 * derivativeValue);
                        f2 = errorFunction(layerId, neuronConnectionId, weightValue, inputTrain[i], classIDTrain[i], w, wOutput);
                        if (f1 >= f2)
                            a = x1;
                        else
                            b = x2;
                    }

                    alphaOptimal = (a + b) / 2;
                }
                wOutput[j] += alphaOptimal * deltaOutput * y[perceptronSize - 1][j];
            }

            //CHECK new sample from test data
            if (i > initialTestStep)
            {
                //MAE = 0, RRSENumerator = 0, RRSEdenominator = 0, RMSE = 0;
                //calculate each output neuron signal
                for (l = 0; l < perceptronSize; l++)
                {
                    for (j = 0; j < dimension; j++)
                    {
                        tmpSum = 0;
                        for (k = 0; k < dimension; k++)
                        {
                            if (l == 0)
                                tmpSum += (double) inputTest[i][j] * w[l][j * dimension + k];
                            else
                                tmpSum += yTmp[l - 1][k] * w[l][j * dimension + k]; //check
                        }
                        yTmp[l][j] = sigmoid(tmpSum);
                    }
                }
                //calculate output value
                tmpSum = 0;
                for (k = 0; k < dimension; k++)
                    tmpSum += yTmp[perceptronSize - 1][k] * wOutput[k];

                yOutput = sigmoid(tmpSum);
                //printf("%f=%f", yOutput, classIDTest[i]);
                MAE += fabs(yOutput - (double) classIDTest[i]);
                RMSE += pow(yOutput - (double) classIDTest[i], 2);
                RRSENumerator += pow((double) classIDTest[i] - yOutput, 2);
                RRSEdenominator += pow((double) classIDTest[i] - targetedMean, 2);
                MAPE += fabs(((double) classIDTest[i] - yOutput) / (double) classIDTest[i]);
                if (fabs(classIDTest[i] - yOutput) < 0.3)
                    count1++;
            }
        }
    }
    printf("\n Training finished");
    printf("\n Elasped time is %.8lf seconds.", double(clock() - end) / (CLOCKS_PER_SEC));
    end = clock();

    //printf("Properly classified: %f \n", (double) count1 / (numberTraining - initialTestStep));

    printf("\nOnline learning: \n");
    printf("MAE: %f |", MAE / (numberTraining - initialTestStep) / maxEpochs);
    printf("RMSE: %f |", sqrt(RMSE / (numberTraining - initialTestStep) / maxEpochs));
    printf("RRSE: %f |", sqrt(RRSENumerator / RRSEdenominator)*100);
    printf("MAPE  %.4lf %% \n", MAPE * 100 / (numberTraining - initialTestStep) / maxEpochs);
    //Evaluation of the results based on test data
    MAE = 0, RRSENumerator = 0, RRSEdenominator = 0, RMSE = 0, MAPE = 0;

    count = 0, count1 = 0;
    for (i = initialTestStep; i < numberTraining; i++)
    {

        //calculate each output neuron signal
        for (l = 0; l < perceptronSize; l++)
        {
            for (j = 0; j < dimension; j++)
            {
                tmpSum = 0;
                for (k = 0; k < dimension; k++)
                {

                    if (l == 0)
                    {
                        tmpSum += (double) inputTest[i][j] * w[l][j * dimension + k];
                    }
                    else
                    {
                        tmpSum += yTmp[l - 1][k] * w[l][j * dimension + k]; //check
                    }
                    //printf("Offline learning: %f \n", w[l][j * dimension + k]);
                }
                yTmp[l][j] = sigmoid(tmpSum);
            }
        }
        //calculate output value
        tmpSum = 0;
        for (k = 0; k < dimension; k++)
        {
            tmpSum += yTmp[perceptronSize - 1][k] * wOutput[k];

        }
        yOutput = sigmoid(tmpSum);
        // printf("Offline learning: %f \n", tmpSum);

        MAE += fabs(yOutput - (double) classIDTest[i]);
        RMSE += pow(yOutput - (double) classIDTest[i], 2);
        RRSENumerator += pow((double) classIDTest[i] - yOutput, 2);
        RRSEdenominator += pow((double) classIDTest[i] - targetedMean, 2);
        MAPE += fabs(((double) classIDTest[i] - yOutput) / (double) classIDTest[i]);
        
        count++;
        // printf(":  %lf  =  %lf\n", classIDTest[i], yOutput);

        if (fabs(classIDTest[i] - yOutput) < 0.3)
            count1++;
    }
    printf("Offline learning: \n");

    printf("MAE: %f |", MAE / numberTesting);
    printf("RMSE: %f |", sqrt(RMSE / numberTesting));
    printf("RRSE: %f |", sqrt(RRSENumerator / RRSEdenominator)*100);
    //printf("MAPE  %.4lf %% \n ", MAPE * 100 / numberTesting);
   
    printf("\nNumber of samples used to train %d \n", initialTestStep);
    return 0;
}

