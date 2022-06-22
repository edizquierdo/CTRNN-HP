// Uncomment the following line for table-based fast sigmoid w/ linear interpolation
//#define FAST_SIGMOID

#include "VectorMatrix.h"
#include "random.h"
#include <iostream>
#include <math.h>

#pragma once


// The sigmoid function

#ifdef FAST_SIGMOID
const int SigTabSize = 400;
const double SigTabRange = 15.0;

double fastsigmoid(double x);
#endif

inline double sigma(double x)
{
  return 1/(1 + exp(-x));
}

inline double sigmoid(double x)
{
#ifndef FAST_SIGMOID
  return sigma(x);
#else
  return fastsigmoid(x);
#endif
}


// The inverse sigmoid function

inline double InverseSigmoid(double y)
{
  return log(y/(1-y));
}


// The CTRNN class declaration

class CTRNN {
    public:
        // The constructor
        CTRNN(int newsize = 0, int newwindowsize = 1, double b=0.0, double bt=20.0, double wt=40.0, double wr=16.0, double br=16.0);
        // The destructor
        ~CTRNN();

        // Accessors
        int CircuitSize(void) {return size;};
        void SetCircuitSize(int newsize, int newwindowsize, double b, double bt, double wt,double newwr, double newbr);
        double NeuronState(int i) {return states[i];};
        double &NeuronStateReference(int i) {return states[i];};
        void SetNeuronState(int i, double value)
            {states[i] = value;outputs[i] = sigmoid(gains[i]*(states[i] + biases[i]));};
        double NeuronOutput(int i) {return outputs[i];};
        double &NeuronOutputReference(int i) {return outputs[i];};
        void SetNeuronOutput(int i, double value)
            {outputs[i] = value; states[i] = InverseSigmoid(value)/gains[i] - biases[i];};
        double NeuronBias(int i) {return biases[i];};
        void SetNeuronBias(int i, double value) {biases[i] = value;};
        double NeuronGain(int i) {return gains[i];};
        void SetNeuronGain(int i, double value) {gains[i] = value;};
        double NeuronTimeConstant(int i) {return taus[i];};
        void SetNeuronTimeConstant(int i, double value) {taus[i] = value;Rtaus[i] = 1/value;};
        double NeuronExternalInput(int i) {return externalinputs[i];};
        double &NeuronExternalInputReference(int i) {return externalinputs[i];};
        void SetNeuronExternalInput(int i, double value) {externalinputs[i] = value;};
        double ConnectionWeight(int from, int to) {return weights[from][to];};
        void SetConnectionWeight(int from, int to, double value) {weights[from][to] = value;};
        // -- NEW
        double NeuronRho(int i) {return rhos[i];};
        void SetNeuronRho(int i, double value) {rhos[i] = value;};
        double PlasticityBoundary(int i) {return boundary[i];};
        void SetPlasticityBoundary(int i, double value) {boundary[i] = value;};
        double NeuronBiasTimeConstant(int i) {return tausBiases[i];};
        void SetNeuronBiasTimeConstant(int i, double value) {tausBiases[i] = value; RtausBiases[i] = 1/value;};
        double ConnectionWeightTimeConstant(int from, int to) {return tausWeights[from][to];};
        void SetConnectionWeightTimeConstant(int from, int to, double value) {tausWeights[from][to] = value; RtausWeights[from][to] = 1/value;};
        // --
        void LesionNeuron(int n)
        {
            for (int i = 1; i<= size; i++) {
                SetConnectionWeight(i,n,0);
                SetConnectionWeight(n,i,0);
            }
        }
        void SetCenterCrossing(void);

        // Input and output
        friend ostream& operator<<(ostream& os, CTRNN& c);
        friend istream& operator>>(istream& is, CTRNN& c);

        // Control
        void RandomizeCircuitState(double lb, double ub);
        void RandomizeCircuitState(double lb, double ub, RandomState &rs);
        void RandomizeCircuitOutput(double lb, double ub);
        void RandomizeCircuitOutput(double lb, double ub, RandomState &rs);
        void EulerStep(double stepsize);
        void RK4Step(double stepsize);

        int size;
        int windowsize; // NEW for AVERAGING
        double wr, br; // NEWER for CAPPING
        TVector<double> states, outputs, biases, gains, taus, Rtaus, externalinputs;
        TVector<double> rhos, tausBiases, RtausBiases, boundary; // NEW
        TVector<double> avgoutputs; // NEW for AVERAGING
        TMatrix<double> weights;
        TMatrix<double> tausWeights, RtausWeights; // NEW
        TMatrix<double> outputhist; // NEW for AVERAGING
        TVector<double> TempStates,TempOutputs,k1,k2,k3,k4;
};
