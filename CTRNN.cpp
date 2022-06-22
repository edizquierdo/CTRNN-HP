#include "CTRNN.h"
#include "random.h"
#include <stdlib.h>

// A fast sigmoid implementation using a table w/ linear interpolation
#ifdef FAST_SIGMOID
int SigTableInitFlag = 0;
double SigTab[SigTabSize];

void InitSigmoidTable(void)
{
  if (!SigTableInitFlag) {
    double DeltaX = SigTabRange/(SigTabSize-1);
    for (int i = 0; i <= SigTabSize-1; i++)
      SigTab[i] = sigma(i * DeltaX);
    SigTableInitFlag = 1;
  }
}

double fastsigmoid(double x)
{
  if (x >= SigTabRange) return 1.0;
  if (x < 0) return 1.0 - fastsigmoid(-x);
  double id;
  double frac = modf(x*(SigTabSize-1)/SigTabRange, &id);
  int i = (int)id;
  double y1 = SigTab[i], y2 = SigTab[i+1];

  return y1 + (y2 - y1) * frac;
}
#endif

// ****************************
// Constructors and Destructors
// ****************************

// The constructor

CTRNN::CTRNN(int newsize, int newwindowsize, double b, double bt, double wt, double wr, double br)
{
	SetCircuitSize(newsize,newwindowsize,b,bt,wt,wr,br);
#ifdef FAST_SIGMOID
  InitSigmoidTable();
#endif
}


// The destructor

CTRNN::~CTRNN()
{
	SetCircuitSize(0,0,0,0,0,0,0);
}


// *********
// Utilities
// *********

// Resize a circuit.

void CTRNN::SetCircuitSize(int newsize, int newwindowsize, double b, double bt, double wt, double newwr, double newbr)
{
	size = newsize;
	states.SetBounds(1,size);
	states.FillContents(0.0);
	outputs.SetBounds(1,size);
	outputs.FillContents(0.0);
	biases.SetBounds(1,size);
	biases.FillContents(0.0);
	gains.SetBounds(1,size);
	gains.FillContents(1.0);
	taus.SetBounds(1,size);
	taus.FillContents(1.0);
	Rtaus.SetBounds(1,size);
	Rtaus.FillContents(1.0);
	externalinputs.SetBounds(1,size);
	externalinputs.FillContents(0.0);
	weights.SetBounds(1,size,1,size);
	weights.FillContents(0.0);
	TempStates.SetBounds(1,size);
	TempOutputs.SetBounds(1,size);
	k1.SetBounds(1,size);
	k2.SetBounds(1,size);
	k3.SetBounds(1,size);
	k4.SetBounds(1,size);

  // NEW
  rhos.SetBounds(1,size);
  rhos.FillContents(0.0);
  tausBiases.SetBounds(1,size);
  tausBiases.FillContents(bt);
  RtausBiases.SetBounds(1,size);
  RtausBiases.FillContents(1/bt);
  boundary.SetBounds(1,size);
  boundary.FillContents(b);
  tausWeights.SetBounds(1,size,1,size);
  tausWeights.FillContents(wt);
  RtausWeights.SetBounds(1,size,1,size);
  RtausWeights.FillContents(1/wt);

  // NEW for AVERAGING
  windowsize = newwindowsize;
  avgoutputs.SetBounds(1,size);
  avgoutputs.FillContents(0.0);
  outputhist.SetBounds(1,size,1,windowsize);
  outputhist.FillContents(0.0);

  // NEW for CAPPING
  wr = newwr;
  br = newbr;
}


// *******
// Control
// *******

// Randomize the states or outputs of a circuit.

void CTRNN::RandomizeCircuitState(double lb, double ub)
{
	for (int i = 1; i <= size; i++)
        SetNeuronState(i, UniformRandom(lb, ub));
  // Fill the window with the first value
  for (int i = 1; i <= size; i++)
    for (int k = 1; k <= windowsize; k++)
      outputhist[i][k] = NeuronOutput(i);
}

void CTRNN::RandomizeCircuitState(double lb, double ub, RandomState &rs)
{
	for (int i = 1; i <= size; i++)
    SetNeuronState(i, rs.UniformRandom(lb, ub));
  // Fill the window with the first value
  for (int i = 1; i <= size; i++)
    for (int k = 1; k <= windowsize; k++)
      outputhist[i][k] = NeuronOutput(i);
}

void CTRNN::RandomizeCircuitOutput(double lb, double ub)
{
	for (int i = 1; i <= size; i++)
        SetNeuronOutput(i, UniformRandom(lb, ub));
  // Fill the window with the first value
  for (int i = 1; i <= size; i++)
    for (int k = 1; k <= windowsize; k++)
      outputhist[i][k] = NeuronOutput(i);
}

void CTRNN::RandomizeCircuitOutput(double lb, double ub, RandomState &rs)
{
	for (int i = 1; i <= size; i++)
    SetNeuronOutput(i, rs.UniformRandom(lb, ub));
  // Fill the window with the first value
  for (int i = 1; i <= size; i++)
    for (int k = 1; k <= windowsize; k++)
      outputhist[i][k] = NeuronOutput(i);
}

// Integrate a circuit one step using Euler integration.

void CTRNN::EulerStep(double stepsize)
{
  // Update the state of all neurons.
  for (int i = 1; i <= size; i++) {
    double input = externalinputs[i];
    for (int j = 1; j <= size; j++)
      input += weights[j][i] * outputs[j];
    states[i] += stepsize * Rtaus[i] * (input - states[i]);
  }
  // Update the outputs of all neurons.
  for (int i = 1; i <= size; i++)
    outputs[i] = sigmoid(gains[i] * (states[i] + biases[i]));

  // Keep track of the running average of the outputs for some predetermined window of time.
  // 1. Update window
  for (int i = 1; i <= size; i++){
    // Slide all the values down by one (effectively deleting the oldest one, and making run for a new one)
    for (int k = 1; k < windowsize; k++){
      outputhist[i][k] = outputhist[i][k+1];
    }
    // Add the new one at the end (in the empty space)
    outputhist[i][windowsize] = outputs[i];
  }
  // 2. Take average
  for (int i = 1; i <= size; i++){
    avgoutputs[i] = 0.0;
    for (int k = 1; k <= windowsize; k++){
      avgoutputs[i] += outputhist[i][k];
    }
    avgoutputs[i] = avgoutputs[i]/windowsize;
  }

  // NEW: Update rho for each neuron.
  for (int i = 1; i <= size; i++) {
    if (avgoutputs[i] < boundary[i]) {
      rhos[i] = (boundary[i] - avgoutputs[i])/boundary[i];
    }
    else{
      if (avgoutputs[i] > (1-boundary[i])){
        rhos[i] = ((1-boundary[i]) - avgoutputs[i])/boundary[i];
      }
      else
      {
        rhos[i] = 0.0;
      }
    }
  }
  // NEW: Update Biases
  for (int i = 1; i <= size; i++){
    biases[i] += stepsize * RtausBiases[i] * rhos[i];
    if (biases[i] > br){
        biases[i] = br;
    }
    else{
        if (biases[i] < -br){
            biases[i] = -br;
        }
    }
  }
  // NEW: Update Weights
  for (int i = 1; i <= size; i++) {
      for (int j = 1; j <= size; j++){
        weights[i][j] += stepsize * RtausWeights[i][j] * rhos[j] * fabs(weights[i][j]);
        if (weights[i][j] > wr){
            weights[i][j] = wr;
        }
        else{
            if (weights[i][j] < -wr){
                weights[i][j] = -wr;
            }
        }
      }
  }
}


// Set the biases of the CTRNN to their center-crossing values

void CTRNN::SetCenterCrossing(void)
{
    double InputWeights, ThetaStar;

    for (int i = 1; i <= CircuitSize(); i++) {
        // Sum the input weights to this neuron
        InputWeights = 0;
        for (int j = 1; j <= CircuitSize(); j++)
            InputWeights += ConnectionWeight(j, i);
        // Compute the corresponding ThetaStar
        ThetaStar = -InputWeights/2;
        SetNeuronBias(i, ThetaStar);
    }
}


// ****************
// Input and Output
// ****************

#include <iomanip>

ostream& operator<<(ostream& os, CTRNN& c)
{
	// Set the precision
	os << setprecision(32);
	// Write the size
	os << c.size << endl << endl;
	// Write the time constants
	for (int i = 1; i <= c.size; i++)
		os << c.taus[i] << " ";
	os << endl << endl;
	// Write the biases
	for (int i = 1; i <= c.size; i++)
		os << c.biases[i] << " ";
	os << endl << endl;
	// Write the gains
	for (int i = 1; i <= c.size; i++)
		os << c.gains[i] << " ";
	os << endl << endl;
	// Write the weights
	for (int i = 1; i <= c.size; i++) {
		for (int j = 1; j <= c.size; j++)
			os << c.weights[i][j] << " ";
		os << endl;
	}
	// Return the ostream
	return os;
}

istream& operator>>(istream& is, CTRNN& c)
{
	// Read the size
	int size;
	is >> size;
	c.SetCircuitSize(size, 1, 1, 1, 1, 1, 1); //windowsize
	// Read the time constants
	for (int i = 1; i <= size; i++) {
		is >> c.taus[i];
		c.Rtaus[i] = 1/c.taus[i];
	}
	// Read the biases
	for (int i = 1; i <= size; i++)
		is >> c.biases[i];
	// Read the gains
	for (int i = 1; i <= size; i++)
		is >> c.gains[i];
	// Read the weights
	for (int i = 1; i <= size; i++)
		for (int j = 1; j <= size; j++)
			is >> c.weights[i][j];
	// Return the istream
	return is;
}
