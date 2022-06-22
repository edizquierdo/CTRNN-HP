#include "TSearch.h"
#include "CTRNN.h"
#include "random.h"

#define PRINTOFILE

// Task params
const double TransientDuration = 50;
const double RunDuration = 50;
const double StepSize = 0.05;

// Param Var
const double WEIGHTNOISE = 0.0;
const double BIASNOISE = 0.0;

// EA params
const int POPSIZE = 96;
const int GENS = 100;
const double MUTVAR = 0.1;
const double CROSSPROB = 0.5;
const double EXPECTED = 1.1;
const double ELITISM = 0.1;

// Parameter variability modality only
const int Repetitions = 10;

// Nervous system params
const int N = 4;
const double WR = 10.0; //10.0;
const double BR = (WR*N)/2;
const double TMIN = 1; //0.5; //1.0;
const double TMAX = 4; //10.0; //4.0;

// Plasticity parameters
const int WS = 1; //1000;					// Window Size of Plastic Rule (in steps size) (so 1 is no window)
const double B = 0.25; 		// Plasticity Low Boundary
const double BT = 20.0;		// Bias Time Constant
const double WT = 40.0;		// Weight Time Constant

int	VectSize = N*N + 2*N;

// ------------------------------------
// Genotype-Phenotype Mapping Functions
// ------------------------------------
void GenPhenMapping(TVector<double> &gen, TVector<double> &phen)
{
	int k = 1;
	// Time-constants
	for (int i = 1; i <= N; i++) {
		phen(k) = MapSearchParameter(gen(k), TMIN, TMAX);
		k++;
	}
	// Bias
	for (int i = 1; i <= N; i++) {
		phen(k) = MapSearchParameter(gen(k), -BR, BR);
		k++;
	}
	// Weights
	for (int i = 1; i <= N; i++) {
			for (int j = 1; j <= N; j++) {
				phen(k) = MapSearchParameter(gen(k), -WR, WR);
				k++;
			}
	}
}

// ------------------------------------
// Fitness function
// ------------------------------------
double FitnessFunction(TVector<double> &genotype, RandomState &rs)
{
		// Map genootype to phenotype
		TVector<double> phenotype;
		phenotype.SetBounds(1, VectSize);
		GenPhenMapping(genotype, phenotype);

		double fit = 0.0;
		// For each circuit, repeat the experiment 10 times
		for (int r = 1; r <= Repetitions; r += 1) {

			// Create the agent
			CTRNN Agent;

			// Instantiate the nervous system
			Agent.SetCircuitSize(N,WS,B,BT,WT,WR,BR);
			int k = 1;
			// Time-constants
			for (int i = 1; i <= N; i++) {
				Agent.SetNeuronTimeConstant(i,phenotype(k));
				k++;
			}
			// Bias
			for (int i = 1; i <= N; i++) {
				Agent.SetNeuronBias(i,phenotype(k)+rs.GaussianRandom(0.0,BIASNOISE));
				k++;
			}
			// Weights
			for (int i = 1; i <= N; i++) {
					for (int j = 1; j <= N; j++) {
						Agent.SetConnectionWeight(i,j,phenotype(k)+rs.GaussianRandom(0.0,WEIGHTNOISE));
						k++;
					}
			}

			// Initialize the state between [-16,16] at random
			Agent.RandomizeCircuitState(-16.0, 16.0, rs);

			// Run the circuit for the initial transient
			for (double time = StepSize; time <= TransientDuration; time += StepSize) {
					Agent.EulerStep(StepSize);
			}

			// Run the circuit to calculate whether it's oscillating or not
			TVector<double> pastNeuronOutput(1,N);
			double activity = 0.0;
			int steps = 0;
			for (double time = StepSize; time <= RunDuration; time += StepSize) {
					for (int i = 1; i <= N; i += 1) {
						pastNeuronOutput[i] = Agent.NeuronOutput(i);
					}
					Agent.EulerStep(StepSize);
					double magnitude = 0.0;
					for (int i = 1; i <= N; i += 1) {
						magnitude += pow((Agent.NeuronOutput(i) - pastNeuronOutput[i])/StepSize, 2);
					}
					activity += sqrt(magnitude);
					steps += 1;
			}
			fit += activity / steps;
		}
		return fit / Repetitions;
}

// ------------------------------------
// Parameter Perturbation Analysis
// ------------------------------------
void TonicPerturbationAnalysis(TVector<double> &genotype)
{
		RandomState rs;
		ofstream tpafile("tpa.dat");

		// Map genootype to phenotype
		TVector<double> phenotype;
		phenotype.SetBounds(1, VectSize);
		GenPhenMapping(genotype, phenotype);

		// For each circuit, test different amounts of noise
		for (double INPUT = -5.0; INPUT <= 5.0; INPUT += 0.1) {

			// For each magnitude of noise, repeat the experiment 100 times
			for (int x = 1; x <= N; x += 1) {

				double fit = 0.0;
				// For each circuit, repeat the experiment 10 times
				for (int k = 1; k <= Repetitions; k += 1) {

					// Create the agent
					CTRNN Agent;

					// Instantiate the nervous system
					Agent.SetCircuitSize(N,WS,B,BT,WT,WR,BR);

					int kk = 1;
					// Time-constants
					for (int i = 1; i <= N; i++) {
						Agent.SetNeuronTimeConstant(i,phenotype(kk));
						kk++;
					}
					// Bias
					for (int i = 1; i <= N; i++) {
						Agent.SetNeuronBias(i,phenotype(kk));
						kk++;
					}
					// Weights
					for (int i = 1; i <= N; i++) {
							for (int j = 1; j <= N; j++) {
								Agent.SetConnectionWeight(i,j,phenotype(kk));
								kk++;
							}
					}

					// Initialize the state between [-16,16] at random
					Agent.RandomizeCircuitState(-16.0, 16.0, rs);

					Agent.SetNeuronExternalInput(x,INPUT);

					// Run the circuit for the initial transient
					for (double time = StepSize; time <= TransientDuration; time += StepSize) {
							Agent.EulerStep(StepSize);
					}

					// Run the circuit to calculate whether it's oscillating or not
					TVector<double> pastNeuronOutput(1,N);
					double activity = 0.0;
					int steps = 0;
					for (double time = StepSize; time <= RunDuration; time += StepSize) {
							for (int i = 1; i <= N; i += 1) {
								pastNeuronOutput[i] = Agent.NeuronOutput(i);
							}
							Agent.EulerStep(StepSize);
							double magnitude = 0.0;
							for (int i = 1; i <= N; i += 1) {
								magnitude += pow((Agent.NeuronOutput(i) - pastNeuronOutput[i])/StepSize, 2);
							}
							activity += sqrt(magnitude);
							steps += 1;
					}
					fit += activity / steps;
				}

				tpafile << INPUT << " " << x << " " << fit / Repetitions << endl;
		}
	//tpafile << endl;
	}
tpafile.close();
}

// ------------------------------------
// Parameter Perturbation Analysis
// ------------------------------------
void ParameterPerturbationAnalysis(TVector<double> &genotype)
{
		RandomState rs;
		ofstream ppafile("ppa.dat");
		// Map genootype to phenotype
		TVector<double> phenotype;
		phenotype.SetBounds(1, VectSize);
		GenPhenMapping(genotype, phenotype);

		// For each circuit, test different amounts of noise
		for (double NOISE = 0.0; NOISE <= 10.0; NOISE += 0.1) {
			double avgperf = 0.0;
			// For each magnitude of noise, repeat the experiment 100 times
			for (int r = 1; r <= 1000; r += 1)
			{

				// Create the agent
				CTRNN Agent;
				TVector<double> TempVector(1,VectSize-N);
				rs.RandomUnitVector(TempVector);
				// Instantiate the nervous system
				Agent.SetCircuitSize(N,WS,B,BT,WT,WR,BR);
				int k = 1;
				// Time-constants
				for (int i = 1; i <= N; i++) {
					Agent.SetNeuronTimeConstant(i,phenotype(k));
					k++;
				}
				// Bias
				for (int i = 1; i <= N; i++) {
					Agent.SetNeuronBias(i,phenotype(k)+ (NOISE * TempVector[k-N]));
					if (Agent.NeuronBias(i) < -BR)
					 	Agent.SetNeuronBias(i, -BR);
					if (Agent.NeuronBias(i) > BR)
						Agent.SetNeuronBias(i, BR);
					k++;
				}
				// Weights
				for (int i = 1; i <= N; i++) {
						for (int j = 1; j <= N; j++) {
							Agent.SetConnectionWeight(i,j,phenotype(k)+(NOISE * TempVector[k-N]));
							if (Agent.ConnectionWeight(i,j) < -WR)
							 	Agent.SetConnectionWeight(i,j,-WR);
							if (Agent.ConnectionWeight(i,j) > WR)
								 Agent.SetConnectionWeight(i,j,WR);
							k++;
						}
				}

				double fit = 0.0;
				// For each circuit, repeat the experiment 10 times
				for (int k = 1; k <= Repetitions; k += 1) {
					// Initialize the state between [-16,16] at random
					Agent.RandomizeCircuitState(-16.0, 16.0, rs);

					// Run the circuit for the initial transient
					for (double time = StepSize; time <= TransientDuration; time += StepSize) {
							Agent.EulerStep(StepSize);
					}

					// Run the circuit to calculate whether it's oscillating or not
					TVector<double> pastNeuronOutput(1,N);
					double activity = 0.0;
					int steps = 0;
					for (double time = StepSize; time <= RunDuration; time += StepSize) {
							for (int i = 1; i <= N; i += 1) {
								pastNeuronOutput[i] = Agent.NeuronOutput(i);
							}
							Agent.EulerStep(StepSize);
							double magnitude = 0.0;
							for (int i = 1; i <= N; i += 1) {
								magnitude += pow((Agent.NeuronOutput(i) - pastNeuronOutput[i])/StepSize, 2);
							}
							activity += sqrt(magnitude);
							steps += 1;
					}
					fit += activity / steps;
				}
			avgperf += fit / Repetitions;
		}
	ppafile << NOISE << " " << avgperf/1000 << endl;
	}
ppafile.close();
}

// ------------------------------------
// Fitness function
// ------------------------------------
double Behavior(TVector<double> &genotype)
{
		// Map genootype to phenotype
		TVector<double> phenotype;
		phenotype.SetBounds(1, VectSize);
		GenPhenMapping(genotype, phenotype);

		ofstream nfile("neural.dat");
		ofstream wfile("weights.dat");
		ofstream bfile("biases.dat");

		double fit = 0.0;
		// For each circuit, repeat the experiment 10 times
		for (int r = 1; r <= 1; r += 1) {

			// Create the agent
			CTRNN Agent;

			// Instantiate the nervous system
			Agent.SetCircuitSize(N,WS,B,BT,WT,WR,BR);
			//cout << phenotype << endl;
			int k = 1;
			// Time-constants
			for (int i = 1; i <= N; i++) {
				Agent.SetNeuronTimeConstant(i,phenotype(k));
				k++;
			}
			// Bias
			for (int i = 1; i <= N; i++) {
				Agent.SetNeuronBias(i,phenotype(k));
				k++;
			}
			// Weights
			for (int i = 1; i <= N; i++) {
					for (int j = 1; j <= N; j++) {
						Agent.SetConnectionWeight(i,j,phenotype(k));
						k++;
					}
			}

			// Initialize the state between [-16,16] at random
			Agent.RandomizeCircuitState(-16.0, 16.0);

			// Run the circuit to calculate whether it's oscillating or not
			TVector<double> pastNeuronOutput(1,N);
			double activity = 0.0;
			for (double time = 0.0; time <= TransientDuration; time += StepSize) {
					for (int i = 1; i <= N; i += 1) {
						pastNeuronOutput[i] = Agent.NeuronOutput(i);
					}
					Agent.EulerStep(StepSize);
					for (int i = 1; i <= N; i += 1) {
						activity += fabs(Agent.NeuronOutput(i) - pastNeuronOutput[i]);
						nfile << Agent.NeuronOutput(i) << " ";
					}
					nfile << endl;
					for (int i = 1; i <= N; i += 1) {
						bfile << Agent.NeuronBias(i) << " ";
						for (int j = 1; j <= N; j += 1) {
							wfile << Agent.ConnectionWeight(i,j) << " ";
						}
					}
					bfile << endl;
					wfile << endl;
			}

			activity = 0.0;
			Agent.boundary.FillContents(0.0);
			for (double time = 0.0; time <= RunDuration; time += StepSize) {
					for (int i = 1; i <= N; i += 1) {
						pastNeuronOutput[i] = Agent.NeuronOutput(i);
					}
					Agent.EulerStep(StepSize);
					for (int i = 1; i <= N; i += 1) {
						activity += fabs(Agent.NeuronOutput(i) - pastNeuronOutput[i]);
						nfile << Agent.NeuronOutput(i) << " ";
					}
					nfile << endl;
					for (int i = 1; i <= N; i += 1) {
						bfile << Agent.NeuronBias(i) << " ";
						for (int j = 1; j <= N; j += 1) {
							wfile << Agent.ConnectionWeight(i,j) << " ";
						}
					}
					bfile << endl;
					wfile << endl;
			}
			//cout << (activity / 300) / sqrt(N) << endl;

		}
		nfile.close();
		bfile.close();
		wfile.close();
		return ((fit / Repetitions) / 300) / sqrt(N);
}

// ------------------------------------
// Fitness function
// ------------------------------------
void Without(TVector<double> &genotype)
{
		// Map genootype to phenotype
		TVector<double> phenotype;
		phenotype.SetBounds(1, VectSize);
		GenPhenMapping(genotype, phenotype);

		ofstream perffile("wandwout.dat");

		double fitW = 0.0, fitWout = 0.0;
		// For each circuit, repeat the experiment 10 times
		for (int r = 1; r <= 100; r += 1) {

			// Create the agent
			CTRNN Agent;

			// Instantiate the nervous system
			Agent.SetCircuitSize(N,WS,B,BT,WT,WR,BR);
			int k = 1;
			// Time-constants
			for (int i = 1; i <= N; i++) {
				Agent.SetNeuronTimeConstant(i,phenotype(k));
				k++;
			}
			// Bias
			for (int i = 1; i <= N; i++) {
				Agent.SetNeuronBias(i,phenotype(k));
				k++;
			}
			// Weights
			for (int i = 1; i <= N; i++) {
					for (int j = 1; j <= N; j++) {
						Agent.SetConnectionWeight(i,j,phenotype(k));
						k++;
					}
			}

			// Initialize the state between [-16,16] at random
			Agent.RandomizeCircuitState(-16.0, 16.0);

			// Run the circuit for the initial transient
			for (double time = StepSize; time <= TransientDuration; time += StepSize) {
					Agent.EulerStep(StepSize);
			}

			// Run the circuit to calculate whether it's oscillating or not
			TVector<double> pastNeuronOutput(1,N);
			double activity = 0.0;
			for (double time = StepSize; time <= 1000; time += StepSize) {
					for (int i = 1; i <= N; i += 1) {
						pastNeuronOutput[i] = Agent.NeuronOutput(i);
					}
					Agent.EulerStep(StepSize);
					for (int i = 1; i <= N; i += 1) {
						activity += fabs(Agent.NeuronOutput(i) - pastNeuronOutput[i]);
					}
			}
			fitW += activity;
			activity = 0.0;
			Agent.boundary.FillContents(0.0);

			// Run the circuit for the initial transient
			for (double time = StepSize; time <= TransientDuration; time += StepSize) {
					Agent.EulerStep(StepSize);
			}

			for (double time = StepSize; time <= 1000; time += StepSize) {
					for (int i = 1; i <= N; i += 1) {
						pastNeuronOutput[i] = Agent.NeuronOutput(i);
					}
					Agent.EulerStep(StepSize);
					for (int i = 1; i <= N; i += 1) {
						activity += fabs(Agent.NeuronOutput(i) - pastNeuronOutput[i]);
					}
			}
			fitWout += activity;

		}
		perffile << ((fitW / 100) / 1000) / sqrt(N) << " " << ((fitWout / 100) / 1000) / sqrt(N);
		perffile.close();
}

// ------------------------------------
// Display functions
// ------------------------------------
void EvolutionaryRunDisplay(int Generation, double BestPerf, double AvgPerf, double PerfVar)
{
	cout << BestPerf << " " << AvgPerf << " " << PerfVar << endl;
}

void ResultsDisplay(TSearch &s)
{
	TVector<double> bestVector;
	ofstream BestIndividualFile;
	TVector<double> phenotype;
	phenotype.SetBounds(1, VectSize);

	// Save the genotype of the best individual
	bestVector = s.BestIndividual();
	BestIndividualFile.open("best.gen.dat");
	BestIndividualFile << bestVector << endl;
	BestIndividualFile.close();
}

// ------------------------------------
// The main program
// ------------------------------------
int main (int argc, const char* argv[]) {

	long IDUM=-time(0);
	TSearch s(VectSize);

#ifdef PRINTOFILE
	ofstream file;
	file.open("evol.dat");
	cout.rdbuf(file.rdbuf());
#endif

  // Configure the search
  s.SetRandomSeed(IDUM);
	s.SetSearchResultsDisplayFunction(ResultsDisplay);
	s.SetPopulationStatisticsDisplayFunction(EvolutionaryRunDisplay);
  s.SetSelectionMode(RANK_BASED);
  s.SetReproductionMode(GENETIC_ALGORITHM);
  s.SetPopulationSize(POPSIZE);
  s.SetMaxGenerations(GENS);
  s.SetCrossoverProbability(CROSSPROB);
  s.SetCrossoverMode(UNIFORM);
	s.SetMutationVariance(MUTVAR);
  s.SetMaxExpectedOffspring(EXPECTED);
  s.SetElitistFraction(ELITISM);
  s.SetSearchConstraint(1);
	//s.SetReEvaluationFlag(1); //  Parameter Variability Modality Only

	s.SetEvaluationFunction(FitnessFunction);
  s.ExecuteSearch();

	ifstream genefile("best.gen.dat");
	TVector<double> genotype(1, VectSize);
	genefile >> genotype;
	Behavior(genotype);
	//Without(genotype);
	//ParameterPerturbationAnalysis(genotype);
	//TonicPerturbationAnalysis(genotype);

  return 0;
}
