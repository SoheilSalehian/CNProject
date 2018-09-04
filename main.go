// TODO
// - Channels
// - Saving weights

package main

import (
	"flag"
	"fmt"
	"log"

	"github.com/pkg/profile"
)

const (
	alpha   float64 = 0.01
	beta1   float64 = 0.9
	beta2   float64 = 0.999
	epsilon float64 = 1e-8
)

var earlyStop = flag.Bool("earlyStop", true, "Enable/Disable early stopping based on NRMSE criteria.")

type Regression struct {
	iterations int
	weights    []float64
}

func (r *Regression) Fit(dataset DataSet) {
	r.weights = adamaxSolver(dataset, r.iterations)
}

func (r *Regression) Predict(x [][]float64) []float64 {
	result, _ := prediction(x, r.weights)
	return result
}

func main() {

	var cpuProfile = flag.Bool("cpuProfile", false, "To enable CPU profiling.")
	if *cpuProfile == true {
		defer profile.Start().Stop()
	}

	iter := flag.Int("iters", 1000, "Number of iterations to run the regression")
	flag.Parse()

	ds := DataSet{}
	err := ds.LoadData("data.csv")
	if err != nil {
		log.Fatal(err)
	}

	ds.Normalize()

	trainSet, testSet := ds.Split(0.7)

	testInputs := make([][]float64, len(testSet))
	testOutputs := make([]float64, len(testSet))

	for i, test := range testSet {
		testInputs[i] = test.X
		testOutputs[i] = test.Y
	}

	lr := Regression{iterations: *iter}
	lr.Fit(trainSet)
	estimate := lr.Predict(testInputs)
	fmt.Println(testOutputs[0], estimate[0])
	fmt.Println("RMSE: ", RMSE(testOutputs, estimate))

	fmt.Println(lr.weights)
}
