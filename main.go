// TODO
// - Adamax
// - Channels
// - Saving weights

package main

import (
	"fmt"
	"log"
)

const (
	alpha   float64 = 0.01
	beta1   float64 = 0.9
	beta2   float64 = 0.999
	epsilon float64 = 1e-8
)

type Regression struct {
	weights []float64
}

func (r *Regression) Fit(dataset DataSet) {
	r.weights = adamaxSolver(dataset)
}

func (r *Regression) Predict(x [][]float64) []float64 {
	result, _ := prediction(x, r.weights)
	return result
}

func main() {
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

	lr := Regression{}
	lr.Fit(trainSet)
	estimate := lr.Predict(testInputs)
	fmt.Println(testOutputs[0], estimate[0])
	fmt.Println("RMSE: ", RMSE(testOutputs, estimate))

	fmt.Println(lr.weights)
}
