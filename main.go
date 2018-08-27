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
	r.weights = adamSolver(dataset)
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

	trainSet, testSet := ds.Split(0.8)

	testInput := make([][]float64, len(testSet))
	testOutput := make([]float64, len(testSet))

	for i, test := range testSet {
		testInput[i] = test.X
		testOutput[i] = test.Y
	}

	lr := Regression{}
	lr.Fit(trainSet)
	estimate := lr.Predict(testInput)
	fmt.Println(testOutput[0], estimate[0])
	fmt.Println("NRMSE: ", NRMSE(testOutput, estimate))
}
