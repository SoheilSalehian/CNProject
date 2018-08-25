package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"
)

type Regression struct {
	weights    []float64
	iterations int
}

const (
	alpha   float64 = 0.001
	beta1   float64 = 0.9
	beta2   float64 = 0.999
	epsilon float64 = 1e-8
)

func (r *Regression) Predict(x [][]float64) []float64 {
	return prediction(x, r.weights)
}

func (r *Regression) Fit(dataset DataSet) error {
	if r.iterations == 0 {
		log.Fatal("Number of iterationss has not been defined.")
	}

	r.weights = adamSolver(dataset, r.iterations)
	return nil
}

func adamSolver(dataset DataSet, iterations int) []float64 {

	m := make([]float64, len(dataset))
	v := make([]float64, len(dataset))
	weights := make([]float64, len(dataset[0].X)+1)

	for i := range weights {
		weights[i] = 0.0
	}

	for t := 1; t < iterations; t++ {
		grad := gradient(dataset, weights, len(dataset))
		for j := range grad {
			m[j] = beta1*m[j] + (1.0-beta1)*grad[j]
			v[j] = beta2*v[j] + (1.0-beta2)*math.Pow(grad[j], 2.0)

			mHat := m[j] / (1.0 - beta1*math.Pow(grad[j], float64(t)))
			vHat := v[j] / (1.0 - beta2*math.Pow(grad[j], float64(t)))

			weights[j] -= (alpha * mHat) / (math.Sqrt(math.Abs(vHat)) + epsilon)
			// if math.IsNaN(weights[j]) {
			// 	fmt.Println("grad: ", mHat, vHat)
			// 	log.Fatal("weight didn't update properly: ", t, j)
			// }
		}
		// fmt.Println(weights)
	}
	return weights
}

func gradient(dataset DataSet, weights []float64, batchSize int) []float64 {

	g := make([]float64, len(weights))

	// FIXME: too many passes on dataset
	input := make([][]float64, len(dataset))
	for i, data := range dataset[0:batchSize] {
		input[i] = data.X
	}

	preds := prediction(input, weights)

	errs := make([]float64, len(preds))
	errSum := 0.0
	for j, data := range dataset[0:batchSize] {
		errs[j] = preds[j] - data.Y
		errSum += errs[j]

		// fmt.Println("Error: ", errs[j])

		for k := 1; k < len(weights); k++ {
			g[k] += (1.0 / float64(len(dataset))) * errs[j] * data.X[k-1]

		}
		g[0] = (1.0 / float64(len(dataset))) * errSum
	}
	fmt.Println(g)

	return g
}

func prediction(x [][]float64, weights []float64) []float64 {
	preds := make([]float64, len(x))
	for i := 0; i < len(x); i++ {
		for j := 1; j < len(x[0]); j++ {
			preds[i] += x[i][j-1] * weights[j]
		}
		preds[i] += weights[0]
	}

	return preds
}

func main() {
	ds := DataSet{}
	err := ds.LoadData("data.csv")
	if err != nil {
		log.Fatal(err)
	}

	trainSet, testSet := ds.Split(0.8)

	testInput := make([][]float64, len(testSet))
	testOutput := make([]float64, len(testSet))

	for i, test := range testSet {
		testInput[i] = test.X
		testOutput[i] = test.Y
	}

	lr := Regression{iterations: 3}
	lr.Fit(trainSet)
	estimate := lr.Predict(testInput)
	fmt.Println(testOutput[0], estimate[0])
	fmt.Println("MSE: ", meanSquaredError(testOutput, estimate))

}

type DataSet []Data

type Data struct {
	X []float64
	Y float64
}

func (d *DataSet) LoadData(fileName string) error {
	f, err := os.Open(fileName)
	if err != nil {
		return fmt.Errorf("could not find csv file %s: %v", fileName, err)
	}
	defer f.Close()

	lines, err := csv.NewReader(f).ReadAll()
	if err != nil {
		return err
	}

	data := make(DataSet, len(lines)-1)

	for i, line := range lines[1:] {
		inputs := []float64{}
		for _, l := range line[2:8] {
			t, err := strconv.Atoi(l)
			if err != nil {
				return err
			}
			inputs = append(inputs, float64(t))
		}
		data[i].X = inputs

		r, err := strconv.Atoi(line[8])
		if err != nil {
			return err
		}
		data[i].Y = float64(r)

	}

	*d = data

	return nil
}

func (d DataSet) Split(p float64) (training, testing DataSet) {
	for i := 0; i < len(d); i++ {
		if p > rand.Float64() {
			training = append(training, d[i])
		} else {
			testing = append(testing, d[i])
		}
	}
	return
}

func meanSquaredError(y, predY []float64) float64 {
	var total float64
	for i := 0; i < len(y); i++ {
		total += (y[i] - predY[i]) * (y[i] - predY[i])
	}

	return total / float64(len(y))
}
