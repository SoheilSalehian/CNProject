package main

import (
	"encoding/csv"
	"errors"
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
	alpha   float64 = 0.01
	beta1   float64 = 0.9
	beta2   float64 = 0.999
	epsilon float64 = 1e-8
)

func (r *Regression) Predict(x [][]float64) []float64 {
	result, _ := prediction(x, r.weights)
	return result
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
	weights := make([]float64, len(dataset[0].X))

	for i := range weights {
		weights[i] = 4.0
	}

	input := make([][]float64, len(dataset))
	for i, data := range dataset {
		input[i] = data.X
	}

	output := make([]float64, len(dataset))
	for i, data := range dataset {
		output[i] = data.Y
	}

	t := 1

	e := 0.0
	for {
		grads, preds := gradients(dataset, weights, len(dataset))
		if t%1000 == 0 {
			fmt.Println("Iteration: ", t, "trainig NRMSE: ", math.Abs(NRMSE(preds, output)))
		}

		e += math.Abs(NRMSE(preds, output))

		if math.Abs(NRMSE(preds, output)) < 0.08 {
			fmt.Printf("----Converged----")
			return weights
		}

		for j := range weights {
			lrt := alpha * (math.Sqrt(1.0 - math.Pow(beta2, float64(t)))) /
				(1.0 - math.Pow(beta1, float64(t)))

			m[j] = beta1*m[j] + (1.0-beta1)*grads[j]
			v[j] = beta2*v[j] + (1.0-beta2)*math.Pow(grads[j], 2.0)

			weights[j] -= lrt * (m[j] / (math.Sqrt(v[j]) + epsilon))
		}
		t += 1
	}
	return weights
}

//TODO: Add minibatches
func gradients(dataset DataSet, weights []float64, batchSize int) ([]float64, []float64) {

	g := make([]float64, len(weights))

	// FIXME: too many passes on dataset
	input := make([][]float64, len(dataset))
	for i, data := range dataset[0:batchSize] {
		input[i] = data.X
	}

	preds, err := prediction(input, weights)
	if err != nil {
		log.Fatal(err)
	}

	errs := make([]float64, len(preds))
	for j, data := range dataset[0:batchSize] {
		errs[j] = data.Y - preds[j]
	}

	g, err = dot(transpose(input), errs)
	if err != nil {
		fmt.Println("dot product for gradient calculations failed.")
		log.Fatal(err)
	}

	for k := range g {
		g[k] /= float64(len(dataset)) * -1.0
	}

	return g, preds
}

func prediction(x [][]float64, weights []float64) ([]float64, error) {

	preds, err := dot(x, weights)
	if err != nil {
		log.Fatal(err)
		return nil, err
	}
	return preds, nil
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

	lr := Regression{iterations: 120000}
	lr.Fit(trainSet)
	estimate := lr.Predict(testInput)
	fmt.Println(testOutput[0], estimate[0])
	fmt.Println("NRMSE: ", NRMSE(testOutput, estimate))
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

func NRMSE(y, predY []float64) float64 {
	var total float64
	for i := 0; i < len(y); i++ {
		total += math.Pow((y[i] - predY[i]), 2)
	}
	min, max := minimum(y), maximum(y)

	return (math.Sqrt(total / float64(len(y)))) / (max - min)
}

func (d DataSet) Normalize() {

	input := make([][]float64, len(d))
	for i, data := range d {
		input[i] = data.X
	}

	features := transpose(input)

	for _, feature := range features {
		min, max := minimum(feature), maximum(feature)
		for k, x := range feature {
			feature[k] = (x - min) / (max - min)
		}
	}

	input = transpose(features)

	for i := range d {
		d[i].X = input[i]
	}

}

func transpose(m [][]float64) [][]float64 {
	r := make([][]float64, len(m[0]))
	for x, _ := range r {
		r[x] = make([]float64, len(m))
	}
	for y, s := range m {
		for x, e := range s {
			r[x][y] = e
		}
	}
	return r
}

func minimum(xx []float64) float64 {
	// fmt.Println(xx)
	min := xx[0]
	for _, x := range xx {
		if x < min {
			min = x
		}
	}
	return min
}

func maximum(xx []float64) float64 {
	max := xx[0]
	for _, x := range xx {
		if x > max {
			max = x
		}
	}
	return max
}

func dot(x [][]float64, y []float64) ([]float64, error) {
	if len(x[0]) != len(y) {
		fmt.Println("X: ", len(x[0]), "Y: ", len(y))
		return nil, errors.New("Incorrect dimensions to do the dot product.")
	}

	out := make([]float64, len(x))
	for i := 0; i < len(x); i++ {
		for j := 0; j < len(y); j++ {
			if len(out) < 1 {
				out = make([]float64, len(y))
			}
			out[i] += x[i][j] * y[j]
		}
	}
	return out, nil
}
