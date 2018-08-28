package main

import (
	"encoding/csv"
	"fmt"
	"math/rand"
	"os"
	"strconv"
)

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
