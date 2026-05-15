package main

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"strings"

	"hugot-gliner2/pkg/gliner"
	"hugot-gliner2/pkg/math"
	"hugot-gliner2/pkg/ortinit"

	"github.com/yalue/onnxruntime_go"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type MedicalSample struct {
	Text   string   `json:"text"`
	Labels []string `json:"labels"`
}

func main() {
	fmt.Println("🏥 Professional Medical Head Training (Single Output Strategy)...")

	err := ortinit.SetupONNX()
	if err != nil {
		log.Fatalf("Failed to setup ONNX: %v", err)
	}
	defer onnxruntime_go.DestroyEnvironment()

	pipeline, err := gliner.NewPipeline(
		"encoder.onnx", "count_embed.onnx", "gliner_classifiers.safetensors",
		"tokenizer_out/tokenizer.json", "tests/testdata/prompt_ids.json", nil,
	)
	if err != nil {
		log.Fatalf("Failed to load pipeline: %v", err)
	}
	defer pipeline.Close()

	data, _ := os.ReadFile("cmd/train/medical_data.json")
	var samples []MedicalSample
	json.Unmarshal(data, &samples)

	g := gorgonia.NewGraph()
	inputSize, hiddenSize, outputSize := 768, 128, 1

	w1 := gorgonia.NewMatrix(g, tensor.Float32, gorgonia.WithShape(hiddenSize, inputSize), gorgonia.WithName("classifier.0.weight"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))
	b1 := gorgonia.NewMatrix(g, tensor.Float32, gorgonia.WithShape(1, hiddenSize), gorgonia.WithName("classifier.0.bias"), gorgonia.WithInit(gorgonia.Zeroes()))
	w2 := gorgonia.NewMatrix(g, tensor.Float32, gorgonia.WithShape(outputSize, hiddenSize), gorgonia.WithName("classifier.2.weight"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))
	b2 := gorgonia.NewMatrix(g, tensor.Float32, gorgonia.WithShape(1, outputSize), gorgonia.WithName("classifier.2.bias"), gorgonia.WithInit(gorgonia.Zeroes()))

	x := gorgonia.NewMatrix(g, tensor.Float32, gorgonia.WithShape(1, inputSize), gorgonia.WithName("x"))
	y := gorgonia.NewMatrix(g, tensor.Float32, gorgonia.WithShape(1, outputSize), gorgonia.WithName("target"))

	l1 := gorgonia.Must(gorgonia.Add(gorgonia.Must(gorgonia.Mul(x, gorgonia.Must(gorgonia.Transpose(w1)))), b1))
	l1_act := gorgonia.Must(gorgonia.Rectify(l1))
	pred := gorgonia.Must(gorgonia.Add(gorgonia.Must(gorgonia.Mul(l1_act, gorgonia.Must(gorgonia.Transpose(w2)))), b2))
	prob := gorgonia.Must(gorgonia.Sigmoid(pred))
	
	loss := gorgonia.Must(gorgonia.Mean(gorgonia.Must(gorgonia.Square(gorgonia.Must(gorgonia.Sub(prob, y))))))
	gorgonia.Grad(loss, w1, b1, w2, b2)

	solver := gorgonia.NewAdamSolver(gorgonia.WithLearnRate(0.005))
	vm := gorgonia.NewTapeMachine(g, gorgonia.BindDualValues(w1, b1, w2, b2))
	defer vm.Close()

	fmt.Println("\n🧠 Starting Real Fine-Tuning (Single Class Symptom)...")
	
	for epoch := 0; epoch < 10; epoch++ {
		totalLoss := float64(0)
		count := 0
		for _, sample := range samples {
			idsArr, wordIndices, words := gliner.TokenizeAndPool(pipeline.GetTokenizer(), sample.Text)
			fullInputIDs := append(pipeline.GetPromptIDs(), idsArr...)
			fullAttentionMask := make([]int64, len(fullInputIDs))
			for i := range fullAttentionMask { fullAttentionMask[i] = 1 }
			
			hiddenStates, err := pipeline.RunEncoder(fullInputIDs, fullAttentionMask, 1, int64(len(fullInputIDs)))
			if err != nil {
				log.Fatalf("Encoder failed: %v", err)
			}
			promptLen := len(pipeline.GetPromptIDs())

			for i, word := range words {
				isSymptom := false
				for _, label := range sample.Labels {
					if strings.Contains(strings.ToLower(label), strings.ToLower(word)) {
						isSymptom = true
						break
					}
				}

				targetVal := []float32{0.0}
				if isSymptom { targetVal = []float32{1.0} }

				tokenIdx := wordIndices[i]
				offsetStart := (promptLen + tokenIdx) * 768
				offsetEnd := (promptLen + tokenIdx + 1) * 768
				
				if offsetEnd > len(hiddenStates) {
					continue
				}
				emb := hiddenStates[offsetStart:offsetEnd]

				gorgonia.Let(x, tensor.New(tensor.WithShape(1, inputSize), tensor.Of(tensor.Float32), tensor.WithBacking(emb)))
				gorgonia.Let(y, tensor.New(tensor.WithShape(1, outputSize), tensor.Of(tensor.Float32), tensor.WithBacking(targetVal)))

				if err := vm.RunAll(); err != nil {
					log.Fatalf("VM failed: %v", err)
				}
				solver.Step(gorgonia.NodesToValueGrads([]*gorgonia.Node{w1, b1, w2, b2}))
				totalLoss += float64(loss.Value().Data().(float32))
				count++
				vm.Reset()
			}
		}
		fmt.Printf("   Epoch %d | Average Loss: %.6f\n", epoch, totalLoss/float64(count))
	}

	exportMap := make(map[string]*math.Tensor)
	toMathTensor := func(n *gorgonia.Node, isBias bool) *math.Tensor {
		t := n.Value().(tensor.Tensor)
		shape := t.Shape()
		if isBias && len(shape) == 2 && shape[0] == 1 {
			shape = []int{shape[1]} // Flatten (1, N) to (N)
		}
		return &math.Tensor{Data: t.Data().([]float32), Shape: shape}
	}
	exportMap["classifier.0.weight"] = toMathTensor(w1, false)
	exportMap["classifier.0.bias"] = toMathTensor(b1, true)
	exportMap["classifier.2.weight"] = toMathTensor(w2, false)
	exportMap["classifier.2.bias"] = toMathTensor(b2, true)
	gliner.SaveSafetensors("medical_head.safetensors", exportMap)

	fmt.Println("\n🚀 Medical Expert fine-tuned successfully!")
}
