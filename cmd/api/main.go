package main

import (
	"fmt"
	"log"
	"os"
	"os/signal"
	"strconv"
	"syscall"
	"time"

	"github.com/yalue/onnxruntime_go"

	"hugot-gliner2/pkg/gliner"
	"hugot-gliner2/pkg/ortinit"
	"hugot-gliner2/pkg/server"
)

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {
	if err := ortinit.SetupONNX(); err != nil {
		return err
	}
	defer onnxruntime_go.DestroyEnvironment()

	pipeline, err := gliner.NewDynamicPipelineWithOptions(
		env("ENCODER_PATH", "encoder.onnx"),
		env("COUNT_EMBED_PATH", "count_embed.onnx"),
		env("CLASSIFIERS_PATH", "gliner_classifiers.safetensors"),
		env("TOKENIZER_PATH", "tokenizer_out/tokenizer.json"),
		gliner.PipelineOptions{
			Provider:                       gliner.ExecutionProvider(env("INFERENCE_PROVIDER", "auto")),
			CoreMLComputeUnits:             env("COREML_COMPUTE_UNITS", "CPUAndGPU"),
			CoreMLModelFormat:              env("COREML_MODEL_FORMAT", "NeuralNetwork"),
			CoreMLModelCacheDirectory:      os.Getenv("COREML_CACHE_DIR"),
			CoreMLProfileComputePlan:       envBool("COREML_PROFILE", false),
			CoreMLLowPrecisionAccumulation: envBool("COREML_LOW_PRECISION", false),
		},
	)
	if err != nil {
		return fmt.Errorf("load GLiNER model: %w", err)
	}
	defer pipeline.Close()

	config := server.DefaultConfig()
	config.ModelID = env("MODEL_ID", server.DefaultModelID)
	config.APIKey = os.Getenv("API_KEY")
	if config.BodyLimit, err = envInt("MAX_BODY_BYTES", config.BodyLimit); err != nil {
		return err
	}
	if config.MaxTextBytes, err = envInt("MAX_TEXT_BYTES", config.MaxTextBytes); err != nil {
		return err
	}
	if config.QueueDepth, err = envInt("INFERENCE_QUEUE_DEPTH", config.QueueDepth); err != nil {
		return err
	}
	if config.InferenceTimeout, err = envDuration("INFERENCE_TIMEOUT", config.InferenceTimeout); err != nil {
		return err
	}

	app, err := server.New(config, pipeline)
	if err != nil {
		return err
	}
	address := env("ADDR", ":8080")
	errCh := make(chan error, 1)
	go func() { errCh <- app.Listen(address) }()
	log.Printf("GLiNER API listening on %s with model %s", address, config.ModelID)

	signals := make(chan os.Signal, 1)
	signal.Notify(signals, syscall.SIGINT, syscall.SIGTERM)
	select {
	case sig := <-signals:
		log.Printf("received %s; shutting down", sig)
		return app.ShutdownWithTimeout(15 * time.Second)
	case err := <-errCh:
		if err == nil {
			return nil
		}
		return err
	}
}

func env(name, fallback string) string {
	if value := os.Getenv(name); value != "" {
		return value
	}
	return fallback
}

func envInt(name string, fallback int) (int, error) {
	value := os.Getenv(name)
	if value == "" {
		return fallback, nil
	}
	parsed, err := strconv.Atoi(value)
	if err != nil || parsed <= 0 {
		return 0, fmt.Errorf("%s must be a positive integer", name)
	}
	return parsed, nil
}

func envDuration(name string, fallback time.Duration) (time.Duration, error) {
	value := os.Getenv(name)
	if value == "" {
		return fallback, nil
	}
	parsed, err := time.ParseDuration(value)
	if err != nil || parsed <= 0 {
		return 0, fmt.Errorf("%s must be a positive duration", name)
	}
	return parsed, nil
}

func envBool(name string, fallback bool) bool {
	value := os.Getenv(name)
	if value == "" {
		return fallback
	}
	parsed, err := strconv.ParseBool(value)
	if err != nil {
		log.Printf("ignoring invalid boolean %s=%q", name, value)
		return fallback
	}
	return parsed
}
