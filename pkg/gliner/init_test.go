package gliner

import (
	"os"
	"testing"

	"github.com/yalue/onnxruntime_go"
)

// TestMain handles setup and teardown for all tests in the package.
// It ensures ONNX is initialized before running any tests that might need it.
func TestMain(m *testing.M) {
	// We call SetupONNX, but we don't fail here if it fails, 
	// because some tests (like math_test.go) don't need it.
	// Individual tests that need ONNX should check if it's initialized.
	_ = SetupONNX()

	code := m.Run()

	if onnxruntime_go.IsInitialized() {
		onnxruntime_go.DestroyEnvironment()
	}

	os.Exit(code)
}

func TestSetupONNX(t *testing.T) {
	// If the environment variable is set, this should definitely work.
	// If not, it might fail in CI/CD environments without ONNX installed, 
	// so we log instead of failing if the env var is missing.
	err := SetupONNX()
	if err != nil {
		if os.Getenv("ONNXRUNTIME_LIB_PATH") != "" {
			t.Fatalf("SetupONNX failed despite ONNXRUNTIME_LIB_PATH being set: %v", err)
		}
		t.Logf("SetupONNX skipped or failed (expected if no lib is found): %v", err)
	} else {
		t.Log("SetupONNX successfully initialized the environment")
	}
}
