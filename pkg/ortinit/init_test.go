package ortinit

import (
	"os"
	"testing"

	"github.com/yalue/onnxruntime_go"
)

// TestMain handles setup and teardown for all tests in the package.
func TestMain(m *testing.M) {
	_ = SetupONNX()

	code := m.Run()

	if onnxruntime_go.IsInitialized() {
		onnxruntime_go.DestroyEnvironment()
	}

	os.Exit(code)
}

func TestSetupONNX(t *testing.T) {
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
