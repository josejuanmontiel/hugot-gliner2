package ortinit

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/yalue/onnxruntime_go"
)

// SetupONNX initializes the ONNX Runtime environment using a robust search strategy:
// 1. Checks the ONNXRUNTIME_LIB_PATH environment variable.
// 2. Checks for the library in the same directory as the executable.
// 3. Defaults to the system's standard library search path with OS-specific names.
func SetupONNX() error {
	if onnxruntime_go.IsInitialized() {
		return nil
	}

	// 1. Priority: Environment variable
	libPath := os.Getenv("ONNXRUNTIME_LIB_PATH")
	if libPath != "" {
		onnxruntime_go.SetSharedLibraryPath(libPath)
		return onnxruntime_go.InitializeEnvironment()
	}

	defaultLib := getDefaultLibName()

	// 2. Strategy: Search relative to executable
	ex, err := os.Executable()
	if err == nil {
		exPath := filepath.Dir(ex)
		localLib := filepath.Join(exPath, defaultLib)
		if _, err := os.Stat(localLib); err == nil {
			onnxruntime_go.SetSharedLibraryPath(localLib)
			return onnxruntime_go.InitializeEnvironment()
		}
	}

	// 3. Last resort: Let the OS find it in standard paths
	onnxruntime_go.SetSharedLibraryPath(defaultLib)

	err = onnxruntime_go.InitializeEnvironment()
	if err != nil {
		return fmt.Errorf("failed to initialize ONNX environment. Set ONNXRUNTIME_LIB_PATH if the library is in a non-standard location: %w", err)
	}

	return nil
}
