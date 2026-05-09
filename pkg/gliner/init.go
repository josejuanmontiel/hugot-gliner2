package gliner

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"

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

	// 2. Strategy: Search relative to executable
	ex, err := os.Executable()
	if err == nil {
		exPath := filepath.Dir(ex)
		var defaultLib string
		switch runtime.GOOS {
		case "windows":
			defaultLib = "onnxruntime.dll"
		case "darwin":
			defaultLib = "libonnxruntime.dylib"
		default: // linux, freebsd, etc.
			defaultLib = "libonnxruntime.so"
		}

		localLib := filepath.Join(exPath, defaultLib)
		if _, err := os.Stat(localLib); err == nil {
			onnxruntime_go.SetSharedLibraryPath(localLib)
			return onnxruntime_go.InitializeEnvironment()
		}
	}

	// 3. Last resort: Let the OS find it in standard paths
	// We still need to set the name, especially for Windows/macOS where it's not always automatic
	switch runtime.GOOS {
	case "windows":
		onnxruntime_go.SetSharedLibraryPath("onnxruntime.dll")
	case "darwin":
		onnxruntime_go.SetSharedLibraryPath("libonnxruntime.dylib")
	default:
		onnxruntime_go.SetSharedLibraryPath("libonnxruntime.so")
	}

	err = onnxruntime_go.InitializeEnvironment()
	if err != nil {
		return fmt.Errorf("failed to initialize ONNX environment. Set ONNXRUNTIME_LIB_PATH if the library is in a non-standard location: %w", err)
	}

	return nil
}
