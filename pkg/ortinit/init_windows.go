//go:build windows

package ortinit

func getDefaultLibName() string {
	return "onnxruntime.dll"
}
