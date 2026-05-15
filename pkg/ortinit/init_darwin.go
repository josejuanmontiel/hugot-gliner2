//go:build darwin

package ortinit

func getDefaultLibName() string {
	return "libonnxruntime.dylib"
}
