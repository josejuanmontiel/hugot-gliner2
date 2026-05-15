//go:build linux

package ortinit

func getDefaultLibName() string {
	return "libonnxruntime.so"
}
