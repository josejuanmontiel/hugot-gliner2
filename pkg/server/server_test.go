package server

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http/httptest"
	"strings"
	"testing"

	"hugot-gliner2/pkg/gliner"
)

type fakeEngine struct {
	ready  error
	result gliner.ExtractionResult
}

func TestSwaggerDocumentation(t *testing.T) {
	app, err := New(DefaultConfig(), &fakeEngine{})
	if err != nil {
		t.Fatal(err)
	}

	specResponse, err := app.Test(httptest.NewRequest("GET", "/openapi.json", nil))
	if err != nil {
		t.Fatal(err)
	}
	if specResponse.StatusCode != 200 || !strings.HasPrefix(specResponse.Header.Get("Content-Type"), "application/json") {
		t.Fatalf("spec response status=%d content-type=%q", specResponse.StatusCode, specResponse.Header.Get("Content-Type"))
	}
	var spec map[string]any
	if err := json.NewDecoder(specResponse.Body).Decode(&spec); err != nil {
		t.Fatalf("invalid OpenAPI JSON: %v", err)
	}
	paths := spec["paths"].(map[string]any)
	if _, ok := paths["/v1/chat/completions"]; !ok {
		t.Fatal("OpenAPI document does not describe chat completions")
	}

	uiResponse, err := app.Test(httptest.NewRequest("GET", "/swagger/", nil))
	if err != nil {
		t.Fatal(err)
	}
	ui, err := io.ReadAll(uiResponse.Body)
	if err != nil {
		t.Fatal(err)
	}
	if uiResponse.StatusCode != 200 || !bytes.Contains(ui, []byte("/openapi.json")) {
		t.Fatalf("unexpected Swagger UI response: status=%d", uiResponse.StatusCode)
	}
}

func (f *fakeEngine) Ready() error { return f.ready }
func (f *fakeEngine) Infer(context.Context, string, gliner.Schema, gliner.InferOptions) (gliner.ExtractionResult, error) {
	return f.result, nil
}

func TestChatCompletionsContract(t *testing.T) {
	engine := &fakeEngine{result: gliner.ExtractionResult{
		Entities:        map[string][]gliner.SpanResult{"person": {{Text: "Ada", Confidence: .95, Start: 0, End: 3}}},
		Classifications: map[string][]gliner.LabelResult{}, Structures: map[string][]map[string]any{}, Relations: map[string][]gliner.RelationMatch{},
	}}
	app, err := New(DefaultConfig(), engine)
	if err != nil {
		t.Fatal(err)
	}
	body := []byte(`{"model":"fastino/gliner2-base-v1","messages":[{"role":"user","content":"Ada"}],"schema":{"entities":["person"]}}`)
	request := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewReader(body))
	request.Header.Set("Content-Type", "application/json")
	response, err := app.Test(request)
	if err != nil {
		t.Fatal(err)
	}
	if response.StatusCode != 200 {
		t.Fatalf("status = %d", response.StatusCode)
	}
	var envelope map[string]any
	if err := json.NewDecoder(response.Body).Decode(&envelope); err != nil {
		t.Fatal(err)
	}
	choices := envelope["choices"].([]any)
	message := choices[0].(map[string]any)["message"].(map[string]any)
	var content map[string]any
	if err := json.Unmarshal([]byte(message["content"].(string)), &content); err != nil {
		t.Fatalf("content is not JSON: %v", err)
	}
	if content["entities"].([]any)[0].(map[string]any)["person"].([]any)[0].(map[string]any)["text"] != "Ada" {
		t.Fatalf("unexpected content: %#v", content)
	}
	if envelope["x_fastino"].(map[string]any)["inference_id"] != nil {
		t.Fatal("inference_id should be null")
	}
}

func TestAuthenticationAndValidation(t *testing.T) {
	config := DefaultConfig()
	config.APIKey = "secret"
	app, err := New(config, &fakeEngine{result: gliner.EmptyExtractionResult(gliner.Schema{})})
	if err != nil {
		t.Fatal(err)
	}
	body := []byte(`{"model":"fastino/gliner2-base-v1","messages":[{"role":"user","content":"Ada"}],"schema":{"entities":["person"]}}`)

	unauthorized := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewReader(body))
	response, _ := app.Test(unauthorized)
	if response.StatusCode != 401 {
		t.Fatalf("unauthorized status = %d", response.StatusCode)
	}

	authorized := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewReader(body))
	authorized.Header.Set("Authorization", "Bearer secret")
	response, _ = app.Test(authorized)
	if response.StatusCode != 200 {
		t.Fatalf("authorized status = %d", response.StatusCode)
	}

	streamBody := []byte(`{"model":"fastino/gliner2-base-v1","messages":[{"role":"user","content":"Ada"}],"schema":{"entities":["person"]},"stream":true}`)
	stream := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewReader(streamBody))
	stream.Header.Set("X-API-Key", "secret")
	response, _ = app.Test(stream)
	if response.StatusCode != 400 {
		t.Fatalf("stream status = %d", response.StatusCode)
	}
}

func TestReadinessAndDeprecatedSchema(t *testing.T) {
	app, err := New(DefaultConfig(), &fakeEngine{ready: errors.New("missing model")})
	if err != nil {
		t.Fatal(err)
	}
	response, _ := app.Test(httptest.NewRequest("GET", "/readyz", nil))
	if response.StatusCode != 503 {
		t.Fatalf("ready status = %d", response.StatusCode)
	}

	app, _ = New(DefaultConfig(), &fakeEngine{result: gliner.EmptyExtractionResult(gliner.Schema{})})
	body := []byte(`{"model":"fastino/gliner2-base-v1","messages":[{"role":"user","content":"Ada"}],"schema":["person"]}`)
	response, _ = app.Test(httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewReader(body)))
	if response.StatusCode != 200 || response.Header.Get("Deprecation") != "true" {
		t.Fatalf("legacy response = %d, deprecation=%q", response.StatusCode, response.Header.Get("Deprecation"))
	}
}
