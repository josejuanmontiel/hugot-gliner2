package server

import (
	"context"
	"crypto/rand"
	"crypto/subtle"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/gofiber/fiber/v2"

	"hugot-gliner2/pkg/gliner"
)

const DefaultModelID = "fastino/gliner2-base-v1"

type Config struct {
	ModelID          string
	APIKey           string
	BodyLimit        int
	MaxTextBytes     int
	InferenceTimeout time.Duration
	QueueDepth       int
}

func DefaultConfig() Config {
	return Config{
		ModelID:          DefaultModelID,
		BodyLimit:        2 * 1024 * 1024,
		MaxTextBytes:     256 * 1024,
		InferenceTimeout: 300 * time.Second,
		QueueDepth:       16,
	}
}

type Server struct {
	config Config
	engine gliner.Engine
	queue  chan struct{}
}

type chatRequest struct {
	Model             string          `json:"model"`
	Messages          []chatMessage   `json:"messages"`
	Schema            json.RawMessage `json:"schema"`
	Threshold         *float64        `json:"threshold"`
	IncludeConfidence *bool           `json:"include_confidence"`
	IncludeSpans      *bool           `json:"include_spans"`
	Store             *bool           `json:"store"`
	Stream            bool            `json:"stream"`
	Task              *string         `json:"task"`
	TaskType          *string         `json:"task_type"`
}

type chatMessage struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"`
}

type errorBody struct {
	Error apiError `json:"error"`
}

type apiError struct {
	Message string  `json:"message"`
	Type    string  `json:"type"`
	Param   *string `json:"param"`
	Code    string  `json:"code"`
}

func New(config Config, engine gliner.Engine) (*fiber.App, error) {
	if engine == nil {
		return nil, errors.New("inference engine is required")
	}
	defaults := DefaultConfig()
	if config.ModelID == "" {
		config.ModelID = defaults.ModelID
	}
	if config.BodyLimit <= 0 {
		config.BodyLimit = defaults.BodyLimit
	}
	if config.MaxTextBytes <= 0 {
		config.MaxTextBytes = defaults.MaxTextBytes
	}
	if config.InferenceTimeout <= 0 {
		config.InferenceTimeout = defaults.InferenceTimeout
	}
	if config.QueueDepth <= 0 {
		config.QueueDepth = defaults.QueueDepth
	}

	s := &Server{config: config, engine: engine, queue: make(chan struct{}, config.QueueDepth)}
	app := fiber.New(fiber.Config{
		BodyLimit:             config.BodyLimit,
		DisableStartupMessage: true,
		ErrorHandler: func(c *fiber.Ctx, err error) error {
			var fiberErr *fiber.Error
			if errors.As(err, &fiberErr) {
				return writeError(c, fiberErr.Code, fiberErr.Message, "invalid_request_error", nil, "http_error")
			}
			return writeError(c, fiber.StatusInternalServerError, "internal server error", "server_error", nil, "internal_error")
		},
	})

	app.Get("/healthz", func(c *fiber.Ctx) error {
		return c.JSON(fiber.Map{"status": "ok"})
	})
	app.Get("/readyz", func(c *fiber.Ctx) error {
		if err := engine.Ready(); err != nil {
			return writeError(c, fiber.StatusServiceUnavailable, err.Error(), "server_error", nil, "model_not_ready")
		}
		response := fiber.Map{"status": "ready", "model": config.ModelID}
		if reporter, ok := engine.(interface{ ExecutionProvider() string }); ok {
			response["execution_provider"] = reporter.ExecutionProvider()
		}
		return c.JSON(response)
	})
	registerDocumentationRoutes(app)
	app.Post("/v1/chat/completions", s.authenticate, s.chatCompletions)
	return app, nil
}

func (s *Server) authenticate(c *fiber.Ctx) error {
	if s.config.APIKey == "" {
		return c.Next()
	}
	provided := c.Get("X-API-Key")
	if provided == "" {
		authorization := c.Get("Authorization")
		if len(authorization) > 7 && strings.EqualFold(authorization[:7], "Bearer ") {
			provided = strings.TrimSpace(authorization[7:])
		}
	}
	if len(provided) != len(s.config.APIKey) || subtle.ConstantTimeCompare([]byte(provided), []byte(s.config.APIKey)) != 1 {
		return writeError(c, fiber.StatusUnauthorized, "invalid or missing API key", "authentication_error", nil, "invalid_api_key")
	}
	return c.Next()
}

func (s *Server) chatCompletions(c *fiber.Ctx) error {
	var request chatRequest
	if err := json.Unmarshal(c.Body(), &request); err != nil {
		return writeError(c, fiber.StatusBadRequest, "request body must be valid JSON", "invalid_request_error", stringPtr("body"), "invalid_json")
	}
	if request.Model == "" {
		return writeError(c, fiber.StatusUnprocessableEntity, "model is required", "invalid_request_error", stringPtr("model"), "missing_field")
	}
	if request.Model != s.config.ModelID {
		return writeError(c, fiber.StatusNotFound, fmt.Sprintf("model %q is not available", request.Model), "invalid_request_error", stringPtr("model"), "model_not_found")
	}
	if request.Stream {
		return writeError(c, fiber.StatusBadRequest, "this deployment supports atomic non-streaming GLiNER inference only", "invalid_request_error", stringPtr("stream"), "streaming_unsupported")
	}

	text, err := extractUserText(request.Messages)
	if err != nil {
		return writeError(c, fiber.StatusUnprocessableEntity, err.Error(), "invalid_request_error", stringPtr("messages"), "invalid_messages")
	}
	if len(text) > s.config.MaxTextBytes {
		return writeError(c, fiber.StatusBadRequest, "input text exceeds the configured limit", "invalid_request_error", stringPtr("messages"), "text_too_large")
	}

	schema, schemaDeprecated, err := gliner.NormalizeSchema(request.Schema)
	if err != nil {
		param := "schema"
		var validation *gliner.SchemaError
		if errors.As(err, &validation) {
			param = validation.Field
		}
		return writeError(c, fiber.StatusBadRequest, err.Error(), "invalid_request_error", &param, "invalid_schema")
	}
	if schemaDeprecated || request.Task != nil || request.TaskType != nil {
		c.Set("Deprecation", "true")
		c.Set("Sunset", "Thu, 31 Dec 2026 23:59:59 GMT")
	}

	threshold := 0.5
	if request.Threshold != nil {
		threshold = *request.Threshold
	}
	if threshold < 0 || threshold > 1 {
		return writeError(c, fiber.StatusUnprocessableEntity, "threshold must be between 0 and 1", "invalid_request_error", stringPtr("threshold"), "invalid_value")
	}
	includeConfidence := true
	if request.IncludeConfidence != nil {
		includeConfidence = *request.IncludeConfidence
	}
	includeSpans := true
	if request.IncludeSpans != nil {
		includeSpans = *request.IncludeSpans
	}

	select {
	case s.queue <- struct{}{}:
		defer func() { <-s.queue }()
	default:
		c.Set("Retry-After", "1")
		return writeError(c, fiber.StatusTooManyRequests, "inference queue is full", "rate_limit_error", nil, "queue_full")
	}

	ctx, cancel := context.WithTimeout(context.Background(), s.config.InferenceTimeout)
	defer cancel()
	result, err := s.engine.Infer(ctx, text, schema, gliner.InferOptions{
		Threshold: threshold, IncludeConfidence: includeConfidence, IncludeSpans: includeSpans,
	})
	if err != nil {
		if errors.Is(err, context.DeadlineExceeded) || errors.Is(ctx.Err(), context.DeadlineExceeded) {
			return writeError(c, fiber.StatusServiceUnavailable, "inference timed out", "server_error", nil, "inference_timeout")
		}
		return writeError(c, fiber.StatusInternalServerError, "inference failed", "server_error", nil, "inference_failed")
	}
	content, err := result.MarshalContent(includeConfidence, includeSpans)
	if err != nil {
		return writeError(c, fiber.StatusInternalServerError, "failed to encode inference result", "server_error", nil, "encoding_failed")
	}

	return c.Status(fiber.StatusOK).JSON(fiber.Map{
		"id":      newCompletionID(),
		"object":  "chat.completion",
		"created": time.Now().Unix(),
		"model":   s.config.ModelID,
		"choices": []any{fiber.Map{
			"index":         0,
			"message":       fiber.Map{"role": "assistant", "content": string(content)},
			"finish_reason": "stop",
		}},
		"usage":     fiber.Map{"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
		"x_fastino": fiber.Map{"inference_id": nil},
	})
}

func extractUserText(messages []chatMessage) (string, error) {
	if len(messages) == 0 {
		return "", errors.New("messages must be a non-empty array")
	}
	for index := len(messages) - 1; index >= 0; index-- {
		if messages[index].Role != "user" {
			continue
		}
		var content string
		if err := json.Unmarshal(messages[index].Content, &content); err != nil {
			return "", errors.New("GLiNER user message content must be a string")
		}
		if strings.TrimSpace(content) != "" {
			return content, nil
		}
	}
	return "", errors.New("messages must contain a non-empty user message")
}

func writeError(c *fiber.Ctx, status int, message, kind string, param *string, code string) error {
	return c.Status(status).JSON(errorBody{Error: apiError{Message: message, Type: kind, Param: param, Code: code}})
}

func stringPtr(value string) *string { return &value }

func newCompletionID() string {
	random := make([]byte, 12)
	if _, err := rand.Read(random); err != nil {
		return fmt.Sprintf("chatcmpl_%d", time.Now().UnixNano())
	}
	return "chatcmpl_" + hex.EncodeToString(random)
}
