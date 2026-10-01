package server

import (
	_ "embed"

	"github.com/gofiber/fiber/v2"
)

//go:embed openapi.json
var openAPISpec []byte

const swaggerHTML = `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>GLiNER API documentation</title>
  <link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css">
</head>
<body>
  <div id="swagger-ui"></div>
  <script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
  <script>
    window.onload = () => SwaggerUIBundle({
      url: "/openapi.json",
      dom_id: "#swagger-ui",
      deepLinking: true,
      displayRequestDuration: true,
      persistAuthorization: true
    });
  </script>
</body>
</html>`

func registerDocumentationRoutes(app *fiber.App) {
	app.Get("/openapi.json", func(c *fiber.Ctx) error {
		c.Set(fiber.HeaderContentType, fiber.MIMEApplicationJSONCharsetUTF8)
		return c.Send(openAPISpec)
	})
	app.Get("/swagger", func(c *fiber.Ctx) error {
		c.Set(fiber.HeaderContentType, fiber.MIMETextHTMLCharsetUTF8)
		return c.SendString(swaggerHTML)
	})
}
