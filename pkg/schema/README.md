# GLiNER2 Schema DSL (Draft)

This package is intended to provide a Fluent API to define extraction schemas directly in Go, bypassing the need for pre-calculated JSON prompt files.

## Future Vision

```go
s := schema.New().
    Entity("Person").
    Entity("Organization").
    Relation("works_at", "Person", "Organization").
    Relation("founded", "Person", "Organization")

pipeline.SetSchema(s)
```

## Roadmap
- [ ] Implement `DynamicPromptBuilder` to tokenize schema entities/relations on-the-fly.
- [ ] Implement `Schema` struct to manage label-to-id mappings.
- [ ] Add support for constrained extraction (e.g. only specific relations).
