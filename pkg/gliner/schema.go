package gliner

import (
	"bytes"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
)

const (
	// These bounds keep dynamic prompts comfortably within the exported
	// checkpoint's practical context window and make request cost predictable.
	MaxSchemaTasks  = 64
	MaxSchemaFields = 256
)

type SchemaError struct {
	Field   string
	Message string
}

func (e *SchemaError) Error() string { return e.Message }

// NormalizeSchema accepts the documented Fastino shapes and selected legacy
// inputs. The returned boolean indicates that deprecation headers are needed.
func NormalizeSchema(raw json.RawMessage) (Schema, bool, error) {
	if len(bytes.TrimSpace(raw)) == 0 || bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return Schema{}, false, schemaErr("schema", "schema is required")
	}
	var value any
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.UseNumber()
	if err := decoder.Decode(&value); err != nil {
		return Schema{}, false, schemaErr("schema", "schema must be valid JSON")
	}

	if flat, ok := value.([]any); ok {
		entities, err := parseEntities(flat)
		if err != nil {
			return Schema{}, false, err
		}
		return Schema{Entities: entities}, true, nil
	}

	object, ok := value.(map[string]any)
	if !ok {
		return Schema{}, false, schemaErr("schema", "schema must be an object")
	}
	var schema Schema
	var err error
	if section, exists := object["entities"]; exists {
		schema.Entities, err = parseEntities(section)
		if err != nil {
			return Schema{}, false, err
		}
	}
	if section, exists := object["classifications"]; exists {
		schema.Classifications, err = parseClassifications(section)
		if err != nil {
			return Schema{}, false, err
		}
	}
	if section, exists := object["structures"]; exists {
		schema.Structures, err = parseStructures(section)
		if err != nil {
			return Schema{}, false, err
		}
	}
	if section, exists := object["relations"]; exists {
		schema.Relations, err = parseRelations(section)
		if err != nil {
			return Schema{}, false, err
		}
	}
	if len(schema.Entities)+len(schema.Classifications)+len(schema.Structures)+len(schema.Relations) == 0 {
		return Schema{}, false, schemaErr("schema", "schema must contain entities, classifications, structures, or relations")
	}
	if err := validateSchemaCardinality(schema); err != nil {
		return Schema{}, false, err
	}
	return schema, false, nil
}

func validateSchemaCardinality(schema Schema) error {
	tasks := len(schema.Structures) + len(schema.Relations) + len(schema.Classifications)
	if len(schema.Entities) > 0 {
		tasks++
	}
	if tasks > MaxSchemaTasks {
		return schemaErr("schema", fmt.Sprintf("schema may contain at most %d tasks", MaxSchemaTasks))
	}
	fields := len(schema.Entities)
	for _, classification := range schema.Classifications {
		fields += len(classification.Labels)
	}
	for _, structure := range schema.Structures {
		fields += len(structure.Fields)
	}
	fields += 2 * len(schema.Relations)
	if fields > MaxSchemaFields {
		return schemaErr("schema", fmt.Sprintf("schema may contain at most %d fields and labels", MaxSchemaFields))
	}
	return nil
}

func parseEntities(value any) ([]EntitySpec, error) {
	var result []EntitySpec
	switch typed := value.(type) {
	case []any:
		for index, item := range typed {
			switch entity := item.(type) {
			case string:
				result = append(result, EntitySpec{Name: strings.TrimSpace(entity)})
			case map[string]any:
				name, _ := entity["name"].(string)
				description, _ := entity["description"].(string)
				result = append(result, EntitySpec{Name: strings.TrimSpace(name), Description: strings.TrimSpace(description)})
			default:
				return nil, schemaErr("schema.entities", fmt.Sprintf("entity %d must be a string or object", index))
			}
		}
	case map[string]any:
		for name, descriptionValue := range typed {
			description, ok := descriptionValue.(string)
			if !ok {
				return nil, schemaErr("schema.entities", "entity descriptions must be strings")
			}
			result = append(result, EntitySpec{Name: strings.TrimSpace(name), Description: strings.TrimSpace(description)})
		}
		sort.Slice(result, func(i, j int) bool { return result[i].Name < result[j].Name })
	default:
		return nil, schemaErr("schema.entities", "entities must be an array or object")
	}
	if err := validateNames("schema.entities", entityNames(result)); err != nil {
		return nil, err
	}
	return result, nil
}

func parseClassifications(value any) ([]ClassificationSpec, error) {
	var entries []map[string]any
	switch typed := value.(type) {
	case []any:
		for index, item := range typed {
			entry, ok := item.(map[string]any)
			if !ok {
				return nil, schemaErr("schema.classifications", fmt.Sprintf("classification %d must be an object", index))
			}
			entries = append(entries, entry)
		}
	case map[string]any:
		for task, config := range typed {
			if labels, ok := config.([]any); ok {
				entries = append(entries, map[string]any{"task": task, "labels": labels})
				continue
			}
			entry, ok := config.(map[string]any)
			if !ok {
				return nil, schemaErr("schema.classifications", "classification shorthand must contain labels")
			}
			copy := make(map[string]any, len(entry)+1)
			for key, item := range entry {
				copy[key] = item
			}
			copy["task"] = task
			entries = append(entries, copy)
		}
		sort.Slice(entries, func(i, j int) bool { return entries[i]["task"].(string) < entries[j]["task"].(string) })
	default:
		return nil, schemaErr("schema.classifications", "classifications must be an array or object")
	}

	result := make([]ClassificationSpec, 0, len(entries))
	for _, entry := range entries {
		task, _ := entry["task"].(string)
		labels, err := stringList(entry["labels"], "schema.classifications.labels")
		if err != nil {
			return nil, err
		}
		if len(labels) < 2 {
			return nil, schemaErr("schema.classifications.labels", "classification tasks require at least two labels")
		}
		item := ClassificationSpec{Task: strings.TrimSpace(task), Labels: labels, TopK: 1, Threshold: 0.5}
		if multi, ok := entry["multi_label"].(bool); ok {
			item.MultiLabel = multi
		}
		if topK, ok := numberAsInt(entry["top_k"]); ok {
			item.TopK = topK
		}
		if threshold, ok := numberAsFloat(entry["threshold"]); ok {
			item.Threshold = threshold
		}
		if item.TopK < 1 || item.TopK > len(item.Labels) {
			return nil, schemaErr("schema.classifications.top_k", "top_k must be between 1 and the number of labels")
		}
		if item.Threshold < 0 || item.Threshold > 1 {
			return nil, schemaErr("schema.classifications.threshold", "classification threshold must be between 0 and 1")
		}
		result = append(result, item)
	}
	if err := validateNames("schema.classifications", classificationNames(result)); err != nil {
		return nil, err
	}
	return result, nil
}

func parseStructures(value any) ([]StructureSpec, error) {
	objects, ok := value.(map[string]any)
	if !ok || len(objects) == 0 {
		return nil, schemaErr("schema.structures", "structures must be a non-empty object")
	}
	result := make([]StructureSpec, 0, len(objects))
	for name, config := range objects {
		fieldsValue := config
		if object, ok := config.(map[string]any); ok {
			fieldsValue = object["fields"]
		}
		items, ok := fieldsValue.([]any)
		if !ok || len(items) == 0 {
			return nil, schemaErr("schema.structures", "each structure must define at least one field")
		}
		structure := StructureSpec{Name: strings.TrimSpace(name)}
		for _, rawField := range items {
			field, err := parseField(rawField)
			if err != nil {
				return nil, err
			}
			structure.Fields = append(structure.Fields, field)
		}
		if err := validateNames("schema.structures.fields", fieldNames(structure.Fields)); err != nil {
			return nil, err
		}
		result = append(result, structure)
	}
	sort.Slice(result, func(i, j int) bool { return result[i].Name < result[j].Name })
	if err := validateNames("schema.structures", structureNames(result)); err != nil {
		return nil, err
	}
	return result, nil
}

func parseField(value any) (FieldSpec, error) {
	switch typed := value.(type) {
	case string:
		parts := strings.SplitN(typed, "::", 3)
		field := FieldSpec{Name: strings.TrimSpace(parts[0]), DType: "str"}
		if len(parts) > 1 {
			field.DType = strings.TrimSpace(parts[1])
		}
		if len(parts) > 2 {
			field.Description = strings.TrimSpace(parts[2])
		}
		if field.Name == "" {
			return FieldSpec{}, schemaErr("schema.structures.fields", "field names cannot be empty")
		}
		if field.DType != "str" && field.DType != "list" {
			return FieldSpec{}, schemaErr("schema.structures.fields", "field type must be str or list")
		}
		return field, nil
	case map[string]any:
		name, _ := typed["name"].(string)
		dtype, _ := typed["dtype"].(string)
		if dtype == "" {
			dtype = "str"
		}
		description, _ := typed["description"].(string)
		field := FieldSpec{Name: strings.TrimSpace(name), DType: dtype, Description: strings.TrimSpace(description)}
		if choices, exists := typed["choices"]; exists {
			parsed, err := stringList(choices, "schema.structures.fields.choices")
			if err != nil {
				return FieldSpec{}, err
			}
			field.Choices = parsed
		}
		if field.Name == "" || (field.DType != "str" && field.DType != "list") {
			return FieldSpec{}, schemaErr("schema.structures.fields", "field requires a name and str or list dtype")
		}
		return field, nil
	default:
		return FieldSpec{}, schemaErr("schema.structures.fields", "fields must be strings or objects")
	}
}

func parseRelations(value any) ([]RelationSpec, error) {
	var result []RelationSpec
	switch typed := value.(type) {
	case []any:
		names, err := stringList(typed, "schema.relations")
		if err != nil {
			return nil, err
		}
		for _, name := range names {
			result = append(result, RelationSpec{Name: name})
		}
	case map[string]any:
		for name, rawConfig := range typed {
			relation := RelationSpec{Name: strings.TrimSpace(name)}
			if rawConfig != nil {
				config, ok := rawConfig.(map[string]any)
				if !ok {
					return nil, schemaErr("schema.relations", "relation configuration must be an object")
				}
				relation.Description, _ = config["description"].(string)
				if threshold, exists := config["threshold"]; exists {
					value, ok := numberAsFloat(threshold)
					if !ok || value < 0 || value > 1 {
						return nil, schemaErr("schema.relations.threshold", "relation threshold must be between 0 and 1")
					}
					relation.Threshold = &value
				}
			}
			result = append(result, relation)
		}
		sort.Slice(result, func(i, j int) bool { return result[i].Name < result[j].Name })
	default:
		return nil, schemaErr("schema.relations", "relations must be an array or object")
	}
	if err := validateNames("schema.relations", relationNames(result)); err != nil {
		return nil, err
	}
	return result, nil
}

func stringList(value any, field string) ([]string, error) {
	items, ok := value.([]any)
	if !ok || len(items) == 0 {
		return nil, schemaErr(field, field+" must be a non-empty array")
	}
	result := make([]string, 0, len(items))
	for _, item := range items {
		text, ok := item.(string)
		if !ok {
			return nil, schemaErr(field, field+" must contain only strings")
		}
		result = append(result, strings.TrimSpace(text))
	}
	if err := validateNames(field, result); err != nil {
		return nil, err
	}
	return result, nil
}

func validateNames(field string, names []string) error {
	if len(names) == 0 {
		return schemaErr(field, field+" cannot be empty")
	}
	seen := make(map[string]struct{}, len(names))
	for _, name := range names {
		if strings.TrimSpace(name) == "" {
			return schemaErr(field, field+" names cannot be empty")
		}
		if _, exists := seen[name]; exists {
			return schemaErr(field, field+" names must be unique")
		}
		seen[name] = struct{}{}
	}
	return nil
}

func schemaErr(field, message string) error { return &SchemaError{Field: field, Message: message} }
func numberAsFloat(value any) (float64, bool) {
	switch n := value.(type) {
	case json.Number:
		v, e := n.Float64()
		return v, e == nil
	case float64:
		return n, true
	default:
		return 0, false
	}
}
func numberAsInt(value any) (int, bool) {
	v, ok := numberAsFloat(value)
	return int(v), ok && v == float64(int(v))
}
func entityNames(values []EntitySpec) []string {
	out := make([]string, len(values))
	for i, v := range values {
		out[i] = v.Name
	}
	return out
}
func classificationNames(values []ClassificationSpec) []string {
	out := make([]string, len(values))
	for i, v := range values {
		out[i] = v.Task
	}
	return out
}
func structureNames(values []StructureSpec) []string {
	out := make([]string, len(values))
	for i, v := range values {
		out[i] = v.Name
	}
	return out
}
func relationNames(values []RelationSpec) []string {
	out := make([]string, len(values))
	for i, v := range values {
		out[i] = v.Name
	}
	return out
}
func fieldNames(values []FieldSpec) []string {
	out := make([]string, len(values))
	for i, v := range values {
		out[i] = v.Name
	}
	return out
}
