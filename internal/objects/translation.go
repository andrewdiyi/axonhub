package objects

type TranslationScope string

const (
	TranslationScopeSystem    TranslationScope = "system"
	TranslationScopeDeveloper TranslationScope = "developer"
	TranslationScopeUser      TranslationScope = "user"
	TranslationScopeAssistant TranslationScope = "assistant"
	TranslationScopeTool      TranslationScope = "tool"
)
