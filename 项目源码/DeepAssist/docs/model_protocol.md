# Model and protocol boundary

DeepAssist's configured project model is **DeepSeek**. The optional
`OpenAICompatibleClient` is an infrastructure adapter for the
`/v1/chat/completions` wire contract; it is not a second model declaration.

“OpenAI-compatible” therefore cannot be inferred from a model name. A local
Qwen deployment is usable through this adapter only if its serving framework
implements the expected request and response fields (including streaming and,
when needed, tool calls). A raw local Qwen runtime without that HTTP contract
is not OpenAI-compatible and needs a separate adapter. This distinction keeps
the DeepAssist resume story and runtime configuration centered on DeepSeek.
