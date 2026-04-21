import Foundation

/// A typed tool declaration passed to `openConversation(tools:)`.
///
/// `parametersJSONSchema` must be a JSON Schema object serialized as a string
/// (e.g. `{"type":"object","properties":{...},"required":[...]}`). The string
/// is embedded directly into the tools payload so callers can keep hand-written
/// schemas verbatim without round-tripping through `JSONSerialization`.
public struct LiteRTLMTool: Sendable, Equatable {
    public let name: String
    public let description: String
    public let parametersJSONSchema: String

    public init(name: String, description: String, parametersJSONSchema: String) {
        self.name = name
        self.description = description
        self.parametersJSONSchema = parametersJSONSchema
    }
}

/// Result of a single typed conversation turn. Either a plain-text reply or one
/// or more tool calls the model wants the host to execute.
public enum LiteRTLMTurn: Sendable {
    case text(String)
    case toolCalls([LiteRTLMEngine.ParsedToolCall])
}

/// Streaming event emitted by `conversationSendTurnStreaming`.
///
/// `.thought` chunks arrive when thinking mode is enabled — surface them in a
/// separate UI channel so users see reasoning before the final answer.
public enum LiteRTLMStreamEvent: Sendable {
    case text(String)
    case thought(String)
    case toolCalls([LiteRTLMEngine.ParsedToolCall])
}

/// Serialize `[LiteRTLMTool]` into the OpenAI-shape array that the Gemma 4
/// chat template expects:
/// `[{"type":"function","function":{"name":..,"description":..,"parameters":..}}]`.
///
/// Throws `LiteRTLMError.invalidToolSchema` if any tool's `parametersJSONSchema`
/// isn't valid JSON — failing early here beats the library silently ignoring
/// a broken tool declaration at send-time.
public func buildToolsJSON(_ tools: [LiteRTLMTool]) throws -> String {
    var entries: [[String: Any]] = []
    entries.reserveCapacity(tools.count)

    for tool in tools {
        guard let schemaData = tool.parametersJSONSchema.data(using: .utf8) else {
            throw LiteRTLMError.invalidToolSchema(
                toolName: tool.name, detail: "schema is not valid UTF-8"
            )
        }
        let schemaObject: Any
        do {
            schemaObject = try JSONSerialization.jsonObject(with: schemaData)
        } catch {
            throw LiteRTLMError.invalidToolSchema(
                toolName: tool.name, detail: "parameters schema is not valid JSON: \(error.localizedDescription)"
            )
        }
        guard schemaObject is [String: Any] else {
            throw LiteRTLMError.invalidToolSchema(
                toolName: tool.name, detail: "parameters schema must be a JSON object"
            )
        }

        entries.append([
            "type": "function",
            "function": [
                "name": tool.name,
                "description": tool.description,
                "parameters": schemaObject,
            ],
        ])
    }

    let data = try JSONSerialization.data(withJSONObject: entries)
    guard let string = String(data: data, encoding: .utf8) else {
        throw LiteRTLMError.invalidToolSchema(
            toolName: "(all)", detail: "failed to encode tools array to UTF-8"
        )
    }
    return string
}
