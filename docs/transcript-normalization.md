# Transcript Normalization

The observer normalizes supported Claude Code and Codex JSONL records into the
existing harness models. It preserves the public event shape by publishing
`message` events with a serialized `Message`, and it uses the existing message
block union where possible:

- `TextBlock` for plain user or assistant text.
- `ThinkingBlock` for plaintext reasoning or thinking summaries.
- `ToolUseBlock` for transcript tool calls with a real transcript id.
- `ToolResultBlock` for transcript tool outputs with a real call id.
- `ImageBlock` for inline base64 image sources or data URLs.

No placeholder message is synthesized. If a transcript record has no meaningful
normalized block, the observer skips it quietly and logs only at debug level.

Sources consulted on 2026-05-11:

- Anthropic Claude tool-use docs, `https://docs.claude.com/en/docs/agents-and-tools/tool-use/overview`,
  for Claude content block names such as `text`, `image`, `tool_use`, and
  `tool_result`.
- Anthropic Claude tool-result docs, `https://platform.claude.com/docs/en/agents-and-tools/tool-use/handle-tool-calls`,
  for tool result content accepting strings or nested blocks.
- OpenAI Responses API reference, `https://platform.openai.com/docs/api-reference/responses`,
  for response items such as `message`, `function_call`,
  `function_call_output`, and reasoning output metadata.
- claude-devtools transcript documentation,
  `https://www.claude-dev.tools/docs/transcripts`, for observed Claude Code
  JSONL transcript conventions.

Implementation choices not directly stated by those sources are based on this
repository's existing models and observer semantics.

## Claude Code Coverage

Supported records:

- `type: "user"` or `type: "assistant"` with `message.content` as a string.
- Content lists containing strings or dictionaries with:
  - `type: "text"` and `text`.
  - `type: "thinking"` with `thinking` or `text`.
  - `type: "tool_use"` with `id`, `name`, and optional `input`.
  - `type: "tool_result"` with `tool_use_id`, optional `content`, and optional
    `is_error`.
  - `type: "image"` with `source.media_type` and `source.data`.

Known metadata record types are ignored quietly:

- `attachment`
- `last-prompt`
- `pr-link`
- `queue-operation`
- `system`

## Codex Coverage

Supported records:

- `response_item` payloads with `type: "message"` and role `user` or
  `assistant`.
- Message content strings, `input_text`, `output_text`, and plain text list
  entries.
- `response_item` payloads with `type: "reasoning"` when `summary`, `content`,
  or `text` contains plaintext.
- `function_call` and `custom_tool_call` payloads with `call_id` or `id`,
  `name`, and optional `arguments`, `input`, or `args`.
- `function_call_output` and `custom_tool_call_output` payloads with `call_id`
  or `id` and optional `output`.
- Inline image data URLs in message content.

Codex `event_msg` payloads with `type: "user_message"`, `type: "agent_message"`,
or `type: "assistant_message"` are treated as stream echoes and ignored. Current
Codex transcripts also emit canonical `response_item` message records for those
turns; normalizing both shapes would duplicate messages.

Known metadata payloads are ignored quietly:

- `agent_message`
- `assistant_message`
- `context_compacted`
- `task_complete`
- `task_started`
- `token_count`
- `user_message`
- `web_search_call`

Known record-level metadata ignored quietly:

- `compacted`
- `session_meta`

## Current Limits

The observer does not read arbitrary local files referenced by transcript image
items. Local image paths are skipped unless the transcript also contains inline
base64 data or a data URL.

Tool calls without a transcript-provided id are skipped. The harness model can
generate ids for new tool blocks, but transcript normalization avoids inventing
ids because that would make external history look more precise than it is.

Reasoning blocks are normalized only when plaintext is present. Encrypted,
redacted, token-count-only, or otherwise opaque reasoning metadata is ignored.

Tool call arguments are decoded from JSON strings when possible. Non-JSON
strings are retained under an `input` key so consumers can still see the raw
call payload without the observer guessing a schema.
