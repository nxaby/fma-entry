// Wire translation and rendering.
//
// Every client request is first parsed into one canonical request. The canonical request is then
// emitted in whichever provider wire was picked (OpenAI, Anthropic or Gemini), always non-streamed:
// a provider answer is a single JSON document, so the answer text and the usage counts come from one
// place and cannot disagree. The client's own wire is rendered last, as JSON or as server-sent
// events in the event order that API expects.
//
// Fidelity matters: the simulated providers echo parts of the request they received (system text,
// the last user message's text, image bytes, tool names, max tokens, stop sequences, JSON mode) and
// the harness compares that echo with what the client originally asked. The canonical form keeps
// those fields exact while the emitters only reshape containers.

import { isRecord, type Json, type Wire } from './providers.ts';

export type ClientApi = 'openai' | 'anthropic';

export interface CanonicalImage {
  mime_type: string;
  base64: string | null;
  url: string | null;
}

export interface CanonicalToolCall {
  id: string;
  name: string;
  arguments: string;
}

export interface CanonicalMessage {
  role: 'user' | 'assistant' | 'tool';
  text: string;
  images: CanonicalImage[];
  tool_calls: CanonicalToolCall[];
  tool_call_id: string | null;
}

export interface CanonicalTool {
  name: string;
  description: string;
  parameters: Json;
}

export type ToolChoice = { mode: 'auto' | 'required' | 'none' | 'tool'; name: string | null } | null;

export interface CanonicalRequest {
  system_text: string;
  messages: CanonicalMessage[];
  tools: CanonicalTool[];
  max_tokens: number | null;
  stops: string[];
  temperature: number | null;
  top_p: number | null;
  json_mode: boolean;
  response_format: Json | null;
  tool_choice: ToolChoice;
}

export interface NormalizedAnswer {
  text: string;
  input_tokens: number;
  output_tokens: number;
  provider_stop: string;
}

function stringOf(v: unknown): string {
  return typeof v === 'string' ? v : '';
}

function numberOrNull(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

function stringList(v: unknown): string[] {
  if (typeof v === 'string') return v === '' ? [] : [v];
  if (!Array.isArray(v)) return [];
  return v.filter((item): item is string => typeof item === 'string' && item !== '');
}

function blockArray(v: unknown): Json[] {
  return Array.isArray(v) ? v.filter(isRecord) : [];
}

/** Text blocks joined by `separator`; a plain string is itself. */
function textBlocks(content: unknown, separator: string): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const parts: string[] = [];
  for (const part of content) {
    if (isRecord(part) && part.type === 'text' && typeof part.text === 'string') parts.push(part.text);
  }
  return parts.join(separator);
}

/** Text of a provider answer: text parts joined with nothing, so the text is exactly what was sent. */
function answerText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const parts: string[] = [];
  for (const part of content) {
    if (isRecord(part) && typeof part.text === 'string') parts.push(part.text);
  }
  return parts.join('');
}

function dataUrlImage(url: string): CanonicalImage | null {
  if (!url.startsWith('data:')) return { mime_type: '', base64: null, url };
  const comma = url.indexOf(',');
  if (comma === -1) return null;
  const header = url.slice(5, comma);
  const pieces = header.split(';');
  const mime = pieces[0] === '' ? 'application/octet-stream' : pieces[0];
  const base64 = pieces.some((piece) => piece.toLowerCase() === 'base64');
  return { mime_type: mime, base64: base64 ? url.slice(comma + 1) : null, url: base64 ? null : url };
}

function openAIImages(content: unknown): CanonicalImage[] {
  const out: CanonicalImage[] = [];
  for (const part of blockArray(content)) {
    if (part.type !== 'image_url') continue;
    const holder = part.image_url;
    const url = typeof holder === 'string' ? holder : isRecord(holder) ? stringOf(holder.url) : '';
    if (url === '') continue;
    const image = dataUrlImage(url);
    if (image !== null) out.push(image);
  }
  return out;
}

function anthropicImages(content: unknown): CanonicalImage[] {
  const out: CanonicalImage[] = [];
  for (const block of blockArray(content)) {
    if (block.type !== 'image' || !isRecord(block.source)) continue;
    const source = block.source;
    if (source.type === 'base64') {
      out.push({ mime_type: stringOf(source.media_type), base64: stringOf(source.data), url: null });
    } else if (source.type === 'url') {
      out.push({ mime_type: '', base64: null, url: stringOf(source.url) });
    }
  }
  return out;
}

function openAIToolChoice(value: unknown): ToolChoice {
  if (typeof value === 'string') {
    if (value === 'auto' || value === 'required' || value === 'none') return { mode: value, name: null };
    return null;
  }
  if (isRecord(value) && value.type === 'function' && isRecord(value.function) && stringOf(value.function.name) !== '') {
    return { mode: 'tool', name: stringOf(value.function.name) };
  }
  return null;
}

function anthropicToolChoice(value: unknown): ToolChoice {
  if (!isRecord(value)) return null;
  if (value.type === 'auto') return { mode: 'auto', name: null };
  if (value.type === 'any') return { mode: 'required', name: null };
  if (value.type === 'none') return { mode: 'none', name: null };
  if (value.type === 'tool' && stringOf(value.name) !== '') return { mode: 'tool', name: stringOf(value.name) };
  return null;
}

function openAITools(body: Json): CanonicalTool[] {
  return blockArray(body.tools).flatMap((tool) => {
    const fn = isRecord(tool.function) ? tool.function : null;
    if (fn === null || stringOf(fn.name) === '') return [];
    return [
      {
        name: stringOf(fn.name),
        description: stringOf(fn.description),
        parameters: isRecord(fn.parameters) ? fn.parameters : { type: 'object', properties: {} },
      },
    ];
  });
}

function anthropicTools(body: Json): CanonicalTool[] {
  return blockArray(body.tools).flatMap((tool) => {
    if (stringOf(tool.name) === '') return [];
    return [
      {
        name: stringOf(tool.name),
        description: stringOf(tool.description),
        parameters: isRecord(tool.input_schema) ? tool.input_schema : { type: 'object', properties: {} },
      },
    ];
  });
}

function openAIToolCalls(value: unknown): CanonicalToolCall[] {
  return blockArray(value).flatMap((call) => {
    const fn = isRecord(call.function) ? call.function : null;
    if (fn === null || stringOf(fn.name) === '') return [];
    return [{ id: stringOf(call.id), name: stringOf(fn.name), arguments: stringOf(fn.arguments) }];
  });
}

/** Parse either client wire into the canonical request. */
export function canonicalRequest(api: ClientApi, body: Json): CanonicalRequest {
  const systemParts: string[] = [];
  const messages: CanonicalMessage[] = [];
  const tools = api === 'openai' ? openAITools(body) : anthropicTools(body);
  const stopKey = api === 'openai' ? body.stop : body.stop_sequences;
  const maxTokens =
    api === 'openai'
      ? (numberOrNull(body.max_tokens) ?? numberOrNull(body.max_completion_tokens))
      : numberOrNull(body.max_tokens);
  let jsonMode = false;
  let responseFormat: Json | null = null;

  for (const message of blockArray(body.messages)) {
    const role = stringOf(message.role);
    if (role === 'system') {
      const text = textBlocks(message.content, '\n');
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (role === 'developer') continue;
    if (role === 'user') {
      const text = textBlocks(message.content, '\n');
      const images = api === 'openai' ? openAIImages(message.content) : anthropicImages(message.content);
      messages.push({ role: 'user', text, images, tool_calls: [], tool_call_id: null });
      if (api === 'anthropic') {
        for (const block of blockArray(message.content)) {
          if (block.type === 'tool_result') {
            messages.push({
              role: 'tool',
              text: typeof block.content === 'string' ? block.content : textBlocks(block.content, '\n'),
              images: [],
              tool_calls: [],
              tool_call_id: stringOf(block.tool_use_id),
            });
          }
        }
      }
      continue;
    }
    if (role === 'assistant') {
      const text = textBlocks(message.content, '\n');
      const calls =
        api === 'openai'
          ? openAIToolCalls(message.tool_calls)
          : blockArray(message.content).flatMap((block) => {
              if (block.type !== 'tool_use' || stringOf(block.name) === '') return [];
              return [{ id: stringOf(block.id), name: stringOf(block.name), arguments: JSON.stringify(isRecord(block.input) ? block.input : {}) }];
            });
      messages.push({ role: 'assistant', text, images: [], tool_calls: calls, tool_call_id: null });
      continue;
    }
    if (role === 'tool') {
      messages.push({
        role: 'tool',
        text: typeof message.content === 'string' ? message.content : textBlocks(message.content, '\n'),
        images: [],
        tool_calls: [],
        tool_call_id: stringOf(message.tool_call_id),
      });
    }
  }

  if (api === 'openai') {
    const systemText = systemParts.join('\n');
    const choice = openAIToolChoice(body.tool_choice);
    if (isRecord(body.response_format)) {
      const type = stringOf(body.response_format.type);
      if (type === 'json_object' || type === 'json_schema') {
        jsonMode = true;
        responseFormat = body.response_format;
      }
    }
    return {
      system_text: systemText,
      messages,
      tools,
      max_tokens: maxTokens,
      stops: stringList(stopKey),
      temperature: numberOrNull(body.temperature),
      top_p: numberOrNull(body.top_p),
      json_mode: jsonMode,
      response_format: responseFormat,
      tool_choice: choice,
    };
  }

  return {
    system_text: textBlocks(body.system, '\n'),
    messages,
    tools,
    max_tokens: maxTokens,
    stops: stringList(stopKey),
    temperature: numberOrNull(body.temperature),
    top_p: numberOrNull(body.top_p),
    json_mode: false,
    response_format: null,
    tool_choice: anthropicToolChoice(body.tool_choice),
  };
}

function openAIUserContent(message: CanonicalMessage): unknown {
  if (message.images.length === 0) return message.text;
  const parts: Json[] = [];
  if (message.text !== '') parts.push({ type: 'text', text: message.text });
  for (const image of message.images) {
    const url =
      image.base64 !== null
        ? `data:${image.mime_type === '' ? 'application/octet-stream' : image.mime_type};base64,${image.base64}`
        : (image.url ?? '');
    if (url !== '') parts.push({ type: 'image_url', image_url: { url } });
  }
  return parts;
}

function anthropicUserContent(message: CanonicalMessage): unknown {
  if (message.images.length === 0) return message.text;
  const blocks: Json[] = [];
  if (message.text !== '') blocks.push({ type: 'text', text: message.text });
  for (const image of message.images) {
    if (image.base64 !== null) {
      blocks.push({
        type: 'image',
        source: { type: 'base64', media_type: image.mime_type === '' ? 'application/octet-stream' : image.mime_type, data: image.base64 },
      });
    } else if (image.url !== null && image.url !== '') {
      blocks.push({ type: 'image', source: { type: 'url', url: image.url } });
    }
  }
  return blocks;
}

function geminiParts(message: CanonicalMessage): Json[] {
  const parts: Json[] = [];
  if (message.text !== '' || message.images.length === 0) parts.push({ text: message.text });
  for (const image of message.images) {
    if (image.base64 !== null) {
      parts.push({
        inlineData: { mimeType: image.mime_type === '' ? 'application/octet-stream' : image.mime_type, data: image.base64 },
      });
    } else if (image.url !== null && image.url !== '') {
      parts.push({ fileData: { fileUri: image.url } });
    }
  }
  for (const call of message.tool_calls) {
    let args: Json = {};
    try {
      const parsed: unknown = JSON.parse(call.arguments === '' ? '{}' : call.arguments);
      if (isRecord(parsed)) args = parsed;
    } catch {
      // A malformed argument string is still forwarded as an empty argument object.
    }
    parts.push({ functionCall: { name: call.name, args } });
  }
  return parts;
}

function openAIToolChoiceOf(choice: ToolChoice): unknown {
  if (choice === null) return undefined;
  if (choice.mode === 'tool') return { type: 'function', function: { name: choice.name } };
  return choice.mode;
}

function anthropicToolChoiceOf(choice: ToolChoice): unknown {
  if (choice === null) return undefined;
  if (choice.mode === 'tool') return { type: 'tool', name: choice.name };
  if (choice.mode === 'required') return { type: 'any' };
  return { type: choice.mode === 'none' ? 'none' : 'auto' };
}

function geminiToolConfig(choice: ToolChoice): Json | undefined {
  if (choice === null) return undefined;
  const mode = choice.mode === 'required' || choice.mode === 'tool' ? 'ANY' : choice.mode === 'none' ? 'NONE' : 'AUTO';
  const config: Json = { functionCallingConfig: { mode } };
  if (choice.mode === 'tool' && choice.name !== null) {
    (config.functionCallingConfig as Json).allowedFunctionNames = [choice.name];
  }
  return { functionCallingConfig: config.functionCallingConfig };
}

/** Request body for the provider wire, with the chosen provider model. Never streams upstream. */
export function upstreamRequest(wire: Wire, canonical: CanonicalRequest, model: string): Json {
  if (wire === 'openai') {
    const messages: Json[] = [];
    if (canonical.system_text !== '') messages.push({ role: 'system', content: canonical.system_text });
    for (const message of canonical.messages) {
      if (message.role === 'user') {
        messages.push({ role: 'user', content: openAIUserContent(message) });
      } else if (message.role === 'assistant') {
        const out: Json = { role: 'assistant', content: message.text };
        if (message.tool_calls.length > 0) {
          out.tool_calls = message.tool_calls.map((call) => ({
            id: call.id,
            type: 'function',
            function: { name: call.name, arguments: call.arguments },
          }));
        }
        messages.push(out);
      } else {
        messages.push({ role: 'tool', tool_call_id: message.tool_call_id ?? '', content: message.text });
      }
    }
    const out: Json = { model, messages };
    if (canonical.max_tokens !== null) out.max_tokens = canonical.max_tokens;
    if (canonical.temperature !== null) out.temperature = canonical.temperature;
    if (canonical.top_p !== null) out.top_p = canonical.top_p;
    if (canonical.stops.length > 0) out.stop = canonical.stops;
    if (canonical.tools.length > 0) {
      out.tools = canonical.tools.map((tool) => ({
        type: 'function',
        function: { name: tool.name, description: tool.description, parameters: tool.parameters },
      }));
    }
    const choice = openAIToolChoiceOf(canonical.tool_choice);
    if (choice !== undefined) out.tool_choice = choice;
    if (canonical.response_format !== null) out.response_format = canonical.response_format;
    return out;
  }

  if (wire === 'anthropic') {
    const messages: Json[] = [];
    for (const message of canonical.messages) {
      if (message.role === 'user') {
        messages.push({ role: 'user', content: anthropicUserContent(message) });
      } else if (message.role === 'assistant') {
        const blocks: Json[] = [];
        if (message.text !== '') blocks.push({ type: 'text', text: message.text });
        for (const call of message.tool_calls) {
          let input: Json = {};
          try {
            const parsed: unknown = JSON.parse(call.arguments === '' ? '{}' : call.arguments);
            if (isRecord(parsed)) input = parsed;
          } catch {
            // Keep the empty input object.
          }
          blocks.push({ type: 'tool_use', id: call.id, name: call.name, input });
        }
        messages.push({ role: 'assistant', content: blocks });
      } else {
        messages.push({
          role: 'user',
          content: [{ type: 'tool_result', tool_use_id: message.tool_call_id ?? '', content: message.text }],
        });
      }
    }
    const out: Json = { model, messages };
    if (canonical.max_tokens !== null) out.max_tokens = canonical.max_tokens;
    if (canonical.system_text !== '') out.system = canonical.system_text;
    if (canonical.temperature !== null) out.temperature = canonical.temperature;
    if (canonical.top_p !== null) out.top_p = canonical.top_p;
    if (canonical.stops.length > 0) out.stop_sequences = canonical.stops;
    if (canonical.tools.length > 0) {
      out.tools = canonical.tools.map((tool) => ({ name: tool.name, description: tool.description, input_schema: tool.parameters }));
    }
    const choice = anthropicToolChoiceOf(canonical.tool_choice);
    if (choice !== undefined) out.tool_choice = choice;
    return out;
  }

  const contents: Json[] = [];
  for (const message of canonical.messages) {
    if (message.role === 'assistant') {
      contents.push({ role: 'model', parts: geminiParts(message) });
    } else if (message.role === 'tool') {
      contents.push({ role: 'user', parts: [{ text: message.text }] });
    } else {
      contents.push({ role: 'user', parts: geminiParts(message) });
    }
  }
  const generationConfig: Json = {};
  if (canonical.max_tokens !== null) generationConfig.maxOutputTokens = canonical.max_tokens;
  if (canonical.temperature !== null) generationConfig.temperature = canonical.temperature;
  if (canonical.top_p !== null) generationConfig.topP = canonical.top_p;
  if (canonical.stops.length > 0) generationConfig.stopSequences = canonical.stops;
  if (canonical.json_mode) generationConfig.responseMimeType = 'application/json';
  const out: Json = { contents };
  if (canonical.system_text !== '') out.systemInstruction = { parts: [{ text: canonical.system_text }] };
  if (Object.keys(generationConfig).length > 0) out.generationConfig = generationConfig;
  if (canonical.tools.length > 0) {
    out.tools = [{ functionDeclarations: canonical.tools.map((tool) => ({ name: tool.name, description: tool.description, parameters: tool.parameters })) }];
  }
  const toolConfig = geminiToolConfig(canonical.tool_choice);
  if (toolConfig !== undefined) out.toolConfig = toolConfig;
  return out;
}

/** Path of a Gemini generateContent call, the model escaped so it is one path segment. */
export function geminiPath(model: string): string {
  return `/v1beta/models/${encodeURIComponent(model)}:generateContent`;
}

/** Read a provider answer document, or null when it does not look like one. */
export function readAnswer(wire: Wire, body: Json): NormalizedAnswer | null {
  if (wire === 'openai') {
    const choice = Array.isArray(body.choices) && isRecord(body.choices[0]) ? body.choices[0] : null;
    if (choice === null || !isRecord(choice.message)) return null;
    const usage = isRecord(body.usage) ? body.usage : {};
    return {
      text: answerText(choice.message.content),
      input_tokens: numberOrNull(usage.prompt_tokens) ?? 0,
      output_tokens: numberOrNull(usage.completion_tokens) ?? 0,
      provider_stop: stringOf(choice.finish_reason),
    };
  }
  if (wire === 'anthropic') {
    if (!Array.isArray(body.content)) return null;
    const usage = isRecord(body.usage) ? body.usage : {};
    return {
      text: answerText(body.content),
      input_tokens: numberOrNull(usage.input_tokens) ?? 0,
      output_tokens: numberOrNull(usage.output_tokens) ?? 0,
      provider_stop: stringOf(body.stop_reason),
    };
  }
  const candidate = Array.isArray(body.candidates) && isRecord(body.candidates[0]) ? body.candidates[0] : null;
  if (candidate === null || !isRecord(candidate.content) || !Array.isArray(candidate.content.parts)) return null;
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return {
    text: answerText(candidate.content.parts),
    input_tokens: numberOrNull(usage.promptTokenCount) ?? 0,
    output_tokens: numberOrNull(usage.candidatesTokenCount) ?? 0,
    provider_stop: stringOf(candidate.finishReason),
  };
}

function stopFamily(providerStop: string): 0 | 1 | 2 | 3 {
  switch (providerStop) {
    case 'length':
    case 'max_tokens':
    case 'MAX_TOKENS':
      return 1;
    case 'tool_calls':
    case 'function_call':
    case 'tool_use':
      return 2;
    case 'content_filter':
    case 'refusal':
    case 'SAFETY':
    case 'RECITATION':
      return 3;
    default:
      return 0;
  }
}

function openAIStop(providerStop: string): string {
  const family = stopFamily(providerStop);
  return family === 1 ? 'length' : family === 2 ? 'tool_calls' : family === 3 ? 'content_filter' : 'stop';
}

function anthropicStop(providerStop: string): string {
  const family = stopFamily(providerStop);
  return family === 1 ? 'max_tokens' : family === 2 ? 'tool_use' : family === 3 ? 'refusal' : 'end_turn';
}

function splitText(text: string, size: number): string[] {
  if (text === '') return [];
  const pieces: string[] = [];
  for (let i = 0; i < text.length; i += size) pieces.push(text.slice(i, i + size));
  return pieces;
}

export function openAIJson(answer: NormalizedAnswer, model: string): Json {
  return {
    id: 'chatcmpl-aggregator',
    object: 'chat.completion',
    created: 0,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: answer.text }, finish_reason: openAIStop(answer.provider_stop) }],
    usage: {
      prompt_tokens: answer.input_tokens,
      completion_tokens: answer.output_tokens,
      total_tokens: answer.input_tokens + answer.output_tokens,
    },
  };
}

export function openAIStream(answer: NormalizedAnswer, model: string): string {
  const id = 'chatcmpl-aggregator';
  const base = { id, object: 'chat.completion.chunk', created: 0, model };
  const events: string[] = [];
  const push = (payload: unknown): void => {
    events.push(`data: ${JSON.stringify(payload)}\n\n`);
  };
  push({ ...base, choices: [{ index: 0, delta: { role: 'assistant' }, finish_reason: null }] });
  for (const piece of splitText(answer.text, 13)) {
    push({ ...base, choices: [{ index: 0, delta: { content: piece }, finish_reason: null }] });
  }
  push({ ...base, choices: [{ index: 0, delta: {}, finish_reason: openAIStop(answer.provider_stop) }] });
  push({
    ...base,
    choices: [],
    usage: {
      prompt_tokens: answer.input_tokens,
      completion_tokens: answer.output_tokens,
      total_tokens: answer.input_tokens + answer.output_tokens,
    },
  });
  events.push('data: [DONE]\n\n');
  return events.join('');
}

export function anthropicJson(answer: NormalizedAnswer, model: string): Json {
  return {
    id: 'msg_aggregator',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: answer.text }],
    stop_reason: anthropicStop(answer.provider_stop),
    stop_sequence: null,
    usage: { input_tokens: answer.input_tokens, output_tokens: answer.output_tokens },
  };
}

export function anthropicStream(answer: NormalizedAnswer, model: string): string {
  const id = 'msg_aggregator';
  const events: string[] = [];
  const push = (name: string, payload: unknown): void => {
    events.push(`event: ${name}\n`);
    events.push(`data: ${JSON.stringify(payload)}\n\n`);
  };
  push('message_start', {
    type: 'message_start',
    message: {
      id,
      type: 'message',
      role: 'assistant',
      model,
      content: [],
      stop_reason: null,
      stop_sequence: null,
      usage: { input_tokens: answer.input_tokens, output_tokens: 0 },
    },
  });
  push('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } });
  for (const piece of splitText(answer.text, 13)) {
    push('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: piece } });
  }
  push('content_block_stop', { type: 'content_block_stop', index: 0 });
  push('message_delta', {
    type: 'message_delta',
    delta: { stop_reason: anthropicStop(answer.provider_stop), stop_sequence: null },
    usage: { output_tokens: answer.output_tokens },
  });
  push('message_stop', { type: 'message_stop' });
  return events.join('');
}
