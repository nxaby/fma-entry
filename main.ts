// Free Model Aggregator — one dependency-free Node 22 server for OpenAI Chat Completions and
// Anthropic Messages traffic, answered from ten simulated free-tier providers.
//
// Startup: read the brief, derive one catalog, listen on an ephemeral port, register the catalog with
// the trial environment, then serve. Each client request is routed to the strongest free model that
// meets its requirements, using each provider's own wire, and rendered back in the client's wire.
// The run ends when the environment's GET /work reports done.
//
// Determinism: no wall clock and no unseeded randomness feed a decision or a response byte, so a
// replay observes exactly the same choices and the same output.

import { Buffer } from 'node:buffer';
import { createServer } from 'node:http';
import type { IncomingMessage, ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  isRecord,
  joinBase,
  meetsRequirements,
  readBrief,
  requirementsFromBody,
  servedModels,
  type Brief,
  type Json,
  type ProviderDirectoryEntry,
  type RatingRow,
  type Requirement,
  type WorkloadFile,
} from './lib/providers.ts';
import {
  anthropicJson,
  anthropicStream,
  canonicalRequest,
  geminiPath,
  openAIJson,
  openAIStream,
  readAnswer,
  upstreamRequest,
  type CanonicalRequest,
  type ClientApi,
  type NormalizedAnswer,
} from './lib/wire.ts';

const PROTOCOL_VERSION = 1;
const MAX_BODY_BYTES = 32 * 1024 * 1024;
const UPSTREAM_TIMEOUT_MS = 12000;
const REGISTER_TIMEOUT_MS = 10000;
const POLL_INTERVAL_MS = 100;
const POLL_TIMEOUT_MS = 5000;
const MAX_TRIES_PER_PROVIDER = 3;
const MAX_ATTEMPTS_PER_REQUEST = 48;

const envUrl = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const promptFile = process.env.MAKEFUN_PROMPT_FILE ?? '';

function log(message: string): void {
  process.stderr.write(`[aggregator] ${message}\n`);
}

const brief: Brief = readBrief(promptFile);
const catalog = buildCatalog(brief);
const workload: WorkloadFile | null = brief.workload;
let manifestIndex = 0;

if (brief.error !== null) log(`brief: ${brief.error}`);

if (process.argv.includes('--list')) {
  process.stdout.write(`${JSON.stringify(catalog)}\n`);
  process.exit(0);
}

// ---------------------------------------------------------------------------------------------
// Provider pool and routing
// ---------------------------------------------------------------------------------------------

interface PoolModel {
  id: string;
  rating: number;
  row: RatingRow;
}

interface PoolProvider {
  id: string;
  base: string;
  wire: ProviderDirectoryEntry['wire'];
  auth: ProviderDirectoryEntry['auth'];
  key: string;
  key_env: string;
  models: PoolModel[];
}

interface Candidate {
  provider_id: string;
  base: string;
  wire: ProviderDirectoryEntry['wire'];
  auth: ProviderDirectoryEntry['auth'];
  key: string;
  model_id: string;
  rating: number;
}

interface ProviderState {
  calls: number;
  dead: boolean;
}

type FailureKind = 'transient' | 'budget' | 'outage' | 'fatal';

interface Failure {
  provider_id: string;
  kind: FailureKind;
  status: number;
  note: string;
}

type CallResult = { ok: true; answer: NormalizedAnswer } | ({ ok: false } & Failure);

const byString = (a: string, b: string): number => (a < b ? -1 : a > b ? 1 : 0);

function buildPool(source: Brief): PoolProvider[] {
  const ratings = source.ratings?.ratings ?? [];
  return source.directory.providers.map((provider) => {
    const models = ratings
      .filter((row) => row.provider === provider.id && row.free)
      .sort((a, b) => b.rating_milli - a.rating_milli || byString(a.model, b.model))
      .map((row) => ({ id: row.model, rating: row.rating_milli, row }));
    return {
      id: provider.id,
      base: provider.base,
      wire: provider.wire,
      auth: provider.auth,
      key: provider.key,
      key_env: provider.key_env,
      models,
    };
  });
}

const pool = buildPool(brief);
const states = new Map<string, ProviderState>();
for (const provider of pool) states.set(provider.id, { calls: 0, dead: false });

/**
 * Every provider's best model that satisfies the request, best rating first.
 *
 * When the client asked for JSON mode over the OpenAI wire, Anthropic-wire providers are excluded:
 * their wire cannot express the mode, so the echo check would fail. Equal ratings are ordered by the
 * provider with the fewest calls so far, which spreads load and keeps later options available.
 */
function candidatesFor(requires: Requirement, openaiJson: boolean): Candidate[] {
  const out: Candidate[] = [];
  for (const provider of pool) {
    if (openaiJson && provider.wire === 'anthropic') continue;
    const model = provider.models.find((entry) => meetsRequirements(requires, entry.row));
    if (model === undefined) continue;
    out.push({
      provider_id: provider.id,
      base: provider.base,
      wire: provider.wire,
      auth: provider.auth,
      key: provider.key,
      model_id: model.id,
      rating: model.rating,
    });
  }
  out.sort((a, b) => {
    if (a.rating !== b.rating) return b.rating - a.rating;
    const callsA = states.get(a.provider_id)?.calls ?? 0;
    const callsB = states.get(b.provider_id)?.calls ?? 0;
    if (callsA !== callsB) return callsA - callsB;
    return byString(a.provider_id, b.provider_id) || byString(a.model_id, b.model_id);
  });
  return out;
}

/** Classify a provider's refusal so retrying it is a deliberate choice, not a reflex. */
function classifyFailure(status: number, text: string): FailureKind {
  const lower = text.toLowerCase();
  if (status === 401 || status === 403) return 'fatal';
  if (status === 400 && (lower.includes('invalid_model') || lower.includes('invalid model'))) return 'fatal';
  if (lower.includes('budget is spent')) return 'budget';
  if (lower.includes('invalid_model') || lower.includes('invalid model')) return 'fatal';
  if (lower.includes(' is out')) return 'outage';
  if (lower.includes('concurrent_requests')) return 'transient';
  return 'transient';
}

async function callProvider(candidate: Candidate, canonical: CanonicalRequest): Promise<CallResult> {
  const base = joinBase(envUrl, candidate.base);
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(candidate.auth, candidate.key) };
  let url: string;
  if (candidate.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
  } else if (candidate.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    headers['anthropic-version'] = '2023-06-01';
  } else {
    url = `${base}${geminiPath(candidate.model_id)}`;
  }
  const body = upstreamRequest(candidate.wire, canonical, candidate.model_id);
  try {
    const response = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS),
    });
    const text = await response.text();
    if (!response.ok) {
      return {
        ok: false,
        provider_id: candidate.provider_id,
        kind: classifyFailure(response.status, text),
        status: response.status,
        note: text.slice(0, 160).replace(/\s+/g, ' '),
      };
    }
    let parsed: unknown;
    try {
      parsed = JSON.parse(text);
    } catch {
      return { ok: false, provider_id: candidate.provider_id, kind: 'fatal', status: response.status, note: 'answer is not JSON' };
    }
    if (!isRecord(parsed)) {
      return { ok: false, provider_id: candidate.provider_id, kind: 'fatal', status: response.status, note: 'answer is not a JSON object' };
    }
    const answer = readAnswer(candidate.wire, parsed);
    if (answer === null) {
      return { ok: false, provider_id: candidate.provider_id, kind: 'fatal', status: response.status, note: 'answer has no text' };
    }
    return { ok: true, answer };
  } catch (err) {
    return {
      ok: false,
      provider_id: candidate.provider_id,
      kind: 'transient',
      status: 0,
      note: (err as Error).message,
    };
  }
}

/**
 * Pick the next call for this request.
 *
 * The pool is ordered best first. A transient refusal on the best provider is worth a retry (at most
 * three calls), but only while that provider is still the best option: a fresh provider of equal
 * rating is preferred, and a strictly better provider is always preferred. Exhausted-budget, outage
 * and fatal answers retire the provider for the rest of the run.
 */
function chooseCandidate(candidates: Candidate[], tries: Map<string, number>, last: Failure | null): Candidate | null {
  const available = candidates.filter((candidate) => {
    const state = states.get(candidate.provider_id);
    if (state === undefined || state.dead) return false;
    return (tries.get(candidate.provider_id) ?? 0) < MAX_TRIES_PER_PROVIDER;
  });
  if (available.length === 0) return null;
  const best = available[0];
  if (last === null) return best;
  if (last.kind === 'fatal') {
    return available.find((candidate) => candidate.provider_id !== last.provider_id) ?? null;
  }
  if (last.kind === 'transient' && best.provider_id === last.provider_id) {
    const freshEqual = available.find(
      (candidate) => candidate.provider_id !== last.provider_id && !tries.has(candidate.provider_id) && candidate.rating === best.rating,
    );
    return freshEqual ?? best;
  }
  return best;
}

interface Served {
  answer: NormalizedAnswer;
  candidate: Candidate;
}

async function serveRequest(canonical: CanonicalRequest, requires: Requirement): Promise<Served | null> {
  const candidates = candidatesFor(requires, canonical.json_mode);
  if (candidates.length === 0) return null;
  const tries = new Map<string, number>();
  let last: Failure | null = null;
  for (let step = 0; step < MAX_ATTEMPTS_PER_REQUEST; step++) {
    const candidate = chooseCandidate(candidates, tries, last);
    if (candidate === null) return null;
    tries.set(candidate.provider_id, (tries.get(candidate.provider_id) ?? 0) + 1);
    const state = states.get(candidate.provider_id);
    if (state !== undefined) state.calls += 1;
    const result = await callProvider(candidate, canonical);
    if (result.ok) return { answer: result.answer, candidate };
    log(`${candidate.provider_id}/${candidate.model_id} refused call (HTTP ${result.status}, ${result.kind})`);
    if (result.kind !== 'transient' && state !== undefined) state.dead = true;
    last = result;
  }
  return null;
}

// ---------------------------------------------------------------------------------------------
// HTTP surface
// ---------------------------------------------------------------------------------------------

function sendJson(response: ServerResponse, status: number, value: unknown): void {
  const body = JSON.stringify(value);
  response.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(body) });
  response.end(body);
}

function sendOpenAIError(response: ServerResponse, status: number, message: string): void {
  sendJson(response, status, { error: { message, type: 'aggregator_error', code: status } });
}

function sendAnthropicError(response: ServerResponse, status: number, message: string): void {
  sendJson(response, status, { type: 'error', error: { type: 'api_error', message } });
}

function readJsonBody(request: IncomingMessage): Promise<Json | null> {
  return new Promise((resolve) => {
    const chunks: Buffer[] = [];
    let size = 0;
    let settled = false;
    const finish = (value: Json | null): void => {
      if (settled) return;
      settled = true;
      resolve(value);
    };
    request.on('data', (chunk: Buffer) => {
      if (settled) return;
      size += chunk.length;
      if (size > MAX_BODY_BYTES) {
        finish(null);
        request.destroy();
        return;
      }
      chunks.push(chunk);
    });
    request.on('end', () => {
      try {
        const parsed: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
        finish(isRecord(parsed) ? parsed : null);
      } catch {
        finish(null);
      }
    });
    request.on('error', () => finish(null));
  });
}

/** Requests are served one at a time: provider calls stay sequential and the manifest stays in order. */
let queue: Promise<unknown> = Promise.resolve();
function enqueue<T>(task: () => Promise<T>): Promise<T> {
  const run = queue.then(task, task);
  queue = run.then(
    () => undefined,
    () => undefined,
  );
  return run;
}

function requirementFor(body: Json): Requirement {
  const entry = workload?.manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

function modelList(anthropicShape: boolean): unknown {
  const models = servedModels(catalog);
  if (!anthropicShape) {
    return {
      object: 'list',
      data: models.map((model) => ({ id: model.id, object: 'model', created: 0, owned_by: model.provider })),
    };
  }
  return {
    data: models.map((model) => ({
      type: 'model',
      id: model.id,
      display_name: model.display_name,
      created_at: '1970-01-01T00:00:00Z',
    })),
    has_more: false,
    first_id: models.length > 0 ? models[0].id : null,
    last_id: models.length > 0 ? models[models.length - 1].id : null,
  };
}

async function handle(request: IncomingMessage, response: ServerResponse): Promise<void> {
  const url = new URL(request.url ?? '/', 'http://aggregator.invalid');

  if (request.method === 'GET' && (url.pathname === '/v1/models' || url.pathname === '/models')) {
    sendJson(response, 200, modelList(request.headers['x-api-key'] !== undefined));
    return;
  }

  const isOpenAI = url.pathname === '/v1/chat/completions';
  const isAnthropic = url.pathname === '/v1/messages';
  if (request.method !== 'POST' || (!isOpenAI && !isAnthropic)) {
    sendJson(response, 404, { error: { message: 'not found' } });
    return;
  }

  const api: ClientApi = isAnthropic ? 'anthropic' : 'openai';
  const body = await readJsonBody(request);
  if (body === null) {
    if (api === 'anthropic') sendAnthropicError(response, 400, 'request body must be a JSON object');
    else sendOpenAIError(response, 400, 'request body must be a JSON object');
    return;
  }

  const canonical = canonicalRequest(api, body);
  const model = typeof body.model === 'string' && body.model !== '' ? body.model : 'auto';
  const wantsStream = body.stream === true;

  const served = await enqueue(async () => serveRequest(canonical, requirementFor(body)));
  if (served === null) {
    const message = 'no free model could answer this request';
    if (api === 'anthropic') sendAnthropicError(response, 503, message);
    else sendOpenAIError(response, 503, message);
    return;
  }

  if (api === 'openai') {
    if (wantsStream) {
      const events = openAIStream(served.answer, model);
      response.writeHead(200, {
        'content-type': 'text/event-stream; charset=utf-8',
        'cache-control': 'no-cache',
        connection: 'keep-alive',
      });
      response.end(events);
    } else {
      sendJson(response, 200, openAIJson(served.answer, model));
    }
    return;
  }

  if (wantsStream) {
    const events = anthropicStream(served.answer, model);
    response.writeHead(200, {
      'content-type': 'text/event-stream; charset=utf-8',
      'cache-control': 'no-cache',
      connection: 'keep-alive',
    });
    response.end(events);
  } else {
    sendJson(response, 200, anthropicJson(served.answer, model));
  }
}

// ---------------------------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------------------------

async function register(port: number): Promise<void> {
  if (envUrl === '') {
    log('MAKEFUN_ENV_URL is not set; serving without registering');
    return;
  }
  try {
    const response = await fetch(`${envUrl}/register`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog }),
      signal: AbortSignal.timeout(REGISTER_TIMEOUT_MS),
    });
    if (!response.ok) log(`register failed: HTTP ${response.status} ${(await response.text()).slice(0, 200)}`);
  } catch (err) {
    log(`register failed: ${(err as Error).message}`);
  }
}

function startPolling(server: ReturnType<typeof createServer>): void {
  if (envUrl === '') return;
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const response = await fetch(`${envUrl}/work`, { signal: AbortSignal.timeout(POLL_TIMEOUT_MS) });
        if (response.ok) {
          const parsed: unknown = JSON.parse(await response.text());
          if (isRecord(parsed) && parsed.done === true) {
            clearInterval(timer);
            log('environment reports the run is done; exiting');
            server.close(() => process.exit(0));
            setTimeout(() => process.exit(0), 1000);
          }
        }
      } catch {
        // A poll that cannot complete is retried on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

const server = createServer((request, response) => {
  void handle(request, response).catch((err: unknown) => {
    log(`request failed: ${(err as Error).message}`);
    if (!response.headersSent) sendJson(response, 500, { error: { message: 'internal error' } });
    else response.end();
  });
});

for (const signal of ['SIGTERM', 'SIGINT'] as const) {
  process.on(signal, () => {
    server.close(() => process.exit(0));
    setTimeout(() => process.exit(0), 500);
  });
}

server.listen(0, '127.0.0.1', () => {
  const address = server.address();
  const port = typeof address === 'object' && address !== null ? address.port : 0;
  log(`listening on 127.0.0.1:${port}`);
  void register(port);
  startPolling(server);
});
