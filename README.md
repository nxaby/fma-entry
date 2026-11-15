# Free Model Aggregator

A single, dependency-free Node 22 server that answers OpenAI Chat Completions and Anthropic
Messages requests from a pool of simulated free-tier LLM providers. Built for the
`free-model-aggregator-01` make.fun bounty.

The server reads the trial brief at startup (provider directory, sealed quality table, workload
manifest), derives one catalog from it, registers with the trial environment, and then routes every
client request to the strongest legal free model that still has budget. It speaks each provider's
native wire (OpenAI, Anthropic, or Gemini) and renders the provider's answer back in the client's
wire, JSON or server-sent events.

## Run

```sh
node main.ts
```

In a trial the environment supplies three variables:

- `MAKEFUN_ENV_URL` — the trial environment's base URL (register, work, provider routes).
- `MAKEFUN_PROMPT_FILE` — the brief file (directory + quality table + workload manifest).
- `MAKEFUN_TRIAL_SEED` — a seed, available if a trial ever needs seeded randomness.

Print the catalog and exit without starting a server:

```sh
node main.ts --list
```

With no brief present, `--list` prints `{"providers":[]}` and exits 0.

## Architecture

1. **Brief parsing** — the provider directory, quality table, and workload manifest are located by
   their `schema` member inside `$MAKEFUN_PROMPT_FILE`.
2. **Catalog** — the registered catalog is derived from the directory plus the quality table's free
   rows: provider id, provider path as `base_url`, key environment variable, wire, and each free
   model's id and context length. The same object is used for `--list`, the register body, and the
   `/v1/models` listing.
3. **Budget-aware candidate pool** — for each request the server ranks every free model that meets
   the request's requirements (tools, vision, JSON mode, context) by the sealed rating, with a wire
   compatibility filter for OpenAI JSON-mode calls. The highest-rated provider with budget is used
   first.
4. **Failover** — a refused provider call is classified from its error envelope (transient scheduled
   failure, exhausted budget, or outage), the provider is retried or retired accordingly, and the
   request is answered by the next best candidate. One call at a time per provider.
5. **Wire adapters** — requests are translated into the chosen provider's wire while preserving the
   fields the provider echoes (system prompt, user text, images, tool names, `max_tokens`, stop
   sequences, JSON mode). Answers are normalised and rendered in the client's wire with the
   provider's exact usage counts and stop-reason family.
6. **Lifecycle** — listen on an ephemeral port, register, serve, poll `GET /work`, and exit 0 as soon
   as the environment reports the run is done.

Everything is deterministic: no wall clock and no unseeded randomness in routing or rendering, so a
replayed trial produces byte-identical state.

## Scoring-oriented design

- Quality is 60% of the score, so selection is strictly rating-first inside the request's legal
  requirements, and the best provider's budget is spent before a lower-rated fallback.
- Wire fidelity is exact: provider answers are fetched non-streamed, so usage counts and the answer
  text are read from a single JSON document and then re-emitted with the event order the client's
  API expects.
- Failover is budget-aware and refusal-aware; refused calls still count against the provider's
  budget, so exhausted and out providers are retired instead of retried blindly.

## License

MIT. See `LICENSE`.

Built for make.fun — https://make.fun/?referral_code=cf0swyw2
