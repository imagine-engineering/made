---
title: Use a Local Model - Zed
description: Configure llama.cpp, Ollama, LM Studio, local OpenAI-compatible servers, and local edit prediction in Zed.
---

# Use a Local Model

Use local models when you run the model on your machine or on infrastructure you control.

| Local path                        | Zed AI features      | External Agents | Terminal Threads | Notes                                              |
| --------------------------------- | -------------------- | --------------- | ---------------- | -------------------------------------------------- |
| llama.cpp                         | Yes                  | Separate config | Separate config  | Configure a llama.cpp server for Zed AI features   |
| LM Studio                         | Yes                  | Separate config | Separate config  | Configure LM Studio for Zed AI features            |
| Ollama                            | Yes                  | Separate config | Separate config  | Configure Ollama for Zed AI features               |
| Local OpenAI-compatible server    | Yes                  | Separate config | Separate config  | Configure base URL, model, and key if needed       |
| Local/self-hosted edit prediction | Edit Prediction only | No              | No               | Uses [Edit Prediction](./edit-prediction.md) setup |

## Copper managed local models

In Copper, choose **Local models** during native-agent setup, or open the llama.cpp provider in AI settings. Downloads start only when you choose **Download**. Choose **Load** to check runtime compatibility; choose **Set as default** to load and persist a local native-agent default. Existing provider choices are preserved until you explicitly change them. Inference failures never switch to a cloud provider.

The setup page distinguishes the selected row, the backend selected for the next load, the loaded model, the saved local default, and the native-agent default. **Ready** means the managed server passed health and model-loading checks. **Active** means that loaded model is also the native-agent default. A conversation may retain its own model selection. Native-agent qualification is reported separately: runtime readiness does not establish reliable tool use.

Capacity recommendations include full weights, the displayed context's FP16 KV cache, runtime overhead, and an additional system reserve. **Recommended**, **Limited headroom**, and **Insufficient capacity** are estimates with reasons, not measured compatibility. Downloaded models do not need download space again; a loaded model is already resident and is not charged against remaining RAM twice. Apple unified memory is one shared CPU/GPU pool. Accelerate speeds up CPU math using system memory; it is not a separate zero-capacity GPU. Open **Hardware details** or a model's **Model details** for the underlying device, artifact and qualification information.

The manually managed server path below remains available through **Use my existing server**.

### JetBrains Qwen3.8–3.6–27B Blend

Copper lists the publisher's Q4_K_M and lower-memory IQ3_S GGUFs as optional text models. Both use a 16,384-token context with an estimated 1.00 GiB FP16 attention KV cache. Q4_K_M requires a 15.66 GiB download and an estimated 20.57 GiB at runtime. IQ3_S requires an 11.74 GiB download and an estimated 15.67 GiB at runtime. Copper budgets a further 1 GiB of system memory for either model. The runtime estimate includes the weights and overhead for recurrent state and other runtime buffers. Native-agent qualification is pending. [JetBrains model card and attribution](https://huggingface.co/JetBrains/Qwen3.8-3.6-27B-blend-GGUF/blob/e5d856b7430523bc465cbd07c78ac4bc0913a8c3/README.md#license-and-attribution)

### Underdog Saluki 27B 1.0

Choose **Download** next to **Underdog Saluki 27B 1.0 (IQ2-mix)** to obtain the publisher's text GGUF. The download is 7.36 GiB. Copper estimates 10.19 GiB at runtime, including a 1.00 GiB FP16 attention KV cache at 16,384 tokens. Copper budgets a further 1 GiB of system memory. Native-agent qualification is pending.

The publisher offers the weights under CC BY-NC 4.0. Commercial use is subject to separate terms from ConwayResearch. The embedded GGUF metadata describes the Apache-licensed upstream model. [Publisher license and attribution](https://huggingface.co/ConwayResearch/Underdog-Saluki-27B-1.0/blob/442e28ec9d89d9e732ddea2f0b9e017db20730a8/README.md#credits-and-license)

### Unsloth Qwen3.8-27B

Copper offers Unsloth's UD-Q4_K_M and lower-memory UD-IQ3_S text GGUFs at 16,384 tokens. The estimates below use Copper's standard decoding mode. Each runtime estimate includes the weights, 1.00 GiB of FP16 attention KV cache, and runtime overhead.

| Variant | Download | Estimated runtime | Available memory required with 1 GiB reserve |
| --- | ---: | ---: | ---: |
| UD-Q4_K_M | 15.43 GiB | 20.28 GiB | 21.28 GiB |
| UD-IQ3_S | 11.21 GiB | 15.02 GiB | 16.02 GiB |

Both files contain an embedded Multi-Token Prediction (MTP) head. In **Local Models**, toggle **Experimental MTP · Live inference unverified** for the variant you want to try. The toggle applies on the next load or reload and starts off each session, including when Copper restores a saved default. The setup summary shows which decoding mode is running. If an MTP load fails, **Retry with standard decoding** turns off MTP for that model and retries. MTP uses the same GGUF download. [Unsloth model card and license](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/4af51b3542f775b0a8e4b310bfb3f17e01c306c9/README.md)

Copper adds a 1 GiB MTP allowance on top of the standard estimate and keeps the separate 1 GiB system reserve. That brings estimated available memory required to 22.28 GiB for UD-Q4_K_M and 17.02 GiB for UD-IQ3_S. Allocation-free Metal probes on a 32 GiB Apple M5 initialized both embedded draft contexts on b11669 and estimated roughly 606 MiB of additional buffers per model. The 1 GiB allowance rounds that estimate up for unmeasured runtime variation and counts the GGUF weights once. Full-weight inference, draft-token acceptance, and native-agent behavior are unverified.

#### Qualify standard decoding and MTP

1. On a machine that clears the estimate and disk check, download one variant through Copper. Record its catalog SHA-256, the bundled runtime's `--version` output (expected build b11669, revision `620d2225cc436ab0b001e3200e590e3b286ef9ab`), hardware, backend, and available memory. Use the same verified file and backend for both modes.
2. For the benchmark, start the bundled `llama-server` on a private loopback port with `--host 127.0.0.1 --port <port> --offline --jinja --parallel 1 --ctx-size 16384 --reasoning off --reasoning-budget 0 --slots --metrics --model <verified-gguf>`. Set a temporary `LLAMA_API_KEY` and use it as a bearer token in requests. For standard decoding add `--spec-type none`. For MTP add `--spec-type draft-mtp --spec-draft-n-max 2`. On the MTP run, confirm `GET /slots` shows `speculative: true` and `n_ctx: 16384` on slot 0. Compare `GET /props`'s `chat_template` SHA-256 with the catalog's `runtime_template_sha256`.
3. In each mode, send the same text prompts to `POST /v1/chat/completions` with `stream: true`, `cache_prompt: false`, and fixed sampling settings and output limits. For example, use temperature 0, seed 42, and a 256-token limit for a code explanation and a code generation prompt. Alternate modes across at least three runs, restarting the server for equivalent cache state. Measure time from request start to the first SSE event with generated text. Record generation tokens per second from the runtime timing summary and peak process memory (on macOS, wrap the server in `/usr/bin/time -l` and record peak memory footprint after shutdown). Keep measured memory distinct from the preflight estimate.
4. Record the before/after deltas from `GET /metrics` for `llamacpp:spec_decode_num_draft_tokens_total` and `llamacpp:spec_decode_num_accepted_tokens_total`. Record accepted divided by drafted for MTP and verify accepted is greater than zero. Report the median latency and generation speed for each mode.
5. In Copper, use **Load** or **Reload with selected decoding** and confirm the running mode. For each mode, start a fresh native-agent conversation in a disposable copy of the same project. Ask the agent to search for a known symbol, read its file, and fix a deliberately failing test. Deny one proposed tool call and verify it has no effect. Approve the edit and test run, then verify the diff and test output. Check that responses stream, cancel a separate long response after text appears, and confirm a new request succeeds. With external networking disconnected, restart Copper and repeat inference using the downloaded model. Re-enable Experimental MTP for the offline MTP run.

## llama.cpp {#llama-cpp}

Use [llama.cpp](https://llama.app) and its built-in server for local models with Zed Agent, Inline Assistant, and similar model-backed Zed AI features.

1. Install llama.cpp from [llama.app](https://llama.app).
2. Start the server in [router mode](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md):

   ```sh
   llama serve
   ```

   It loads models from the llama.cpp cache on demand. To download and run a specific model in one step, pass `-hf`:

   ```sh
   llama serve -hf unsloth/gemma-4-26B-A4B-it-GGUF:BF16
   ```

3. In Zed, select a llama.cpp model from the model dropdown.

Zed automatically discovers the served models with their context length and tool/vision capabilities. In router mode these are refined once a model loads, via the server's `/models/sse` stream (which requires a recent llama.cpp build). To list models yourself instead, set `auto_discover` to `false`:

```json [settings]
{
  "language_models": {
    "llama.cpp": {
      "api_url": "http://localhost:8080",
      "auto_discover": false,
      "available_models": [
        {
          "name": "gemma-4-12b-it-GGUF:BF16",
          "display_name": "gemma-4-12b-it-GGUF:BF16",
          "max_tokens": 32768,
          "supports_tools": true,
          "supports_images": false
        }
      ]
    }
  }
}
```

### llama.cpp Context Length {#llama-cpp-context}

Zed uses the context length the server reports (`/props`). Override it for all models with `context_window`, or per model with `max_tokens` in `available_models`:

```json [settings]
{
  "language_models": {
    "llama.cpp": {
      "context_window": 8192
    }
  }
}
```

If your llama.cpp server requires a key, enter it in the provider UI or set `LLAMACPP_API_KEY`. For a remote server, set the API URL to its endpoint and provide the key (set on the server with `--api-key`).

## Ollama {#ollama}

Use Ollama for local models with Zed Agent, Inline Assistant, and similar model-backed Zed AI features.

1. Download and install Ollama from [ollama.com/download](https://ollama.com/download).
2. Pull a model:

   ```sh
   ollama pull mistral
   ```

3. Make sure the Ollama server is running. On macOS, open Ollama.app. On Linux or from a shell, run:

   ```sh
   ollama serve
   ```

4. In Zed, select an Ollama model from the model dropdown.

Zed automatically discovers models that Ollama has pulled. To disable autodiscovery and list models yourself, configure `auto_discover`:

```json [settings]
{
  "language_models": {
    "ollama": {
      "api_url": "http://localhost:11434",
      "auto_discover": false,
      "available_models": [
        {
          "name": "qwen2.5-coder",
          "display_name": "qwen 2.5 coder",
          "max_tokens": 32768,
          "supports_tools": true,
          "supports_thinking": true,
          "supports_images": true
        }
      ]
    }
  }
}
```

### Ollama Context Length {#ollama-context}

Zed requests to Ollama include context length as the `num_ctx` parameter. By default, Zed uses `4096` tokens.

Set a context length for all Ollama models:

```json [settings]
{
  "language_models": {
    "ollama": {
      "context_window": 8192
    }
  }
}
```

You can also configure context length per model with `max_tokens` in `available_models`.

If your Ollama server requires a key, enter the key in the provider UI or set `OLLAMA_API_KEY`. For remote Ollama services such as Ollama Turbo, set the API URL to the remote endpoint and provide an API key.

## LM Studio {#lm-studio}

Use LM Studio for local models with Zed Agent, Inline Assistant, and similar model-backed Zed AI features.

1. Download and install [LM Studio](https://lmstudio.ai/download).
2. Download at least one model in LM Studio, or use the LM Studio CLI:

   ```sh
   lms get qwen2.5-coder-7b
   ```

3. Start the LM Studio API server:

   ```sh
   lms server start
   ```

4. In Zed, select an LM Studio model from the model dropdown.

If your LM Studio server requires a key, enter the key in the provider UI or set `LMSTUDIO_API_KEY`.

## Local OpenAI-Compatible Servers {#openai-compatible}

Use [OpenAI-compatible endpoints](./use-api-access.md#openai-compatible) for local or self-hosted servers that expose an OpenAI-compatible API.

## Local Edit Prediction {#edit-prediction}

Edit Prediction has its own provider setup. See [Edit Prediction](./edit-prediction.md) for local and self-hosted edit prediction options.

## Agent Path Boundaries {#agent-path-boundaries}

This page covers local models configured in Zed. External Agents and terminal CLIs may have their own local-model setup; configure those in the agent or CLI.
