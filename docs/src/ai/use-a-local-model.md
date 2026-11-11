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
