<script lang="ts">
  import { fade } from "svelte/transition"
  import { onMount } from "svelte"
  import type {
    OllamaConnection,
    DockerModelRunnerConnection,
  } from "$lib/types"
  import FormElement from "$lib/utils/form_element.svelte"
  import FormContainer from "$lib/utils/form_container.svelte"
  import { KilnError, createKilnError } from "$lib/utils/error_handlers"
  import { client, base_url } from "$lib/api_client"
  import Warning from "$lib/ui/warning.svelte"
  import { available_tuning_models } from "$lib/stores/fine_tune_store"
  import { clear_available_models_cache } from "$lib/stores"
  import { get_provider_image } from "$lib/ui/provider_image"
  import posthog from "posthog-js"
  import { goto } from "$app/navigation"
  import { setCopilotConnected } from "$lib/stores/copilot_connection_store"

  export let onboarding = false
  export let highlight_finetune = false
  export let required_providers: string[] = []

  type Provider = {
    name: string
    id: string
    description: string
    featured: boolean
    hide_in_onboarding?: boolean
    pill_text?: string
    api_key_steps?: string[]
    api_key_warning?: string
    api_key_fields?: string[]
    optional_fields?: string[]
  }
  const providers: Provider[] = [
    {
      name: "OpenRouter.ai",
      id: "openrouter",
      description:
        "Proxies requests to OpenAI, Anthropic, and more. Works with almost any model.",
      featured: !highlight_finetune,
      api_key_steps: [
        "Go to https://openrouter.ai/settings/keys",
        "Create a new API Key",
        "Copy the new API Key, paste it below and click 'Connect'",
      ],
    },
    {
      name: "Kiln Pro",
      id: "kiln_copilot",
      description: "Optimize your evals and AI performance with Kiln Pro.",
      featured: !highlight_finetune,
      hide_in_onboarding: true,
    },
    {
      name: "OpenAI",
      id: "openai",
      description: "The OG home to GPT-4o and more. Supports fine-tuning.",
      featured: false,
      pill_text: highlight_finetune ? "Tuneable" : undefined,
      api_key_steps: [
        "Create an OpenAI Platform account at https://platform.openai.com/signup and add a payment method.",
        "Go to https://platform.openai.com/account/api-keys",
        "Click 'Create new secret key'",
        "Copy the new secret key, paste it below and click 'Connect'",
      ],
      api_key_warning:
        "Note: the OpenAI API requires a separate account from ChatGPT.",
    },
    {
      name: "Ollama",
      id: "ollama",
      description: "Run models locally. No API key required.",
      featured: false,
    },
    {
      name: "Docker Model Runner",
      id: "docker_model_runner",
      description: "Run models locally with Docker. No API key required.",
      featured: false,
    },
    {
      name: "Groq",
      id: "groq",
      description: "Exceptionally fast inference on custom hardware.",
      featured: false,
      api_key_steps: [
        "Go to https://console.groq.com/keys",
        "Create an API Key",
        "Copy the new key, paste it below and click 'Connect'",
      ],
    },
    {
      name: "Fireworks AI",
      id: "fireworks_ai",
      description: "Open models (Llama, Phi), plus the ability to fine-tune.",
      pill_text: highlight_finetune ? "Tuneable" : undefined,
      api_key_steps: [
        "Go to https://app.fireworks.ai/settings/users/api-keys",
        "Create a new API Key and paste it below",
        "Go to https://app.fireworks.ai/settings/account",
        "Copy the Account ID, paste it below, and click 'Connect'",
      ],
      featured: false,
      api_key_fields: ["API Key", "Account ID"],
    },
    {
      name: "Anthropic",
      id: "anthropic",
      description: "The home of Sonnet, Haiku, and Opus.",
      featured: false,
      api_key_steps: [
        "Go to https://console.anthropic.com/settings/keys",
        "Create a new API Key",
        "Copy the new API Key, paste it below and click 'Connect'",
      ],
      api_key_fields: ["API Key"],
    },
    {
      name: "Google Gemini API",
      id: "gemini_api",
      description:
        "Google's Gemini API (aka AI Studio). Not to be confused with Gemini Enterprise Agent Platform (formerly Vertex AI).",
      featured: false,
      api_key_steps: [
        "Go to https://aistudio.google.com/app/apikey",
        "Create a new API Key",
        "Copy the new API Key, paste it below and click 'Connect'",
      ],
      api_key_fields: ["API Key"],
    },
    {
      name: "Gemini Enterprise Agent Platform",
      id: "vertex",
      description:
        "Google's Gemini Enterprise Agent Platform (formerly Vertex AI). Not to be confused with Gemini API.",
      featured: false,
      pill_text: highlight_finetune ? "Tuneable" : undefined,
      api_key_steps: [
        "Create a Google Cloud account.",
        "Install the gcloud CLI, then run `gcloud auth application-default login` in the terminal. This will add Google Vertex credentials to your environment.",
        "Create a project in the console, enable Vertex AI for that project, and click 'Enable Recommended APIs' in the Vertex AI console.",
        "Add the project ID below. Be sure to use the project ID, not the project name.",
        "Add a Google Cloud location, example: 'global'. We suggest 'global' as it has more recent Gemini models.",
        "Click connect.",
      ],
      api_key_fields: ["Project ID", "Project Location"],
      api_key_warning:
        "With Vertex AI, you must deploy some models manually.\nSee our docs for details: https://docs.kiln.tech/docs/models-and-ai-providers#google-vertex-ai",
    },
    {
      name: "Azure OpenAI",
      id: "azure_openai",
      description: "Microsoft's Azure OpenAI API.",
      featured: false,
      api_key_steps: [
        "Open the Azure portal, and navigate to the Azure OpenAI resource you want to use. Create a new resource if you don't have one, we suggest 'East US2' for maximal model support.",
        "Open the Keys & Endpoint section. Find your API Key and Endpoint URL. The Endpoint URL will look like https://<your-resource-name>.openai.azure.com",
        "Copy the API Key and Endpoint URL, paste them below and click 'Connect'",
      ],
      api_key_fields: ["API Key", "Endpoint URL"],
      api_key_warning:
        "With Azure OpenAI, you must deploy each model manually.\nSee our docs for details: https://docs.kiln.tech/docs/models-and-ai-providers#azure-openai-api",
    },
    {
      name: "Hugging Face",
      id: "huggingface",
      description: "AI community hub, with many models.",
      featured: false,
      api_key_steps: [
        "Go to https://huggingface.co/settings/tokens",
        "Create a new Access Token",
        "Copy the new Access Token, paste it below and click 'Connect'",
      ],
      api_key_fields: ["API Key"],
    },
    {
      name: "Together.ai",
      id: "together_ai",
      description: "Inference service from Together.ai",
      featured: false,
      pill_text: highlight_finetune ? "Tuneable" : undefined,
      api_key_steps: [
        "Create a Together account.",
        "Create an API Key (or user key) here: https://api.together.ai/settings/api-keys",
        "Copy the API Key, paste it below and click 'Connect'",
      ],
      api_key_fields: ["API Key"],
    },
    {
      name: "Amazon Bedrock",
      id: "amazon_bedrock",
      description: "So your company has an AWS contract?",
      featured: false,
      api_key_steps: [
        "Go to https://us-west-2.console.aws.amazon.com/bedrock/home?region=us-west-2#/overview - be sure to select us-west-2, as it has the most models, and Kiln defaults to this region",
        "Request model access for supported models like Llama and Mistral",
        "Create an IAM Key using this guide https://docs.aws.amazon.com/IAM/latest/UserGuide/id_credentials_access-keys.html and be sure to select 'AmazonBedrockFullAccess' policy when creating the IAM user",
        "Get the access key ID and secret access key for the new user. Paste them below and click 'Connect'",
      ],
      api_key_warning:
        "Bedrock is quite difficult to setup.\nFor beginners we suggest other providers, like OpenRouter, as they are easier to set up and have more models.",
      api_key_fields: ["Access Key", "Secret Key"],
    },
    {
      name: "SiliconFlow (硅基流动)",
      id: "siliconflow_cn",
      description: "AI provider for users in China.",
      api_key_warning:
        "SiliconFlow.cn is a Chinese provider. It is not available to users outside of China.",
      featured: false,
      api_key_steps: [
        "Go to https://cloud.siliconflow.cn/account/ak",
        "Create a new API Key",
        "Copy the new API Key, paste it below and click 'Connect'",
      ],
      api_key_fields: ["API Key"],
    },
    {
      name: "Cerebras",
      id: "cerebras",
      description: "Exceptionally fast inference on custom hardware.",
      featured: false,
      api_key_steps: [
        "Go to https://cloud.cerebras.ai/platform",
        "Create a new API Key",
        "Copy the new API Key, paste it below and click 'Connect'",
      ],
      api_key_fields: ["API Key"],
    },
    {
      name: "Featherless AI",
      id: "featherless_ai",
      description: "Serverless inference for thousands of open models.",
      featured: false,
      api_key_steps: [
        "Go to https://featherless.ai/account/api-keys",
        "Create a new API Key",
        "Copy the new API Key, paste it below and click 'Connect'",
      ],
      api_key_fields: ["API Key"],
    },
    {
      name: "Weights & Biases",
      id: "wandb",
      hide_in_onboarding: true,
      description: "Track and visualize your experiments.",
      featured: false,
      api_key_steps: [
        "Create a Weights & Biases account at https://wandb.ai , or host your own instance.",
        "Enter your API key below. If you use hosted W&B, you can find it here: https://wandb.ai/settings#api ",
        "Optional: if you host your own instance, set the base URL below.",
        "Optional: enter a custom W&B entity. If you don't enter one, your default entity will be used. For example, if your workspace URL is https://wandb.ai/nico/project, then your entity is nico.",
        "Click 'Connect'",
      ],
      api_key_fields: [
        "API Key",
        "Custom Entity - Optional",
        "Base URL - Optional",
      ],
      optional_fields: ["Base URL - Optional", "Custom Entity - Optional"],
    },
    {
      name: "Custom API",
      id: "openai_compatible",
      description: "Connect any OpenAI compatible API.",
      featured: false,
    },
  ]

  type ProviderStatus = {
    connected: boolean
    error: string | null
    custom_description: string | null
    connecting: boolean
  }
  let status: { [key: string]: ProviderStatus } = {
    ollama: {
      connected: false,
      connecting: false,
      error: null,
      custom_description: null,
    },
    docker_model_runner: {
      connected: false,
      connecting: false,
      error: null,
      custom_description: null,
    },
    openai: {
      connected: false,
      connecting: false,
      error: null,
      custom_description: null,
    },
    openrouter: {
      connected: false,
      connecting: false,
      error: null,
      custom_description: null,
    },
    groq: {
      connected: false,
      connecting: false,
      error: null,
      custom_description: null,
    },
    amazon_bedrock: {
      connected: false,
      connecting: false,
      error: null,
      custom_description: null,
    },
    fireworks_ai: {
      connected: false,
      connecting: false,
      error: null,
      custom_description: null,
    },
    anthropic: {
      connected: false,
      connecting: false,
      error: null,
      custom_description: null,
    },
    vertex: {
      connected: false,
      connecting: false,
      error: null,
      custom_description: null,
    },
    gemini_api: {
      connected: false,
      connecting: false,
      error: null,
      custom_description: null,
    },
    huggingface: {
      connected: false,
      connecting: false,
      error: null,
      custom_description: null,
    },
    azure_openai: {
      connected: false,
      connecting: false,
      error: null,
      custom_description: null,
    },
    together_ai: {
      connected: false,
      connecting: false,
      error: null,
      custom_description: null,
    },
    openai_compatible: {
      connected: false,
      connecting: false,
      error: null,
      custom_description: null,
    },
    wandb: {
      connected: false,
      connecting: false,
      error: null,
      custom_description: null,
    },
    siliconflow_cn: {
      connected: false,
      connecting: false,
      error: null,
      custom_description: null,
    },
    cerebras: {
      connected: false,
      connecting: false,
      error: null,
      custom_description: null,
    },
    featherless_ai: {
      connected: false,
      connecting: false,
      error: null,
      custom_description: null,
    },
    kiln_copilot: {
      connected: false,
      connecting: false,
      error: null,
      custom_description: null,
    },
  }

  export let has_connected_providers = false
  $: has_connected_providers = Object.values(status).some(
    (provider) => provider.connected,
  )
  export let intermediate_step = false
  export let centered = false
  let api_key_provider: Provider | null = null
  $: {
    intermediate_step = api_key_provider != null
  }

  const disconnect_provider = async (provider: Provider) => {
    if (provider.id === "ollama") {
      alert(
        "Ollama automatically connects to the localhost Ollama instance when it is running. It can't be manually disconnected. To change your preferred Ollama URL, turn off your Ollama instance, then return to this screen.",
      )
      return
    }
    if (
      !confirm(
        "Are you sure you want to disconnect this provider? Your connection details will be deleted and cannot be recovered.",
      )
    ) {
      return
    }
    try {
      const { error: disconnect_error } = await client.POST(
        "/api/provider/disconnect_api_key",
        {
          params: {
            query: {
              provider_id: provider.id,
            },
          },
        },
      )
      if (disconnect_error) {
        throw disconnect_error
      }

      posthog.capture("disconnect_provider", {
        provider_id: provider.id,
      })

      status[provider.id].connected = false

      if (provider.id === "kiln_copilot") {
        setCopilotConnected(false)
      }

      // Clear the available models list
      available_tuning_models.set(null)
      // Clear the available models cache so it refreshes next time
      clear_available_models_cache()
    } catch (e) {
      console.error("disconnect_provider error", e)
      alert("Failed to disconnect provider. Unknown error.")
      return
    }
  }

  const connect_provider = (provider: Provider) => {
    if (status[provider.id].connected) {
      return
    }
    if (provider.id === "ollama") {
      connect_ollama()
    }
    if (provider.id === "docker_model_runner") {
      connect_docker_model_runner()
    }
    if (provider.id === "openai_compatible") {
      show_custom_api_dialog()
    }
    if (provider.id === "kiln_copilot") {
      const isSettings = window.location.pathname.includes("/settings/")
      const route = isSettings
        ? "/settings/providers/kiln_pro"
        : "/setup/connect_providers/kiln_pro"
      goto(route)
      return
    }

    if (provider.api_key_steps) {
      api_key_provider = provider
    }
  }

  let custom_ollama_url: string | null = null

  const connect_ollama = async (user_initiated: boolean = true) => {
    status.ollama.connected = false
    status.ollama.connecting = user_initiated

    let data: OllamaConnection | null = null
    try {
      const { data: req_data, error: req_error } = await client.POST(
        "/api/provider/ollama/connect",
        {
          params: {
            query: {
              custom_ollama_url: custom_ollama_url || undefined,
            },
          },
        },
      )
      if (req_error) {
        throw req_error
      }
      data = req_data
    } catch (e) {
      if (
        e &&
        typeof e === "object" &&
        "message" in e &&
        typeof e.message === "string"
      ) {
        status.ollama.error = e.message
      } else {
        status.ollama.error = "Failed to connect. Ensure Ollama app is running."
      }
      status.ollama.connected = false
      return
    } finally {
      status.ollama.connecting = false
    }
    // Check min version number. We require 0.5+ for structured output
    if (data.version) {
      const version_parts = data.version.split(".")
      if (version_parts.length >= 2) {
        const major = parseInt(version_parts[0])
        const minor = parseInt(version_parts[1])
        if (major < 0 || (major == 0 && minor < 5)) {
          status.ollama.error =
            "Ollama version must be 0.5.0 or higher. Please update Ollama."
          status.ollama.connected = false
          return
        }
      }
    }
    if (
      data.supported_models.length === 0 &&
      (!data.untested_models || data.untested_models.length === 0)
    ) {
      status.ollama.error =
        "Ollama running, but no models available. Install some using ollama cli (e.g. 'ollama pull llama3.1')."
      return
    }
    status.ollama.error = null
    status.ollama.connected = true
    // Clear the available models cache so it refreshes next time
    clear_available_models_cache()
    const supported_models_str =
      data.supported_models.length > 0
        ? "The following supported models are available: " +
          data.supported_models.join(", ") +
          ". "
        : "No supported models are installed -- we suggest installing some (e.g. 'ollama pull llama3.1'). "
    const untested_models_str =
      data.untested_models && data.untested_models.length > 0
        ? "The following untested models are installed: " +
          data.untested_models.join(", ") +
          ". "
        : ""
    const custom_url_str =
      custom_ollama_url && custom_ollama_url == "http://localhost:11434"
        ? ""
        : "Custom Ollama URL: " + custom_ollama_url
    status.ollama.custom_description =
      "Ollama connected. " +
      supported_models_str +
      untested_models_str +
      custom_url_str
  }

  let docker_model_runner_custom_url: string | null = null

  const connect_docker_model_runner = async (
    user_initiated: boolean = true,
  ) => {
    status.docker_model_runner.connected = false
    status.docker_model_runner.connecting = user_initiated

    let data: DockerModelRunnerConnection | null = null
    try {
      const { data: req_data, error: req_error } = await client.POST(
        "/api/provider/docker_model_runner/connect",
        {
          params: {
            query: {
              docker_model_runner_custom_url:
                docker_model_runner_custom_url || undefined,
            },
          },
        },
      )
      if (req_error) {
        throw req_error
      }
      data = req_data
    } catch (e) {
      if (
        e &&
        typeof e === "object" &&
        "message" in e &&
        typeof e.message === "string"
      ) {
        status.docker_model_runner.error = e.message
      } else {
        status.docker_model_runner.error =
          "Failed to connect. Ensure Docker Model Runner is running."
      }
      status.docker_model_runner.connected = false
      return
    } finally {
      status.docker_model_runner.connecting = false
    }

    if (
      data.supported_models.length === 0 &&
      (!data.untested_models || data.untested_models.length === 0)
    ) {
      status.docker_model_runner.error =
        "Docker Model Runner running, but no models available. Pull some models first: https://hub.docker.com/u/ai"
      return
    }
    status.docker_model_runner.error = null
    status.docker_model_runner.connected = true
    // Clear the available models cache so it refreshes next time
    clear_available_models_cache()
    const supported_models_str =
      data.supported_models.length > 0
        ? "The following supported models are available: " +
          data.supported_models.join(", ") +
          ". "
        : "No supported models are loaded. "
    const untested_models_str =
      data.untested_models && data.untested_models.length > 0
        ? "The following untested models are loaded: " +
          data.untested_models.join(", ") +
          ". "
        : ""
    const custom_url_str =
      docker_model_runner_custom_url &&
      docker_model_runner_custom_url !==
        "http://localhost:12434/engines/llama.cpp"
        ? "Custom Docker Model Runner URL: " + docker_model_runner_custom_url
        : ""
    status.docker_model_runner.custom_description =
      "Docker Model Runner connected. " +
      supported_models_str +
      untested_models_str +
      custom_url_str
  }

  let api_key_issue = false
  let api_key_submitting = false
  let api_key_message: string | null = null
  const submit_api_key = async () => {
    const apiKeyFields = document.getElementById(
      "api-key-fields",
    ) as HTMLDivElement
    const inputs = apiKeyFields.querySelectorAll("input")
    const apiKeyData: Record<string, string> = {}
    for (const input of inputs) {
      apiKeyData[input.id] = input.value
      if (!input.value) {
        if (api_key_provider?.optional_fields?.includes(input.id)) {
          delete apiKeyData[input.id]
        } else {
          api_key_issue = true
          return
        }
      }
    }

    api_key_issue = false
    api_key_message = null
    api_key_submitting = true
    try {
      const provider_id = api_key_provider ? api_key_provider.id : ""
      let res = await fetch(base_url + "/api/provider/connect_api_key", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          provider: provider_id,
          key_data: apiKeyData,
        }),
      })
      let data = await res.json()

      if (res.status !== 200) {
        api_key_message =
          data.message || "Failed to connect to provider. Unknown error."
        return
      }

      posthog.capture("connect_provider", {
        provider_id: provider_id,
      })

      api_key_issue = false
      api_key_message = null
      status[provider_id].connected = true
      api_key_provider = null

      // Clear the available models list
      available_tuning_models.set(null)
      // Clear the available models cache so it refreshes next time
      clear_available_models_cache()
    } catch (e) {
      console.error("submit_api_key error", e)
      api_key_message = "Failed to connect to provider (Exception: " + e + ")"
      api_key_issue = true
      return
    } finally {
      api_key_submitting = false
    }
  }

  let loading_initial_providers = true
  let initial_load_failure = false
  type CustomOpenAICompatibleProvider = {
    name: string
    base_url: string
    api_key: string
  }
  let custom_openai_compatible_providers: CustomOpenAICompatibleProvider[] = []
  const check_existing_providers = async () => {
    try {
      let res = await fetch(base_url + "/api/settings")
      let data = await res.json()
      if (data["open_ai_api_key"]) {
        status.openai.connected = true
      }
      if (data["groq_api_key"]) {
        status.groq.connected = true
      }
      if (data["bedrock_access_key"] && data["bedrock_secret_key"]) {
        status.amazon_bedrock.connected = true
      }
      if (data["open_router_api_key"]) {
        status.openrouter.connected = true
      }
      if (data["fireworks_api_key"] && data["fireworks_account_id"]) {
        status.fireworks_ai.connected = true
      }
      if (data["vertex_project_id"] && data["vertex_location"]) {
        status.vertex.connected = true
      }
      if (data["ollama_base_url"]) {
        custom_ollama_url = data["ollama_base_url"]
      }
      if (data["docker_model_runner_base_url"]) {
        docker_model_runner_custom_url = data["docker_model_runner_base_url"]
      }
      if (data["anthropic_api_key"]) {
        status.anthropic.connected = true
      }
      if (data["gemini_api_key"]) {
        status.gemini_api.connected = true
      }
      if (data["azure_openai_api_key"] && data["azure_openai_endpoint"]) {
        status.azure_openai.connected = true
      }
      if (data["huggingface_api_key"]) {
        status.huggingface.connected = true
      }
      if (data["together_api_key"]) {
        status.together_ai.connected = true
      }
      if (data["siliconflow_cn_api_key"]) {
        status.siliconflow_cn.connected = true
      }
      if (data["wandb_api_key"]) {
        status.wandb.connected = true
      }
      if (data["cerebras_api_key"]) {
        status.cerebras.connected = true
      }
      if (data["featherless_ai_api_key"]) {
        status.featherless_ai.connected = true
      }
      if (data["kiln_copilot_api_key"]) {
        status.kiln_copilot.connected = true
      }
      if (
        data["openai_compatible_providers"] &&
        data["openai_compatible_providers"].length > 0
      ) {
        status.openai_compatible.connected = true
        custom_openai_compatible_providers = data["openai_compatible_providers"]
      }
    } catch (e) {
      console.error("check_existing_providers error", e)
      initial_load_failure = true
    } finally {
      loading_initial_providers = false
    }
  }

  onMount(async () => {
    await check_existing_providers()

    // Check Ollama every load, as it can be closed. More ephemeral (and local/cheap/fast)
    connect_ollama(false).then(() => {
      // Clear the error as the user didn't initiate this run
      status["ollama"].error = null
    })
    // Check Docker Model Runner every load, as it can be closed. More ephemeral (and local/cheap/fast)
    connect_docker_model_runner(false).then(() => {
      // Clear the error as the user didn't initiate this run
      status["docker_model_runner"].error = null
    })
  })

  function show_custom_ollama_url_dialog() {
    // @ts-expect-error showModal is not a method on HTMLElement
    document.getElementById("ollama_dialog")?.showModal()
  }

  function show_docker_model_runner_custom_url_dialog() {
    // @ts-expect-error showModal is not a method on HTMLElement
    document.getElementById("docker_model_runner_dialog")?.showModal()
  }

  function show_custom_api_dialog() {
    // @ts-expect-error showModal is not a method on HTMLElement
    document.getElementById("openai_compatible_dialog")?.showModal()
  }

  let new_provider_name = ""
  let new_provider_base_url = ""
  let new_provider_api_key = ""
  let adding_new_provider = false
  let new_provider_error: KilnError | null = null
  async function add_new_provider() {
    try {
      adding_new_provider = true
      if (!new_provider_base_url.startsWith("http")) {
        throw new Error("Base URL must start with http")
      }

      const { error: save_error } = await client.POST(
        "/api/provider/openai_compatible",
        {
          body: {
            name: new_provider_name,
            base_url: new_provider_base_url,
            api_key: new_provider_api_key,
          },
        },
      )
      if (save_error) {
        throw save_error
      }

      // Refresh to trigger the UI update
      custom_openai_compatible_providers = [
        ...custom_openai_compatible_providers,
        {
          name: new_provider_name,
          base_url: new_provider_base_url,
          api_key: new_provider_api_key,
        },
      ]

      // Reset the form
      new_provider_name = ""
      new_provider_base_url = ""
      new_provider_api_key = ""
      new_provider_error = null

      status.openai_compatible.connected = true
      // Clear the available models cache so it refreshes next time
      clear_available_models_cache()
      // @ts-expect-error daisyui does not add types
      document.getElementById("openai_compatible_dialog")?.close()
    } catch (e) {
      new_provider_error = createKilnError(e)
    } finally {
      adding_new_provider = false
    }
  }

  async function remove_openai_compatible_provider_at_index(index: number) {
    if (index < 0 || index >= custom_openai_compatible_providers.length) {
      return
    }
    try {
      let provider = custom_openai_compatible_providers[index]

      const { error: delete_error } = await client.DELETE(
        "/api/provider/openai_compatible",
        {
          params: {
            query: {
              name: provider.name,
            },
          },
        },
      )
      if (delete_error) {
        throw delete_error
      }

      // Update UI
      custom_openai_compatible_providers =
        custom_openai_compatible_providers.filter(
          (v, _) => v.name !== provider.name,
        )
      if (custom_openai_compatible_providers.length === 0) {
        status.openai_compatible.connected = false
      }
      // Clear the available models cache so it refreshes next time
      clear_available_models_cache()
    } catch (e) {
      alert("Failed to remove provider: " + e)
    }
  }
</script>

<div class="w-full {centered && 'flex flex-col items-center'}">
  {#if api_key_provider}
    <div class="grow h-full max-w-[400px] flex flex-col place-content-center">
      <div class="grow"></div>

      <h1 class="text-xl font-medium flex-none text-center">
        Connect {api_key_provider.name}
      </h1>

      {#if api_key_provider.api_key_warning}
        <div class="pt-4">
          <Warning
            warning_color="warning"
            warning_message={api_key_provider.api_key_warning}
            trusted={true}
          />
        </div>
      {/if}

      <ol class="flex-none my-2 text-gray-700">
        {#each api_key_provider.api_key_steps || [] as step}
          <li class="list-decimal pl-1 mx-8 my-4">
            <!-- eslint-disable-next-line svelte/no-at-html-tags -->
            {@html step.replace(
              /https?:\/\/\S+/g,
              '<a href="$&" class="link underline" target="_blank">$&</a>',
            )}
          </li>
        {/each}
      </ol>
      {#if api_key_message}
        <p class="text-error text-center pb-4">{api_key_message}</p>
      {/if}
      <div class="flex flex-row gap-4 items-center">
        <div class="grow flex flex-col gap-2" id="api-key-fields">
          {#each api_key_provider.api_key_fields || ["API Key"] as field}
            <input
              type="text"
              id={field}
              placeholder={field}
              class="input input-bordered w-full max-w-[300px] {api_key_issue
                ? 'input-error'
                : ''}"
            />
          {/each}
        </div>
        <button
          class="btn min-w-[130px]"
          on:click={submit_api_key}
          disabled={api_key_submitting}
        >
          {#if api_key_submitting}
            <div class="loading loading-spinner loading-md"></div>
          {:else}
            Connect
          {/if}
        </button>
      </div>
      <button
        class="link text-center text-sm mt-8"
        on:click={() => (api_key_provider = null)}
      >
        Cancel setting up {api_key_provider.name}
      </button>
      <div class="grow-[1.5]"></div>
    </div>
  {:else}
    <div
      class="w-full grid grid-cols-1 xl:grid-cols-2 gap-y-6 gap-x-24 max-w-lg xl:max-w-screen-xl"
    >
      {#each providers as provider}
        {#if !onboarding || !provider.hide_in_onboarding}
          {@const is_connected =
            status[provider.id] && status[provider.id].connected}
          <div class="flex flex-row gap-4 items-center">
            <img
              src={get_provider_image(provider.id)}
              alt={provider.name}
              class="flex-none p-1 size-10 mx-1"
            />
            <div class="flex flex-col grow pr-4">
              <h3
                class="text-base font-medium flex flex-row gap-2 items-center"
              >
                {provider.name}
                {#if required_providers.includes(provider.id)}
                  <div class="badge badge-sm badge-warning">Required</div>
                {:else if provider.featured}
                  <div class="badge badge-sm badge-secondary">Recommended</div>
                {:else if provider.pill_text}
                  <div class="badge badge-sm badge-primary">
                    {provider.pill_text}
                  </div>
                {/if}
              </h3>
              {#if status[provider.id] && status[provider.id].error}
                <p class="text-sm text-error" in:fade>
                  {status[provider.id].error}
                </p>
              {:else}
                <p class="text-sm text-gray-500">
                  {status[provider.id].custom_description ||
                    provider.description}
                </p>
              {/if}
              {#if provider.id === "ollama" && status[provider.id] && status[provider.id].error}
                <button
                  class="link text-left text-sm text-gray-500"
                  on:click={show_custom_ollama_url_dialog}
                >
                  Set Custom Ollama URL
                </button>
              {/if}
              {#if provider.id === "docker_model_runner" && status[provider.id] && status[provider.id].error}
                <button
                  class="link text-left text-sm text-gray-500"
                  on:click={show_docker_model_runner_custom_url_dialog}
                >
                  Set Docker Model Runner Custom URL
                </button>
              {/if}
            </div>

            {#if loading_initial_providers}
              <!-- Light loading state-->
              <div class="btn md:min-w-[100px] skeleton bg-base-200"></div>
              &nbsp;
            {:else if is_connected && provider.id === "openai_compatible"}
              <button
                class="btn md:min-w-[100px]"
                on:click={() => show_custom_api_dialog()}
              >
                Manage
              </button>
            {:else if is_connected}
              <button
                class="btn md:min-w-[100px] hover:btn-error group"
                on:click={() => disconnect_provider(provider)}
              >
                <img
                  src="/images/circle-check.svg"
                  class="size-6 group-hover:hidden"
                  alt="Connected"
                />
                <span class="text-xs hidden group-hover:inline">Disconnect</span
                >
              </button>
            {:else if status[provider.id].connecting}
              <div class="btn md:min-w-[100px]">
                <div class=" loading loading-spinner loading-md"></div>
              </div>
            {:else if initial_load_failure}
              <div>
                <div class="btn md:min-w-[100px] btn-error text-xs">Error</div>
                <div class="text-xs text-gray-500 text-center pt-1">
                  Reload page
                </div>
              </div>
            {:else}
              <button
                class="btn md:min-w-[100px]"
                on:click={() => connect_provider(provider)}
              >
                Connect
              </button>
            {/if}
          </div>
        {/if}
      {/each}
    </div>
  {/if}
</div>

<dialog id="ollama_dialog" class="modal">
  <div class="modal-box">
    <form method="dialog">
      <button
        class="btn btn-sm text-xl btn-circle btn-ghost absolute right-2 top-2 focus:outline-none"
        >✕</button
      >
    </form>

    <h3 class="text-lg font-bold">Custom Ollama URL</h3>
    <p class="text-sm font-light mb-8">
      By default, Kiln attempts to connect to Ollama running on localhost:11434.
      If you run Ollama on a custom URL or port, enter it here to connect.
    </p>
    <FormElement
      id="ollama_url"
      label="Ollama URL"
      info_description="It should include the http prefix and the port number. For example, http://localhost:11434"
      bind:value={custom_ollama_url}
      placeholder="http://localhost:11434"
    />
    <div class="flex flex-row gap-4 items-center mt-4 justify-end">
      <form method="dialog">
        <button class="btn">Cancel</button>
      </form>
      <button
        class="btn btn-primary"
        disabled={!custom_ollama_url}
        on:click={() => {
          connect_ollama(true)
          // @ts-expect-error showModal is not a method on HTMLElement
          document.getElementById("ollama_dialog")?.close()
        }}
      >
        Connect
      </button>
    </div>
  </div>
  <form method="dialog" class="modal-backdrop">
    <button>close</button>
  </form>
</dialog>

<dialog id="docker_model_runner_dialog" class="modal">
  <div class="modal-box">
    <form method="dialog">
      <button
        class="btn btn-sm text-xl btn-circle btn-ghost absolute right-2 top-2 focus:outline-none"
        >✕</button
      >
    </form>

    <h3 class="text-lg font-bold">Custom Docker Model Runner URL</h3>
    <p class="text-sm font-light mb-8">
      By default, Kiln attempts to connect to Docker Model Runner at
      http://localhost:12434/engines. If you run it on a different host or port,
      enter the full base URL here.
    </p>
    <FormElement
      id="docker_model_runner_url"
      label="Docker Model Runner URL"
      info_description="It should include the http prefix, host, port and engine path. For example, http://localhost:12434/engines/llama.cpp"
      bind:value={docker_model_runner_custom_url}
      placeholder="http://localhost:12434/engines"
    />
    <div class="flex flex-row gap-4 items-center mt-4 justify-end">
      <form method="dialog">
        <button class="btn">Cancel</button>
      </form>
      <button
        class="btn btn-primary"
        disabled={!docker_model_runner_custom_url}
        on:click={() => {
          connect_docker_model_runner(true)
          // @ts-expect-error showModal is not a method on HTMLElement
          document.getElementById("docker_model_runner_dialog")?.close()
        }}
      >
        Connect
      </button>
    </div>
  </div>
  <form method="dialog" class="modal-backdrop">
    <button>close</button>
  </form>
</dialog>

<dialog id="openai_compatible_dialog" class="modal">
  <div class="modal-box">
    <form method="dialog">
      <button
        class="btn btn-sm text-xl btn-circle btn-ghost absolute right-2 top-2 focus:outline-none"
        >✕</button
      >
    </form>

    <h3 class="text-lg font-bold flex flex-row gap-4">Connect Custom APIs</h3>
    <p class="text-sm font-light mb-8">
      Connect any OpenAI-compatible API by adding a base URL and API key.
    </p>
    {#if custom_openai_compatible_providers.length > 0}
      <div class="flex flex-col gap-2">
        <div class="font-medium">Existing APIs</div>
        {#each custom_openai_compatible_providers as provider, index}
          <div class="flex flex-row gap-3 card bg-base-200 px-4 items-center">
            <div class="text-sm">{provider.name}</div>
            <div class="text-sm text-gray-500 grow truncate">
              {provider.base_url}
            </div>
            <button
              class="btn btn-sm btn-ghost"
              on:click={() => remove_openai_compatible_provider_at_index(index)}
            >
              Remove
            </button>
          </div>
        {/each}
      </div>
    {/if}
    <div class="flex flex-col gap-2 mt-8">
      <div class="font-medium">Add New API</div>
      <FormContainer
        submit_label="Add"
        on:submit={add_new_provider}
        gap={2}
        submitting={adding_new_provider}
        error={new_provider_error}
      >
        <FormElement
          id="name"
          label="API Name"
          bind:value={new_provider_name}
          placeholder="My home server"
          info_description="A name for this endpoint for you to use. Example: 'My home server'"
        />
        <FormElement
          id="base_url"
          label="Base URL"
          bind:value={new_provider_base_url}
          placeholder="https://.../v1"
          info_description="The base URL of an OpenAI compatible API. For example, https://openrouter.ai/api/v1"
        />
        <FormElement
          id="api_key"
          label="API Key"
          optional={true}
          bind:value={new_provider_api_key}
          placeholder="sk-..."
          info_description="The API key for the OpenAI compatible API."
        />
      </FormContainer>
    </div>

    <form method="dialog" class="modal-backdrop">
      <button>close</button>
    </form>
  </div>
</dialog>
