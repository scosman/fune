import type { components } from "./api_schema"

// A type for a button which can appear on an app page
export type ActionButton = {
  label?: string
  icon?: string
  handler?: () => void
  href?: string
  primary?: boolean
  notice?: boolean
  shortcut?: string
  disabled?: boolean
  loading?: boolean
}

// Project-Input is a variant with path
export type Project = components["schemas"]["Project-Input"]
export type Task = components["schemas"]["Task"]
export type TurnMode = components["schemas"]["TurnMode"]
export type TaskRun = components["schemas"]["TaskRun-Input"]
export type TaskRunOutput = components["schemas"]["TaskRun-Output"]
export type TaskRequirement = components["schemas"]["TaskRequirement"]
export type TaskOutputRating = components["schemas"]["TaskOutputRating-Output"]
export type TaskOutputRatingType = components["schemas"]["TaskOutputRatingType"]
export type RequirementRating = components["schemas"]["RequirementRating"]
export type RatingType = components["schemas"]["TaskOutputRatingType"]
export type AvailableModels = components["schemas"]["AvailableModels"]
export type ProviderModels = components["schemas"]["ProviderModels"]
export type ProviderModel = components["schemas"]["ProviderModel"]
export type ModelDetails = components["schemas"]["ModelDetails"]
export type DatasetSplit = components["schemas"]["DatasetSplit"]
export type Finetune = components["schemas"]["Finetune"]
export type FinetuneProvider = components["schemas"]["FinetuneProvider"]
export type FineTuneParameter = components["schemas"]["FineTuneParameter"]
export type FinetuneWithStatus = components["schemas"]["FinetuneWithStatus"]
export type OllamaConnection = components["schemas"]["OllamaConnection"]
export type DockerModelRunnerConnection =
  components["schemas"]["DockerModelRunnerConnection"]
export type RunSummary = components["schemas"]["RunSummary"]
export type PromptResponse = components["schemas"]["PromptResponse"]
export type ApiPrompt = components["schemas"]["ApiPrompt"]
export type ChatStrategy = components["schemas"]["ChatStrategy"]
export type EvalOutputScore = components["schemas"]["EvalOutputScore"]
export type EvalTemplateId = components["schemas"]["EvalTemplateId"]
export type Eval = components["schemas"]["Eval"]
export type EvalConfigType = components["schemas"]["EvalConfigType"]
export type EvalDataType = components["schemas"]["EvalDataType"]
export type EvalConfig = components["schemas"]["EvalConfig"]
export type TaskRunConfig = components["schemas"]["TaskRunConfig"]
export type DataGuide = components["schemas"]["DataGuide"]
export type KilnAgentRunConfigProperties =
  components["schemas"]["KilnAgentRunConfigProperties"]
export type McpRunConfigProperties =
  components["schemas"]["McpRunConfigProperties"]
export type RunConfigProperties =
  | KilnAgentRunConfigProperties
  | McpRunConfigProperties
export type JinjaInputTransform = components["schemas"]["JinjaInputTransform"]
export type InputTransform = NonNullable<
  KilnAgentRunConfigProperties["input_transform"]
>
export type EvalResultSummary = components["schemas"]["EvalResultSummary"]
export type EvalRunResult = components["schemas"]["EvalRunResult"]
export type EvalConfigCompareSummary =
  components["schemas"]["EvalConfigCompareSummary"]
export type EvalRun = components["schemas"]["EvalRun"]
export type EvalRunWithTrace = components["schemas"]["EvalRunWithTrace"]
export type EvalProgress = components["schemas"]["EvalProgress"]
export type RatingOption = components["schemas"]["RatingOption"]
export type RatingOptionResponse = components["schemas"]["RatingOptionResponse"]
export type FinetuneDatasetInfo = components["schemas"]["FinetuneDatasetInfo"]
export type FinetuneDatasetTagInfo =
  components["schemas"]["FinetuneDatasetTagInfo"]
export type StructuredOutputMode = components["schemas"]["StructuredOutputMode"]
export type KilnDocument = components["schemas"]["Document"]
export type KilnDocumentKind = components["schemas"]["Kind"]
export type ExtractorConfig = components["schemas"]["ExtractorConfig"]
export type ExtractionSummary = components["schemas"]["ExtractionSummary"]
export type ExtractorType = components["schemas"]["ExtractorType"]
export type OutputFormat = components["schemas"]["OutputFormat"]
export type ExtractionProgress = components["schemas"]["ExtractionProgress"]
export type EmbeddingConfig = components["schemas"]["EmbeddingConfig"]
export type EmbeddingProperties = components["schemas"]["EmbeddingProperties"]
export type EmbeddingModelDetails =
  components["schemas"]["EmbeddingModelDetails"]
export type EmbeddingProvider = components["schemas"]["EmbeddingProvider"]
export type EmbeddingModelName = components["schemas"]["EmbeddingModelName"]
export type ChunkerConfig = components["schemas"]["ChunkerConfig"]
export type ChunkerType = components["schemas"]["ChunkerType"]
export type CreateChunkerConfigRequest =
  components["schemas"]["CreateChunkerConfigRequest"]
export type ModelProviderName = components["schemas"]["ModelProviderName"]
export type RagProgress = components["schemas"]["RagProgress"]
export type RagConfigWithSubConfigs =
  components["schemas"]["RagConfigWithSubConfigs"]
export type VectorStoreConfig = components["schemas"]["VectorStoreConfig"]
export type VectorStoreType = components["schemas"]["VectorStoreType"]
export type RerankerConfig = components["schemas"]["RerankerConfig"]
export type RerankerType =
  components["schemas"]["RerankerConfig"]["properties"]["type"]
export type RerankerProvider = components["schemas"]["RerankerProvider"]
export type RerankerModelDetails = components["schemas"]["RerankerModelDetails"]
export type LogMessage = components["schemas"]["LogMessage"]
export type BulkCreateDocumentsResponse =
  components["schemas"]["BulkCreateDocumentsResponse"]
export type KilnToolServerDescription =
  components["schemas"]["KilnToolServerDescription"]
export type KilnTaskToolDescription =
  components["schemas"]["KilnTaskToolDescription"]
export type ExternalToolServer = components["schemas"]["ExternalToolServer"]
export type ExternalToolServerApiDescription =
  components["schemas"]["ExternalToolServerApiDescription"]
export type ExternalToolApiDescription =
  components["schemas"]["ExternalToolApiDescription"]
export type ToolServerType = components["schemas"]["ToolServerType"]
export type ToolSetType = components["schemas"]["ToolSetType"]
export type ToolApiDescription = components["schemas"]["ToolApiDescription"]
export type ToolSetApiDescription =
  components["schemas"]["ToolSetApiDescription"]
export type LocalServerProperties =
  components["schemas"]["LocalServerProperties"]
export type RemoteServerProperties =
  components["schemas"]["RemoteServerProperties"]
export type KilnTaskServerProperties =
  components["schemas"]["KilnTaskServerProperties"]
export type TaskToolCompatibility =
  components["schemas"]["TaskToolCompatibility"]
export type UserModelEntry = components["schemas"]["UserModelEntry"]
export type AvailableProviderInfo =
  components["schemas"]["AvailableProviderInfo"]

export type TraceMessage =
  | components["schemas"]["ChatCompletionDeveloperMessageParam"]
  | components["schemas"]["ChatCompletionSystemMessageParam"]
  | components["schemas"]["ChatCompletionUserMessageParam"]
  | components["schemas"]["ChatCompletionAssistantMessageParamWrapper"]
  | components["schemas"]["ChatCompletionToolMessageParamWrapper"]
  | components["schemas"]["ChatCompletionFunctionMessageParam"]
export type Trace = TraceMessage[]
export type ErrorWithTrace = components["schemas"]["ErrorWithTrace"]
export type ToolCallMessageParam =
  components["schemas"]["ChatCompletionMessageFunctionToolCallParam"]
export type RunChainEntry = components["schemas"]["RunChainEntry"]
export type SearchToolApiDescription =
  components["schemas"]["SearchToolApiDescription"]
export type CodeToolResponse = components["schemas"]["CodeToolResponse"]
export type CodeToolCreateResponse =
  components["schemas"]["CodeToolCreateResponse"]
export type TestCodeToolResponse = components["schemas"]["TestCodeToolResponse"]
export type ToolCallLogEntryResponse =
  components["schemas"]["ToolCallLogEntryResponse"]
export type Skill = components["schemas"]["SkillResponse"]
export type DocumentLibraryState = components["schemas"]["DocumentLibraryState"]
export type Spec = components["schemas"]["Spec"]
export type EvalStatus = components["schemas"]["EvalStatus"]
// Status moved from specs to evals; the old name is kept for existing call sites.
export type SpecStatus = EvalStatus
export type Priority = components["schemas"]["Priority"]
export type Feedback = components["schemas"]["Feedback"]
export type FeedbackSource = components["schemas"]["FeedbackSource"]

// Copilot API types
export type SyntheticDataGenerationStepConfigApi =
  components["schemas"]["SyntheticDataGenerationStepConfigApi"]
export type SyntheticDataGenerationSessionConfigApi =
  components["schemas"]["SyntheticDataGenerationSessionConfigApi"]
export type TaskMetadataApi = components["schemas"]["TaskMetadataApi"]
export type ReviewedExample = components["schemas"]["ReviewedExample"]
export type SampleApi = components["schemas"]["SampleApi"]
export type SubsampleBatchOutputItemApi =
  components["schemas"]["SubsampleBatchOutputItemApi"]
export type SpecProperties =
  | components["schemas"]["AppropriateToolUseProperties"]
  | components["schemas"]["DesiredBehaviourProperties"]
  | components["schemas"]["IssueProperties"]
  | components["schemas"]["ToneProperties"]
  | components["schemas"]["FormattingProperties"]
  | components["schemas"]["LocalizationProperties"]
  | components["schemas"]["AppropriateToolUseProperties"]
  | components["schemas"]["ReferenceAnswerAccuracyProperties"]
  | components["schemas"]["FactualCorrectnessProperties"]
  | components["schemas"]["HallucinationsProperties"]
  | components["schemas"]["CompletenessProperties"]
  | components["schemas"]["ToxicityProperties"]
  | components["schemas"]["BiasProperties"]
  | components["schemas"]["MaliciousnessProperties"]
  | components["schemas"]["NsfwProperties"]
  | components["schemas"]["TabooProperties"]
  | components["schemas"]["JailbreakProperties"]
  | components["schemas"]["PromptLeakageProperties"]
export type SpecType = SpecProperties["spec_type"]
export type SubmitAnswersRequest = components["schemas"]["SubmitAnswersRequest"]
export type SpecificationInput = components["schemas"]["SpecificationInput"]
export type QuestionSet = components["schemas"]["QuestionSet"]
export type QuestionWithAnswer = components["schemas"]["QuestionWithAnswer"]
export type AnswerOptionWithSelection =
  components["schemas"]["AnswerOptionWithSelection"]

export type PublicPromptOptimizationJobStatusResponse =
  components["schemas"]["PublicPromptOptimizationJobStatusResponse"]
export type PromptOptimizationJob =
  components["schemas"]["PromptOptimizationJob"]
export type JobStatus = components["schemas"]["JobStatus"]

// Type helpers for ExternalToolServerApiDescription properties

type ToolPropsByType = {
  remote_mcp: RemoteServerProperties
  local_mcp: LocalServerProperties
}

export function toolIsType<T extends keyof ToolPropsByType>(
  x: ExternalToolServerApiDescription,
  t: T,
): asserts x is ExternalToolServerApiDescription & {
  type: T
  properties: ToolPropsByType[T]
} {
  if (x.type !== t) {
    throw new Error(`Tool type mismatch: ${x.type} !== ${t}`)
  }
}

export function isToolType<T extends keyof ToolPropsByType>(
  x: ExternalToolServerApiDescription,
  t: T,
): x is ExternalToolServerApiDescription & {
  type: T
  properties: ToolPropsByType[T]
} {
  // Throw an error if the tool is not of the given type
  toolIsType(x, t)
  return true
}

export function isKilnAgentRunConfig(
  config: RunConfigProperties | null | undefined,
): config is KilnAgentRunConfigProperties {
  return config?.type === "kiln_agent"
}

export function isMcpRunConfig(
  config: RunConfigProperties | null | undefined,
): config is McpRunConfigProperties {
  return config?.type === "mcp"
}
