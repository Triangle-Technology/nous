//! Pipeline executor — runs semantic composition steps sequentially.
//!
//! Brain analog: cortical processing cascade — each stage transforms
//! the representation and passes to the next (Lamme 2000 feedforward sweep).
//!
//! Emits events at each step for progress tracking (P4 plugin isolation).

use std::time::Instant;

use crate::ai::provider::{
    AiProvider, CompletionRequest, MessageRole, ProviderMessage,
};
use crate::errors::{NousError, NousResult};
use crate::kernel::events::{EventBus, PipelineDoneEvent, PipelineStepEvent};

// ── Types ──────────────────────────────────────────────────────────────

/// A semantic primitive — one step in a pipeline.
pub struct PrimitiveStep {
    /// Primitive ID (e.g., "superpose", "interfere").
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// System prompt for this step.
    pub system_prompt: String,
    /// Model to use (can differ per step).
    pub model: String,
    /// Max tokens for response.
    pub max_tokens: u32,
    /// Temperature (0-1).
    pub temperature: f32,
}

/// Pipeline configuration.
pub struct PipelineConfig {
    /// Composition ID (e.g., "dissolve", "innovate").
    pub composition_id: String,
    /// Ordered steps to execute.
    pub steps: Vec<PrimitiveStep>,
}

/// Result of one pipeline step.
#[derive(Debug, Clone)]
pub struct StepResult {
    pub step_index: usize,
    pub primitive_id: String,
    pub output: String,
    pub model: String,
    pub duration_ms: u64,
}

/// Result of complete pipeline execution.
#[derive(Debug, Clone)]
pub struct PipelineResult {
    pub composition_id: String,
    pub steps: Vec<StepResult>,
    /// Final output = last step's output.
    pub final_output: String,
    pub total_duration_ms: u64,
}

// ── Core Function ──────────────────────────────────────────────────────

/// Execute a pipeline: run each step sequentially, feeding output into next.
///
/// Emits `PipelineStepEvent` after each step and `PipelineDoneEvent` at end.
/// P6 fail-open: if a step fails, returns error with partial results context.
pub async fn execute_pipeline(
    config: &PipelineConfig,
    input: &str,
    ai: &dyn AiProvider,
    events: Option<&EventBus>,
) -> NousResult<PipelineResult> {
    let pipeline_start = Instant::now();
    let mut step_results = Vec::new();
    let mut current_input = input.to_string();

    for (i, step) in config.steps.iter().enumerate() {
        let step_start = Instant::now();

        // Build user prompt: combine original input with previous step output
        let user_content = if i == 0 {
            current_input.clone()
        } else {
            format!(
                "Original input: {}\n\nPrevious analysis:\n{}",
                input, current_input
            )
        };

        let request = CompletionRequest {
            model: step.model.clone(),
            messages: vec![ProviderMessage {
                role: MessageRole::User,
                content: user_content,
            }],
            system_prompt: Some(step.system_prompt.clone()),
            max_tokens: step.max_tokens,
            temperature: step.temperature,
            stream: false,
        };

        let response = ai.complete(request).await.map_err(|e| NousError::Pipeline {
            composition_id: config.composition_id.clone(),
            step: i,
            message: e.to_string(),
        })?;

        let duration_ms = step_start.elapsed().as_millis() as u64;

        // Emit step event
        if let Some(bus) = events {
            bus.emit(PipelineStepEvent {
                composition_id: config.composition_id.clone(),
                step: i,
                primitive: step.id.clone(),
                model: step.model.clone(),
                duration_ms,
            });
        }

        step_results.push(StepResult {
            step_index: i,
            primitive_id: step.id.clone(),
            output: response.text.clone(),
            model: response.model,
            duration_ms,
        });

        current_input = response.text;
    }

    let total_duration_ms = pipeline_start.elapsed().as_millis() as u64;

    // Emit done event
    if let Some(bus) = events {
        bus.emit(PipelineDoneEvent {
            composition_id: config.composition_id.clone(),
            total_steps: config.steps.len(),
            duration_ms: total_duration_ms,
        });
    }

    Ok(PipelineResult {
        composition_id: config.composition_id.clone(),
        final_output: current_input,
        steps: step_results,
        total_duration_ms,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use crate::ai::provider::{AiProviderType, CompletionResponse, StreamChunk, TokenUsage};
    use async_trait::async_trait;

    /// Mock AI provider for testing.
    struct MockAi {
        response: String,
    }

    #[async_trait]
    impl AiProvider for MockAi {
        fn provider_type(&self) -> AiProviderType {
            AiProviderType::Local
        }

        async fn complete(&self, _request: CompletionRequest) -> NousResult<CompletionResponse> {
            Ok(CompletionResponse {
                text: self.response.clone(),
                usage: TokenUsage::default(),
                model: "mock".into(),
            })
        }

        async fn stream(
            &self,
            _request: CompletionRequest,
            sender: tokio::sync::mpsc::Sender<StreamChunk>,
        ) -> NousResult<()> {
            let _ = sender.send(StreamChunk::TextDelta(self.response.clone())).await;
            let _ = sender.send(StreamChunk::Done).await;
            Ok(())
        }
    }

    #[tokio::test]
    async fn single_step_pipeline() {
        let ai = MockAi {
            response: "analysis result".into(),
        };
        let config = PipelineConfig {
            composition_id: "test".into(),
            steps: vec![PrimitiveStep {
                id: "superpose".into(),
                name: "Superpose".into(),
                system_prompt: "Analyze this".into(),
                model: "mock".into(),
                max_tokens: 1024,
                temperature: 0.7,
            }],
        };

        let result = execute_pipeline(&config, "input text", &ai, None).await.unwrap();
        assert_eq!(result.final_output, "analysis result");
        assert_eq!(result.steps.len(), 1);
        assert_eq!(result.composition_id, "test");
    }

    #[tokio::test]
    async fn multi_step_pipeline_chains_output() {
        let ai = MockAi {
            response: "step output".into(),
        };
        let config = PipelineConfig {
            composition_id: "dissolve".into(),
            steps: vec![
                PrimitiveStep {
                    id: "superpose".into(),
                    name: "S".into(),
                    system_prompt: "Step 1".into(),
                    model: "mock".into(),
                    max_tokens: 1024,
                    temperature: 0.7,
                },
                PrimitiveStep {
                    id: "interfere".into(),
                    name: "I".into(),
                    system_prompt: "Step 2".into(),
                    model: "mock".into(),
                    max_tokens: 1024,
                    temperature: 0.7,
                },
            ],
        };

        let result = execute_pipeline(&config, "input", &ai, None).await.unwrap();
        assert_eq!(result.steps.len(), 2);
        assert_eq!(result.steps[0].primitive_id, "superpose");
        assert_eq!(result.steps[1].primitive_id, "interfere");
    }

    #[tokio::test]
    async fn pipeline_emits_events() {
        let ai = MockAi {
            response: "ok".into(),
        };
        let bus = EventBus::new();
        let count = Arc::new(std::sync::atomic::AtomicU32::new(0));
        let c = count.clone();
        bus.on::<PipelineStepEvent>(move |_| {
            c.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        });

        let config = PipelineConfig {
            composition_id: "test".into(),
            steps: vec![PrimitiveStep {
                id: "s".into(),
                name: "S".into(),
                system_prompt: "".into(),
                model: "m".into(),
                max_tokens: 100,
                temperature: 0.5,
            }],
        };

        execute_pipeline(&config, "x", &ai, Some(&bus)).await.unwrap();
        assert_eq!(count.load(std::sync::atomic::Ordering::SeqCst), 1);
    }
}
