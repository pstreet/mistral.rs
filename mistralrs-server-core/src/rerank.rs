//! Jina/Cohere-compatible rerank endpoint.

use anyhow::{anyhow, Context, Error as AnyhowError, Result};
use axum::{
    extract::{rejection::JsonRejection, Json, State},
    response::IntoResponse,
};
use futures::future::join_all;
use mistralrs_core::{
    Constraint, MistralRs, NormalRequest, Request, RequestMessage, Response, SamplingParams,
};
use tokio::sync::mpsc::Receiver;

use crate::{
    handler_core::{
        base_process_non_streaming_response, create_response_channel, openai_error_from_error,
        send_request_with_model, ApiError, ApiErrorKind, ModelErrorMessage,
    },
    openai::{RerankRequest, RerankResponse, RerankResult, RerankUsage},
    types::{ExtractedMistralRsState, SharedMistralRsState},
    util::validate_model_name,
};

/// Represents different types of rerank responses.
pub enum RerankResponder {
    Json(RerankResponse),
    InternalError(AnyhowError),
    ValidationError(AnyhowError),
}

impl IntoResponse for RerankResponder {
    fn into_response(self) -> axum::response::Response {
        match self {
            RerankResponder::Json(s) => Json(s).into_response(),
            RerankResponder::InternalError(e) => {
                openai_error_from_error(e.as_ref(), ApiErrorKind::Internal)
            }
            RerankResponder::ValidationError(e) => {
                openai_error_from_error(e.as_ref(), ApiErrorKind::InvalidRequest)
            }
        }
    }
}

#[utoipa::path(
    post,
    tag = "Mistral.rs",
    path = "/v1/rerank",
    request_body = RerankRequest,
    responses((status = 200, description = "Rerank scores", body = RerankResponse))
)]
pub async fn rerank(
    State(state): ExtractedMistralRsState,
    payload: Result<Json<RerankRequest>, JsonRejection>,
) -> RerankResponder {
    let oairequest = match payload {
        Ok(Json(request)) => request,
        Err(error) => {
            return validation_error(AnyhowError::new(ApiError::from_json_rejection(error)));
        }
    };
    let repr = serde_json::to_string(&oairequest).expect("Serialization of rerank request failed.");
    MistralRs::maybe_log_request(state.clone(), repr);

    if let Err(e) = validate_model_name(&oairequest.model, state.clone()) {
        return validation_error(e);
    }

    if oairequest.documents.is_empty() {
        return validation_error(AnyhowError::new(ApiError::new(
            ApiErrorKind::InvalidRequest,
            "documents must contain at least one entry.",
            Some("invalid_documents"),
            Some("documents"),
        )));
    }

    let model_override = if oairequest.model == "default" {
        None
    } else {
        Some(oairequest.model.clone())
    };

    let futures = oairequest.documents.iter().map(|document| {
        let state = state.clone();
        let model_override = model_override.clone();
        let query = oairequest.query.clone();
        async move {
            fetch_rerank_score(
                state,
                query,
                document.text().to_string(),
                model_override.as_deref(),
                oairequest.truncate_sequence.unwrap_or(false),
            )
            .await
        }
    });

    let results = join_all(futures).await;
    let mut scored = Vec::with_capacity(results.len());
    let mut total_prompt_tokens: usize = 0;
    let mut total_tokens: usize = 0;
    for (index, result) in results.into_iter().enumerate() {
        match result {
            Ok(ScoreWithUsage {
                score,
                prompt_tokens,
                total_tokens: item_total_tokens,
            }) => {
                scored.push((index, score));
                total_prompt_tokens = total_prompt_tokens.saturating_add(prompt_tokens);
                total_tokens = total_tokens.saturating_add(item_total_tokens);
            }
            Err(e) => {
                MistralRs::maybe_log_error(state.clone(), e.as_ref());
                return internal_error(e);
            }
        }
    }

    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut results: Vec<RerankResult> = scored
        .into_iter()
        .map(|(index, relevance_score)| RerankResult {
            index: index as u32,
            relevance_score,
        })
        .collect();
    if let Some(top_n) = oairequest.top_n {
        results.truncate(top_n);
    }

    let response = RerankResponse {
        id: format!("rerank_{}", uuid::Uuid::new_v4().simple()),
        model: oairequest.model,
        results,
        usage: RerankUsage {
            prompt_tokens: saturating_to_u32(total_prompt_tokens),
            total_tokens: saturating_to_u32(total_tokens),
        },
    };

    MistralRs::maybe_log_response(state.clone(), &response);

    RerankResponder::Json(response)
}

struct ScoreWithUsage {
    score: f32,
    prompt_tokens: usize,
    total_tokens: usize,
}

async fn fetch_rerank_score(
    state: SharedMistralRsState,
    query: String,
    document: String,
    model_id: Option<&str>,
    truncate_sequence: bool,
) -> Result<ScoreWithUsage> {
    let (tx, mut rx) = create_response_channel(Some(1));

    let request = Request::Normal(Box::new(NormalRequest {
        id: state.next_request_id(),
        queued_at: None,
        messages: RequestMessage::Rerank { query, document },
        sampling_params: SamplingParams::deterministic(),
        seed: None,
        response: tx,
        return_logprobs: false,
        is_streaming: false,
        suffix: None,
        constraint: Constraint::None,
        tool_choice: None,
        tools: None,
        logits_processors: None,
        return_raw_logits: false,
        web_search_options: None,
        enable_code_execution: false,
        enable_shell: false,
        shell_options: None,
        code_execution_permission: None,
        code_execution_approval_notifier: None,
        agent_permission: None,
        agent_approval_handler: None,
        agent_approval_notifier: None,
        max_tool_rounds: None,
        tool_dispatch_url: None,
        model_id: model_id.map(|m| m.to_string()),
        adapter: None,
        truncate_sequence,
        session_id: None,
        files: None,
        input_files: Vec::new(),
    }));

    send_request_with_model(&state, request, model_id)
        .await
        .context("Failed to dispatch rerank request")?;

    process_rerank_response(&mut rx, state.clone()).await
}

async fn process_rerank_response(
    rx: &mut Receiver<Response>,
    state: SharedMistralRsState,
) -> Result<ScoreWithUsage> {
    base_process_non_streaming_response(
        rx,
        state.clone(),
        |_, response| match response {
            Response::Embeddings {
                embeddings,
                prompt_tokens,
                total_tokens,
            } => {
                anyhow::ensure!(
                    embeddings.len() == 1,
                    "Expected a single rerank score, got {embeddings:?}"
                );
                Ok(ScoreWithUsage {
                    score: embeddings[0],
                    prompt_tokens,
                    total_tokens,
                })
            }
            Response::ValidationError(e) => Err(AnyhowError::new(ApiError::from_error(
                e.as_ref(),
                ApiErrorKind::InvalidRequest,
            ))),
            Response::InternalError(e) => {
                MistralRs::maybe_log_error(state.clone(), e.as_ref());
                Err(anyhow!(e))
            }
            Response::ModelError(msg, _) => {
                MistralRs::maybe_log_error(state.clone(), &ModelErrorMessage(msg));
                Err(AnyhowError::new(ApiError::model_error()))
            }
            Response::Done(_)
            | Response::Chunk(_)
            | Response::CompletionDone(_)
            | Response::CompletionChunk(_)
            | Response::CompletionModelError(_, _)
            | Response::ImageGeneration(_)
            | Response::Speech { .. }
            | Response::Raw { .. }
            | Response::AgenticToolCallProgress { .. }
            | Response::BlockDenoisingProgress(_)
            | Response::AgenticToolApprovalRequired { .. }
            | Response::File(_) => Err(anyhow!(
                "Received unexpected response type from rerank request."
            )),
        },
        |_, err| Err(anyhow!(err)),
    )
    .await
}

fn validation_error<E>(err: E) -> RerankResponder
where
    E: Into<AnyhowError>,
{
    let err = err.into();
    RerankResponder::ValidationError(err)
}

fn internal_error<E>(err: E) -> RerankResponder
where
    E: Into<AnyhowError>,
{
    let err = err.into();
    RerankResponder::InternalError(err)
}

fn saturating_to_u32(value: usize) -> u32 {
    if value > u32::MAX as usize {
        u32::MAX
    } else {
        value as u32
    }
}
