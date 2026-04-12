use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use serde::Serialize;

#[derive(Clone)]
pub struct Metrics {
    inner: Arc<MetricsInner>,
}

struct MetricsInner {
    total_requests: AtomicU64,
    total_prompt_tokens: AtomicU64,
    total_completion_tokens: AtomicU64,
    total_inference_ms: AtomicU64,
}

impl Metrics {
    pub fn new() -> Self {
        Self { inner: Arc::new(MetricsInner {
            total_requests: AtomicU64::new(0),
            total_prompt_tokens: AtomicU64::new(0),
            total_completion_tokens: AtomicU64::new(0),
            total_inference_ms: AtomicU64::new(0),
        })}
    }

    pub fn record_request(&self, prompt_tokens: u64, completion_tokens: u64, inference_ms: u64) {
        self.inner.total_requests.fetch_add(1, Ordering::Relaxed);
        self.inner.total_prompt_tokens.fetch_add(prompt_tokens, Ordering::Relaxed);
        self.inner.total_completion_tokens.fetch_add(completion_tokens, Ordering::Relaxed);
        self.inner.total_inference_ms.fetch_add(inference_ms, Ordering::Relaxed);
    }

    pub fn snapshot(&self) -> MetricsSnapshot {
        let total_requests = self.inner.total_requests.load(Ordering::Relaxed);
        let total_completion_tokens = self.inner.total_completion_tokens.load(Ordering::Relaxed);
        let total_inference_ms = self.inner.total_inference_ms.load(Ordering::Relaxed);
        let avg_tokens_per_sec = if total_inference_ms > 0 {
            (total_completion_tokens as f64 / total_inference_ms as f64) * 1000.0
        } else { 0.0 };
        MetricsSnapshot {
            total_requests,
            total_prompt_tokens: self.inner.total_prompt_tokens.load(Ordering::Relaxed),
            total_completion_tokens,
            total_inference_ms,
            avg_tokens_per_sec,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct MetricsSnapshot {
    pub total_requests: u64,
    pub total_prompt_tokens: u64,
    pub total_completion_tokens: u64,
    pub total_inference_ms: u64,
    pub avg_tokens_per_sec: f64,
}
