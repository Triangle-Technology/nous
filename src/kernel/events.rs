//! EventBus — typed publish/subscribe for plugin communication (P4).
//!
//! Brain analog: cortical broadcast — modules communicate via shared signals,
//! not direct connections (Global Workspace, Dehaene 2001).
//!
//! Plugins use EventBus instead of direct imports (P4 plugin isolation).

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::sync::Mutex;

/// Type-erased event handler.
type BoxedHandler = Box<dyn Fn(&dyn Any) + Send + Sync>;

/// Event bus for decoupled module communication.
///
/// Supports typed events: `emit::<MyEvent>(data)` and `on::<MyEvent>(handler)`.
/// Thread-safe via `Arc<Mutex<>>`.
pub struct EventBus {
    handlers: Mutex<HashMap<TypeId, Vec<BoxedHandler>>>,
}

impl EventBus {
    pub fn new() -> Self {
        Self {
            handlers: Mutex::new(HashMap::new()),
        }
    }

    /// Subscribe to events of type `T`.
    ///
    /// Returns a subscription ID (for future unsubscribe support).
    pub fn on<T: 'static + Send + Sync>(&self, handler: impl Fn(&T) + Send + Sync + 'static) -> usize {
        let mut handlers = self.handlers.lock().unwrap_or_else(|e| e.into_inner());
        let type_id = TypeId::of::<T>();
        let wrapped: BoxedHandler = Box::new(move |any| {
            if let Some(event) = any.downcast_ref::<T>() {
                handler(event);
            }
        });
        let list = handlers.entry(type_id).or_default();
        list.push(wrapped);
        list.len() - 1
    }

    /// Emit an event to all subscribers of type `T`.
    pub fn emit<T: 'static + Send + Sync>(&self, event: T) {
        let handlers = self.handlers.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(list) = handlers.get(&TypeId::of::<T>()) {
            for handler in list {
                handler(&event);
            }
        }
    }

    /// Number of handlers registered for type `T`.
    pub fn handler_count<T: 'static>(&self) -> usize {
        let handlers = self.handlers.lock().unwrap_or_else(|e| e.into_inner());
        handlers
            .get(&TypeId::of::<T>())
            .map(|list| list.len())
            .unwrap_or(0)
    }
}

impl Default for EventBus {
    fn default() -> Self {
        Self::new()
    }
}

// ── Well-Known Events ──────────────────────────────────────────────────

/// Emitted when a plugin finishes loading.
#[derive(Debug, Clone)]
pub struct PluginLoadedEvent {
    pub plugin_id: String,
}

/// Emitted when a pipeline step completes.
#[derive(Debug, Clone)]
pub struct PipelineStepEvent {
    pub composition_id: String,
    pub step: usize,
    pub primitive: String,
    pub model: String,
    pub duration_ms: u64,
}

/// Emitted when a pipeline completes.
#[derive(Debug, Clone)]
pub struct PipelineDoneEvent {
    pub composition_id: String,
    pub total_steps: usize,
    pub duration_ms: u64,
}

/// Emitted when a reasoning dimension analysis completes.
#[derive(Debug, Clone)]
pub struct DimensionDoneEvent {
    pub dimension_id: String,
    pub duration_ms: u64,
    pub tokens: u64,
}

/// Emitted on conversation switch.
#[derive(Debug, Clone)]
pub struct ConversationSwitchEvent {
    pub new_conversation_id: String,
    pub old_conversation_id: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU32, Ordering};

    #[derive(Debug, Clone)]
    struct TestEvent {
        value: u32,
    }

    #[test]
    fn emit_and_receive() {
        let bus = EventBus::new();
        let received = Arc::new(AtomicU32::new(0));
        let r = received.clone();

        bus.on::<TestEvent>(move |e| {
            r.store(e.value, Ordering::SeqCst);
        });

        bus.emit(TestEvent { value: 42 });
        assert_eq!(received.load(Ordering::SeqCst), 42);
    }

    #[test]
    fn multiple_handlers() {
        let bus = EventBus::new();
        let count = Arc::new(AtomicU32::new(0));
        let c1 = count.clone();
        let c2 = count.clone();

        bus.on::<TestEvent>(move |_| {
            c1.fetch_add(1, Ordering::SeqCst);
        });
        bus.on::<TestEvent>(move |_| {
            c2.fetch_add(1, Ordering::SeqCst);
        });

        bus.emit(TestEvent { value: 1 });
        assert_eq!(count.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn no_cross_type_leak() {
        let bus = EventBus::new();
        let received = Arc::new(AtomicU32::new(0));
        let r = received.clone();

        #[derive(Debug)]
        struct OtherEvent;

        bus.on::<TestEvent>(move |_| {
            r.store(1, Ordering::SeqCst);
        });

        bus.emit(OtherEvent);
        assert_eq!(received.load(Ordering::SeqCst), 0); // Not triggered
    }

    #[test]
    fn handler_count() {
        let bus = EventBus::new();
        assert_eq!(bus.handler_count::<TestEvent>(), 0);
        bus.on::<TestEvent>(|_| {});
        assert_eq!(bus.handler_count::<TestEvent>(), 1);
    }

    #[test]
    fn emit_no_handlers_is_noop() {
        let bus = EventBus::new();
        bus.emit(TestEvent { value: 99 }); // Should not panic
    }
}
