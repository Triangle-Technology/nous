//! Plugin system — SemanticPlugin trait + PluginRegistry.
//!
//! Brain analog: cortical modules — self-contained processing units
//! that communicate via the event bus (Global Workspace broadcast).
//!
//! P4: Plugins never import each other directly. All communication
//! goes through EventBus or trait-based kernel services.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;

use crate::errors::NousResult;
use crate::kernel::events::EventBus;

/// Core kernel services available to plugins during initialization.
pub struct KernelServices {
    pub events: Arc<EventBus>,
}

/// The plugin trait — implement this to extend Nous.
///
/// Lifecycle: register → boot (initialize) → running → shutdown (destroy).
/// Plugin crashes must not crash the kernel (P6 fail-open).
#[async_trait]
pub trait SemanticPlugin: Send + Sync {
    /// Unique identifier (e.g., "cognitive-memory", "triz").
    fn id(&self) -> &str;
    fn name(&self) -> &str;
    fn version(&self) -> &str;
    fn description(&self) -> &str;

    /// Called during kernel boot. Set up subscriptions, load state.
    async fn initialize(&self, services: &KernelServices) -> NousResult<()>;

    /// Called during kernel shutdown. Clean up resources.
    async fn destroy(&self) -> NousResult<()>;

    /// Capabilities this plugin provides (for registry indexing).
    fn capabilities(&self) -> Vec<PluginCapability> {
        vec![]
    }
}

/// What a plugin provides — indexed by the registry for fast lookup.
#[derive(Debug, Clone)]
pub enum PluginCapability {
    /// A semantic primitive (SUPERPOSE, INTERFERE, etc.).
    Primitive { id: String, name: String },
    /// A composition (DISSOLVE, INNOVATE, etc.).
    Composition { id: String, name: String },
    /// A reasoning dimension.
    Dimension { id: String, label: String },
    /// A theory implementation (metadata only).
    Theory { id: String, name: String },
}

/// Registry for fast plugin + capability lookup.
pub struct PluginRegistry {
    plugins: Vec<Arc<dyn SemanticPlugin>>,
    capabilities: Vec<(String, PluginCapability)>, // (plugin_id, capability)
    plugin_map: HashMap<String, usize>,            // id → index
}

impl PluginRegistry {
    pub fn new() -> Self {
        Self {
            plugins: Vec::new(),
            capabilities: Vec::new(),
            plugin_map: HashMap::new(),
        }
    }

    /// Mutable: adds plugin to registry and indexes its capabilities.
    /// Requires mutation because registration accumulates plugins during boot.
    pub fn register(&mut self, plugin: Arc<dyn SemanticPlugin>) {
        let id = plugin.id().to_string();
        let index = self.plugins.len();

        // Index capabilities
        for cap in plugin.capabilities() {
            self.capabilities.push((id.clone(), cap));
        }

        self.plugin_map.insert(id, index);
        self.plugins.push(plugin);
    }

    /// Get a plugin by ID.
    pub fn get(&self, id: &str) -> Option<&Arc<dyn SemanticPlugin>> {
        self.plugin_map.get(id).map(|&i| &self.plugins[i])
    }

    /// All registered plugins.
    pub fn all(&self) -> &[Arc<dyn SemanticPlugin>] {
        &self.plugins
    }

    /// Find all capabilities of a specific type.
    pub fn find_primitives(&self) -> Vec<(&str, &str)> {
        self.capabilities
            .iter()
            .filter_map(|(_plugin_id, cap)| match cap {
                PluginCapability::Primitive { id, name } => Some((id.as_str(), name.as_str())),
                _ => None,
            })
            .collect()
    }

    /// Find all compositions.
    pub fn find_compositions(&self) -> Vec<(&str, &str)> {
        self.capabilities
            .iter()
            .filter_map(|(_plugin_id, cap)| match cap {
                PluginCapability::Composition { id, name } => Some((id.as_str(), name.as_str())),
                _ => None,
            })
            .collect()
    }

    /// Find all dimensions.
    pub fn find_dimensions(&self) -> Vec<(&str, &str)> {
        self.capabilities
            .iter()
            .filter_map(|(_plugin_id, cap)| match cap {
                PluginCapability::Dimension { id, label } => Some((id.as_str(), label.as_str())),
                _ => None,
            })
            .collect()
    }

    /// Number of registered plugins.
    pub fn count(&self) -> usize {
        self.plugins.len()
    }
}

impl Default for PluginRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestPlugin {
        id: String,
    }

    #[async_trait]
    impl SemanticPlugin for TestPlugin {
        fn id(&self) -> &str { &self.id }
        fn name(&self) -> &str { "Test Plugin" }
        fn version(&self) -> &str { "0.1.0" }
        fn description(&self) -> &str { "A test plugin" }

        async fn initialize(&self, _services: &KernelServices) -> NousResult<()> { Ok(()) }
        async fn destroy(&self) -> NousResult<()> { Ok(()) }

        fn capabilities(&self) -> Vec<PluginCapability> {
            vec![PluginCapability::Theory {
                id: "test-theory".into(),
                name: "Test Theory".into(),
            }]
        }
    }

    #[test]
    fn register_and_lookup() {
        let mut registry = PluginRegistry::new();
        let plugin = Arc::new(TestPlugin { id: "test".into() });
        registry.register(plugin);

        assert_eq!(registry.count(), 1);
        assert!(registry.get("test").is_some());
        assert!(registry.get("nonexistent").is_none());
    }

    #[test]
    fn capability_indexing() {
        let mut registry = PluginRegistry::new();
        registry.register(Arc::new(TestPlugin { id: "p1".into() }));

        // Test plugin registers a Theory capability
        let dims = registry.find_dimensions();
        assert!(dims.is_empty()); // No dimensions

        let primitives = registry.find_primitives();
        assert!(primitives.is_empty()); // No primitives
    }
}
