//! Handling of `client/registerCapability` and `client/unregisterCapability` requests,
//! which let language servers toggle their capabilities at runtime.
//!
//! Multiple registrations of the same capability may coexist: the most recent one is
//! "active" and mirrored into the server's [`lsp::ServerCapabilities`], while the rest
//! (including the statically declared capability, if any) are kept to be restored when
//! the active registration is unregistered. Document change and save options
//! remain separate from the server-wide capabilities because they depend on the document.

use anyhow::{Context as _, Result};
use collections::HashSet;
use globset::{GlobBuilder, GlobMatcher};
use gpui::{Context, SharedString};
use lsp::{
    CompletionOptions, DiagnosticServerCapabilities, LanguageServer, LanguageServerId, OneOf,
};

use std::path::Path;

use crate::lsp_store::{
    LanguageServerState, LspStore, RenamePathsWatchedForServer, lsp_workspace_diagnostics_refresh,
    notify_server_capabilities_updated,
};

#[derive(Debug)]
pub(super) enum RegistrationSource {
    Static,
    Dynamic(String),
}

impl RegistrationSource {
    pub(super) fn registration_id(&self) -> Option<&str> {
        match self {
            RegistrationSource::Static => None,
            RegistrationSource::Dynamic(id) => Some(id),
        }
    }
}

pub(super) type CapabilityRegistrations<T> = Vec<(RegistrationSource, T)>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CapabilityUnregistration {
    NotFound,
    Removed { active_capability_changed: bool },
}

impl CapabilityUnregistration {
    fn removed(self) -> bool {
        self != CapabilityUnregistration::NotFound
    }

    fn capability_changed(self) -> bool {
        self == CapabilityUnregistration::Removed {
            active_capability_changed: true,
        }
    }
}

#[derive(Default, Debug)]
pub(super) struct DynamicRegistrations {
    pub(super) did_change_watched_files: HashSet<String>,
    pub(super) did_change: DocumentSelectorRegistrations<lsp::TextDocumentSyncKind>,
    pub(super) did_save: DocumentSelectorRegistrations<bool>,
    pub(super) diagnostics: CapabilityRegistrations<DiagnosticServerCapabilities>,
    workspace_folders: CapabilityRegistrations<lsp::WorkspaceFoldersServerCapabilities>,
    workspace_symbol: CapabilityRegistrations<OneOf<bool, lsp::WorkspaceSymbolOptions>>,
    file_operations: CapabilityRegistrations<lsp::WorkspaceFileOperationsServerCapabilities>,
    execute_command: CapabilityRegistrations<lsp::ExecuteCommandOptions>,
    range_formatting: CapabilityRegistrations<OneOf<bool, lsp::DocumentRangeFormattingOptions>>,
    on_type_formatting: CapabilityRegistrations<lsp::DocumentOnTypeFormattingOptions>,
    formatting: CapabilityRegistrations<OneOf<bool, lsp::DocumentFormattingOptions>>,
    pub(super) range_formatting_selectors: DocumentSelectorRegistrations<()>,
    pub(super) on_type_formatting_selectors:
        DocumentSelectorRegistrations<lsp::DocumentOnTypeFormattingOptions>,
    pub(super) formatting_selectors: DocumentSelectorRegistrations<()>,
    rename: CapabilityRegistrations<OneOf<bool, lsp::RenameOptions>>,
    code_action: CapabilityRegistrations<lsp::CodeActionProviderCapability>,
    pub(super) definition_selectors: DocumentSelectorRegistrations<()>,
    definition: CapabilityRegistrations<OneOf<bool, lsp::DefinitionOptions>>,
    completion: CapabilityRegistrations<CompletionOptions>,
    pub(super) completion_selectors: DocumentSelectorRegistrations<CompletionOptions>,
    pub(super) hover_selectors: DocumentSelectorRegistrations<()>,
    hover: CapabilityRegistrations<lsp::HoverProviderCapability>,
    signature_help: CapabilityRegistrations<lsp::SignatureHelpOptions>,
    color: CapabilityRegistrations<lsp::ColorProviderCapability>,
    folding_range: CapabilityRegistrations<lsp::FoldingRangeProviderCapability>,
    document_link: CapabilityRegistrations<lsp::DocumentLinkOptions>,
    semantic_tokens: CapabilityRegistrations<lsp::SemanticTokensServerCapabilities>,
    inlay_hint: CapabilityRegistrations<OneOf<bool, lsp::InlayHintServerCapabilities>>,
    code_lens: CapabilityRegistrations<lsp::CodeLensOptions>,
    document_symbol: CapabilityRegistrations<OneOf<bool, lsp::DocumentSymbolOptions>>,
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct ChangeRegistrationOptions {
    document_selector: Option<lsp::DocumentSelector>,
    sync_kind: lsp::TextDocumentSyncKind,
}

#[derive(Debug)]
pub(super) struct DocumentSelectorRegistrations<T>(Vec<DocumentSelectorRegistration<T>>);

impl<T> Default for DocumentSelectorRegistrations<T> {
    fn default() -> Self {
        Self(Vec::new())
    }
}

#[derive(Debug)]
struct DocumentSelectorRegistration<T> {
    id: String,
    selector: Option<Vec<DocumentSelectorFilter>>,
    options: T,
}

#[derive(Debug)]
struct DocumentSelectorFilter {
    language: Option<String>,
    scheme: Option<String>,
    pattern: Option<GlobMatcher>,
}

#[derive(Clone, Copy)]
pub(super) struct DocumentSelectorTarget<'a> {
    pub uri: &'a lsp::Uri,
    pub absolute_path: &'a Path,
    pub worktree_path: &'a str,
    pub language_id: Option<&'a str>,
    pub matches_client_selector: bool,
}

impl<T: Clone> DocumentSelectorRegistrations<T> {
    fn register(
        &mut self,
        id: String,
        selector: Option<lsp::DocumentSelector>,
        options: T,
    ) -> Result<()> {
        let selector = selector
            .map(|filters| {
                filters
                    .into_iter()
                    .map(|filter| {
                        let pattern = filter
                            .pattern
                            .as_deref()
                            .map(|pattern| {
                                GlobBuilder::new(pattern)
                                    .literal_separator(true)
                                    .build()
                                    .map(|glob| glob.compile_matcher())
                                    .with_context(|| {
                                        format!("Invalid LSP document selector pattern {pattern:?}")
                                    })
                            })
                            .transpose()?;
                        Ok(DocumentSelectorFilter {
                            language: filter.language,
                            scheme: filter.scheme,
                            pattern,
                        })
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .transpose()?;
        if let Some(registration) = self.0.iter_mut().find(|registration| registration.id == id) {
            registration.selector = selector;
            registration.options = options;
        } else {
            self.0.push(DocumentSelectorRegistration {
                id,
                selector,
                options,
            });
        }
        Ok(())
    }

    fn unregister(&mut self, id: &str) {
        self.0.retain(|registration| registration.id != id);
    }

    pub(super) fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub(super) fn matching_options(&self, target: DocumentSelectorTarget<'_>) -> Option<T> {
        self.0.iter().rev().find_map(|registration| {
            let matches = match &registration.selector {
                Some(filters) => filters.iter().any(|filter| {
                    filter
                        .language
                        .as_deref()
                        .is_none_or(|language| Some(language) == target.language_id)
                        && filter
                            .scheme
                            .as_deref()
                            .is_none_or(|scheme| scheme == target.uri.scheme())
                        && filter.pattern.as_ref().is_none_or(|pattern| {
                            pattern.is_match(target.absolute_path)
                                || pattern.is_match(target.worktree_path)
                        })
                }),
                None => target.matches_client_selector,
            };
            matches.then(|| registration.options.clone())
        })
    }
}

impl LspStore {
    fn register_document_selector<T: Clone>(
        &mut self,
        server_id: LanguageServerId,
        id: String,
        selector: Option<lsp::DocumentSelector>,
        options: T,
        selectors: impl FnOnce(&mut DynamicRegistrations) -> &mut DocumentSelectorRegistrations<T>,
    ) -> Result<()> {
        let registrations = self
            .as_local_mut()
            .context("Expected LSP Store to be local")?
            .language_server_dynamic_registrations
            .entry(server_id)
            .or_default();
        selectors(registrations).register(id, selector, options)
    }

    fn unregister_document_selector<T: Clone>(
        &mut self,
        server_id: LanguageServerId,
        id: &str,
        selectors: impl FnOnce(&mut DynamicRegistrations) -> &mut DocumentSelectorRegistrations<T>,
    ) {
        if let Some(registrations) = self.as_local_mut().and_then(|local| {
            local
                .language_server_dynamic_registrations
                .get_mut(&server_id)
        }) {
            selectors(registrations).unregister(id);
        }
    }

    /// Returns `true` when the registration changed the server's active capability value:
    /// duplicate-ID replacements of non-active registrations and (re-)registrations with
    /// options identical to the active ones do not.
    fn register_dynamic_capability<T: Clone + PartialEq>(
        &mut self,
        server: &LanguageServer,
        method: &str,
        registration_id: String,
        options: T,
        cx: &mut Context<Self>,
        registrations_of: impl FnOnce(&mut DynamicRegistrations) -> &mut CapabilityRegistrations<T>,
        capability_of: impl Fn(&mut lsp::ServerCapabilities) -> &mut Option<T>,
    ) -> anyhow::Result<bool> {
        let server_id = server.server_id();
        let local = self
            .as_local_mut()
            .context("Expected LSP Store to be local")?;
        let registrations = registrations_of(
            local
                .language_server_dynamic_registrations
                .entry(server_id)
                .or_default(),
        );
        if registrations.is_empty() {
            let mut initial_capabilities = server.capabilities();
            if let Some(static_options) = capability_of(&mut initial_capabilities).take() {
                registrations.push((RegistrationSource::Static, static_options));
            }
        }
        let previously_active = registrations.last().map(|(_, options)| options.clone());
        if let Some(index) = registrations
            .iter()
            .position(|(source, _)| source.registration_id() == Some(registration_id.as_str()))
        {
            log::warn!(
                "Received a duplicate {method} registration with ID {registration_id}, replacing the previous one"
            );
            registrations[index].1 = options;
        } else {
            registrations.push((RegistrationSource::Dynamic(registration_id), options));
        }
        let active = registrations.last().map(|(_, options)| options.clone());
        let active_changed = active != previously_active;
        if active_changed {
            server.update_capabilities(|capabilities| *capability_of(capabilities) = active);
            notify_server_capabilities_updated(server, cx);
        }
        Ok(active_changed)
    }

    fn unregister_dynamic_capability<T: Clone + PartialEq>(
        &mut self,
        server: &LanguageServer,
        unregistration: &lsp::Unregistration,
        cx: &mut Context<Self>,
        registrations_of: impl FnOnce(&mut DynamicRegistrations) -> &mut CapabilityRegistrations<T>,
        capability_of: impl FnOnce(&mut lsp::ServerCapabilities) -> &mut Option<T>,
    ) -> anyhow::Result<CapabilityUnregistration> {
        let server_id = server.server_id();
        let local = self
            .as_local_mut()
            .context("Expected LSP Store to be local")?;
        let registrations = local
            .language_server_dynamic_registrations
            .get_mut(&server_id)
            .map(registrations_of);
        let index = registrations.as_ref().and_then(|registrations| {
            registrations.iter().position(|(source, _)| {
                source.registration_id() == Some(unregistration.id.as_str())
            })
        });
        let (Some(registrations), Some(index)) = (registrations, index) else {
            log::warn!(
                "Attempted to unregister non-existent {} registration with ID {}",
                unregistration.method,
                unregistration.id
            );
            return Ok(CapabilityUnregistration::NotFound);
        };
        let removed_active = index + 1 == registrations.len();
        let (_, removed_options) = registrations.remove(index);
        let restored = registrations.last().map(|(_, options)| options.clone());
        if !removed_active || restored.as_ref() == Some(&removed_options) {
            return Ok(CapabilityUnregistration::Removed {
                active_capability_changed: false,
            });
        }
        server.update_capabilities(|capabilities| *capability_of(capabilities) = restored);
        notify_server_capabilities_updated(server, cx);
        Ok(CapabilityUnregistration::Removed {
            active_capability_changed: true,
        })
    }

    fn update_paths_watched_for_rename(&mut self, server: &LanguageServer) {
        let Some(local) = self.as_local_mut() else {
            return;
        };
        let watcher = server
            .capabilities()
            .workspace
            .and_then(|workspace| workspace.file_operations)
            .and_then(|file_operations| {
                let did_rename_caps = file_operations.did_rename.as_ref();
                let will_rename_caps = file_operations.will_rename.as_ref();
                did_rename_caps.or(will_rename_caps)?;
                Some(
                    RenamePathsWatchedForServer::default()
                        .with_did_rename_patterns(did_rename_caps)
                        .with_will_rename_patterns(will_rename_caps),
                )
            });
        match watcher {
            Some(watcher) => {
                local
                    .language_server_paths_watched_for_rename
                    .insert(server.server_id(), watcher);
            }
            None => {
                local
                    .language_server_paths_watched_for_rename
                    .remove(&server.server_id());
            }
        }
    }

    fn apply_completion_triggers(&self, server: &LanguageServer, cx: &mut Context<Self>) {
        let server_id = server.server_id();
        let Some(local) = self.as_local() else {
            return;
        };
        let mut buffers_with_language_server = Vec::new();
        for handle in self.buffer_store.read(cx).buffers() {
            let buffer_id = handle.read(cx).remote_id();
            if local
                .buffers_opened_in_servers
                .get(&buffer_id)
                .filter(|s| s.contains(&server_id))
                .is_some()
            {
                buffers_with_language_server.push(handle);
            }
        }
        for handle in buffers_with_language_server {
            handle.update(cx, |buffer, cx| {
                let triggers = local.completion_triggers_for_buffer(buffer, server, cx);
                buffer.set_completion_triggers(server_id, triggers, cx);
            });
        }
    }

    pub(super) fn register_server_capabilities(
        &mut self,
        server_id: LanguageServerId,
        params: lsp::RegistrationParams,
        cx: &mut Context<Self>,
    ) -> anyhow::Result<()> {
        let server = self
            .language_server_for_id(server_id)
            .with_context(|| format!("no server {server_id} found"))?;
        for reg in params.registrations {
            match reg.method.as_str() {
                "workspace/didChangeWatchedFiles" => {
                    if let Some(options) = reg.register_options {
                        let notify = if let Some(local_lsp_store) = self.as_local_mut() {
                            let caps = serde_json::from_value(options)?;
                            local_lsp_store
                                .on_lsp_did_change_watched_files(server_id, &reg.id, caps, cx);
                            true
                        } else {
                            false
                        };
                        if notify {
                            notify_server_capabilities_updated(&server, cx);
                        }
                    }
                }
                "workspace/didChangeConfiguration" => {
                    // Ignore payload since we notify clients of setting changes unconditionally, relying on them pulling the latest settings.
                }
                "workspace/didChangeWorkspaceFolders" => {
                    // In this case register options is an empty object, we can ignore it
                    let caps = lsp::WorkspaceFoldersServerCapabilities {
                        supported: Some(true),
                        change_notifications: Some(OneOf::Right(reg.id.clone())),
                    };
                    self.register_dynamic_capability(
                        &server,
                        &reg.method,
                        reg.id,
                        caps,
                        cx,
                        |registrations| &mut registrations.workspace_folders,
                        |capabilities| {
                            &mut capabilities
                                .workspace
                                .get_or_insert_default()
                                .workspace_folders
                        },
                    )?;
                }
                "workspace/symbol" => {
                    let options = parse_register_capabilities(reg.register_options)?;
                    self.register_dynamic_capability(
                        &server,
                        &reg.method,
                        reg.id,
                        options,
                        cx,
                        |registrations| &mut registrations.workspace_symbol,
                        |capabilities| &mut capabilities.workspace_symbol_provider,
                    )?;
                }
                "workspace/fileOperations" => {
                    if let Some(options) = reg.register_options {
                        let caps = serde_json::from_value(options)?;
                        if self.register_dynamic_capability(
                            &server,
                            &reg.method,
                            reg.id,
                            caps,
                            cx,
                            |registrations| &mut registrations.file_operations,
                            |capabilities| {
                                &mut capabilities
                                    .workspace
                                    .get_or_insert_default()
                                    .file_operations
                            },
                        )? {
                            self.update_paths_watched_for_rename(&server);
                        }
                    }
                }
                "workspace/executeCommand" => {
                    if let Some(options) = reg.register_options {
                        let options = serde_json::from_value(options)?;
                        self.register_dynamic_capability(
                            &server,
                            &reg.method,
                            reg.id,
                            options,
                            cx,
                            |registrations| &mut registrations.execute_command,
                            |capabilities| &mut capabilities.execute_command_provider,
                        )?;
                    }
                }
                "textDocument/rangeFormatting" => {
                    let selector = parse_document_selector(reg.register_options.as_ref())?;
                    let options = parse_register_capabilities(reg.register_options)?;
                    self.register_document_selector(
                        server_id,
                        reg.id.clone(),
                        selector,
                        (),
                        |registrations| &mut registrations.range_formatting_selectors,
                    )?;
                    self.register_dynamic_capability(
                        &server,
                        &reg.method,
                        reg.id,
                        options,
                        cx,
                        |registrations| &mut registrations.range_formatting,
                        |capabilities| &mut capabilities.document_range_formatting_provider,
                    )?;
                }
                "textDocument/onTypeFormatting" => {
                    let selector = parse_document_selector(reg.register_options.as_ref())?;
                    if let Some(options) = reg
                        .register_options
                        .map(serde_json::from_value::<lsp::DocumentOnTypeFormattingOptions>)
                        .transpose()?
                    {
                        self.register_document_selector(
                            server_id,
                            reg.id.clone(),
                            selector,
                            options.clone(),
                            |registrations| &mut registrations.on_type_formatting_selectors,
                        )?;
                        self.register_dynamic_capability(
                            &server,
                            &reg.method,
                            reg.id,
                            options,
                            cx,
                            |registrations| &mut registrations.on_type_formatting,
                            |capabilities| &mut capabilities.document_on_type_formatting_provider,
                        )?;
                    }
                }
                "textDocument/formatting" => {
                    let selector = parse_document_selector(reg.register_options.as_ref())?;
                    let options = parse_register_capabilities(reg.register_options)?;
                    self.register_document_selector(
                        server_id,
                        reg.id.clone(),
                        selector,
                        (),
                        |registrations| &mut registrations.formatting_selectors,
                    )?;
                    self.register_dynamic_capability(
                        &server,
                        &reg.method,
                        reg.id,
                        options,
                        cx,
                        |registrations| &mut registrations.formatting,
                        |capabilities| &mut capabilities.document_formatting_provider,
                    )?;
                }
                "textDocument/rename" => {
                    let options = parse_register_capabilities(reg.register_options)?;
                    self.register_dynamic_capability(
                        &server,
                        &reg.method,
                        reg.id,
                        options,
                        cx,
                        |registrations| &mut registrations.rename,
                        |capabilities| &mut capabilities.rename_provider,
                    )?;
                }
                "textDocument/inlayHint" => {
                    let options = parse_register_capabilities(reg.register_options)?;
                    if self.register_dynamic_capability(
                        &server,
                        &reg.method,
                        reg.id,
                        options,
                        cx,
                        |registrations| &mut registrations.inlay_hint,
                        |capabilities| &mut capabilities.inlay_hint_provider,
                    )? {
                        self.refresh_inlay_hints(server_id, cx);
                    }
                }
                "textDocument/documentSymbol" => {
                    let options = parse_register_capabilities(reg.register_options)?;
                    if self.register_dynamic_capability(
                        &server,
                        &reg.method,
                        reg.id,
                        options,
                        cx,
                        |registrations| &mut registrations.document_symbol,
                        |capabilities| &mut capabilities.document_symbol_provider,
                    )? {
                        self.refresh_document_symbols(Some(server_id), cx);
                    }
                }
                "textDocument/codeAction" => {
                    let options = parse_register_capabilities(reg.register_options)?;
                    let provider = match options {
                        OneOf::Left(value) => lsp::CodeActionProviderCapability::Simple(value),
                        OneOf::Right(caps) => caps,
                    };
                    self.register_dynamic_capability(
                        &server,
                        &reg.method,
                        reg.id,
                        provider,
                        cx,
                        |registrations| &mut registrations.code_action,
                        |capabilities| &mut capabilities.code_action_provider,
                    )?;
                }
                "textDocument/definition" => {
                    let selector = parse_document_selector(reg.register_options.as_ref())?;
                    let options = parse_register_capabilities(reg.register_options)?;
                    self.register_document_selector(
                        server_id,
                        reg.id.clone(),
                        selector,
                        (),
                        |registrations| &mut registrations.definition_selectors,
                    )?;
                    self.register_dynamic_capability(
                        &server,
                        &reg.method,
                        reg.id,
                        options,
                        cx,
                        |registrations| &mut registrations.definition,
                        |capabilities| &mut capabilities.definition_provider,
                    )?;
                }
                "textDocument/completion" => {
                    let selector = parse_document_selector(reg.register_options.as_ref())?;
                    if let Some(options) = reg
                        .register_options
                        .map(serde_json::from_value::<CompletionOptions>)
                        .transpose()?
                    {
                        self.register_document_selector(
                            server_id,
                            reg.id.clone(),
                            selector,
                            options.clone(),
                            |registrations| &mut registrations.completion_selectors,
                        )?;
                        self.register_dynamic_capability(
                            &server,
                            &reg.method,
                            reg.id,
                            options,
                            cx,
                            |registrations| &mut registrations.completion,
                            |capabilities| &mut capabilities.completion_provider,
                        )?;
                        self.apply_completion_triggers(&server, cx);
                    }
                }
                "textDocument/hover" => {
                    let selector = parse_document_selector(reg.register_options.as_ref())?;
                    let options = parse_register_capabilities(reg.register_options)?;
                    let provider = match options {
                        OneOf::Left(value) => lsp::HoverProviderCapability::Simple(value),
                        OneOf::Right(caps) => caps,
                    };
                    self.register_document_selector(
                        server_id,
                        reg.id.clone(),
                        selector,
                        (),
                        |registrations| &mut registrations.hover_selectors,
                    )?;
                    self.register_dynamic_capability(
                        &server,
                        &reg.method,
                        reg.id,
                        provider,
                        cx,
                        |registrations| &mut registrations.hover,
                        |capabilities| &mut capabilities.hover_provider,
                    )?;
                }
                "textDocument/signatureHelp" => {
                    if let Some(caps) = reg
                        .register_options
                        .map(serde_json::from_value)
                        .transpose()?
                    {
                        self.register_dynamic_capability(
                            &server,
                            &reg.method,
                            reg.id,
                            caps,
                            cx,
                            |registrations| &mut registrations.signature_help,
                            |capabilities| &mut capabilities.signature_help_provider,
                        )?;
                    }
                }
                "textDocument/didChange" => {
                    if let Some(options) = reg
                        .register_options
                        .map(serde_json::from_value::<ChangeRegistrationOptions>)
                        .transpose()?
                    {
                        self.as_local_mut()
                            .context("Expected LSP Store to be local")?
                            .language_server_dynamic_registrations
                            .entry(server_id)
                            .or_default()
                            .did_change
                            .register(reg.id, options.document_selector, options.sync_kind)?;
                    }
                }
                "textDocument/didSave" => {
                    let options: lsp::TextDocumentSaveRegistrationOptions = serde_json::from_value(
                        reg.register_options
                            .unwrap_or_else(|| serde_json::json!({})),
                    )?;
                    self.as_local_mut()
                        .context("Expected LSP Store to be local")?
                        .language_server_dynamic_registrations
                        .entry(server_id)
                        .or_default()
                        .did_save
                        .register(
                            reg.id,
                            options.text_document_registration_options.document_selector,
                            options.include_text.unwrap_or(false),
                        )?;
                }
                "textDocument/codeLens" => {
                    if let Some(options) = reg
                        .register_options
                        .map(serde_json::from_value)
                        .transpose()?
                    {
                        if self.register_dynamic_capability(
                            &server,
                            &reg.method,
                            reg.id,
                            options,
                            cx,
                            |registrations| &mut registrations.code_lens,
                            |capabilities| &mut capabilities.code_lens_provider,
                        )? {
                            self.refresh_code_lens(Some(server_id), cx);
                        }
                    }
                }
                "textDocument/diagnostic" => {
                    if let Some(caps) = reg
                        .register_options
                        .map(serde_json::from_value::<DiagnosticServerCapabilities>)
                        .transpose()?
                    {
                        let registration_id = reg.id.clone();
                        self.register_dynamic_capability(
                            &server,
                            &reg.method,
                            reg.id,
                            caps.clone(),
                            cx,
                            |registrations| &mut registrations.diagnostics,
                            |capabilities| &mut capabilities.diagnostic_provider,
                        )?;

                        let supports_workspace_diagnostics =
                            |capabilities: &DiagnosticServerCapabilities| match capabilities {
                                DiagnosticServerCapabilities::Options(diagnostic_options) => {
                                    diagnostic_options.workspace_diagnostics
                                }
                                DiagnosticServerCapabilities::RegistrationOptions(
                                    diagnostic_registration_options,
                                ) => {
                                    diagnostic_registration_options
                                        .diagnostic_options
                                        .workspace_diagnostics
                                }
                            };

                        let local = self
                            .as_local_mut()
                            .context("Expected LSP Store to be local")?;
                        let state = local
                            .language_servers
                            .get_mut(&server_id)
                            .context("Could not obtain Language Servers state")?;
                        if let LanguageServerState::Running {
                            workspace_diagnostics_refresh_tasks,
                            ..
                        } = state
                        {
                            workspace_diagnostics_refresh_tasks
                                .remove(&Some(registration_id.clone()));
                            if supports_workspace_diagnostics(&caps)
                                && let Some(task) = lsp_workspace_diagnostics_refresh(
                                    Some(registration_id.clone()),
                                    caps,
                                    server.clone(),
                                    cx,
                                )
                            {
                                workspace_diagnostics_refresh_tasks
                                    .insert(Some(registration_id), task);
                            }
                        }

                        let _ = self.pull_document_diagnostics_for_server(server_id, None, cx);
                    }
                }
                "textDocument/documentColor" => {
                    let options = parse_register_capabilities(reg.register_options)?;
                    let provider = match options {
                        OneOf::Left(value) => lsp::ColorProviderCapability::Simple(value),
                        OneOf::Right(caps) => caps,
                    };
                    if self.register_dynamic_capability(
                        &server,
                        &reg.method,
                        reg.id,
                        provider,
                        cx,
                        |registrations| &mut registrations.color,
                        |capabilities| &mut capabilities.color_provider,
                    )? {
                        self.refresh_document_colors(Some(server_id), cx);
                    }
                }
                "textDocument/foldingRange" => {
                    let options = parse_register_capabilities(reg.register_options)?;
                    let provider = match options {
                        OneOf::Left(value) => lsp::FoldingRangeProviderCapability::Simple(value),
                        OneOf::Right(caps) => caps,
                    };
                    if self.register_dynamic_capability(
                        &server,
                        &reg.method,
                        reg.id,
                        provider,
                        cx,
                        |registrations| &mut registrations.folding_range,
                        |capabilities| &mut capabilities.folding_range_provider,
                    )? {
                        self.refresh_folding_ranges(Some(server_id), cx);
                    }
                }
                "textDocument/documentLink" => {
                    if let Some(caps) = reg
                        .register_options
                        .map(serde_json::from_value)
                        .transpose()?
                    {
                        if self.register_dynamic_capability(
                            &server,
                            &reg.method,
                            reg.id,
                            caps,
                            cx,
                            |registrations| &mut registrations.document_link,
                            |capabilities| &mut capabilities.document_link_provider,
                        )? {
                            self.refresh_document_links(Some(server_id), cx);
                        }
                    }
                }
                "textDocument/semanticTokens" => {
                    if let Some(caps) = reg
                        .register_options
                        .map(serde_json::from_value::<lsp::SemanticTokensRegistrationOptions>)
                        .transpose()?
                    {
                        if self.register_dynamic_capability(
                            &server,
                            &reg.method,
                            reg.id,
                            lsp::SemanticTokensServerCapabilities::SemanticTokensRegistrationOptions(caps),
                            cx,
                            |registrations| &mut registrations.semantic_tokens,
                            |capabilities| &mut capabilities.semantic_tokens_provider,
                        )? {
                            // Re-query already-open buffers, which would otherwise keep
                            // tree-sitter-only highlighting until edited.
                            self.refresh_semantic_tokens(server_id, cx);
                        }
                    }
                }
                _ => log::warn!("unhandled capability registration: {reg:?}"),
            }
        }

        Ok(())
    }

    pub(super) fn unregister_server_capabilities(
        &mut self,
        server_id: LanguageServerId,
        params: lsp::UnregistrationParams,
        cx: &mut Context<Self>,
    ) -> anyhow::Result<()> {
        let server = self
            .language_server_for_id(server_id)
            .with_context(|| format!("no server {server_id} found"))?;
        for unreg in params.unregisterations.iter() {
            match unreg.method.as_str() {
                "workspace/didChangeWatchedFiles" => {
                    let notify = if let Some(local_lsp_store) = self.as_local_mut() {
                        local_lsp_store
                            .on_lsp_unregister_did_change_watched_files(server_id, &unreg.id, cx);
                        true
                    } else {
                        false
                    };
                    if notify {
                        notify_server_capabilities_updated(&server, cx);
                    }
                }
                "workspace/didChangeConfiguration" => {
                    // Ignore payload since we notify clients of setting changes unconditionally, relying on them pulling the latest settings.
                }
                "workspace/didChangeWorkspaceFolders" => {
                    self.unregister_dynamic_capability(
                        &server,
                        unreg,
                        cx,
                        |registrations| &mut registrations.workspace_folders,
                        |capabilities| {
                            &mut capabilities
                                .workspace
                                .get_or_insert_default()
                                .workspace_folders
                        },
                    )?;
                }
                "workspace/symbol" => {
                    self.unregister_dynamic_capability(
                        &server,
                        unreg,
                        cx,
                        |registrations| &mut registrations.workspace_symbol,
                        |capabilities| &mut capabilities.workspace_symbol_provider,
                    )?;
                }
                "workspace/fileOperations" => {
                    if self
                        .unregister_dynamic_capability(
                            &server,
                            unreg,
                            cx,
                            |registrations| &mut registrations.file_operations,
                            |capabilities| {
                                &mut capabilities
                                    .workspace
                                    .get_or_insert_default()
                                    .file_operations
                            },
                        )?
                        .capability_changed()
                    {
                        self.update_paths_watched_for_rename(&server);
                    }
                }
                "workspace/executeCommand" => {
                    self.unregister_dynamic_capability(
                        &server,
                        unreg,
                        cx,
                        |registrations| &mut registrations.execute_command,
                        |capabilities| &mut capabilities.execute_command_provider,
                    )?;
                }
                "textDocument/rangeFormatting" => {
                    self.unregister_document_selector(server_id, &unreg.id, |registrations| {
                        &mut registrations.range_formatting_selectors
                    });
                    self.unregister_dynamic_capability(
                        &server,
                        unreg,
                        cx,
                        |registrations| &mut registrations.range_formatting,
                        |capabilities| &mut capabilities.document_range_formatting_provider,
                    )?;
                }
                "textDocument/onTypeFormatting" => {
                    self.unregister_document_selector(server_id, &unreg.id, |registrations| {
                        &mut registrations.on_type_formatting_selectors
                    });
                    self.unregister_dynamic_capability(
                        &server,
                        unreg,
                        cx,
                        |registrations| &mut registrations.on_type_formatting,
                        |capabilities| &mut capabilities.document_on_type_formatting_provider,
                    )?;
                }
                "textDocument/formatting" => {
                    self.unregister_document_selector(server_id, &unreg.id, |registrations| {
                        &mut registrations.formatting_selectors
                    });
                    self.unregister_dynamic_capability(
                        &server,
                        unreg,
                        cx,
                        |registrations| &mut registrations.formatting,
                        |capabilities| &mut capabilities.document_formatting_provider,
                    )?;
                }
                "textDocument/rename" => {
                    self.unregister_dynamic_capability(
                        &server,
                        unreg,
                        cx,
                        |registrations| &mut registrations.rename,
                        |capabilities| &mut capabilities.rename_provider,
                    )?;
                }
                "textDocument/codeAction" => {
                    self.unregister_dynamic_capability(
                        &server,
                        unreg,
                        cx,
                        |registrations| &mut registrations.code_action,
                        |capabilities| &mut capabilities.code_action_provider,
                    )?;
                }
                "textDocument/definition" => {
                    self.unregister_document_selector(server_id, &unreg.id, |registrations| {
                        &mut registrations.definition_selectors
                    });
                    self.unregister_dynamic_capability(
                        &server,
                        unreg,
                        cx,
                        |registrations| &mut registrations.definition,
                        |capabilities| &mut capabilities.definition_provider,
                    )?;
                }
                "textDocument/completion" => {
                    self.unregister_document_selector(server_id, &unreg.id, |registrations| {
                        &mut registrations.completion_selectors
                    });
                    self.unregister_dynamic_capability(
                        &server,
                        unreg,
                        cx,
                        |registrations| &mut registrations.completion,
                        |capabilities| &mut capabilities.completion_provider,
                    )?;
                    self.apply_completion_triggers(&server, cx);
                }
                "textDocument/hover" => {
                    self.unregister_document_selector(server_id, &unreg.id, |registrations| {
                        &mut registrations.hover_selectors
                    });
                    self.unregister_dynamic_capability(
                        &server,
                        unreg,
                        cx,
                        |registrations| &mut registrations.hover,
                        |capabilities| &mut capabilities.hover_provider,
                    )?;
                }
                "textDocument/signatureHelp" => {
                    self.unregister_dynamic_capability(
                        &server,
                        unreg,
                        cx,
                        |registrations| &mut registrations.signature_help,
                        |capabilities| &mut capabilities.signature_help_provider,
                    )?;
                }
                "textDocument/semanticTokens" => {
                    if self
                        .unregister_dynamic_capability(
                            &server,
                            unreg,
                            cx,
                            |registrations| &mut registrations.semantic_tokens,
                            |capabilities| &mut capabilities.semantic_tokens_provider,
                        )?
                        .capability_changed()
                    {
                        self.refresh_semantic_tokens(server_id, cx);
                    }
                }
                "textDocument/didChange" | "textDocument/didSave" => {
                    let local = self
                        .as_local_mut()
                        .context("Expected LSP Store to be local")?;
                    if let Some(registrations) = local
                        .language_server_dynamic_registrations
                        .get_mut(&server_id)
                    {
                        if unreg.method == "textDocument/didChange" {
                            registrations.did_change.unregister(&unreg.id);
                        } else {
                            registrations.did_save.unregister(&unreg.id);
                        }
                    }
                }
                "textDocument/inlayHint" => {
                    if self
                        .unregister_dynamic_capability(
                            &server,
                            unreg,
                            cx,
                            |registrations| &mut registrations.inlay_hint,
                            |capabilities| &mut capabilities.inlay_hint_provider,
                        )?
                        .capability_changed()
                    {
                        self.refresh_inlay_hints(server_id, cx);
                    }
                }
                "textDocument/documentSymbol" => {
                    if self
                        .unregister_dynamic_capability(
                            &server,
                            unreg,
                            cx,
                            |registrations| &mut registrations.document_symbol,
                            |capabilities| &mut capabilities.document_symbol_provider,
                        )?
                        .capability_changed()
                    {
                        self.refresh_document_symbols(Some(server_id), cx);
                    }
                }
                "textDocument/codeLens" => {
                    if self
                        .unregister_dynamic_capability(
                            &server,
                            unreg,
                            cx,
                            |registrations| &mut registrations.code_lens,
                            |capabilities| &mut capabilities.code_lens_provider,
                        )?
                        .capability_changed()
                    {
                        self.refresh_code_lens(Some(server_id), cx);
                    }
                }
                "textDocument/diagnostic" => {
                    if self
                        .unregister_dynamic_capability(
                            &server,
                            unreg,
                            cx,
                            |registrations| &mut registrations.diagnostics,
                            |capabilities| &mut capabilities.diagnostic_provider,
                        )?
                        .removed()
                    {
                        let local = self
                            .as_local_mut()
                            .context("Expected LSP Store to be local")?;
                        let state = local
                            .language_servers
                            .get_mut(&server_id)
                            .context("Could not obtain Language Servers state")?;
                        if let LanguageServerState::Running {
                            workspace_diagnostics_refresh_tasks,
                            ..
                        } = state
                        {
                            workspace_diagnostics_refresh_tasks.remove(&Some(unreg.id.clone()));
                        }

                        self.clear_unregistered_diagnostics(
                            server_id,
                            SharedString::from(unreg.id.clone()),
                            cx,
                        )?;
                    }
                }
                "textDocument/documentColor" => {
                    if self
                        .unregister_dynamic_capability(
                            &server,
                            unreg,
                            cx,
                            |registrations| &mut registrations.color,
                            |capabilities| &mut capabilities.color_provider,
                        )?
                        .capability_changed()
                    {
                        self.refresh_document_colors(Some(server_id), cx);
                    }
                }
                "textDocument/foldingRange" => {
                    if self
                        .unregister_dynamic_capability(
                            &server,
                            unreg,
                            cx,
                            |registrations| &mut registrations.folding_range,
                            |capabilities| &mut capabilities.folding_range_provider,
                        )?
                        .capability_changed()
                    {
                        self.refresh_folding_ranges(Some(server_id), cx);
                    }
                }
                "textDocument/documentLink" => {
                    if self
                        .unregister_dynamic_capability(
                            &server,
                            unreg,
                            cx,
                            |registrations| &mut registrations.document_link,
                            |capabilities| &mut capabilities.document_link_provider,
                        )?
                        .capability_changed()
                    {
                        self.refresh_document_links(Some(server_id), cx);
                    }
                }
                _ => log::warn!("unhandled capability unregistration: {unreg:?}"),
            }
        }

        Ok(())
    }
}

fn parse_document_selector(
    options: Option<&serde_json::Value>,
) -> Result<Option<lsp::DocumentSelector>> {
    Ok(options
        .and_then(|options| options.get("documentSelector"))
        .cloned()
        .map(serde_json::from_value::<Option<lsp::DocumentSelector>>)
        .transpose()?
        .flatten())
}

// Registration with registerOptions as null, should fallback to true.
// https://github.com/microsoft/vscode-languageserver-node/blob/d90a87f9557a0df9142cfb33e251cfa6fe27d970/client/src/common/client.ts#L2133
fn parse_register_capabilities<T: serde::de::DeserializeOwned>(
    register_options: Option<serde_json::Value>,
) -> Result<OneOf<bool, T>> {
    Ok(match register_options {
        Some(options) => OneOf::Right(serde_json::from_value::<T>(options)?),
        None => OneOf::Left(true),
    })
}
