mod handlers;
mod schema;

use std::{collections::HashMap, sync::Arc};
use tower_lsp_server::{LspService, Server, ls_types::*};
use dashmap::DashMap;
use parking_lot::RwLock;
use tree_sitter::Parser;

pub use schema::Schema;

pub struct Backend {
    db: DashMap<Uri, DocumentState>,
    workspace_roots: RwLock<Vec<Uri>>,
    schema_cache: DashMap<String, Arc<Schema>>,
}

#[derive(Debug, Clone)]
pub struct DocumentState {
    pub uri: Uri,
    pub content: String,
    pub version: i32,
    pub tree: tree_sitter::Tree,
    pub symbols: SymbolTable,
    pub schema_ref: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub struct SymbolTable {
    pub namespaces: HashMap<String, NamespaceInfo>,
    pub assignments: Vec<AssignmentInfo>,
}

#[derive(Debug, Clone)]
pub struct NamespaceInfo {
    pub name: String,
    pub range: Range,
}

#[derive(Debug, Clone)]
pub struct AssignmentInfo {
    pub name: String,
    pub full_path: String,
    pub range: Range,
    pub value_type: Option<ValueType>,
    pub is_internal: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ValueType {
    Number,
    Float,
    String,
    Boolean,
    Length,
    Time,
    Range { from_type: Option<Box<ValueType>>, to_type: Option<Box<ValueType>> },
    Array(Box<ValueType>),
    Path,
    Unknown,
}

impl Backend {
    pub fn new() -> Self {
        Self {
            db: DashMap::new(),
            workspace_roots: RwLock::new(Vec::new()),
            schema_cache: DashMap::new(),
        }
    }

    pub fn add_workspace_root(&self, root: Uri) {
        self.workspace_roots.write().push(root);
    }

    pub fn get_document(&self, uri: &Uri) -> Option<DocumentState> {
        self.db.get(uri).map(|v| v.clone())
    }

    pub fn update_document(&self, uri: Uri, state: DocumentState) {
        self.db.insert(uri, state);
    }

    pub fn remove_document(&self, uri: &Uri) {
        self.db.remove(uri);
    }

    pub fn get_schema(&self, schema_ref: &str) -> Option<Arc<Schema>> {
        self.schema_cache.get(schema_ref).map(|v| v.clone())
    }

    pub fn cache_schema(&self, schema_ref: String, schema: Arc<Schema>) {
        self.schema_cache.insert(schema_ref, schema);
    }
}

impl Default for Backend {
    fn default() -> Self {
        Self::new()
    }
}

impl DocumentState {
    pub fn new(uri: Uri, content: String, version: i32) -> Self {
        let mut parser = Parser::new();
        parser.set_language(&tree_sitter_ratslang::LANGUAGE.into())
            .expect("Failed to set language");
        
        let tree = parser.parse(&content, None).expect("Failed to parse");
        let symbols = SymbolTable::from_tree(&tree, &content);
        let schema_ref = symbols.find_schema_ref(&content);
        
        Self { uri, content, version, tree, symbols, schema_ref }
    }

    pub fn update(&mut self, content: String, version: i32) {
        let mut parser = Parser::new();
        parser.set_language(&tree_sitter_ratslang::LANGUAGE.into())
            .expect("Failed to set language");
        
        let tree = parser.parse(&content, None).expect("Failed to parse");
        let symbols = SymbolTable::from_tree(&tree, &content);
        let schema_ref = symbols.find_schema_ref(&content);
        
        self.content = content;
        self.version = version;
        self.tree = tree;
        self.symbols = symbols;
        self.schema_ref = schema_ref;
    }

    pub fn byte_to_position(byte: usize, content: &str) -> Position {
        let text = &content[..byte.min(content.len())];
        let lines: Vec<&str> = text.lines().collect();
        Position {
            line: (lines.len().saturating_sub(1)) as u32,
            character: lines.last().map(|l| l.len() as u32).unwrap_or(0),
        }
    }

    pub fn collect_diagnostics(&self) -> Vec<Diagnostic> {
        let mut diagnostics = Vec::new();
        self.collect_parse_errors(&mut diagnostics);
        self.collect_type_errors(&mut diagnostics);
        diagnostics
    }

    fn collect_parse_errors(&self, diagnostics: &mut Vec<Diagnostic>) {
        fn find_errors(node: tree_sitter::Node, diagnostics: &mut Vec<Diagnostic>, content: &str) {
            if node.is_error() || node.is_missing() {
                let start = DocumentState::byte_to_position(node.start_byte(), content);
                let end = DocumentState::byte_to_position(node.end_byte(), content);
                diagnostics.push(Diagnostic {
                    range: Range { start, end },
                    severity: Some(DiagnosticSeverity::ERROR),
                    message: if node.is_missing() {
                        format!("Missing: {}", node.kind())
                    } else {
                        "Syntax error".to_string()
                    },
                    source: Some("ratslang".to_string()),
                    ..Default::default()
                });
            }
            for i in 0..node.child_count() {
                if let Some(child) = node.child(i) {
                    find_errors(child, diagnostics, content);
                }
            }
        }
        find_errors(self.tree.root_node(), diagnostics, &self.content);
    }

    fn collect_type_errors(&self, diagnostics: &mut Vec<Diagnostic>) {
        for assignment in &self.symbols.assignments {
            if let Some(value_type) = &assignment.value_type {
                if let ValueType::Range { from_type, to_type } = value_type {
                    if let (Some(from), Some(to)) = (from_type, to_type) {
                        if !Self::types_compatible_for_range(from, to) {
                            diagnostics.push(Diagnostic {
                                range: assignment.range,
                                severity: Some(DiagnosticSeverity::ERROR),
                                message: format!("Range bounds have incompatible types: {:?} and {:?}", from, to),
                                source: Some("ratslang".to_string()),
                                ..Default::default()
                            });
                        }
                    }
                }
            }
        }
    }

    fn types_compatible_for_range(a: &ValueType, b: &ValueType) -> bool {
        matches!(
            (a, b),
            (ValueType::Number, ValueType::Number)
                | (ValueType::Float, ValueType::Float)
                | (ValueType::Length, ValueType::Length)
                | (ValueType::Time, ValueType::Time)
        )
    }
}

impl SymbolTable {
    pub fn from_tree(tree: &tree_sitter::Tree, content: &str) -> Self {
        let mut symbols = SymbolTable::default();
        let root = tree.root_node();
        symbols.walk_statements(root, content, "");
        symbols
    }

    fn walk_statements(&mut self, node: tree_sitter::Node, content: &str, current_ns: &str) {
        for i in 0..node.child_count() {
            if let Some(child) = node.child(i) {
                match child.kind() {
                    "namespace_block" => self.process_namespace_block(child, content, current_ns),
                    "assignment" => self.process_assignment(child, content, current_ns),
                    _ => {}
                }
            }
        }
    }

    fn process_namespace_block(&mut self, node: tree_sitter::Node, content: &str, current_ns: &str) {
        if let Some(name_node) = node.child(0) {
            let name = name_node.utf8_text(content.as_bytes()).unwrap_or("").to_string();
            let full_path = if current_ns.is_empty() {
                name.clone()
            } else {
                format!("{}.{}", current_ns, name)
            };

            let start = DocumentState::byte_to_position(name_node.start_byte(), content);
            let end = DocumentState::byte_to_position(node.end_byte(), content);
            
            self.namespaces.insert(full_path.clone(), NamespaceInfo {
                name: name.clone(),
                range: Range { start, end },
            });

            if let Some(block) = node.child_by_field_name("body") {
                self.walk_statements(block, content, &full_path);
            }
        }
    }

    fn process_assignment(&mut self, node: tree_sitter::Node, content: &str, current_ns: &str) {
        if let Some(lhs_node) = node.child(0) {
            let name = lhs_node.utf8_text(content.as_bytes()).unwrap_or("").to_string();
            let is_internal = name.starts_with('_');
            
            let full_path = if current_ns.is_empty() {
                name.clone()
            } else {
                format!("{}.{}", current_ns, name)
            };

            let start = DocumentState::byte_to_position(lhs_node.start_byte(), content);
            let end = DocumentState::byte_to_position(node.end_byte(), content);
            
            let value_type = self.infer_value_type(node, content);

            self.assignments.push(AssignmentInfo {
                name,
                full_path,
                range: Range { start, end },
                value_type,
                is_internal,
            });
        }
    }

    fn infer_value_type(&self, node: tree_sitter::Node, content: &str) -> Option<ValueType> {
        for i in 0..node.child_count() {
            if let Some(child) = node.child(i) {
                if child.kind() == "=" {
                    if let Some(rhs) = node.child(i + 1) {
                        return Some(self.infer_node_type(rhs, content));
                    }
                }
            }
        }
        None
    }

    fn infer_node_type(&self, node: tree_sitter::Node, content: &str) -> ValueType {
        match node.kind() {
            "number" => {
                let text = node.utf8_text(content.as_bytes()).unwrap_or("");
                if text.contains('.') { ValueType::Float } else { ValueType::Number }
            }
            "time_quantity" => ValueType::Time,
            "length_quantity" => ValueType::Length,
            "string" => ValueType::String,
            "boolean" => ValueType::Boolean,
            "range" => {
                let mut from_type = None;
                let mut to_type = None;
                for i in 0..node.child_count() {
                    if let Some(child) = node.child(i) {
                        match child.kind() {
                            "number" | "time_quantity" | "length_quantity" => {
                                let t = self.infer_node_type(child, content);
                                if from_type.is_none() {
                                    from_type = Some(Box::new(t));
                                } else {
                                    to_type = Some(Box::new(t));
                                }
                            }
                            _ => {}
                        }
                    }
                }
                ValueType::Range { from_type, to_type }
            }
            "array" => ValueType::Array(Box::new(ValueType::Unknown)),
            "path" => ValueType::Path,
            _ => ValueType::Unknown,
        }
    }

    pub fn find_schema_ref(&self, content: &str) -> Option<String> {
        for assignment in &self.assignments {
            if assignment.name == "_schema" {
                let line_idx = assignment.range.start.line as usize;
                if let Some(line) = content.lines().nth(line_idx) {
                    if let Some(eq_pos) = line.find('=') {
                        let value = line[eq_pos + 1..].trim().trim_matches('"');
                        return Some(value.to_string());
                    }
                }
            }
        }
        None
    }

    pub fn get_namespaces_at_path(&self, path: &str) -> Vec<String> {
        self.namespaces
            .keys()
            .filter(|k| {
                if path.is_empty() {
                    !k.contains('.')
                } else {
                    k.starts_with(&format!("{}.", path)) 
                        && k.matches('.').count() == path.matches('.').count() + 1
                }
            })
            .map(|s| {
                let parts: Vec<&str> = s.split('.').collect();
                parts.last().unwrap_or(&"").to_string()
            })
            .collect()
    }
}

pub struct RatslangBackend {
    backend: Backend,
    client: tower_lsp_server::Client,
}

impl RatslangBackend {
    pub fn new(client: tower_lsp_server::Client) -> Self {
        Self { backend: Backend::new(), client }
    }
}

impl tower_lsp_server::LanguageServer for RatslangBackend {
    async fn initialize(&self, params: InitializeParams) -> tower_lsp_server::jsonrpc::Result<InitializeResult> {
        if let Some(folders) = params.workspace_folders {
            for folder in folders {
                self.backend.add_workspace_root(folder.uri);
            }
        }

        Ok(InitializeResult {
            server_info: Some(ServerInfo {
                name: "ratslang-lsp".to_string(),
                version: Some(env!("CARGO_PKG_VERSION").to_string()),
            }),
            capabilities: ServerCapabilities {
                text_document_sync: Some(TextDocumentSyncCapability::Kind(TextDocumentSyncKind::INCREMENTAL)),
                completion_provider: Some(CompletionOptions {
                    resolve_provider: Some(false),
                    trigger_characters: Some(vec![".".to_string()]),
                    ..Default::default()
                }),
                code_action_provider: Some(CodeActionProviderCapability::Simple(true)),
                hover_provider: Some(HoverProviderCapability::Simple(true)),
                document_symbol_provider: Some(OneOf::Left(true)),
                ..Default::default()
            },
            offset_encoding: None,
        })
    }

    async fn initialized(&self, _: InitializedParams) {
        self.client.log_message(MessageType::INFO, "ratslang-lsp initialized").await;
    }

    async fn shutdown(&self) -> tower_lsp_server::jsonrpc::Result<()> {
        Ok(())
    }

    async fn did_open(&self, params: DidOpenTextDocumentParams) {
        let uri = params.text_document.uri;
        let content = params.text_document.text;
        let version = params.text_document.version;
        
        let doc = DocumentState::new(uri.clone(), content, version);
        
        if let Some(schema_ref) = &doc.schema_ref {
            self.load_schema(&uri, schema_ref).await;
        }
        
        let diagnostics = doc.collect_diagnostics();
        self.backend.update_document(uri.clone(), doc);
        self.client.publish_diagnostics(uri, diagnostics, None).await;
    }

    async fn did_change(&self, params: DidChangeTextDocumentParams) {
        let uri = params.text_document.uri;
        let version = params.text_document.version;
        
        if let Some(change) = params.content_changes.last() {
            let doc = if let Some(mut existing) = self.backend.get_document(&uri) {
                existing.update(change.text.clone(), version);
                existing
            } else {
                DocumentState::new(uri.clone(), change.text.clone(), version)
            };
            
            if let Some(schema_ref) = doc.schema_ref.clone() {
                self.load_schema(&uri, &schema_ref).await;
            }
            
            let diagnostics = doc.collect_diagnostics();
            self.backend.update_document(uri.clone(), doc);
            self.client.publish_diagnostics(uri, diagnostics, None).await;
        }
    }

    async fn did_close(&self, params: DidCloseTextDocumentParams) {
        self.backend.remove_document(&params.text_document.uri);
    }

    async fn completion(&self, params: CompletionParams) -> tower_lsp_server::jsonrpc::Result<Option<CompletionResponse>> {
        let uri = params.text_document_position.text_document.uri;
        let position = params.text_document_position.position;

        if let Some(doc) = self.backend.get_document(&uri) {
            let schema = doc.schema_ref.as_ref()
                .and_then(|s| self.backend.get_schema(s));
            let items = handlers::completion::handle_completion(&doc, position, schema.as_ref().map(|v| v.as_ref()));
            return Ok(Some(CompletionResponse::Array(items)));
        }
        Ok(None)
    }

    async fn hover(&self, params: HoverParams) -> tower_lsp_server::jsonrpc::Result<Option<Hover>> {
        let uri = params.text_document_position_params.text_document.uri;
        let position = params.text_document_position_params.position;

        if let Some(doc) = self.backend.get_document(&uri) {
            let schema = doc.schema_ref.as_ref()
                .and_then(|s| self.backend.get_schema(s));
            return Ok(handlers::hover::handle_hover(&doc, position, schema.as_ref().map(|v| v.as_ref())));
        }
        Ok(None)
    }

    async fn code_action(&self, params: CodeActionParams) -> tower_lsp_server::jsonrpc::Result<Option<CodeActionResponse>> {
        let uri = params.text_document.uri;

        if let Some(doc) = self.backend.get_document(&uri) {
            let schema = doc.schema_ref.as_ref()
                .and_then(|s| self.backend.get_schema(s));
            let actions = handlers::code_actions::handle_code_actions(&doc, params.range, schema.as_ref().map(|v| v.as_ref()));
            return Ok(Some(actions));
        }
        Ok(None)
    }

    async fn document_symbol(&self, params: DocumentSymbolParams) -> tower_lsp_server::jsonrpc::Result<Option<DocumentSymbolResponse>> {
        if let Some(doc) = self.backend.get_document(&params.text_document.uri) {
            let symbols = handlers::symbols::handle_document_symbols(&doc);
            return Ok(Some(DocumentSymbolResponse::Flat(symbols)));
        }
        Ok(None)
    }
}

impl RatslangBackend {
    async fn load_schema(&self, doc_uri: &Uri, schema_ref: &str) {
        if self.backend.get_schema(schema_ref).is_some() {
            return;
        }

        let schema = if schema_ref.starts_with("http://") || schema_ref.starts_with("https://") {
            match reqwest::get(schema_ref).await {
                Ok(resp) => match resp.text().await {
                    Ok(content) => Some(Schema::from_ratslang_source(&content)),
                    Err(e) => {
                        self.client.log_message(MessageType::WARNING, format!("Failed to load schema: {}", e)).await;
                        None
                    }
                }
                Err(e) => {
                    self.client.log_message(MessageType::WARNING, format!("Failed to fetch schema: {}", e)).await;
                    None
                }
            }
        } else {
            doc_uri.to_file_path()
                .and_then(|path| path.parent().map(|p| p.to_path_buf()))
                .and_then(|dir| {
                    let schema_path = dir.join(schema_ref);
                    std::fs::read_to_string(schema_path).ok()
                })
                .map(|content| Schema::from_ratslang_source(&content))
        };

        if let Some(schema) = schema {
            self.backend.cache_schema(schema_ref.to_string(), Arc::new(schema));
            self.client.log_message(MessageType::INFO, format!("Loaded schema: {}", schema_ref)).await;
        }
    }
}

pub async fn run_server() {
    let stdin = tokio::io::stdin();
    let stdout = tokio::io::stdout();
    
    let (service, socket) = LspService::new(RatslangBackend::new);
    Server::new(stdin, stdout, socket).serve(service).await;
}
