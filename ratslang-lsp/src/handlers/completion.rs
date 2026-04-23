use crate::{DocumentState, Schema};
use tower_lsp_server::ls_types::*;

pub fn handle_completion(
    doc: &DocumentState,
    position: Position,
    schema: Option<&Schema>,
) -> Vec<CompletionItem> {
    let mut items = Vec::new();
    let prefix = get_completion_prefix(doc, position);

    if let Some(schema) = schema {
        let (props, namespaces) = schema.get_children_at_path(&prefix);
        for prop in props {
            if let Some(prop_info) = schema.get_property(&if prefix.is_empty() {
                prop.to_string()
            } else {
                format!("{}.{}", prefix, prop)
            }) {
                items.push(schema_prop_to_completion(prop, prop_info, &prefix));
            }
        }
        for ns in namespaces {
            items.push(CompletionItem {
                label: ns.to_string(),
                kind: Some(CompletionItemKind::MODULE),
                detail: schema
                    .namespaces
                    .get(ns)
                    .and_then(|n| n.description.clone()),
                ..Default::default()
            });
        }
    }

    let in_file_items = get_in_file_completions(doc, &prefix);
    items.extend(in_file_items);

    items
}

fn get_completion_prefix(doc: &DocumentState, position: Position) -> String {
    let line_idx = position.line as usize;
    if let Some(line) = doc.content.lines().nth(line_idx) {
        let col = position.character as usize;
        let before_cursor: String = line.chars().take(col).collect();

        if before_cursor.ends_with('.') {
            if let Some(dot_pos) = before_cursor.rfind('.') {
                return before_cursor[..dot_pos].to_string();
            }
        } else if let Some(last_dot) = before_cursor.rfind('.') {
            return before_cursor[..last_dot].to_string();
        }
    }
    String::new()
}

fn get_in_file_completions(doc: &DocumentState, prefix: &str) -> Vec<CompletionItem> {
    let mut items = Vec::new();

    for assignment in &doc.symbols.assignments {
        if prefix.is_empty() {
            if !assignment.name.contains('.') && !assignment.is_internal {
                items.push(CompletionItem {
                    label: assignment.name.clone(),
                    kind: Some(CompletionItemKind::PROPERTY),
                    detail: assignment.value_type.as_ref().map(|t| format!("{:?}", t)),
                    ..Default::default()
                });
            }
        } else if assignment.full_path.starts_with(&format!("{}.", prefix)) {
            let remaining = &assignment.full_path[prefix.len() + 1..];
            if !remaining.contains('.') {
                items.push(CompletionItem {
                    label: remaining.to_string(),
                    kind: Some(CompletionItemKind::PROPERTY),
                    detail: assignment.value_type.as_ref().map(|t| format!("{:?}", t)),
                    ..Default::default()
                });
            }
        }
    }

    for ns_name in doc.symbols.get_namespaces_at_path(prefix) {
        items.push(CompletionItem {
            label: ns_name,
            kind: Some(CompletionItemKind::MODULE),
            ..Default::default()
        });
    }

    items
}

fn schema_prop_to_completion(
    name: &str,
    prop: &crate::schema::SchemaProperty,
    _prefix: &str,
) -> CompletionItem {
    let insert_text = if prop.recommended {
        if let Some(default) = &prop.default {
            format!("{} = {}", name, default)
        } else {
            format!("{} = ", name)
        }
    } else {
        name.to_string()
    };

    CompletionItem {
        label: name.to_string(),
        kind: Some(CompletionItemKind::PROPERTY),
        detail: prop.value_type.clone(),
        documentation: prop
            .description
            .as_ref()
            .map(|d| Documentation::String(d.clone())),
        insert_text: Some(insert_text),
        insert_text_format: Some(InsertTextFormat::PLAIN_TEXT),
        ..Default::default()
    }
}
