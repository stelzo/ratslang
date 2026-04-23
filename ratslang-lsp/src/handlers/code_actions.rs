use crate::{DocumentState, Schema};
use tower_lsp_server::ls_types::*;

pub fn handle_code_actions(
    doc: &DocumentState,
    range: Range,
    schema: Option<&Schema>,
) -> Vec<CodeActionOrCommand> {
    let mut actions = Vec::new();

    if let Some(schema) = schema {
        let recommended = schema.get_recommended_properties();
        let missing: Vec<_> = recommended
            .into_iter()
            .filter(|(path, _)| !doc.symbols.assignments.iter().any(|a| &a.full_path == path))
            .collect();

        if !missing.is_empty() {
            for (path, default) in &missing {
                let parts: Vec<&str> = path.split('.').collect();
                let name = parts.last().copied().unwrap_or(&path.as_str());

                let edit = WorkspaceEdit {
                    changes: Some(
                        vec![(
                            doc.uri.clone(),
                            vec![TextEdit {
                                range,
                                new_text: format!("{} = {}\n", name, default),
                            }],
                        )]
                        .into_iter()
                        .collect(),
                    ),
                    ..Default::default()
                };

                actions.push(CodeActionOrCommand::CodeAction(CodeAction {
                    title: format!("Insert default for {}", path),
                    kind: Some(CodeActionKind::QUICKFIX),
                    edit: Some(edit),
                    ..Default::default()
                }));
            }

            actions.push(CodeActionOrCommand::CodeAction(CodeAction {
                title: "Insert all recommended properties".to_string(),
                kind: Some(CodeActionKind::QUICKFIX),
                command: Some(Command {
                    title: "Insert recommended".to_string(),
                    command: "ratslang.insertRecommended".to_string(),
                    arguments: Some(vec![
                        serde_json::to_value(doc.uri.clone()).unwrap_or_default(),
                        serde_json::to_value(missing).unwrap_or_default(),
                    ]),
                }),
                ..Default::default()
            }));
        }
    }

    actions
}
