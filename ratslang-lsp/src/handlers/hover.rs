use crate::{DocumentState, Schema};
use tower_lsp_server::ls_types::*;

pub fn handle_hover(
    doc: &DocumentState,
    position: Position,
    schema: Option<&Schema>,
) -> Option<Hover> {
    let line_idx = position.line as usize;
    let line = doc.content.lines().nth(line_idx)?;
    let col = position.character as usize;

    let word_start = line[..col.min(line.len())]
        .char_indices()
        .rev()
        .take_while(|(_i, c)| c.is_alphanumeric() || *c == '_' || *c == '.')
        .last()
        .map(|(i, _)| i)
        .unwrap_or(col);

    let word_end = line[col.min(line.len())..]
        .char_indices()
        .take_while(|(_, c)| c.is_alphanumeric() || *c == '_' || *c == '.')
        .last()
        .map(|(i, c)| col + i + c.len_utf8())
        .unwrap_or(col);

    if word_start >= word_end {
        return None;
    }

    let word = &line[word_start..word_end.min(line.len())];

    if let Some(schema) = schema {
        if let Some(prop) = schema.get_property(word) {
            let mut contents = String::new();

            if let Some(desc) = &prop.description {
                contents.push_str(&format!("{}\n\n", desc));
            }

            if let Some(t) = &prop.value_type {
                contents.push_str(&format!("**Type:** `{}`\n\n", t));
            }

            if prop.recommended {
                contents.push_str("**Recommended**\n\n");
            }

            if let Some(default) = &prop.default {
                contents.push_str(&format!("**Default:** `{}`", default));
            }

            return Some(Hover {
                contents: HoverContents::Markup(MarkupContent {
                    kind: MarkupKind::Markdown,
                    value: contents,
                }),
                range: Some(Range {
                    start: Position {
                        line: position.line,
                        character: word_start as u32,
                    },
                    end: Position {
                        line: position.line,
                        character: word_end as u32,
                    },
                }),
            });
        }
    }

    None
}
