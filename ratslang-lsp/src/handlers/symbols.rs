use crate::DocumentState;
use tower_lsp_server::ls_types::*;

#[allow(deprecated)]
pub fn handle_document_symbols(doc: &DocumentState) -> Vec<SymbolInformation> {
    let mut symbols = Vec::new();

    for assignment in &doc.symbols.assignments {
        symbols.push(SymbolInformation {
            name: assignment.full_path.clone(),
            kind: if assignment.is_internal {
                SymbolKind::CONSTANT
            } else {
                SymbolKind::PROPERTY
            },
            location: Location {
                uri: doc.uri.clone(),
                range: assignment.range,
            },
            container_name: None,
            tags: None,
            deprecated: None,
        });
    }

    for (name, ns) in &doc.symbols.namespaces {
        symbols.push(SymbolInformation {
            name: name.clone(),
            kind: SymbolKind::NAMESPACE,
            location: Location {
                uri: doc.uri.clone(),
                range: ns.range,
            },
            container_name: None,
            tags: None,
            deprecated: None,
        });
    }

    symbols
}
