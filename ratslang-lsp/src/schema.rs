use std::collections::HashMap;

#[derive(Debug, Clone, Default)]
pub struct Schema {
    pub properties: HashMap<String, SchemaProperty>,
    pub namespaces: HashMap<String, SchemaNamespace>,
}

#[derive(Debug, Clone, Default)]
pub struct SchemaProperty {
    pub value_type: Option<String>,
    pub description: Option<String>,
    pub default: Option<String>,
    pub recommended: bool,
    pub options: Vec<String>,
}

#[derive(Debug, Clone, Default)]
pub struct SchemaNamespace {
    pub description: Option<String>,
    pub properties: HashMap<String, SchemaProperty>,
    pub namespaces: HashMap<String, SchemaNamespace>,
}

impl Schema {
    pub fn from_ratslang_source(source: &str) -> Self {
        let mut schema = Schema::default();
        let mut current_annotations: Vec<SchemaAnnotation> = Vec::new();
        let mut namespace_stack: Vec<String> = Vec::new();
        let mut brace_depth: usize = 0;

        for line in source.lines() {
            let trimmed = line.trim();

            if trimmed.starts_with('#') {
                let comment = trimmed.trim_start_matches('#').trim();
                if let Some(annotation) = SchemaAnnotation::parse(comment) {
                    current_annotations.push(annotation);
                }
                continue;
            }

            if trimmed.is_empty() {
                continue;
            }

            if !current_annotations.is_empty() {
                if trimmed.ends_with('{') {
                    let name = trimmed.trim_end_matches('{').trim();
                    let full_path = if namespace_stack.is_empty() {
                        name.to_string()
                    } else {
                        format!("{}.{}", namespace_stack.join("."), name)
                    };
                    schema.apply_annotations_to_namespace(&full_path, &current_annotations);
                    namespace_stack.push(name.to_string());
                    brace_depth += 1;
                } else if let Some(eq_pos) = trimmed.find('=') {
                    let name = trimmed[..eq_pos].trim();
                    let full_path = if namespace_stack.is_empty() {
                        name.to_string()
                    } else {
                        format!("{}.{}", namespace_stack.join("."), name)
                    };
                    schema.apply_annotations_to_property(&full_path, &current_annotations);
                }
                current_annotations.clear();
            } else if trimmed.ends_with('{') {
                let name = trimmed.trim_end_matches('{').trim();
                namespace_stack.push(name.to_string());
                brace_depth += 1;
            }

            if trimmed.starts_with('}') {
                namespace_stack.pop();
                brace_depth = brace_depth.saturating_sub(1);
            }
        }

        schema
    }

    fn apply_annotations_to_property(&mut self, path: &str, annotations: &[SchemaAnnotation]) {
        let parts: Vec<&str> = path.split('.').collect();

        if parts.len() == 1 {
            let mut prop = SchemaProperty::default();
            for ann in annotations {
                match ann {
                    SchemaAnnotation::Type(t) => prop.value_type = Some(t.clone()),
                    SchemaAnnotation::Description(d) => prop.description = Some(d.clone()),
                    SchemaAnnotation::Default(d) => prop.default = Some(d.clone()),
                    SchemaAnnotation::Recommended => prop.recommended = true,
                    SchemaAnnotation::Options(opts) => prop.options = opts.clone(),
                }
            }
            self.properties.insert(path.to_string(), prop);
        } else {
            let ns_name = parts[0];
            let rest = parts[1..].join(".");
            let ns = self.namespaces.entry(ns_name.to_string()).or_default();
            Self::apply_property_to_namespace(ns, &rest, annotations);
        }
    }

    fn apply_property_to_namespace(
        ns: &mut SchemaNamespace,
        path: &str,
        annotations: &[SchemaAnnotation],
    ) {
        let parts: Vec<&str> = path.split('.').collect();

        if parts.len() == 1 {
            let mut prop = SchemaProperty::default();
            for ann in annotations {
                match ann {
                    SchemaAnnotation::Type(t) => prop.value_type = Some(t.clone()),
                    SchemaAnnotation::Description(d) => prop.description = Some(d.clone()),
                    SchemaAnnotation::Default(d) => prop.default = Some(d.clone()),
                    SchemaAnnotation::Recommended => prop.recommended = true,
                    SchemaAnnotation::Options(opts) => prop.options = opts.clone(),
                }
            }
            ns.properties.insert(path.to_string(), prop);
        } else {
            let next_ns = parts[0];
            let rest = parts[1..].join(".");
            let inner_ns = ns.namespaces.entry(next_ns.to_string()).or_default();
            Self::apply_property_to_namespace(inner_ns, &rest, annotations);
        }
    }

    fn apply_annotations_to_namespace(&mut self, path: &str, annotations: &[SchemaAnnotation]) {
        let parts: Vec<&str> = path.split('.').collect();

        if parts.len() == 1 {
            let ns = self.namespaces.entry(path.to_string()).or_default();
            for ann in annotations {
                if let SchemaAnnotation::Description(d) = ann {
                    ns.description = Some(d.clone());
                }
            }
        } else {
            let first = parts[0];
            let rest = parts[1..].join(".");
            let ns = self.namespaces.entry(first.to_string()).or_default();
            Self::apply_namespace_recursive(ns, &rest, annotations);
        }
    }

    fn apply_namespace_recursive(
        ns: &mut SchemaNamespace,
        path: &str,
        annotations: &[SchemaAnnotation],
    ) {
        let parts: Vec<&str> = path.split('.').collect();

        if parts.len() == 1 {
            let inner_ns = ns.namespaces.entry(path.to_string()).or_default();
            for ann in annotations {
                if let SchemaAnnotation::Description(d) = ann {
                    inner_ns.description = Some(d.clone());
                }
            }
        } else {
            let first = parts[0];
            let rest = parts[1..].join(".");
            let inner_ns = ns.namespaces.entry(first.to_string()).or_default();
            Self::apply_namespace_recursive(inner_ns, &rest, annotations);
        }
    }

    pub fn get_property(&self, path: &str) -> Option<&SchemaProperty> {
        let parts: Vec<&str> = path.split('.').collect();
        self.get_property_recursive(&parts, &self.properties, &self.namespaces)
    }

    fn get_property_recursive<'a>(
        &'a self,
        parts: &[&str],
        props: &'a HashMap<String, SchemaProperty>,
        nss: &'a HashMap<String, SchemaNamespace>,
    ) -> Option<&'a SchemaProperty> {
        if parts.is_empty() {
            return None;
        }

        if parts.len() == 1 {
            return props.get(parts[0]);
        }

        if let Some(ns) = nss.get(parts[0]) {
            self.get_property_recursive(&parts[1..], &ns.properties, &ns.namespaces)
        } else {
            None
        }
    }

    pub fn get_children_at_path(&self, path: &str) -> (Vec<&str>, Vec<&str>) {
        let parts: Vec<&str> = if path.is_empty() {
            Vec::new()
        } else {
            path.split('.').collect()
        };
        self.get_children_recursive(&parts, &self.properties, &self.namespaces)
    }

    fn get_children_recursive<'a>(
        &'a self,
        parts: &[&str],
        props: &'a HashMap<String, SchemaProperty>,
        nss: &'a HashMap<String, SchemaNamespace>,
    ) -> (Vec<&'a str>, Vec<&'a str>) {
        if parts.is_empty() {
            return (
                props.keys().map(|s| s.as_str()).collect(),
                nss.keys().map(|s| s.as_str()).collect(),
            );
        }

        if let Some(ns) = nss.get(parts[0]) {
            self.get_children_recursive(&parts[1..], &ns.properties, &ns.namespaces)
        } else {
            (Vec::new(), Vec::new())
        }
    }

    pub fn get_recommended_properties(&self) -> Vec<(String, String)> {
        let mut result = Vec::new();
        self.collect_recommended_recursive(&self.properties, &self.namespaces, "", &mut result);
        result
    }

    fn collect_recommended_recursive(
        &self,
        props: &HashMap<String, SchemaProperty>,
        nss: &HashMap<String, SchemaNamespace>,
        prefix: &str,
        result: &mut Vec<(String, String)>,
    ) {
        for (name, prop) in props {
            if prop.recommended {
                let full_path = if prefix.is_empty() {
                    name.clone()
                } else {
                    format!("{}.{}", prefix, name)
                };
                let default = prop.default.clone().unwrap_or_default();
                result.push((full_path, default));
            }
        }

        for (name, ns) in nss {
            let new_prefix = if prefix.is_empty() {
                name.clone()
            } else {
                format!("{}.{}", prefix, name)
            };
            self.collect_recommended_recursive(&ns.properties, &ns.namespaces, &new_prefix, result);
        }
    }
}

#[derive(Debug, Clone)]
enum SchemaAnnotation {
    Type(String),
    Description(String),
    Default(String),
    Recommended,
    Options(Vec<String>),
}

impl SchemaAnnotation {
    fn parse(comment: &str) -> Option<Self> {
        if !comment.starts_with('@') {
            return None;
        }

        let content = comment[1..].trim();

        if content == "recommended" {
            return Some(SchemaAnnotation::Recommended);
        }

        if let Some(eq_pos) = content.find('=') {
            let key = content[..eq_pos].trim();
            let value = content[eq_pos + 1..].trim().trim_matches('"');

            return match key {
                "type" => Some(SchemaAnnotation::Type(value.to_string())),
                "description" => Some(SchemaAnnotation::Description(value.to_string())),
                "default" => Some(SchemaAnnotation::Default(value.to_string())),
                "options" => {
                    let opts: Vec<String> = value
                        .trim_matches(|c| c == '[' || c == ']')
                        .split(',')
                        .map(|s| s.trim().trim_matches('"').to_string())
                        .collect();
                    Some(SchemaAnnotation::Options(opts))
                }
                _ => None,
            };
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_schema_from_source() {
        let source = r#"
# @type = string
# @description = "ROS topic for the lidar"
# @default = "/ouster/points"
topic = "/ouster/points"

# @recommended
# @type = number
resolution = 50

sensor {
    # @type = length
    range = 100m
}
"#;
        let schema = Schema::from_ratslang_source(source);

        assert!(schema.properties.contains_key("topic"));
        let topic = schema.properties.get("topic").unwrap();
        assert_eq!(topic.value_type, Some("string".to_string()));
        assert_eq!(
            topic.description,
            Some("ROS topic for the lidar".to_string())
        );

        assert!(schema.namespaces.contains_key("sensor"));
    }

    #[test]
    fn test_get_property_at_path() {
        let source = r#"
sensor {
    # @type = length
    range = 100m
}
"#;
        let schema = Schema::from_ratslang_source(source);

        let prop = schema.get_property("sensor.range");
        assert!(prop.is_some(), "Expected to find sensor.range property");
        assert_eq!(prop.unwrap().value_type, Some("length".to_string()));
    }
}
