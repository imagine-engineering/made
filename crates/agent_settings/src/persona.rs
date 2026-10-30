use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt,
    path::{Component, Path},
};

use serde::Deserialize;

const MAX_MANIFEST_BYTES: usize = 1_048_576;
const MAX_FRONTMATTER_BYTES: usize = 1_048_576;
const MAX_BODY_BYTES: usize = 262_144;
const MAX_INSTRUCTIONS_BYTES: usize = 262_144;
const MAX_PERSONAS: usize = 256;

#[derive(Clone, PartialEq)]
pub struct PersonaPack {
    pub id: String,
    pub name: String,
    pub version: String,
    pub description: Option<String>,
    pub instructions: Option<String>,
    pub mcp_config_path: Option<String>,
    pub hooks_config_path: Option<String>,
    pub personas: Vec<Persona>,
}

impl fmt::Debug for PersonaPack {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PersonaPack")
            .field("id", &self.id)
            .field("name", &self.name)
            .field("version", &self.version)
            .field("description", &self.description)
            .field(
                "instructions",
                &self.instructions.as_ref().map(|_| "<redacted>"),
            )
            .field("mcp_config_path", &self.mcp_config_path)
            .field("hooks_config_path", &self.hooks_config_path)
            .field("personas", &self.personas)
            .finish()
    }
}

#[derive(Clone, PartialEq)]
pub struct Persona {
    pub source_path: String,
    pub name: String,
    pub display_name: String,
    pub description: String,
    pub avatar: Option<String>,
    pub version: String,
    pub author: Option<String>,
    pub skills: Vec<String>,
    pub mcp_servers: Vec<PersonaMcpServer>,
    pub hooks: Option<PersonaHooks>,
    pub prompt: String,
    pub behavior: PersonaBehavior,
}

impl fmt::Debug for Persona {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Persona")
            .field("source_path", &self.source_path)
            .field("name", &self.name)
            .field("display_name", &self.display_name)
            .field("description", &self.description)
            .field("avatar", &self.avatar)
            .field("version", &self.version)
            .field("author", &self.author)
            .field("skills", &self.skills)
            .field("mcp_servers", &self.mcp_servers)
            .field("hooks", &self.hooks)
            .field("prompt", &"<redacted>")
            .field("behavior", &self.behavior)
            .finish()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PersonaBehavior {
    pub model: Option<String>,
    pub runtime: Option<String>,
    pub temperature: Option<f64>,
    pub max_context_tokens: Option<u64>,
    pub subscribe: Option<Vec<String>>,
    pub triggers: Option<PersonaTriggers>,
    pub thread_replies: bool,
    pub broadcast_replies: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PersonaTriggers {
    pub mentions: bool,
    pub keywords: Vec<String>,
    pub all_messages: bool,
}

#[derive(Clone, PartialEq, Eq)]
pub struct PersonaMcpServer {
    pub name: String,
    pub command: String,
    pub args: Vec<String>,
    pub environment: BTreeMap<String, String>,
}

impl fmt::Debug for PersonaMcpServer {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PersonaMcpServer")
            .field("name", &self.name)
            .field("command", &"<redacted>")
            .field("arguments_len", &self.args.len())
            .field("environment", &self.environment.keys().collect::<Vec<_>>())
            .finish()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PersonaHooks {
    pub on_start: Option<String>,
    pub on_stop: Option<String>,
    pub on_message: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PersonaPackError {
    DuplicateSourcePath(String),
    InvalidPath(String),
    MissingFile(String),
    ManifestTooLarge,
    MalformedManifest(String),
    InvalidManifestField(&'static str),
    TooManyPersonas,
    PersonaFrontmatterTooLarge(String),
    PersonaBodyTooLarge(String),
    MalformedPersona { path: String, reason: String },
    DuplicatePersona(String),
    DuplicateMcpServer { persona: String, server: String },
    InstructionsTooLarge(String),
}

impl fmt::Display for PersonaPackError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateSourcePath(path) => write!(formatter, "duplicate source path: {path}"),
            Self::InvalidPath(path) => write!(formatter, "invalid pack-relative path: {path}"),
            Self::MissingFile(path) => write!(formatter, "pack file is missing: {path}"),
            Self::ManifestTooLarge => write!(formatter, "persona pack manifest is too large"),
            Self::MalformedManifest(reason) => {
                write!(formatter, "persona pack manifest is malformed: {reason}")
            }
            Self::InvalidManifestField(field) => {
                write!(formatter, "persona pack manifest has an invalid {field}")
            }
            Self::TooManyPersonas => write!(formatter, "persona pack contains too many personas"),
            Self::PersonaFrontmatterTooLarge(path) => {
                write!(formatter, "persona frontmatter is too large: {path}")
            }
            Self::PersonaBodyTooLarge(path) => {
                write!(formatter, "persona body is too large: {path}")
            }
            Self::MalformedPersona { path, reason } => {
                write!(formatter, "persona is malformed at {path}: {reason}")
            }
            Self::DuplicatePersona(name) => write!(formatter, "duplicate persona name: {name}"),
            Self::DuplicateMcpServer { persona, server } => {
                write!(
                    formatter,
                    "persona {persona} has duplicate MCP server {server}"
                )
            }
            Self::InstructionsTooLarge(path) => {
                write!(formatter, "pack instructions are too large: {path}")
            }
        }
    }
}

impl Error for PersonaPackError {}

#[derive(Deserialize)]
struct Manifest {
    id: Option<String>,
    name: Option<String>,
    version: Option<String>,
    description: Option<String>,
    #[serde(default)]
    personas: Vec<String>,
    pack_instructions: Option<String>,
    mcp_config: Option<String>,
    hooks_config: Option<String>,
    defaults: Option<BehavioralDefaults>,
}

#[derive(Clone, Default, Deserialize)]
struct BehavioralDefaults {
    model: Option<String>,
    temperature: Option<f64>,
    max_context_tokens: Option<u64>,
    #[serde(default)]
    subscribe: Option<Vec<String>>,
    #[serde(alias = "respond_to")]
    triggers: Option<TriggerInput>,
    thread_replies: Option<bool>,
    broadcast_replies: Option<bool>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PersonaFrontmatter {
    name: Option<String>,
    display_name: Option<String>,
    description: Option<String>,
    avatar: Option<String>,
    version: Option<String>,
    author: Option<String>,
    #[serde(default)]
    skills: Vec<String>,
    #[serde(default)]
    mcp_servers: Vec<McpServerInput>,
    #[serde(default)]
    subscribe: Option<Vec<String>>,
    #[serde(alias = "respond_to")]
    triggers: Option<TriggerInput>,
    model: Option<String>,
    runtime: Option<String>,
    temperature: Option<f64>,
    max_context_tokens: Option<u64>,
    thread_replies: Option<bool>,
    broadcast_replies: Option<bool>,
    hooks: Option<HooksInput>,
}

#[derive(Clone, Deserialize)]
struct TriggerInput {
    mentions: Option<bool>,
    #[serde(default)]
    keywords: Vec<String>,
    all_messages: Option<bool>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct McpServerInput {
    name: String,
    command: String,
    #[serde(default)]
    args: Vec<String>,
    #[serde(default)]
    env: BTreeMap<String, String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HooksInput {
    on_start: Option<String>,
    on_stop: Option<String>,
    on_message: Option<String>,
}

pub fn parse_persona_pack<I, P, C>(
    manifest_json: &str,
    files: I,
) -> Result<PersonaPack, PersonaPackError>
where
    I: IntoIterator<Item = (P, C)>,
    P: Into<String>,
    C: Into<String>,
{
    if manifest_json.len() > MAX_MANIFEST_BYTES {
        return Err(PersonaPackError::ManifestTooLarge);
    }

    let manifest: Manifest = serde_json::from_str(manifest_json).map_err(|error| {
        PersonaPackError::MalformedManifest(format!(
            "JSON syntax or schema error at line {}, column {}",
            error.line(),
            error.column()
        ))
    })?;
    let id = required_manifest_text(manifest.id, "id")?;
    let name = required_manifest_text(manifest.name, "name")?;
    let version = required_manifest_text(manifest.version, "version")?;
    if manifest.personas.is_empty() {
        return Err(PersonaPackError::InvalidManifestField("personas"));
    }
    if manifest.personas.len() > MAX_PERSONAS {
        return Err(PersonaPackError::TooManyPersonas);
    }

    let files = collect_files(files)?;
    let instructions_path = manifest.pack_instructions.as_deref().or_else(|| {
        files
            .contains_key("instructions.md")
            .then_some("instructions.md")
    });
    let instructions = instructions_path
        .map(|path| {
            validate_relative_path(path, None)?;
            let content = files
                .get(path)
                .ok_or_else(|| PersonaPackError::MissingFile(path.to_string()))?;
            if content.len() > MAX_INSTRUCTIONS_BYTES {
                return Err(PersonaPackError::InstructionsTooLarge(path.to_string()));
            }
            Ok(content.clone())
        })
        .transpose()?;

    let mcp_config_path = validate_optional_path(manifest.mcp_config)?;
    let hooks_config_path = validate_optional_path(manifest.hooks_config)?;
    let defaults = manifest.defaults.unwrap_or_default();
    validate_behavioral_defaults(&defaults)?;

    let mut names = BTreeSet::new();
    let mut personas = Vec::with_capacity(manifest.personas.len());
    for path in manifest.personas {
        validate_relative_path(&path, Some(".persona.md"))?;
        let content = files
            .get(&path)
            .ok_or_else(|| PersonaPackError::MissingFile(path.clone()))?;
        let persona = parse_persona(&path, content, &version, &defaults)?;
        if !names.insert(persona.name.clone()) {
            return Err(PersonaPackError::DuplicatePersona(persona.name));
        }
        personas.push(persona);
    }

    Ok(PersonaPack {
        id,
        name,
        version,
        description: manifest.description,
        instructions,
        mcp_config_path,
        hooks_config_path,
        personas,
    })
}

fn collect_files<I, P, C>(files: I) -> Result<BTreeMap<String, String>, PersonaPackError>
where
    I: IntoIterator<Item = (P, C)>,
    P: Into<String>,
    C: Into<String>,
{
    let mut collected = BTreeMap::new();
    for (path, content) in files {
        let path = path.into();
        validate_relative_path(&path, None)?;
        if collected.insert(path.clone(), content.into()).is_some() {
            return Err(PersonaPackError::DuplicateSourcePath(path));
        }
    }
    Ok(collected)
}

fn parse_persona(
    path: &str,
    content: &str,
    pack_version: &str,
    defaults: &BehavioralDefaults,
) -> Result<Persona, PersonaPackError> {
    let (frontmatter, prompt) = split_frontmatter(path, content)?;
    if frontmatter.len() > MAX_FRONTMATTER_BYTES {
        return Err(PersonaPackError::PersonaFrontmatterTooLarge(
            path.to_string(),
        ));
    }
    if prompt.len() > MAX_BODY_BYTES {
        return Err(PersonaPackError::PersonaBodyTooLarge(path.to_string()));
    }

    let frontmatter: PersonaFrontmatter =
        serde_yaml_ng::from_str(frontmatter).map_err(|error| {
            let reason = error.location().map_or_else(
                || "YAML syntax or schema error".to_string(),
                |location| {
                    format!(
                        "YAML syntax or schema error at line {}, column {}",
                        location.line(),
                        location.column()
                    )
                },
            );
            PersonaPackError::MalformedPersona {
                path: path.to_string(),
                reason,
            }
        })?;
    validate_behavior(path, &frontmatter)?;
    let name = required_persona_text(path, frontmatter.name, "name")?;
    validate_slug(&name).map_err(|reason| malformed(path, reason))?;
    let display_name = required_persona_text(path, frontmatter.display_name, "display_name")?;
    let description = required_persona_text(path, frontmatter.description, "description")?;

    let mut server_names = BTreeSet::new();
    let mut mcp_servers = Vec::with_capacity(frontmatter.mcp_servers.len());
    for server in frontmatter.mcp_servers {
        if server.name.trim().is_empty() || server.command.trim().is_empty() {
            return Err(malformed(
                path,
                "MCP server name and command must be non-empty",
            ));
        }
        if !server_names.insert(server.name.clone()) {
            return Err(PersonaPackError::DuplicateMcpServer {
                persona: name,
                server: server.name,
            });
        }
        mcp_servers.push(PersonaMcpServer {
            name: server.name,
            command: server.command,
            args: server.args,
            environment: server.env,
        });
    }

    let triggers = frontmatter
        .triggers
        .or_else(|| defaults.triggers.clone())
        .map(resolve_triggers);
    let hooks = frontmatter.hooks.map(|hooks| PersonaHooks {
        on_start: hooks.on_start,
        on_stop: hooks.on_stop,
        on_message: hooks.on_message,
    });

    Ok(Persona {
        source_path: path.to_string(),
        name,
        display_name,
        description,
        avatar: frontmatter.avatar,
        version: frontmatter
            .version
            .unwrap_or_else(|| pack_version.to_string()),
        author: frontmatter.author,
        skills: frontmatter.skills,
        mcp_servers,
        hooks,
        prompt: prompt.to_string(),
        behavior: PersonaBehavior {
            model: frontmatter.model.or_else(|| defaults.model.clone()),
            runtime: frontmatter.runtime,
            temperature: frontmatter.temperature.or(defaults.temperature),
            max_context_tokens: frontmatter
                .max_context_tokens
                .or(defaults.max_context_tokens),
            subscribe: frontmatter.subscribe.or_else(|| defaults.subscribe.clone()),
            triggers,
            thread_replies: frontmatter
                .thread_replies
                .or(defaults.thread_replies)
                .unwrap_or(true),
            broadcast_replies: frontmatter
                .broadcast_replies
                .or(defaults.broadcast_replies)
                .unwrap_or(false),
        },
    })
}

fn split_frontmatter<'a>(
    path: &str,
    content: &'a str,
) -> Result<(&'a str, &'a str), PersonaPackError> {
    let rest = content
        .strip_prefix("---\n")
        .or_else(|| content.strip_prefix("---\r\n"))
        .ok_or_else(|| malformed(path, "missing opening frontmatter delimiter"))?;
    let mut offset = 0;
    loop {
        let position = rest[offset..]
            .find("\n---")
            .map(|position| position + offset)
            .ok_or_else(|| malformed(path, "missing closing frontmatter delimiter"))?;
        let after_delimiter = position + 4;
        match rest.as_bytes().get(after_delimiter) {
            None => return Ok((&rest[..position], "")),
            Some(b'\n') => return Ok((&rest[..position], &rest[after_delimiter + 1..])),
            Some(b'\r') if rest.as_bytes().get(after_delimiter + 1) == Some(&b'\n') => {
                return Ok((&rest[..position], &rest[after_delimiter + 2..]));
            }
            _ => offset = after_delimiter,
        }
    }
}

fn resolve_triggers(triggers: TriggerInput) -> PersonaTriggers {
    PersonaTriggers {
        mentions: triggers.mentions.unwrap_or(true),
        keywords: triggers.keywords,
        all_messages: triggers.all_messages.unwrap_or(false),
    }
}

fn validate_behavior(path: &str, frontmatter: &PersonaFrontmatter) -> Result<(), PersonaPackError> {
    if frontmatter
        .temperature
        .is_some_and(|value| !value.is_finite())
    {
        return Err(malformed(path, "temperature must be finite"));
    }
    if frontmatter.max_context_tokens == Some(0) {
        return Err(malformed(
            path,
            "max_context_tokens must be greater than zero",
        ));
    }
    if frontmatter.prompt_fields_contain_nul() {
        return Err(malformed(path, "metadata may not contain NUL characters"));
    }
    for relative_path in frontmatter
        .skills
        .iter()
        .chain(
            frontmatter
                .avatar
                .iter()
                .chain(frontmatter.hooks.iter().flat_map(|hooks| {
                    hooks
                        .on_start
                        .iter()
                        .chain(&hooks.on_stop)
                        .chain(&hooks.on_message)
                })),
        )
    {
        validate_relative_path(relative_path, None)?;
    }
    Ok(())
}

impl PersonaFrontmatter {
    fn prompt_fields_contain_nul(&self) -> bool {
        self.name
            .iter()
            .chain(&self.display_name)
            .chain(&self.description)
            .chain(&self.avatar)
            .chain(&self.version)
            .chain(&self.author)
            .chain(&self.model)
            .chain(&self.runtime)
            .any(|value| value.contains('\0'))
    }
}

fn validate_behavioral_defaults(defaults: &BehavioralDefaults) -> Result<(), PersonaPackError> {
    if defaults.temperature.is_some_and(|value| !value.is_finite()) {
        return Err(PersonaPackError::InvalidManifestField(
            "defaults.temperature",
        ));
    }
    if defaults.max_context_tokens == Some(0) {
        return Err(PersonaPackError::InvalidManifestField(
            "defaults.max_context_tokens",
        ));
    }
    Ok(())
}

fn validate_optional_path(path: Option<String>) -> Result<Option<String>, PersonaPackError> {
    if let Some(path) = path.as_deref() {
        validate_relative_path(path, None)?;
    }
    Ok(path)
}

fn validate_relative_path(
    path: &str,
    required_suffix: Option<&str>,
) -> Result<(), PersonaPackError> {
    let candidate = Path::new(path);
    if path.is_empty()
        || path.contains('\\')
        || path.contains('\0')
        || required_suffix.is_some_and(|suffix| !path.ends_with(suffix))
        || candidate.is_absolute()
        || candidate
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(PersonaPackError::InvalidPath(path.to_string()));
    }
    Ok(())
}

fn validate_slug(slug: &str) -> Result<(), &'static str> {
    let mut characters = slug.chars();
    let Some(first) = characters.next() else {
        return Err("persona name must be non-empty");
    };
    if slug.len() > 64
        || !first.is_ascii_lowercase() && !first.is_ascii_digit()
        || characters.any(|character| {
            !character.is_ascii_lowercase()
                && !character.is_ascii_digit()
                && character != '_'
                && character != '-'
        })
    {
        return Err("persona name must match ^[a-z0-9][a-z0-9_-]{0,63}$");
    }
    Ok(())
}

fn required_manifest_text(
    value: Option<String>,
    field: &'static str,
) -> Result<String, PersonaPackError> {
    value
        .filter(|value| !value.trim().is_empty())
        .ok_or(PersonaPackError::InvalidManifestField(field))
}

fn required_persona_text(
    path: &str,
    value: Option<String>,
    field: &'static str,
) -> Result<String, PersonaPackError> {
    value
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| {
            malformed(
                path,
                match field {
                    "name" => "missing or empty name",
                    "display_name" => "missing or empty display_name",
                    "description" => "missing or empty description",
                    _ => "missing required field",
                },
            )
        })
}

fn malformed(path: &str, reason: impl Into<String>) -> PersonaPackError {
    PersonaPackError::MalformedPersona {
        path: path.to_string(),
        reason: reason.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn persona(name: &str, extra: &str, prompt: &str) -> String {
        format!(
            "---\nname: {name}\ndisplay_name: {name}\ndescription: Test persona\n{extra}---\n{prompt}"
        )
    }

    #[test]
    fn parses_valid_pack_in_manifest_order() {
        let manifest = r#"{
            "id":"com.example.persona-pack",
            "name":"Example Pack",
            "version":"1.2.3",
            "description":"Example personas",
            "personas":["personas/b.persona.md","personas/a.persona.md"],
            "pack_instructions":"instructions.md",
            "mcp_config":".mcp.json",
            "ops_category":"ignored"
        }"#;
        let pack = parse_persona_pack(
            manifest,
            [
                (
                    "personas/a.persona.md",
                    persona("alpha", "version: 2.0.0\n", "Alpha prompt"),
                ),
                (
                    "personas/b.persona.md",
                    persona(
                        "beta",
                        "avatar: assets/beta.png\nskills: [skills/review]\nmcp_servers:\n  - name: search\n    command: search-server\n    args: [--stdio]\n    env:\n      TOKEN: ${TOKEN}\nhooks:\n  on_start: hooks/start.sh\n",
                        "Beta prompt",
                    ),
                ),
                ("instructions.md", "Pack instructions".to_string()),
            ],
        )
        .expect("valid pack should parse");

        assert_eq!(pack.id, "com.example.persona-pack");
        assert_eq!(pack.personas[0].name, "beta");
        assert_eq!(pack.personas[0].version, "1.2.3");
        assert_eq!(pack.personas[0].prompt, "Beta prompt");
        assert_eq!(pack.personas[1].name, "alpha");
        assert_eq!(pack.personas[1].version, "2.0.0");
        assert_eq!(pack.instructions.as_deref(), Some("Pack instructions"));
        assert_eq!(pack.mcp_config_path.as_deref(), Some(".mcp.json"));
    }

    #[test]
    fn inherits_defaults_with_explicit_empty_and_shallow_overrides() {
        let manifest = r#"{
            "id":"inheritance-pack",
            "name":"Inheritance Pack",
            "version":"1.0.0",
            "personas":["personas/inherited.persona.md","personas/override.persona.md"],
            "defaults":{
                "model":"provider:default",
                "temperature":0.7,
                "subscribe":["general"],
                "triggers":{"mentions":false,"keywords":["default"],"all_messages":true},
                "thread_replies":false,
                "broadcast_replies":true
            }
        }"#;
        let pack = parse_persona_pack(
            manifest,
            [
                (
                    "personas/inherited.persona.md",
                    persona("inherited", "", "Inherited"),
                ),
                (
                    "personas/override.persona.md",
                    persona(
                        "override",
                        "subscribe: []\ntriggers:\n  keywords: [local]\nthread_replies: true\n",
                        "Override",
                    ),
                ),
            ],
        )
        .expect("inherited pack should parse");

        let inherited = &pack.personas[0].behavior;
        assert_eq!(inherited.model.as_deref(), Some("provider:default"));
        assert_eq!(
            inherited.subscribe.as_deref(),
            Some(&["general".to_string()][..])
        );
        assert_eq!(
            inherited.triggers,
            Some(PersonaTriggers {
                mentions: false,
                keywords: vec!["default".to_string()],
                all_messages: true,
            })
        );
        assert!(!inherited.thread_replies);
        assert!(inherited.broadcast_replies);

        let overridden = &pack.personas[1].behavior;
        assert_eq!(overridden.subscribe, Some(Vec::new()));
        assert_eq!(
            overridden.triggers,
            Some(PersonaTriggers {
                mentions: true,
                keywords: vec!["local".to_string()],
                all_messages: false,
            })
        );
        assert!(overridden.thread_replies);
    }

    #[test]
    fn rejects_conflicting_personas_and_mcp_servers() {
        let manifest = r#"{
            "id":"conflict-pack","name":"Conflict Pack","version":"1.0.0",
            "personas":["personas/a.persona.md","personas/b.persona.md"]
        }"#;
        let duplicate_persona = parse_persona_pack(
            manifest,
            [
                ("personas/a.persona.md", persona("same", "", "A")),
                ("personas/b.persona.md", persona("same", "", "B")),
            ],
        );
        assert_eq!(
            duplicate_persona,
            Err(PersonaPackError::DuplicatePersona("same".to_string()))
        );

        let duplicate_server = parse_persona_pack(
            r#"{"id":"conflict-pack","name":"Conflict Pack","version":"1","personas":["personas/a.persona.md"]}"#,
            [(
                "personas/a.persona.md",
                persona(
                    "same",
                    "mcp_servers:\n  - {name: server, command: first}\n  - {name: server, command: second}\n",
                    "A",
                ),
            )],
        );
        assert_eq!(
            duplicate_server,
            Err(PersonaPackError::DuplicateMcpServer {
                persona: "same".to_string(),
                server: "server".to_string(),
            })
        );
    }

    #[test]
    fn rejects_malformed_manifests_personas_and_paths() {
        let cases = [
            parse_persona_pack("{}", std::iter::empty::<(&str, &str)>()),
            parse_persona_pack(
                r#"{"id":"bad","name":"Bad","version":"1","personas":["../bad.persona.md"]}"#,
                std::iter::empty::<(&str, &str)>(),
            ),
            parse_persona_pack(
                r#"{"id":"bad","name":"Bad","version":"1","personas":["bad.persona.md"]}"#,
                [("bad.persona.md", "name: missing-delimiters")],
            ),
            parse_persona_pack(
                r#"{"id":"bad","name":"Bad","version":"1","personas":["bad.persona.md"]}"#,
                [(
                    "bad.persona.md",
                    "---\nname: Uppercase\ndisplay_name: Bad\ndescription: Bad\nunknown: true\n---\nBad",
                )],
            ),
        ];

        assert!(cases.into_iter().all(|result| result.is_err()));
    }
}
