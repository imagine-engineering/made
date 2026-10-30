mod audio_input_output_setup;
mod audio_test_window;
mod edit_prediction_provider_setup;
#[cfg(feature = "agentic")]
mod external_agents_page;
mod feature_flags;
#[cfg(feature = "agentic")]
mod llm_providers_page;
#[cfg(feature = "agentic")]
mod mcp_servers_page;
#[cfg(feature = "agentic")]
mod sandbox_settings;
#[cfg(feature = "agentic")]
mod skill_creator;
#[cfg(feature = "agentic")]
mod skills_setup;
#[cfg(feature = "agentic")]
mod tool_permissions_setup;

pub(crate) use audio_input_output_setup::{
    render_input_audio_device_dropdown, render_output_audio_device_dropdown,
};
pub(crate) use audio_test_window::open_audio_test_window;
pub(crate) use edit_prediction_provider_setup::render_edit_prediction_setup_page;
#[cfg(feature = "agentic")]
pub(crate) use external_agents_page::{
    CustomAgentForm, render_add_agent_popover, render_external_agents_page,
};
pub(crate) use feature_flags::render_feature_flags_page;
#[cfg(feature = "agentic")]
pub(crate) use llm_providers_page::{
    LlmProviderForm, render_add_llm_provider_popover, render_llm_providers_page,
};
#[cfg(feature = "agentic")]
pub(crate) use mcp_servers_page::{
    McpServerForm, render_add_server_popover, render_mcp_servers_page,
};
#[cfg(feature = "agentic")]
pub(crate) use sandbox_settings::render_sandbox_settings_page;
#[cfg(feature = "agentic")]
pub use skill_creator::SkillCreatorOpenMode;
#[cfg(feature = "agentic")]
pub(crate) use skill_creator::{
    SkillCreatorEvent, SkillCreatorPage, render_skill_creator_page, skill_url_from_clipboard,
};
#[cfg(all(test, feature = "agentic"))]
pub(crate) use skills_setup::displayed_skills;
#[cfg(feature = "agentic")]
pub(crate) use skills_setup::render_skills_setup_page;
#[cfg(feature = "agentic")]
pub(crate) use tool_permissions_setup::render_tool_permissions_setup_page;

#[cfg(feature = "agentic")]
pub use tool_permissions_setup::{
    render_copy_path_tool_config, render_create_directory_tool_config,
    render_delete_path_tool_config, render_edit_file_tool_config, render_fetch_tool_config,
    render_move_path_tool_config, render_skill_tool_config, render_terminal_tool_config,
    render_web_search_tool_config, render_write_file_tool_config,
};
