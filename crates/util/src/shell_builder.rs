use std::borrow::Cow;

use crate::shell::get_system_shell;
use crate::shell::{Shell, ShellKind};

/// ShellBuilder is used to turn a user-requested task into a
/// program that can be executed by the shell.
pub struct ShellBuilder {
    /// The shell to run
    program: String,
    args: Vec<String>,
    interactive: bool,
    /// Whether to redirect stdin to /dev/null for the spawned command as a subshell.
    redirect_stdin: bool,
    kind: ShellKind,
}

impl ShellBuilder {
    /// Create a new ShellBuilder as configured.
    pub fn new(shell: &Shell, is_windows: bool) -> Self {
        let (program, args) = match shell {
            Shell::System => (get_system_shell(), Vec::new()),
            Shell::Program(shell) => (shell.clone(), Vec::new()),
            Shell::WithArguments { program, args, .. } => (program.clone(), args.clone()),
        };

        let kind = ShellKind::new(&program, is_windows);
        Self {
            program,
            args,
            interactive: true,
            kind,
            redirect_stdin: false,
        }
    }
    pub fn non_interactive(mut self) -> Self {
        self.interactive = false;
        self
    }

    /// Returns the label to show in the terminal tab
    pub fn command_label(&self, command_to_use_in_label: &str) -> String {
        if command_to_use_in_label.trim().is_empty() {
            self.program.clone()
        } else {
            match self.kind {
                ShellKind::PowerShell | ShellKind::Pwsh => {
                    format!("{} -C '{}'", self.program, command_to_use_in_label)
                }
                ShellKind::Cmd => {
                    format!("{} /C \"{}\"", self.program, command_to_use_in_label)
                }
                ShellKind::Posix
                | ShellKind::Nushell
                | ShellKind::Fish
                | ShellKind::Csh
                | ShellKind::Tcsh
                | ShellKind::Rc
                | ShellKind::Xonsh
                | ShellKind::Elvish => {
                    let interactivity = self.interactive.then_some("-i ").unwrap_or_default();
                    format!(
                        "{PROGRAM} {interactivity}-c '{command_to_use_in_label}'",
                        PROGRAM = self.program
                    )
                }
            }
        }
    }

    pub fn redirect_stdin_to_dev_null(mut self) -> Self {
        self.redirect_stdin = true;
        self
    }

    /// Returns the program and arguments to run this task in a shell.
    pub fn build(
        mut self,
        task_command: Option<String>,
        task_args: &[String],
    ) -> (String, Vec<String>) {
        if let Some(task_command) = task_command {
            let task_command = if !task_args.is_empty() {
                match self.kind.try_quote_prefix_aware(&task_command) {
                    Some(task_command) => task_command.into_owned(),
                    None => task_command,
                }
            } else {
                task_command
            };
            let mut combined_command = task_args.iter().fold(task_command, |mut command, arg| {
                command.push(' ');
                let shell_variable = self.kind.to_shell_variable(arg);
                command.push_str(&match self.kind.try_quote(&shell_variable) {
                    Some(shell_variable) => shell_variable,
                    None => Cow::Owned(shell_variable),
                });
                command
            });
            if self.redirect_stdin {
                match self.kind {
                    ShellKind::Posix => {
                        // Perform the STDIN redirection prior to the actual
                        // command on a separate line, so that it is already
                        // active if the command contains a syntax error.
                        // Otherwise, with -i, dash will fall back to an
                        // interactive shell in this case.
                        combined_command.insert_str(0, "exec </dev/null\n");
                    }
                    ShellKind::Fish => {
                        combined_command.insert_str(0, "begin; ");
                        combined_command.push_str("; end </dev/null");
                    }
                    ShellKind::Nushell
                    | ShellKind::Csh
                    | ShellKind::Tcsh
                    | ShellKind::Rc
                    | ShellKind::Xonsh
                    | ShellKind::Elvish => {
                        combined_command.insert(0, '(');
                        combined_command.push_str("\n) </dev/null");
                    }
                    ShellKind::PowerShell | ShellKind::Pwsh => {
                        combined_command.insert_str(0, "$null | & {");
                        combined_command.push_str("}");
                    }
                    ShellKind::Cmd => {
                        combined_command.push_str("< NUL");
                    }
                }
            }

            self.args
                .extend(self.kind.args_for_shell(self.interactive, combined_command));
        }

        (self.program, self.args)
    }

    // This should not exist, but our task infra is broken beyond repair right now
    #[doc(hidden)]
    pub fn build_no_quote(
        self,
        task_command: Option<String>,
        task_args: &[String],
    ) -> (String, Vec<String>) {
        self.build_with_literal_prefix(task_command, task_args, 0)
    }

    /// Quotes an internal command and its leading arguments while leaving the
    /// remaining Task arguments available for the existing shell substitution.
    pub fn build_with_literal_prefix(
        self,
        task_command: Option<String>,
        task_args: &[String],
        literal_prefix_args: usize,
    ) -> (String, Vec<String>) {
        self.build_with_command_intent(task_command, task_args, false, literal_prefix_args)
    }

    pub fn build_with_command_intent(
        mut self,
        task_command: Option<String>,
        task_args: &[String],
        literal_command: bool,
        literal_prefix_args: usize,
    ) -> (String, Vec<String>) {
        if let Some(task_command) = task_command {
            let task_command = if literal_command
                && !matches!(
                    self.kind,
                    ShellKind::PowerShell | ShellKind::Pwsh | ShellKind::Cmd
                ) {
                self.kind
                    .try_quote(&task_command)
                    .map(|quoted| self.kind.prepend_command_prefix(&quoted).into_owned())
                    .unwrap_or(task_command)
            } else if literal_prefix_args > 0 {
                self.kind
                    .try_quote_prefix_aware(&task_command)
                    .map(Cow::into_owned)
                    .unwrap_or(task_command)
            } else {
                task_command
            };
            let mut combined_command =
                task_args
                    .iter()
                    .enumerate()
                    .fold(task_command, |mut command, (index, arg)| {
                        command.push(' ');
                        if index < literal_prefix_args {
                            let quoted = self.kind.try_quote(arg).unwrap_or(Cow::Borrowed(arg));
                            command.push_str(&quoted);
                        } else {
                            command.push_str(&self.kind.to_shell_variable(arg));
                        }
                        command
                    });
            if self.redirect_stdin {
                match self.kind {
                    ShellKind::Posix => {
                        combined_command.insert_str(0, "exec </dev/null\n");
                    }
                    ShellKind::Fish => {
                        combined_command.insert_str(0, "begin; ");
                        combined_command.push_str("; end </dev/null");
                    }
                    ShellKind::Nushell
                    | ShellKind::Csh
                    | ShellKind::Tcsh
                    | ShellKind::Rc
                    | ShellKind::Xonsh
                    | ShellKind::Elvish => {
                        combined_command.insert(0, '(');
                        combined_command.push_str("\n) </dev/null");
                    }
                    ShellKind::PowerShell | ShellKind::Pwsh => {
                        combined_command.insert_str(0, "$null | & {");
                        combined_command.push_str("}");
                    }
                    ShellKind::Cmd => {
                        combined_command.push_str("< NUL");
                    }
                }
            }

            self.args
                .extend(self.kind.args_for_shell(self.interactive, combined_command));
        }

        (self.program, self.args)
    }

    /// Builds a `smol::process::Command` with the given task command and arguments.
    ///
    /// Prefer this over manually constructing a command with the output of `Self::build`,
    /// as this method handles `cmd` weirdness on windows correctly.
    pub fn build_smol_command(
        self,
        task_command: Option<String>,
        task_args: &[String],
    ) -> smol::process::Command {
        smol::process::Command::from(self.build_std_command(task_command, task_args))
    }

    /// Builds a `std::process::Command` with the given task command and arguments.
    ///
    /// Prefer this over manually constructing a command with the output of `Self::build`,
    /// as this method handles `cmd` weirdness on windows correctly.
    pub fn build_std_command(
        self,
        mut task_command: Option<String>,
        task_args: &[String],
    ) -> std::process::Command {
        #[cfg(windows)]
        let kind = self.kind;
        if task_args.is_empty() {
            task_command = task_command
                .as_ref()
                .map(|cmd| self.kind.try_quote_prefix_aware(&cmd).map(Cow::into_owned))
                .unwrap_or(task_command);
        }
        let (program, args) = self.build(task_command, task_args);

        let mut child = crate::command::new_std_command(program);

        #[cfg(windows)]
        if kind == ShellKind::Cmd {
            use std::os::windows::process::CommandExt;

            for arg in args {
                child.raw_arg(arg);
            }
        } else {
            child.args(args);
        }

        #[cfg(not(windows))]
        child.args(args);

        child
    }

    pub fn kind(&self) -> ShellKind {
        self.kind
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn literal_command_intent_preserves_unqualified_windows_shell_behavior() {
        for shell in ["powershell.exe", "pwsh.exe", "cmd.exe"] {
            for arguments in [Vec::new(), vec!["literal argument".to_string()]] {
                let shell = Shell::Program(shell.into());
                let command = Some(r"C:\Program Files\profiler.exe".to_string());
                let legacy = ShellBuilder::new(&shell, true).build_with_literal_prefix(
                    command.clone(),
                    &arguments,
                    arguments.len(),
                );
                let requested = ShellBuilder::new(&shell, true).build_with_command_intent(
                    command,
                    &arguments,
                    true,
                    arguments.len(),
                );
                assert_eq!(
                    requested, legacy,
                    "command-position Windows qualification is still pending"
                );
            }
        }
    }

    #[test]
    fn literal_prefix_quotes_helper_but_preserves_task_substitution() {
        let shell = Shell::Program("sh".to_owned());
        let (program, args) = ShellBuilder::new(&shell, false).build_with_literal_prefix(
            Some("/host path/server".to_owned()),
            &[
                "observe-task".to_owned(),
                "--socket".to_owned(),
                "/private/socket path".to_owned(),
                "--".to_owned(),
                "cargo".to_owned(),
                "$ZED_TEST_TASK_VALUE".to_owned(),
            ],
            5,
        );

        assert_eq!(program, "sh");
        assert_eq!(
            args,
            [
                "-i",
                "-c",
                "'/host path/server' observe-task --socket '/private/socket path' -- cargo $ZED_TEST_TASK_VALUE"
            ]
        );
    }

    #[test]
    fn test_nu_shell_variable_substitution() {
        let shell = Shell::Program("nu".to_owned());
        let shell_builder = ShellBuilder::new(&shell, false);

        let (program, args) = shell_builder.build(
            Some("echo".into()),
            &[
                "${hello}".to_string(),
                "$world".to_string(),
                "nothing".to_string(),
                "--$something".to_string(),
                "$".to_string(),
                "${test".to_string(),
            ],
        );

        assert_eq!(program, "nu");
        assert_eq!(
            args,
            vec![
                "-i",
                "-c",
                "echo '$env.hello' '$env.world' nothing '--($env.something)' '$' '${test'"
            ]
        );
    }

    #[test]
    fn redirect_stdin_to_dev_null_precedence() {
        let shell = Shell::Program("nu".to_owned());
        let shell_builder = ShellBuilder::new(&shell, false);

        let (program, args) = shell_builder
            .redirect_stdin_to_dev_null()
            .build(Some("echo".into()), &["nothing".to_string()]);

        assert_eq!(program, "nu");
        assert_eq!(args, vec!["-i", "-c", "(echo nothing\n) </dev/null"]);
    }

    #[test]
    fn redirect_stdin_to_dev_null_fish() {
        let shell = Shell::Program("fish".to_owned());
        let shell_builder = ShellBuilder::new(&shell, false);

        let (program, args) = shell_builder
            .redirect_stdin_to_dev_null()
            .build(Some("echo".into()), &["test".to_string()]);

        assert_eq!(program, "fish");
        assert_eq!(args, vec!["-i", "-c", "begin; echo test; end </dev/null"]);
    }

    #[test]
    fn redirect_stdin_to_dev_null_preserves_heredoc() {
        let shell = Shell::Program("sh".to_owned());
        let shell_builder = ShellBuilder::new(&shell, false);

        let command = "cat <<EOF\nhello\nEOF";
        let (program, args) = shell_builder
            .redirect_stdin_to_dev_null()
            .build(Some(command.into()), &[]);

        assert_eq!(program, "sh");
        assert_eq!(
            args,
            vec!["-i", "-c", "exec </dev/null\ncat <<EOF\nhello\nEOF"]
        );
    }

    #[test]
    fn non_interactive_omits_interactive_flag() {
        // Headless hosts (e.g. the eval CLI) build the agent's shell command
        // non-interactively so it works without a controlling TTY.
        let shell = Shell::Program("sh".to_owned());
        let shell_builder = ShellBuilder::new(&shell, false).non_interactive();

        let (program, args) = shell_builder.build(Some("echo hello".into()), &[]);

        assert_eq!(program, "sh");
        assert_eq!(args, vec!["-c", "echo hello"]);
        assert!(
            !args.iter().any(|arg| arg == "-i"),
            "non-interactive shell command must not include `-i`"
        );
    }

    #[test]
    fn does_not_quote_sole_command_only() {
        let shell = Shell::Program("fish".to_owned());
        let shell_builder = ShellBuilder::new(&shell, false);

        let (program, args) = shell_builder.build(Some("echo".into()), &[]);

        assert_eq!(program, "fish");
        assert_eq!(args, vec!["-i", "-c", "echo"]);

        let shell = Shell::Program("fish".to_owned());
        let shell_builder = ShellBuilder::new(&shell, false);

        let (program, args) = shell_builder.build(Some("echo oo".into()), &[]);

        assert_eq!(program, "fish");
        assert_eq!(args, vec!["-i", "-c", "echo oo"]);
    }
}
