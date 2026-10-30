//! TTY related functionality.

use std::ffi::{CStr, CString};
use std::fmt;
use std::fs::File;
use std::io::{Error, ErrorKind, Read, Result};
use std::mem::MaybeUninit;
use std::os::fd::OwnedFd;
use std::os::unix::ffi::OsStrExt;
use std::os::unix::io::AsRawFd;
use std::os::unix::net::UnixStream;
use std::os::unix::process::CommandExt;
#[cfg(target_os = "macos")]
use std::path::Path;
use std::process::{Child, Command};
use std::sync::Arc;
use std::{env, ptr};

use libc::{F_GETFL, F_SETFL, O_NONBLOCK, TIOCSCTTY, c_int, fcntl};
use log::error;
use polling::{Event, PollMode, Poller};
use rustix_openpty::openpty;
use rustix_openpty::rustix::termios::Winsize;
#[cfg(any(target_os = "linux", target_os = "macos"))]
use rustix_openpty::rustix::termios::{self, InputModes, OptionalActions};
use signal_hook::low_level::{pipe as signal_pipe, unregister as unregister_signal};
use signal_hook::{SigId, consts as sigconsts};

use crate::event::{OnResize, WindowSize};
use crate::tty::{ChildEvent, EventedPty, EventedReadWrite, Options};

// Interest in PTY read/writes.
pub(crate) const PTY_READ_WRITE_TOKEN: usize = 0;

// Interest in new child events.
pub(crate) const PTY_CHILD_EVENT_TOKEN: usize = 1;

macro_rules! die {
    ($($arg:tt)*) => {{
        error!($($arg)*);
        std::process::exit(1);
    }};
}

/// Really only needed on BSD, but should be fine elsewhere.
fn set_controlling_terminal(fd: c_int) -> Result<()> {
    let res = unsafe {
        // TIOSCTTY changes based on platform and the `ioctl` call is different
        // based on architecture (32/64). So a generic cast is used to make sure
        // there are no issues. To allow such a generic cast the clippy warning
        // is disabled.
        #[allow(clippy::cast_lossless)]
        libc::ioctl(fd, TIOCSCTTY as _, 0)
    };

    if res == 0 { Ok(()) } else { Err(Error::last_os_error()) }
}

/// Signal mask to apply to a spawned PTY child before exec.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct SignalMask(libc::sigset_t);

impl SignalMask {
    /// Capture the calling thread's current signal mask.
    pub fn current() -> Result<Self> {
        let mut signal_set = MaybeUninit::<libc::sigset_t>::uninit();

        // `pthread_sigmask` only writes the kernel-relevant portion of `sigset_t` (e.g. 8 bytes
        // on 64-bit glibc, where `sigset_t` is 128 bytes). Zero the whole struct first so the
        // remaining padding is deterministic, otherwise reading it (in `PartialEq`) would be
        // undefined behavior and could spuriously compare unequal.
        if unsafe { libc::sigemptyset(signal_set.as_mut_ptr()) } != 0 {
            return Err(Error::last_os_error());
        }

        let result = unsafe {
            libc::pthread_sigmask(libc::SIG_SETMASK, ptr::null(), signal_set.as_mut_ptr())
        };

        if result != 0 {
            Err(Error::from_raw_os_error(result))
        } else {
            Ok(Self(unsafe { signal_set.assume_init() }))
        }
    }

    fn apply(self) -> Result<()> {
        let result = unsafe { libc::sigprocmask(libc::SIG_SETMASK, &self.0, ptr::null_mut()) };

        if result == -1 { Err(Error::last_os_error()) } else { Ok(()) }
    }
}

impl fmt::Debug for SignalMask {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_tuple("SignalMask").finish()
    }
}

#[derive(Debug)]
struct Passwd<'a> {
    name: &'a str,
    dir: &'a str,
    shell: &'a str,
}

/// Return a Passwd struct with pointers into the provided buf.
///
/// # Unsafety
///
/// If `buf` is changed while `Passwd` is alive, bad thing will almost certainly happen.
fn get_pw_entry(buf: &mut [i8; 1024]) -> Result<Passwd<'_>> {
    // Create zeroed passwd struct.
    let mut entry: MaybeUninit<libc::passwd> = MaybeUninit::uninit();

    let mut res: *mut libc::passwd = ptr::null_mut();

    // Try and read the pw file.
    let uid = unsafe { libc::getuid() };
    let status = unsafe {
        libc::getpwuid_r(uid, entry.as_mut_ptr(), buf.as_mut_ptr() as *mut _, buf.len(), &mut res)
    };
    let entry = unsafe { entry.assume_init() };

    if status < 0 {
        return Err(Error::other("getpwuid_r failed"));
    }

    if res.is_null() {
        return Err(Error::other("pw not found"));
    }

    // Sanity check.
    assert_eq!(entry.pw_uid, uid);

    // Build a borrowed Passwd struct.
    Ok(Passwd {
        name: unsafe { CStr::from_ptr(entry.pw_name).to_str().unwrap() },
        dir: unsafe { CStr::from_ptr(entry.pw_dir).to_str().unwrap() },
        shell: unsafe { CStr::from_ptr(entry.pw_shell).to_str().unwrap() },
    })
}

pub struct Pty {
    child: Child,
    file: File,
    signals: UnixStream,
    sig_id: SigId,
}

impl Pty {
    pub fn child(&self) -> &Child {
        &self.child
    }

    pub fn child_mut(&mut self) -> &mut Child {
        &mut self.child
    }

    pub fn file(&self) -> &File {
        &self.file
    }
}

/// User information that is required for a new shell session.
struct ShellUser {
    user: String,
    home: String,
    shell: String,
}

impl ShellUser {
    /// look for shell, username, longname, and home dir in the respective environment variables
    /// before falling back on looking into `passwd`.
    fn from_env() -> Result<Self> {
        let mut buf = [0; 1024];
        let pw = get_pw_entry(&mut buf);

        match (env::var("USER"), env::var("HOME"), env::var("SHELL")) {
            (Ok(user), Ok(home), Ok(shell)) => Ok(Self { user, home, shell }),
            (user, home, shell) => {
                let passwd = pw?;
                Ok(Self {
                    user: user.unwrap_or_else(|_| passwd.name.to_owned()),
                    home: home.unwrap_or_else(|_| passwd.dir.to_owned()),
                    shell: shell.unwrap_or_else(|_| passwd.shell.to_owned()),
                })
            },
        }
    }
}

#[cfg(not(target_os = "macos"))]
fn default_shell_command(shell: &str, _user: &str, _home: &str) -> Command {
    Command::new(shell)
}

#[cfg(target_os = "macos")]
fn default_shell_command(shell: &str, user: &str, home: &str) -> Command {
    let shell_name = shell.rsplit('/').next().unwrap();

    // On macOS, use the `login` command so the shell will appear as a tty session.
    let mut login_command = Command::new("/usr/bin/login");

    // Exec the shell with argv[0] prepended by '-' so it becomes a login shell.
    // `login` normally does this itself, but `-l` disables this.
    let exec = format!("exec -a -{} {}", shell_name, shell);

    // Since we use -l, `login` will not change directory to the user's home. However,
    // `login` only checks the current working directory for a .hushlogin file, causing
    // it to miss any in the user's home directory. We can fix this by doing the check
    // ourselves and passing `-q`
    let has_home_hushlogin = Path::new(home).join(".hushlogin").exists();

    // -f: Bypasses authentication for the already-logged-in user.
    // -l: Skips changing directory to $HOME and prepending '-' to argv[0].
    // -p: Preserves the environment.
    // -q: Act as if `.hushlogin` exists.
    //
    // XXX: we use zsh here over sh due to `exec -a`.
    let flags = if has_home_hushlogin { "-qflp" } else { "-flp" };
    login_command.args([flags, user, "/bin/zsh", "-fc", &exec]);
    login_command
}

/// Create a new TTY and return a handle to interact with it.
pub fn new(config: &Options, window_size: WindowSize, window_id: u64) -> Result<Pty> {
    new_with_environment(config, window_size, window_id, false)
}

/// Create a TTY with optional removal of inherited process environment variables.
pub fn new_with_environment(
    config: &Options,
    window_size: WindowSize,
    window_id: u64,
    clear_environment: bool,
) -> Result<Pty> {
    let pty = openpty(None, Some(&window_size.to_winsize()))?;
    let (master, slave) = (pty.controller, pty.user);
    from_fd_with_environment(config, window_id, master, slave, clear_environment)
}

/// Create a new TTY from a PTY's file descriptors.
pub fn from_fd(config: &Options, window_id: u64, master: OwnedFd, slave: OwnedFd) -> Result<Pty> {
    from_fd_with_environment(config, window_id, master, slave, false)
}

#[expect(
    clippy::disallowed_methods,
    reason = "The upstream synchronous PTY owns std::process::Child and SIGCHLD; Zed calls it from its background executor"
)]
fn from_fd_with_environment(
    config: &Options,
    window_id: u64,
    master: OwnedFd,
    slave: OwnedFd,
    clear_environment: bool,
) -> Result<Pty> {
    let master_fd = master.as_raw_fd();
    let slave_fd = slave.as_raw_fd();

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    if let Ok(mut termios) = termios::tcgetattr(&master) {
        // Set character encoding to UTF-8.
        termios.input_modes.set(InputModes::IUTF8, true);
        let _ = termios::tcsetattr(&master, OptionalActions::Now, &termios);
    }

    let user = ShellUser::from_env()?;

    let mut builder = if let Some(shell) = config.shell.as_ref() {
        let mut cmd = Command::new(&shell.program);
        cmd.args(shell.args.as_slice());
        cmd
    } else {
        default_shell_command(&user.shell, &user.user, &user.home)
    };

    // Setup child stdin/stdout/stderr as slave fd of PTY.
    builder.stdin(slave.try_clone()?);
    builder.stderr(slave.try_clone()?);
    builder.stdout(slave);

    if clear_environment {
        builder.env_clear();
    }

    // Setup shell environment.
    let window_id = window_id.to_string();
    builder.env("ALACRITTY_WINDOW_ID", &window_id);
    builder.env("USER", user.user);
    builder.env("HOME", user.home);
    // Set Window ID for clients relying on X11 hacks.
    builder.env("WINDOWID", window_id);
    if !clear_environment {
        for (key, value) in &config.env {
            builder.env(key, value);
        }
    }

    // Prevent child processes from inheriting linux-specific startup notification env.
    builder.env_remove("XDG_ACTIVATION_TOKEN");
    builder.env_remove("DESKTOP_STARTUP_ID");

    if clear_environment {
        // The opted-in caller supplies the final map, including any desired runtime stamps.
        builder.envs(&config.env);
    }

    let working_directory = if clear_environment {
        config
            .working_directory
            .as_ref()
            .map(|path| {
                CString::new(path.as_os_str().as_bytes())
                    .map_err(|_| Error::new(ErrorKind::InvalidInput, "PTY directory contains NUL"))
            })
            .transpose()?
    } else {
        config
            .working_directory
            .as_ref()
            .and_then(|path| CString::new(path.as_os_str().as_bytes()).ok())
    };
    let child_signal_mask = config.child_signal_mask;
    unsafe {
        builder.pre_exec(move || {
            // Create a new process group.
            let err = libc::setsid();
            if err == -1 {
                return Err(Error::last_os_error());
            }

            // Only legacy callers permit an invalid directory to fall back to inherited cwd.
            if let Some(working_directory) = working_directory.as_ref() {
                if libc::chdir(working_directory.as_ptr()) == -1 && clear_environment {
                    return Err(Error::last_os_error());
                }
            }

            set_controlling_terminal(slave_fd)?;

            // No longer need slave/master fds.
            libc::close(slave_fd);
            libc::close(master_fd);

            if let Some(child_signal_mask) = child_signal_mask {
                child_signal_mask.apply()?;
            }

            libc::signal(libc::SIGCHLD, libc::SIG_DFL);
            libc::signal(libc::SIGHUP, libc::SIG_DFL);
            libc::signal(libc::SIGINT, libc::SIG_DFL);
            libc::signal(libc::SIGQUIT, libc::SIG_DFL);
            libc::signal(libc::SIGTERM, libc::SIG_DFL);
            libc::signal(libc::SIGALRM, libc::SIG_DFL);

            Ok(())
        });
    }

    // Prepare signal handling before spawning child.
    let (signals, sig_id) = {
        let (sender, recv) = UnixStream::pair()?;

        // Register the recv end of the pipe for SIGCHLD.
        let sig_id = signal_pipe::register(sigconsts::SIGCHLD, sender)?;
        recv.set_nonblocking(true)?;
        (recv, sig_id)
    };

    match builder.spawn() {
        Ok(child) => {
            unsafe {
                // Maybe this should be done outside of this function so nonblocking
                // isn't forced upon consumers. Although maybe it should be?
                set_nonblocking(master_fd)?;
            }

            Ok(Pty { child, file: File::from(master), signals, sig_id })
        },
        Err(err) => {
            if clear_environment {
                unregister_signal(sig_id);
            }
            Err(Error::new(
                err.kind(),
                format!(
                    "Failed to spawn command '{}': {}",
                    builder.get_program().to_string_lossy(),
                    err
                ),
            ))
        },
    }
}

impl Drop for Pty {
    fn drop(&mut self) {
        // Make sure the PTY is terminated properly.
        unsafe {
            libc::kill(self.child.id() as i32, libc::SIGHUP);
        }

        // Clear signal-hook handler.
        unregister_signal(self.sig_id);

        let _ = self.child.wait();
    }
}

impl EventedReadWrite for Pty {
    type Reader = File;
    type Writer = File;

    #[inline]
    unsafe fn register(
        &mut self,
        poll: &Arc<Poller>,
        mut interest: Event,
        poll_opts: PollMode,
    ) -> Result<()> {
        interest.key = PTY_READ_WRITE_TOKEN;
        unsafe {
            poll.add_with_mode(&self.file, interest, poll_opts)?;
        }

        unsafe {
            poll.add_with_mode(
                &self.signals,
                Event::readable(PTY_CHILD_EVENT_TOKEN),
                PollMode::Level,
            )
        }
    }

    #[inline]
    fn reregister(
        &mut self,
        poll: &Arc<Poller>,
        mut interest: Event,
        poll_opts: PollMode,
    ) -> Result<()> {
        interest.key = PTY_READ_WRITE_TOKEN;
        poll.modify_with_mode(&self.file, interest, poll_opts)?;

        poll.modify_with_mode(
            &self.signals,
            Event::readable(PTY_CHILD_EVENT_TOKEN),
            PollMode::Level,
        )
    }

    #[inline]
    fn deregister(&mut self, poll: &Arc<Poller>) -> Result<()> {
        poll.delete(&self.file)?;
        poll.delete(&self.signals)
    }

    #[inline]
    fn reader(&mut self) -> &mut File {
        &mut self.file
    }

    #[inline]
    fn writer(&mut self) -> &mut File {
        &mut self.file
    }
}

impl EventedPty for Pty {
    #[inline]
    fn next_child_event(&mut self) -> Option<ChildEvent> {
        // See if there has been a SIGCHLD.
        let mut buf = [0u8; 1];
        if let Err(err) = self.signals.read(&mut buf) {
            if err.kind() != ErrorKind::WouldBlock {
                error!("Error reading from signal pipe: {err}");
            }
            return None;
        }

        // Match on the child process.
        match self.child.try_wait() {
            Err(err) => {
                error!("Error checking child process termination: {err}");
                None
            },
            Ok(None) => None,
            Ok(exit_status) => Some(ChildEvent::Exited(exit_status)),
        }
    }
}

impl OnResize for Pty {
    /// Resize the PTY.
    ///
    /// Tells the kernel that the window size changed with the new pixel
    /// dimensions and line/column counts.
    fn on_resize(&mut self, window_size: WindowSize) {
        let win = window_size.to_winsize();

        let res = unsafe { libc::ioctl(self.file.as_raw_fd(), libc::TIOCSWINSZ, &win as *const _) };

        if res < 0 {
            die!("ioctl TIOCSWINSZ failed: {}", Error::last_os_error());
        }
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;
    use std::mem::MaybeUninit;
    use std::os::unix::process::ExitStatusExt;
    use std::ptr;
    use std::sync::mpsc;
    use std::thread;
    use std::time::{Duration, Instant};

    use crate::event::WindowSize;
    use crate::tty::{Options, Shell};

    struct SignalMaskGuard {
        previous: libc::sigset_t,
    }

    impl SignalMaskGuard {
        fn block_signal(signal: libc::c_int) -> Self {
            let mut blocked = MaybeUninit::<libc::sigset_t>::uninit();
            let mut previous = MaybeUninit::<libc::sigset_t>::uninit();

            unsafe {
                assert_eq!(libc::sigemptyset(blocked.as_mut_ptr()), 0);
                let mut blocked = blocked.assume_init();
                assert_eq!(libc::sigaddset(&mut blocked, signal), 0);
                assert_eq!(
                    libc::pthread_sigmask(libc::SIG_BLOCK, &blocked, previous.as_mut_ptr()),
                    0
                );

                Self { previous: previous.assume_init() }
            }
        }
    }

    impl Drop for SignalMaskGuard {
        fn drop(&mut self) {
            unsafe {
                assert_eq!(
                    libc::pthread_sigmask(libc::SIG_SETMASK, &self.previous, ptr::null_mut()),
                    0
                );
            }
        }
    }

    fn sleep_command_options(child_signal_mask: Option<super::SignalMask>) -> Options {
        Options {
            shell: Some(Shell::new("/bin/sleep".to_owned(), vec!["30".to_owned()])),
            child_signal_mask,
            ..Options::default()
        }
    }

    fn window_size() -> WindowSize {
        WindowSize { num_lines: 24, num_cols: 80, cell_width: 8, cell_height: 16 }
    }

    fn wait_for_child_exit(
        pty: &mut super::Pty,
        timeout: Duration,
    ) -> Option<std::process::ExitStatus> {
        let deadline = Instant::now() + timeout;
        loop {
            if let Some(status) = pty.child.try_wait().expect("failed to poll PTY child") {
                return Some(status);
            }

            if Instant::now() >= deadline {
                return None;
            }

            thread::sleep(Duration::from_millis(10));
        }
    }

    fn assert_ctrl_c_terminates_sleep_child(child_signal_mask: Option<super::SignalMask>) {
        let mut pty = super::new(&sleep_command_options(child_signal_mask), window_size(), 0)
            .expect("failed to spawn PTY child");

        pty.file.write_all(b"\x03").expect("failed to write Ctrl-C to PTY");

        let status = wait_for_child_exit(&mut pty, Duration::from_secs(2))
            .expect("PTY child did not exit after Ctrl-C");

        assert_eq!(status.signal(), Some(libc::SIGINT));
    }

    #[test]
    fn ctrl_c_reaches_child_spawned_with_default_signal_mask() {
        assert_ctrl_c_terminates_sleep_child(None);
    }

    #[test]
    fn ctrl_c_reaches_child_spawned_with_sigint_blocked_on_parent_thread() {
        let child_signal_mask =
            super::SignalMask::current().expect("failed to capture signal mask");
        let _signal_mask_guard = SignalMaskGuard::block_signal(libc::SIGINT);

        assert_ctrl_c_terminates_sleep_child(Some(child_signal_mask));
    }

    /// Spawns a `/bin/sleep` PTY child the way Zed does: the signal mask is captured on a
    /// "foreground" thread (where terminal signals are unblocked) while the PTY itself is
    /// created on a separate "background" thread that has `signal` blocked.
    ///
    /// On macOS, Zed's background executor runs work on libdispatch (GCD) worker threads,
    /// which start with terminal signals blocked. Without forwarding the foreground mask
    /// through `Options::child_signal_mask`, the forked child inherits that blocked mask
    /// and never receives the signal.
    fn spawn_sleep_child_with_foreground_signal_mask(signal: libc::c_int) -> super::Pty {
        let (mask_tx, mask_rx) = mpsc::channel();

        // "Foreground" thread: capture the mask while `signal` is unblocked, then hand it off.
        let foreground = thread::spawn(move || {
            let mask = super::SignalMask::current().expect("failed to capture signal mask");
            mask_tx.send(mask).expect("failed to forward captured signal mask");
        });

        // "Background" thread: block `signal`, then spawn the child with the captured mask.
        let background = thread::spawn(move || {
            let _signal_mask_guard = SignalMaskGuard::block_signal(signal);
            let mask = mask_rx.recv().expect("failed to receive captured signal mask");
            super::new(&sleep_command_options(Some(mask)), window_size(), 0)
                .expect("failed to spawn PTY child")
        });

        foreground.join().expect("foreground thread panicked");
        background.join().expect("background thread panicked")
    }

    /// Asserts that a signal Alacritty does *not* reset to `SIG_DFL` in its `pre_exec` hook is
    /// still deliverable to a child spawned on a background thread that had it blocked, because
    /// the captured foreground mask is applied before exec.
    ///
    /// Alacritty only resets the dispositions of `SIGCHLD`, `SIGHUP`, `SIGINT`, `SIGQUIT`,
    /// `SIGTERM` and `SIGALRM`. Every other signal relies entirely on the inherited signal mask
    /// being correct, so this is where forwarding the foreground mask actually matters.
    fn assert_signal_reaches_child_spawned_on_background_thread(signal: libc::c_int) {
        let mut pty = spawn_sleep_child_with_foreground_signal_mask(signal);
        let pid = pty.child.id() as libc::pid_t;

        assert_eq!(
            unsafe { libc::kill(pid, signal) },
            0,
            "failed to send signal {signal} to PTY child: {}",
            std::io::Error::last_os_error(),
        );

        let status = wait_for_child_exit(&mut pty, Duration::from_secs(2))
            .unwrap_or_else(|| panic!("PTY child did not exit after signal {signal}"));

        assert_eq!(status.signal(), Some(signal));
    }

    // The signals below are representative of those Alacritty does not reset: their default
    // action is to terminate, and they are not used by the Rust test runtime, so delivery can
    // be observed via process termination. Stop-default (`SIGTSTP`/`SIGTTIN`/`SIGTTOU`) and
    // ignore-default (`SIGWINCH`) signals can't be observed this way and are not covered here.
    #[test]
    fn sigusr1_reaches_child_spawned_on_background_thread() {
        assert_signal_reaches_child_spawned_on_background_thread(libc::SIGUSR1);
    }

    #[test]
    fn sigusr2_reaches_child_spawned_on_background_thread() {
        assert_signal_reaches_child_spawned_on_background_thread(libc::SIGUSR2);
    }

    #[test]
    fn sigvtalrm_reaches_child_spawned_on_background_thread() {
        assert_signal_reaches_child_spawned_on_background_thread(libc::SIGVTALRM);
    }

    /// Without forwarding the foreground mask, a child spawned on a thread that has a signal
    /// blocked inherits that blocked mask and never receives the signal. This pins down the
    /// regression that broke Ctrl-C in the terminal when spawning moved to the background
    /// executor (zed-industries/zed#42234, #42411).
    #[test]
    fn signal_blocked_on_spawn_thread_is_not_delivered_without_foreground_mask() {
        let signal = libc::SIGUSR1;

        let mut pty = thread::spawn(move || {
            let _signal_mask_guard = SignalMaskGuard::block_signal(signal);
            super::new(&sleep_command_options(None), window_size(), 0)
                .expect("failed to spawn PTY child")
        })
        .join()
        .expect("spawn thread panicked");

        let pid = pty.child.id() as libc::pid_t;
        assert_eq!(
            unsafe { libc::kill(pid, signal) },
            0,
            "failed to send signal {signal} to PTY child: {}",
            std::io::Error::last_os_error(),
        );

        // The signal is blocked in the child, so it must still be running after a grace period.
        assert!(
            wait_for_child_exit(&mut pty, Duration::from_millis(500)).is_none(),
            "child exited after a blocked signal was sent; it should have stayed blocked",
        );

        // Clean up the still-running child.
        unsafe {
            libc::kill(pid, libc::SIGKILL);
        }
    }
    fn exact_environment_output(pty: &mut super::Pty) -> Vec<u8> {
        use std::io::Read as _;
        let deadline = Instant::now() + Duration::from_secs(2);
        let mut output = Vec::new();
        let mut buffer = [0_u8; 4096];
        let mut status = None;
        loop {
            match pty.file.read(&mut buffer) {
                Ok(0) => {
                    if status.is_some() {
                        break;
                    }
                },
                Ok(length) => {
                    assert!(output.len() + length <= 1024 * 1024, "bounded fixture output");
                    output.extend_from_slice(&buffer[..length]);
                },
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    if status.is_some() {
                        break;
                    }
                },
                Err(error) if error.raw_os_error() == Some(libc::EIO) => {
                    if status.is_some() {
                        break;
                    }
                },
                Err(error) => panic!("PTY receiver failed: {error}"),
            }
            if status.is_none() {
                status = pty.child.try_wait().expect("actual child status");
            }
            assert!(Instant::now() < deadline, "actual PTY child/EOF deadline");
            thread::sleep(Duration::from_millis(5));
        }
        assert!(status.expect("actual completed PTY child").success());
        output
    }

    #[test]
    fn exact_environment_does_not_inherit_and_applies_complete_map_last() {
        let inherited_path = std::env::var("PATH").expect("fixture parent has inherited PATH");
        let final_environment = std::collections::HashMap::from([
            ("USER".to_owned(), "controlled-user".to_owned()),
            ("HOME".to_owned(), "/private/tmp".to_owned()),
            ("WINDOWID".to_owned(), "123".to_owned()),
            ("ALACRITTY_WINDOW_ID".to_owned(), "123".to_owned()),
            ("CONTROLLED_EXACT_VALUE".to_owned(), "spaces $ literal".to_owned()),
        ]);
        let options = Options {
            shell: Some(Shell::new("/usr/bin/env".to_owned(), Vec::new())),
            env: final_environment.clone(),
            ..Options::default()
        };
        let mut exact = super::new_with_environment(&options, window_size(), 77, true)
            .expect("same real PTY constructor with opted-in clearing");
        let output = exact_environment_output(&mut exact);
        let received = String::from_utf8(output)
            .map_err(|_| ())
            .expect("controlled UTF8 environment")
            .lines()
            .map(|line| {
                let (key, value) = line.split_once('=').expect("actual env receiver entry");
                (key.to_owned(), value.to_owned())
            })
            .collect::<std::collections::HashMap<_, _>>();
        assert!(
            received == final_environment,
            "Actual opted-in PTY environment differs from its complete controlled map"
        );
        drop(exact);
        let mut legacy = super::new(&options, window_size(), 77).expect("ordinary inherited PTY");
        let output = exact_environment_output(&mut legacy);
        let expected_path = format!("PATH={inherited_path}");
        assert!(
            String::from_utf8(output)
                .map_err(|_| ())
                .expect("legacy fixture environment")
                .lines()
                .any(|line| line == expected_path)
        );
    }

    #[test]
    fn exact_environment_rejects_failed_directory_and_nul_with_fresh_recovery() {
        use std::os::unix::ffi::OsStringExt as _;
        let parent = std::env::temp_dir().join(format!(
            "alacritty-exact-directory-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("fixture clock")
                .as_nanos(),
        ));
        let mut options = Options {
            shell: Some(Shell::new("/usr/bin/true".to_owned(), Vec::new())),
            working_directory: Some(parent.join("missing")),
            ..Options::default()
        };
        assert!(super::new_with_environment(&options, window_size(), 0, true).is_err());
        let mut legacy = super::new(&options, window_size(), 0)
            .expect("legacy missing-directory fallback is preserved");
        assert!(
            wait_for_child_exit(&mut legacy, Duration::from_secs(2))
                .expect("legacy actual completion")
                .success()
        );
        drop(legacy);
        options.working_directory = Some(std::path::PathBuf::from(std::ffi::OsString::from_vec(
            b"/private/tmp/invalid\0directory".to_vec(),
        )));
        assert_eq!(
            super::new_with_environment(&options, window_size(), 0, true)
                .err()
                .expect("NUL must reject before child")
                .kind(),
            std::io::ErrorKind::InvalidInput,
        );
        std::fs::create_dir(&parent).expect("fresh genuine directory");
        options.working_directory = Some(parent.clone());
        let mut fresh = super::new_with_environment(&options, window_size(), 0, true)
            .expect("fresh opted-in actual child");
        assert!(
            wait_for_child_exit(&mut fresh, Duration::from_secs(2))
                .expect("fresh actual completion")
                .success()
        );
        drop(fresh);
        std::fs::remove_dir(parent).expect("owned fixture cleanup");
    }
}

/// Types that can produce a `Winsize`.
pub trait ToWinsize {
    /// Get a `Winsize`.
    fn to_winsize(self) -> Winsize;
}

impl ToWinsize for WindowSize {
    fn to_winsize(self) -> Winsize {
        let ws_row = self.num_lines as libc::c_ushort;
        let ws_col = self.num_cols as libc::c_ushort;

        let ws_xpixel = ws_col * self.cell_width as libc::c_ushort;
        let ws_ypixel = ws_row * self.cell_height as libc::c_ushort;
        Winsize { ws_row, ws_col, ws_xpixel, ws_ypixel }
    }
}

unsafe fn set_nonblocking(fd: c_int) -> Result<()> {
    let res = unsafe { fcntl(fd, F_SETFL, fcntl(fd, F_GETFL, 0) | O_NONBLOCK) };
    if res == 0 { Ok(()) } else { Err(Error::last_os_error()) }
}

#[test]
fn test_get_pw_entry() {
    let mut buf: [i8; 1024] = [0; 1024];
    let _pw = get_pw_entry(&mut buf).unwrap();
}
