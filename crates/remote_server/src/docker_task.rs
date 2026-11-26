use anyhow::{Context as _, Result, ensure};
use std::{
    io::{Read as _, Write as _},
    os::{
        fd::AsRawFd as _,
        linux::net::SocketAddrExt as _,
        unix::{
            net::{SocketAddr, UnixListener, UnixStream},
            process::{CommandExt as _, ExitStatusExt as _},
        },
    },
    process::Command,
    time::{Duration, Instant},
};

const CANCEL_TIMEOUT: Duration = Duration::from_secs(10);
const POLL_INTERVAL: Duration = Duration::from_millis(25);
const STOPPED: [u8; 1] = [1];

fn address(identity: &str) -> Result<SocketAddr> {
    ensure!(
        identity.len() == 32
            && identity
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "Invalid Docker Task identity"
    );
    let user = unsafe { libc::geteuid() };
    Ok(SocketAddr::from_abstract_name(format!(
        "made-docker-task-{user}-{identity}"
    ))?)
}

fn same_user(stream: &UnixStream) -> Result<bool> {
    let mut credentials = libc::ucred {
        pid: 0,
        uid: 0,
        gid: 0,
    };
    let mut size = size_of::<libc::ucred>() as libc::socklen_t;
    let result = unsafe {
        libc::getsockopt(
            stream.as_raw_fd(),
            libc::SOL_SOCKET,
            libc::SO_PEERCRED,
            (&mut credentials as *mut libc::ucred).cast(),
            &mut size,
        )
    };
    if result != 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    Ok(credentials.uid == unsafe { libc::geteuid() })
}

fn accept_cancellation(listener: &UnixListener) -> Result<Option<UnixStream>> {
    match listener.accept() {
        Ok((stream, _)) if same_user(&stream)? => Ok(Some(stream)),
        Ok(_) => Ok(None),
        Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => Ok(None),
        Err(error) => Err(error.into()),
    }
}

fn has_live_group_members(group: libc::pid_t) -> Result<bool> {
    for entry in std::fs::read_dir("/proc")? {
        let entry = entry?;
        if !entry
            .file_name()
            .as_encoded_bytes()
            .iter()
            .all(u8::is_ascii_digit)
        {
            continue;
        }
        let stat = match std::fs::read(entry.path().join("stat")) {
            Ok(stat) => stat,
            Err(error)
                if matches!(
                    error.kind(),
                    std::io::ErrorKind::NotFound | std::io::ErrorKind::PermissionDenied
                ) =>
            {
                continue;
            }
            Err(error) => return Err(error.into()),
        };
        let Some(delimiter) = stat.iter().rposition(|byte| *byte == b')') else {
            continue;
        };
        let mut fields = stat[delimiter + 1..]
            .split(|byte| byte.is_ascii_whitespace())
            .filter(|field| !field.is_empty());
        let state = fields.next();
        let process_group = fields
            .nth(1)
            .and_then(|field| std::str::from_utf8(field).ok()?.parse().ok());
        if process_group == Some(group) && !matches!(state, Some(b"Z" | b"X")) {
            return Ok(true);
        }
    }
    Ok(false)
}

#[expect(
    clippy::disallowed_methods,
    reason = "Docker Task control runs outside GPUI"
)]
pub(super) fn run(identity: &str, mut command: Command) -> Result<i32> {
    let listener = UnixListener::bind_addr(&address(identity)?)?;
    listener.set_nonblocking(true)?;
    if let Some(mut stream) = accept_cancellation(&listener)? {
        stream.write_all(&STOPPED)?;
        return Ok(130);
    }
    command.process_group(0);
    if unsafe { libc::isatty(libc::STDIN_FILENO) } == 1 {
        // The Docker PTY must deliver input and terminal-generated signals to the Task.
        unsafe {
            command.pre_exec(|| {
                let handler = libc::signal(libc::SIGTTOU, libc::SIG_IGN);
                let result = libc::tcsetpgrp(libc::STDIN_FILENO, libc::getpid());
                let error = std::io::Error::last_os_error();
                libc::signal(libc::SIGTTOU, handler);
                if result == 0 { Ok(()) } else { Err(error) }
            });
        }
    }
    let mut child = command.spawn().context("Could not start the Docker Task")?;
    let group = libc::pid_t::try_from(child.id())?;
    loop {
        if let Some(mut stream) = accept_cancellation(&listener)? {
            // The command leader remains unreaped until the signal is delivered.
            if unsafe { libc::kill(-group, libc::SIGKILL) } != 0 {
                let error = std::io::Error::last_os_error();
                if error.raw_os_error() != Some(libc::ESRCH) {
                    return Err(error).context("Could not stop the Docker Task process group");
                }
            }
            let deadline = Instant::now() + CANCEL_TIMEOUT;
            while has_live_group_members(group)? {
                ensure!(
                    Instant::now() < deadline,
                    "Docker Task processes did not stop"
                );
                std::thread::sleep(POLL_INTERVAL);
            }
            child.wait()?;
            stream.write_all(&STOPPED)?;
            return Ok(130);
        }
        if let Some(status) = child.try_wait()? {
            return Ok(status
                .code()
                .unwrap_or_else(|| 128 + status.signal().unwrap_or(1)));
        }
        std::thread::sleep(POLL_INTERVAL);
    }
}

#[expect(
    clippy::disallowed_methods,
    reason = "Docker Task control runs outside GPUI"
)]
pub(super) fn cancel(identity: &str) -> Result<()> {
    let address = address(identity)?;
    let deadline = Instant::now() + CANCEL_TIMEOUT;
    let mut stream = loop {
        match UnixStream::connect_addr(&address) {
            Ok(stream) => break stream,
            Err(error) if error.kind() == std::io::ErrorKind::ConnectionRefused => {
                ensure!(
                    Instant::now() < deadline,
                    "Docker Task did not acknowledge cancellation"
                );
                std::thread::sleep(POLL_INTERVAL);
            }
            Err(error) => return Err(error.into()),
        }
    };
    ensure!(same_user(&stream)?, "Docker Task control user changed");
    stream.set_read_timeout(Some(CANCEL_TIMEOUT))?;
    let mut response = [0];
    stream.read_exact(&mut response)?;
    ensure!(
        response == STOPPED,
        "Docker Task cancellation was not confirmed"
    );
    Ok(())
}
