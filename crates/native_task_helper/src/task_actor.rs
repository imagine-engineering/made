use std::{
    fs::{File, Metadata},
    io,
    os::{fd::AsRawFd, unix::fs::MetadataExt},
    process::Child,
};

// This proves original-process identity only. Host Task admission, nonce, image
// authorization and the original Task completion must still be checked by their owners.
pub struct OriginalTaskActor {
    process_id: u32,
    expected_image: File,
    image_identity: ImageIdentity,
    lifetime: ProcessLifetime,
    retired: bool,
}

pub struct AuthenticatedTaskActor {
    actor: OriginalTaskActor,
}

impl OriginalTaskActor {
    /// Call inside the spawn owner, before handing the child to a waiter/event loop.
    /// A later PID lookup cannot prove the original actor after the child is reaped.
    pub fn from_std_child(child: &mut Child, expected_image: &File) -> io::Result<Self> {
        if child.try_wait()?.is_some() {
            return Err(exited_actor());
        }
        let mut actor = Self::pin_unwaited_child(child.id(), expected_image)?;
        if child.try_wait()?.is_some() {
            return Err(exited_actor());
        }
        actor.revalidate()?;
        Ok(actor)
    }

    /// Call before creating a status future or reader that can wait this child.
    /// A smol status future can retain a shared child independently of this borrow.
    pub fn from_smol_child(
        child: &mut smol::process::Child,
        expected_image: &File,
    ) -> io::Result<Self> {
        if child.try_status()?.is_some() {
            return Err(exited_actor());
        }
        let mut actor = Self::pin_unwaited_child(child.id(), expected_image)?;
        if child.try_status()?.is_some() {
            return Err(exited_actor());
        }
        actor.revalidate()?;
        Ok(actor)
    }

    fn pin_unwaited_child(process_id: u32, expected_image: &File) -> io::Result<Self> {
        let image_identity = ImageIdentity::read(expected_image)?;
        let lifetime = ProcessLifetime::pin(process_id)?;
        let mut actor = Self {
            process_id,
            expected_image: expected_image.try_clone()?,
            image_identity,
            lifetime,
            retired: false,
        };
        actor.revalidate()?;
        Ok(actor)
    }

    pub fn authenticate_peer<S: AsRawFd>(
        mut self,
        connection: &S,
    ) -> io::Result<AuthenticatedTaskActor> {
        let peer = peer_identity(connection.as_raw_fd())?;
        if peer.process_id != self.process_id || peer.user_id != current_user_id() {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "Task connection does not belong to its original actor",
            ));
        }
        self.revalidate()?;
        Ok(AuthenticatedTaskActor { actor: self })
    }

    fn revalidate(&mut self) -> io::Result<()> {
        if self.retired {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "Original Task actor lease was retired",
            ));
        }
        let result = (|| {
            self.lifetime.ensure_running()?;
            if ImageIdentity::read(&self.expected_image)? != self.image_identity
                || ImageIdentity::read(&self.lifetime.executable()?)? != self.image_identity
            {
                return Err(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    "Original Task actor image changed",
                ));
            }
            self.lifetime.ensure_running()
        })();
        // An actor can exec back to an old image. Exclusive ownership and the
        // latched failure prevent that from renewing the original lease.
        if result.is_err() {
            self.retired = true;
        }
        result
    }
}

impl AuthenticatedTaskActor {
    pub fn revalidate(&mut self) -> io::Result<()> {
        self.actor.revalidate()
    }
}

#[derive(PartialEq, Eq)]
struct ImageIdentity {
    device: u64,
    inode: u64,
    size: u64,
    mode: u32,
    user: u32,
    group: u32,
    modified: (i64, i64),
    changed: (i64, i64),
}

impl ImageIdentity {
    fn read(file: &File) -> io::Result<Self> {
        let metadata = file.metadata()?;
        if !metadata.is_file() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Task image is not a regular file",
            ));
        }
        Ok(Self::from_metadata(metadata))
    }

    fn from_metadata(metadata: Metadata) -> Self {
        Self {
            device: metadata.dev(),
            inode: metadata.ino(),
            size: metadata.size(),
            mode: metadata.mode(),
            user: metadata.uid(),
            group: metadata.gid(),
            modified: (metadata.mtime(), metadata.mtime_nsec()),
            changed: (metadata.ctime(), metadata.ctime_nsec()),
        }
    }
}

struct PeerIdentity {
    process_id: u32,
    user_id: u32,
}

fn current_user_id() -> u32 {
    // geteuid has no failure result and does not change process credentials.
    unsafe { libc::geteuid() }
}

fn exited_actor() -> io::Error {
    io::Error::new(
        io::ErrorKind::NotConnected,
        "Original Task actor has exited",
    )
}

#[cfg(target_os = "linux")]
use linux::{ProcessLifetime, peer_identity};

#[cfg(target_os = "macos")]
use macos::{ProcessLifetime, peer_identity};

#[cfg(target_os = "linux")]
mod linux {
    use super::{PeerIdentity, exited_actor};
    use std::{
        ffi::CStr,
        fs::{File, OpenOptions},
        io::{self, Read},
        os::{
            fd::{AsRawFd, FromRawFd, OwnedFd, RawFd},
            unix::fs::OpenOptionsExt,
        },
    };

    pub(super) struct ProcessLifetime {
        descriptor: OwnedFd,
        directory: File,
        start_time: u64,
    }

    impl ProcessLifetime {
        pub(super) fn pin(process_id: u32) -> io::Result<Self> {
            let process_id = libc::pid_t::try_from(process_id).map_err(|_| {
                io::Error::new(io::ErrorKind::InvalidInput, "Invalid original Task actor")
            })?;
            if process_id <= 0 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "Invalid original Task actor",
                ));
            }
            // The caller still owns the unreaped Child. This descriptor is never
            // opened from a socket/frame PID or after starting its status waiter.
            #[cfg(test)]
            observation_failure::check(observation_failure::Boundary::Pidfd, &[])?;
            let descriptor = unsafe { libc::syscall(libc::SYS_pidfd_open, process_id, 0) };
            if descriptor < 0 {
                return Err(io::Error::last_os_error());
            }
            // A successful pidfd_open returns a new owned file descriptor.
            let descriptor = unsafe { OwnedFd::from_raw_fd(descriptor as RawFd) };
            #[cfg(test)]
            observation_failure::check(
                observation_failure::Boundary::ProcDirectory,
                &[descriptor.as_raw_fd()],
            )?;
            let directory = OpenOptions::new()
                .read(true)
                .custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC)
                .open(format!("/proc/{process_id}"))?;
            let start_time = read_start_time(&directory)?;
            let lifetime = Self {
                descriptor,
                directory,
                start_time,
            };
            lifetime.ensure_running()?;
            Ok(lifetime)
        }

        pub(super) fn ensure_running(&self) -> io::Result<()> {
            let mut descriptor = libc::pollfd {
                fd: self.descriptor.as_raw_fd(),
                events: libc::POLLIN,
                revents: 0,
            };
            // Zero timeout only checks lifetime; readiness is never an exit status.
            let count = unsafe { libc::poll(&mut descriptor, 1, 0) };
            if count < 0 {
                return Err(io::Error::last_os_error());
            }
            if count != 0 || descriptor.revents != 0 {
                return Err(exited_actor());
            }
            if read_start_time(&self.directory)? != self.start_time {
                return Err(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    "Original Task actor lifetime changed",
                ));
            }
            Ok(())
        }

        pub(super) fn executable(&self) -> io::Result<File> {
            #[cfg(test)]
            observation_failure::check(
                observation_failure::Boundary::Executable,
                &[self.descriptor.as_raw_fd(), self.directory.as_raw_fd()],
            )?;
            open_proc_file(&self.directory, c"exe")
        }
    }

    #[cfg(test)]
    pub(super) mod observation_failure {
        use std::{cell::RefCell, io, os::fd::RawFd};

        #[derive(Clone, Copy, PartialEq, Eq)]
        pub(crate) enum Boundary {
            Pidfd,
            ProcDirectory,
            Executable,
        }

        struct Failure {
            boundary: Boundary,
            observed: Option<Vec<RawFd>>,
        }

        thread_local! {
            static FAILURE: RefCell<Option<Failure>> = const { RefCell::new(None) };
        }

        pub(super) fn check(boundary: Boundary, descriptors: &[RawFd]) -> io::Result<()> {
            FAILURE.with(|failure| {
                let mut failure = failure.borrow_mut();
                if let Some(failure) = failure.as_mut()
                    && failure.boundary == boundary
                {
                    assert!(failure.observed.is_none(), "Denied observation was retried");
                    failure.observed = Some(descriptors.to_vec());
                    return Err(io::Error::from_raw_os_error(libc::EACCES));
                }
                Ok(())
            })
        }

        pub(crate) fn during<T>(
            boundary: Boundary,
            operation: impl FnOnce() -> io::Result<T>,
        ) -> (io::Result<T>, Vec<RawFd>) {
            struct Reset;
            impl Drop for Reset {
                fn drop(&mut self) {
                    FAILURE.with(|failure| failure.replace(None));
                }
            }
            FAILURE.with(|failure| {
                assert!(failure.borrow().is_none(), "Nested observation fault");
                failure.replace(Some(Failure {
                    boundary,
                    observed: None,
                }));
            });
            let reset = Reset;
            let result = operation();
            let descriptors = FAILURE.with(|failure| {
                failure
                    .borrow_mut()
                    .as_mut()
                    .and_then(|failure| failure.observed.take())
                    .expect("The denied observation boundary was never reached")
            });
            drop(reset);
            (result, descriptors)
        }
    }

    fn open_proc_file(directory: &File, name: &CStr) -> io::Result<File> {
        // exe is a proc magic link and must be followed. The retained proc
        // directory prevents a reused numeric PID from retargeting this lookup.
        let descriptor = unsafe {
            libc::openat(
                directory.as_raw_fd(),
                name.as_ptr(),
                libc::O_RDONLY | libc::O_CLOEXEC,
            )
        };
        if descriptor < 0 {
            return Err(io::Error::last_os_error());
        }
        // openat returned a new descriptor owned by this operation.
        Ok(unsafe { File::from_raw_fd(descriptor) })
    }

    fn read_start_time(directory: &File) -> io::Result<u64> {
        let mut bytes = Vec::new();
        open_proc_file(directory, c"stat")?
            .take(16 * 1024 + 1)
            .read_to_end(&mut bytes)?;
        if bytes.len() > 16 * 1024 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Original Task actor identity exceeds its limit",
            ));
        }
        let text = std::str::from_utf8(&bytes).map_err(|_| {
            io::Error::new(io::ErrorKind::InvalidData, "Invalid Task actor identity")
        })?;
        // comm may itself contain spaces and parentheses; its final ')' ends field 2.
        text.rsplit_once(')')
            .and_then(|(_, fields)| fields.split_whitespace().nth(19))
            .and_then(|value| value.parse().ok())
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Invalid Task actor start identity",
                )
            })
    }

    pub(super) fn peer_identity(connection: RawFd) -> io::Result<PeerIdentity> {
        let mut credentials = std::mem::MaybeUninit::<libc::ucred>::uninit();
        let mut length = std::mem::size_of::<libc::ucred>() as libc::socklen_t;
        // Credentials are supplied by the connected Unix socket, not its payload.
        let result = unsafe {
            libc::getsockopt(
                connection,
                libc::SOL_SOCKET,
                libc::SO_PEERCRED,
                credentials.as_mut_ptr().cast(),
                &mut length,
            )
        };
        if result != 0 {
            return Err(io::Error::last_os_error());
        }
        if length as usize != std::mem::size_of::<libc::ucred>() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid Task peer credentials",
            ));
        }
        // getsockopt succeeded and wrote the complete checked-size credentials.
        let credentials = unsafe { credentials.assume_init() };
        let process_id = u32::try_from(credentials.pid)
            .ok()
            .filter(|process_id| *process_id != 0)
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "Invalid Task peer process")
            })?;
        Ok(PeerIdentity {
            process_id,
            user_id: credentials.uid,
        })
    }
}

#[cfg(target_os = "macos")]
mod macos {
    use super::PeerIdentity;
    use std::{fs::File, io, os::fd::RawFd};

    pub(super) struct ProcessLifetime;

    fn unsupported_image_binding() -> io::Error {
        io::Error::new(
            io::ErrorKind::Unsupported,
            "macOS original Task image binding requires the existing kernel-image policy",
        )
    }

    impl ProcessLifetime {
        pub(super) fn pin(_process_id: u32) -> io::Result<Self> {
            // Reopening proc_pidpath can read a replacement file rather than the
            // executable actually mapped by the original child. The existing
            // Project image issuer must establish that authority before enabling this path.
            Err(unsupported_image_binding())
        }

        pub(super) fn ensure_running(&self) -> io::Result<()> {
            Err(unsupported_image_binding())
        }

        pub(super) fn executable(&self) -> io::Result<File> {
            Err(unsupported_image_binding())
        }
    }

    pub(super) fn peer_identity(_connection: RawFd) -> io::Result<PeerIdentity> {
        Err(unsupported_image_binding())
    }
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::*;
    use std::{
        io::{Read, Write},
        os::unix::{net::UnixListener, process::CommandExt},
        path::{Path, PathBuf},
        process::{Command, Stdio},
        time::{Duration, Instant},
    };

    struct ActorFixture {
        child: Child,
        listener: UnixListener,
        marker: PathBuf,
        _directory: tempfile::TempDir,
    }

    impl ActorFixture {
        fn spawn() -> io::Result<Self> {
            Self::spawn_image(&std::env::current_exe()?, &[])
        }

        #[expect(
            clippy::disallowed_methods,
            reason = "This synchronous process test must supply an actual std Child to its constructor"
        )]
        fn spawn_image(image: &Path, environment: &[(&str, &Path)]) -> io::Result<Self> {
            let directory = tempfile::tempdir()?;
            let socket = directory.path().join("actor.sock");
            let marker = directory.path().join("workload-started");
            let listener = UnixListener::bind(&socket)?;
            listener.set_nonblocking(true)?;
            let child = Command::new(image)
                .args([
                    "--exact",
                    "task_actor::tests::original_task_actor_fixture",
                    "--nocapture",
                ])
                .env("ZED_TEST_TASK_ACTOR_SOCKET", socket)
                .env("ZED_TEST_TASK_ACTOR_MARKER", &marker)
                .env("ZED_TEST_TASK_ACTOR_ROLE", "actor")
                .envs(environment.iter().copied())
                .stdin(Stdio::null())
                .spawn()?;
            Ok(Self {
                child,
                listener,
                marker,
                _directory: directory,
            })
        }

        fn connect(&mut self) -> io::Result<std::os::unix::net::UnixStream> {
            let deadline = Instant::now() + Duration::from_secs(5);
            loop {
                match self.listener.accept() {
                    Ok((stream, _)) => {
                        stream.set_read_timeout(Some(Duration::from_secs(5)))?;
                        stream.set_write_timeout(Some(Duration::from_secs(5)))?;
                        return Ok(stream);
                    }
                    Err(error) if error.kind() == io::ErrorKind::WouldBlock => {}
                    Err(error) => return Err(error),
                }
                if self.child.try_wait()?.is_some() {
                    return Err(io::Error::other("Actor fixture exited before connection"));
                }
                if Instant::now() >= deadline {
                    return Err(io::Error::new(
                        io::ErrorKind::TimedOut,
                        "Actor fixture timed out",
                    ));
                }
                std::thread::sleep(Duration::from_millis(5));
            }
        }

        fn wait(&mut self) -> io::Result<std::process::ExitStatus> {
            let deadline = Instant::now() + Duration::from_secs(5);
            loop {
                if let Some(status) = self.child.try_wait()? {
                    return Ok(status);
                }
                if Instant::now() >= deadline {
                    return Err(io::Error::new(
                        io::ErrorKind::TimedOut,
                        "Actor exit timed out",
                    ));
                }
                std::thread::sleep(Duration::from_millis(5));
            }
        }
    }

    impl Drop for ActorFixture {
        fn drop(&mut self) {
            match self.child.try_wait() {
                Ok(Some(_)) => return,
                Ok(None) => {}
                Err(error) => eprintln!("Actor fixture status failed during cleanup: {error}"),
            }
            if let Err(error) = self.child.kill() {
                eprintln!("Actor fixture kill failed during cleanup: {error}");
            }
            if let Err(error) = self.child.wait() {
                eprintln!("Actor fixture reap failed during cleanup: {error}");
            }
        }
    }

    fn expected_image() -> io::Result<File> {
        File::open(std::env::current_exe()?)
    }

    #[test]
    fn original_task_actor_fixture() -> io::Result<()> {
        let Some(socket) = std::env::var_os("ZED_TEST_TASK_ACTOR_SOCKET") else {
            return Ok(());
        };
        let mut stream = std::os::unix::net::UnixStream::connect(socket)?;
        stream.set_read_timeout(Some(Duration::from_secs(5)))?;
        let mut control = [0];
        stream.read_exact(&mut control)?;
        match std::env::var_os("ZED_TEST_TASK_ACTOR_ROLE").as_deref() {
            Some(role) if role == std::ffi::OsStr::new("fd-observation") => {
                if control != [b'F'] {
                    return Err(io::Error::other("Unexpected FD observation control"));
                }
                check_observation_denials_release_owned_descriptors()?;
                let marker = std::env::var_os("ZED_TEST_TASK_ACTOR_MARKER")
                    .ok_or_else(|| io::Error::other("FD observation marker is missing"))?;
                std::fs::write(marker, b"fd-observation-complete")?;
                return Ok(());
            }
            Some(role) if role != std::ffi::OsStr::new("actor") => {
                return Err(io::Error::other("Unexpected actor fixture role"));
            }
            _ => {}
        }
        match control {
            [b'A'] => {
                let marker = std::env::var_os("ZED_TEST_TASK_ACTOR_MARKER")
                    .ok_or_else(|| io::Error::other("Actor fixture marker is missing"))?;
                std::fs::write(marker, b"started")?;
                Ok(())
            }
            [b'E'] => Err(Command::new("/bin/sleep").arg("10").exec()),
            [b'R'] => {
                let next = std::env::var_os("ZED_TEST_TASK_ACTOR_NEXT_IMAGE")
                    .ok_or_else(|| io::Error::other("Actor next image is missing"))?;
                let original = std::env::var_os("ZED_TEST_TASK_ACTOR_RETURN_IMAGE")
                    .ok_or_else(|| io::Error::other("Actor return image is missing"))?;
                Err(Command::new(next)
                    .args([
                        "--exact",
                        "task_actor::tests::original_task_actor_fixture",
                        "--nocapture",
                    ])
                    .env("ZED_TEST_TASK_ACTOR_NEXT_IMAGE", original)
                    .exec())
            }
            [b'Q'] => Ok(()),
            _ => Err(io::Error::other("Unexpected actor fixture control")),
        }
    }

    #[test]
    fn original_task_actor_accepts_actual_owned_std_child() -> io::Result<()> {
        let mut fixture = ActorFixture::spawn()?;
        let actor = OriginalTaskActor::from_std_child(&mut fixture.child, &expected_image()?)?;
        let mut connection = fixture.connect()?;
        assert!(!fixture.marker.exists());
        let mut admitted = actor.authenticate_peer(&connection)?;
        admitted.revalidate()?;
        connection.write_all(b"A")?;
        assert!(fixture.wait()?.success());
        assert_eq!(std::fs::read(&fixture.marker)?, b"started");
        assert!(admitted.revalidate().is_err());
        assert!(admitted.revalidate().is_err());
        Ok(())
    }

    #[test]
    fn original_task_actor_rejects_foreign_same_image_child() -> io::Result<()> {
        let mut original = ActorFixture::spawn()?;
        let mut foreign = ActorFixture::spawn()?;
        let actor = OriginalTaskActor::from_std_child(&mut original.child, &expected_image()?)?;
        let connection = foreign.connect()?;
        match actor.authenticate_peer(&connection) {
            Err(error) => assert_eq!(error.kind(), io::ErrorKind::PermissionDenied),
            Ok(_) => panic!("An unrelated actor was admitted"),
        }
        assert!(!original.marker.exists());
        assert!(!foreign.marker.exists());
        Ok(())
    }

    #[test]
    fn original_task_actor_rejects_cached_completed_child() -> io::Result<()> {
        let mut fixture = ActorFixture::spawn()?;
        let mut connection = fixture.connect()?;
        connection.write_all(b"Q")?;
        assert!(fixture.wait()?.success());
        assert!(OriginalTaskActor::from_std_child(&mut fixture.child, &expected_image()?).is_err());
        assert!(!fixture.marker.exists());
        Ok(())
    }

    #[test]
    fn original_task_actor_rejects_exit_before_admission() -> io::Result<()> {
        let mut fixture = ActorFixture::spawn()?;
        let actor = OriginalTaskActor::from_std_child(&mut fixture.child, &expected_image()?)?;
        let mut connection = fixture.connect()?;
        connection.write_all(b"Q")?;
        assert!(fixture.wait()?.success());
        assert!(actor.authenticate_peer(&connection).is_err());
        assert!(!fixture.marker.exists());
        Ok(())
    }

    #[test]
    fn original_task_actor_rejects_actual_exec_image_replacement() -> io::Result<()> {
        let mut fixture = ActorFixture::spawn()?;
        let mut actor = OriginalTaskActor::from_std_child(&mut fixture.child, &expected_image()?)?;
        let mut connection = fixture.connect()?;
        connection.write_all(b"E")?;
        let deadline = Instant::now() + Duration::from_secs(5);
        while actor.revalidate().is_ok() {
            if Instant::now() >= deadline {
                return Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    "Actor image did not change",
                ));
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(fixture.child.try_wait()?.is_none());
        assert!(actor.authenticate_peer(&connection).is_err());
        assert!(!fixture.marker.exists());
        Ok(())
    }

    #[test]
    fn original_task_actor_accepts_actual_owned_smol_child() -> io::Result<()> {
        let directory = tempfile::tempdir()?;
        let socket = directory.path().join("actor.sock");
        let marker = directory.path().join("workload-started");
        let listener = UnixListener::bind(&socket)?;
        listener.set_nonblocking(true)?;
        let mut command = smol::process::Command::new(std::env::current_exe()?);
        command
            .args([
                "--exact",
                "task_actor::tests::original_task_actor_fixture",
                "--nocapture",
            ])
            .env("ZED_TEST_TASK_ACTOR_SOCKET", socket)
            .env("ZED_TEST_TASK_ACTOR_MARKER", &marker)
            .stdin(Stdio::null())
            .kill_on_drop(true)
            .reap_on_drop(true);
        let mut child = command.spawn()?;
        let actor = OriginalTaskActor::from_smol_child(&mut child, &expected_image()?)?;
        let deadline = Instant::now() + Duration::from_secs(5);
        let mut connection = loop {
            match listener.accept() {
                Ok((connection, _)) => break connection,
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => {}
                Err(error) => return Err(error),
            }
            if child.try_status()?.is_some() || Instant::now() >= deadline {
                return Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    "Smol actor did not connect",
                ));
            }
            std::thread::sleep(Duration::from_millis(5));
        };
        connection.set_write_timeout(Some(Duration::from_secs(5)))?;
        assert!(!marker.exists());
        let mut admitted = actor.authenticate_peer(&connection)?;
        admitted.revalidate()?;
        connection.write_all(b"A")?;
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            if let Some(status) = child.try_status()? {
                assert!(status.success());
                break;
            }
            if Instant::now() >= deadline {
                return Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    "Smol actor did not exit",
                ));
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(std::fs::read(marker)?, b"started");
        assert!(admitted.revalidate().is_err());
        assert!(admitted.revalidate().is_err());
        Ok(())
    }

    #[test]
    fn original_task_actor_rejects_running_path_replaced_with_expected_image() -> io::Result<()> {
        let directory = tempfile::tempdir()?;
        let image = directory.path().join("original-image");
        let replacement = directory.path().join("replacement-image");
        std::fs::copy(std::env::current_exe()?, &image)?;
        std::fs::copy("/bin/sleep", &replacement)?;
        let mut fixture = ActorFixture::spawn_image(&image, &[])?;
        let mut connection = fixture.connect()?;
        std::fs::rename(replacement, &image)?;
        let expected_replacement = File::open(&image)?;
        match OriginalTaskActor::from_std_child(&mut fixture.child, &expected_replacement) {
            Err(error) => assert_eq!(error.kind(), io::ErrorKind::PermissionDenied),
            Ok(_) => panic!("A pathname replacement was treated as the running image"),
        }
        assert!(fixture.child.try_wait()?.is_none());
        assert!(!fixture.marker.exists());
        connection.write_all(b"Q")?;
        assert!(fixture.wait()?.success());
        Ok(())
    }

    #[test]
    fn original_task_actor_failed_then_restored_image_cannot_renew() -> io::Result<()> {
        let directory = tempfile::tempdir()?;
        let original_image = std::env::current_exe()?;
        let second_image = directory.path().join("second-image");
        std::fs::copy(&original_image, &second_image)?;
        let mut fixture = ActorFixture::spawn_image(
            &original_image,
            &[
                ("ZED_TEST_TASK_ACTOR_NEXT_IMAGE", &second_image),
                ("ZED_TEST_TASK_ACTOR_RETURN_IMAGE", &original_image),
            ],
        )?;
        let actor = OriginalTaskActor::from_std_child(&mut fixture.child, &expected_image()?)?;
        let mut original_connection = fixture.connect()?;
        let mut admitted = actor.authenticate_peer(&original_connection)?;
        original_connection.write_all(b"R")?;
        let mut second_connection = fixture.connect()?;
        assert_eq!(
            peer_identity(second_connection.as_raw_fd())?.process_id,
            fixture.child.id()
        );
        match admitted.revalidate() {
            Err(error) => assert_eq!(error.kind(), io::ErrorKind::PermissionDenied),
            Ok(()) => panic!("An exec to a different executable did not retire the lease"),
        }
        second_connection.write_all(b"R")?;
        let mut restored_connection = fixture.connect()?;
        assert_eq!(
            peer_identity(restored_connection.as_raw_fd())?.process_id,
            fixture.child.id()
        );
        // Check actual kernel image identity has returned to the original. The
        // following refusal must be sticky failure, not an image still changing.
        assert!(
            ImageIdentity::read(&admitted.actor.lifetime.executable()?)?
                == admitted.actor.image_identity
        );
        assert!(fixture.child.try_wait()?.is_none());
        assert!(admitted.revalidate().is_err());
        assert!(admitted.revalidate().is_err());
        assert!(!fixture.marker.exists());
        restored_connection.write_all(b"Q")?;
        assert!(fixture.wait()?.success());
        Ok(())
    }

    type SmolActorStatus = std::pin::Pin<
        Box<dyn std::future::Future<Output = io::Result<std::process::ExitStatus>> + Send>,
    >;

    struct SmolActorFixture {
        child: smol::process::Child,
        listener: UnixListener,
        marker: PathBuf,
        _directory: tempfile::TempDir,
    }

    impl SmolActorFixture {
        fn spawn() -> io::Result<Self> {
            let directory = tempfile::tempdir()?;
            let socket = directory.path().join("actor.sock");
            let marker = directory.path().join("workload-started");
            let listener = UnixListener::bind(&socket)?;
            listener.set_nonblocking(true)?;
            let mut command = smol::process::Command::new(std::env::current_exe()?);
            command
                .args([
                    "--exact",
                    "task_actor::tests::original_task_actor_fixture",
                    "--nocapture",
                ])
                .env("ZED_TEST_TASK_ACTOR_SOCKET", socket)
                .env("ZED_TEST_TASK_ACTOR_MARKER", &marker)
                .env("ZED_TEST_TASK_ACTOR_ROLE", "actor")
                .stdin(Stdio::null())
                .kill_on_drop(true)
                .reap_on_drop(true);
            Ok(Self {
                child: command.spawn()?,
                listener,
                marker,
                _directory: directory,
            })
        }

        fn connect(&mut self) -> io::Result<std::os::unix::net::UnixStream> {
            let deadline = Instant::now() + Duration::from_secs(5);
            loop {
                match self.listener.accept() {
                    Ok((stream, _)) => {
                        stream.set_read_timeout(Some(Duration::from_secs(5)))?;
                        stream.set_write_timeout(Some(Duration::from_secs(5)))?;
                        return Ok(stream);
                    }
                    Err(error) if error.kind() == io::ErrorKind::WouldBlock => {}
                    Err(error) => return Err(error),
                }
                if self.child.try_status()?.is_some() || Instant::now() >= deadline {
                    return Err(io::Error::new(
                        io::ErrorKind::TimedOut,
                        "Smol actor did not connect",
                    ));
                }
                std::thread::sleep(Duration::from_millis(5));
            }
        }

        fn pin_before_status_wait(
            &mut self,
            image: &File,
        ) -> io::Result<(OriginalTaskActor, SmolActorStatus)> {
            let actor = OriginalTaskActor::from_smol_child(&mut self.child, image)?;
            let status = Box::pin(self.child.status());
            Ok((actor, status))
        }
    }

    impl Drop for SmolActorFixture {
        fn drop(&mut self) {
            match self.child.try_status() {
                Ok(Some(_)) => return,
                Ok(None) => {}
                Err(error) => eprintln!("Smol actor status failed during cleanup: {error}"),
            }
            if let Err(error) = self.child.kill() {
                eprintln!("Smol actor kill failed during cleanup: {error}");
            }
            if let Err(error) = wait_smol_status(self.child.status()) {
                eprintln!("Smol actor completion unverified during cleanup: {error}");
            }
        }
    }

    fn wait_smol_status(
        status: impl std::future::Future<Output = io::Result<std::process::ExitStatus>>,
    ) -> io::Result<std::process::ExitStatus> {
        smol::block_on(smol::future::race(status, async {
            // This standalone process fixture has no GPUI executor to own its timeout.
            smol::unblock(|| std::thread::sleep(Duration::from_secs(5))).await;
            Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "Smol actor exit timed out",
            ))
        }))
    }

    #[test]
    fn original_task_actor_rejects_actual_smol_cached_completion() -> io::Result<()> {
        let mut fixture = SmolActorFixture::spawn()?;
        let mut connection = fixture.connect()?;
        connection.write_all(b"Q")?;
        let status = wait_smol_status(fixture.child.status())?;
        assert!(status.success());
        assert!(fixture.child.try_status()?.is_some());
        match OriginalTaskActor::from_smol_child(&mut fixture.child, &expected_image()?) {
            Err(error) => assert_eq!(error.kind(), io::ErrorKind::NotConnected),
            Ok(_) => panic!("A cached completed smol Child was admitted"),
        }
        assert!(!fixture.marker.exists());
        Ok(())
    }

    #[test]
    fn original_task_actor_earlier_status_future_cannot_admit_reaped_child() -> io::Result<()> {
        let mut fixture = SmolActorFixture::spawn()?;
        let original_process = fixture.child.id();
        // The real async-process future retains its shared Child without this borrow.
        // This deliberately violates the required spawn-owner ordering.
        let earlier_status = fixture.child.status();
        let mut connection = fixture.connect()?;
        assert_eq!(
            peer_identity(connection.as_raw_fd())?.process_id,
            original_process
        );
        connection.write_all(b"Q")?;
        assert!(wait_smol_status(earlier_status)?.success());
        assert!(fixture.child.try_status()?.is_some());
        match OriginalTaskActor::from_smol_child(&mut fixture.child, &expected_image()?) {
            Err(error) => assert_eq!(error.kind(), io::ErrorKind::NotConnected),
            Ok(_) => panic!("An earlier waiter's completed original child was admitted"),
        }
        assert!(!fixture.marker.exists());
        Ok(())
    }

    #[test]
    fn original_task_actor_pin_precedes_its_smol_status_wait() -> io::Result<()> {
        let mut fixture = SmolActorFixture::spawn()?;
        let (actor, status) = fixture.pin_before_status_wait(&expected_image()?)?;
        let mut connection = fixture.connect()?;
        assert_eq!(
            peer_identity(connection.as_raw_fd())?.process_id,
            fixture.child.id()
        );
        let mut admitted = actor.authenticate_peer(&connection)?;
        admitted.revalidate()?;
        assert!(!fixture.marker.exists());
        connection.write_all(b"A")?;
        assert!(wait_smol_status(status)?.success());
        assert_eq!(std::fs::read(&fixture.marker)?, b"started");
        assert!(fixture.child.try_status()?.is_some());
        assert!(admitted.revalidate().is_err());
        assert!(admitted.revalidate().is_err());
        Ok(())
    }

    fn open_descriptor_count() -> io::Result<usize> {
        let count = std::fs::read_dir("/proc/self/fd")?
            .take(4097)
            .try_fold(0, |count, entry| entry.map(|_| count + 1))?;
        if count > 4096 {
            return Err(io::Error::other(
                "Actor descriptor census exceeds its limit",
            ));
        }
        Ok(count)
    }

    #[test]
    fn original_task_actor_observation_denials_release_owned_descriptors() -> io::Result<()> {
        let mut fixture = ActorFixture::spawn_image(
            &std::env::current_exe()?,
            &[("ZED_TEST_TASK_ACTOR_ROLE", Path::new("fd-observation"))],
        )?;
        let mut connection = fixture.connect()?;
        assert!(!fixture.marker.exists());
        connection.write_all(b"F")?;
        assert!(fixture.wait()?.success());
        assert_eq!(std::fs::read(&fixture.marker)?, b"fd-observation-complete");
        Ok(())
    }

    fn check_observation_denials_release_owned_descriptors() -> io::Result<()> {
        use super::linux::observation_failure::{Boundary, during};
        let mut fixture = ActorFixture::spawn()?;
        let image = expected_image()?;
        let mut connection = fixture.connect()?;
        for boundary in [
            Boundary::Pidfd,
            Boundary::ProcDirectory,
            Boundary::Executable,
        ] {
            let descriptors_before = open_descriptor_count()?;
            let (result, descriptors) = during(boundary, || {
                OriginalTaskActor::from_std_child(&mut fixture.child, &image)
            });
            match result {
                Err(error) => assert_eq!(error.raw_os_error(), Some(libc::EACCES)),
                Ok(_) => panic!("A denied kernel observation minted an actor"),
            }
            for descriptor in descriptors {
                let result = unsafe { libc::fcntl(descriptor, libc::F_GETFD) };
                assert_eq!(result, -1, "A refused actor retained an owned descriptor");
                assert_eq!(io::Error::last_os_error().raw_os_error(), Some(libc::EBADF));
            }
            assert_eq!(open_descriptor_count()?, descriptors_before);
            assert!(fixture.child.try_wait()?.is_none());
            assert!(!fixture.marker.exists());
        }
        connection.write_all(b"Q")?;
        assert!(fixture.wait()?.success());
        assert!(!fixture.marker.exists());
        let mut recovery = ActorFixture::spawn()?;
        let actor = OriginalTaskActor::from_std_child(&mut recovery.child, &image)?;
        let mut connection = recovery.connect()?;
        let mut admitted = actor.authenticate_peer(&connection)?;
        admitted.revalidate()?;
        connection.write_all(b"A")?;
        assert!(recovery.wait()?.success());
        assert_eq!(std::fs::read(&recovery.marker)?, b"started");
        assert!(admitted.revalidate().is_err());
        Ok(())
    }
}

#[cfg(all(test, target_os = "macos"))]
mod macos_tests {
    use super::*;

    #[test]
    fn original_task_actor_requires_macos_kernel_image_policy() -> io::Result<()> {
        let mut command = smol::process::Command::new("/bin/sleep");
        command.arg("5").kill_on_drop(true).reap_on_drop(true);
        let mut child = command.spawn()?;
        let image = File::open("/bin/sleep")?;
        match OriginalTaskActor::from_smol_child(&mut child, &image) {
            Err(error) => assert_eq!(error.kind(), io::ErrorKind::Unsupported),
            Ok(_) => panic!("macOS actor was claimed image-bound without kernel-image policy"),
        }
        child.kill()?;
        smol::block_on(child.status())?;
        Ok(())
    }
}
