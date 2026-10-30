use std::{error::Error, fmt, time::Duration};

use gpui::{Context, IntoElement, Render, Role, Task, Window, px, rgb};
use nostr_compat::pairing::{MAX_NIP_AB_QR_BYTES, NIP_AB_SESSION_MILLIS, PairingQr};
use qrcode::{EcLevel, QrCode, types::Color as QrColor};
use ui::{Button, ButtonStyle, prelude::*};
use zeroize::Zeroizing;

const MAX_PAIRING_UI_SESSION_ID_BYTES: usize = 128;
const MAX_PAIRING_CREDENTIAL_IDENTIFIER_BYTES: usize = 512;
const QR_CELL_PIXELS: f32 = 3.;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativePairingSessionId(String);

impl NativePairingSessionId {
    pub fn parse(value: impl Into<String>) -> Result<Self, NativePairingUiError> {
        let value = value.into();
        if value.is_empty()
            || value.len() > MAX_PAIRING_UI_SESSION_ID_BYTES
            || !value
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
        {
            return Err(NativePairingUiError::InvalidServiceResponse);
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

pub struct NativePairingSource {
    session_id: NativePairingSessionId,
    qr: PairingQr,
    expires_at_millis: u64,
    expires_in_millis: u64,
}

impl NativePairingSource {
    pub fn new(
        session_id: NativePairingSessionId,
        qr: PairingQr,
        expires_at_millis: u64,
        expires_in_millis: u64,
    ) -> Result<Self, NativePairingUiError> {
        validate_expiry(expires_at_millis, expires_in_millis)?;
        Ok(Self {
            session_id,
            qr,
            expires_at_millis,
            expires_in_millis,
        })
    }
}

impl fmt::Debug for NativePairingSource {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("NativePairingSource")
            .field("session_id", &self.session_id)
            .field("qr", &"[REDACTED]")
            .field("expires_at_millis", &self.expires_at_millis)
            .field("expires_in_millis", &self.expires_in_millis)
            .finish()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativePairingConfirmation {
    session_id: NativePairingSessionId,
    sas_code: String,
    expires_at_millis: u64,
    expires_in_millis: u64,
}

impl NativePairingConfirmation {
    pub fn new(
        session_id: NativePairingSessionId,
        sas_code: impl Into<String>,
        expires_at_millis: u64,
        expires_in_millis: u64,
    ) -> Result<Self, NativePairingUiError> {
        let sas_code = sas_code.into();
        if sas_code.len() != 6 || !sas_code.bytes().all(|byte| byte.is_ascii_digit()) {
            return Err(NativePairingUiError::InvalidServiceResponse);
        }
        validate_expiry(expires_at_millis, expires_in_millis)?;
        Ok(Self {
            session_id,
            sas_code,
            expires_at_millis,
            expires_in_millis,
        })
    }

    pub fn session_id(&self) -> &NativePairingSessionId {
        &self.session_id
    }

    pub fn sas_code(&self) -> &str {
        &self.sas_code
    }

    pub const fn expires_at_millis(&self) -> u64 {
        self.expires_at_millis
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativePairingImportReceipt {
    credential_identifier: String,
    public_key: [u8; 32],
}

impl NativePairingImportReceipt {
    pub fn new(
        credential_identifier: impl Into<String>,
        public_key: [u8; 32],
    ) -> Result<Self, NativePairingUiError> {
        let credential_identifier = credential_identifier.into();
        if credential_identifier.trim().is_empty()
            || credential_identifier.len() > MAX_PAIRING_CREDENTIAL_IDENTIFIER_BYTES
            || credential_identifier.chars().any(char::is_control)
            || public_key == [0; 32]
        {
            return Err(NativePairingUiError::InvalidServiceResponse);
        }
        Ok(Self {
            credential_identifier,
            public_key,
        })
    }

    pub fn credential_identifier(&self) -> &str {
        &self.credential_identifier
    }

    pub const fn public_key(&self) -> [u8; 32] {
        self.public_key
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativePairingServiceError {
    Unavailable,
    AuthorizationDenied,
    Expired,
    Conflict,
    LockedKeyring,
    ImportFailed,
    InvalidResponse,
}

pub trait NativePairingService {
    fn create_source(&mut self) -> Result<NativePairingSource, NativePairingServiceError>;

    fn preview_import(
        &mut self,
        qr: &PairingQr,
    ) -> Result<NativePairingConfirmation, NativePairingServiceError>;

    fn confirm_import(
        &mut self,
        session_id: &NativePairingSessionId,
    ) -> Result<NativePairingImportReceipt, NativePairingServiceError>;

    fn cancel(
        &mut self,
        session_id: &NativePairingSessionId,
    ) -> Result<(), NativePairingServiceError>;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativePairingPhase {
    Idle,
    Scanning,
    DisplayingQr,
    AwaitingConfirmation,
    Completed,
    Cancelled,
    Expired,
    Failed,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativePairingNotice {
    CorruptQr,
    ServiceUnavailable,
    AuthorizationDenied,
    SessionChanged,
    LockedKeyring,
    ImportFailed,
    InvalidServiceResponse,
}

impl NativePairingNotice {
    const fn label(self) -> &'static str {
        match self {
            Self::CorruptQr => "That pairing QR code is invalid. Scan a new code.",
            Self::ServiceUnavailable => "Pairing is temporarily unavailable. Try again.",
            Self::AuthorizationDenied => "Pairing was not authorized.",
            Self::SessionChanged => "The pairing session expired or changed. Start again.",
            Self::LockedKeyring => {
                "Unlock the system keyring, then retry this exact identity import."
            }
            Self::ImportFailed => "The paired identity could not be imported safely.",
            Self::InvalidServiceResponse => "Pairing returned an invalid response. Start again.",
        }
    }
}

struct PairingQrMatrix {
    width: usize,
    cells: Vec<bool>,
}

impl PairingQrMatrix {
    fn from_qr(qr: &PairingQr) -> Result<Self, NativePairingUiError> {
        let encoded = Zeroizing::new(
            qr.encode()
                .map_err(|_| NativePairingUiError::InvalidServiceResponse)?,
        );
        if encoded.len() > MAX_NIP_AB_QR_BYTES {
            return Err(NativePairingUiError::InvalidServiceResponse);
        }
        let code = QrCode::with_error_correction_level(encoded.as_bytes(), EcLevel::M)
            .map_err(|_| NativePairingUiError::InvalidServiceResponse)?;
        let width = code.width();
        let cells = code
            .to_colors()
            .into_iter()
            .map(|color| color == QrColor::Dark)
            .collect::<Vec<_>>();
        if width == 0 || cells.len() != width.saturating_mul(width) {
            return Err(NativePairingUiError::InvalidServiceResponse);
        }
        Ok(Self { width, cells })
    }

    #[cfg(test)]
    fn dark_cell_count(&self) -> usize {
        self.cells.iter().filter(|cell| **cell).count()
    }
}

impl fmt::Debug for PairingQrMatrix {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PairingQrMatrix")
            .field("width", &self.width)
            .field("cells", &"[REDACTED]")
            .finish()
    }
}

struct DisplayedSource {
    session_id: NativePairingSessionId,
    matrix: PairingQrMatrix,
}

enum NativePairingState {
    Idle,
    Scanning,
    Displaying(DisplayedSource),
    Confirming(NativePairingConfirmation),
    Completed(NativePairingImportReceipt),
    Cancelled,
    Expired,
    Failed,
}

pub struct NativePairingView {
    service: Box<dyn NativePairingService>,
    state: NativePairingState,
    notice: Option<NativePairingNotice>,
    expiry_task: Task<()>,
}

impl NativePairingView {
    pub fn new(service: impl NativePairingService + 'static) -> Self {
        Self {
            service: Box::new(service),
            state: NativePairingState::Idle,
            notice: None,
            expiry_task: Task::ready(()),
        }
    }

    pub const fn phase(&self) -> NativePairingPhase {
        match self.state {
            NativePairingState::Idle => NativePairingPhase::Idle,
            NativePairingState::Scanning => NativePairingPhase::Scanning,
            NativePairingState::Displaying(_) => NativePairingPhase::DisplayingQr,
            NativePairingState::Confirming(_) => NativePairingPhase::AwaitingConfirmation,
            NativePairingState::Completed(_) => NativePairingPhase::Completed,
            NativePairingState::Cancelled => NativePairingPhase::Cancelled,
            NativePairingState::Expired => NativePairingPhase::Expired,
            NativePairingState::Failed => NativePairingPhase::Failed,
        }
    }

    pub const fn notice(&self) -> Option<NativePairingNotice> {
        self.notice
    }

    pub fn confirmation(&self) -> Option<&NativePairingConfirmation> {
        match &self.state {
            NativePairingState::Confirming(confirmation) => Some(confirmation),
            _ => None,
        }
    }

    pub fn completion(&self) -> Option<&NativePairingImportReceipt> {
        match &self.state {
            NativePairingState::Completed(receipt) => Some(receipt),
            _ => None,
        }
    }

    pub fn create_source(&mut self, cx: &mut Context<Self>) -> Result<(), NativePairingUiError> {
        if self.active_session_id().is_some() {
            return Err(NativePairingUiError::SessionActive);
        }
        let source = self.service.create_source().map_err(|error| {
            self.fail(error);
            NativePairingUiError::Service(error)
        })?;
        if validate_expiry(source.expires_at_millis, source.expires_in_millis).is_err() {
            self.fail(NativePairingServiceError::InvalidResponse);
            cx.notify();
            return Err(NativePairingUiError::InvalidServiceResponse);
        }
        let matrix = PairingQrMatrix::from_qr(&source.qr).inspect_err(|_| {
            self.fail(NativePairingServiceError::InvalidResponse);
            cx.notify();
        })?;
        let expires_in_millis = source.expires_in_millis;
        self.state = NativePairingState::Displaying(DisplayedSource {
            session_id: source.session_id,
            matrix,
        });
        self.notice = None;
        self.schedule_expiry(expires_in_millis, cx);
        cx.notify();
        Ok(())
    }

    pub fn begin_scan(&mut self, cx: &mut Context<Self>) {
        if self.active_session_id().is_none() {
            self.state = NativePairingState::Scanning;
            self.notice = None;
            cx.notify();
        }
    }

    pub fn submit_scanned_qr(
        &mut self,
        raw_qr: &str,
        cx: &mut Context<Self>,
    ) -> Result<(), NativePairingUiError> {
        if !matches!(
            self.state,
            NativePairingState::Scanning | NativePairingState::Failed
        ) {
            return Err(NativePairingUiError::NotScanning);
        }
        let qr = PairingQr::parse(raw_qr).map_err(|_| {
            self.state = NativePairingState::Failed;
            self.notice = Some(NativePairingNotice::CorruptQr);
            cx.notify();
            NativePairingUiError::CorruptQr
        })?;
        let confirmation = self.service.preview_import(&qr).map_err(|error| {
            self.fail(error);
            cx.notify();
            NativePairingUiError::Service(error)
        })?;
        if validate_expiry(
            confirmation.expires_at_millis,
            confirmation.expires_in_millis,
        )
        .is_err()
        {
            self.fail(NativePairingServiceError::InvalidResponse);
            cx.notify();
            return Err(NativePairingUiError::InvalidServiceResponse);
        }
        let expires_in_millis = confirmation.expires_in_millis;
        self.state = NativePairingState::Confirming(confirmation);
        self.notice = None;
        self.schedule_expiry(expires_in_millis, cx);
        cx.notify();
        Ok(())
    }

    pub fn confirm_import(&mut self, cx: &mut Context<Self>) -> Result<(), NativePairingUiError> {
        let session_id = self
            .confirmation()
            .map(|confirmation| confirmation.session_id.clone())
            .ok_or(NativePairingUiError::ConfirmationUnavailable)?;
        match self.service.confirm_import(&session_id) {
            Ok(receipt) => {
                self.state = NativePairingState::Completed(receipt);
                self.notice = None;
                self.expiry_task = Task::ready(());
                cx.notify();
                Ok(())
            }
            Err(error) => {
                if error == NativePairingServiceError::LockedKeyring {
                    self.notice = Some(NativePairingNotice::LockedKeyring);
                } else {
                    self.fail(error);
                }
                cx.notify();
                Err(NativePairingUiError::Service(error))
            }
        }
    }

    pub fn cancel(&mut self, cx: &mut Context<Self>) -> Result<(), NativePairingUiError> {
        if let Some(session_id) = self.active_session_id().cloned() {
            if let Err(error) = self.service.cancel(&session_id) {
                self.notice = Some(map_notice(error));
                cx.notify();
                return Err(NativePairingUiError::Service(error));
            }
        } else if !matches!(self.state, NativePairingState::Scanning) {
            return Err(NativePairingUiError::SessionInactive);
        }
        self.state = NativePairingState::Cancelled;
        self.notice = None;
        self.expiry_task = Task::ready(());
        cx.notify();
        Ok(())
    }

    pub fn restart_scan(&mut self, cx: &mut Context<Self>) {
        if matches!(self.state, NativePairingState::Failed) {
            self.state = NativePairingState::Scanning;
            self.notice = None;
            cx.notify();
        }
    }

    fn active_session_id(&self) -> Option<&NativePairingSessionId> {
        match &self.state {
            NativePairingState::Displaying(source) => Some(&source.session_id),
            NativePairingState::Confirming(confirmation) => Some(&confirmation.session_id),
            _ => None,
        }
    }

    fn fail(&mut self, error: NativePairingServiceError) {
        self.state = NativePairingState::Failed;
        self.notice = Some(map_notice(error));
        self.expiry_task = Task::ready(());
    }

    fn schedule_expiry(&mut self, expires_in_millis: u64, cx: &mut Context<Self>) {
        self.expiry_task = cx.spawn(async move |this, cx| {
            cx.background_executor()
                .timer(Duration::from_millis(expires_in_millis))
                .await;
            if this
                .update(cx, |this, cx| {
                    if this.active_session_id().is_some() {
                        this.state = NativePairingState::Expired;
                        this.notice = None;
                        cx.notify();
                    }
                })
                .is_err()
            {
                return;
            }
        });
    }
}

impl Render for NativePairingView {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        v_flex()
            .id("native-pairing-flow")
            .role(Role::Region)
            .aria_label("Secure device pairing")
            .size_full()
            .gap_3()
            .p_3()
            .child(div().text_ui(cx).child("Pair a device"))
            .when_some(self.notice, |this, notice| {
                this.child(
                    v_flex()
                        .id("native-pairing-notice")
                        .role(Role::Alert)
                        .aria_label(notice.label())
                        .gap_1()
                        .child(notice.label())
                        .when(matches!(self.state, NativePairingState::Failed), |this| {
                            this.child(
                                Button::new("native-pairing-retry-scan", "Scan a new code")
                                    .style(ButtonStyle::Subtle)
                                    .on_click(cx.listener(|this, _, _window, cx| {
                                        this.restart_scan(cx);
                                    })),
                            )
                        })
                        .when(notice == NativePairingNotice::LockedKeyring, |this| {
                            this.child(
                                Button::new("native-pairing-retry-import", "Retry import")
                                    .style(ButtonStyle::Filled)
                                    .on_click(cx.listener(|this, _, _window, cx| {
                                        if this.confirm_import(cx).is_err() {
                                            cx.notify();
                                        }
                                    })),
                            )
                        }),
                )
            })
            .child(self.render_state(cx))
    }
}

impl NativePairingView {
    fn render_state(&self, cx: &mut Context<Self>) -> gpui::AnyElement {
        match &self.state {
            NativePairingState::Idle => v_flex()
                .gap_2()
                .child("Show a QR code to send an identity, or scan one to import an identity.")
                .child(
                    h_flex()
                        .gap_1()
                        .child(
                            Button::new("native-pairing-create", "Show pairing QR")
                                .style(ButtonStyle::Filled)
                                .on_click(cx.listener(|this, _, _window, cx| {
                                    if this.create_source(cx).is_err() {
                                        cx.notify();
                                    }
                                })),
                        )
                        .child(
                            Button::new("native-pairing-scan", "Scan pairing QR")
                                .style(ButtonStyle::Subtle)
                                .on_click(cx.listener(|this, _, _window, cx| {
                                    this.begin_scan(cx);
                                })),
                        ),
                )
                .into_any_element(),
            NativePairingState::Scanning => v_flex()
                .id("native-pairing-scanner")
                .role(Role::Status)
                .aria_label("Pairing QR scanner ready")
                .gap_2()
                .child("Align the pairing QR code inside the scanner.")
                .child(
                    Button::new("native-pairing-cancel-scan", "Cancel")
                        .style(ButtonStyle::Subtle)
                        .on_click(cx.listener(|this, _, _window, cx| {
                            if this.cancel(cx).is_err() {
                                cx.notify();
                            }
                        })),
                )
                .into_any_element(),
            NativePairingState::Displaying(source) => v_flex()
                .id("native-pairing-display")
                .role(Role::Status)
                .aria_label("Pairing QR code. Expires in two minutes")
                .gap_2()
                .items_center()
                .child("Scan this code on the device receiving the identity.")
                .child(render_qr_matrix(&source.matrix))
                .child("This code expires after two minutes.")
                .child(
                    Button::new("native-pairing-cancel-display", "Cancel pairing")
                        .style(ButtonStyle::Subtle)
                        .on_click(cx.listener(|this, _, _window, cx| {
                            if this.cancel(cx).is_err() {
                                cx.notify();
                            }
                        })),
                )
                .into_any_element(),
            NativePairingState::Confirming(confirmation) => v_flex()
                .id("native-pairing-confirmation")
                .role(Role::Region)
                .aria_label("Confirm pairing security code before importing the identity")
                .gap_2()
                .items_center()
                .child("Confirm that both devices show this security code:")
                .child(format_sas(confirmation.sas_code()))
                .child("This confirmation expires with the pairing session.")
                .child(
                    h_flex()
                        .gap_1()
                        .child(
                            Button::new("native-pairing-confirm", "Codes match — import")
                                .style(ButtonStyle::Filled)
                                .on_click(cx.listener(|this, _, _window, cx| {
                                    if this.confirm_import(cx).is_err() {
                                        cx.notify();
                                    }
                                })),
                        )
                        .child(
                            Button::new("native-pairing-deny", "Cancel pairing")
                                .style(ButtonStyle::Subtle)
                                .on_click(cx.listener(|this, _, _window, cx| {
                                    if this.cancel(cx).is_err() {
                                        cx.notify();
                                    }
                                })),
                        ),
                )
                .into_any_element(),
            NativePairingState::Completed(receipt) => v_flex()
                .id("native-pairing-complete")
                .role(Role::Status)
                .aria_label("Paired identity imported securely")
                .gap_1()
                .child("Identity imported")
                .child(public_key_fingerprint(receipt.public_key()))
                .into_any_element(),
            NativePairingState::Cancelled => div()
                .id("native-pairing-cancelled")
                .role(Role::Status)
                .aria_label("Pairing cancelled")
                .child("Pairing cancelled")
                .into_any_element(),
            NativePairingState::Expired => div()
                .id("native-pairing-expired")
                .role(Role::Alert)
                .aria_label("Pairing session expired. Start again")
                .child("Pairing session expired. Start again.")
                .into_any_element(),
            NativePairingState::Failed => div()
                .id("native-pairing-failed")
                .role(Role::Status)
                .aria_label("Pairing stopped safely")
                .child("No identity was imported.")
                .into_any_element(),
        }
    }
}

fn render_qr_matrix(matrix: &PairingQrMatrix) -> impl IntoElement {
    v_flex()
        .id("native-pairing-qr")
        .role(Role::Image)
        .aria_label("Pairing QR code")
        .p_2()
        .bg(rgb(0xffffff))
        .children(matrix.cells.chunks(matrix.width).map(|row| {
            h_flex().children(row.iter().map(|dark| {
                div().size(px(QR_CELL_PIXELS)).flex_none().bg(rgb(if *dark {
                    0x000000
                } else {
                    0xffffff
                }))
            }))
        }))
}

fn format_sas(sas_code: &str) -> String {
    format!("{} {}", &sas_code[..3], &sas_code[3..])
}

fn public_key_fingerprint(public_key: [u8; 32]) -> String {
    let prefix = public_key[..4]
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    format!("Public key {prefix}…")
}

fn validate_expiry(
    expires_at_millis: u64,
    expires_in_millis: u64,
) -> Result<(), NativePairingUiError> {
    if expires_at_millis == 0 || expires_in_millis == 0 || expires_in_millis > NIP_AB_SESSION_MILLIS
    {
        return Err(NativePairingUiError::InvalidServiceResponse);
    }
    Ok(())
}

const fn map_notice(error: NativePairingServiceError) -> NativePairingNotice {
    match error {
        NativePairingServiceError::Unavailable => NativePairingNotice::ServiceUnavailable,
        NativePairingServiceError::AuthorizationDenied => NativePairingNotice::AuthorizationDenied,
        NativePairingServiceError::Expired | NativePairingServiceError::Conflict => {
            NativePairingNotice::SessionChanged
        }
        NativePairingServiceError::LockedKeyring => NativePairingNotice::LockedKeyring,
        NativePairingServiceError::ImportFailed => NativePairingNotice::ImportFailed,
        NativePairingServiceError::InvalidResponse => NativePairingNotice::InvalidServiceResponse,
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativePairingUiError {
    CorruptQr,
    InvalidServiceResponse,
    SessionActive,
    SessionInactive,
    NotScanning,
    ConfirmationUnavailable,
    Service(NativePairingServiceError),
}

impl fmt::Display for NativePairingUiError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::CorruptQr => "pairing QR code is invalid",
            Self::InvalidServiceResponse => "pairing service response is invalid",
            Self::SessionActive => "a pairing session is already active",
            Self::SessionInactive => "no pairing session is active",
            Self::NotScanning => "pairing scanner is not active",
            Self::ConfirmationUnavailable => "pairing confirmation is unavailable",
            Self::Service(_) => "pairing service request failed",
        })
    }
}

impl Error for NativePairingUiError {}

#[cfg(test)]
mod tests {
    use std::{
        cell::{Cell, RefCell},
        collections::VecDeque,
        rc::Rc,
    };

    use gpui::TestAppContext;

    use super::*;

    const VALID_QR: &str = "nostrpair://79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798?secret=0101010101010101010101010101010101010101010101010101010101010101&relay=wss%3A%2F%2Frelay.example.com%2F&v=1";

    #[derive(Default)]
    struct ServiceState {
        created: RefCell<VecDeque<Result<NativePairingSource, NativePairingServiceError>>>,
        previews: RefCell<VecDeque<Result<NativePairingConfirmation, NativePairingServiceError>>>,
        imports: RefCell<VecDeque<Result<NativePairingImportReceipt, NativePairingServiceError>>>,
        preview_calls: Cell<usize>,
        import_calls: Cell<usize>,
        cancelled: RefCell<Vec<String>>,
    }

    #[derive(Clone)]
    struct QueueService(Rc<ServiceState>);

    impl NativePairingService for QueueService {
        fn create_source(&mut self) -> Result<NativePairingSource, NativePairingServiceError> {
            self.0
                .created
                .borrow_mut()
                .pop_front()
                .ok_or(NativePairingServiceError::InvalidResponse)?
        }

        fn preview_import(
            &mut self,
            _qr: &PairingQr,
        ) -> Result<NativePairingConfirmation, NativePairingServiceError> {
            self.0.preview_calls.set(self.0.preview_calls.get() + 1);
            self.0
                .previews
                .borrow_mut()
                .pop_front()
                .ok_or(NativePairingServiceError::InvalidResponse)?
        }

        fn confirm_import(
            &mut self,
            _session_id: &NativePairingSessionId,
        ) -> Result<NativePairingImportReceipt, NativePairingServiceError> {
            self.0.import_calls.set(self.0.import_calls.get() + 1);
            self.0
                .imports
                .borrow_mut()
                .pop_front()
                .ok_or(NativePairingServiceError::InvalidResponse)?
        }

        fn cancel(
            &mut self,
            session_id: &NativePairingSessionId,
        ) -> Result<(), NativePairingServiceError> {
            self.0
                .cancelled
                .borrow_mut()
                .push(session_id.as_str().to_owned());
            Ok(())
        }
    }

    fn session_id() -> NativePairingSessionId {
        NativePairingSessionId::parse("pairing-session-1").expect("session ID")
    }

    fn qr() -> PairingQr {
        PairingQr::parse(VALID_QR).expect("valid QR")
    }

    fn source() -> NativePairingSource {
        NativePairingSource::new(session_id(), qr(), 120_100, NIP_AB_SESSION_MILLIS)
            .expect("source")
    }

    fn confirmation() -> NativePairingConfirmation {
        NativePairingConfirmation::new(session_id(), "123456", 120_100, NIP_AB_SESSION_MILLIS)
            .expect("confirmation")
    }

    fn receipt() -> NativePairingImportReceipt {
        NativePairingImportReceipt::new("zed-nostr://credential/v1/imported", [7; 32])
            .expect("receipt")
    }

    fn service(
        created: Vec<Result<NativePairingSource, NativePairingServiceError>>,
        previews: Vec<Result<NativePairingConfirmation, NativePairingServiceError>>,
        imports: Vec<Result<NativePairingImportReceipt, NativePairingServiceError>>,
    ) -> (QueueService, Rc<ServiceState>) {
        let state = Rc::new(ServiceState {
            created: RefCell::new(created.into()),
            previews: RefCell::new(previews.into()),
            imports: RefCell::new(imports.into()),
            ..ServiceState::default()
        });
        (QueueService(state.clone()), state)
    }

    #[gpui::test]
    fn pairing_displays_a_real_qr_and_scans_into_confirmation(cx: &mut TestAppContext) {
        let (source_service, _) = service(vec![Ok(source())], Vec::new(), Vec::new());
        let source_view = cx.new(|_| NativePairingView::new(source_service));
        source_view
            .update(cx, NativePairingView::create_source)
            .expect("display source QR");
        assert_eq!(
            source_view.read_with(cx, |view, _| view.phase()),
            NativePairingPhase::DisplayingQr
        );
        source_view.read_with(cx, |view, _| {
            let NativePairingState::Displaying(source) = &view.state else {
                panic!("display state")
            };
            assert!(source.matrix.width >= 21);
            assert!(source.matrix.dark_cell_count() > 0);
            assert!(source.matrix.dark_cell_count() < source.matrix.cells.len());
        });

        let (target_service, state) = service(Vec::new(), vec![Ok(confirmation())], Vec::new());
        let target_view = cx.new(|_| NativePairingView::new(target_service));
        target_view.update(cx, NativePairingView::begin_scan);
        target_view
            .update(cx, |view, cx| view.submit_scanned_qr(VALID_QR, cx))
            .expect("scan QR");
        assert_eq!(
            target_view.read_with(cx, |view, _| view.phase()),
            NativePairingPhase::AwaitingConfirmation
        );
        assert_eq!(state.preview_calls.get(), 1);
    }

    #[gpui::test]
    fn pairing_expires_on_the_gpui_scheduler_boundary(cx: &mut TestAppContext) {
        let (service, _) = service(vec![Ok(source())], Vec::new(), Vec::new());
        let view = cx.new(|_| NativePairingView::new(service));
        view.update(cx, NativePairingView::create_source)
            .expect("display source QR");

        cx.executor()
            .advance_clock(Duration::from_millis(NIP_AB_SESSION_MILLIS - 1));
        cx.run_until_parked();
        assert_eq!(
            view.read_with(cx, |view, _| view.phase()),
            NativePairingPhase::DisplayingQr
        );
        cx.executor().advance_clock(Duration::from_millis(1));
        cx.run_until_parked();
        assert_eq!(
            view.read_with(cx, |view, _| view.phase()),
            NativePairingPhase::Expired
        );
    }

    #[gpui::test]
    fn corrupt_qr_fails_before_the_pairing_service(cx: &mut TestAppContext) {
        let (service, state) = service(Vec::new(), Vec::new(), Vec::new());
        let view = cx.new(|_| NativePairingView::new(service));
        view.update(cx, NativePairingView::begin_scan);

        assert_eq!(
            view.update(cx, |view, cx| {
                view.submit_scanned_qr("nostrpair://corrupt", cx)
            }),
            Err(NativePairingUiError::CorruptQr)
        );
        assert_eq!(state.preview_calls.get(), 0);
        assert_eq!(
            view.read_with(cx, |view, _| (view.phase(), view.notice())),
            (
                NativePairingPhase::Failed,
                Some(NativePairingNotice::CorruptQr)
            )
        );
    }

    #[gpui::test]
    fn cancellation_is_terminal_and_cancels_the_exact_session(cx: &mut TestAppContext) {
        let (service, state) = service(vec![Ok(source())], Vec::new(), Vec::new());
        let view = cx.new(|_| NativePairingView::new(service));
        view.update(cx, NativePairingView::create_source)
            .expect("display source QR");

        view.update(cx, NativePairingView::cancel)
            .expect("cancel session");
        assert_eq!(
            view.read_with(cx, |view, _| view.phase()),
            NativePairingPhase::Cancelled
        );
        assert_eq!(state.cancelled.borrow().as_slice(), ["pairing-session-1"]);
        cx.executor()
            .advance_clock(Duration::from_millis(NIP_AB_SESSION_MILLIS));
        cx.run_until_parked();
        assert_eq!(
            view.read_with(cx, |view, _| view.phase()),
            NativePairingPhase::Cancelled
        );
    }

    #[gpui::test]
    fn import_requires_explicit_sas_confirmation(cx: &mut TestAppContext) {
        let (service, state) = service(Vec::new(), vec![Ok(confirmation())], vec![Ok(receipt())]);
        let view = cx.new(|_| NativePairingView::new(service));
        view.update(cx, NativePairingView::begin_scan);
        view.update(cx, |view, cx| view.submit_scanned_qr(VALID_QR, cx))
            .expect("scan QR");
        assert_eq!(state.import_calls.get(), 0);
        assert_eq!(
            view.read_with(cx, |view, _| view
                .confirmation()
                .map(|value| value.sas_code().to_owned())),
            Some("123456".to_owned())
        );

        view.update(cx, NativePairingView::confirm_import)
            .expect("confirmed import");
        assert_eq!(state.import_calls.get(), 1);
        assert_eq!(
            view.read_with(cx, |view, _| view.phase()),
            NativePairingPhase::Completed
        );
        assert_eq!(
            view.read_with(cx, |view, _| {
                view.completion()
                    .map(|receipt| receipt.credential_identifier().to_owned())
            }),
            Some("zed-nostr://credential/v1/imported".to_owned())
        );
    }

    #[gpui::test]
    fn locked_keyring_retains_confirmation_for_exact_retry(cx: &mut TestAppContext) {
        let (service, state) = service(
            Vec::new(),
            vec![Ok(confirmation())],
            vec![Err(NativePairingServiceError::LockedKeyring), Ok(receipt())],
        );
        let view = cx.new(|_| NativePairingView::new(service));
        view.update(cx, NativePairingView::begin_scan);
        view.update(cx, |view, cx| view.submit_scanned_qr(VALID_QR, cx))
            .expect("scan QR");

        assert_eq!(
            view.update(cx, NativePairingView::confirm_import),
            Err(NativePairingUiError::Service(
                NativePairingServiceError::LockedKeyring
            ))
        );
        assert_eq!(
            view.read_with(cx, |view, _| (view.phase(), view.notice())),
            (
                NativePairingPhase::AwaitingConfirmation,
                Some(NativePairingNotice::LockedKeyring)
            )
        );
        assert_eq!(state.import_calls.get(), 1);

        view.update(cx, NativePairingView::confirm_import)
            .expect("retry after unlocking keyring");
        assert_eq!(state.import_calls.get(), 2);
        assert_eq!(
            view.read_with(cx, |view, _| view.phase()),
            NativePairingPhase::Completed
        );
        let debug = view.read_with(cx, |view, _| format!("{:?}", view.completion()));
        assert!(!debug.contains("0101010101010101"));
        assert!(!debug.contains("nostrpair://"));
    }
}
