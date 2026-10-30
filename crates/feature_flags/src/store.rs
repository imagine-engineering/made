use std::any::TypeId;
use std::sync::Arc;

use collections::HashMap;
use fs::Fs;
use gpui::{App, BorrowAppContext, Subscription};
use settings::{Settings, SettingsStore, update_settings_file};

use crate::{FeatureFlag, FeatureFlagValue, FeatureFlagsSettings, ZED_DISABLE_STAFF};

pub struct FeatureFlagDescriptor {
    pub name: &'static str,
    pub variants: fn() -> Vec<FeatureFlagVariant>,
    pub on_variant_key: fn() -> &'static str,
    pub default_variant_key: fn() -> &'static str,
    pub enabled_for_all: fn() -> bool,
    pub enabled_for_staff: fn() -> bool,
    pub allow_local_override: fn() -> bool,
    pub requires_local_opt_in: fn() -> bool,
    pub description: fn() -> &'static str,
    pub has_flag: fn(&FeatureFlagStore, &App) -> bool,
    pub type_id: fn() -> TypeId,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FeatureFlagVariant {
    pub override_key: &'static str,
    pub label: &'static str,
}

inventory::collect!(FeatureFlagDescriptor);

#[doc(hidden)]
pub mod __private {
    pub use inventory;
}

/// Submits a [`FeatureFlagDescriptor`] for this flag so it shows up in the
/// configuration UI and in `FeatureFlagStore::known_flags()`.
#[macro_export]
macro_rules! register_feature_flag {
    ($flag:ty) => {
        $crate::__private::inventory::submit! {
            $crate::FeatureFlagDescriptor {
                name: <$flag as $crate::FeatureFlag>::NAME,
                variants: || {
                    <<$flag as $crate::FeatureFlag>::Value as $crate::FeatureFlagValue>::all_variants()
                        .iter()
                        .map(|v| $crate::FeatureFlagVariant {
                            override_key: <<$flag as $crate::FeatureFlag>::Value as $crate::FeatureFlagValue>::override_key(v),
                            label: <<$flag as $crate::FeatureFlag>::Value as $crate::FeatureFlagValue>::label(v),
                        })
                        .collect()
                },
                on_variant_key: || {
                    <<$flag as $crate::FeatureFlag>::Value as $crate::FeatureFlagValue>::override_key(
                        &<<$flag as $crate::FeatureFlag>::Value as $crate::FeatureFlagValue>::on_variant(),
                    )
                },
                default_variant_key: || {
                    <<$flag as $crate::FeatureFlag>::Value as $crate::FeatureFlagValue>::override_key(
                        &<<$flag as $crate::FeatureFlag>::Value as ::std::default::Default>::default(),
                    )
                },
                enabled_for_all: <$flag as $crate::FeatureFlag>::enabled_for_all,
                enabled_for_staff: <$flag as $crate::FeatureFlag>::enabled_for_staff,
                allow_local_override: <$flag as $crate::FeatureFlag>::allow_local_override,
                requires_local_opt_in: <$flag as $crate::FeatureFlag>::requires_local_opt_in,
                description: <$flag as $crate::FeatureFlag>::description,
                has_flag: |store, cx| store.has_flag::<$flag>(cx),
                type_id: || std::any::TypeId::of::<$flag>(),
            }
        }
    };
}

#[derive(Default)]
pub struct FeatureFlagStore {
    staff: bool,
    server_flags: HashMap<String, String>,
    server_flags_received: bool,

    _settings_subscription: Option<Subscription>,
}

impl FeatureFlagStore {
    pub fn init(cx: &mut App) {
        let subscription = cx.observe_global::<SettingsStore>(|cx| {
            // Touch the global so anything observing `FeatureFlagStore` re-runs
            cx.update_default_global::<FeatureFlagStore, _>(|_, _| {});
        });

        cx.update_default_global::<FeatureFlagStore, _>(|store, _| {
            store._settings_subscription = Some(subscription);
        });
    }

    pub fn known_flags() -> impl Iterator<Item = &'static FeatureFlagDescriptor> {
        let mut seen = collections::HashSet::default();
        inventory::iter::<FeatureFlagDescriptor>().filter(move |d| seen.insert((d.type_id)()))
    }

    pub fn is_staff(&self) -> bool {
        self.staff
    }

    /// Whether feature flag overrides from settings should be honored.
    ///
    /// Ordinary overrides are a staff-only affordance. Flags explicitly
    /// allowing local overrides use [`Self::overrides_enabled_for`] instead.
    /// Debug builds are always treated as staff, and `ZED_DISABLE_STAFF`
    /// forces the user to be treated as non-staff for testing.
    pub fn overrides_enabled(&self) -> bool {
        (cfg!(debug_assertions) || self.staff) && !*ZED_DISABLE_STAFF
    }

    pub fn overrides_enabled_for(&self, descriptor: &FeatureFlagDescriptor) -> bool {
        self.overrides_enabled() || (descriptor.allow_local_override)()
    }

    pub fn has_overridable_flags(&self) -> bool {
        Self::known_flags().any(|descriptor| self.overrides_enabled_for(descriptor))
    }

    pub fn server_flags_received(&self) -> bool {
        self.server_flags_received
    }

    pub fn set_staff(&mut self, staff: bool) {
        self.staff = staff;
    }

    pub fn update_server_flags(&mut self, staff: bool, flags: Vec<String>) {
        self.staff = staff;
        self.server_flags_received = true;
        self.server_flags.clear();
        for flag in flags {
            self.server_flags.insert(flag.clone(), flag);
        }
    }

    /// The user's override key for this flag, read directly from
    /// [`FeatureFlagsSettings`].
    pub fn override_for<'a>(flag_name: &str, cx: &'a App) -> Option<&'a str> {
        if Self::known_flags()
            .any(|descriptor| descriptor.name == flag_name && (descriptor.requires_local_opt_in)())
        {
            return Self::local_override_for(flag_name, cx);
        }
        FeatureFlagsSettings::get_global(cx)
            .overrides
            .get(flag_name)
            .map(String::as_str)
    }

    fn local_override_for<'a>(flag_name: &str, cx: &'a App) -> Option<&'a str> {
        cx.try_global::<SettingsStore>()?
            .raw_local_settings()?
            .feature_flags
            .as_ref()?
            .get(flag_name)
            .map(String::as_str)
    }

    /// Applies an override by writing to `settings.json`. The store's own
    /// `overrides` field will be updated when the settings-store observer
    /// fires. Pass the [`FeatureFlagValue::override_key`] of the variant
    /// you want forced.
    pub fn set_override(flag_name: &str, override_key: String, fs: Arc<dyn Fs>, cx: &App) {
        let flag_name = flag_name.to_owned();
        update_settings_file(fs, cx, move |content, _| {
            content
                .feature_flags
                .get_or_insert_default()
                .insert(flag_name, override_key);
        });
    }

    /// Removes any override for the given flag from `settings.json`. Leaves
    /// an empty `"feature_flags"` object rather than removing the key
    /// entirely so the user can see it's still a meaningful settings surface.
    pub fn clear_override(flag_name: &str, fs: Arc<dyn Fs>, cx: &App) {
        let flag_name = flag_name.to_owned();
        update_settings_file(fs, cx, move |content, _| {
            if let Some(map) = content.feature_flags.as_mut() {
                map.remove(&flag_name);
            }
        });
    }

    /// The resolved value of the flag for the current user, taking overrides,
    /// `enabled_for_all`, staff rules, and server flags into account. Explicit
    /// local opt-in flags use only the original local settings file.
    pub fn try_flag_value<T: FeatureFlag>(&self, cx: &App) -> Option<T::Value> {
        self.flag_value_with_policy::<T>(
            cx,
            self.overrides_enabled(),
            (cfg!(debug_assertions) || self.staff) && !*ZED_DISABLE_STAFF,
        )
    }

    fn flag_value_with_policy<T: FeatureFlag>(
        &self,
        cx: &App,
        ordinary_overrides_enabled: bool,
        staff_enabled: bool,
    ) -> Option<T::Value> {
        if T::requires_local_opt_in() {
            return Self::local_override_for(T::NAME, cx).and_then(variant_from_key::<T::Value>);
        }

        // Ordinary global rollout keeps its existing precedence over local overrides.
        if T::enabled_for_all() && !T::requires_local_opt_in() {
            return Some(T::Value::on_variant());
        }

        // Only apply overrides when they are specifically enabled.
        if ordinary_overrides_enabled || T::allow_local_override() {
            if let Some(override_key) = FeatureFlagsSettings::get_global(cx).overrides.get(T::NAME)
            {
                return variant_from_key::<T::Value>(override_key);
            }
        }

        if T::requires_local_opt_in() {
            return None;
        }

        // Staff default: resolve to the enabled variant.
        if staff_enabled && T::enabled_for_staff() {
            return Some(T::Value::on_variant());
        }

        // Server-delivered flag.
        if let Some(wire) = self.server_flags.get(T::NAME) {
            return T::Value::from_wire(wire);
        }

        None
    }

    /// Whether the flag resolves to its "on" value. Best for presence-style
    /// flags. For enum flags with meaningful non-default variants, prefer
    /// [`crate::FeatureFlagAppExt::flag_value`].
    pub fn has_flag<T: FeatureFlag>(&self, cx: &App) -> bool {
        self.try_flag_value::<T>(cx)
            .is_some_and(|v| v == T::Value::on_variant())
    }

    pub fn has_flag_by_name(&self, name: &str, cx: &App) -> bool {
        Self::known_flags()
            .find(|descriptor| descriptor.name == name)
            .is_some_and(|descriptor| (descriptor.has_flag)(self, cx))
    }

    /// Mirrors the resolution order of [`Self::try_flag_value`], but falls
    /// back to the [`Default`] variant when no rule applies so the UI always
    /// shows *something* selected — matching what
    /// [`crate::FeatureFlagAppExt::flag_value`] would return.
    pub fn resolved_key(&self, descriptor: &FeatureFlagDescriptor, cx: &App) -> &'static str {
        if (descriptor.requires_local_opt_in)() {
            return Self::local_override_for(descriptor.name, cx)
                .and_then(|requested| {
                    (descriptor.variants)()
                        .into_iter()
                        .find(|variant| variant.override_key == requested)
                        .map(|variant| variant.override_key)
                })
                .unwrap_or_else(|| (descriptor.default_variant_key)());
        }

        let on_variant_key = (descriptor.on_variant_key)();

        if Self::is_forced_on(descriptor) {
            return on_variant_key;
        }

        // Only apply overrides when they are specifically enabled.
        if self.overrides_enabled_for(descriptor) {
            if let Some(requested) = FeatureFlagsSettings::get_global(cx)
                .overrides
                .get(descriptor.name)
            {
                if let Some(variant) = (descriptor.variants)()
                    .into_iter()
                    .find(|v| v.override_key == requested.as_str())
                {
                    return variant.override_key;
                }
            }
        }

        if (descriptor.requires_local_opt_in)() {
            return (descriptor.default_variant_key)();
        }

        if (cfg!(debug_assertions) || self.staff)
            && !*ZED_DISABLE_STAFF
            && (descriptor.enabled_for_staff)()
        {
            return on_variant_key;
        }

        if self.server_flags.contains_key(descriptor.name) {
            return on_variant_key;
        }

        (descriptor.default_variant_key)()
    }

    /// Whether this flag is forced on by `enabled_for_all` and therefore not
    /// user-overridable. The UI uses this to render the row as disabled.
    pub fn is_forced_on(descriptor: &FeatureFlagDescriptor) -> bool {
        (descriptor.enabled_for_all)() && !(descriptor.requires_local_opt_in)()
    }

    /// Fallback used when the store isn't installed as a global yet (e.g. very
    /// early in startup). Matches the pre-existing default behavior.
    pub fn has_flag_default<T: FeatureFlag>() -> bool {
        if T::requires_local_opt_in() {
            return false;
        }
        if T::enabled_for_all() {
            return true;
        }
        cfg!(debug_assertions) && T::enabled_for_staff() && !*ZED_DISABLE_STAFF
    }
}

fn variant_from_key<V: FeatureFlagValue>(key: &str) -> Option<V> {
    V::all_variants()
        .iter()
        .find(|v| v.override_key() == key)
        .cloned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EnumFeatureFlag, FeatureFlag, PresenceFlag};
    use gpui::UpdateGlobal;
    use settings::SettingsStore;

    struct DemoFlag;
    impl FeatureFlag for DemoFlag {
        const NAME: &'static str = "demo";
        type Value = PresenceFlag;
        fn enabled_for_staff() -> bool {
            false
        }
    }

    #[derive(Clone, Copy, PartialEq, Eq, Debug, EnumFeatureFlag)]
    enum Intensity {
        #[default]
        Low,
        High,
    }

    struct IntensityFlag;
    impl FeatureFlag for IntensityFlag {
        const NAME: &'static str = "intensity";
        type Value = Intensity;
        fn enabled_for_all() -> bool {
            true
        }
    }

    fn init_settings_store(cx: &mut App) {
        let store = SettingsStore::test(cx);
        cx.set_global(store);
        SettingsStore::update_global(cx, |store, _| {
            store.register_setting::<FeatureFlagsSettings>();
        });
    }

    fn set_override(name: &str, value: &str, cx: &mut App) {
        SettingsStore::update_global(cx, |store: &mut SettingsStore, cx| {
            store.update_user_settings(cx, |content| {
                content
                    .feature_flags
                    .get_or_insert_default()
                    .insert(name.to_string(), value.to_string());
            });
        });
    }

    #[gpui::test]
    fn server_flag_enables_presence(cx: &mut App) {
        init_settings_store(cx);
        let mut store = FeatureFlagStore::default();
        assert!(!store.has_flag::<DemoFlag>(cx));
        store.update_server_flags(false, vec!["demo".to_string()]);
        assert!(store.has_flag::<DemoFlag>(cx));
    }

    #[gpui::test]
    fn off_override_beats_server_flag(cx: &mut App) {
        init_settings_store(cx);
        let mut store = FeatureFlagStore::default();
        store.update_server_flags(false, vec!["demo".to_string()]);
        set_override(DemoFlag::NAME, "off", cx);
        assert!(!store.has_flag::<DemoFlag>(cx));
        assert_eq!(
            store.try_flag_value::<DemoFlag>(cx),
            Some(PresenceFlag::Off)
        );
    }

    #[gpui::test]
    fn enabled_for_all_wins_over_override(cx: &mut App) {
        init_settings_store(cx);
        let store = FeatureFlagStore::default();
        set_override(IntensityFlag::NAME, "high", cx);
        assert_eq!(
            store.try_flag_value::<IntensityFlag>(cx),
            Some(Intensity::Low)
        );
    }

    #[gpui::test]
    fn enum_override_selects_specific_variant(cx: &mut App) {
        init_settings_store(cx);
        let store = FeatureFlagStore::default();
        // Staff path would normally resolve to `Low`; the override pushes
        // us to `High` instead.
        set_override("enum-demo", "high", cx);

        struct EnumDemo;
        impl FeatureFlag for EnumDemo {
            const NAME: &'static str = "enum-demo";
            type Value = Intensity;
        }

        assert_eq!(store.try_flag_value::<EnumDemo>(cx), Some(Intensity::High));
    }

    #[gpui::test]
    fn unknown_variant_key_resolves_to_none(cx: &mut App) {
        init_settings_store(cx);
        let store = FeatureFlagStore::default();
        set_override("enum-demo", "nonsense", cx);

        struct EnumDemo;
        impl FeatureFlag for EnumDemo {
            const NAME: &'static str = "enum-demo";
            type Value = Intensity;
        }

        assert_eq!(store.try_flag_value::<EnumDemo>(cx), None);
    }

    #[gpui::test]
    fn on_override_enables_without_server_or_staff(cx: &mut App) {
        init_settings_store(cx);
        let store = FeatureFlagStore::default();
        set_override(DemoFlag::NAME, "on", cx);
        assert!(store.has_flag::<DemoFlag>(cx));
    }

    /// No rule applies, so the store's `try_flag_value` returns `None`. The
    /// `FeatureFlagAppExt::flag_value` path (used by most callers) falls
    /// back to [`Default`], which for `PresenceFlag` is `Off`.
    #[gpui::test]
    fn presence_flag_defaults_to_off(cx: &mut App) {
        init_settings_store(cx);
        let store = FeatureFlagStore::default();
        assert_eq!(store.try_flag_value::<DemoFlag>(cx), None);
        assert_eq!(PresenceFlag::default(), PresenceFlag::Off);
    }

    #[gpui::test]
    fn on_flags_ready_waits_for_server_flags(cx: &mut gpui::TestAppContext) {
        use crate::FeatureFlagAppExt;
        use std::cell::Cell;
        use std::rc::Rc;

        cx.update(|cx| {
            init_settings_store(cx);
            FeatureFlagStore::init(cx);
        });

        let fired = Rc::new(Cell::new(false));
        cx.update({
            let fired = fired.clone();
            |cx| cx.on_flags_ready(move |_, _| fired.set(true)).detach()
        });

        // Settings-triggered no-op touch must not fire on_flags_ready.
        cx.update(|cx| cx.update_default_global::<FeatureFlagStore, _>(|_, _| {}));
        cx.run_until_parked();
        assert!(!fired.get());

        // Server flags arrive — now it should fire.
        cx.update(|cx| cx.update_flags(true, vec![]));
        cx.run_until_parked();
        assert!(fired.get());
    }
}

#[cfg(test)]
mod experimental_tests {
    use super::*;
    use crate::{
        NotebookFeatureFlag, PresenceFlag, RustCoverageExperimentalFeatureFlag,
        RustExactDoctestActionsExperimentalFeatureFlag,
        RustNamedCargoPresetsExperimentalFeatureFlag, RustNativeProfilingExperimentalFeatureFlag,
    };
    use gpui::UpdateGlobal;

    fn initialize(cx: &mut App) {
        let settings_store = SettingsStore::test(cx);
        cx.set_global(settings_store);
        SettingsStore::update_global(cx, |store, _| {
            store.register_setting::<FeatureFlagsSettings>();
        });
    }

    fn override_flag(name: &str, value: &str, cx: &mut App) {
        SettingsStore::update_global(cx, |store, cx| {
            store.update_user_settings(cx, |content| {
                content
                    .feature_flags
                    .get_or_insert_default()
                    .insert(name.into(), value.into());
            });
        });
    }

    fn check_explicit_local_flag<T: FeatureFlag<Value = PresenceFlag>>(cx: &mut App) {
        let mut store = FeatureFlagStore::default();
        assert_eq!(store.flag_value_with_policy::<T>(cx, false, false), None);
        assert_eq!(store.flag_value_with_policy::<T>(cx, true, true), None);
        store.update_server_flags(true, vec![T::NAME.into()]);
        assert_eq!(store.flag_value_with_policy::<T>(cx, false, false), None);
        assert_eq!(store.try_flag_value::<T>(cx), None);
        assert!(!FeatureFlagStore::has_flag_default::<T>());
        let descriptor = FeatureFlagStore::known_flags()
            .find(|flag| flag.name == T::NAME)
            .expect("registered experimental flag");
        assert_eq!(store.resolved_key(descriptor, cx), "off");
        override_flag(T::NAME, "on", cx);
        assert_eq!(
            store.flag_value_with_policy::<T>(cx, false, false),
            Some(PresenceFlag::On)
        );
        assert_eq!(store.try_flag_value::<T>(cx), Some(PresenceFlag::On));
        assert!(store.has_flag_by_name(T::NAME, cx));
        assert!(!store.has_flag_by_name("unknown-experimental-flag", cx));
        store.update_server_flags(true, vec![T::NAME.into()]);
        override_flag(T::NAME, "off", cx);
        assert_eq!(
            store.flag_value_with_policy::<T>(cx, false, false),
            Some(PresenceFlag::Off)
        );
        assert_eq!(store.try_flag_value::<T>(cx), Some(PresenceFlag::Off));
        assert!(!store.has_flag_by_name(T::NAME, cx));
        let descriptor = FeatureFlagStore::known_flags()
            .find(|flag| flag.name == T::NAME)
            .expect("registered experimental flag");
        assert_eq!(store.resolved_key(descriptor, cx), "off");
        assert!(store.overrides_enabled_for(descriptor));
        override_flag(T::NAME, "unrecognized", cx);
        assert_eq!(store.try_flag_value::<T>(cx), None);
        assert_eq!(store.resolved_key(descriptor, cx), "off");
        SettingsStore::update_global(cx, |store, cx| {
            store.update_user_settings(cx, |content| {
                if let Some(overrides) = &mut content.feature_flags {
                    overrides.remove(T::NAME);
                }
            });
        });
        assert_eq!(store.try_flag_value::<T>(cx), None);
        assert_eq!(store.resolved_key(descriptor, cx), "off");
    }

    #[gpui::test]
    fn experimental_flags_resolve_explicit_local_values_in_nonstaff_policy(cx: &mut App) {
        initialize(cx);
        check_explicit_local_flag::<RustCoverageExperimentalFeatureFlag>(cx);
        check_explicit_local_flag::<RustNativeProfilingExperimentalFeatureFlag>(cx);
        check_explicit_local_flag::<RustNamedCargoPresetsExperimentalFeatureFlag>(cx);
        check_explicit_local_flag::<RustExactDoctestActionsExperimentalFeatureFlag>(cx);
    }

    #[gpui::test]
    fn ordinary_flags_keep_staff_override_and_server_policy(cx: &mut App) {
        initialize(cx);
        let mut store = FeatureFlagStore::default();
        assert!(!store.has_flag_by_name(crate::AgentThreadWorktreeLabelFlag::NAME, cx));
        let descriptor = FeatureFlagStore::known_flags()
            .find(|flag| flag.name == crate::AgentThreadWorktreeLabelFlag::NAME)
            .expect("registered ordinary enum flag");
        assert_eq!(
            store.resolved_key(descriptor, cx),
            (descriptor.on_variant_key)()
        );
        store.update_server_flags(
            false,
            vec![crate::AgentThreadWorktreeLabelFlag::NAME.into()],
        );
        assert!(store.has_flag_by_name(crate::AgentThreadWorktreeLabelFlag::NAME, cx));
        override_flag(NotebookFeatureFlag::NAME, "on", cx);
        assert_eq!(
            store.flag_value_with_policy::<NotebookFeatureFlag>(cx, false, false),
            None
        );
        assert_eq!(
            store.flag_value_with_policy::<NotebookFeatureFlag>(cx, true, true),
            Some(PresenceFlag::On)
        );
        override_flag(NotebookFeatureFlag::NAME, "off", cx);
        store.update_server_flags(false, vec![NotebookFeatureFlag::NAME.into()]);
        assert_eq!(
            store.flag_value_with_policy::<NotebookFeatureFlag>(cx, false, false),
            Some(PresenceFlag::On)
        );
        assert_eq!(
            store.flag_value_with_policy::<NotebookFeatureFlag>(cx, true, true),
            Some(PresenceFlag::Off)
        );
    }

    #[test]
    fn only_four_experimental_descriptors_allow_nonstaff_local_overrides() {
        let mut names = FeatureFlagStore::known_flags()
            .filter(|flag| (flag.allow_local_override)())
            .map(|flag| {
                assert!(!(flag.enabled_for_staff)());
                assert!(!(flag.enabled_for_all)());
                assert!((flag.requires_local_opt_in)());
                assert!((flag.description)().starts_with("Experimental:"));
                assert_eq!((flag.default_variant_key)(), "off");
                flag.name
            })
            .collect::<Vec<_>>();
        names.sort_unstable();
        assert_eq!(
            names,
            [
                "rust-coverage-experimental",
                "rust-exact-doctest-actions-experimental",
                "rust-named-cargo-presets-experimental",
                "rust-native-profiling-experimental"
            ]
        );
    }
}

#[cfg(test)]
mod experimental_observer_tests {
    use crate::{
        FeatureFlag, FeatureFlagAppExt, FeatureFlagStore, FeatureFlagsSettings, PresenceFlag,
        RustCoverageExperimentalFeatureFlag,
    };
    use gpui::{TestAppContext, UpdateGlobal};
    use settings::SettingsStore;
    use std::{cell::RefCell, rc::Rc};

    #[gpui::test]
    fn experimental_local_setting_changes_notify_existing_flag_observers(cx: &mut TestAppContext) {
        let observed = Rc::new(RefCell::new(Vec::new()));
        let subscription = cx.update(|cx| {
            let settings_store = SettingsStore::test(cx);
            cx.set_global(settings_store);
            SettingsStore::update_global(cx, |store, _| {
                store.register_setting::<FeatureFlagsSettings>()
            });
            FeatureFlagStore::init(cx);
            cx.update_flags(
                false,
                vec![RustCoverageExperimentalFeatureFlag::NAME.into()],
            );
            assert!(!cx.has_flag::<RustCoverageExperimentalFeatureFlag>());
            assert!(!cx.has_flag_by_name(RustCoverageExperimentalFeatureFlag::NAME));
            assert!(!cx.has_flag_by_name("unknown-experimental-flag"));
            let observed = observed.clone();
            cx.observe_flag::<RustCoverageExperimentalFeatureFlag, _>(move |value, _| {
                observed.borrow_mut().push(value)
            })
        });
        cx.run_until_parked();
        for value in ["on", "off"] {
            cx.update(|cx| {
                SettingsStore::update_global(cx, |store, cx| {
                    store.update_user_settings(cx, |content| {
                        content.feature_flags.get_or_insert_default().insert(
                            RustCoverageExperimentalFeatureFlag::NAME.into(),
                            value.into(),
                        );
                    });
                })
            });
            cx.run_until_parked();
            assert_eq!(
                cx.update(|cx| cx.has_flag::<RustCoverageExperimentalFeatureFlag>()),
                value == "on"
            );
            assert_eq!(
                cx.update(|cx| cx.has_flag_by_name(RustCoverageExperimentalFeatureFlag::NAME)),
                value == "on"
            );
        }
        assert!(
            observed
                .borrow()
                .ends_with(&[PresenceFlag::On, PresenceFlag::Off])
        );
        drop(subscription);
    }
}

#[cfg(test)]
mod experimental_local_origin_tests {
    use super::*;
    use crate::{FeatureFlagAppExt as _, PresenceFlag, RustCoverageExperimentalFeatureFlag};
    use gpui::UpdateGlobal as _;

    #[gpui::test]
    fn experimental_opt_in_ignores_merged_sources_and_honors_only_local_user_root(cx: &mut App) {
        let settings_store = SettingsStore::test(cx);
        cx.set_global(settings_store);
        FeatureFlagStore::init(cx);
        let descriptor = FeatureFlagStore::known_flags()
            .find(|descriptor| descriptor.name == RustCoverageExperimentalFeatureFlag::NAME)
            .expect("existing experimental descriptor");
        SettingsStore::update_global(cx, |settings, cx| {
            settings.update_default_settings(cx, |content| {
                content.feature_flags.get_or_insert_default().insert(
                    RustCoverageExperimentalFeatureFlag::NAME.into(),
                    "on".into(),
                );
            });
            settings
                .set_global_settings(
                    r#"{"feature_flags":{"rust-coverage-experimental":"on"}}"#,
                    cx,
                )
                .result()
                .expect("global settings");
            settings
                .set_server_settings(
                    r#"{"feature_flags":{"rust-coverage-experimental":"on"}}"#,
                    cx,
                )
                .expect("nonlocal server settings");
            settings.set_user_settings(
                r#"{"macos":{"feature_flags":{"rust-coverage-experimental":"on"}},"profiles":{"experiment":{"settings":{"feature_flags":{"rust-coverage-experimental":"on"}}}}}"#, cx)
                .result().expect("derived user settings");
        });
        assert_eq!(
            FeatureFlagsSettings::get_global(cx)
                .overrides
                .get(RustCoverageExperimentalFeatureFlag::NAME)
                .map(String::as_str),
            Some("on")
        );
        assert!(!cx.has_flag::<RustCoverageExperimentalFeatureFlag>());
        assert!(!cx.has_flag_by_name(RustCoverageExperimentalFeatureFlag::NAME));
        assert_eq!(
            FeatureFlagStore::override_for(RustCoverageExperimentalFeatureFlag::NAME, cx),
            None
        );
        assert_eq!(
            cx.global::<FeatureFlagStore>().resolved_key(descriptor, cx),
            "off"
        );
        for value in ["on", "off"] {
            SettingsStore::update_global(cx, |settings, cx| {
                settings.update_user_settings(cx, |content| {
                    content.feature_flags.get_or_insert_default().insert(
                        RustCoverageExperimentalFeatureFlag::NAME.into(),
                        value.into(),
                    );
                });
            });
            assert_eq!(
                cx.has_flag::<RustCoverageExperimentalFeatureFlag>(),
                value == "on"
            );
            assert_eq!(
                cx.has_flag_by_name(RustCoverageExperimentalFeatureFlag::NAME),
                value == "on"
            );
            assert_eq!(
                cx.global::<FeatureFlagStore>().resolved_key(descriptor, cx),
                value
            );
            assert_eq!(
                FeatureFlagStore::override_for(RustCoverageExperimentalFeatureFlag::NAME, cx),
                Some(value)
            );
        }
    }

    #[gpui::test]
    fn experimental_host_local_origin_is_explicit_and_nonlocal_server_update_revokes_it(
        cx: &mut App,
    ) {
        let settings_store = SettingsStore::test(cx);
        cx.set_global(settings_store);
        FeatureFlagStore::init(cx);
        let on = r#"{"feature_flags":{"rust-coverage-experimental":"on"}}"#;
        SettingsStore::update_global(cx, |settings, cx| {
            settings
                .set_server_settings(on, cx)
                .expect("nonlocal source");
        });
        assert!(!cx.has_flag::<RustCoverageExperimentalFeatureFlag>());
        SettingsStore::update_global(cx, |settings, cx| {
            settings
                .set_local_server_settings(on, cx)
                .expect("explicit host-local source");
        });
        assert!(cx.has_flag::<RustCoverageExperimentalFeatureFlag>());
        assert_eq!(
            cx.global::<FeatureFlagStore>()
                .try_flag_value::<RustCoverageExperimentalFeatureFlag>(cx),
            Some(PresenceFlag::On)
        );
        SettingsStore::update_global(cx, |settings, cx| {
            settings
                .set_server_settings(on, cx)
                .expect("same bytes from nonlocal source");
        });
        assert!(!cx.has_flag::<RustCoverageExperimentalFeatureFlag>());
        SettingsStore::update_global(cx, |settings, cx| {
            settings
                .set_local_server_settings(
                    r#"{"feature_flags":{"rust-coverage-experimental":"off"}}"#,
                    cx,
                )
                .expect("actual host-local revocation");
        });
        assert!(!cx.has_flag::<RustCoverageExperimentalFeatureFlag>());
    }
}
