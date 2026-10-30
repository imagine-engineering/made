import 'dart:async';
import 'dart:convert';

import 'contracts.dart';

const mobileCollaborationStateKey = 'zed_collaboration_mobile_state_v1';

abstract interface class MobileKeyValueStore {
  Future<String?> read(String key);

  Future<void> write(String key, String value);
}

class MobileStateSnapshot {
  MobileStateSnapshot({
    required List<MobileCollaborationBinding> bindings,
    required this.activeIdentityKey,
    required List<MobileEntityRecord> records,
  }) : bindings = List.unmodifiable(bindings),
       records = List.unmodifiable(records);

  final List<MobileCollaborationBinding> bindings;
  final String? activeIdentityKey;
  final List<MobileEntityRecord> records;

  MobileCollaborationBinding? get activeBinding {
    final key = activeIdentityKey;
    if (key == null) return null;
    for (final binding in bindings) {
      if (binding.identityKey == key) return binding;
    }
    return null;
  }

  List<MobileEntityRecord> recordsFor(MobileCollaborationBinding binding) =>
      records
          .where(
            (record) =>
                record.accountId == binding.accountId &&
                record.communityId == binding.communityId,
          )
          .toList(growable: false);
}

class MobileCollaborationStateStore {
  MobileCollaborationStateStore(this._store);

  final MobileKeyValueStore _store;
  Future<void> _tail = Future.value();

  Future<MobileStateSnapshot> load() => _serialized(_load);

  Future<MobileStateSnapshot> applyBootstrap(
    MobileCollaborationBinding binding,
    MobileBootstrap bootstrap,
  ) => _serialized(() async {
    if (bootstrap.accountId != binding.accountId ||
        bootstrap.communityId != binding.communityId ||
        bootstrap.profileId != binding.profileId ||
        bootstrap.records.any(
          (record) =>
              record.accountId != binding.accountId ||
              record.communityId != binding.communityId,
        )) {
      throw invalidResponse();
    }

    final state = await _loadMutable();
    _mergeBinding(state.bindings, binding, replace: true);
    for (final record in bootstrap.records) {
      _mergeRecord(state.records, record);
    }
    state.activeIdentityKey = binding.identityKey;
    await _save(state);
    return state.snapshot();
  });

  Future<MobileStateSnapshot> importResolvedLegacyBinding(
    MobileCollaborationBinding resolved,
  ) => _serialized(() async {
    final state = await _loadMutable();
    _mergeBinding(state.bindings, resolved, replace: true);
    state.activeIdentityKey ??= resolved.identityKey;
    await _save(state);
    return state.snapshot();
  });

  Future<MobileStateSnapshot> switchActive({
    required String accountId,
    required String communityId,
  }) => _serialized(() async {
    final identityKey =
        '${validAccountId(accountId)}/${validUuid(communityId)}';
    final state = await _loadMutable();
    if (!state.bindings.containsKey(identityKey)) {
      throw const MobileCollaborationException(
        MobileCollaborationErrorKind.stateConflict,
        'The selected account and community are not stored on this device.',
      );
    }
    state.activeIdentityKey = identityKey;
    await _save(state);
    return state.snapshot();
  });

  Future<T> _serialized<T>(Future<T> Function() operation) {
    final completer = Completer<T>();
    _tail = _tail.then((_) async {
      try {
        completer.complete(await operation());
      } catch (error, stackTrace) {
        completer.completeError(error, stackTrace);
      }
    });
    return completer.future;
  }

  Future<MobileStateSnapshot> _load() async {
    final state = await _loadMutable();
    return state.snapshot();
  }

  Future<_MutableState> _loadMutable() async {
    final raw = await _store.read(mobileCollaborationStateKey);
    if (raw == null) return _MutableState.empty();
    final json = decodeJsonObject(raw);
    if (requiredInt(json, 'schema_version') != mobileStateSchemaVersion) {
      throw invalidResponse();
    }
    final bindingValues = json['bindings'];
    final recordValues = json['records'];
    if (bindingValues is! List<Object?> ||
        bindingValues.length > 64 ||
        recordValues is! List<Object?> ||
        recordValues.length > 50000) {
      throw invalidResponse();
    }

    final bindings = <String, MobileCollaborationBinding>{};
    for (final value in bindingValues) {
      _mergeBinding(bindings, MobileCollaborationBinding.fromJson(value));
    }
    final records = <String, MobileEntityRecord>{};
    for (final value in recordValues) {
      _mergeRecord(records, MobileEntityRecord.fromJson(value));
    }
    final active = json['active_identity_key'];
    if (active != null &&
        (active is! String || !bindings.containsKey(active))) {
      throw invalidResponse();
    }
    return _MutableState(
      bindings: bindings,
      records: records,
      activeIdentityKey: active as String?,
    );
  }

  Future<void> _save(_MutableState state) async {
    if (state.bindings.length > 64 || state.records.length > 50000) {
      throw const MobileCollaborationException(
        MobileCollaborationErrorKind.stateConflict,
        'The mobile collaboration state exceeds its local bounds.',
      );
    }
    final bindings = state.bindings.values.toList()
      ..sort((left, right) => left.identityKey.compareTo(right.identityKey));
    final records = state.records.values.toList()
      ..sort((left, right) => left.identityKey.compareTo(right.identityKey));
    final encoded = jsonEncode({
      'schema_version': mobileStateSchemaVersion,
      'active_identity_key': state.activeIdentityKey,
      'bindings': bindings.map((binding) => binding.toJson()).toList(),
      'records': records.map((record) => record.toJson()).toList(),
    });
    if (encoded.length > 4 * 1024 * 1024) {
      throw const MobileCollaborationException(
        MobileCollaborationErrorKind.stateConflict,
        'The mobile collaboration state exceeds its local size limit.',
      );
    }
    await _store.write(mobileCollaborationStateKey, encoded);
  }
}

class _MutableState {
  _MutableState({
    required this.bindings,
    required this.records,
    required this.activeIdentityKey,
  });

  factory _MutableState.empty() =>
      _MutableState(bindings: {}, records: {}, activeIdentityKey: null);

  final Map<String, MobileCollaborationBinding> bindings;
  final Map<String, MobileEntityRecord> records;
  String? activeIdentityKey;

  MobileStateSnapshot snapshot() => MobileStateSnapshot(
    bindings: bindings.values.toList(),
    activeIdentityKey: activeIdentityKey,
    records: records.values.toList(),
  );
}

void _mergeBinding(
  Map<String, MobileCollaborationBinding> bindings,
  MobileCollaborationBinding incoming, {
  bool replace = false,
}) {
  final existing = bindings[incoming.identityKey];
  if (existing != null && !existing.hasSameCanonicalIdentity(incoming)) {
    throw const MobileCollaborationException(
      MobileCollaborationErrorKind.stateConflict,
      'A stored account and community binding conflicts with canonical identity.',
    );
  }
  if (existing != null && !replace && !existing.hasSameStoredValue(incoming)) {
    throw const MobileCollaborationException(
      MobileCollaborationErrorKind.stateConflict,
      'Duplicate stored bindings disagree about canonical authority.',
    );
  }
  bindings[incoming.identityKey] = incoming;
}

void _mergeRecord(
  Map<String, MobileEntityRecord> records,
  MobileEntityRecord incoming,
) {
  final existing = records[incoming.identityKey];
  if (existing == null || incoming.revision > existing.revision) {
    records[incoming.identityKey] = incoming;
    return;
  }
  if (incoming.revision < existing.revision) return;
  if (incoming.payloadDigest != existing.payloadDigest) {
    throw const MobileCollaborationException(
      MobileCollaborationErrorKind.stateConflict,
      'A canonical entity revision has conflicting local content.',
    );
  }
}
