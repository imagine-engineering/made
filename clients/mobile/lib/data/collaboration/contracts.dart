import 'dart:convert';

const mobileClientId = 'buzz-mobile';
const mobileClientVersion = '0.0.0+1';
const mobileCompatibilityPath = '/v1/collaboration/compatibility';
const mobileStateSchemaVersion = 1;

enum MobileCompatibilityAccess { read, write }

enum MobileEntityKind {
  channel,
  message,
  directMessage,
  readState,
  searchResult,
  media,
}

enum MobileCollaborationErrorKind {
  upgradeRequired,
  readOnly,
  authenticationDenied,
  invalidRequest,
  invalidResponse,
  stateConflict,
  serviceUnavailable,
}

class MobileCollaborationException implements Exception {
  const MobileCollaborationException(
    this.kind,
    this.message, {
    this.minimumVersion,
    this.maximumVersion,
    this.retryable = false,
  });

  final MobileCollaborationErrorKind kind;
  final String message;
  final String? minimumVersion;
  final String? maximumVersion;
  final bool retryable;

  @override
  String toString() => 'MobileCollaborationException($kind)';
}

class MobileCollaborationBinding {
  MobileCollaborationBinding({
    required String accountId,
    required String communityId,
    required String profileId,
    required String nostrPublicKey,
    required Uri serviceOrigin,
    required Uri relayUrl,
    required String credentialReference,
  }) : accountId = validAccountId(accountId),
       communityId = validUuid(communityId),
       profileId = validUuid(profileId),
       nostrPublicKey = validHex(nostrPublicKey, 64),
       serviceOrigin = validServiceOrigin(serviceOrigin),
       relayUrl = validRelayUrl(relayUrl),
       credentialReference = validOpaqueReference(credentialReference);

  final String accountId;
  final String communityId;
  final String profileId;
  final String nostrPublicKey;
  final Uri serviceOrigin;
  final Uri relayUrl;
  final String credentialReference;

  String get identityKey => '$accountId/$communityId';

  Map<String, Object> toJson() => {
    'account_id': accountId,
    'community_id': communityId,
    'profile_id': profileId,
    'nostr_public_key': nostrPublicKey,
    'service_origin': serviceOrigin.toString(),
    'relay_url': relayUrl.toString(),
    'credential_reference': credentialReference,
  };

  factory MobileCollaborationBinding.fromJson(Object? value) {
    final json = jsonObject(value);
    return MobileCollaborationBinding(
      accountId: requiredString(json, 'account_id'),
      communityId: requiredString(json, 'community_id'),
      profileId: requiredString(json, 'profile_id'),
      nostrPublicKey: requiredString(json, 'nostr_public_key'),
      serviceOrigin: requiredUri(json, 'service_origin'),
      relayUrl: requiredUri(json, 'relay_url'),
      credentialReference: requiredString(json, 'credential_reference'),
    );
  }

  bool hasSameCanonicalIdentity(MobileCollaborationBinding other) =>
      accountId == other.accountId &&
      communityId == other.communityId &&
      profileId == other.profileId;

  bool hasSameStoredValue(MobileCollaborationBinding other) =>
      hasSameCanonicalIdentity(other) &&
      nostrPublicKey == other.nostrPublicKey &&
      serviceOrigin == other.serviceOrigin &&
      relayUrl == other.relayUrl &&
      credentialReference == other.credentialReference;
}

class MobileEntityRecord {
  MobileEntityRecord({
    required String accountId,
    required String communityId,
    required this.kind,
    required String entityId,
    required int revision,
    required String payloadDigest,
  }) : accountId = validAccountId(accountId),
       communityId = validUuid(communityId),
       entityId = validEntityId(entityId),
       revision = validRevision(revision),
       payloadDigest = validHex(payloadDigest, 64);

  final String accountId;
  final String communityId;
  final MobileEntityKind kind;
  final String entityId;
  final int revision;
  final String payloadDigest;

  String get identityKey => '$accountId/$communityId/${kind.name}/$entityId';

  Map<String, Object> toJson() => {
    'account_id': accountId,
    'community_id': communityId,
    'kind': kind.name,
    'entity_id': entityId,
    'revision': revision,
    'payload_digest': payloadDigest,
  };

  factory MobileEntityRecord.fromJson(Object? value) {
    final json = jsonObject(value);
    final kindName = requiredString(json, 'kind');
    final kind = MobileEntityKind.values.where(
      (candidate) => candidate.name == kindName,
    );
    if (kind.length != 1) throw invalidResponse();
    return MobileEntityRecord(
      accountId: requiredString(json, 'account_id'),
      communityId: requiredString(json, 'community_id'),
      kind: kind.single,
      entityId: requiredString(json, 'entity_id'),
      revision: requiredInt(json, 'revision'),
      payloadDigest: requiredString(json, 'payload_digest'),
    );
  }
}

class MobileBootstrap {
  MobileBootstrap({
    required String accountId,
    required String communityId,
    required String profileId,
    required List<MobileEntityRecord> records,
  }) : accountId = validAccountId(accountId),
       communityId = validUuid(communityId),
       profileId = validUuid(profileId),
       records = List.unmodifiable(records);

  final String accountId;
  final String communityId;
  final String profileId;
  final List<MobileEntityRecord> records;

  factory MobileBootstrap.fromJson(Object? value) {
    final json = jsonObject(value);
    final records = json['records'];
    if (records is! List<Object?> || records.length > 10000) {
      throw invalidResponse();
    }
    return MobileBootstrap(
      accountId: requiredString(json, 'account_id'),
      communityId: requiredString(json, 'community_id'),
      profileId: requiredString(json, 'profile_id'),
      records: records.map(MobileEntityRecord.fromJson).toList(),
    );
  }
}

Map<String, Object?> jsonObject(Object? value) {
  if (value is! Map<String, Object?>) throw invalidResponse();
  return value;
}

Map<String, Object?> decodeJsonObject(String value) {
  if (value.length > 1024 * 1024) throw invalidResponse();
  Object? decoded;
  try {
    decoded = jsonDecode(value);
  } on FormatException {
    throw invalidResponse();
  }
  return jsonObject(decoded);
}

String requiredString(
  Map<String, Object?> json,
  String key, {
  int maximumLength = 4096,
}) {
  final value = json[key];
  if (value is! String ||
      value.isEmpty ||
      value.length > maximumLength ||
      containsControlCharacter(value)) {
    throw invalidResponse();
  }
  return value;
}

int requiredInt(Map<String, Object?> json, String key) {
  final value = json[key];
  if (value is! int) throw invalidResponse();
  return value;
}

Uri requiredUri(Map<String, Object?> json, String key) {
  final value = requiredString(json, key);
  final uri = Uri.tryParse(value);
  if (uri == null) throw invalidResponse();
  return uri;
}

String validAccountId(String value) {
  if (!RegExp(r'^[1-9][0-9]{0,19}$').hasMatch(value)) {
    throw invalidRequest();
  }
  return value;
}

String validUuid(String value) {
  final normalized = value.toLowerCase();
  if (!RegExp(
        r'^[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}$',
      ).hasMatch(normalized) ||
      normalized == '00000000-0000-0000-0000-000000000000') {
    throw invalidRequest();
  }
  return normalized;
}

String validHex(String value, int length) {
  if (value.length != length || !RegExp(r'^[0-9a-f]+$').hasMatch(value)) {
    throw invalidRequest();
  }
  return value;
}

String validEntityId(String value) {
  if (value.isEmpty || value.length > 256 || containsControlCharacter(value)) {
    throw invalidRequest();
  }
  return value;
}

String validOpaqueReference(String value) {
  if (value.isEmpty || value.length > 256 || containsControlCharacter(value)) {
    throw invalidRequest();
  }
  return value;
}

int validRevision(int value) {
  if (value < 1) throw invalidRequest();
  return value;
}

Uri validServiceOrigin(Uri value) {
  final localHttp = value.scheme == 'http' && isLoopbackHost(value.host);
  if ((value.scheme != 'https' && !localHttp) ||
      value.userInfo.isNotEmpty ||
      value.query.isNotEmpty ||
      value.fragment.isNotEmpty ||
      (value.path.isNotEmpty && value.path != '/')) {
    throw invalidRequest();
  }
  return value.replace(path: '/', query: null, fragment: null);
}

Uri validRelayUrl(Uri value) {
  final localWebSocket = value.scheme == 'ws' && isLoopbackHost(value.host);
  if ((value.scheme != 'wss' && !localWebSocket) ||
      value.userInfo.isNotEmpty ||
      value.query.isNotEmpty ||
      value.fragment.isNotEmpty) {
    throw invalidRequest();
  }
  return value;
}

bool isLoopbackHost(String value) =>
    value == 'localhost' ||
    value.endsWith('.localhost') ||
    value == '127.0.0.1' ||
    value == '::1';

bool containsControlCharacter(String value) =>
    value.runes.any((codePoint) => codePoint <= 31 || codePoint == 127);

MobileCollaborationException invalidRequest() =>
    const MobileCollaborationException(
      MobileCollaborationErrorKind.invalidRequest,
      'The mobile collaboration request is invalid.',
    );

MobileCollaborationException invalidResponse() =>
    const MobileCollaborationException(
      MobileCollaborationErrorKind.invalidResponse,
      'The collaboration service returned an invalid response.',
    );
