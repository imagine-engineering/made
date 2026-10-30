import 'dart:convert';

import 'contracts.dart';
import 'storage.dart';

const _requestTimeout = Duration(seconds: 15);
const _maximumResponseCharacters = 1024 * 1024;
const _requiredFeatures = [
  'invites',
  'communities',
  'channels',
  'messages',
  'direct-messages',
  'media',
  'pairing',
  'push',
];

class MobileTransportRequest {
  const MobileTransportRequest({
    required this.method,
    required this.url,
    required this.headers,
    this.body,
    this.timeout = _requestTimeout,
  });

  final String method;
  final Uri url;
  final Map<String, String> headers;
  final String? body;
  final Duration timeout;
}

class MobileTransportResponse {
  const MobileTransportResponse({required this.statusCode, required this.body});

  final int statusCode;
  final String body;
}

abstract interface class MobileCollaborationTransport {
  Future<MobileTransportResponse> send(MobileTransportRequest request);
}

abstract interface class MobileRequestAuthorizer {
  Future<String> authorize({
    required MobileCollaborationBinding binding,
    required String method,
    required Uri url,
    String? body,
  });
}

class MobileCollaborationClient {
  MobileCollaborationClient({
    required this.transport,
    required this.authorizer,
    required this.stateStore,
  });

  final MobileCollaborationTransport transport;
  final MobileRequestAuthorizer authorizer;
  final MobileCollaborationStateStore stateStore;

  Future<MobileStateSnapshot> activate(
    MobileCollaborationBinding binding,
  ) async {
    await _negotiate(binding.serviceOrigin, MobileCompatibilityAccess.read);
    final url = binding.serviceOrigin.replace(
      pathSegments: [
        'v1',
        'collaboration',
        'accounts',
        binding.accountId,
        'communities',
        binding.communityId,
        'bootstrap',
      ],
      queryParameters: {'profile_id': binding.profileId},
    );
    final authorization = await _authorize(binding, 'GET', url);
    final response = await _send(
      MobileTransportRequest(
        method: 'GET',
        url: url,
        headers: {'Authorization': authorization},
      ),
    );
    if (response.statusCode == 401 || response.statusCode == 403) {
      throw const MobileCollaborationException(
        MobileCollaborationErrorKind.authenticationDenied,
        'This account cannot access the selected community.',
      );
    }
    if (response.statusCode != 200) {
      throw MobileCollaborationException(
        MobileCollaborationErrorKind.serviceUnavailable,
        'The collaboration service could not load this community.',
        retryable: response.statusCode >= 500,
      );
    }
    MobileBootstrap bootstrap;
    try {
      bootstrap = MobileBootstrap.fromJson(_responseJson(response));
    } on MobileCollaborationException catch (error) {
      if (error.kind != MobileCollaborationErrorKind.invalidRequest) rethrow;
      throw invalidResponse();
    }
    return stateStore.applyBootstrap(binding, bootstrap);
  }

  Future<void> requireWriteCompatibility(MobileCollaborationBinding binding) =>
      _negotiate(binding.serviceOrigin, MobileCompatibilityAccess.write);

  Future<void> _negotiate(
    Uri serviceOrigin,
    MobileCompatibilityAccess access,
  ) async {
    final url = serviceOrigin.resolve(mobileCompatibilityPath);
    final response = await _send(
      MobileTransportRequest(
        method: 'POST',
        url: url,
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'client_id': mobileClientId,
          'client_version': mobileClientVersion,
          'access': access.name,
          'protocols': [
            {'id': 'collaboration-http', 'version': 1},
            {'id': 'nostr-ingress', 'version': 1},
            {'id': 'nip-ab', 'version': 1},
            {'id': 'nip44-payload', 'version': 2},
          ],
          'features': _requiredFeatures,
        }),
      ),
    );
    if (response.statusCode != 200 && response.statusCode != 426) {
      throw MobileCollaborationException(
        MobileCollaborationErrorKind.serviceUnavailable,
        'The collaboration service could not negotiate this mobile version.',
        retryable: response.statusCode >= 500,
      );
    }
    final json = _responseJson(response);
    final outcome = requiredString(json, 'outcome');
    if (response.statusCode == 426 || outcome == 'upgrade_required') {
      throw MobileCollaborationException(
        MobileCollaborationErrorKind.upgradeRequired,
        'Buzz mobile $mobileClientVersion is unsupported. Upgrade before continuing.',
        minimumVersion: optionalString(json, 'minimum_client_version'),
        maximumVersion: optionalString(json, 'maximum_client_version'),
      );
    }
    if (access == MobileCompatibilityAccess.write && outcome == 'read_only') {
      throw const MobileCollaborationException(
        MobileCollaborationErrorKind.readOnly,
        'This mobile version can read collaboration data but cannot write it.',
      );
    }
    final policyVersion = json['policy_version'];
    final selectedFeatures = json['selected_features'];
    if (response.statusCode != 200 ||
        policyVersion is! int ||
        policyVersion < 1 ||
        outcome != 'supported' ||
        json['client_id'] != mobileClientId ||
        json['retryable'] != false ||
        selectedFeatures is! List<Object?> ||
        _requiredFeatures.any(
          (feature) => !selectedFeatures.contains(feature),
        )) {
      throw invalidResponse();
    }
  }

  Future<String> _authorize(
    MobileCollaborationBinding binding,
    String method,
    Uri url,
  ) async {
    String authorization;
    try {
      authorization = await authorizer.authorize(
        binding: binding,
        method: method,
        url: url,
      );
    } on MobileCollaborationException {
      rethrow;
    } catch (_) {
      throw const MobileCollaborationException(
        MobileCollaborationErrorKind.authenticationDenied,
        'The device credential could not authorize this request.',
      );
    }
    if (authorization.isEmpty ||
        authorization.length > 8192 ||
        containsControlCharacter(authorization)) {
      throw const MobileCollaborationException(
        MobileCollaborationErrorKind.authenticationDenied,
        'The device credential returned invalid authorization.',
      );
    }
    return authorization;
  }

  Future<MobileTransportResponse> _send(MobileTransportRequest request) async {
    try {
      return await transport.send(request);
    } on MobileCollaborationException {
      rethrow;
    } catch (_) {
      throw const MobileCollaborationException(
        MobileCollaborationErrorKind.serviceUnavailable,
        'The collaboration service is unavailable.',
        retryable: true,
      );
    }
  }

  Map<String, Object?> _responseJson(MobileTransportResponse response) {
    if (response.body.length > _maximumResponseCharacters) {
      throw invalidResponse();
    }
    return decodeJsonObject(response.body);
  }
}

String? optionalString(Map<String, Object?> json, String key) {
  final value = json[key];
  if (value is String &&
      value.isNotEmpty &&
      value.length <= 128 &&
      !containsControlCharacter(value)) {
    return value;
  }
  return null;
}
