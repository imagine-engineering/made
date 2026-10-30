import 'dart:convert';

import '../../lib/data/collaboration/client.dart';
import '../../lib/data/collaboration/contracts.dart';
import '../../lib/data/collaboration/storage.dart';
import '../../lib/platform/collaboration_lifecycle/lifecycle.dart';
import '../../lib/platform/collaboration_lifecycle/pairing.dart';

const communityId = '018fbe5f-6f37-7b40-8fb3-1c8d64057001';
const profileId = '018fbe5f-6f37-7b40-8fb3-1c8d64057002';
const nostrPublicKey =
    'abababababababababababababababababababababababababababababababab';
const digestOne =
    '1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a'
    '1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a';
const digestTwo =
    '2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b'
    '2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b';
const pairingSecret =
    '3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c'
    '3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c';

Future<void> main() async {
  await supportedVersionCoversFeaturesOfflineAndPrivacy();
  print('ok - supported version covers features, offline state, and privacy');
  await deviceAndSimulatorReconnectAndWakeMatrix();
  print('ok - device and simulator reconnect and wake matrix');
  await frozenAndMigratedPairingVersionCompletes();
  print('ok - frozen and migrated pairing version completes');
  await minimumVersionStopsBeforeIdentityResourceOrStorage();
  print('ok - minimum version stops before identity, resource, or storage');
}

Future<void> supportedVersionCoversFeaturesOfflineAndPrivacy() async {
  expectEqual(mobileClientVersion, '0.0.0+1');
  final harness = MobileHarness();
  final state = await harness.client.activate(harness.binding);

  expectEqual(state.records.length, 6);
  expectEqual(state.records.map((record) => record.kind).toSet(), {
    MobileEntityKind.channel,
    MobileEntityKind.message,
    MobileEntityKind.directMessage,
    MobileEntityKind.readState,
    MobileEntityKind.searchResult,
    MobileEntityKind.media,
  });
  expectTrue(state.records.every((record) => record.revision == 1));
  final stored = harness.keyValueStore.values[mobileCollaborationStateKey]!;
  for (final privateMarker in [
    'nsec',
    'private message body',
    'direct message ciphertext',
    'search query',
    'media bytes',
  ]) {
    expectFalse(stored.contains(privateMarker));
  }

  harness.transport.offline = true;
  final authorizationCalls = harness.authorizer.calls;
  await expectCollaborationError(
    harness.client.activate(harness.binding),
    MobileCollaborationErrorKind.serviceUnavailable,
  );
  expectEqual(harness.authorizer.calls, authorizationCalls);
  final offline = await harness.stateStore.load();
  expectEqual(offline.records.length, 6);
  expectEqual(offline.activeBinding?.identityKey, harness.binding.identityKey);
}

Future<void> deviceAndSimulatorReconnectAndWakeMatrix() async {
  for (final profile in [
    ApprovedMobilePushProfile.iosProduction,
    ApprovedMobilePushProfile.iosSandbox,
  ]) {
    final harness = MobileHarness();
    await harness.client.activate(harness.binding);
    final connection = MatrixConnection(harness.client, harness.binding);
    final lifecycle = MobileCollaborationLifecycle(connection: connection);

    await lifecycle.enterBackground(1000);
    await lifecycle.tick(6000);
    harness.transport.recordRevision = 2;
    await lifecycle.enterForeground(7000);
    expectEqual(connection.actions, ['disconnect', 'reconnect', 'fetch']);
    var state = await harness.stateStore.load();
    expectEqual(state.records.length, 6);
    expectTrue(state.records.every((record) => record.revision == 2));

    await lifecycle.enterBackground(8000);
    await lifecycle.tick(13000);
    connection.actions.clear();
    final wake = MobilePushWake.fromJson({
      'aps': <String, Object?>{
        'alert': <String, Object?>{'body': 'Reconnect to your relay now'},
        'mutable-content': 1,
      },
    });
    final lease = MobilePushLease(
      profile: profile,
      leaseGeneration: 4,
      endpointGeneration: 7,
      expiresAtMillis: 20000,
      revoked: false,
    );
    expectTrue(await lifecycle.handlePushWake(wake, lease, 14000));
    expectEqual(connection.actions, ['reconnect', 'fetch', 'disconnect']);
    state = await harness.stateStore.load();
    expectEqual(state.records.length, 6);
    expectEqual(
      pushProfileName(profile),
      profile == ApprovedMobilePushProfile.iosProduction
          ? 'buzz-ios-production'
          : 'buzz-ios-sandbox',
    );
  }
}

Future<void> frozenAndMigratedPairingVersionCompletes() async {
  final harness = MobileHarness();
  final port = MatrixPairingPort();
  final coordinator = MobilePairingCoordinator(
    compatibility: ClientPairingGate(harness.client, harness.binding),
    port: port,
  );
  final offer =
      'nostrpair://$nostrPublicKey?secret=$pairingSecret'
      '&relay=wss%3A%2F%2Fpairing.example.test';
  final session = await coordinator.start(offer, 1000);
  expectEqual(session.offer.version, 1);
  expectEqual(session.sasCode, '654321');
  await session.confirm(2000);
  final completion = await session.finish(3000);
  expectEqual(completion.nostrPublicKey, nostrPublicKey);
  expectEqual(completion.credentialReference, 'protected-paired-identity');
  expectEqual(session.stage, MobilePairingStage.complete);
  expectThrowsPairing(
    () => MobilePairingOffer.parse('$offer&v=2'),
    MobilePairingErrorKind.invalidOffer,
  );
}

Future<void> minimumVersionStopsBeforeIdentityResourceOrStorage() async {
  final harness = MobileHarness(upgradeRequired: true);
  final error = await expectCollaborationError(
    harness.client.activate(harness.binding),
    MobileCollaborationErrorKind.upgradeRequired,
  );
  expectEqual(error.minimumVersion, '0.0.1+2');
  expectEqual(error.maximumVersion, '0.0.1+5');
  expectEqual(harness.authorizer.calls, 0);
  expectEqual(harness.transport.resourceRequests, 0);
  expectEqual(harness.keyValueStore.writes, 0);

  final pairingPort = MatrixPairingPort();
  final coordinator = MobilePairingCoordinator(
    compatibility: ClientPairingGate(harness.client, harness.binding),
    port: pairingPort,
  );
  await expectCollaborationError(
    coordinator.start(
      'nostrpair://$nostrPublicKey?secret=$pairingSecret'
      '&relay=wss%3A%2F%2Fpairing.example.test&v=1',
      1000,
    ),
    MobileCollaborationErrorKind.upgradeRequired,
  );
  expectEqual(pairingPort.openCalls, 0);
  expectEqual(harness.transport.resourceRequests, 0);
  expectEqual(harness.keyValueStore.writes, 0);
}

class MobileHarness {
  MobileHarness({bool upgradeRequired = false})
    : keyValueStore = MemoryKeyValueStore(),
      transport = MatrixTransport(upgradeRequired: upgradeRequired),
      authorizer = MatrixAuthorizer() {
    stateStore = MobileCollaborationStateStore(keyValueStore);
    binding = MobileCollaborationBinding(
      accountId: '42',
      communityId: communityId,
      profileId: profileId,
      nostrPublicKey: nostrPublicKey,
      serviceOrigin: Uri.parse('https://collaboration.example.test'),
      relayUrl: Uri.parse('wss://relay.example.test'),
      credentialReference: 'protected-device-identity',
    );
    client = MobileCollaborationClient(
      transport: transport,
      authorizer: authorizer,
      stateStore: stateStore,
    );
  }

  final MemoryKeyValueStore keyValueStore;
  final MatrixTransport transport;
  final MatrixAuthorizer authorizer;
  late final MobileCollaborationStateStore stateStore;
  late final MobileCollaborationBinding binding;
  late final MobileCollaborationClient client;
}

class MemoryKeyValueStore implements MobileKeyValueStore {
  final values = <String, String>{};
  int writes = 0;

  @override
  Future<String?> read(String key) async => values[key];

  @override
  Future<void> write(String key, String value) async {
    writes += 1;
    values[key] = value;
  }
}

class MatrixAuthorizer implements MobileRequestAuthorizer {
  int calls = 0;

  @override
  Future<String> authorize({
    required MobileCollaborationBinding binding,
    required String method,
    required Uri url,
    String? body,
  }) async {
    calls += 1;
    expectEqual(method, 'GET');
    expectEqual(url.queryParameters['profile_id'], binding.profileId);
    return 'Nostr canonical-mobile-request';
  }
}

class MatrixTransport implements MobileCollaborationTransport {
  MatrixTransport({this.upgradeRequired = false});

  final bool upgradeRequired;
  bool offline = false;
  int recordRevision = 1;
  int resourceRequests = 0;

  @override
  Future<MobileTransportResponse> send(MobileTransportRequest request) async {
    if (offline) throw StateError('offline');
    if (request.url.path == mobileCompatibilityPath) {
      final body = jsonObject(jsonDecode(request.body!));
      expectEqual(body['client_id'], mobileClientId);
      expectEqual(body['client_version'], mobileClientVersion);
      if (upgradeRequired) {
        return response({
          'policy_version': 1,
          'outcome': 'upgrade_required',
          'client_id': mobileClientId,
          'minimum_client_version': '0.0.1+2',
          'maximum_client_version': '0.0.1+5',
          'selected_features': <Object>[],
          'retryable': false,
        }, statusCode: 426);
      }
      return response({
        'policy_version': 1,
        'outcome': 'supported',
        'client_id': mobileClientId,
        'minimum_client_version': mobileClientVersion,
        'maximum_client_version': mobileClientVersion,
        'selected_features': body['features'],
        'retryable': false,
      });
    }

    resourceRequests += 1;
    expectEqual(
      request.headers['Authorization'],
      'Nostr canonical-mobile-request',
    );
    final records = <Map<String, Object?>>[];
    for (final kind in MobileEntityKind.values) {
      records.add({
        'account_id': '42',
        'community_id': communityId,
        'kind': kind.name,
        'entity_id': '${kind.name}-canonical-id',
        'revision': recordRevision,
        'payload_digest': recordRevision == 1 ? digestOne : digestTwo,
      });
    }
    return response({
      'account_id': '42',
      'community_id': communityId,
      'profile_id': profileId,
      'records': records,
    });
  }
}

class MatrixConnection implements MobileCollaborationConnection {
  MatrixConnection(this.client, this.binding);

  final MobileCollaborationClient client;
  final MobileCollaborationBinding binding;
  final actions = <String>[];

  @override
  Future<void> disconnect() async => actions.add('disconnect');

  @override
  Future<void> fetchCanonicalState() async {
    actions.add('fetch');
    await client.activate(binding);
  }

  @override
  Future<void> reconnect() async => actions.add('reconnect');
}

class ClientPairingGate implements MobilePairingCompatibilityGate {
  ClientPairingGate(this.client, this.binding);

  final MobileCollaborationClient client;
  final MobileCollaborationBinding binding;

  @override
  Future<void> requireNipAbWrite() => client.requireWriteCompatibility(binding);
}

class MatrixPairingPort implements MobileNipAbPort {
  int openCalls = 0;

  @override
  Future<void> cancel() async {}

  @override
  Future<void> confirmSas() async {}

  @override
  Future<MobilePairingCompletion> finish() async => MobilePairingCompletion(
    nostrPublicKey: nostrPublicKey,
    credentialReference: 'protected-paired-identity',
  );

  @override
  Future<String> open(MobilePairingOffer offer, int expiresAtMillis) async {
    openCalls += 1;
    expectEqual(offer.sourcePublicKey, nostrPublicKey);
    expectEqual(expiresAtMillis, 1000 + nipAbSessionMilliseconds);
    return '654321';
  }
}

MobileTransportResponse response(
  Map<String, Object?> body, {
  int statusCode = 200,
}) => MobileTransportResponse(statusCode: statusCode, body: jsonEncode(body));

Future<MobileCollaborationException> expectCollaborationError(
  Future<Object?> future,
  MobileCollaborationErrorKind kind,
) async {
  try {
    await future;
  } on MobileCollaborationException catch (error) {
    expectEqual(error.kind, kind);
    return error;
  }
  throw StateError('expected MobileCollaborationException($kind)');
}

void expectThrowsPairing(
  void Function() operation,
  MobilePairingErrorKind kind,
) {
  try {
    operation();
  } on MobilePairingException catch (error) {
    expectEqual(error.kind, kind);
    return;
  }
  throw StateError('expected MobilePairingException($kind)');
}

void expectEqual(Object? actual, Object? expected) {
  if (actual is List<Object?> && expected is List<Object?>) {
    if (actual.length == expected.length &&
        Iterable<int>.generate(
          actual.length,
        ).every((index) => actual[index] == expected[index])) {
      return;
    }
  } else if (actual is Set<Object?> && expected is Set<Object?>) {
    if (actual.length == expected.length && actual.containsAll(expected)) {
      return;
    }
  } else if (actual == expected) {
    return;
  }
  throw StateError('expected $expected, received $actual');
}

void expectTrue(bool value) {
  if (!value) throw StateError('expected true');
}

void expectFalse(bool value) {
  if (value) throw StateError('expected false');
}
