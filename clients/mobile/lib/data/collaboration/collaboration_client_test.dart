import 'dart:convert';

import 'client.dart';
import 'contracts.dart';
import 'storage.dart';

const communityOne = '018fbe5f-6f37-7b40-8fb3-1c8d64057001';
const communityTwo = '018fbe5f-6f37-7b40-8fb3-1c8d64057002';
const profileOne = '018fbe5f-6f37-7b40-8fb3-1c8d64057003';
const profileTwo = '018fbe5f-6f37-7b40-8fb3-1c8d64057004';
const nostrPublicKey =
    'abababababababababababababababababababababababababababababababab';
const digestOne =
    '1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a'
    '1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a';
const digestTwo =
    '2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b'
    '2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b';

Future<void> main() async {
  await _upgradeStopsBeforeIdentityOrStorage();
  print('ok - upgrade stops before identity or storage');
  await _switchesCanonicalAccountAndCommunity();
  print('ok - switches canonical account and community');
  await _doesNotDuplicateBindingsOrLocalRecords();
  print('ok - does not duplicate bindings or local records');
}

Future<void> _upgradeStopsBeforeIdentityOrStorage() async {
  final keyValueStore = MemoryKeyValueStore();
  final stateStore = MobileCollaborationStateStore(keyValueStore);
  final transport = FakeTransport(upgradeRequired: true);
  final authorizer = FakeAuthorizer();
  final client = MobileCollaborationClient(
    transport: transport,
    authorizer: authorizer,
    stateStore: stateStore,
  );

  await expectCollaborationError(
    client.activate(binding(accountId: '1', communityId: communityOne)),
    MobileCollaborationErrorKind.upgradeRequired,
  );
  expectEqual(authorizer.calls, 0);
  expectEqual(transport.resourceRequests, 0);
  expectEqual((await stateStore.load()).bindings.length, 0);
  expectEqual(keyValueStore.writes, 0);
}

Future<void> _switchesCanonicalAccountAndCommunity() async {
  final stateStore = MobileCollaborationStateStore(MemoryKeyValueStore());
  final transport = FakeTransport();
  final client = MobileCollaborationClient(
    transport: transport,
    authorizer: FakeAuthorizer(),
    stateStore: stateStore,
  );
  final first = binding(accountId: '1', communityId: communityOne);
  final second = binding(
    accountId: '2',
    communityId: communityTwo,
    profileId: profileTwo,
  );

  await client.activate(first);
  var state = await client.activate(second);
  expectEqual(state.bindings.length, 2);
  expectEqual(state.activeBinding?.accountId, '2');
  expectEqual(state.activeBinding?.communityId, communityTwo);
  expectEqual(state.recordsFor(first).length, 1);
  expectEqual(state.recordsFor(second).length, 1);

  state = await stateStore.switchActive(
    accountId: first.accountId,
    communityId: first.communityId,
  );
  expectEqual(state.activeBinding?.accountId, '1');
  expectEqual(state.activeBinding?.communityId, communityOne);
  expectEqual(state.recordsFor(second).length, 1);
}

Future<void> _doesNotDuplicateBindingsOrLocalRecords() async {
  final keyValueStore = MemoryKeyValueStore();
  final stateStore = MobileCollaborationStateStore(keyValueStore);
  final transport = FakeTransport();
  final client = MobileCollaborationClient(
    transport: transport,
    authorizer: FakeAuthorizer(),
    stateStore: stateStore,
  );
  final current = binding(accountId: '1', communityId: communityOne);

  await stateStore.importResolvedLegacyBinding(current);
  await stateStore.importResolvedLegacyBinding(current);
  await client.activate(current);
  var state = await client.activate(current);
  expectEqual(state.bindings.length, 1);
  expectEqual(state.records.length, 1);
  expectEqual(state.records.single.revision, 1);

  transport.recordRevision = 2;
  state = await client.activate(current);
  expectEqual(state.bindings.length, 1);
  expectEqual(state.records.length, 1);
  expectEqual(state.records.single.revision, 2);
}

MobileCollaborationBinding binding({
  required String accountId,
  required String communityId,
  String profileId = profileOne,
}) => MobileCollaborationBinding(
  accountId: accountId,
  communityId: communityId,
  profileId: profileId,
  nostrPublicKey: nostrPublicKey,
  serviceOrigin: Uri.parse('https://collaboration.example.test'),
  relayUrl: Uri.parse('wss://relay.example.test'),
  credentialReference: 'device-key-$accountId-$communityId',
);

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

class FakeAuthorizer implements MobileRequestAuthorizer {
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
    expectEqual(body, null);
    return 'Nostr signed-request';
  }
}

class FakeTransport implements MobileCollaborationTransport {
  FakeTransport({this.upgradeRequired = false});

  final bool upgradeRequired;
  int resourceRequests = 0;
  int recordRevision = 1;

  @override
  Future<MobileTransportResponse> send(MobileTransportRequest request) async {
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
    expectEqual(request.headers['Authorization'], 'Nostr signed-request');
    final segments = request.url.pathSegments;
    expectEqual(segments.length, 7);
    expectEqual(segments[0], 'v1');
    expectEqual(segments[1], 'collaboration');
    expectEqual(segments[2], 'accounts');
    expectEqual(segments[4], 'communities');
    expectEqual(segments[6], 'bootstrap');
    final accountId = segments[3];
    final communityId = segments[5];
    final profileId = request.url.queryParameters['profile_id'];
    expectTrue(profileId != null);
    return response({
      'account_id': accountId,
      'community_id': communityId,
      'profile_id': profileId,
      'records': [
        {
          'account_id': accountId,
          'community_id': communityId,
          'kind': 'channel',
          'entity_id': 'channel-general',
          'revision': recordRevision,
          'payload_digest': recordRevision == 1 ? digestOne : digestTwo,
        },
      ],
    });
  }
}

MobileTransportResponse response(
  Map<String, Object?> body, {
  int statusCode = 200,
}) => MobileTransportResponse(statusCode: statusCode, body: jsonEncode(body));

Future<void> expectCollaborationError(
  Future<Object?> future,
  MobileCollaborationErrorKind kind,
) async {
  try {
    await future;
  } on MobileCollaborationException catch (error) {
    expectEqual(error.kind, kind);
    return;
  }
  throw StateError('expected MobileCollaborationException($kind)');
}

void expectEqual(Object? actual, Object? expected) {
  if (actual != expected) {
    throw StateError('expected $expected, received $actual');
  }
}

void expectTrue(bool value) {
  if (!value) throw StateError('expected true');
}
