import '../../data/collaboration/contracts.dart';
import 'lifecycle.dart';
import 'pairing.dart';

const sourcePublicKey =
    'abababababababababababababababababababababababababababababababab';
const sessionSecret =
    '1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a'
    '1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a';
const pairingOffer =
    'nostrpair://$sourcePublicKey?secret=$sessionSecret'
    '&relay=wss%3A%2F%2Fpairing.example.test&v=1';

Future<void> main() async {
  await backgroundAndForegroundPreserveGraceAndRefetch();
  print('ok - background and foreground preserve grace and refetch');
  await networkRestoreReconnectsAndRefetches();
  print('ok - network restore reconnects and refetches');
  await pushWakeIsFixedAndRevokedLeaseStaysClosed();
  print('ok - push wake is fixed and revoked lease stays closed');
  await pairingCompletesOnceAndExpiresAtTheDeadline();
  print('ok - pairing completes once and expires at the deadline');
  await upgradeStopsBeforePairingTransport();
  print('ok - upgrade stops before pairing transport');
}

Future<void> backgroundAndForegroundPreserveGraceAndRefetch() async {
  final connection = RecordingConnection();
  final lifecycle = MobileCollaborationLifecycle(connection: connection);

  await lifecycle.enterBackground(1000);
  await lifecycle.enterForeground(4000);
  expectEqual(connection.actions, <String>[]);
  expectEqual(lifecycle.phase, MobileLifecyclePhase.foreground);

  await lifecycle.enterBackground(10000);
  await lifecycle.tick(14999);
  expectEqual(connection.actions, <String>[]);
  await lifecycle.tick(15000);
  expectEqual(connection.actions, ['disconnect']);
  expectEqual(lifecycle.phase, MobileLifecyclePhase.background);

  await lifecycle.enterForeground(16000);
  expectEqual(connection.actions, ['disconnect', 'reconnect', 'fetch']);
  expectEqual(lifecycle.phase, MobileLifecyclePhase.foreground);
}

Future<void> networkRestoreReconnectsAndRefetches() async {
  final connection = RecordingConnection();
  final lifecycle = MobileCollaborationLifecycle(connection: connection);

  await lifecycle.networkRestored(1000);
  expectEqual(connection.actions, ['reconnect', 'fetch']);
  await lifecycle.enterBackground(2000);
  await lifecycle.networkRestored(3000);
  expectEqual(connection.actions, ['reconnect', 'fetch']);
}

Future<void> pushWakeIsFixedAndRevokedLeaseStaysClosed() async {
  final wake = MobilePushWake.fromJson({
    'aps': <String, Object?>{
      'alert': <String, Object?>{'body': 'Reconnect to your relay now'},
      'mutable-content': 1,
    },
  });
  expectThrowsFormat(
    () => MobilePushWake.fromJson({
      'aps': <String, Object?>{
        'alert': <String, Object?>{'body': 'Read a private message'},
        'mutable-content': 1,
      },
    }),
  );

  final connection = RecordingConnection();
  final lifecycle = MobileCollaborationLifecycle(connection: connection);
  final active = MobilePushLease(
    profile: ApprovedMobilePushProfile.iosSandbox,
    leaseGeneration: 2,
    endpointGeneration: 3,
    expiresAtMillis: 20000,
    revoked: false,
  );
  expectTrue(await lifecycle.handlePushWake(wake, active, 5000));
  expectEqual(connection.actions, ['reconnect', 'fetch']);

  await lifecycle.enterBackground(6000);
  await lifecycle.tick(11000);
  connection.actions.clear();
  expectTrue(await lifecycle.handlePushWake(wake, active, 12000));
  expectEqual(connection.actions, ['reconnect', 'fetch', 'disconnect']);

  final revoked = MobilePushLease(
    profile: ApprovedMobilePushProfile.iosSandbox,
    leaseGeneration: 3,
    endpointGeneration: 3,
    expiresAtMillis: 20000,
    revoked: true,
  );
  expectFalse(await lifecycle.handlePushWake(wake, revoked, 13000));
  expectEqual(connection.actions, ['reconnect', 'fetch', 'disconnect']);
  expectEqual(
    pushProfileName(ApprovedMobilePushProfile.iosProduction),
    'buzz-ios-production',
  );
  expectThrowsFormat(() => parsePushProfile('fcm'));
}

Future<void> pairingCompletesOnceAndExpiresAtTheDeadline() async {
  final compatibility = RecordingCompatibility();
  final successfulPort = RecordingPairingPort();
  final coordinator = MobilePairingCoordinator(
    compatibility: compatibility,
    port: successfulPort,
  );
  final session = await coordinator.start(pairingOffer, 1000);
  expectEqual(session.sasCode, '123456');
  expectEqual(session.offer.version, 1);
  expectEqual(session.offer.mode, MobilePairingMode.receiveIdentity);
  expectEqual(successfulPort.openedExpiry, 1000 + nipAbSessionMilliseconds);
  await session.confirm(2000);
  final completion = await session.finish(3000);
  expectEqual(completion.nostrPublicKey, sourcePublicKey);
  expectEqual(session.stage, MobilePairingStage.complete);

  final expiringPort = RecordingPairingPort();
  final expiring = await MobilePairingCoordinator(
    compatibility: compatibility,
    port: expiringPort,
  ).start(pairingOffer, 10000);
  await expiring.confirm(11000);
  await expectPairingError(
    expiring.finish(10000 + nipAbSessionMilliseconds),
    MobilePairingErrorKind.expired,
  );
  expectEqual(expiring.stage, MobilePairingStage.expired);
  expectEqual(expiringPort.cancelCalls, 1);
}

Future<void> upgradeStopsBeforePairingTransport() async {
  final compatibility = RecordingCompatibility(upgradeRequired: true);
  final port = RecordingPairingPort();
  final coordinator = MobilePairingCoordinator(
    compatibility: compatibility,
    port: port,
  );

  await expectCollaborationError(
    coordinator.start(pairingOffer, 1000),
    MobileCollaborationErrorKind.upgradeRequired,
  );
  expectEqual(compatibility.calls, 1);
  expectEqual(port.openCalls, 0);
}

class RecordingConnection implements MobileCollaborationConnection {
  final actions = <String>[];

  @override
  Future<void> disconnect() async => actions.add('disconnect');

  @override
  Future<void> fetchCanonicalState() async => actions.add('fetch');

  @override
  Future<void> reconnect() async => actions.add('reconnect');
}

class RecordingCompatibility implements MobilePairingCompatibilityGate {
  RecordingCompatibility({this.upgradeRequired = false});

  final bool upgradeRequired;
  int calls = 0;

  @override
  Future<void> requireNipAbWrite() async {
    calls += 1;
    if (upgradeRequired) {
      throw const MobileCollaborationException(
        MobileCollaborationErrorKind.upgradeRequired,
        'Upgrade required.',
        minimumVersion: '0.0.1+2',
      );
    }
  }
}

class RecordingPairingPort implements MobileNipAbPort {
  int openCalls = 0;
  int confirmCalls = 0;
  int finishCalls = 0;
  int cancelCalls = 0;
  int? openedExpiry;

  @override
  Future<void> cancel() async {
    cancelCalls += 1;
  }

  @override
  Future<void> confirmSas() async {
    confirmCalls += 1;
  }

  @override
  Future<MobilePairingCompletion> finish() async {
    finishCalls += 1;
    return MobilePairingCompletion(
      nostrPublicKey: sourcePublicKey,
      credentialReference: 'protected-mobile-identity',
    );
  }

  @override
  Future<String> open(MobilePairingOffer offer, int expiresAtMillis) async {
    openCalls += 1;
    openedExpiry = expiresAtMillis;
    expectEqual(offer.sourcePublicKey, sourcePublicKey);
    return '123456';
  }
}

Future<void> expectPairingError(
  Future<Object?> future,
  MobilePairingErrorKind kind,
) async {
  try {
    await future;
  } on MobilePairingException catch (error) {
    expectEqual(error.kind, kind);
    return;
  }
  throw StateError('expected MobilePairingException($kind)');
}

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

void expectThrowsFormat(void Function() operation) {
  try {
    operation();
  } on FormatException {
    return;
  }
  throw StateError('expected FormatException');
}

void expectEqual(Object? actual, Object? expected) {
  if (actual is List<Object?> && expected is List<Object?>) {
    if (actual.length == expected.length &&
        Iterable<int>.generate(
          actual.length,
        ).every((index) => actual[index] == expected[index])) {
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
