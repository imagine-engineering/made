enum MobileLifecyclePhase { foreground, gracePeriod, background }

enum ApprovedMobilePushProfile { iosProduction, iosSandbox }

abstract interface class MobileCollaborationConnection {
  Future<void> disconnect();

  Future<void> reconnect();

  Future<void> fetchCanonicalState();
}

class MobilePushLease {
  MobilePushLease({
    required this.profile,
    required this.leaseGeneration,
    required this.endpointGeneration,
    required this.expiresAtMillis,
    required this.revoked,
  }) {
    if (leaseGeneration < 1 || endpointGeneration < 1 || expiresAtMillis < 1) {
      throw const FormatException('Invalid push lease');
    }
  }

  final ApprovedMobilePushProfile profile;
  final int leaseGeneration;
  final int endpointGeneration;
  final int expiresAtMillis;
  final bool revoked;
}

class MobilePushWake {
  const MobilePushWake._();

  factory MobilePushWake.fromJson(Map<String, Object?> json) {
    if (json.length != 1) {
      throw const FormatException('Invalid push wake');
    }
    final aps = json['aps'];
    if (aps is! Map<String, Object?> ||
        aps.length != 2 ||
        aps['mutable-content'] != 1) {
      throw const FormatException('Invalid push wake');
    }
    final alert = aps['alert'];
    if (alert is! Map<String, Object?> ||
        alert.length != 1 ||
        alert['body'] != 'Reconnect to your relay now') {
      throw const FormatException('Invalid push wake');
    }
    return const MobilePushWake._();
  }

  bool isAuthorizedBy(MobilePushLease lease, int nowMillis) =>
      nowMillis > 0 && nowMillis < lease.expiresAtMillis && !lease.revoked;
}

class MobileCollaborationLifecycle {
  MobileCollaborationLifecycle({
    required this.connection,
    this.backgroundGrace = const Duration(seconds: 5),
  }) {
    if (backgroundGrace <= Duration.zero ||
        backgroundGrace > const Duration(seconds: 30)) {
      throw const FormatException('Invalid background grace period');
    }
  }

  final MobileCollaborationConnection connection;
  final Duration backgroundGrace;

  MobileLifecyclePhase _phase = MobileLifecyclePhase.foreground;
  int? _backgroundedAtMillis;
  bool _disconnectedForBackground = false;
  Future<void> _tail = Future.value();

  MobileLifecyclePhase get phase => _phase;

  Future<void> enterBackground(int nowMillis) => _serialized(() async {
    _validNow(nowMillis);
    _phase = MobileLifecyclePhase.gracePeriod;
    _backgroundedAtMillis = nowMillis;
    _disconnectedForBackground = false;
  });

  Future<void> tick(int nowMillis) => _serialized(() async {
    _validNow(nowMillis);
    final backgroundedAt = _backgroundedAtMillis;
    if (_phase != MobileLifecyclePhase.gracePeriod ||
        backgroundedAt == null ||
        nowMillis - backgroundedAt < backgroundGrace.inMilliseconds) {
      return;
    }
    await connection.disconnect();
    _phase = MobileLifecyclePhase.background;
    _disconnectedForBackground = true;
  });

  Future<void> enterForeground(int nowMillis) => _serialized(() async {
    _validNow(nowMillis);
    final backgroundedAt = _backgroundedAtMillis;
    final needsReconnect =
        _disconnectedForBackground ||
        (backgroundedAt != null &&
            nowMillis - backgroundedAt >= backgroundGrace.inMilliseconds);
    _phase = MobileLifecyclePhase.foreground;
    _backgroundedAtMillis = null;
    _disconnectedForBackground = false;
    if (needsReconnect) await _reconnectAndFetch();
  });

  Future<void> networkRestored(int nowMillis) => _serialized(() async {
    _validNow(nowMillis);
    if (_phase == MobileLifecyclePhase.foreground) {
      await _reconnectAndFetch();
    }
  });

  Future<bool> handlePushWake(
    MobilePushWake wake,
    MobilePushLease lease,
    int nowMillis,
  ) => _serialized(() async {
    _validNow(nowMillis);
    if (!wake.isAuthorizedBy(lease, nowMillis)) return false;
    await connection.reconnect();
    try {
      await connection.fetchCanonicalState();
    } finally {
      if (_phase != MobileLifecyclePhase.foreground) {
        await connection.disconnect();
      }
    }
    return true;
  });

  Future<void> _reconnectAndFetch() async {
    await connection.reconnect();
    await connection.fetchCanonicalState();
  }

  Future<T> _serialized<T>(Future<T> Function() operation) {
    final result = _tail.then((_) => operation());
    _tail = result.then<void>((_) {}, onError: (_, _) {});
    return result;
  }
}

ApprovedMobilePushProfile parsePushProfile(Object? value) => switch (value) {
  'buzz-ios-production' => ApprovedMobilePushProfile.iosProduction,
  'buzz-ios-sandbox' => ApprovedMobilePushProfile.iosSandbox,
  _ => throw const FormatException('Unsupported push profile'),
};

String pushProfileName(ApprovedMobilePushProfile profile) => switch (profile) {
  ApprovedMobilePushProfile.iosProduction => 'buzz-ios-production',
  ApprovedMobilePushProfile.iosSandbox => 'buzz-ios-sandbox',
};

int requiredPositiveInt(Object? value) {
  if (value is! int || value < 1) {
    throw const FormatException('Expected a positive integer');
  }
  return value;
}

String requiredString(Object? value) {
  if (value is! String || value.isEmpty || value.length > 128) {
    throw const FormatException('Expected a bounded string');
  }
  return value;
}

void _validNow(int nowMillis) {
  if (nowMillis < 1) throw const FormatException('Invalid current time');
}
