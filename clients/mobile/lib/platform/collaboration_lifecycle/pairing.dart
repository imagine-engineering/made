const nipAbSessionMilliseconds = 120000;

enum MobilePairingMode { receiveIdentity, sendIdentity }

enum MobilePairingStage {
  confirmingSas,
  transferring,
  complete,
  cancelled,
  expired,
}

enum MobilePairingErrorKind { invalidOffer, invalidState, expired }

class MobilePairingException implements Exception {
  const MobilePairingException(this.kind, this.message);

  final MobilePairingErrorKind kind;
  final String message;

  @override
  String toString() => 'MobilePairingException($kind)';
}

class MobilePairingOffer {
  MobilePairingOffer._({
    required this.sourcePublicKey,
    required this.sessionSecret,
    required this.relays,
    required this.version,
    required this.mode,
  });

  factory MobilePairingOffer.parse(String value) {
    if (value.isEmpty || value.length > 2048) throw invalidOffer();
    final uri = Uri.tryParse(value);
    if (uri == null ||
        uri.scheme != 'nostrpair' ||
        uri.host.isEmpty ||
        uri.userInfo.isNotEmpty ||
        uri.hasPort ||
        uri.path.isNotEmpty ||
        uri.fragment.isNotEmpty) {
      throw invalidOffer();
    }
    final sourcePublicKey = validLowerHex(uri.host, 64);
    final parameters = uri.queryParametersAll;
    if (parameters.keys.any(
      (key) => !{'secret', 'relay', 'v', 'mode'}.contains(key),
    )) {
      throw invalidOffer();
    }
    final secretValues = parameters['secret'];
    final versionValues = parameters['v'];
    final modeValues = parameters['mode'];
    if (secretValues?.length != 1 ||
        (versionValues != null && versionValues.length != 1) ||
        (modeValues != null && modeValues.length != 1)) {
      throw invalidOffer();
    }
    final sessionSecret = validLowerHex(secretValues!.single, 64);
    if (RegExp(r'^0+$').hasMatch(sessionSecret)) throw invalidOffer();
    final version = versionValues == null
        ? 1
        : int.tryParse(versionValues.single);
    if (version != 1) throw invalidOffer();

    final relayValues = parameters['relay'];
    if (relayValues == null || relayValues.isEmpty || relayValues.length > 4) {
      throw invalidOffer();
    }
    final relays = <Uri>[];
    for (final value in relayValues) {
      final relay = Uri.tryParse(value);
      if (relay == null ||
          !_validPairingRelay(relay) ||
          relays.contains(relay)) {
        throw invalidOffer();
      }
      relays.add(relay);
    }
    final mode = switch (modeValues?.single) {
      null => MobilePairingMode.receiveIdentity,
      'recover' => MobilePairingMode.sendIdentity,
      _ => throw invalidOffer(),
    };
    return MobilePairingOffer._(
      sourcePublicKey: sourcePublicKey,
      sessionSecret: sessionSecret,
      relays: List.unmodifiable(relays),
      version: version!,
      mode: mode,
    );
  }

  final String sourcePublicKey;
  final String sessionSecret;
  final List<Uri> relays;
  final int version;
  final MobilePairingMode mode;
}

class MobilePairingCompletion {
  MobilePairingCompletion({
    required String nostrPublicKey,
    required String credentialReference,
  }) : nostrPublicKey = validCompletionPublicKey(nostrPublicKey),
       credentialReference = validCredentialReference(credentialReference);

  final String nostrPublicKey;
  final String credentialReference;
}

abstract interface class MobilePairingCompatibilityGate {
  Future<void> requireNipAbWrite();
}

abstract interface class MobileNipAbPort {
  Future<String> open(MobilePairingOffer offer, int expiresAtMillis);

  Future<void> confirmSas();

  Future<MobilePairingCompletion> finish();

  Future<void> cancel();
}

class MobilePairingCoordinator {
  MobilePairingCoordinator({required this.compatibility, required this.port});

  final MobilePairingCompatibilityGate compatibility;
  final MobileNipAbPort port;

  Future<MobilePairingSession> start(String rawOffer, int nowMillis) async {
    validNow(nowMillis);
    await compatibility.requireNipAbWrite();
    final offer = MobilePairingOffer.parse(rawOffer);
    final expiresAtMillis = nowMillis + nipAbSessionMilliseconds;
    final sasCode = await port.open(offer, expiresAtMillis);
    if (!RegExp(r'^[0-9]{6}$').hasMatch(sasCode)) {
      await port.cancel();
      throw const MobilePairingException(
        MobilePairingErrorKind.invalidState,
        'The NIP-AB engine returned an invalid SAS code.',
      );
    }
    return MobilePairingSession._(
      offer: offer,
      port: port,
      sasCode: sasCode,
      expiresAtMillis: expiresAtMillis,
    );
  }
}

class MobilePairingSession {
  MobilePairingSession._({
    required this.offer,
    required MobileNipAbPort port,
    required this.sasCode,
    required this.expiresAtMillis,
  }) : _port = port;

  final MobilePairingOffer offer;
  final MobileNipAbPort _port;
  final String sasCode;
  final int expiresAtMillis;

  MobilePairingStage _stage = MobilePairingStage.confirmingSas;
  Future<void> _tail = Future.value();

  MobilePairingStage get stage => _stage;

  Future<void> confirm(int nowMillis) => _serialized(() async {
    await _requireActive(nowMillis, MobilePairingStage.confirmingSas);
    await _port.confirmSas();
    _stage = MobilePairingStage.transferring;
  });

  Future<MobilePairingCompletion> finish(int nowMillis) =>
      _serialized(() async {
        await _requireActive(nowMillis, MobilePairingStage.transferring);
        final completion = await _port.finish();
        if (offer.mode == MobilePairingMode.receiveIdentity &&
            completion.nostrPublicKey != offer.sourcePublicKey) {
          await _port.cancel();
          _stage = MobilePairingStage.cancelled;
          throw const MobilePairingException(
            MobilePairingErrorKind.invalidState,
            'The transferred identity does not match the NIP-AB source.',
          );
        }
        _stage = MobilePairingStage.complete;
        return completion;
      });

  Future<void> cancel() => _serialized(() async {
    if (_stage == MobilePairingStage.complete ||
        _stage == MobilePairingStage.cancelled ||
        _stage == MobilePairingStage.expired) {
      return;
    }
    await _port.cancel();
    _stage = MobilePairingStage.cancelled;
  });

  Future<void> _requireActive(
    int nowMillis,
    MobilePairingStage expected,
  ) async {
    validNow(nowMillis);
    if (nowMillis >= expiresAtMillis) {
      await _port.cancel();
      _stage = MobilePairingStage.expired;
      throw const MobilePairingException(
        MobilePairingErrorKind.expired,
        'The NIP-AB pairing session expired.',
      );
    }
    if (_stage != expected) {
      throw const MobilePairingException(
        MobilePairingErrorKind.invalidState,
        'The NIP-AB pairing transition is invalid.',
      );
    }
  }

  Future<T> _serialized<T>(Future<T> Function() operation) {
    final result = _tail.then((_) => operation());
    _tail = result.then<void>((_) {}, onError: (_, _) {});
    return result;
  }
}

String validLowerHex(String value, int length) {
  if (value.length != length ||
      !RegExp('^[0-9a-f]{$length}\$').hasMatch(value)) {
    throw invalidOffer();
  }
  return value;
}

String validCompletionPublicKey(String value) {
  if (value.length != 64 || !RegExp(r'^[0-9a-f]+$').hasMatch(value)) {
    throw const MobilePairingException(
      MobilePairingErrorKind.invalidState,
      'The paired identity is invalid.',
    );
  }
  return value;
}

String validCredentialReference(String value) {
  if (value.isEmpty ||
      value.length > 256 ||
      value.runes.any((codePoint) => codePoint <= 31 || codePoint == 127)) {
    throw const MobilePairingException(
      MobilePairingErrorKind.invalidState,
      'The protected credential reference is invalid.',
    );
  }
  return value;
}

bool _validPairingRelay(Uri relay) {
  final localWebSocket =
      relay.scheme == 'ws' &&
      (relay.host == 'localhost' ||
          relay.host.endsWith('.localhost') ||
          relay.host == '127.0.0.1' ||
          relay.host == '::1');
  return (relay.scheme == 'wss' || localWebSocket) &&
      relay.userInfo.isEmpty &&
      relay.query.isEmpty &&
      relay.fragment.isEmpty;
}

void validNow(int nowMillis) {
  if (nowMillis < 1) {
    throw const MobilePairingException(
      MobilePairingErrorKind.invalidState,
      'The pairing clock is invalid.',
    );
  }
}

MobilePairingException invalidOffer() => const MobilePairingException(
  MobilePairingErrorKind.invalidOffer,
  'The NIP-AB pairing offer is invalid.',
);
