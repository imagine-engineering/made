use uuid::Uuid;

use crate::{AggregateId, CommunityId, PrincipalId};

const ZED_CHANNEL_BINDING_NAMESPACE: Uuid =
    Uuid::from_u128(0x5d64_4f62_3d17_5eaa_9b80_e5fa_4c6b_0182);

pub fn community_id_for_legacy_root_channel(root_channel_id: u64) -> CommunityId {
    CommunityId::from_uuid(Uuid::new_v5(
        &ZED_CHANNEL_BINDING_NAMESPACE,
        format!("community:{root_channel_id}").as_bytes(),
    ))
}

pub fn channel_id_for_legacy_channel(channel_id: u64) -> AggregateId {
    AggregateId::from_uuid(Uuid::new_v5(
        &ZED_CHANNEL_BINDING_NAMESPACE,
        format!("channel:{channel_id}").as_bytes(),
    ))
}

pub fn principal_id_for_legacy_user(community_id: CommunityId, legacy_user_id: u64) -> PrincipalId {
    let mut identity = community_id.as_uuid().as_bytes().to_vec();
    identity.extend_from_slice(&legacy_user_id.to_be_bytes());
    PrincipalId::from_uuid(Uuid::new_v5(&ZED_CHANNEL_BINDING_NAMESPACE, &identity))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_bindings_are_stable_and_scoped() {
        let first = community_id_for_legacy_root_channel(1);
        let second = community_id_for_legacy_root_channel(2);
        assert_ne!(first, second);
        assert_eq!(
            channel_id_for_legacy_channel(7),
            channel_id_for_legacy_channel(7)
        );
        assert_ne!(
            principal_id_for_legacy_user(first, 9),
            principal_id_for_legacy_user(second, 9)
        );
    }
}
