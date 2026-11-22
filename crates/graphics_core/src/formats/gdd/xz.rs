use std::io::{Cursor, Read};

use lzma_rust2::XzReader;

use super::{GddArchiveBudget, GddArchiveError, StagedArchive, StagedEntries};

const TAR_BLOCK_BYTES: usize = 512;
const XZ_HEADER_BYTES: usize = 12;
const XZ_BLOCK_HEADER_BYTES: usize = 12;
const XZ_FOOTER_BYTES: usize = 12;
const CRC64_BYTES: usize = 8;

pub(super) fn stage_xz(
    bytes: &[u8],
    budget: &GddArchiveBudget,
) -> Result<StagedArchive, GddArchiveError> {
    if bytes.len() > budget.max_archive_bytes {
        return Err(GddArchiveError::BudgetExceeded);
    }
    let tar_byte_budget = budget
        .max_total_entry_bytes
        .saturating_add((budget.max_entries as u64).saturating_mul(4096))
        .saturating_add(2 * TAR_BLOCK_BYTES as u64);
    let tar_bytes = decode_xz_tar(bytes, tar_byte_budget)?;
    let mut archive = tar::Archive::new(Cursor::new(tar_bytes.as_slice()));
    let mut entries = StagedEntries::new(budget);
    for entry in archive.entries()? {
        let entry = entry?;
        if !entry.header().entry_type().is_file() {
            return Err(GddArchiveError::InconsistentEntries);
        }
        let name = std::str::from_utf8(&entry.path_bytes())
            .map_err(|_| GddArchiveError::InconsistentEntries)?
            .to_owned();
        let size = entry.size();
        entries.stage(name, size, entry)?;
    }

    let tail_start = usize::try_from(archive.into_inner().position())
        .ok()
        .and_then(|position| position.checked_sub(TAR_BLOCK_BYTES))
        .ok_or(GddArchiveError::InconsistentEntries)?;
    let tail = tar_bytes
        .get(tail_start..)
        .ok_or(GddArchiveError::InconsistentEntries)?;
    if tail.len() < 2 * TAR_BLOCK_BYTES
        || !tar_bytes.len().is_multiple_of(TAR_BLOCK_BYTES)
        || tail.iter().any(|byte| *byte != 0)
    {
        return Err(GddArchiveError::InconsistentEntries);
    }
    entries.finish()
}

fn decode_xz_tar(bytes: &[u8], limit: u64) -> Result<Vec<u8>, GddArchiveError> {
    let size = xz_tar_stream_size(bytes, limit)?;
    let mut decoder = XzReader::new(Cursor::new(bytes), false);
    let mut output = Vec::new();
    decoder.by_ref().take(size + 1).read_to_end(&mut output)?;
    if u64::try_from(output.len()) != Ok(size)
        || decoder.into_inner().position() != bytes.len() as u64
    {
        return Err(GddArchiveError::InconsistentEntries);
    }
    Ok(output)
}

fn xz_tar_stream_size(bytes: &[u8], limit: u64) -> Result<u64, GddArchiveError> {
    let stream_header = bytes
        .first_chunk::<XZ_HEADER_BYTES>()
        .filter(|header| header.starts_with(b"\xFD7zXZ\0\0\x04"))
        .ok_or(GddArchiveError::UnsupportedEnvelope)?;
    let block_end = XZ_HEADER_BYTES + XZ_BLOCK_HEADER_BYTES;
    let block_header = bytes
        .get(XZ_HEADER_BYTES..block_end)
        .filter(|header| header.starts_with(&[2, 0, 0x21, 1]))
        .ok_or(GddArchiveError::UnsupportedEnvelope)?;
    // The pinned exporter uses one LZMA2 block with the default 8 MiB
    // dictionary. The decoder allocates the dictionary before emitting bytes.
    if block_header[4] > 22 {
        return Err(GddArchiveError::BudgetExceeded);
    }
    let footer = bytes
        .last_chunk::<XZ_FOOTER_BYTES>()
        .filter(|footer| footer[8..10] == stream_header[6..8] && footer[10..] == *b"YZ")
        .ok_or(GddArchiveError::InconsistentEntries)?;
    let backward_size = u32::from_le_bytes([footer[4], footer[5], footer[6], footer[7]]);
    let index_size = usize::try_from((u64::from(backward_size) + 1) * 4)
        .map_err(|_| GddArchiveError::InconsistentEntries)?;
    let index_end = bytes.len() - footer.len();
    let index_start = index_end
        .checked_sub(index_size)
        .filter(|start| *start >= block_end)
        .ok_or(GddArchiveError::InconsistentEntries)?;
    if !bytes[index_start..index_end].starts_with(&[0, 1]) {
        return Err(GddArchiveError::UnsupportedEnvelope);
    }

    let body = &bytes[block_end..index_start];
    let mut chunks = body;
    let size = bounded_chunk_length(&mut chunks, limit)?;
    let compressed_bytes = body.len() - chunks.len();
    let padding = (4 - compressed_bytes % 4) % 4;
    // A second block must be rejected before the decoder sees its dictionary.
    if chunks.len() != padding + CRC64_BYTES {
        return Err(GddArchiveError::InconsistentEntries);
    }
    Ok(size)
}

fn bounded_chunk_length(chunks: &mut &[u8], limit: u64) -> Result<u64, GddArchiveError> {
    let mut total = 0_u64;
    loop {
        let (&control, remaining) = chunks
            .split_first()
            .ok_or(GddArchiveError::InconsistentEntries)?;
        if control == 0 {
            *chunks = remaining;
            return Ok(total);
        }
        let (encoded_size, decoded_size, header_bytes) = match control {
            1 | 2 => {
                let length = remaining
                    .first_chunk::<2>()
                    .map(|size| usize::from(u16::from_be_bytes(*size)) + 1)
                    .ok_or(GddArchiveError::InconsistentEntries)?;
                (length, length as u64, 2)
            }
            0x80..=0xFF => {
                let size = remaining
                    .first_chunk::<4>()
                    .ok_or(GddArchiveError::InconsistentEntries)?;
                let decoded_high = u64::from(control & 0x1F) << 16;
                let decoded_low = u64::from(u16::from_be_bytes([size[0], size[1]]));
                let encoded = usize::from(u16::from_be_bytes([size[2], size[3]])) + 1;
                (
                    encoded,
                    decoded_high + decoded_low + 1,
                    4 + usize::from(control >= 0xC0),
                )
            }
            _ => return Err(GddArchiveError::InconsistentEntries),
        };
        total = total
            .checked_add(decoded_size)
            .filter(|total| *total <= limit && *total < u64::MAX)
            .ok_or(GddArchiveError::BudgetExceeded)?;
        *chunks = remaining
            .get(header_bytes + encoded_size..)
            .ok_or(GddArchiveError::InconsistentEntries)?;
    }
}
