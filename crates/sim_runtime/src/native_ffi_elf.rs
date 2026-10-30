use sim_types::{CancellationError, CancellationToken};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

const MAXIMUM_ELF_TABLE_BYTES: usize = 64 * 1024 * 1024;
const CANCELLATION_CHUNK_BYTES: usize = 64 * 1024;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct NativeElfDynamicContract {
    pub(crate) symbols: BTreeSet<String>,
    pub(crate) symbol_identities: BTreeMap<String, Vec<NativeElfSymbolIdentity>>,
    pub(crate) needed: BTreeSet<String>,
    pub(crate) soname: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct NativeElfSymbolIdentity {
    pub(crate) binding: u8,
    pub(crate) kind: u8,
    pub(crate) visibility: u8,
    pub(crate) section_index: u16,
    pub(crate) value: u64,
    pub(crate) size: u64,
    pub(crate) executable: bool,
    pub(crate) version: Option<NativeElfSymbolVersion>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct NativeElfSymbolVersion {
    pub(crate) name: String,
    pub(crate) is_default: bool,
}

impl NativeElfDynamicContract {
    pub(crate) fn symbols(&self) -> &BTreeSet<String> {
        &self.symbols
    }

    pub(crate) fn symbol_identities(&self) -> &BTreeMap<String, Vec<NativeElfSymbolIdentity>> {
        &self.symbol_identities
    }

    pub(crate) fn needed(&self) -> &BTreeSet<String> {
        &self.needed
    }

    pub(crate) fn soname(&self) -> Option<&str> {
        self.soname.as_deref()
    }
}

#[derive(Debug, Error)]
pub(crate) enum NativeElfInspectionError {
    #[error("native ELF inspection was cancelled")]
    Cancelled(#[from] CancellationError),
    #[error("invalid native ELF object: {0}")]
    Invalid(String),
}

#[derive(Clone, Copy)]
struct ElfProgramHeader {
    kind: u32,
    flags: u32,
    file_offset: usize,
    virtual_address: u64,
    file_size: usize,
    memory_size: u64,
}

#[derive(Clone, Copy)]
struct ElfDynamicSymbolTable {
    section_index: usize,
    file_offset: usize,
    size: usize,
    entry_size: usize,
}

pub(crate) fn inspect_elf64_dynamic_contract(
    bytes: &[u8],
    expected_machine: u16,
    cancellation: &CancellationToken,
) -> Result<NativeElfDynamicContract, NativeElfInspectionError> {
    cancellation.check()?;
    inspect_elf64_dynamic_contract_inner(bytes, expected_machine, cancellation)
        .map_err(NativeElfInspectionError::Invalid)
}

fn inspect_elf64_dynamic_contract_inner(
    bytes: &[u8],
    expected_machine: u16,
    cancellation: &CancellationToken,
) -> Result<NativeElfDynamicContract, String> {
    check(cancellation)?;
    if bytes.get(0..4) != Some(b"\x7fELF")
        || bytes.get(4) != Some(&2)
        || bytes.get(5) != Some(&1)
        || bytes.get(6) != Some(&1)
    {
        return Err("expected a little-endian ELF64 object".to_owned());
    }
    if read_u16(bytes, 16)? != 3 || read_u16(bytes, 18)? != expected_machine {
        return Err(format!(
            "expected an ELF64 shared object for machine {expected_machine}"
        ));
    }
    if read_u32(bytes, 20)? != 1 {
        return Err("expected the current ELF header version".to_owned());
    }
    if read_u16(bytes, 52)? != 64 {
        return Err("expected the ELF64 header size".to_owned());
    }
    let program_offset = usize::try_from(read_u64(bytes, 32)?)
        .map_err(|_| "program table offset exceeds address space".to_owned())?;
    let program_entry_size = usize::from(read_u16(bytes, 54)?);
    let program_count = usize::from(read_u16(bytes, 56)?);
    if program_entry_size < 56 || program_count == 0 {
        return Err("ELF program table is absent or malformed".to_owned());
    }
    let program_table_size = program_entry_size
        .checked_mul(program_count)
        .ok_or_else(|| "ELF program table size overflowed".to_owned())?;
    checked_range(bytes, program_offset, program_table_size)?;
    let mut programs = Vec::with_capacity(program_count);
    for index in 0..program_count {
        check(cancellation)?;
        let offset = program_offset
            .checked_add(
                index
                    .checked_mul(program_entry_size)
                    .ok_or_else(|| "ELF program offset overflowed".to_owned())?,
            )
            .ok_or_else(|| "ELF program offset overflowed".to_owned())?;
        let header = checked_slice(bytes, offset, program_entry_size)?;
        let file_offset = usize::try_from(read_u64(header, 8)?)
            .map_err(|_| "program file offset exceeds address space".to_owned())?;
        let file_size = usize::try_from(read_u64(header, 32)?)
            .map_err(|_| "program file size exceeds address space".to_owned())?;
        let memory_size = read_u64(header, 40)?;
        if u64::try_from(file_size).unwrap_or(u64::MAX) > memory_size {
            return Err("program file size exceeds its memory size".to_owned());
        }
        checked_range(bytes, file_offset, file_size)?;
        programs.push(ElfProgramHeader {
            kind: read_u32(header, 0)?,
            flags: read_u32(header, 4)?,
            file_offset,
            virtual_address: read_u64(header, 16)?,
            file_size,
            memory_size,
        });
    }
    if !programs.iter().any(|program| program.kind == 1) {
        return Err("ELF object has no loadable segment".to_owned());
    }
    let dynamic_segments = programs
        .iter()
        .filter(|program| program.kind == 2)
        .copied()
        .collect::<Vec<_>>();
    if dynamic_segments.len() != 1 {
        return Err("ELF object must have exactly one PT_DYNAMIC segment".to_owned());
    }
    let dynamic_segment = dynamic_segments[0];
    if dynamic_segment.file_size == 0
        || dynamic_segment.file_size > MAXIMUM_ELF_TABLE_BYTES
        || dynamic_segment.file_size % 16 != 0
    {
        return Err("PT_DYNAMIC has an invalid bounded size".to_owned());
    }
    if !programs
        .iter()
        .filter(|program| program.kind == 1)
        .any(|program| {
            contains_virtual_range(
                *program,
                dynamic_segment.virtual_address,
                dynamic_segment.file_size,
            )
        })
    {
        return Err("PT_DYNAMIC is not contained in a loadable segment".to_owned());
    }
    let dynamic_table = checked_slice(
        bytes,
        dynamic_segment.file_offset,
        dynamic_segment.file_size,
    )?;
    let mut string_address = None;
    let mut string_size = None;
    let mut symbol_address = None;
    let mut symbol_entry_size = None;
    let mut needed_offsets = Vec::new();
    let mut soname_offset = None;
    let mut version_symbol_address = None;
    let mut version_definition_address = None;
    let mut version_definition_count = None;
    let mut found_null = false;
    for entry in dynamic_table.chunks_exact(16) {
        check(cancellation)?;
        let tag = read_i64(entry, 0)?;
        let value = read_u64(entry, 8)?;
        match tag {
            0 => {
                found_null = true;
                break;
            }
            1 => needed_offsets.push(value),
            5 => assign_once(&mut string_address, value, "DT_STRTAB")?,
            6 => assign_once(&mut symbol_address, value, "DT_SYMTAB")?,
            10 => assign_once(&mut string_size, value, "DT_STRSZ")?,
            11 => assign_once(&mut symbol_entry_size, value, "DT_SYMENT")?,
            14 => assign_once(&mut soname_offset, value, "DT_SONAME")?,
            0x6fff_fff0 => assign_once(&mut version_symbol_address, value, "DT_VERSYM")?,
            0x6fff_fffc => assign_once(&mut version_definition_address, value, "DT_VERDEF")?,
            0x6fff_fffd => assign_once(&mut version_definition_count, value, "DT_VERDEFNUM")?,
            15 | 29 => {
                return Err(
                    "ELF RPATH and RUNPATH are forbidden for certified libraries".to_owned(),
                );
            }
            _ => {}
        }
    }
    if !found_null {
        return Err("PT_DYNAMIC has no terminating DT_NULL entry".to_owned());
    }
    let string_address = string_address.ok_or_else(|| "PT_DYNAMIC has no DT_STRTAB".to_owned())?;
    let string_size =
        usize::try_from(string_size.ok_or_else(|| "PT_DYNAMIC has no DT_STRSZ".to_owned())?)
            .map_err(|_| "dynamic string-table size exceeds address space".to_owned())?;
    if string_size == 0 || string_size > MAXIMUM_ELF_TABLE_BYTES {
        return Err("dynamic string table has an invalid bounded size".to_owned());
    }
    let symbol_address = symbol_address.ok_or_else(|| "PT_DYNAMIC has no DT_SYMTAB".to_owned())?;
    let symbol_entry_size =
        usize::try_from(symbol_entry_size.ok_or_else(|| "PT_DYNAMIC has no DT_SYMENT".to_owned())?)
            .map_err(|_| "dynamic-symbol entry size exceeds address space".to_owned())?;
    if symbol_entry_size != 24 {
        return Err("DT_SYMENT does not match the ELF64 symbol size".to_owned());
    }
    let string_offset = virtual_to_file_offset(&programs, string_address, string_size)?;
    let strings = checked_slice(bytes, string_offset, string_size)?;
    let has_version_dynamic_contract = version_symbol_address.is_some()
        || version_definition_address.is_some()
        || version_definition_count.is_some();
    if has_version_dynamic_contract
        && (version_symbol_address.is_none()
            || version_definition_address.is_none()
            || version_definition_count.is_none())
    {
        return Err("PT_DYNAMIC has an incomplete GNU symbol-version contract".to_owned());
    }

    let section_offset = usize::try_from(read_u64(bytes, 40)?)
        .map_err(|_| "section table offset exceeds address space".to_owned())?;
    let section_entry_size = usize::from(read_u16(bytes, 58)?);
    let section_count = usize::from(read_u16(bytes, 60)?);
    if section_entry_size < 64 || section_count == 0 {
        return Err("ELF section table is absent or malformed".to_owned());
    }
    let section_table_size = section_entry_size
        .checked_mul(section_count)
        .ok_or_else(|| "ELF section table size overflowed".to_owned())?;
    checked_range(bytes, section_offset, section_table_size)?;
    let section = |index: usize| -> Result<&[u8], String> {
        if index >= section_count {
            return Err("ELF section link is out of bounds".to_owned());
        }
        let offset = section_offset
            .checked_add(
                index
                    .checked_mul(section_entry_size)
                    .ok_or_else(|| "ELF section offset overflowed".to_owned())?,
            )
            .ok_or_else(|| "ELF section offset overflowed".to_owned())?;
        checked_slice(bytes, offset, section_entry_size)
    };
    let mut dynamic_symbols = None;
    for index in 0..section_count {
        check(cancellation)?;
        let header = section(index)?;
        if read_u32(header, 4)? != 11 {
            continue;
        }
        let symbol_offset = usize::try_from(read_u64(header, 24)?)
            .map_err(|_| "dynamic-symbol offset exceeds address space".to_owned())?;
        let symbol_size = usize::try_from(read_u64(header, 32)?)
            .map_err(|_| "dynamic-symbol size exceeds address space".to_owned())?;
        let string_index = usize::try_from(read_u32(header, 40)?)
            .map_err(|_| "string-table index exceeds address space".to_owned())?;
        let section_symbol_entry_size = usize::try_from(read_u64(header, 56)?)
            .map_err(|_| "dynamic-symbol entry size exceeds address space".to_owned())?;
        if section_symbol_entry_size != symbol_entry_size
            || symbol_size == 0
            || symbol_size > MAXIMUM_ELF_TABLE_BYTES
            || symbol_size % section_symbol_entry_size != 0
        {
            return Err("dynamic-symbol table has an invalid entry size".to_owned());
        }
        let symbol_virtual_address = read_u64(header, 16)?;
        let string_header = section(string_index)?;
        if read_u32(string_header, 4)? != 3 || read_u64(string_header, 16)? != string_address {
            return Err("dynamic-symbol table does not link to a string table".to_owned());
        }
        let section_string_offset = usize::try_from(read_u64(string_header, 24)?)
            .map_err(|_| "string-table offset exceeds address space".to_owned())?;
        let section_string_size = usize::try_from(read_u64(string_header, 32)?)
            .map_err(|_| "string-table size exceeds address space".to_owned())?;
        if symbol_virtual_address != symbol_address
            || symbol_offset != virtual_to_file_offset(&programs, symbol_address, symbol_size)?
            || section_string_offset != string_offset
            || section_string_size != string_size
            || read_u64(header, 8)? & 2 == 0
        {
            continue;
        }
        if dynamic_symbols.is_some() {
            return Err(
                "multiple sections claim the loader-consumed dynamic-symbol table".to_owned(),
            );
        }
        dynamic_symbols = Some(ElfDynamicSymbolTable {
            section_index: index,
            file_offset: symbol_offset,
            size: symbol_size,
            entry_size: section_symbol_entry_size,
        });
    }
    let dynamic_symbols = dynamic_symbols.ok_or_else(|| {
        "ELF object has no section matching the loader-consumed dynamic-symbol table".to_owned()
    })?;
    let symbol_count = dynamic_symbols.size / dynamic_symbols.entry_size;
    let symbol_versions = inspect_symbol_versions(
        bytes,
        section_offset,
        section_entry_size,
        section_count,
        dynamic_symbols.section_index,
        symbol_count,
        strings,
        string_offset,
        string_size,
        version_symbol_address,
        version_definition_address,
        version_definition_count,
        &programs,
        cancellation,
    )?;
    let table = checked_slice(bytes, dynamic_symbols.file_offset, dynamic_symbols.size)?;
    let mut symbols = BTreeSet::new();
    let mut symbol_identities = BTreeMap::<String, Vec<NativeElfSymbolIdentity>>::new();
    for (symbol_index, entry) in table.chunks_exact(dynamic_symbols.entry_size).enumerate() {
        check(cancellation)?;
        let name_offset = usize::try_from(read_u32(entry, 0)?)
            .map_err(|_| "symbol name offset exceeds address space".to_owned())?;
        if name_offset == 0 {
            continue;
        }
        let name = dynamic_string(strings, name_offset, cancellation)?;
        if name.is_empty() {
            continue;
        }
        let section_index = read_u16(entry, 6)?;
        let value = read_u64(entry, 8)?;
        let size = read_u64(entry, 16)?;
        let executable = value != 0
            && programs
                .iter()
                .filter(|program| program.kind == 1 && program.flags & 1 != 0)
                .any(|program| {
                    contains_virtual_range(
                        *program,
                        value,
                        usize::try_from(size.max(1)).unwrap_or(usize::MAX),
                    )
                });
        let version = symbol_versions
            .as_ref()
            .and_then(|versions| versions.get(symbol_index))
            .cloned()
            .flatten();
        symbol_identities
            .entry(name.to_owned())
            .or_default()
            .push(NativeElfSymbolIdentity {
                binding: entry[4] >> 4,
                kind: entry[4] & 0x0f,
                visibility: entry[5] & 0x03,
                section_index,
                value,
                size,
                executable,
                version,
            });
        if section_index != 0 {
            symbols.insert(name.to_owned());
        }
    }
    let mut needed = BTreeSet::new();
    for offset in needed_offsets {
        let offset = usize::try_from(offset)
            .map_err(|_| "DT_NEEDED string offset exceeds address space".to_owned())?;
        let dependency = dynamic_string(strings, offset, cancellation)?.to_owned();
        if !needed.insert(dependency.clone()) {
            return Err(format!(
                "PT_DYNAMIC contains duplicate DT_NEEDED entry {dependency}"
            ));
        }
    }
    let soname = soname_offset
        .map(|offset| {
            usize::try_from(offset)
                .map_err(|_| "DT_SONAME string offset exceeds address space".to_owned())
                .and_then(|offset| {
                    dynamic_string(strings, offset, cancellation).map(ToOwned::to_owned)
                })
        })
        .transpose()?;
    check(cancellation)?;
    Ok(NativeElfDynamicContract {
        symbols,
        symbol_identities,
        needed,
        soname,
    })
}

#[allow(clippy::too_many_arguments)]
fn inspect_symbol_versions(
    bytes: &[u8],
    section_offset: usize,
    section_entry_size: usize,
    section_count: usize,
    symbol_section_index: usize,
    symbol_count: usize,
    strings: &[u8],
    string_offset: usize,
    string_size: usize,
    version_symbol_address: Option<u64>,
    version_definition_address: Option<u64>,
    version_definition_count: Option<u64>,
    programs: &[ElfProgramHeader],
    cancellation: &CancellationToken,
) -> Result<Option<Vec<Option<NativeElfSymbolVersion>>>, String> {
    let mut version_symbol_section = None;
    let mut version_definition_section = None;
    for index in 0..section_count {
        check(cancellation)?;
        let header = section_header(
            bytes,
            section_offset,
            section_entry_size,
            section_count,
            index,
        )?;
        match read_u32(header, 4)? {
            0x6fff_ffff
                if usize::try_from(read_u32(header, 40)?).ok() == Some(symbol_section_index) =>
            {
                if version_symbol_section.replace(index).is_some() {
                    return Err(
                        "multiple GNU symbol-version tables target the dynamic symbols".to_owned(),
                    );
                }
            }
            0x6fff_fffd => {
                let linked_string_index = usize::try_from(read_u32(header, 40)?).map_err(|_| {
                    "version-definition string-table index exceeds address space".to_owned()
                })?;
                let linked = section_header(
                    bytes,
                    section_offset,
                    section_entry_size,
                    section_count,
                    linked_string_index,
                )?;
                if read_u32(linked, 4)? == 3
                    && usize::try_from(read_u64(linked, 24)?).ok() == Some(string_offset)
                    && usize::try_from(read_u64(linked, 32)?).ok() == Some(string_size)
                {
                    if version_definition_section.replace(index).is_some() {
                        return Err(
                            "multiple GNU version-definition tables target the dynamic strings"
                                .to_owned(),
                        );
                    }
                }
            }
            _ => {}
        }
    }
    let has_sections = version_symbol_section.is_some() || version_definition_section.is_some();
    let has_dynamic = version_symbol_address.is_some()
        || version_definition_address.is_some()
        || version_definition_count.is_some();
    if !has_sections && !has_dynamic {
        return Ok(None);
    }
    let version_symbol_section = version_symbol_section
        .ok_or_else(|| "ELF GNU version contract has no symbol-version section".to_owned())?;
    let version_definition_section = version_definition_section
        .ok_or_else(|| "ELF GNU version contract has no version-definition section".to_owned())?;
    let expected_symbol_address =
        version_symbol_address.ok_or_else(|| "PT_DYNAMIC has no DT_VERSYM".to_owned())?;
    let expected_definition_address =
        version_definition_address.ok_or_else(|| "PT_DYNAMIC has no DT_VERDEF".to_owned())?;
    let expected_definition_count = usize::try_from(
        version_definition_count.ok_or_else(|| "PT_DYNAMIC has no DT_VERDEFNUM".to_owned())?,
    )
    .map_err(|_| "DT_VERDEFNUM exceeds address space".to_owned())?;
    if expected_definition_count == 0 || expected_definition_count > 4096 {
        return Err("DT_VERDEFNUM has an invalid bounded count".to_owned());
    }

    let symbol_header = section_header(
        bytes,
        section_offset,
        section_entry_size,
        section_count,
        version_symbol_section,
    )?;
    let symbol_address = read_u64(symbol_header, 16)?;
    let symbol_offset = usize::try_from(read_u64(symbol_header, 24)?)
        .map_err(|_| "GNU symbol-version offset exceeds address space".to_owned())?;
    let symbol_size = usize::try_from(read_u64(symbol_header, 32)?)
        .map_err(|_| "GNU symbol-version size exceeds address space".to_owned())?;
    let symbol_entry_size = usize::try_from(read_u64(symbol_header, 56)?)
        .map_err(|_| "GNU symbol-version entry size exceeds address space".to_owned())?;
    let expected_symbol_size = symbol_count
        .checked_mul(2)
        .ok_or_else(|| "GNU symbol-version size overflowed".to_owned())?;
    if symbol_address != expected_symbol_address
        || symbol_offset != virtual_to_file_offset(programs, symbol_address, symbol_size)?
        || symbol_size != expected_symbol_size
        || symbol_entry_size != 2
        || read_u64(symbol_header, 8)? & 2 == 0
    {
        return Err("GNU symbol-version table does not match the loader contract".to_owned());
    }

    let definition_header = section_header(
        bytes,
        section_offset,
        section_entry_size,
        section_count,
        version_definition_section,
    )?;
    let definition_address = read_u64(definition_header, 16)?;
    let definition_offset = usize::try_from(read_u64(definition_header, 24)?)
        .map_err(|_| "GNU version-definition offset exceeds address space".to_owned())?;
    let definition_size = usize::try_from(read_u64(definition_header, 32)?)
        .map_err(|_| "GNU version-definition size exceeds address space".to_owned())?;
    if definition_size == 0
        || definition_size > MAXIMUM_ELF_TABLE_BYTES
        || definition_address != expected_definition_address
        || definition_offset
            != virtual_to_file_offset(programs, definition_address, definition_size)?
        || read_u64(definition_header, 8)? & 2 == 0
    {
        return Err("GNU version-definition table does not match the loader contract".to_owned());
    }
    let definitions = checked_slice(bytes, definition_offset, definition_size)?;
    let mut definition_names = BTreeMap::new();
    let mut definition_offset_in_section = 0_usize;
    for definition_index in 0..expected_definition_count {
        check(cancellation)?;
        let definition = checked_slice(definitions, definition_offset_in_section, 20)?;
        if read_u16(definition, 0)? != 1 || read_u16(definition, 6)? != 1 {
            return Err("GNU version definition has an unsupported layout".to_owned());
        }
        let version_flags = read_u16(definition, 2)?;
        let version_index = read_u16(definition, 4)? & 0x7fff;
        if version_index == 0
            || (version_index == 1 && version_flags != 1)
            || (version_index > 1 && version_flags != 0)
        {
            return Err("GNU version definition has an invalid base identity".to_owned());
        }
        let auxiliary_offset = usize::try_from(read_u32(definition, 12)?)
            .map_err(|_| "GNU version auxiliary offset exceeds address space".to_owned())?;
        let auxiliary_position = definition_offset_in_section
            .checked_add(auxiliary_offset)
            .ok_or_else(|| "GNU version auxiliary offset overflowed".to_owned())?;
        let auxiliary = checked_slice(definitions, auxiliary_position, 8)?;
        if read_u32(auxiliary, 4)? != 0 {
            return Err("GNU version definition has an unexpected auxiliary chain".to_owned());
        }
        let name_offset = usize::try_from(read_u32(auxiliary, 0)?)
            .map_err(|_| "GNU version name offset exceeds address space".to_owned())?;
        let name = dynamic_string(strings, name_offset, cancellation)?;
        if name.is_empty()
            || definition_names
                .insert(version_index, name.to_owned())
                .is_some()
        {
            return Err(
                "GNU version definitions contain an empty or duplicate identity".to_owned(),
            );
        }
        let next = usize::try_from(read_u32(definition, 16)?)
            .map_err(|_| "GNU version-definition next offset exceeds address space".to_owned())?;
        if definition_index + 1 == expected_definition_count {
            if next != 0 {
                return Err("GNU version-definition chain exceeds DT_VERDEFNUM".to_owned());
            }
        } else if next < 20 {
            return Err("GNU version-definition chain ended early or cycled".to_owned());
        } else {
            definition_offset_in_section = definition_offset_in_section
                .checked_add(next)
                .ok_or_else(|| "GNU version-definition chain overflowed".to_owned())?;
        }
    }

    let version_symbols = checked_slice(bytes, symbol_offset, symbol_size)?;
    let mut versions = Vec::with_capacity(symbol_count);
    for entry in version_symbols.chunks_exact(2) {
        check(cancellation)?;
        let raw = read_u16(entry, 0)?;
        let index = raw & 0x7fff;
        let version = if index <= 1 {
            None
        } else {
            Some(NativeElfSymbolVersion {
                name: definition_names
                    .get(&index)
                    .ok_or_else(|| {
                        "GNU symbol version references an undefined identity".to_owned()
                    })?
                    .clone(),
                is_default: raw & 0x8000 == 0,
            })
        };
        versions.push(version);
    }
    Ok(Some(versions))
}

fn check(cancellation: &CancellationToken) -> Result<(), String> {
    cancellation.check().map_err(|error| error.to_string())
}

fn assign_once(slot: &mut Option<u64>, value: u64, name: &str) -> Result<(), String> {
    if slot.replace(value).is_some() {
        Err(format!("PT_DYNAMIC contains duplicate {name}"))
    } else {
        Ok(())
    }
}

fn contains_virtual_range(program: ElfProgramHeader, address: u64, length: usize) -> bool {
    let Ok(length) = u64::try_from(length) else {
        return false;
    };
    let Some(range_end) = address.checked_add(length) else {
        return false;
    };
    let Some(program_end) = program.virtual_address.checked_add(program.memory_size) else {
        return false;
    };
    address >= program.virtual_address && range_end <= program_end
}

fn virtual_to_file_offset(
    programs: &[ElfProgramHeader],
    address: u64,
    length: usize,
) -> Result<usize, String> {
    let mut resolved = None;
    for program in programs.iter().filter(|program| program.kind == 1) {
        if !contains_virtual_range(*program, address, length) {
            continue;
        }
        let delta = usize::try_from(address - program.virtual_address)
            .map_err(|_| "virtual-address delta exceeds address space".to_owned())?;
        let end = delta
            .checked_add(length)
            .ok_or_else(|| "virtual-address range overflowed".to_owned())?;
        if end > program.file_size {
            continue;
        }
        let offset = program
            .file_offset
            .checked_add(delta)
            .ok_or_else(|| "mapped file offset overflowed".to_owned())?;
        if resolved.is_some_and(|prior| prior != offset) {
            return Err("virtual address maps ambiguously to multiple file offsets".to_owned());
        }
        resolved = Some(offset);
    }
    resolved.ok_or_else(|| "dynamic virtual address is not file-backed by PT_LOAD".to_owned())
}

fn dynamic_string<'a>(
    strings: &'a [u8],
    offset: usize,
    cancellation: &CancellationToken,
) -> Result<&'a str, String> {
    let suffix = strings
        .get(offset..)
        .ok_or_else(|| "dynamic string is outside the string table".to_owned())?;
    for (chunk_index, chunk) in suffix.chunks(CANCELLATION_CHUNK_BYTES).enumerate() {
        check(cancellation)?;
        if let Some(position) = chunk.iter().position(|byte| *byte == 0) {
            let end = chunk_index
                .checked_mul(CANCELLATION_CHUNK_BYTES)
                .and_then(|start| start.checked_add(position))
                .ok_or_else(|| "dynamic string length overflowed".to_owned())?;
            return std::str::from_utf8(&suffix[..end])
                .map_err(|_| "dynamic string is not UTF-8".to_owned());
        }
    }
    Err("dynamic string is not NUL terminated".to_owned())
}

fn checked_range(bytes: &[u8], offset: usize, length: usize) -> Result<(), String> {
    offset
        .checked_add(length)
        .filter(|end| *end <= bytes.len())
        .map(|_| ())
        .ok_or_else(|| "ELF range is out of bounds".to_owned())
}

fn checked_slice(bytes: &[u8], offset: usize, length: usize) -> Result<&[u8], String> {
    checked_range(bytes, offset, length)?;
    Ok(&bytes[offset..offset + length])
}

fn section_header(
    bytes: &[u8],
    section_offset: usize,
    section_entry_size: usize,
    section_count: usize,
    index: usize,
) -> Result<&[u8], String> {
    if index >= section_count {
        return Err("ELF section link is out of bounds".to_owned());
    }
    let offset = section_offset
        .checked_add(
            index
                .checked_mul(section_entry_size)
                .ok_or_else(|| "ELF section offset overflowed".to_owned())?,
        )
        .ok_or_else(|| "ELF section offset overflowed".to_owned())?;
    checked_slice(bytes, offset, section_entry_size)
}

fn read_u16(bytes: &[u8], offset: usize) -> Result<u16, String> {
    let value = checked_slice(bytes, offset, 2)?;
    Ok(u16::from_le_bytes([value[0], value[1]]))
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, String> {
    let value = checked_slice(bytes, offset, 4)?;
    Ok(u32::from_le_bytes([value[0], value[1], value[2], value[3]]))
}

fn read_u64(bytes: &[u8], offset: usize) -> Result<u64, String> {
    let value = checked_slice(bytes, offset, 8)?;
    Ok(u64::from_le_bytes([
        value[0], value[1], value[2], value[3], value[4], value[5], value[6], value[7],
    ]))
}

fn read_i64(bytes: &[u8], offset: usize) -> Result<i64, String> {
    let value = checked_slice(bytes, offset, 8)?;
    Ok(i64::from_le_bytes([
        value[0], value[1], value[2], value[3], value[4], value[5], value[6], value[7],
    ]))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    fn write_u16(bytes: &mut [u8], offset: usize, value: u16) {
        bytes[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
    }

    fn write_u32(bytes: &mut [u8], offset: usize, value: u32) {
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }

    fn write_u64(bytes: &mut [u8], offset: usize, value: u64) {
        bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
    }

    fn section_data_offset(bytes: &[u8], kind: u32) -> Option<usize> {
        let section_offset = usize::try_from(read_u64(bytes, 40).ok()?).ok()?;
        let section_entry_size = usize::from(read_u16(bytes, 58).ok()?);
        let section_count = usize::from(read_u16(bytes, 60).ok()?);
        (0..section_count).find_map(|index| {
            let header = section_header(
                bytes,
                section_offset,
                section_entry_size,
                section_count,
                index,
            )
            .ok()?;
            (read_u32(header, 4).ok()? == kind)
                .then(|| usize::try_from(read_u64(header, 24).ok()?).ok())
                .flatten()
        })
    }

    pub(crate) fn fixture(
        machine: u16,
        symbols: &BTreeSet<String>,
        needed: &[&str],
        runpath: Option<&str>,
        soname: &str,
    ) -> Vec<u8> {
        let mut strings = vec![0_u8];
        let mut names = Vec::new();
        for symbol in symbols {
            names.push(u32::try_from(strings.len()).unwrap_or_default());
            strings.extend_from_slice(symbol.as_bytes());
            strings.push(0);
        }
        let mut needed_offsets = Vec::new();
        for dependency in needed {
            needed_offsets.push(u64::try_from(strings.len()).unwrap_or_default());
            strings.extend_from_slice(dependency.as_bytes());
            strings.push(0);
        }
        let runpath_offset = runpath.map(|path| {
            let offset = u64::try_from(strings.len()).unwrap_or_default();
            strings.extend_from_slice(path.as_bytes());
            strings.push(0);
            offset
        });
        let soname_offset = u64::try_from(strings.len()).unwrap_or_default();
        strings.extend_from_slice(soname.as_bytes());
        strings.push(0);
        let version_namespace = match soname {
            "libavcodec.so.61" => "LIBAVCODEC_61",
            "libavformat.so.61" => "LIBAVFORMAT_61",
            "libavutil.so.59" => "LIBAVUTIL_59",
            "libswresample.so.5" => "LIBSWRESAMPLE_5",
            "libswscale.so.8" => "LIBSWSCALE_8",
            _ => "ZED_FIXTURE_1",
        };
        let version_name_offset = u32::try_from(strings.len()).unwrap_or_default();
        strings.extend_from_slice(version_namespace.as_bytes());
        strings.push(0);
        let program_offset = 64_usize;
        let program_entry_size = 56_usize;
        let program_count = 2_usize;
        let string_offset = 192_usize;
        let symbol_offset = (string_offset + strings.len() + 7) & !7;
        let symbol_size = (symbols.len() + 1) * 24;
        let version_symbol_offset = symbol_offset + symbol_size;
        let version_symbol_size = (symbols.len() + 1) * 2;
        let version_definition_offset = (version_symbol_offset + version_symbol_size + 3) & !3;
        let version_definition_size = 56_usize;
        let dynamic_offset = (version_definition_offset + version_definition_size + 7) & !7;
        let dynamic_entries = 9 + needed_offsets.len() + usize::from(runpath_offset.is_some());
        let dynamic_size = dynamic_entries * 16;
        let section_offset = (dynamic_offset + dynamic_size + 7) & !7;
        let mut bytes = vec![0_u8; section_offset + 6 * 64];
        bytes[..4].copy_from_slice(b"\x7fELF");
        bytes[4] = 2;
        bytes[5] = 1;
        bytes[6] = 1;
        write_u16(&mut bytes, 16, 3);
        write_u16(&mut bytes, 18, machine);
        write_u32(&mut bytes, 20, 1);
        write_u64(
            &mut bytes,
            32,
            u64::try_from(program_offset).unwrap_or_default(),
        );
        write_u64(
            &mut bytes,
            40,
            u64::try_from(section_offset).unwrap_or_default(),
        );
        write_u16(&mut bytes, 52, 64);
        write_u16(
            &mut bytes,
            54,
            u16::try_from(program_entry_size).unwrap_or_default(),
        );
        write_u16(
            &mut bytes,
            56,
            u16::try_from(program_count).unwrap_or_default(),
        );
        write_u16(&mut bytes, 58, 64);
        write_u16(&mut bytes, 60, 6);
        let file_length = u64::try_from(bytes.len()).unwrap_or_default();
        write_u32(&mut bytes, program_offset, 1);
        write_u32(&mut bytes, program_offset + 4, 5);
        write_u64(&mut bytes, program_offset + 32, file_length);
        write_u64(&mut bytes, program_offset + 40, file_length);
        write_u64(&mut bytes, program_offset + 48, 8);
        let dynamic_program = program_offset + program_entry_size;
        write_u32(&mut bytes, dynamic_program, 2);
        write_u32(&mut bytes, dynamic_program + 4, 4);
        write_u64(
            &mut bytes,
            dynamic_program + 8,
            u64::try_from(dynamic_offset).unwrap_or_default(),
        );
        write_u64(
            &mut bytes,
            dynamic_program + 16,
            u64::try_from(dynamic_offset).unwrap_or_default(),
        );
        write_u64(
            &mut bytes,
            dynamic_program + 32,
            u64::try_from(dynamic_size).unwrap_or_default(),
        );
        write_u64(
            &mut bytes,
            dynamic_program + 40,
            u64::try_from(dynamic_size).unwrap_or_default(),
        );
        write_u64(&mut bytes, dynamic_program + 48, 8);
        bytes[string_offset..string_offset + strings.len()].copy_from_slice(&strings);
        for (index, name_offset) in names.into_iter().enumerate() {
            let entry = symbol_offset + (index + 1) * 24;
            write_u32(&mut bytes, entry, name_offset);
            bytes[entry + 4] = 0x12;
            write_u16(&mut bytes, entry + 6, 1);
            write_u64(
                &mut bytes,
                entry + 8,
                u64::try_from(entry).unwrap_or_default(),
            );
            write_u64(&mut bytes, entry + 16, 1);
            write_u16(&mut bytes, version_symbol_offset + (index + 1) * 2, 2);
        }
        write_u16(&mut bytes, version_definition_offset, 1);
        write_u16(&mut bytes, version_definition_offset + 2, 1);
        write_u16(&mut bytes, version_definition_offset + 4, 1);
        write_u16(&mut bytes, version_definition_offset + 6, 1);
        write_u32(&mut bytes, version_definition_offset + 12, 20);
        write_u32(&mut bytes, version_definition_offset + 16, 28);
        write_u32(
            &mut bytes,
            version_definition_offset + 20,
            u32::try_from(soname_offset).unwrap_or_default(),
        );
        let callable_version_definition = version_definition_offset + 28;
        write_u16(&mut bytes, callable_version_definition, 1);
        write_u16(&mut bytes, callable_version_definition + 4, 2);
        write_u16(&mut bytes, callable_version_definition + 6, 1);
        write_u32(&mut bytes, callable_version_definition + 12, 20);
        write_u32(
            &mut bytes,
            callable_version_definition + 20,
            version_name_offset,
        );
        let strings_header = section_offset + 64;
        write_u32(&mut bytes, strings_header + 4, 3);
        write_u64(
            &mut bytes,
            strings_header + 16,
            u64::try_from(string_offset).unwrap_or_default(),
        );
        write_u64(
            &mut bytes,
            strings_header + 24,
            u64::try_from(string_offset).unwrap_or_default(),
        );
        write_u64(
            &mut bytes,
            strings_header + 32,
            u64::try_from(strings.len()).unwrap_or_default(),
        );
        let symbols_header = section_offset + 128;
        write_u32(&mut bytes, symbols_header + 4, 11);
        write_u64(&mut bytes, symbols_header + 8, 2);
        write_u64(
            &mut bytes,
            symbols_header + 16,
            u64::try_from(symbol_offset).unwrap_or_default(),
        );
        write_u64(
            &mut bytes,
            symbols_header + 24,
            u64::try_from(symbol_offset).unwrap_or_default(),
        );
        write_u64(
            &mut bytes,
            symbols_header + 32,
            u64::try_from(symbol_size).unwrap_or_default(),
        );
        write_u32(&mut bytes, symbols_header + 40, 1);
        write_u64(&mut bytes, symbols_header + 56, 24);
        let mut dynamic_index = 0_usize;
        for (tag, value) in [
            (5, u64::try_from(string_offset).unwrap_or_default()),
            (10, u64::try_from(strings.len()).unwrap_or_default()),
            (6, u64::try_from(symbol_offset).unwrap_or_default()),
            (11, 24),
            (14, soname_offset),
            (
                0x6fff_fff0,
                u64::try_from(version_symbol_offset).unwrap_or_default(),
            ),
            (
                0x6fff_fffc,
                u64::try_from(version_definition_offset).unwrap_or_default(),
            ),
            (0x6fff_fffd, 2),
        ] {
            let entry = dynamic_offset + dynamic_index * 16;
            write_u64(&mut bytes, entry, tag);
            write_u64(&mut bytes, entry + 8, value);
            dynamic_index += 1;
        }
        for offset in needed_offsets {
            let entry = dynamic_offset + dynamic_index * 16;
            write_u64(&mut bytes, entry, 1);
            write_u64(&mut bytes, entry + 8, offset);
            dynamic_index += 1;
        }
        if let Some(offset) = runpath_offset {
            let entry = dynamic_offset + dynamic_index * 16;
            write_u64(&mut bytes, entry, 29);
            write_u64(&mut bytes, entry + 8, offset);
            dynamic_index += 1;
        }
        write_u64(&mut bytes, dynamic_offset + dynamic_index * 16, 0);
        let dynamic_header = section_offset + 192;
        write_u32(&mut bytes, dynamic_header + 4, 6);
        write_u64(
            &mut bytes,
            dynamic_header + 16,
            u64::try_from(dynamic_offset).unwrap_or_default(),
        );
        write_u64(
            &mut bytes,
            dynamic_header + 24,
            u64::try_from(dynamic_offset).unwrap_or_default(),
        );
        write_u64(
            &mut bytes,
            dynamic_header + 32,
            u64::try_from(dynamic_size).unwrap_or_default(),
        );
        write_u32(&mut bytes, dynamic_header + 40, 1);
        write_u64(&mut bytes, dynamic_header + 56, 16);
        let version_symbol_header = section_offset + 256;
        write_u32(&mut bytes, version_symbol_header + 4, 0x6fff_ffff);
        write_u64(&mut bytes, version_symbol_header + 8, 2);
        write_u64(
            &mut bytes,
            version_symbol_header + 16,
            u64::try_from(version_symbol_offset).unwrap_or_default(),
        );
        write_u64(
            &mut bytes,
            version_symbol_header + 24,
            u64::try_from(version_symbol_offset).unwrap_or_default(),
        );
        write_u64(
            &mut bytes,
            version_symbol_header + 32,
            u64::try_from(version_symbol_size).unwrap_or_default(),
        );
        write_u32(&mut bytes, version_symbol_header + 40, 2);
        write_u64(&mut bytes, version_symbol_header + 56, 2);
        let version_definition_header = section_offset + 320;
        write_u32(&mut bytes, version_definition_header + 4, 0x6fff_fffd);
        write_u64(&mut bytes, version_definition_header + 8, 2);
        write_u64(
            &mut bytes,
            version_definition_header + 16,
            u64::try_from(version_definition_offset).unwrap_or_default(),
        );
        write_u64(
            &mut bytes,
            version_definition_header + 24,
            u64::try_from(version_definition_offset).unwrap_or_default(),
        );
        write_u64(
            &mut bytes,
            version_definition_header + 32,
            u64::try_from(version_definition_size).unwrap_or_default(),
        );
        write_u32(&mut bytes, version_definition_header + 40, 1);
        write_u32(&mut bytes, version_definition_header + 44, 2);
        bytes
    }

    #[test]
    fn elf_inspection_binds_machine_soname_symbols_and_dependencies() {
        let symbols = BTreeSet::from(["avcodec_open2".to_owned(), "avcodec_send_frame".to_owned()]);
        let bytes = fixture(
            62,
            &symbols,
            &["libavutil.so.59", "libc.so.6"],
            None,
            "libavcodec.so.61",
        );
        let contract = inspect_elf64_dynamic_contract(&bytes, 62, &CancellationToken::default())
            .expect("synthetic ELF should be admitted");
        assert_eq!(contract.soname(), Some("libavcodec.so.61"));
        assert_eq!(contract.symbols(), &symbols);
        for identities in contract.symbol_identities().values() {
            assert_eq!(identities.len(), 1);
            let Some(identity) = identities.first() else {
                continue;
            };
            assert_eq!(identity.binding, 1);
            assert_eq!(identity.kind, 2);
            assert_eq!(identity.visibility, 0);
            assert!(identity.executable);
            assert_eq!(
                identity.version.as_ref(),
                Some(&NativeElfSymbolVersion {
                    name: "LIBAVCODEC_61".to_owned(),
                    is_default: true,
                })
            );
        }
        assert_eq!(
            contract.needed(),
            &BTreeSet::from(["libavutil.so.59".to_owned(), "libc.so.6".to_owned()])
        );
        assert!(
            inspect_elf64_dynamic_contract(&bytes, 183, &CancellationToken::default()).is_err()
        );
    }

    #[test]
    fn elf_inspection_rejects_invalid_elf64_header_contract() {
        let bytes = fixture(
            62,
            &BTreeSet::from(["avcodec_open2".to_owned()]),
            &[],
            None,
            "libavcodec.so.61",
        );

        for invalid_version in [0, 2] {
            let mut invalid = bytes.clone();
            write_u32(&mut invalid, 20, invalid_version);
            let error = inspect_elf64_dynamic_contract(&invalid, 62, &CancellationToken::default())
                .expect_err("a non-current ELF header version must fail closed");
            assert!(error.to_string().contains("ELF header version"));
        }

        for invalid_size in [0, 63] {
            let mut invalid = bytes.clone();
            write_u16(&mut invalid, 52, invalid_size);
            let error = inspect_elf64_dynamic_contract(&invalid, 62, &CancellationToken::default())
                .expect_err("a non-ELF64 header size must fail closed");
            assert!(error.to_string().contains("ELF64 header size"));
        }
    }

    #[test]
    fn elf_inspection_rejects_duplicate_needed_entries() {
        let bytes = fixture(
            62,
            &BTreeSet::from(["avcodec_open2".to_owned()]),
            &["libc.so.6", "libc.so.6"],
            None,
            "libavcodec.so.61",
        );
        let error = inspect_elf64_dynamic_contract(&bytes, 62, &CancellationToken::default())
            .expect_err("duplicate DT_NEEDED entries must fail closed");
        assert!(error.to_string().contains("duplicate DT_NEEDED"));
    }

    #[test]
    fn elf_inspection_preserves_non_callable_and_hidden_version_metadata() {
        let symbols = BTreeSet::from(["avcodec_version".to_owned()]);
        let bytes = fixture(62, &symbols, &[], None, "libavcodec.so.61");

        let mut non_callable = bytes.clone();
        let symbol_entry = section_data_offset(&non_callable, 11)
            .and_then(|offset| offset.checked_add(24))
            .expect("fixture dynamic symbol entry must exist");
        non_callable[symbol_entry + 4] = 0x1a;
        let contract =
            inspect_elf64_dynamic_contract(&non_callable, 62, &CancellationToken::default())
                .expect("inspector must preserve IFUNC metadata for the trust owner");
        let identity = contract
            .symbol_identities()
            .get("avcodec_version")
            .and_then(|identities| identities.first())
            .expect("fixture symbol identity must exist");
        assert_eq!(identity.kind, 10);

        let mut hidden_version = bytes;
        let version_entry = section_data_offset(&hidden_version, 0x6fff_ffff)
            .and_then(|offset| offset.checked_add(2))
            .expect("fixture symbol-version entry must exist");
        write_u16(&mut hidden_version, version_entry, 0x8002);
        let contract =
            inspect_elf64_dynamic_contract(&hidden_version, 62, &CancellationToken::default())
                .expect("inspector must preserve hidden GNU version metadata");
        let version = contract
            .symbol_identities()
            .get("avcodec_version")
            .and_then(|identities| identities.first())
            .and_then(|identity| identity.version.as_ref())
            .expect("fixture symbol version must exist");
        assert!(!version.is_default);
    }

    #[test]
    fn elf_inspection_rejects_embedded_search_paths_and_cancellation() {
        let bytes = fixture(
            183,
            &BTreeSet::new(),
            &[],
            Some("/ambient"),
            "libavutil.so.59",
        );
        assert!(
            inspect_elf64_dynamic_contract(&bytes, 183, &CancellationToken::default()).is_err()
        );
        let cancellation = CancellationToken::default();
        cancellation.cancel();
        assert!(matches!(
            inspect_elf64_dynamic_contract(&bytes, 183, &cancellation),
            Err(NativeElfInspectionError::Cancelled(_))
        ));
    }
}
