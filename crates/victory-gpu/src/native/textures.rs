//! Title textures on the native host: the console's tiled layouts (`victory_formats::texture`)
//! detiled from title memory into backend textures.
//!
//! A title texture lives in title memory in the console's swizzled layout; the backend's texture is
//! Apple's own. Each texture the host samples is detiled into a linear staging buffer and uploaded,
//! mip by mip and slice by slice, when the memory's content changed since the last time (a hash of
//! the memory, taken when work is submitted). Render targets and depth targets are the GPU's: they
//! start empty and are not read from memory. With `VICTORY_TEXTURE_BACK=1`, textures compute writes
//! are retiled back into title memory ([`Back`]).
//!
//! shortcut: the detile runs on the CPU (`Layout::detile`, the code the D3D12 host uses too) and
//! a thread hashes the memory of every sampled texture round after round (see `Watch`); a compute detile that reads the
//! pool buffer directly and a per-page change test (what `watch.rs` gives the D3D12 host) are the
//! next step. A change detiles the whole texture.
//!
//! A submission's textures are read, hashed and detiled on a few threads without the host's lock
//! ([`prepare`]); only the probes that pick them ([`Title::sync_jobs`]) and the uploads
//! ([`Title::sync_apply`]) hold it. Under the lock that work stopped every thread of a title that
//! streams (hundreds of MiB of textures a second), for most of each second.

use super::descriptors::format;
use super::host::NativeHost;
use super::title::Title;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use victory_backend::resources::*;
use victory_formats::texture::{Dimension, Layout};

/// `D3D12_RESOURCE_FLAGS`.
const RENDER_TARGET: u32 = 1;
const DEPTH_STENCIL: u32 = 2;
const UNORDERED_ACCESS: u32 = 4;

/// A title texture as the title made it: its description and where its memory is.
#[derive(Clone)]
pub struct TextureInfo {
    pub dimension: Dimension,
    /// DXGI format number, the console's (before `pc_format`).
    pub format: u32,
    pub width: u32,
    pub height: u32,
    /// Array length, or depth for 3D.
    pub depth_or_layers: u32,
    pub flags: u32,
    /// The layout of plane 0 (depth, for depth and stencil).
    pub layout: Layout,
    pub memory: *const u8,
    pub size: usize,
}

impl TextureInfo {
    pub fn render_target(&self) -> bool {
        self.flags & (RENDER_TARGET | DEPTH_STENCIL) != 0
    }

    /// The backend's description of it; `None` for formats or shapes the host does not know.
    pub fn desc(&self) -> Option<TextureDesc> {
        let depth = self.flags & DEPTH_STENCIL != 0;
        let mut format = format(self.format)?;
        // A typeless 32-bit float is a depth target when the texture is one.
        if depth && format == Format::R32Float {
            format = Format::D32Float;
        }
        if depth && format == Format::R16Unorm {
            format = Format::D16Unorm;
        }
        let (dim, d) = match self.dimension {
            Dimension::Tex3d => (TexDim::D3, self.depth_or_layers as usize),
            _ if self.depth_or_layers > 1 => (TexDim::D2Array, self.depth_or_layers as usize),
            _ => (TexDim::D2, 1),
        };
        Some(TextureDesc {
            dim,
            format,
            width: self.width as usize,
            height: self.height.max(1) as usize,
            depth_or_layers: d,
            mips: self.layout.mips.len().max(1),
            render_target: self.render_target(),
            unordered_access: self.flags & UNORDERED_ACCESS != 0,
        })
    }
}

/// Uploads and change probes must see the same bytes, including across holes in title memory.
/// False if they could not be read at once (a hole: zeros there).
fn read_texture_bytes(at: *const u8, out: &mut [u8]) -> bool {
    read_texture_bytes_known(at, out, false)
}

/// [`read_texture_bytes`] when `known_partial` says an earlier whole read of this texture met an
/// unmapped or protected range: skips the initial whole-range kernel read that would copy the
/// readable prefix and then fail at the hole, and returns `true` again once every page is readable.
fn read_texture_bytes_known(at: *const u8, out: &mut [u8], known_partial: bool) -> bool {
    #[cfg(target_os = "macos")]
    return read_texture_chunk(
        at,
        out,
        known_partial,
        super::safe_read,
        victory_platform::mem::next_region,
    );
    #[cfg(not(target_os = "macos"))]
    {
        let _ = known_partial;
        let read = super::safe_read(at, out);
        if !read {
            out.fill(0);
        }
        read
    }
}

#[cfg(target_os = "macos")]
fn read_texture_chunk(
    at: *const u8,
    out: &mut [u8],
    known_partial: bool,
    mut read: impl FnMut(*const u8, &mut [u8]) -> bool,
    mut next_region: impl FnMut(usize) -> Option<victory_platform::mem::Region>,
) -> bool {
    use victory_platform::mem::{Protect, page_size};

    if out.is_empty() {
        return true;
    }
    let base = at.addr();
    let page = page_size();
    let single_page = out.len() <= page - base % page;
    if (!known_partial || single_page) && read(at, out) {
        return true;
    }
    if single_page {
        out.fill(0);
        return false;
    }
    let Some(end) = base.checked_add(out.len()) else {
        out.fill(0);
        return false;
    };
    let mut cur = base;
    let mut whole = true;
    while cur < end {
        let Some(region) = next_region(cur) else {
            break;
        };
        let start = cur.max(region.base);
        let stop = end.min(region.base.saturating_add(region.len));
        if start >= stop {
            break;
        }
        if cur < start {
            out[cur - base..start - base].fill(0);
            whole = false;
        }
        if region.protect.0 & Protect::READ.0 == 0 {
            out[start - base..stop - base].fill(0);
            whole = false;
        } else if !read(start as *const u8, &mut out[start - base..stop - base]) {
            whole = false;
            // The mapping may have changed after the query. Salvage this span page by page.
            let mut pos = start;
            while pos < stop {
                let n = (page - pos % page).min(stop - pos);
                let part = &mut out[pos - base..pos - base + n];
                if !read(pos as *const u8, part) {
                    part.fill(0);
                }
                pos += n;
            }
        }
        cur = stop;
    }
    if cur < end {
        out[cur - base..].fill(0);
        whole = false;
    }
    whole
}

/// Reads `len` bytes of title memory (zeros where it cannot be read: a page the title never mapped).
pub fn read_memory(at: *const u8, len: usize) -> Vec<u8> {
    let mut out = vec![0u8; len];
    let _ = read_into(at as usize, &mut out, false);
    out
}

/// Fills `out` from title memory at `at`, with zeros where it cannot be read; whether some of it
/// could not be read at once (`VICTORY_TRACE_TEXTURE_SYNC=1` counts those textures).
fn read_into(at: usize, out: &mut [u8], known_partial: bool) -> bool {
    !read_texture_bytes_known(at as *const u8, out, known_partial)
}

/// Host texels as title memory holds them: the reverse of [`to_host`].
fn from_host(format: u32, texels: &mut [u8]) {
    if format == 115 {
        for t in texels.as_chunks_mut::<2>().0 {
            *t = u16::from_le_bytes(*t).rotate_right(4).to_le_bytes();
        }
    }
}

/// Title texels as the host format holds them: DXGI `B4G4R4A4_UNORM` (115) has blue in the low
/// nibble and alpha in the top one, Metal's `ABGR4Unorm` alpha low and red top. The host's own
/// writes and reads agree with each other, so only bytes from title memory are converted.
pub fn to_host(format: u32, texels: &mut [u8]) {
    if format == 115 {
        for t in texels.as_chunks_mut::<2>().0 {
            *t = u16::from_le_bytes(*t).rotate_left(4).to_le_bytes();
        }
    }
}

/// One mip of one slice, detiled: what `ResourceBackend::upload_texture` takes.
struct Level {
    mip: usize,
    slice: usize,
    texels: Vec<u8>,
    row: usize,
    slice_pitch: usize,
}

/// Detiles every mip of every slice of `tiled` (a texture's whole memory): the levels detiled,
/// and the error that stopped it short if one did.
fn detile(layout: &Layout, dimension: Dimension, format: u32, tiled: &[u8]) -> (Vec<Level>, Result<(), String>) {
    let slices = if dimension == Dimension::Tex3d { 1 } else { layout.slices };
    let n = layout.desc.element_bytes as usize;
    let mut levels = Vec::new();
    for mip in 0..layout.mips.len() as u32 {
        let m = &layout.mips[mip as usize];
        let row = m.width as usize * n;
        let slice_pitch = row * m.height as usize;
        let depth = layout.subresource_depth(mip).max(1) as usize;
        for slice in 0..slices {
            let mut texels = vec![0u8; slice_pitch * depth];
            if let Err(e) = layout.detile(tiled, mip, slice, &mut texels, row, slice_pitch) {
                return (levels, Err(format!("detile mip {mip} slice {slice}: {e:?}")));
            }
            to_host(format, &mut texels);
            levels.push(Level { mip: mip as usize, slice: slice as usize, texels, row, slice_pitch });
        }
    }
    (levels, Ok(()))
}

fn upload_levels<B: ResourceBackend>(backend: &B, tex: &B::Texture, levels: &[Level]) {
    for l in levels {
        backend.upload_texture(tex, l.mip, l.slice, &l.texels, l.row, l.slice_pitch);
    }
}

/// Detiles every mip of every slice of `tiled` (the texture's whole memory) into `tex`.
pub fn upload<B: ResourceBackend>(
    backend: &B,
    tex: &B::Texture,
    info: &TextureInfo,
    tiled: &[u8],
) -> Result<(), String> {
    let (levels, result) = detile(&info.layout, info.dimension, info.format, tiled);
    upload_levels(backend, tex, &levels);
    result
}

/// The reverse of [`upload`]: every mip of every slice of `tex`, retiled into `tiled` (the
/// texture's whole memory; bytes between subresources stay as they are).
fn download<B: ResourceBackend>(
    backend: &B,
    tex: &B::Texture,
    info: &TextureInfo,
    tiled: &mut [u8],
) -> Result<(), String> {
    let layout = &info.layout;
    let slices = if info.dimension == Dimension::Tex3d {
        1
    } else {
        layout.slices
    };
    let n = layout.desc.element_bytes as usize;
    for mip in 0..layout.mips.len() as u32 {
        let m = &layout.mips[mip as usize];
        let row = m.width as usize * n;
        let slice_pitch = row * m.height as usize;
        let depth = layout.subresource_depth(mip).max(1) as usize;
        let mut linear = vec![0u8; slice_pitch * depth];
        for slice in 0..slices {
            backend.read_texture(tex, mip as usize, slice as usize, &mut linear, row, slice_pitch);
            from_host(info.format, &mut linear);
            layout
                .tile(tiled, mip, slice, &linear, row, slice_pitch)
                .map_err(|e| format!("tile mip {mip} slice {slice}: {e:?}"))?;
        }
    }
    Ok(())
}

/// The largest texture copied back (the ocean maps are small; a guess until a world run's
/// "copied back" lines show what is written).
const BACK_LIMIT: usize = 4 << 20;

/// Textures compute work wrote, to be copied back into title memory once it completes
/// (`VICTORY_TEXTURE_BACK=1`). Host textures are the GPU's own, so without this the title's CPU
/// reads what was there before (VIctory-macOS issue #46: boats sink).
///
/// shortcut: copied back before a fence value is shown to the title; a title that instead polls a
/// label the GPU writes into memory may read a frame late.
pub(super) struct Back<B: PresentBackend> {
    /// A fence of ours per title queue, and the last value queued on it.
    fences: HashMap<usize, (Rc<B::Fence>, u64)>,
    /// Submissions not yet copied back: reached when the fence has the value.
    pending: Vec<(Rc<B::Fence>, u64, Vec<usize>)>,
    /// Textures already logged.
    logged: std::collections::HashSet<usize>,
}

impl<B: PresentBackend> Default for Back<B> {
    fn default() -> Self {
        Back {
            fences: HashMap::new(),
            pending: Vec::new(),
            logged: Default::default(),
        }
    }
}

impl<B: PresentBackend> Back<B> {
    /// The signal the next work of title queue `key` carries (`None` with copy-back off): in the
    /// work's own command buffer, so it does not hang on the order command buffers complete in.
    pub(super) fn next(&mut self, host: &NativeHost<B>, key: usize) -> Option<(Rc<B::Fence>, u64)> {
        if !host.collecting_written() {
            return None;
        }
        let (fence, value) = match self.fences.entry(key) {
            std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
            std::collections::hash_map::Entry::Vacant(e) => match host.backend.create_fence(0) {
                Ok(f) => e.insert((Rc::new(f), 0)),
                Err(error) => {
                    tracing::warn!(%error, "no fence to copy textures back by");
                    return None;
                }
            },
        };
        *value += 1;
        Some((fence.clone(), *value))
    }

    /// That work went to the GPU: if it wrote textures, they are copied back once `signal` is
    /// reached.
    pub(super) fn submitted(&mut self, host: &NativeHost<B>, signal: Option<(Rc<B::Fence>, u64)>) {
        let written = host.take_written();
        if let Some((fence, value)) = signal
            && !written.is_empty()
        {
            self.pending.push((fence, value, written));
        }
    }
}

/// Textures whose memory the scanner thread watches, and those it found changed.
///
/// Hashing every sampled texture's memory on every submission (150 ms at 170 MiB, under the
/// title lock the title's other threads wait on) made a title that streams textures slow to a
/// crawl as they accumulated. A thread of its own hashes them in rounds, off the lock, and a
/// submission only looks at the textures it marked.
/// shortcut: a change is seen one round later (a round is as long as the hashing takes, about
/// 100 ms for 150 MiB), so a texture written just before it is sampled may show its old content
/// for a few frames; a per-page write test would make that exact.
#[derive(Default)]
struct Watch {
    /// Title texture key to (address, bytes, whether it is a render target: its first sight is
    /// its baseline, not a change; which texture of that key: [`Tracked::serial`]).
    entries: HashMap<usize, (usize, usize, bool, u64)>,
    dirty: std::collections::HashSet<usize>,
    /// The hash of each texture's memory as last read whole, by the scanner or by a submission.
    /// What a submission read, the host texture holds: the scanner does not report it again (it
    /// used to report every new texture and every write a probe had caught already, and each
    /// report is the whole texture read once more).
    seen: HashMap<usize, blake3::Hash>,
    /// Textures whose last whole read met an unmapped or protected range (fine mips not mapped yet).
    partial: std::collections::HashSet<usize>,
}

impl Watch {
    /// Texture `key` is gone. Its hash goes with it: the key (an object's address, a placement)
    /// names another texture later, and a render target compared with the old one's hash had
    /// its memory uploaded over what was drawn into it.
    fn released(&mut self, key: usize) {
        self.entries.remove(&key);
        self.dirty.remove(&key);
        self.seen.remove(&key);
        self.partial.remove(&key);
    }
}

type Watched = std::sync::Mutex<Option<Watch>>;

static WATCH: Watched = std::sync::Mutex::new(None);

fn watch<R>(f: impl FnOnce(&mut Watch) -> R) -> R {
    watch_in(&WATCH, f)
}

fn watch_in<R>(watched: &Watched, f: impl FnOnce(&mut Watch) -> R) -> R {
    f(watched.lock().unwrap_or_else(|e| e.into_inner()).get_or_insert_default())
}

/// One round of the scanner: hashes the memory of each texture `watched` lists, without holding
/// it, and marks those that changed since they were last read whole.
fn scan_round(watched: &Watched, buf: &mut [u8]) {
    let list: Vec<(usize, (usize, usize, bool, u64))> = watch_in(watched, |w| {
        let Watch { entries, seen, partial, .. } = w;
        seen.retain(|k, _| entries.contains_key(k));
        partial.retain(|k| entries.contains_key(k));
        entries.iter().map(|(&k, &v)| (k, v)).collect()
    });
    for &(key, entry) in &list {
        let (at, size, target, _) = entry;
        // What it is compared with; `None` (skipped) if it was released since the list.
        let known = |w: &Watch| (w.entries.get(&key) == Some(&entry)).then(|| w.seen.get(&key).copied());
        let Some((before, was_partial)) = watch_in(watched, |w| known(w).map(|b| (b, w.partial.contains(&key)))) else {
            continue;
        };
        let mut h = blake3::Hasher::new();
        let mut unreadable = false;
        for off in (0..size).step_by(buf.len()) {
            let n = buf.len().min(size - off);
            unreadable |= !read_texture_bytes_known((at + off) as *const u8, &mut buf[..n], was_partial);
            h.update(&buf[..n]);
        }
        let h = h.finalize();
        watch_in(watched, |w| {
            // Released, or read by a submission while it was hashed here: this reading is the
            // older one, and the next round compares with the submission's.
            if known(w) != Some(before) {
                return;
            }
            if unreadable {
                w.partial.insert(key);
            } else {
                w.partial.remove(&key);
            }
            w.seen.insert(key, h);
            if before != Some(h) && !(target && before.is_none()) {
                w.dirty.insert(key);
            }
        });
    }
}

fn start_scanner() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let _ = std::thread::Builder::new().name("victory texture scan".into()).spawn(|| {
            let mut buf = vec![0u8; 1 << 20];
            loop {
                scan_round(&WATCH, &mut buf);
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
        });
    });
}

/// A texture the host keeps for a title texture.
pub struct Tracked<B: ResourceBackend> {
    pub texture: Rc<B::Texture>,
    info: TextureInfo,
    /// Which texture this is: its key (an object's address, a placement) names another one later.
    serial: u64,
    /// The hash of the memory last uploaded; `None` for targets and before the first upload.
    last: Option<blake3::Hash>,
    /// Uploads of its memory so far.
    pub uploads: u64,
    /// Submissions left in which its memory is probed at submission rather than left to the
    /// scanner (0: cold), and the hash of its samples when it was last read whole (a probe that
    /// reads others reports a write).
    hot: u32,
    probe: Option<blake3::Hash>,
    /// Whether its last write showed in the sampled probe (a movie's planes, a streamed texture:
    /// written whole): then it is read whole only every `WHOLE_EVERY` probes (backing off across
    /// consecutive unchanged whole checks), and probes skipped since. Set at each upload
    /// (`sync_apply`), and returned to periodic checks once a follow-up whole read finds no write.
    sampled_sees_writes: bool,
    unchanged_whole: u8,
    skipped: u32,
    /// Whether its last whole read met an unmapped or protected range (fine mips not mapped yet).
    partial: bool,
    /// Its last upload was all zeros: it is probed (`Title::blank_keys`) until written, however
    /// long that takes (a movie's planes are made blank about 40 s before the movie decodes into
    /// them; once left to the scanner, their first frames showed green for a second or two).
    blank: bool,
    /// Render-target views already made, kept so the same attachment keeps its identity.
    #[cfg(target_os = "macos")]
    target_views: Vec<(ViewDesc, Rc<B::Texture>)>,
    /// The GPU wrote it (a copy, draw, clear or dispatch): its host texture holds what title
    /// memory does not, and is never filled from that memory again.
    gpu_written: bool,
}

impl<B: ResourceBackend> Tracked<B> {
    /// Whether the host texture is no longer filled from title memory: the GPU wrote it
    /// ([`Title::gpu_written`]). One never uploaded and not a render target takes its first upload.
    fn kept_from_memory(&self) -> bool {
        self.gpu_written && (self.uploads > 0 || self.info.render_target())
    }
}

/// The other format of an sRGB pair.
#[cfg(target_os = "macos")]
fn srgb_twin(f: Format) -> Option<Format> {
    Some(match f {
        Format::Rgba8Unorm => Format::Rgba8UnormSrgb,
        Format::Rgba8UnormSrgb => Format::Rgba8Unorm,
        Format::Bgra8Unorm => Format::Bgra8UnormSrgb,
        Format::Bgra8UnormSrgb => Format::Bgra8Unorm,
        _ => return None,
    })
}

/// Submissions a texture whose memory changed stays hot without a further change.
const HOT: u32 = 600;
/// Time one submission spends probing hot textures, and again blank ones. Boot movies and UI
/// glyph caches have a few hot textures and a few hundred blank ones (the movie's planes among
/// them, blank until the title decodes into them), all of which a submission must reach: a count
/// limit left a plane's first frames unseen (solid green). When world streaming realizes thousands
/// of blank textures the time bounds the probing and the cursor resumes where the last submission
/// stopped, while the scanner thread catches the rest off the host lock.
///
/// shortcut: past a few thousand blank textures a write is noticed a few submissions late (unless
/// the scanner catches it first); at least `MIN_SYNC_HOT` of each are probed whatever the time.
const PROBE_BUDGET: std::time::Duration = std::time::Duration::from_millis(2);
/// Hot textures a submission probes at least, however long the probes take.
const MIN_SYNC_HOT: usize = 32;
/// Pages of a large hot texture's memory a probe reads, spread over it.
const PROBE_PAGES: usize = 8;
/// Bytes sampled per page when probing a large hot texture.
const PROBE_SAMPLE: usize = 256;
/// Hot textures up to this size whose last write its samples missed are read whole at every probe
/// (a glyph cache gets a few glyphs at a time). Those written so that the samples showed it (a
/// streamed texture, written whole after its blank upload) are read whole now and then
/// (`WHOLE_EVERY`): in the world they are thousands, and read whole at every probe they were 2,000
/// textures and 500 MiB a second, most of what a submission read.
///
/// shortcut: an atlas whose first glyphs reach a sampled page gets its next ones up to
/// `WHOLE_EVERY` probes late, once (then its samples have missed a write); per-sample hashes
/// would tell a partial write from a whole one.
const PROBE_WHOLE: usize = 4 << 20;

/// A hot texture whose writes show in its samples (or whose follow-up whole check after a missed
/// write found nothing further) is read whole once in this many probes, doubling after each
/// further unchanged whole check up to `WHOLE_EVERY << WHOLE_BACKOFF_MAX`, to catch a write that
/// misses the samples (a glyph added to an atlas a movie-like rewrite went through). Reading the
/// movie's chroma planes whole at every submission cost 15 ms a submission, halving the movie's
/// frame rate.
pub(super) const WHOLE_EVERY: u32 = 8;
/// Maximum doubling steps (`WHOLE_EVERY << 5` = 256 probes) for repeated unchanged whole checks
/// while the background scanner continues to watch every texture (a guess).
const WHOLE_BACKOFF_MAX: u32 = 5;
/// Bytes of hot textures one submission reads whole to check them, once `MIN_SYNC_HOT` were
/// probed: the reading is off the host's lock but not free (a guess at what a submission can
/// spare: about 5 ms of one thread under Rosetta).
const WHOLE_BUDGET: usize = 8 << 20;
/// Of those, bytes of the ones read whole only now and then (`WHOLE_EVERY`: their samples showed
/// their last write). In the world they are nearly all of the hot textures and the title submits
/// some 30 times a frame; one that is due and not read stays due (a guess, as above: 1 ms).
const WHOLE_SPARE: usize = 1 << 20;

/// Hash of the bytes a probe of `size` bytes at `at` reads ([`probe_read`]).
fn probe_hash(at: usize, size: usize, whole: bool, scratch: &mut Vec<u8>) -> blake3::Hash {
    let mut hasher = blake3::Hasher::new();
    probe_read(at, size, whole, scratch, |b| {
        hasher.update(b);
    });
    hasher.finalize()
}

/// Whether the `PROBE_SAMPLE` bytes in the middle of the `size` bytes at `at` are not all zero:
/// a blank texture's probe, one read (a few hundred blank textures are probed every submission).
///
/// shortcut: a first write that misses the middle (a glyph in a blank atlas) waits for the
/// scanner; textures written whole (streaming, a movie's planes) are seen at once.
fn probe_nonzero(at: usize, size: usize, scratch: &mut Vec<u8>) -> bool {
    let off = (size / 2) & !4095;
    let n = PROBE_SAMPLE.min(size - off);
    if scratch.len() < n {
        scratch.resize(n, 0);
    }
    read_texture_bytes((at + off) as *const u8, &mut scratch[..n]);
    scratch[..n].iter().any(|&b| b != 0)
}

/// Hands `seen` the bytes a probe of `size` bytes at `at` reads: all of them if `whole`, else
/// `PROBE_PAGES` samples of `PROBE_SAMPLE` bytes spread across the texture, reusing `scratch`.
///
/// shortcut: when sampled, a write that misses every sampled window goes unseen until the
/// scanner's next round; a texture the title rewrites whole (a movie's planes) is caught at once.
fn probe_read(at: usize, size: usize, whole: bool, scratch: &mut Vec<u8>, mut seen: impl FnMut(&[u8])) {
    if whole || size <= PROBE_PAGES * PROBE_SAMPLE {
        const CHUNK: usize = 1 << 20;
        let cap = size.min(CHUNK);
        if scratch.len() < cap {
            scratch.resize(cap, 0);
        }
        for off in (0..size).step_by(CHUNK) {
            let n = CHUNK.min(size - off);
            read_texture_bytes((at + off) as *const u8, &mut scratch[..n]);
            seen(&scratch[..n]);
        }
        return;
    }
    const PAGE: usize = 4096;
    let pages = size.div_ceil(PAGE);
    if scratch.len() < PROBE_SAMPLE {
        scratch.resize(PROBE_SAMPLE, 0);
    }
    for k in 0..PROBE_PAGES {
        let off = k * pages / PROBE_PAGES * PAGE;
        let n = PROBE_SAMPLE.min(size - off);
        read_texture_bytes((at + off) as *const u8, &mut scratch[..n]);
        seen(&scratch[..n]);
    }
}

/// For the trace: hot textures a probe read whole, and their bytes.
static WHOLE_PROBES: [std::sync::atomic::AtomicU64; 2] = [std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0)];
/// For the trace: placements that took the place of another at their address, and those of them
/// that had been at that address before.
static REPLACED: [std::sync::atomic::AtomicU64; 2] = [std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0)];

/// Probes `keys` round-robin from `cursor`, each at most once. An expired deadline, or
/// `WHOLE_BUDGET` of whole reads, stops the round after at least `MIN_SYNC_HOT` probes (no deadline:
/// neither does). Those whose memory changed, and the hot ones to be read whole to tell, go to
/// `dirty`. Hot textures that stay unchanged cool down and leave; blank ones (`blank`) stay until
/// they are written.
fn probe_round<B: ResourceBackend>(
    textures: &mut Textures<B>,
    keys: &mut Vec<usize>,
    cursor: &mut usize,
    until: Option<std::time::Instant>,
    blank: bool,
    scratch: &mut Vec<u8>,
    dirty: &mut std::collections::HashSet<usize>,
) {
    let mut left = keys.len();
    // Where the round starts: keys before it are probed after the wrap.
    let start = *cursor % keys.len().max(1);
    let mut probed = 0;
    // Bytes of the textures to be read whole, and of those among them read so only now and then.
    let mut whole = 0;
    let mut spare = 0;
    while left > 0 && !keys.is_empty() {
        if probed >= MIN_SYNC_HOT && until.is_some_and(|until| std::time::Instant::now() >= until || whole >= WHOLE_BUDGET) {
            break;
        }
        left -= 1;
        let idx = *cursor % keys.len();
        let key = keys[idx];
        let Some(t) = textures.get_mut(&key).filter(|t| if blank { t.blank } else { t.hot > 0 }) else {
            leave_round(keys, idx, start, cursor);
            continue;
        };
        probed += 1;
        let at = t.info.memory as usize;
        let leave = if blank {
            // Written at last.
            let written = probe_nonzero(at, t.info.size, scratch);
            if written {
                dirty.insert(key);
                t.blank = false;
            }
            written
        } else {
            let sampled = probe_hash(at, t.info.size, false, scratch);
            // Its samples at the last upload (`sync_apply` moves them on).
            let before = t.probe;
            let changed = before != Some(sampled);
            if !changed && t.uploads >= 2 && t.info.size <= PROBE_WHOLE {
                let every = if t.sampled_sees_writes {
                    WHOLE_EVERY << u32::from(t.unchanged_whole.saturating_sub(1)).min(WHOLE_BACKOFF_MAX)
                } else {
                    1
                };
                t.skipped = t.skipped.saturating_add(1);
                if t.skipped >= every && !(t.sampled_sees_writes && spare >= WHOLE_SPARE) {
                    t.skipped = 0;
                    if t.sampled_sees_writes {
                        spare += t.info.size;
                    }
                    // Read whole with the textures the submission reads anyway, without the host's
                    // lock ([`prepare`]), and uploaded if that is not what the host texture holds
                    // (a write the samples missed: `sync_apply`). Hashed here, the hot textures
                    // of a title that streams were hundreds of MiB a second under the lock.
                    dirty.insert(key);
                    whole += t.info.size;
                    WHOLE_PROBES[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    WHOLE_PROBES[1].fetch_add(t.info.size as u64, std::sync::atomic::Ordering::Relaxed);
                }
            }
            if changed {
                dirty.insert(key);
            } else {
                t.hot -= 1;
            }
            t.hot == 0
        };
        if leave {
            leave_round(keys, idx, start, cursor);
        } else {
            *cursor = idx + 1;
        }
    }
}

/// Takes `keys[idx]` out of a [`probe_round`] that started at `start`. The last key moves into its
/// place: if it was at or past `start` and the round has wrapped, it was probed already and the
/// cursor moves past it; otherwise it is still to be probed and the cursor stays on `idx`.
/// (Probing it again cost the round its last step, and the key just before `start` went
/// unprobed: a blank texture's first frame came a submission late whenever another blank texture
/// was written in the same round.)
fn leave_round(keys: &mut Vec<usize>, idx: usize, start: usize, cursor: &mut usize) {
    keys.swap_remove(idx);
    // `keys.len()` is now where the moved key was.
    *cursor = if idx < start && keys.len() >= start { idx + 1 } else { idx };
}

/// The key of a texture a view names by address and description (no resource object): a number no
/// object address has (the top bit is set).
pub fn placed_key(address: u64, description: &[u32; 14]) -> usize {
    let mut h = blake3::Hasher::new();
    h.update(&address.to_le_bytes());
    for (i, w) in description.iter().enumerate() {
        // Words 1 and 13 are padding (stack bytes in the title's records).
        h.update(&(if i == 1 || i == 13 { 0 } else { *w }).to_le_bytes());
    }
    let b = h.finalize();
    (u64::from_le_bytes(b.as_bytes()[..8].try_into().expect("8 bytes")) | 1 << 63) as usize
}

impl<B: PresentBackend> Title<B> {
    /// Turns copying GPU-written textures back into title memory on or off.
    pub fn set_texture_back(&mut self, on: bool) {
        self.host.collect_written(on);
    }

    /// Copies back what completed submissions wrote.
    pub(super) fn copy_back(&mut self) {
        let mut done = Vec::new();
        let backend = &self.host.backend;
        self.back.pending.retain(|(fence, value, keys)| {
            let reached = backend.completed(fence) >= *value;
            if reached {
                done.extend(keys.iter().copied());
            }
            !reached
        });
        done.sort_unstable();
        done.dedup();
        for key in done {
            self.write_back(key);
        }
    }

    /// Retiles the host texture of `key` into its title memory.
    fn write_back(&mut self, key: usize) {
        // Not a texture (a buffer view), or released since: nothing to copy.
        let Some(t) = self.textures.get_mut(&key) else { return };
        let info = &t.info;
        // Render and depth targets are left out: nothing says the title reads one.
        if info.flags & UNORDERED_ACCESS == 0 || info.render_target() {
            return;
        }
        let described = (info.format, info.width, info.height, info.depth_or_layers, info.size);
        if info.size > BACK_LIMIT {
            if self.back.logged.insert(key) {
                tracing::info!(texture = key, ?described, "a GPU-written texture is too large to copy back");
            }
            return;
        }
        let mut tiled = read_memory(info.memory, info.size);
        // The title wrote the memory since the host texture was filled from it: its bytes stay,
        // and the next submission uploads them.
        if t.last != Some(blake3::hash(&tiled)) {
            return;
        }
        if let Err(error) = download(&self.host.backend, &t.texture, info, &mut tiled) {
            if self.back.logged.insert(key) {
                tracing::warn!(texture = key, ?described, %error, "a GPU-written texture was not copied back");
            }
            return;
        }
        let hash = blake3::hash(&tiled);
        if t.last == Some(hash) {
            return;
        }
        // Through the pool's own mapping; `None` when the title unmapped the memory meanwhile.
        let Ok((_buffer, at)) = self.host.resolve(info.memory as u64, info.size as u64) else { return };
        // SAFETY: `resolve` gives `size` mapped, writable bytes of the pool; the buffer is kept.
        unsafe { std::ptr::copy_nonoverlapping(tiled.as_ptr(), at.cast_mut(), tiled.len()) };
        // What the host texture holds, so the scanner's sighting of this write uploads nothing.
        t.last = Some(hash);
        if self.back.logged.insert(key) {
            tracing::info!(texture = key, ?described, layout = ?info.layout.desc, "a GPU-written texture is copied back into title memory");
        }
    }

    /// A title texture exists. Its host texture is made when a view, a render target or a clear
    /// first names it (most of a title's textures are never used), and filled from title memory
    /// when work next goes to the GPU.
    pub fn texture_created(&mut self, key: usize, info: TextureInfo) {
        self.pending.insert(key, info);
    }

    /// Makes the host texture of title texture `key` if it has none yet and registers it for
    /// views. `false` when there is none (not a title texture, or one the host cannot represent).
    pub fn realize(&mut self, key: usize) -> bool {
        if self.textures.contains_key(&key) {
            return true;
        }
        let Some(info) = self.pending.remove(&key) else {
            return false;
        };
        let Some(desc) = info.desc() else {
            // Loud once per format: views of it get no texture and their draws sample nothing.
            static SEEN: std::sync::Mutex<Vec<u32>> = std::sync::Mutex::new(Vec::new());
            let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
            if !seen.contains(&info.format) {
                seen.push(info.format);
                tracing::warn!(format = info.format, w = info.width, h = info.height, flags = info.flags, "a title texture format the host does not represent (first of this format)");
            }
            return false;
        };
        let tex = match self.host.texture(&desc) {
            Ok(t) => t,
            Err(e) => {
                tracing::warn!(error = %e, ?desc, "no host texture");
                return false;
            }
        };
        self.host.register_texture(key, &tex);
        static SERIAL: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let serial = SERIAL.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // Depth-stencil textures are GPU-written depth targets whose CPU tile layout (e.g.
        // 4-byte or 2-byte plane 0) does not match Metal's combined depth-stencil texel layout,
        // so watching or uploading CPU memory cannot produce valid replaceRegion transfers.
        if !desc.format.is_depth() {
            // Render targets too: what the title (CPU, DMA) writes into their memory is uploaded
            // when it changes, as it is not otherwise seen (a GPU write never changes title memory).
            watch(|w| w.entries.insert(key, (info.memory as usize, info.size, info.render_target(), serial)));
            start_scanner();
            // Debugging (VICTORY_MAGENTA=1): render targets start magenta, so a frame that stays so
            // was never drawn into, and a black one was drawn black.
            if info.render_target() && env_set!("VICTORY_MAGENTA") {
                let m = &info.layout.mips[0];
                let n = info.layout.desc.element_bytes as usize;
                let row = m.width as usize * n;
                let pixel = [0xffu8, 0, 0xff, 0xff];
                let linear: Vec<u8> = (0..row * m.height as usize).map(|i| pixel[i % 4]).collect();
                self.host.backend.upload_texture(&tex, 0, 0, &linear, row, row * m.height as usize);
            }
            if !info.render_target() {
                self.unloaded.insert(key);
            }
        }
        self.textures.insert(
            key,
            Tracked {
                texture: tex,
                info,
                serial,
                last: None,
                uploads: 0,
                hot: 0,
                probe: None,
                sampled_sees_writes: false,
                unchanged_whole: 0,
                skipped: 0,
                partial: false,
                blank: false,
                #[cfg(target_os = "macos")]
                target_views: Vec::new(),
                gpu_written: false,
            },
        );
        true
    }

    /// The format of the host texture of title texture `key`.
    pub fn format_of(&self, key: usize) -> Option<Format> {
        self.textures.get(&key).and_then(|t| t.info.desc()).map(|d| d.format)
    }

    /// The host texture of title texture `key`.
    pub fn texture_of(&mut self, key: usize) -> Option<Rc<B::Texture>> {
        self.realize(key);
        self.textures.get(&key).map(|t| t.texture.clone())
    }

    /// Resolves a render-target view to its attachment and its effective format. The base
    /// attachment can use the texture itself. A narrowed view also supplies the mip's extent.
    #[cfg(target_os = "macos")]
    pub(crate) fn target_of(
        &mut self,
        record: crate::descriptor::RtvRecord,
    ) -> Option<(Rc<B::Texture>, Format)> {
        let key = record.resource as usize;
        self.realize(key);
        let t = self.textures.get_mut(&key)?;
        let desc = t.info.desc()?;
        let (mip, slice) = record.native_rtv_subresource(
            t.info.dimension,
            t.info.depth_or_layers,
            desc.mips,
            t.info.layout.desc.samples,
        )?;
        let base = (mip, slice) == (0, 0);
        let own = desc.format;
        let twin = format(record.format()).filter(|&f| srgb_twin(own) == Some(f));
        // A failed format reinterpretation can fall back to the resource format, but must
        // keep the same mip and slice.
        for format in twin.into_iter().chain([own]) {
            if base && format == own {
                return Some((t.texture.clone(), own));
            }
            let view = ViewDesc {
                format,
                mips: if base { (0, usize::MAX) } else { (mip, 1) },
                slices: if base { (0, usize::MAX) } else { (slice, 1) },
            };
            if let Some((_, texture)) = t.target_views.iter().find(|(cached, _)| *cached == view) {
                return Some((texture.clone(), format));
            }
            match self.host.backend.create_texture_view(&t.texture, &view) {
                Ok(texture) => {
                    let texture = Rc::new(texture);
                    t.target_views.push((view, texture.clone()));
                    return Some((texture, format));
                }
                Err(e) => tracing::warn!(error = %e, ?view, "no host view for a render target view"),
            }
        }
        None
    }

    /// The GPU writes title texture `key` (made already): a copy, draw or clear recorded into it,
    /// or a prepared dispatch's unordered-access view of it. From now on the host texture is the
    /// GPU's and the CPU-side sync no longer fills it from title memory (which titles change under
    /// their render targets: uploaded, that was block noise over what the GPU drew).
    /// A sampled texture not uploaded yet still takes its first upload, at the submission and so
    /// before the write runs (a copy of a region keeps the rest).
    /// shortcut: what the title writes into its memory afterwards is not seen; when a title is
    /// found to refill a GPU-written texture from the CPU, upload a change again.
    pub fn gpu_written(&mut self, key: usize) {
        // A placed view sharing an object's texture ([`Self::alias_placed`]) writes the object's.
        let key = self.shared.iter().find(|(_, (_, placed))| *placed == key).map_or(key, |(&object, _)| object);
        if let Some(t) = self.textures.get_mut(&key) {
            t.gpu_written = true;
            (t.hot, t.blank) = (0, false);
        }
    }

    /// How many times title texture `key`'s memory was detiled into its host texture.
    pub fn uploads_of(&self, key: usize) -> u64 {
        self.textures.get(&key).map_or(0, |t| t.uploads)
    }

    pub fn texture_released(&mut self, key: usize) {
        self.textures.remove(&key);
        // A placed view that shares this texture ([`Self::alias_placed`]) goes with it.
        if let Some((address, placed)) = self.shared.remove(&key)
            && self.placed.get(&address).is_some_and(|&(k, _)| k == placed)
        {
            self.placed.remove(&address);
            self.host.release_resource(placed);
        }
        self.pending.remove(&key);
        self.unloaded.remove(&key);
        self.hot_keys.retain(|&k| k != key);
        self.blank_keys.retain(|&k| k != key);
        watch(|w| w.released(key));
    }

    /// A view placed texture `key` at title `address` (in pool `pool`). The title has no object to
    /// release for it, so the texture another description placed at that address before goes
    /// now (guess: the title is done with it once it places another there), and every texture
    /// placed in a pool goes when the pool is released ([`Self::sync_dirty`]).
    pub fn texture_placed(&mut self, address: u64, key: usize, pool: usize) {
        if let Some((old, _)) = self.placed.insert(address, (key, pool))
            && old != key
        {
            if env_set!("VICTORY_TRACE_TEXTURE_SYNC") {
                // Whether two descriptions keep taking one address from each other.
                static WERE: std::sync::Mutex<Option<std::collections::HashSet<(u64, usize)>>> = std::sync::Mutex::new(None);
                let mut were = WERE.lock().unwrap_or_else(|e| e.into_inner());
                let were = were.get_or_insert_default();
                were.insert((address, old));
                REPLACED[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                REPLACED[1].fetch_add(u64::from(were.contains(&(address, key))), std::sync::atomic::Ordering::Relaxed);
            }
            self.host.forget_resource(old);
            self.texture_released(old);
        }
    }

    /// A placed R9G9B9E5 texture `key` laid out exactly as a live R32_UINT resource object at the
    /// same memory (the title reads as floats what compute wrote as integers): `key` names the
    /// object's host texture, which its views cast, instead of a copy only title memory fills.
    /// `false` (nothing done) for any other placed texture.
    pub fn alias_placed(&mut self, address: u64, key: usize, info: &TextureInfo) -> bool {
        let over = |o: &TextureInfo| {
            o.format == 42 && o.memory == info.memory && o.size == info.size && o.layout == info.layout
        };
        // Object keys are addresses; placed keys have the top bit set ([`placed_key`]).
        let object = |k: usize| k >> 63 == 0;
        if info.format != 67 {
            return false;
        }
        let Some(obj) = self
            .textures
            .iter()
            .map(|(&k, t)| (k, &t.info))
            .chain(self.pending.iter().map(|(&k, i)| (k, i)))
            .find(|&(k, i)| object(k) && over(i))
            .map(|(k, _)| k)
        else {
            return false;
        };
        let Some(texture) = self.texture_of(obj) else { return false };
        self.host.register_texture(key, &texture);
        self.shared.insert(obj, (address, key));
        static ALIASED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = ALIASED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        if n <= 1000 {
            tracing::info!(placed = format!("{key:#x}"), object = format!("{obj:#x}"), aliased = n, "a placed R9G9B9E5 view shares an R32_UINT texture");
        }
        true
    }

    /// Releases the textures placed in pools the title released since the last call.
    fn release_placed(&mut self) {
        let gone = self.host.released_pools();
        if gone.is_empty() {
            return;
        }
        let dead: Vec<u64> = self
            .placed
            .iter()
            .filter(|(_, (_, pool))| gone.contains(pool))
            .map(|(&a, _)| a)
            .collect();
        for a in dead {
            let (key, _) = self.placed.remove(&a).expect("listed above");
            self.host.release_resource(key);
            self.texture_released(key);
        }
    }

    /// Brings the textures the host samples up to date with title memory: those whose memory
    /// changed since their last upload are detiled again.
    pub fn sync_textures(&mut self) {
        let keys: Vec<usize> = self
            .textures
            .iter()
            .filter(|(_, t)| !t.info.render_target())
            .map(|(&k, _)| k)
            .collect();
        let jobs = self.jobs(keys);
        self.sync(jobs);
    }

    /// What a submission does: the textures not uploaded yet, and those the scanner or hot probe
    /// found changed since. (`glue::submit` takes the same steps, the middle one without the
    /// host's lock.)
    pub fn sync_dirty(&mut self) {
        let jobs = self.sync_jobs();
        self.sync(jobs);
    }

    #[cfg(all(test, target_os = "macos"))]
    pub(super) fn sync_dirty_without_probe_deadline(&mut self) {
        let jobs = self.sync_jobs_within(None);
        self.sync(jobs);
    }

    /// Reads `jobs` and uploads what changed.
    pub(super) fn sync(&mut self, jobs: Vec<Job>) {
        for batch in batches(jobs) {
            let prepared = prepare(&batch);
            self.sync_apply(prepared);
        }
    }

    /// The textures a submission must read: those not uploaded yet, and those the scanner or
    /// the probes of hot and blank textures found changed.
    pub(super) fn sync_jobs(&mut self) -> Vec<Job> {
        self.sync_jobs_within(Some(PROBE_BUDGET))
    }

    /// [`Self::sync_jobs`] with `budget` of time for each round of probes (none: every hot and
    /// blank texture is probed).
    fn sync_jobs_within(&mut self, budget: Option<std::time::Duration>) -> Vec<Job> {
        self.release_placed();
        let mut dirty = watch(|w| std::mem::take(&mut w.dirty));
        // Hot textures (a movie's planes, a glyph cache) are probed synchronously, round-robin within
        // a time budget so world streaming (thousands of blank textures) never stalls submissions;
        // the ones that changed lately first, then the blank ones.
        let started = std::time::Instant::now();
        let until = budget.map(|budget| started + budget);
        probe_round(&mut self.textures, &mut self.hot_keys, &mut self.hot_cursor, until, false, &mut self.probe_buf, &mut dirty);
        // The blank ones get a budget of their own: a busy movie must not starve its next planes.
        let hot = std::time::Instant::now();
        let until = budget.map(|budget| hot + budget);
        probe_round(&mut self.textures, &mut self.blank_keys, &mut self.blank_cursor, until, true, &mut self.probe_buf, &mut dirty);
        if env_set!("VICTORY_TRACE_TEXTURE_SYNC") {
            SyncCost { probe: [hot - started, hot.elapsed()], hot: self.hot_keys.len() as u64, ..Default::default() }.log();
        }
        let unloaded = std::mem::take(&mut self.unloaded);
        self.jobs(keys_to_sync(&unloaded, &dirty))
    }

    pub(super) fn jobs(&self, keys: Vec<usize>) -> Vec<Job> {
        let job = |key| {
            let t = self.textures.get(&key)?;
            // Depth-stencil memory is not laid out as the host's texels are (`realize`).
            if t.info.desc().is_some_and(|d| d.format.is_depth()) || t.kept_from_memory() {
                return None;
            }
            Some(Job {
                key,
                serial: t.serial,
                memory: t.info.memory as usize,
                size: t.info.size,
                format: t.info.format,
                dimension: t.info.dimension,
                layout: t.info.layout.clone(),
                last: t.last,
                partial: t.partial,
            })
        };
        keys.into_iter().filter_map(job).collect()
    }

    /// Uploads what [`prepare`] detiled, into the textures that are still the ones it read.
    pub(super) fn sync_apply(&mut self, prepared: Prepared) {
        let Title {
            host,
            textures,
            hot_keys,
            blank_keys,
            stats,
            ..
        } = self;
        let started = std::time::Instant::now();
        let mut cost = SyncCost { batches: 1, prepare: prepared.took, ..Default::default() };
        let mut read = Vec::new();
        for r in prepared.ready {
            cost.textures += 1;
            cost.bytes += r.bytes as u64;
            cost.read += r.cost[0];
            cost.hash += r.cost[1];
            cost.detile += r.cost[2];
            cost.unreadable += u64::from(r.unreadable);
            // Released while it was read (its key may name another texture by now).
            // Or a copy into it was recorded while it was read: the GPU's from now on.
            let Some(t) = textures.get_mut(&r.key).filter(|t| t.serial == r.serial && !t.kept_from_memory()) else {
                continue;
            };
            let key = r.key;
            t.partial = r.unreadable;
            read.push((key, t.serial, r.hash, r.unreadable));
            cost.first += u64::from(t.last.is_none());
            // What a probe of this memory reads: the next one reports a write since, not this
            // content again (a texture's first probe used to, and that was a whole read wasted).
            let sampled = t.probe.replace(r.probe);
            // `last` again: a copy back into title memory may have moved it meanwhile.
            let Some((levels, detiled)) = r.levels.filter(|_| t.last != Some(r.hash)) else {
                // Nothing changed since the last upload: if its previous write missed the samples,
                // this follow-up whole read has checked for an immediate next write, and further
                // checks return to `WHOLE_EVERY` within `WHOLE_SPARE` with backoff.
                t.sampled_sees_writes = true;
                t.unchanged_whole = t.unchanged_whole.saturating_add(1);
                cost.unchanged += 1;
                continue;
            };
            t.unchanged_whole = 0;
            if let Some(sampled) = sampled {
                // Whoever found this write: one its samples do not show has it read whole at
                // every probe while it is hot, one they show only now and then.
                t.sampled_sees_writes = sampled != r.probe;
            }
            cost.blank += u64::from(r.blank);
            if !t.info.render_target() {
                if r.blank && !t.blank {
                    blank_keys.push(key);
                } else if !r.blank && t.uploads > 0 {
                    // Written again after its first upload: hot.
                    if t.hot == 0 {
                        hot_keys.push(key);
                    }
                    t.hot = HOT;
                }
                t.blank = r.blank;
            }
            // Pass trace: uploads from title memory into textures the GPU writes (targets, UAVs).
            if super::pass_on() && (t.info.render_target() || t.info.flags & UNORDERED_ACCESS != 0) {
                eprintln!("PASSUPLOAD key={key:#x} fmt={} {}x{} at={:#x} rt={} uav={} uploads={} blank={} unreadable={}", t.info.format, t.info.width, t.info.height, t.info.memory as usize, t.info.render_target(), t.info.flags & UNORDERED_ACCESS != 0, t.uploads, r.blank, r.unreadable);
            }
            upload_levels(&host.backend, &t.texture, &levels);
            t.last = Some(r.hash);
            match detiled {
                Ok(()) => {
                    t.uploads += 1;
                    stats.detiles += 1;
                }
                Err(e) => {
                    stats.undetilable += 1;
                    tracing::warn!(texture = key, error = %e, "a title texture was not detiled");
                }
            }
        }
        watch(|w| {
            for (key, serial, hash, unreadable) in read {
                if w.entries.get(&key).is_some_and(|e| e.3 == serial) {
                    if unreadable {
                        w.partial.insert(key);
                    } else {
                        w.partial.remove(&key);
                    }
                    // What the scanner reported while this was read is this same content.
                    if w.seen.insert(key, hash) == Some(hash) {
                        w.dirty.remove(&key);
                    }
                }
            }
        });
        if env_set!("VICTORY_TRACE_TEXTURE_SYNC") {
            cost.upload = started.elapsed();
            cost.log();
        }
    }
}

/// A texture whose memory a submission reads: what the reading needs, taken under the host's lock
/// so that [`prepare`] runs without it.
pub(super) struct Job {
    pub(super) key: usize,
    serial: u64,
    memory: usize,
    size: usize,
    format: u32,
    dimension: Dimension,
    layout: Layout,
    /// The hash of what the host texture holds: memory that still hashes to it is not detiled.
    last: Option<blake3::Hash>,
    /// Whether an earlier whole read met an unmapped or protected range.
    partial: bool,
}

/// What a [`Job`]'s memory held when [`prepare`] read it.
struct Ready {
    key: usize,
    serial: u64,
    bytes: usize,
    hash: blake3::Hash,
    /// What a sampled probe of that memory hashes to ([`probe_hash`]).
    probe: blake3::Hash,
    /// Some of it could not be read (and counts as zeros).
    unreadable: bool,
    /// All zeros.
    blank: bool,
    /// Detiled, and how that went; `None` when the host texture holds this content already.
    levels: Option<(Vec<Level>, Result<(), String>)>,
    /// Time reading, hashing and detiling.
    cost: [std::time::Duration; 3],
}

/// What [`prepare`] made of a batch of jobs, and how long it took.
pub(super) struct Prepared {
    ready: Vec<Ready>,
    took: std::time::Duration,
}

/// Threads a submission's textures are read on (the submitting thread is one of them).
const SYNC_THREADS: usize = 6;
/// Texture memory read at a time: its detiled copy is held until it is uploaded.
const SYNC_BATCH: usize = 256 << 20;

/// `jobs` in runs of at most `SYNC_BATCH` bytes of texture memory (a larger texture alone).
pub(super) fn batches(jobs: Vec<Job>) -> Vec<Vec<Job>> {
    let mut out: Vec<Vec<Job>> = Vec::new();
    let mut bytes = 0;
    for job in jobs {
        if out.is_empty() || bytes + job.size > SYNC_BATCH {
            out.push(Vec::new());
            bytes = 0;
        }
        bytes += job.size;
        out.last_mut().expect("pushed above").push(job);
    }
    out
}

impl Job {
    /// Reads its memory into `tiled` (kept from job to job: a fresh buffer each time maps and
    /// faults in new pages) and detiles it unless it is what the host texture holds.
    fn read(&self, tiled: &mut Vec<u8>) -> Ready {
        let started = std::time::Instant::now();
        // Sampled before the whole read: a write between the two is then a change to the next
        // probe, not part of what it compares with.
        let probe = probe_hash(self.memory, self.size, false, tiled);
        if tiled.len() < self.size {
            tiled.resize(self.size, 0);
        }
        let tiled = &mut tiled[..self.size];
        let unreadable = read_into(self.memory, tiled, self.partial);
        let read = started.elapsed();
        let hash = blake3::hash(tiled);
        let hashed = started.elapsed();
        let levels = (self.last != Some(hash)).then(|| detile(&self.layout, self.dimension, self.format, tiled));
        Ready {
            key: self.key,
            serial: self.serial,
            bytes: self.size,
            hash,
            probe,
            unreadable,
            blank: levels.is_some() && tiled.iter().all(|&b| b == 0),
            levels,
            cost: [read, hashed - read, started.elapsed() - hashed],
        }
    }
}

/// Reads, hashes and detiles the textures of `jobs`, in their order. This is the bulk of a texture
/// sync and needs nothing of the host, so it runs without the host's lock, on a few threads.
pub(super) fn prepare(jobs: &[Job]) -> Prepared {
    let started = std::time::Instant::now();
    let next = std::sync::atomic::AtomicUsize::new(0);
    let done = std::sync::Mutex::new(Vec::with_capacity(jobs.len()));
    let work = || {
        let mut tiled = Vec::new();
        loop {
            let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let Some(job) = jobs.get(i) else { break };
            let ready = job.read(&mut tiled);
            done.lock().unwrap_or_else(|e| e.into_inner()).push((i, ready));
        }
    };
    std::thread::scope(|s| {
        for _ in 1..SYNC_THREADS.min(jobs.len()) {
            // Without the thread, the others (this one at least) read its share.
            let _ = std::thread::Builder::new().name("victory texture read".into()).spawn_scoped(s, work);
        }
        work();
    });
    let mut done = done.into_inner().unwrap_or_else(|e| e.into_inner());
    done.sort_by_key(|&(i, _)| i);
    Prepared { ready: done.into_iter().map(|(_, r)| r).collect(), took: started.elapsed() }
}

/// What syncing textures cost: logged with `VICTORY_TRACE_TEXTURE_SYNC=1`, summed over each second.
#[derive(Default)]
struct SyncCost {
    batches: u64,
    textures: u64,
    /// Never uploaded before (a new texture, or a render target's memory written for the first time).
    first: u64,
    /// Read and hashed, and what the host texture holds already.
    unchanged: u64,
    /// All zeros (detiled and uploaded all the same).
    blank: u64,
    /// With memory that could not be read.
    unreadable: u64,
    bytes: u64,
    /// Hot textures at a submission (the last of the second is logged).
    hot: u64,
    /// Under the host's lock: probing hot textures and blank ones, and uploading.
    probe: [std::time::Duration; 2],
    upload: std::time::Duration,
    /// What the reading took the submission, without the host's lock.
    prepare: std::time::Duration,
    /// On the reading threads, summed: reading title memory, hashing it, detiling it.
    read: std::time::Duration,
    hash: std::time::Duration,
    detile: std::time::Duration,
}

impl SyncCost {
    fn log(self) {
        static SUM: std::sync::Mutex<Option<(std::time::Instant, [u64; 14])>> = std::sync::Mutex::new(None);
        let mut sum = SUM.lock().unwrap_or_else(|e| e.into_inner());
        let (since, s) = sum.get_or_insert_with(|| (std::time::Instant::now(), [0; 14]));
        let us = |d: std::time::Duration| d.as_micros() as u64;
        let add = [
            self.batches, self.textures, self.first, self.unchanged, self.blank, self.unreadable, self.bytes,
            us(self.probe[0]), us(self.probe[1]), us(self.upload), us(self.prepare), us(self.read), us(self.hash), us(self.detile),
        ];
        s.iter_mut().zip(add).for_each(|(s, a)| *s += a);
        static HOT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        if self.hot > 0 {
            HOT.store(self.hot, std::sync::atomic::Ordering::Relaxed);
        }
        if since.elapsed().as_secs() >= 1 {
            let take = |a: &std::sync::atomic::AtomicU64| a.swap(0, std::sync::atomic::Ordering::Relaxed);
            tracing::info!(
                "texture sync in the last {:.1} s: {} batches, {} textures ({} for the first time, {} unchanged, {} blank, {} partly unreadable), {} MiB; under the lock: probes of {} hot textures {} ms ({} read whole, {} MiB), of blank ones {} ms, uploads {} ms; off it: {} ms (on its threads: read {} ms, hash {} ms, detile {} ms); placements replaced {} ({} by one that was there before)",
                since.elapsed().as_secs_f32(), s[0], s[1], s[2], s[3], s[4], s[5], s[6] >> 20,
                take(&HOT), s[7] / 1000, take(&WHOLE_PROBES[0]), take(&WHOLE_PROBES[1]) >> 20, s[8] / 1000, s[9] / 1000,
                s[10] / 1000, s[11] / 1000, s[12] / 1000, s[13] / 1000,
                take(&REPLACED[0]), take(&REPLACED[1])
            );
            *sum = None;
        }
    }
}

fn keys_to_sync(unloaded: &HashSet<usize>, dirty: &HashSet<usize>) -> Vec<usize> {
    unloaded.iter().chain(dirty.difference(unloaded)).copied().collect()
}

/// The textures of a title host, by title resource key.
pub type Textures<B> = HashMap<usize, Tracked<B>>;

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;

    #[test]
    fn sync_keys_include_overlaps_once_and_keep_unloaded_first() {
        let unloaded = (0..20_000).collect();
        let dirty = (10_000..30_000).collect();
        let mut keys = keys_to_sync(&unloaded, &dirty);

        let (first, rest) = keys.split_at_mut(unloaded.len());
        first.sort_unstable();
        rest.sort_unstable();
        assert!(first.iter().copied().eq(0..20_000));
        assert!(rest.iter().copied().eq(20_000..30_000));
    }

    fn watched(entries: impl Into<HashMap<usize, (usize, usize, bool, u64)>>) -> Watched {
        Watched::new(Some(Watch { entries: entries.into(), ..Default::default() }))
    }

    fn dirty_this_round(watched: &Watched) -> std::collections::HashSet<usize> {
        scan_round(watched, &mut [0; 2]);
        watch_in(watched, |w| std::mem::take(&mut w.dirty))
    }

    fn seen(watched: &Watched) -> HashMap<usize, blake3::Hash> {
        watch_in(watched, |w| w.seen.clone())
    }

    #[test]
    fn scanner_forgets_removed_keys_and_reports_them_when_they_reappear() {
        let bytes = [1u8, 2, 3];
        let entry = (bytes.as_ptr() as usize, bytes.len(), false, 0);
        let w = watched([(1, entry), (2, entry)]);

        assert_eq!(dirty_this_round(&w), [1, 2].into());
        assert!(dirty_this_round(&w).is_empty());

        watch_in(&w, |w| w.entries.remove(&1));
        assert!(dirty_this_round(&w).is_empty());
        assert_eq!(seen(&w), HashMap::from([(2, blake3::hash(&bytes))]));

        watch_in(&w, |w| w.entries.insert(1, entry));
        assert_eq!(dirty_this_round(&w), [1].into());
    }

    #[test]
    fn scanner_uses_the_first_render_target_hash_as_its_baseline() {
        let mut bytes = [3u8];
        let w = watched([(1, (bytes.as_ptr() as usize, bytes.len(), true, 0))]);
        assert!(dirty_this_round(&w).is_empty());
        assert_eq!(seen(&w)[&1], blake3::hash(&bytes));

        bytes[0] = 4;
        assert_eq!(dirty_this_round(&w), [1].into());
        assert_eq!(seen(&w)[&1], blake3::hash(&bytes));
    }

    /// What a submission read whole is what the scanner compares with: it reports the write after
    /// it, not that content once more.
    #[test]
    fn scanner_compares_with_what_a_submission_read() {
        let mut bytes = [3u8];
        let w = watched([(1, (bytes.as_ptr() as usize, bytes.len(), false, 0))]);
        assert_eq!(dirty_this_round(&w), [1].into());
        bytes[0] = 4;
        // A submission read it meanwhile (`sync_apply`).
        watch_in(&w, |w| w.seen.insert(1, blake3::hash(&bytes)));
        assert!(dirty_this_round(&w).is_empty());
        bytes[0] = 5;
        assert_eq!(dirty_this_round(&w), [1].into());
    }

    /// A render target released and made again under the same key (an object's address is used
    /// again) starts over: its first sight is its baseline. Compared with the old texture's hash
    /// it counted as written, and its memory was uploaded over what the GPU had drawn into it.
    #[test]
    fn scanner_starts_over_for_a_render_target_made_again_under_a_released_key() {
        let mut bytes = [3u8];
        let entry = |serial| (bytes.as_ptr() as usize, bytes.len(), true, serial);
        let (old, new) = (entry(0), entry(1));
        let w = watched([(1, old)]);
        assert!(dirty_this_round(&w).is_empty());
        assert_eq!(seen(&w)[&1], blake3::hash(&bytes));

        // Released and made again between two rounds, its memory holding something else by then.
        watch_in(&w, |w| {
            w.released(1);
            w.entries.insert(1, new);
        });
        bytes[0] = 6;
        assert!(dirty_this_round(&w).is_empty(), "the new texture's first sight is not a change");
        assert_eq!(seen(&w)[&1], blake3::hash(&bytes));
        bytes[0] = 7;
        assert_eq!(dirty_this_round(&w), [1].into(), "a write into it after that is");
    }

    /// Every mip, array slice and depth slice goes up and comes back where it was: 1D, 2D and 3D,
    /// element sizes 1 to 8, the shapes a world run names (#46).
    #[test]
    fn download_undoes_upload_for_every_mip_and_slice() {
        use victory_formats::texture::{Desc, Swizzle};
        let Ok(metal) = victory_metal::metal::Metal::new() else { return };
        use Dimension::{Tex1d, Tex2d, Tex3d};
        // (dimension, DXGI format, element bytes, width, height, depth or layers, mips)
        let shapes = [
            (Tex1d, 28, 4, 256, 1, 1, 1),
            (Tex2d, 61, 1, 128, 64, 1, 5),
            (Tex2d, 10, 8, 256, 256, 1, 9),
            (Tex2d, 41, 4, 64, 64, 4, 6),
            (Tex3d, 41, 4, 256, 256, 16, 1),
            (Tex3d, 17, 8, 64, 64, 64, 1),
            (Tex3d, 28, 4, 32, 32, 32, 6),
        ];
        let mut ran = 0;
        for (dimension, format, element_bytes, width, height, depth, mips) in shapes {
            for swizzle in [Swizzle::LINEAR, Swizzle::SW_4KB_S, Swizzle::SW_64KB_S, Swizzle::SW_64KB_D_X] {
                let what = format!("{dimension:?} {format} {width}x{height}x{depth} mips {mips} {swizzle:?}");
                let desc = Desc { dimension, width, height, depth, mips, element_bytes, block: 1, samples: 1, swizzle };
                // Not every mode exists for every shape.
                let Ok(layout) = Layout::new(&desc) else { continue };
                let mut want = vec![0u8; layout.size as usize];
                let slices = if dimension == Tex3d { 1 } else { layout.slices };
                let tiled = (0..layout.mips.len() as u32).all(|mip| {
                    let m = &layout.mips[mip as usize];
                    let row = (m.width * element_bytes) as usize;
                    let pitch = row * m.height as usize;
                    (0..slices).all(|slice| {
                        let n = pitch * layout.subresource_depth(mip).max(1) as usize;
                        let linear: Vec<u8> = (0..n).map(|i| (i % 251) as u8 ^ (mip * 37 + slice * 11) as u8).collect();
                        layout.tile(&mut want, mip, slice, &linear, row, pitch).is_ok()
                    })
                });
                if !tiled {
                    continue;
                }
                let info = TextureInfo {
                    dimension,
                    format,
                    width,
                    height,
                    depth_or_layers: depth,
                    flags: UNORDERED_ACCESS,
                    size: want.len(),
                    layout,
                    memory: std::ptr::null(),
                };
                let tex = metal.create_texture(&info.desc().unwrap()).unwrap();
                upload(&metal, &tex, &info, &want).unwrap();
                let mut got = vec![0u8; want.len()];
                download(&metal, &tex, &info, &mut got).unwrap();
                let at = got.iter().zip(&want).position(|(a, b)| a != b);
                assert!(at.is_none(), "{what}: differs first at byte {at:?}");
                eprintln!("round trip: {what}");
                ran += 1;
            }
        }
        // Linear at least, for each shape.
        assert!(ran >= shapes.len(), "{ran}");
    }

    #[test]
    fn from_host_undoes_to_host() {
        let title: Vec<u8> = (0..=255).collect();
        for format in [28, 115] {
            let mut texels = title.clone();
            to_host(format, &mut texels);
            from_host(format, &mut texels);
            assert_eq!(texels, title, "{format}");
        }
    }

    #[test]
    fn an_inaccessible_megabyte_takes_one_failed_read_and_one_region_query() {
        use victory_metal::mem::Region;
        use victory_platform::mem::{Protect, next_region, protect};

        let len = 1 << 20;
        let memory = Region::new(len, None).unwrap();
        // SAFETY: the test owns this entire mapping until the reader finishes.
        unsafe { protect(memory.ptr, len, Protect::NO_ACCESS) }.unwrap();
        let mut out = vec![0xff; len];
        let (mut reads, mut queries) = (0, 0);
        read_texture_chunk(
            memory.ptr,
            &mut out,
            false,
            |at, out| {
                reads += 1;
                crate::native::safe_read(at, out)
            },
            |at| {
                queries += 1;
                next_region(at)
            },
        );
        assert!(out.iter().all(|&b| b == 0));
        assert_eq!((reads, queries), (1, 1));
    }

    #[test]
    fn a_region_that_changes_after_the_query_still_preserves_readable_pages() {
        use victory_metal::mem::Region;
        use victory_platform::mem::{Protect, next_region, page_size, protect};

        let page = page_size();
        let memory = Region::new(4 * page, None).unwrap();
        // SAFETY: the test owns all four pages and writes them before changing protection.
        unsafe {
            memory.ptr.write_bytes(0x72, 4 * page);
            protect(memory.ptr, page, Protect::NO_ACCESS).unwrap();
        }
        let mut out = vec![0xff; 4 * page - 2];
        read_texture_chunk(
            memory.ptr.wrapping_add(1),
            &mut out,
            false,
            crate::native::safe_read,
            |at| {
                let region = next_region(at);
                if at == memory.ptr.addr() + page {
                    // SAFETY: the third page is ours. Revoke access just after querying the span.
                    unsafe { protect(memory.ptr.add(2 * page), page, Protect::NO_ACCESS) }.unwrap();
                }
                region
            },
        );
        for (offset, &byte) in out.iter().enumerate() {
            let readable = matches!((offset + 1) / page, 1 | 3);
            assert_eq!(byte, if readable { 0x72 } else { 0 });
        }
    }

    #[test]
    fn a_texture_read_preserves_texels_on_both_sides_of_an_unreadable_page() {
        use victory_metal::mem::Region;
        use victory_platform::mem::{Protect, page_size, protect};

        let page = page_size();
        let memory = Region::new(3 * page, None).unwrap();
        // SAFETY: all three pages belong to this test. Protect only the middle one after filling it.
        unsafe {
            memory.ptr.write_bytes(0x31, 2 * page);
            memory.ptr.add(2 * page).write_bytes(0x72, page);
            protect(memory.ptr.add(page), page, Protect::NO_ACCESS).unwrap();
        }
        let start = memory.ptr.wrapping_add(page - 7);
        let mut expected = vec![0; 7 + page + 11];
        expected[..7].fill(0x31);
        expected[7 + page..].fill(0x72);

        let bytes = read_memory(start, expected.len());
        assert_eq!(&bytes[..7], &expected[..7]);
        assert!(bytes[7..7 + page].iter().all(|&b| b == 0));
        assert_eq!(&bytes[7 + page..], &expected[7 + page..]);
        let mut scratch = vec![0xff; expected.len()];
        assert_eq!(
            probe_hash(start as usize, expected.len(), true, &mut scratch),
            blake3::hash(&expected)
        );
    }

    #[test]
    fn probe_hash_detects_writes_in_sampled_and_whole_modes() {
        let mut mem = vec![0u8; 16 * 4096];
        let mut scratch = Vec::new();
        let h0_sampled = probe_hash(mem.as_ptr() as usize, mem.len(), false, &mut scratch);
        let h0_whole = probe_hash(mem.as_ptr() as usize, mem.len(), true, &mut scratch);
        // Flip a byte in the first sampled window.
        mem[42] = 0xAB;
        let h1_sampled = probe_hash(mem.as_ptr() as usize, mem.len(), false, &mut scratch);
        let h1_whole = probe_hash(mem.as_ptr() as usize, mem.len(), true, &mut scratch);
        assert_ne!(h0_sampled, h1_sampled);
        assert_ne!(h0_whole, h1_whole);
        // Flip a byte outside sampled windows: whole mode catches it.
        mem[42] = 0;
        mem[1000] = 0xCD;
        assert_ne!(h0_whole, probe_hash(mem.as_ptr() as usize, mem.len(), true, &mut scratch));
    }

    #[test]
    fn a_known_partial_chunk_skips_the_whole_read_until_every_page_is_readable() {
        use victory_metal::mem::Region;
        use victory_platform::mem::{Protect, next_region, page_size, protect};

        let page = page_size();
        let memory = Region::new(4 * page, None).unwrap();
        // SAFETY: the test owns all four pages; pages 0..2 hold coarse-mip data and 2..4 are unmapped/no-access.
        unsafe {
            memory.ptr.write_bytes(0x5a, 2 * page);
            protect(memory.ptr.add(2 * page), 2 * page, Protect::NO_ACCESS).unwrap();
        }

        let mut out = vec![0xff; 4 * page];
        let (mut reads, mut queries) = (0, 0);
        let whole = read_texture_chunk(
            memory.ptr,
            &mut out,
            false,
            |at, buf| {
                reads += 1;
                crate::native::safe_read(at, buf)
            },
            |at| {
                queries += 1;
                next_region(at)
            },
        );
        assert!(!whole);
        assert_eq!((reads, queries), (2, 2));
        assert!(out[..2 * page].iter().all(|&b| b == 0x5a));
        assert!(out[2 * page..].iter().all(|&b| b == 0));

        // Once marked partial, the doomed whole-span read is skipped and only readable spans are read.
        out.fill(0xff);
        (reads, queries) = (0, 0);
        let whole = read_texture_chunk(
            memory.ptr,
            &mut out,
            true,
            |at, buf| {
                reads += 1;
                crate::native::safe_read(at, buf)
            },
            |at| {
                queries += 1;
                next_region(at)
            },
        );
        assert!(!whole);
        assert_eq!((reads, queries), (1, 2));
        assert!(out[..2 * page].iter().all(|&b| b == 0x5a));
        assert!(out[2 * page..].iter().all(|&b| b == 0));

        // SAFETY: restore access to pages 2..4 (fine mips mapped) and fill them.
        unsafe {
            protect(memory.ptr.add(2 * page), 2 * page, Protect::READ_WRITE).unwrap();
            memory.ptr.add(2 * page).write_bytes(0xa5, 2 * page);
        }
        out.fill(0xff);
        (reads, queries) = (0, 0);
        let whole = read_texture_chunk(
            memory.ptr,
            &mut out,
            true,
            |at, buf| {
                reads += 1;
                crate::native::safe_read(at, buf)
            },
            |at| {
                queries += 1;
                next_region(at)
            },
        );
        assert!(whole, "once every page is readable again, the partial flag clears");
        assert_eq!((reads, queries), (1, 1));
        assert!(out[..2 * page].iter().all(|&b| b == 0x5a));
        assert!(out[2 * page..].iter().all(|&b| b == 0xa5));
    }
}
