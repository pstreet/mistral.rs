// Release freed heap pages back to the OS. Harmless when the global
// allocator is not mimalloc (its heaps are then empty).
pub(crate) fn collect() {
    // SAFETY: mi_collect only reclaims free blocks, never live ones.
    unsafe {
        libmimalloc_sys::mi_collect(true);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collect_runs() {
        collect();
    }
}
