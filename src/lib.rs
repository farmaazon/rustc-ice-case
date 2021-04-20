#![allow(dead_code)]

#![warn(missing_copy_implementations)]
#![warn(missing_debug_implementations)]
#![warn(missing_docs)]
#![warn(trivial_casts)]
#![warn(trivial_numeric_casts)]
#![warn(unsafe_code)]
#![warn(unused_import_braces)]
#![warn(unused_qualifications)]
#![recursion_limit="512"]

pub mod dom;

use nalgebra::*;

/// Class for buffer items, like `f32` or `Vector<f32>`.
///
/// WebGL buffers contain primitive values only, so for example, two `Vector3<f32>` are represented
/// as six `f32` values. This trait defines fast conversions (views) for the underlying flat data
/// storage.
pub trait Storable: Copy + Sized {

    /// The primitive type which this type is build of. In case of the most primitive types, like
    /// `f32` this type may be set to itself.
    type Cell: Storable;

    /// Converts from a buffer slice to a slice of items.
    fn slice_to_items(buffer: &[Self]) -> &[Self::Cell];
}


// === Type Families ===

/// Item accessor.
pub type Cell <T> = <T as Storable>::Cell;


// === Instances ===

impl Storable for f32 {
    type Cell = Self;
    fn slice_to_items       (buffer: &    [Self]) -> &    [Self::Cell] { buffer }
}

impl<T:Storable<Cell=T>,R,C> Storable for MatrixMN<T,R,C>
    where T : Scalar + Storable,
          R : DimName,
          C : DimName,
          DefaultAllocator: nalgebra::allocator::Allocator<T,R,C>,
          <DefaultAllocator as nalgebra::allocator::Allocator<T,R,C>>::Buffer:Copy,  {
    type Cell = T;

    #[allow(unsafe_code)]
    fn slice_to_items(buffer: &[Self]) -> &[Self::Cell] {
        // This code casts slice to matrix. This is safe because `MatrixMN`
        // uses `nalgebra::Owned` allocator, which resolves to array defined as
        // `#[repr(C)]` under the hood.
        let len = buffer.len() * <R as DimName>::dim() * <C as DimName>::dim();
        unsafe { std::slice::from_raw_parts(buffer.as_ptr().cast(), len) }
    }
}

/// Extension method for viewing into wasm's linear memory.
#[allow(unsafe_code)]
pub trait JsBufferView {
    /// Create a JS typed array which is a view into wasm's linear memory at the slice specified.
    ///
    /// This function returns a new typed array which is a view into wasm's memory. This view does
    /// not copy the underlying data.
    ///
    /// # Safety
    ///
    /// Views into WebAssembly memory are only valid so long as the backing buffer isn't resized in
    /// JS. Once this function is called any future calls to `Box::new` (or malloc of any form) may
    /// cause the returned value here to be invalidated. Use with caution!
    ///
    /// Additionally the returned object can be safely mutated but the input slice isn't guaranteed
    /// to be mutable.
    ///
    /// Finally, the returned object is disconnected from the input slice's lifetime, so there's no
    /// guarantee that the data is read at the right time.
    fn js_buffer_view(&self) -> usize;
}


// === Instances ===

#[allow(unsafe_code)]
impl JsBufferView for [f32] {
    fn js_buffer_view(&self) -> usize {
        4
    }
}


#[allow(unsafe_code)]
impl<T: Storable<Cell=T>,R,C> JsBufferView for [MatrixMN<T,R,C>]
    where
          T : Scalar + Storable,
          R : DimName,
          C : DimName,
          DefaultAllocator: nalgebra::allocator::Allocator<T,R,C>,
          <DefaultAllocator as nalgebra::allocator::Allocator<T,R,C>>::Buffer:Copy,
          MatrixMN<T,R,C>         : Storable,
          [Cell<MatrixMN<T,R,C>>] : JsBufferView {
    fn js_buffer_view(&self) -> usize {
        <MatrixMN<T,R,C> as Storable>::slice_to_items(self).js_buffer_view()
    }
}

#[allow(unsafe_code)]
impl<T: Storable<Cell=T>,R,C> JsBufferView for MatrixMN<T,R,C>
    where T : Scalar + Storable,
          R : DimName,
          C : DimName,
          DefaultAllocator: nalgebra::allocator::Allocator<T,R,C>,
          <DefaultAllocator as nalgebra::allocator::Allocator<T,R,C>>::Buffer:Copy, {
    fn js_buffer_view(&self) -> usize {
        4
    }
}
