use crate::JsBufferView;
use nalgebra::Matrix4;

/// Sets the object transform as the CSS style property.
#[allow(unsafe_code)]
pub fn set_object_transform(matrix:&Matrix4<f32>) {
    // Views to WASM memory are only valid as long the backing buffer isn't
    // resized. Check documentation of IntoFloat32ArrayView trait for more
    // details.
    unsafe {
        matrix.js_buffer_view();
        // js::set_object_transform(&dom,&matrix_array);
    }
}
