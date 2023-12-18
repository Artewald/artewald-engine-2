use nalgebra_glm as glm;

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct UniformBufferObject {
    pub model: glm::Mat4,
    pub view: glm::Mat4,
    pub proj: glm::Mat4,
}
