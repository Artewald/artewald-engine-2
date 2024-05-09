

// The manager needs (per shader):
// - all vertices (per object)
// - all indices (per object)
// - all textures (if the shader is set up to have multiple textures per object we need to have multiple texture arrays)
// - all uniform buffer data that is the same for the object type (not dynamic)
// - all "uniform buffer" data (if the shader is set up to have multiple uniform buffers per object we need to have multiple shader storage buffers)
//     - This is the per instance object data (UNIFORM_BUFFER_DYNAMIC)
pub struct ObjectManager {
    
}