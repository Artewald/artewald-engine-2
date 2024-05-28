#version 450

// If you have double numbers the location has to be increase by 2 instead of 1
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec2 texCoord;

layout(set = 0, binding = 0) buffer InstanceData {
    mat4 model[];
} instanceData;
layout(set = 0, binding = 1) uniform ObjectTypeData {
    mat4 view_proj;
} objectTypeData;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 fragTexCoord;

void main() {
    gl_Position = objectTypeData.view_proj * instanceData.model[gl_InstanceIndex] * vec4(inPosition, 1.0);
    fragColor = inColor;
    fragTexCoord = texCoord;
}