#version 450

layout(location = 0) in vec4 inPos;

layout(binding = 0) uniform LightUniformBufferObject{
	mat4 model;
	mat4 view;
	mat4 proj;
	vec4 lightPos;
	vec4 cameraPos;
}ubo;

void main(){
	gl_Position = ubo.proj * ubo.view * ubo.model * inPos;
}