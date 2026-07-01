"""GLSL shader sources for all render passes (PRD §12.3)."""

# ---------------------------------------------------------------------------
# Flat-color shader — roads, route strip, ground plane, marker
# ---------------------------------------------------------------------------

FLAT_VERT = """
#version 330 core
in vec3 in_position;
uniform mat4 mvp;
void main() {
    gl_Position = mvp * vec4(in_position, 1.0);
}
"""

FLAT_FRAG = """
#version 330 core
uniform vec4 color;
out vec4 f_color;
void main() {
    f_color = color;
}
"""

# ---------------------------------------------------------------------------
# Building diffuse shader — buildings (PRD §12.3, NFR-VIS-005)
# Uses abs(dot) for two-sided diffuse so inward wall normals still appear lit.
# ---------------------------------------------------------------------------

BUILDING_VERT = """
#version 330 core
in vec3 in_position;
in vec3 in_normal;
uniform mat4 mvp;
out vec3 v_normal;
void main() {
    gl_Position = mvp * vec4(in_position, 1.0);
    v_normal = in_normal;
}
"""

BUILDING_FRAG = """
#version 330 core
in vec3 v_normal;
uniform vec4 color;
out vec4 f_color;
void main() {
    vec3 light = normalize(vec3(0.4, 1.0, 0.3));
    float diff = abs(dot(normalize(v_normal), light));
    float ambient = 0.35;
    float brightness = ambient + (1.0 - ambient) * diff;
    f_color = vec4(color.rgb * brightness, color.a);
}
"""
