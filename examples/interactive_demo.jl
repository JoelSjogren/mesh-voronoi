# Interactive 2D demo: click to place points, drag to place segments, watch
# the exact generalized-Voronoi output complex update live, and hover over a
# cell to see the constraints that define it. Pure OpenGL: a raw GLFW window
# and a hand-written fragment shader do all the drawing -- no plotting
# library, no scene graph. The picture itself is drawn by the shader
# evaluating every point/segment feature's distance formula directly, per
# pixel, on the GPU -- not a polygon approximation -- so curved (parabolic)
# bisectors render perfectly smoothly at any zoom.
#
# Uses MeshVoronoi (the dimension-generic package) exclusively -- the old
# half-edge/DCEL prototype (kept for reference under first-prototype/) is
# deprecated and not used anywhere here. The construction backend feeds
# the shader a flat list of features
# (quadratic + validity region per input sub-simplex) that the background
# is painted from directly, pixel by pixel -- and hover highlighting (which
# cell/edge/vertex the cursor is over) is computed the exact same way, by
# the shader, per pixel, off that same feature list (`hover_winner_indices`
# finds which features are tied at the cursor on the CPU, once; the shader
# tints any pixel whose own winner matches). This deliberately does *not*
# reconstruct the hovered cell's geometry from `cx` and triangulate it --
# an earlier version did, and that CPU-side approach was the source of a
# long, genuinely deep bug-chase (documented in triangulation-difficulties.md)
# that a pure per-pixel shader computation can't have, since it's the exact
# same computation already proven correct for the main diagram. `cx` itself
# is still what the construction *builds*, just no longer what hovering
# *queries* -- see the interactive demo's own `winners_at`-based stdout
# print, which never used `cx` for this either.
#
# Launch from the command line:
#   julia --project=/home/joel/claude/mesh-voronoi /home/joel/claude/mesh-voronoi/examples/interactive_demo.jl
#
# Controls:
#   - Left-click empty space: add an isolated point.
#   - Left-click-drag from one spot to another: add a segment between them
#     (each end snaps to an existing vertex if you land close to one).
#   - Right-click: clear the canvas and start over.
#   - Scroll: zoom in/out, centered on the cursor.
#   - A small semitransparent box follows the cursor; the output cell it's
#     in gets a white tint, and the specific input point/segment currently
#     winning there gets highlighted in yellow -- separate indicators,
#     since they come from separate queries (see above). Get close enough
#     to a shared boundary between two cells and a thin cyan line traces
#     it instead -- boundaries have a small hover "thickness" of their
#     own, since a boundary's label (a genuine tie) is exactly the
#     interesting information a zero-width interface could otherwise never
#     show. Get close enough to a vertex where three or more cells meet at
#     once and a small cyan blob marks it instead -- a multi-way tie no
#     single incident edge can show on its own (each edge only ever
#     carries the 2-way tie between the two cells it separates). All three
#     are the exact same underlying mechanism (the shader tinting a pixel
#     when the feature(s) tied at the cursor are also (nearly) tied right
#     there), not three different code paths.
#   - A small text box near the cursor shows the hovered output subcell's
#     own raw `cx` node id alongside its label (e.g. "C14={v7}", a
#     debugging reference back into `cx.nodes`), plus the winning
#     feature's own mindist -- the minimum distance from that subcell's
#     own geometric extent (not wherever the cursor happens to sit within
#     or near it) to its winning site -- rendered with a self-contained
#     bitmap font (no font file, no FreeType: a hardcoded 5x7 glyph table
#     rasterized to a texture on the fly). Nothing about hover is printed
#     to the terminal; only `status` (load/save/error messages) does that.
#   - R: refine -- cut every current segment in half at its midpoint.
#   - S: save the current drawing as a named preset (prompts on stdin).
#   - L: list every available preset and load one (prompts on stdin) --
#     the built-ins (`BUILTIN_PRESETS`, e.g. "house", "benchmark") lettered
#     a, b, c, ..., then anything saved to disk numbered 1, 2, 3, .... One
#     discoverable menu for everything, rather than a dedicated key per
#     built-in preset.
#
# Every add re-runs only the one new incremental step of the construction
# (`insert_entry!`) against what's already been built -- not a full
# recompute from scratch.

using MeshVoronoi
using GLFW
using ModernGL
using LinearAlgebra
using StaticArrays
using Random

const CANVAS_LO = -8.0
const CANVAS_HI = 8.0
const SNAP_RADIUS = 0.4   # data units: both "close enough to reuse a vertex" and "close enough that a drag counts as a click"
const MAX_FEATURES = 256
const WINDOW_SIZE = 900
const POINT_SIZE_FRACTION = 0.0056   # of framebuffer width, e.g. ~5px at WINDOW_SIZE=900
const LINE_WIDTH_FRACTION = 0.0052   # of framebuffer width, e.g. ~5px at WINDOW_SIZE=900 -- almost as thick as the point marker, still visibly under it
const ZOOM_FACTOR_PER_SCROLL = 0.9   # per unit of scroll yoffset; <1 so scrolling up (positive yoffset) zooms in
const CURSOR_BOX_HALF_FRACTION = 0.012   # of the current view's half-width, so the box looks a constant size on screen regardless of zoom
const EDGE_HOVER_THICKNESS_FRACTION = 0.02   # of the current view's half-width -- how close the cursor must get to a boundary edge (world-space, scaled by zoom the same way as CURSOR_BOX_HALF_FRACTION) before it's reported as hovering the boundary rather than whichever cell technically contains the exact pixel
const CELL_HOVER_COLOR = (1.0f0, 1.0f0, 1.0f0)      # white -- a single hovered feature (the "cell" case)
const BOUNDARY_HOVER_COLOR = (0.1f0, 0.8f0, 1.0f0)  # cyan -- 2+ hovered features mutually near-tied (edge/vertex case)
const INPUT_HIGHLIGHT_COLOR = (1.0f0, 0.82f0, 0.0f0, 0.95f0)   # yellow
const PRESETS_FILE = joinpath(@__DIR__, "presets.txt")   # plain text, not JSON -- no need for a new dependency for this

# ---------------------------------------------------------------------------
# Raw GL plumbing: no GLAbstraction/GLMakie helpers here, just ModernGL's
# 1:1 bindings to the C API, so shader compilation and uniform upload are
# spelled out by hand.

function compile_shader(shader_type, source::String)
    id = glCreateShader(shader_type)
    GC.@preserve source begin
        srcs = Ptr{GLchar}[Ptr{GLchar}(pointer(source))]
        lens = GLint[sizeof(source)]
        glShaderSource(id, 1, srcs, lens)
    end
    glCompileShader(id)
    status = GLint[0]
    glGetShaderiv(id, GL_COMPILE_STATUS, status)
    if status[1] == GL_FALSE
        loglen = GLint[0]
        glGetShaderiv(id, GL_INFO_LOG_LENGTH, loglen)
        buf = Vector{UInt8}(undef, max(loglen[1], 1))
        glGetShaderInfoLog(id, loglen[1], C_NULL, buf)
        error("shader failed to compile:\n" * String(buf[1:max(loglen[1] - 1, 0)]))
    end
    return id
end

function link_program(vs, fs)
    prog = glCreateProgram()
    glAttachShader(prog, vs)
    glAttachShader(prog, fs)
    glLinkProgram(prog)
    status = GLint[0]
    glGetProgramiv(prog, GL_LINK_STATUS, status)
    if status[1] == GL_FALSE
        loglen = GLint[0]
        glGetProgramiv(prog, GL_INFO_LOG_LENGTH, loglen)
        buf = Vector{UInt8}(undef, max(loglen[1], 1))
        glGetProgramInfoLog(prog, loglen[1], C_NULL, buf)
        error("shader program failed to link:\n" * String(buf[1:max(loglen[1] - 1, 0)]))
    end
    return prog
end

# ---------------------------------------------------------------------------
# Background shader: every feature's quadratic + validity region goes in as
# uniform arrays, and the shader itself picks the per-pixel winner -- the
# GPU analogue of evaluating every feature and taking the minimum, run once
# per pixel instead of once per grid sample.

const BG_VERTEX_SRC = """
#version 330 core
out vec2 frag_uv;
void main() {
    vec2 uv = vec2(0.0, 0.0);
    if ((gl_VertexID & 1) != 0) uv.x = 1.0;
    if ((gl_VertexID & 2) != 0) uv.y = 1.0;
    // The 3 vertices (0,0)/(1,0)/(0,1) form one big triangle that overshoots
    // the screen so its visible portion covers the whole viewport; over
    // just the visible portion, `uv` only ranges over [0,0.5], so it must be
    // doubled here to get `frag_uv` covering the full [0,1] range main()
    // below assumes.
    frag_uv = uv * 2.0;
    gl_Position = vec4(uv * 4.0 - 1.0, 0.0, 1.0);
}
"""

const BG_FRAGMENT_SRC = """
#version 330 core
#define MAX_FEATURES $(MAX_FEATURES)
#define MAX_HOVER 8
in vec2 frag_uv;
out vec4 fragment_color;

uniform vec2 uCanvasLo;
uniform vec2 uCanvasHi;
uniform int uCount;
uniform int uType[MAX_FEATURES];
uniform vec2 uPoint[MAX_FEATURES];
uniform vec2 uLineN[MAX_FEATURES];
uniform float uLineD[MAX_FEATURES];
uniform int uValidType[MAX_FEATURES];
uniform vec2 uLoN[MAX_FEATURES];
uniform float uLoD[MAX_FEATURES];
uniform vec2 uHiN[MAX_FEATURES];
uniform float uHiD[MAX_FEATURES];
uniform vec3 uColor[MAX_FEATURES];

// Hover highlighting is computed right here, per pixel, against the same
// feature list the diagram itself is drawn from -- deliberately *not* by
// reconstructing a cell's polygon on the CPU and triangulating it (that
// used to be how this worked; see the interactive-demo docstring and the
// project's own notes on why that approach was dropped). `uHoverIdx` holds
// the feature indices currently tied at the cursor (found on the CPU by
// the same distance comparison this shader already does, evaluated once
// at a single point -- not by any geometry lookup); a pixel is tinted if
// its own winner is the (single) hovered feature (`uHoverCount==1`, the
// "cell" case), or if *all* of the hovered features are mutually within
// `uHoverThickness` of each other at that pixel (`uHoverCount>=2` -- the
// same mechanism naturally produces a thin line for a 2-way tie and a
// small blob for a 3+-way tie, with no separate code paths needed).
uniform int uHoverCount;
uniform int uHoverIdx[MAX_HOVER];
uniform float uHoverThickness;
uniform vec3 uHoverColor;

// The same *pair* of features can genuinely tie along more than one
// disjoint stored edge (e.g. two segments bordering each other in two
// separate places, with a third feature's own territory wedged between)
// -- so "which features are tied" alone doesn't pin down "which specific
// edge is under the cursor". Without this bound, the 2-way tie band above
// would paint *every* such location at once, not just the one actually
// hovered. `uHoverBoundA`/`uHoverBoundB` are that specific edge's own two
// endpoints (world space; only meaningful when `uHoverBounded != 0`, set
// only for the genuine edge-hover case -- a single-feature cell hover
// needs no such bound, since it isn't scanning for a tie locus at all).
uniform int uHoverBounded;
uniform vec2 uHoverBoundA;
uniform vec2 uHoverBoundB;

// `tol` is a world-space linear slack on top of the feature's own exact
// validity boundary (its `uLoN`/`uHiN` are unit normals, so the dot
// products here are already linear distances, the same units `tol`
// itself is in). The main per-pixel winner-selection loop below needs
// this vanishingly tight (an actual winner shouldn't leak outside its own
// validity region) -- but the hover tie-highlight block needs it wide,
// around `uHoverThickness`: two features of the *same* segment whose
// validity regions are deliberately complementary (e.g. "beyond an
// endpoint" and "interior", meeting exactly at the perpendicular plane
// through that endpoint with essentially zero overlap) would otherwise
// both be "valid" only on a literal zero-width line, collapsing the
// tie-highlight band to sub-pixel width -- effectively invisible -- even
// though the tie itself is completely genuine.
bool feature_valid(int i, vec2 x, float tol) {
    if (uValidType[i] == 0) return true;
    bool loOk = dot(uLoN[i], x) - uLoD[i] >= -tol;
    if (uValidType[i] == 1) return loOk;
    bool hiOk = dot(uHiN[i], x) - uHiD[i] >= -tol;
    return loOk && hiOk;
}

float feature_dist(int i, vec2 x) {
    if (uType[i] == 0) {
        return length(x - uPoint[i]);
    }
    return abs(dot(uLineN[i], x) - uLineD[i]);
}

// Unit gradient of `feature_dist(i, ·)` at `x` -- both a point's and a
// line's own (unsquared) distance field have unit-magnitude gradient
// everywhere (away from the point itself), which is what makes this
// meaningful: it's used below to turn a raw *difference in distance
// values* into an actual estimated perpendicular distance to the tie
// locus between two features, since those aren't the same thing (see the
// comment above the tie-highlight block for why that distinction matters).
vec2 feature_grad(int i, vec2 x) {
    if (uType[i] == 0) {
        vec2 d = x - uPoint[i];
        float len = length(d);
        return len > 1e-9 ? d / len : vec2(1.0, 0.0);
    }
    float side = dot(uLineN[i], x) - uLineD[i];
    return side >= 0.0 ? uLineN[i] : -uLineN[i];
}

// Whether `x` falls within the specific hovered edge's own bounded extent
// (see `uHoverBounded`'s own comment) -- `x`'s projection onto the chord
// between the edge's two endpoints, as a fraction of that chord, allowed a
// little slack (in the same `uHoverThickness` world-space units the tie
// band's own width already uses) past each end. Exact for a straight edge;
// for a curved one this is the chord, not the true arc, so it's an
// approximation -- still a hard bound instead of the previous "anywhere on
// the whole canvas" non-bound, which is what actually matters here.
bool in_hover_bound(vec2 x) {
    if (uHoverBounded == 0) return true;
    vec2 d = uHoverBoundB - uHoverBoundA;
    float len2 = dot(d, d);
    if (len2 < 1e-12) return true;
    float t = dot(x - uHoverBoundA, d) / len2;
    float slack = uHoverThickness / sqrt(len2);
    return t >= -slack && t <= 1.0 + slack;
}

void main() {
    vec2 x = uCanvasLo + frag_uv * (uCanvasHi - uCanvasLo);
    float best = 1e30;
    int bestI = -1;
    for (int i = 0; i < MAX_FEATURES; i++) {
        if (i >= uCount) break;
        if (!feature_valid(i, x, 1e-6)) continue;

        float v;
        if (uType[i] == 0) {
            vec2 d = x - uPoint[i];
            v = dot(d, d);
        } else {
            float t = dot(uLineN[i], x) - uLineD[i];
            v = t * t;
        }
        if (v < best) { best = v; bestI = i; }
    }
    if (bestI < 0) {
        fragment_color = vec4(1.0, 1.0, 1.0, 1.0);
        return;
    }
    // Both sides of a segment's own interior are genuinely the same cell
    // (its label/face carries no side information -- see the project's own
    // sub/super-cell duality notes), so, unlike an earlier version of this
    // shader, they're not shaded differently here: doing so would visually
    // imply a boundary that doesn't actually exist in the underlying
    // complex.
    vec3 base = uColor[bestI];

    if (uHoverCount == 1) {
        if (bestI == uHoverIdx[0]) {
            base = mix(base, uHoverColor, 0.35);
        }
    } else if (uHoverCount >= 2) {
        bool in_set = false;
        for (int k = 0; k < uHoverCount; k++) {
            if (bestI == uHoverIdx[k]) in_set = true;
        }
        if (in_set) {
            bool all_valid = true;
            float dmin = 1e30;
            float dmax = -1e30;
            vec2 grads[MAX_HOVER];
            for (int k = 0; k < uHoverCount; k++) {
                int fi = uHoverIdx[k];
                if (!feature_valid(fi, x, uHoverThickness)) { all_valid = false; break; }
                float dd = feature_dist(fi, x);
                dmin = min(dmin, dd);
                dmax = max(dmax, dd);
                grads[k] = feature_grad(fi, x);
            }
            if (all_valid) {
                // `dmax - dmin` alone is *not* a uniform-width band in
                // world space: how fast it grows as you move away from the
                // true tie locus depends on the local angle between the
                // tied features' own gradients, which varies along a
                // boundary and, in the two-feature case, is only constant
                // by symmetry exactly at the midpoint -- and varies most
                // sharply of all right where three or more bisectors
                // converge (a vertex), which is exactly where a raw
                // `dmax - dmin` threshold looked wildly non-uniform. This
                // is a standard implicit-surface distance estimate
                // (`|f(x)| / |∇f(x)|`, generalized to "how much does the
                // tightest pairwise gap change per unit step"): dividing by
                // the *smallest* pairwise gradient-difference magnitude
                // among the hovered set gives a conservative estimate of
                // the true perpendicular distance to the tie locus, so the
                // highlighted band comes out a roughly constant world-space
                // width everywhere, triple points included.
                float min_grad_diff = 1e30;
                for (int a = 0; a < uHoverCount; a++) {
                    for (int b = a + 1; b < uHoverCount; b++) {
                        min_grad_diff = min(min_grad_diff, length(grads[a] - grads[b]));
                    }
                }
                float approx_dist = (dmax - dmin) / max(min_grad_diff, 0.05);
                if (approx_dist <= uHoverThickness && in_hover_bound(x)) {
                    base = mix(base, uHoverColor, 0.6);
                }
            }
        }
    }
    fragment_color = vec4(base, 1.0);
}
"""

# Overlay shader: draws the raw input (points/segments) in solid black on
# top of the background, given vertex positions directly in canvas (world)
# coordinates.
const OVERLAY_VERTEX_SRC = """
#version 330 core
layout(location = 0) in vec2 pos;
uniform vec2 uCanvasLo;
uniform vec2 uCanvasHi;
uniform float uPointSize;
void main() {
    vec2 ndc = (pos - uCanvasLo) / (uCanvasHi - uCanvasLo) * 2.0 - 1.0;
    gl_Position = vec4(ndc, 0.0, 1.0);
    gl_PointSize = uPointSize;
}
"""

const OVERLAY_FRAGMENT_SRC = """
#version 330 core
out vec4 fragment_color;
void main() {
    fragment_color = vec4(0.0, 0.0, 0.0, 1.0);
}
"""

# Highlight shader: a general-purpose "draw whatever's uploaded, in a given
# uniform color" pass, blended over everything drawn so far -- used for the
# three hover indicators (the semitransparent cursor box, the detected
# output subcell's fill, and the detected input point/segment's outline),
# which differ only in what geometry gets uploaded and which color/alpha is
# set before drawing.
const HIGHLIGHT_VERTEX_SRC = """
#version 330 core
layout(location = 0) in vec2 pos;
uniform vec2 uCanvasLo;
uniform vec2 uCanvasHi;
uniform float uPointSize;
void main() {
    vec2 ndc = (pos - uCanvasLo) / (uCanvasHi - uCanvasLo) * 2.0 - 1.0;
    gl_Position = vec4(ndc, 0.0, 1.0);
    gl_PointSize = uPointSize;
}
"""

const HIGHLIGHT_FRAGMENT_SRC = """
#version 330 core
out vec4 fragment_color;
uniform vec4 uColor;
void main() {
    fragment_color = uColor;
}
"""

# Screen-space shader for the hover text box: vertex positions are given
# directly in window pixel coordinates (origin top-left, y downward --
# matching GLFW's own cursor-position convention) rather than world
# coordinates, so the box stays a constant size and position relative to
# the cursor regardless of the current zoom/pan. Doubles as both the
# text's background panel (`uUseTexture=0`, solid `uColor`) and the glyph
# quad itself (`uUseTexture=1`, `uColor`'s alpha modulated by the bound
# texture's red channel, which holds per-pixel glyph coverage) so only one
# program/VAO pair is needed for both draws.
const SCREEN_VERTEX_SRC = """
#version 330 core
layout(location = 0) in vec2 pos;
layout(location = 1) in vec2 uv;
uniform vec2 uViewport;
out vec2 frag_uv;
void main() {
    vec2 ndc = vec2(pos.x / uViewport.x * 2.0 - 1.0, 1.0 - pos.y / uViewport.y * 2.0);
    gl_Position = vec4(ndc, 0.0, 1.0);
    frag_uv = uv;
}
"""

const SCREEN_FRAGMENT_SRC = """
#version 330 core
in vec2 frag_uv;
out vec4 fragment_color;
uniform sampler2D uTex;
uniform vec4 uColor;
uniform int uUseTexture;
void main() {
    float a = uColor.a;
    if (uUseTexture != 0) {
        a *= texture(uTex, frag_uv).r;
    }
    fragment_color = vec4(uColor.rgb, a);
}
"""

# ---------------------------------------------------------------------------

flatten2(vs::Vector{SVector{2,Float32}}) = isempty(vs) ? Float32[] : reduce(vcat, ([v[1], v[2]] for v in vs))
flatten3(vs::Vector{SVector{3,Float32}}) = isempty(vs) ? Float32[] : reduce(vcat, ([v[1], v[2], v[3]] for v in vs))

"""
Golden-angle hue rotation: a simple, dependency-free way to get visually
distinct colors for an a-priori-unknown number of labels.
"""
function hue_color(k::Int)
    h = mod(k * 137.508, 360.0)
    s, v = 0.55, 0.95
    c = v * s
    x = c * (1 - abs(mod(h / 60, 2) - 1))
    m = v - c
    r, g, b = if h < 60
        (c, x, 0.0)
    elseif h < 120
        (x, c, 0.0)
    elseif h < 180
        (0.0, c, x)
    elseif h < 240
        (0.0, x, c)
    elseif h < 300
        (x, 0.0, c)
    else
        (c, 0.0, x)
    end
    return SVector{3,Float32}(r + m, g + m, b + m)
end

# One flattened, GPU-ready description of a single feature: mirrors
# `GFeature` (its quadratic + validity region), plus the color assigned to
# its face-group.
struct ShaderFeature
    type::Int32               # 0 = point, 1 = line
    point::SVector{2,Float32}
    line_n::SVector{2,Float32}
    line_d::Float32
    valid_type::Int32         # 0 = whole plane, 1 = halfplane, 2 = strip
    lo_n::SVector{2,Float32}
    lo_d::Float32
    hi_n::SVector{2,Float32}
    hi_d::Float32
    color::SVector{3,Float32}
end

function shader_feature(f::GFeature{2}, color::SVector{3,Float32})
    quad = f.quad
    if quad isa AffineQuadratic{2,0,Float64}
        type, point, line_n, line_d = Int32(0), SVector{2,Float32}(quad.p), SVector(0.0f0, 0.0f0), 0.0f0
    else
        # K=1: the segment's own supporting line, anchor `quad.p` and unit
        # direction `quad.basis[:,1]` -- converted to the shader's
        # normal-form `(n·x-d)^2` by rotating the direction 90°.
        t̂ = quad.basis[:, 1]
        n = SVector(-t̂[2], t̂[1])
        type, point, line_n, line_d = Int32(1), SVector(0.0f0, 0.0f0), SVector{2,Float32}(n), Float32(dot(n, quad.p))
    end
    nv = length(f.validity)
    if nv == 0
        vtype, lo_n, lo_d, hi_n, hi_d = Int32(0), SVector(0.0f0, 0.0f0), 0.0f0, SVector(0.0f0, 0.0f0), 0.0f0
    elseif nv == 1
        vtype = Int32(1)
        lo_n, lo_d = SVector{2,Float32}(f.validity[1].n), Float32(f.validity[1].d)
        hi_n, hi_d = SVector(0.0f0, 0.0f0), 0.0f0
    else
        vtype = Int32(2)
        lo_n, lo_d = SVector{2,Float32}(f.validity[1].n), Float32(f.validity[1].d)
        hi_n, hi_d = SVector{2,Float32}(f.validity[2].n), Float32(f.validity[2].d)
    end
    return ShaderFeature(type, point, line_n, line_d, vtype, lo_n, lo_d, hi_n, hi_d, color)
end

# Groups every feature by its face (every feature sharing a face gets the
# same color, e.g. a segment's endpoint and its interior), assigning each
# group a distinct color.
function build_shader_features(feats::Vector{GFeature{2}})
    ids = Dict{Vector{Int},Int}()
    out = ShaderFeature[]
    for f in feats
        face_key = sort(collect(f.face))
        k = get!(ids, face_key, length(ids) + 1)
        push!(out, shader_feature(f, hue_color(k)))
    end
    return out
end

function set_background_uniforms!(prog, feats::Vector{ShaderFeature}, lo::SVector{2,Float64}, hi::SVector{2,Float64})
    n = min(length(feats), MAX_FEATURES)
    fs = feats[1:n]
    loc(name) = glGetUniformLocation(prog, name)
    glUniform2f(loc("uCanvasLo"), Float32(lo[1]), Float32(lo[2]))
    glUniform2f(loc("uCanvasHi"), Float32(hi[1]), Float32(hi[2]))
    glUniform1i(loc("uCount"), Int32(n))
    n == 0 && return
    glUniform1iv(loc("uType"), n, Int32[f.type for f in fs])
    glUniform2fv(loc("uPoint"), n, flatten2([f.point for f in fs]))
    glUniform2fv(loc("uLineN"), n, flatten2([f.line_n for f in fs]))
    glUniform1fv(loc("uLineD"), n, Float32[f.line_d for f in fs])
    glUniform1iv(loc("uValidType"), n, Int32[f.valid_type for f in fs])
    glUniform2fv(loc("uLoN"), n, flatten2([f.lo_n for f in fs]))
    glUniform1fv(loc("uLoD"), n, Float32[f.lo_d for f in fs])
    glUniform2fv(loc("uHiN"), n, flatten2([f.hi_n for f in fs]))
    glUniform1fv(loc("uHiD"), n, Float32[f.hi_d for f in fs])
    glUniform3fv(loc("uColor"), n, flatten3([f.color for f in fs]))
end

"""
Uploads the hover-tinting uniforms (see `BG_FRAGMENT_SRC`'s own docstring
comment for the mechanism): `hover_idx` are 1-based Julia indices into the
same feature list `set_background_uniforms!` just uploaded, converted to
the shader's 0-based array here.
"""
function set_hover_uniforms!(prog, hover_idx::Vector{Int}, thickness::Float64, color::NTuple{3,Float32};
    bound::Union{Nothing,NTuple{2,Pt{2,Float64}}}=nothing)
    loc(name) = glGetUniformLocation(prog, name)
    n = min(length(hover_idx), 8)
    glUniform1i(loc("uHoverCount"), Int32(n))
    n > 0 && glUniform1iv(loc("uHoverIdx"), n, Int32[i - 1 for i in hover_idx[1:n]])
    glUniform1f(loc("uHoverThickness"), Float32(thickness))
    glUniform3f(loc("uHoverColor"), color...)
    glUniform1i(loc("uHoverBounded"), bound === nothing ? Int32(0) : Int32(1))
    a, b = bound === nothing ? (Pt{2,Float64}(0.0, 0.0), Pt{2,Float64}(0.0, 0.0)) : bound
    glUniform2f(loc("uHoverBoundA"), Float32(a[1]), Float32(a[2]))
    glUniform2f(loc("uHoverBoundB"), Float32(b[1]), Float32(b[2]))
end

function gen_vao_vbo()
    vao = GLuint[0]
    glGenVertexArrays(1, vao)
    glBindVertexArray(vao[1])
    vbo = GLuint[0]
    glGenBuffers(1, vbo)
    glBindBuffer(GL_ARRAY_BUFFER, vbo[1])
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, C_NULL)
    glEnableVertexAttribArray(0)
    glBindVertexArray(0)
    return vao[1], vbo[1]
end

function upload!(vao, vbo, data::Vector{Float32})
    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    if isempty(data)
        glBufferData(GL_ARRAY_BUFFER, 0, C_NULL, GL_DYNAMIC_DRAW)
    else
        glBufferData(GL_ARRAY_BUFFER, sizeof(data), data, GL_DYNAMIC_DRAW)
    end
    glBindVertexArray(0)
end

# ---------------------------------------------------------------------------
# Self-contained bitmap-font text rendering: no font file, no FreeType --
# just a hardcoded 5x7 glyph table (uppercase letters, digits, and the
# handful of symbols the hover labels actually use), rasterized on the CPU
# into a single-channel texture and drawn as one textured quad via the
# screen-space shader above. Text is upper-cased before rendering (the
# glyph table only covers uppercase), which is fine for a debug/info HUD.

const GLYPH_W, GLYPH_H = 5, 7
const GLYPH_SCALE = 3   # each font pixel becomes a GLYPH_SCALE x GLYPH_SCALE screen-pixel block
const GLYPH_GAP = 1     # blank columns between characters, in font pixels
const LINE_GAP = 2      # blank rows between lines, in font pixels

const RAW_GLYPHS = Dict{Char,Vector{String}}(
    'A' => [".###.", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"],
    'B' => ["####.", "#...#", "#...#", "####.", "#...#", "#...#", "####."],
    'C' => [".####", "#....", "#....", "#....", "#....", "#....", ".####"],
    'D' => ["###..", "#..#.", "#...#", "#...#", "#...#", "#..#.", "###.."],
    'E' => ["#####", "#....", "#....", "####.", "#....", "#....", "#####"],
    'F' => ["#####", "#....", "#....", "####.", "#....", "#....", "#...."],
    'G' => [".####", "#....", "#....", "#.###", "#...#", "#...#", ".####"],
    'H' => ["#...#", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"],
    'I' => ["#####", "..#..", "..#..", "..#..", "..#..", "..#..", "#####"],
    'J' => ["..###", "...#.", "...#.", "...#.", "...#.", "#..#.", ".##.."],
    'K' => ["#...#", "#..#.", "#.#..", "##...", "#.#..", "#..#.", "#...#"],
    'L' => ["#....", "#....", "#....", "#....", "#....", "#....", "#####"],
    'M' => ["#...#", "##.##", "#.#.#", "#...#", "#...#", "#...#", "#...#"],
    'N' => ["#...#", "##..#", "#.#.#", "#..##", "#...#", "#...#", "#...#"],
    'O' => [".###.", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."],
    'P' => ["####.", "#...#", "#...#", "####.", "#....", "#....", "#...."],
    'Q' => [".###.", "#...#", "#...#", "#...#", "#.#.#", "#..#.", ".##.#"],
    'R' => ["####.", "#...#", "#...#", "####.", "#.#..", "#..#.", "#...#"],
    'S' => [".####", "#....", "#....", ".###.", "....#", "....#", "####."],
    'T' => ["#####", "..#..", "..#..", "..#..", "..#..", "..#..", "..#.."],
    'U' => ["#...#", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."],
    'V' => ["#...#", "#...#", "#...#", "#...#", "#...#", ".#.#.", "..#.."],
    'W' => ["#...#", "#...#", "#...#", "#.#.#", "#.#.#", "##.##", "#...#"],
    'X' => ["#...#", "#...#", ".#.#.", "..#..", ".#.#.", "#...#", "#...#"],
    'Y' => ["#...#", "#...#", ".#.#.", "..#..", "..#..", "..#..", "..#.."],
    'Z' => ["#####", "....#", "...#.", "..#..", ".#...", "#....", "#####"],
    '0' => [".###.", "#...#", "#..##", "#.#.#", "##..#", "#...#", ".###."],
    '1' => ["..#..", ".##..", "..#..", "..#..", "..#..", "..#..", "#####"],
    '2' => [".###.", "#...#", "....#", "...#.", "..#..", ".#...", "#####"],
    '3' => ["####.", "....#", "....#", ".###.", "....#", "....#", "####."],
    '4' => ["...#.", "..##.", ".#.#.", "#..#.", "#####", "...#.", "...#."],
    '5' => ["#####", "#....", "#....", "####.", "....#", "....#", "####."],
    '6' => [".###.", "#....", "#....", "####.", "#...#", "#...#", ".###."],
    '7' => ["#####", "....#", "...#.", "..#..", ".#...", ".#...", ".#..."],
    '8' => [".###.", "#...#", "#...#", ".###.", "#...#", "#...#", ".###."],
    '9' => [".###.", "#...#", "#...#", ".####", "....#", "....#", ".###."],
    ' ' => [".....", ".....", ".....", ".....", ".....", ".....", "....."],
    '-' => [".....", ".....", ".....", "#####", ".....", ".....", "....."],
    '=' => [".....", ".....", "#####", ".....", "#####", ".....", "....."],
    ':' => [".....", "..#..", ".....", ".....", "..#..", ".....", "....."],
    '(' => ["...#.", "..#..", ".#...", ".#...", ".#...", "..#..", "...#."],
    ')' => [".#...", "..#..", "...#.", "...#.", "...#.", "..#..", ".#..."],
    '.' => [".....", ".....", ".....", ".....", ".....", "..#..", "....."],
    ',' => [".....", ".....", ".....", ".....", "..#..", "..#..", ".#..."],
    '|' => ["..#..", "..#..", "..#..", "..#..", "..#..", "..#..", "..#.."],
    '{' => ["..##.", "..#..", ".#...", "##...", ".#...", "..#..", "..##."],
    '}' => [".##..", "..#..", "...#.", "...##", "...#.", "..#..", ".##.."],
)

function parse_glyph(rows::Vector{String})
    return ntuple(GLYPH_H) do r
        row = rows[r]
        UInt8(sum(row[c] == '#' ? (1 << (GLYPH_W - c)) : 0 for c in 1:GLYPH_W))
    end
end

const FONT = Dict{Char,NTuple{GLYPH_H,UInt8}}(c => parse_glyph(rows) for (c, rows) in RAW_GLYPHS)

"""
Rasterizes `text` (upper-cased; a character missing from `FONT` renders as
a blank cell; `\\n` starts a new line) into a single-channel 0/255
greyscale bitmap, nearest-neighbor scaled up by `GLYPH_SCALE` from the raw
5x7 font -- returns `(pixels, width, height)`, `pixels` row-major with row
0 (index range `1:width`) as the *top* of the rendered text. Multi-line
text is left-aligned; `width` is wide enough for the longest line, shorter
lines just leave the rest of their own row blank.
"""
function rasterize_text(text::AbstractString)
    lines = split(uppercase(text), '\n')
    cell_w = GLYPH_W + GLYPH_GAP
    line_h = GLYPH_H + LINE_GAP
    width = maximum(max(length(l) * cell_w - GLYPH_GAP, 1) for l in lines) * GLYPH_SCALE
    height = (length(lines) * line_h - LINE_GAP) * GLYPH_SCALE
    pixels = zeros(UInt8, width * height)
    for (li, line) in enumerate(lines)
        y0 = (li - 1) * line_h * GLYPH_SCALE
        for (i, ch) in enumerate(line)
            glyph = get(FONT, ch, FONT[' '])
            x0 = (i - 1) * cell_w * GLYPH_SCALE
            for gr in 1:GLYPH_H, gc in 1:GLYPH_W
                bit = (glyph[gr] >> (GLYPH_W - gc)) & 1
                bit == 0 && continue
                for sy in 1:GLYPH_SCALE, sx in 1:GLYPH_SCALE
                    px = x0 + (gc - 1) * GLYPH_SCALE + sx - 1
                    py = y0 + (gr - 1) * GLYPH_SCALE + sy - 1
                    pixels[py*width+px+1] = 0xff
                end
            end
        end
    end
    return pixels, width, height
end

function gen_text_texture()
    tex_ref = GLuint[0]
    glGenTextures(1, tex_ref)
    glBindTexture(GL_TEXTURE_2D, tex_ref[1])
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glBindTexture(GL_TEXTURE_2D, 0)
    return tex_ref[1]
end

function upload_text_texture!(tex, pixels::Vector{UInt8}, width::Int, height::Int)
    glBindTexture(GL_TEXTURE_2D, tex)
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, width, height, 0, GL_RED, GL_UNSIGNED_BYTE, pixels)
    glBindTexture(GL_TEXTURE_2D, 0)
end

# Like `gen_vao_vbo`, but for interleaved (pos::vec2, uv::vec2) vertices --
# what the screen-space shader's glyph/box quads need, versus the plain
# `pos`-only layout every world-space draw call uses.
function gen_text_vao_vbo()
    vao = GLuint[0]
    glGenVertexArrays(1, vao)
    glBindVertexArray(vao[1])
    vbo = GLuint[0]
    glGenBuffers(1, vbo)
    glBindBuffer(GL_ARRAY_BUFFER, vbo[1])
    stride = 4 * sizeof(Float32)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, stride, Ptr{Cvoid}(0))
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, stride, Ptr{Cvoid}(2 * sizeof(Float32)))
    glEnableVertexAttribArray(1)
    glBindVertexArray(0)
    return vao[1], vbo[1]
end

# A pixel-space quad (top-left `(x0,y0)`, size `w`x`h`) with UVs oriented so
# the quad's top edge samples the *first* row of the texture data passed to
# `upload_text_texture!` -- matching `rasterize_text`'s row-0-is-the-top
# convention.
function text_quad_data(x0, y0, w, h)
    return Float32[
        x0, y0, 0.0, 0.0,
        x0+w, y0, 1.0, 0.0,
        x0+w, y0+h, 1.0, 1.0,
        x0, y0+h, 0.0, 1.0,
    ]
end

function nearest_vertex(coords, pos)
    isempty(coords) && return nothing
    best_i, best_d = 0, Inf
    for (i, p) in enumerate(coords)
        d = norm(p - pos)
        if d < best_d
            best_i, best_d = i, d
        end
    end
    return best_d < SNAP_RADIUS ? best_i : nothing
end

# Hover query: works directly off the flat feature list, exactly the way
# the shader itself decides a winner per pixel -- the cell complex (`cx`)
# only exists to *build* this list correctly, not to answer point queries.
function winners_at(feats::Vector{GFeature{2}}, pos::SVector{2,Float64}; atol=1e-9)
    valid = [f for f in feats if is_valid(f.validity, pos)]
    isempty(valid) && return GFeature{2}[]
    ds = [sqdist(f.quad, pos) for f in valid]
    m = minimum(ds)
    return [f for (f, d) in zip(valid, ds) if d <= m + atol * max(1.0, m)]
end

"""
Like `winners_at`, but returns the *indices* into `feats` (matching the
shader's own uniform array layout one-to-one, so a caller can hand them
straight to the shader for hover tinting) of every feature within
`thickness` -- a linear, world-space distance, not a tiny numerical tie
tolerance -- of the closest one at `pos`. This deliberately reuses the
exact same feature-distance computation the shader itself does (`feats`,
`sqdist`), evaluated once at a single point, so what gets highlighted can
never disagree with what the shader actually draws -- unlike the earlier
approach (reconstructing the hovered cell's polygon from `cx` on the
CPU), this needs no geometry lookup at all.

Deliberately *not* `is_valid` (whose own tolerance is a fixed, tiny
`1e-9`) for the validity prefilter: two features of the same segment are
routinely validity-*complementary* (e.g. "beyond an endpoint" and
"interior" meet exactly at the perpendicular plane through that endpoint,
essentially zero overlap), so requiring the cursor to land within `1e-9`
of that exact line before either counts as a candidate would make this
genuine tie practically undetectable -- the mouse is never going to land
that precisely. Widening the tolerance to `thickness` here (the same
"how close counts as close enough" the rest of this function already
uses for the actual tie distance) is what makes the shader's own matching
widened check (`feature_valid`'s `tol` argument) actually reachable: if a
tied feature never makes it into `valid_idx`/`hover_idx` here in the
first place, no amount of shader-side leniency helps.
"""
function hover_winner_indices(feats::Vector{GFeature{2}}, pos::SVector{2,Float64}, thickness::Float64)
    valid_idx = [i for (i, f) in enumerate(feats) if all(r -> dot(r.n, pos) - r.d >= -thickness, f.validity)]
    isempty(valid_idx) && return Int[]
    # `sqdist` is a quadratic evaluation, not a literally-squared quantity --
    # right at (or extremely close to) a genuine tie, floating-point
    # cancellation can push it a hair below zero (e.g. -1.1e-16) even though
    # the true value is exactly 0; `sqrt` of a negative Real throws, so clamp
    # first rather than let a real tie point crash the hover query.
    ds = [sqrt(max(0.0, sqdist(feats[i].quad, pos))) for i in valid_idx]
    m = minimum(ds)
    return [valid_idx[k] for (k, d) in enumerate(ds) if d - m <= thickness]
end

"""
The exact minimum of `sqdist(q, ·)` along the straight segment from `a` to
`b`. `sqdist` restricted to a line is itself a quadratic in the
parameter `t` (its own quadratic part, `to_quadric(q).M`, is always
positive semi-definite -- a projector -- so this parabola always opens
upward): the true minimum is either at that parabola's own unconstrained
vertex (clamped into `[0,1]`, if the segment's direction has any
component outside the site's own subspace) or, when the segment runs
entirely *within* that subspace (the quadratic term vanishes -- e.g. a
segment along a line-type site's own supporting line), constant along
the whole segment, so any point on it will do.
"""
function min_sqdist_on_segment(q::AffineQuadratic{2,K,Float64}, a::Pt{2,Float64}, b::Pt{2,Float64}) where {K}
    quad = to_quadric(q)
    d = b - a
    Md = quad.M * d
    A = dot(d, Md)
    B = 2 * dot(a, Md) + 2 * dot(quad.b, d)
    t = A > 1e-14 ? clamp(-B / (2A), 0.0, 1.0) : 0.0
    return max(0.0, evaluate(quad, a + t * d))
end

"""
mindist: the minimum, over every point in `dim`-dimensional subcell `id`'s
own geometric extent, of the winning (tied) feature's own distance to
that point -- the diagram's own canonical distance function (the thing
that's minimized everywhere to decide who wins where), restricted to and
minimized over the specific hovered subcell, not evaluated at wherever
the cursor itself happens to sit within or near it. A vertex's extent is
the single point itself; an edge's is its (possibly curved) arc, walked
as its own tessellated polyline (`edge_polyline`) and minimized exactly
along each straight sub-segment (`min_sqdist_on_segment`); a cell's is
its full 2D region -- exactly 0 if the winning *point* feature's own site
lies inside the cell (checked directly, since that's the common,
correctly-constructed case: a feature's own site is normally inside its
own territory), otherwise (or for a line-type winner, where "inside"
doesn't simply apply) the minimum over the cell's boundary polygon, same
per-edge minimization as the edge case. A well-formed cell's own mindist
should therefore come out at or near 0; a meaningfully nonzero value on a
plain cell (not an edge/vertex tie, where nonzero is completely normal)
is a red flag that the cell doesn't actually reach its own winning site --
exactly the shape of bug this session's construction issues have taken.
"""
function subcell_mindist(cx::CellComplex{2}, feats::Vector{GFeature{2}}, kind::Symbol, id::Int)
    face = first(cx.nodes[id].label)
    f = only(ff for ff in feats if ff.face == face)
    if kind == :vertex
        return sqrt(max(0.0, sqdist(f.quad, cx.nodes[id].point)))
    end
    pts = kind == :edge ? edge_polyline(cx, id) : polygon_vertices_2d(cx, id)
    if kind == :cell && length(f.face) == 1 && point_in_polygon_2d(pts, f.quad.p)
        return 0.0
    end
    n = length(pts)
    d2 = minimum(min_sqdist_on_segment(f.quad, pts[i], pts[kind == :cell ? mod1(i + 1, n) : i+1]) for i in 1:(kind == :cell ? n : n - 1))
    return sqrt(max(0.0, d2))
end

function describe_face(face::Set{Int})
    return length(face) == 1 ? "vertex $(only(face))" : "segment interior $(join(sort(collect(face)), "-"))"
end

function describe_winners(ws::Vector{GFeature{2}})
    isempty(ws) && return "(outside the diagram)"
    prefix = length(ws) > 1 ? "TIE: " : ""
    return prefix * join([describe_face(f.face) for f in ws], "   |   ")
end

# A single face, compactly: "v2" for a vertex, "e4-5" for a segment
# interior -- the on-screen hover box's building block, once space (a
# small window, a many-way tie) makes `describe_face`'s full words too
# wide to be worth it there.
function compact_face(face::Set{Int})
    idxs = sort(collect(face))
    return length(idxs) == 1 ? "v$(idxs[1])" : "e" * join(idxs, "-")
end

# Reads straight off a `cx` cell's own `label` (a `Set{Set{VertexIdx}}`)
# instead of the flat feature list, so what the on-screen hover box shows
# is the output complex's own bookkeeping (post merge-fix), not a
# re-derivation from `feats` -- rendered as compact set notation
# (`{v2, v3}`, not "TIE: vertex 2   |   vertex 3") since the box itself
# already makes clear this is a label; spelling that out in the text too
# is just noise. Wraps onto multiple lines once there'd be more than a
# couple of tied faces, rather than growing one line arbitrarily wide.
function compact_label(label::Label; max_line_chars::Int=28)
    isempty(label) && return ""
    faces = sort([sort(collect(f)) for f in label])
    parts = [compact_face(Set(f)) for f in faces]
    lines = String[]
    cur = "{"
    for (i, p) in enumerate(parts)
        piece = i == 1 ? p : ", " * p
        if cur != "{" && length(cur) + length(piece) > max_line_chars
            push!(lines, cur)
            cur = "  " * p
        else
            cur *= piece
        end
    end
    push!(lines, cur * "}")
    return join(lines, "\n")
end

# ---------------------------------------------------------------------------
# Presets: a preset is a `Vector` of `(:point, p)` / `(:segment, pa, pb)`
# entries -- deliberately index-free (unlike the live `entries` state, whose
# `(:point, p, idx)` / `(:segment, pa, pb, idxa, idxb)` entries carry vertex
# indices that only make sense for *that* drawing) so a preset can be
# replayed into a fresh, empty complex and get its own fresh indices.

# "House" always means a house with a chimney: a square base, a triangular
# roof, and a small chimney rectangle whose sides cross the right roof
# slope (so it visibly pokes through, not just sits beside it).
#
# The unperturbed, "obvious" numbers -- deliberately kept even though this
# currently hits the still-open "curved bisector crossing one edge twice"
# gap (see the dashboard's /reports/curved-twice write-up and the task
# tracking it) partway through the roof. An earlier version of this preset
# quietly swapped in a shallower roof and a 2-point chimney that happened
# to avoid the gap -- that was the wrong call: it hid a real limitation
# behind a lucky choice of numbers instead of surfacing it. If this
# doesn't work, it doesn't work; that's the honest state of the
# construction right now, not something to paper over here.
const HOUSE_PRESET = [
    (:segment, SVector(-3.0, -3.0), SVector(3.0, -3.0)),   # base
    (:segment, SVector(3.0, -3.0), SVector(3.0, 2.0)),     # right wall
    (:segment, SVector(3.0, 2.0), SVector(-3.0, 2.0)),     # top of walls
    (:segment, SVector(-3.0, 2.0), SVector(-3.0, -3.0)),   # left wall
    (:segment, SVector(-3.0, 2.0), SVector(0.0, 5.0)),     # roof, left slope
    (:segment, SVector(0.0, 5.0), SVector(3.0, 2.0)),      # roof, right slope
    (:segment, SVector(1.3, 3.0), SVector(1.3, 4.6)),      # chimney, left side
    (:segment, SVector(1.9, 3.0), SVector(1.9, 4.6)),      # chimney, right side
    (:segment, SVector(1.3, 4.6), SVector(1.9, 4.6)),      # chimney, top
]

# A large, reproducible (fixed seed, matching `examples/benchmark_bvh.jl`'s
# own configuration exactly) point cloud -- loading it interactively lets
# you feel both sides of the ~100x point-location speedup that standalone
# benchmark measures: construction alone takes roughly a minute (the same
# per-cell independent clipping cost the BVH doesn't touch), but once
# built, hovering stays fast regardless -- the shader highlights per pixel
# off the flat feature list (see `hover_winner_indices`), not by querying
# `cx`'s ~74000 stored nodes.
const BENCHMARK_PRESET = let
    Random.seed!(1)   # matches examples/benchmark_bvh.jl's own seed
    lo, hi = -200.0, 200.0
    [(:point, SVector(lo + rand() * (hi - lo), lo + rand() * (hi - lo))) for _ in 1:1400]
end

# Built-in presets always offered first (in this order) in the load menu,
# ahead of whatever's been saved to `PRESETS_FILE` -- see `do_load_prompt!`.
const BUILTIN_PRESETS = [
    "house" => HOUSE_PRESET,
    "benchmark" => BENCHMARK_PRESET,
]

entries_to_preset(entries) = [e[1] === :point ? (:point, e[2]) : (:segment, e[2], e[3]) for e in entries]

"""
Append `name => preset` to `PRESETS_FILE` in a small line-based text format
(no need for a JSON dependency for something this simple):

    ### name
    point x y
    segment x1 y1 x2 y2
    ...

Saving the same name twice just appends a second block; `load_presets_file`
reads them in order into a `Dict`, so the later one wins -- good enough for
a single-user local demo, not meant to be a real database.
"""
function save_preset(name::AbstractString, preset)
    open(PRESETS_FILE, "a") do io
        println(io, "### ", name)
        for e in preset
            if e[1] === :point
                p = e[2]
                println(io, "point ", p[1], " ", p[2])
            else
                pa, pb = e[2], e[3]
                println(io, "segment ", pa[1], " ", pa[2], " ", pb[1], " ", pb[2])
            end
        end
    end
    return nothing
end

function load_presets_file()
    presets = Dict{String,Vector{Any}}()
    isfile(PRESETS_FILE) || return presets
    name = nothing
    current = Any[]
    for raw in readlines(PRESETS_FILE)
        line = strip(raw)
        isempty(line) && continue
        if startswith(line, "### ")
            name !== nothing && (presets[name] = current)
            name = line[5:end]
            current = Any[]
        elseif startswith(line, "point ")
            parts = split(line)
            push!(current, (:point, SVector(parse(Float64, parts[2]), parse(Float64, parts[3]))))
        elseif startswith(line, "segment ")
            parts = split(line)
            push!(current, (:segment,
                SVector(parse(Float64, parts[2]), parse(Float64, parts[3])),
                SVector(parse(Float64, parts[4]), parse(Float64, parts[5]))))
        end
    end
    name !== nothing && (presets[name] = current)
    return presets
end

# Assembles all persistent state and GL objects the running app needs, and
# returns `(window, state)`. Split out from the render loop so the
# gesture/incremental-step logic can be exercised programmatically (see the
# bottom of this file) without pumping a real event loop -- `state` exposes
# the mutable pieces needed for that.
function build_app(; visible=true)
    coords = SVector{2,Float64}[]
    entries = Any[]   # (:point, p, idx) or (:segment, pa, pb, idxa, idxb), in insertion order
    lo0, hi0 = SVector(CANVAS_LO, CANVAS_LO), SVector(CANVAS_HI, CANVAS_HI)
    cx = init_bbox_complex(Val(2), lo0, hi0)[1]
    feats = GFeature{2}[]

    GLFW.WindowHint(GLFW.VISIBLE, visible)
    GLFW.WindowHint(GLFW.RESIZABLE, false)
    GLFW.WindowHint(GLFW.CONTEXT_VERSION_MAJOR, 3)
    GLFW.WindowHint(GLFW.CONTEXT_VERSION_MINOR, 3)
    GLFW.WindowHint(GLFW.OPENGL_PROFILE, GLFW.OPENGL_CORE_PROFILE)
    GLFW.WindowHint(GLFW.OPENGL_FORWARD_COMPAT, true)
    window = GLFW.CreateWindow(WINDOW_SIZE, WINDOW_SIZE, "mesh-voronoi-nd -- interactive demo")
    GLFW.MakeContextCurrent(window)
    GLFW.SwapInterval(1)
    glEnable(GL_PROGRAM_POINT_SIZE)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)   # needed for the semitransparent hover indicators

    bg_prog = link_program(compile_shader(GL_VERTEX_SHADER, BG_VERTEX_SRC), compile_shader(GL_FRAGMENT_SHADER, BG_FRAGMENT_SRC))
    overlay_prog = link_program(compile_shader(GL_VERTEX_SHADER, OVERLAY_VERTEX_SRC), compile_shader(GL_FRAGMENT_SHADER, OVERLAY_FRAGMENT_SRC))
    highlight_prog = link_program(compile_shader(GL_VERTEX_SHADER, HIGHLIGHT_VERTEX_SRC), compile_shader(GL_FRAGMENT_SHADER, HIGHLIGHT_FRAGMENT_SRC))
    screen_prog = link_program(compile_shader(GL_VERTEX_SHADER, SCREEN_VERTEX_SRC), compile_shader(GL_FRAGMENT_SHADER, SCREEN_FRAGMENT_SRC))
    bg_vao_ref = GLuint[0]
    glGenVertexArrays(1, bg_vao_ref)
    bg_vao = bg_vao_ref[1]
    points_vao, points_vbo = gen_vao_vbo()
    lines_vao, lines_vbo = gen_vao_vbo()
    highlight_vao, highlight_vbo = gen_vao_vbo()
    screen_vao, screen_vbo = gen_text_vao_vbo()
    text_tex = gen_text_texture()
    last_hover_text = Ref("")
    hover_text_size = Ref((0, 0))
    points_n = Ref(0)
    lines_n = Ref(0)
    test_cursor_override = Ref{Union{Nothing,Tuple{Float64,Float64}}}(nothing)

    bg_feats = Ref(ShaderFeature[])

    # The view (what portion of the fixed CANVAS_LO..CANVAS_HI domain is on
    # screen) is separate from that domain itself -- zooming never touches
    # `cx`/`feats`, only how the same underlying features get mapped to
    # pixels. Kept square (same half-extent in both dimensions) since the
    # window itself is square.
    view_center = Ref(SVector((CANVAS_LO + CANVAS_HI) / 2, (CANVAS_LO + CANVAS_HI) / 2))
    view_half = Ref((CANVAS_HI - CANVAS_LO) / 2)
    view_lo() = view_center[] .- view_half[]
    view_hi() = view_center[] .+ view_half[]

    function rebuild!()
        bg_feats[] = build_shader_features(feats)
        pts = Float32[]
        for e in entries
            e[1] === :point || continue
            p = e[2]
            append!(pts, (Float32(p[1]), Float32(p[2])))
        end
        points_n[] = length(pts) ÷ 2
        upload!(points_vao, points_vbo, pts)

        lns = Float32[]
        for e in entries
            e[1] === :segment || continue
            a, b = e[2], e[3]
            append!(lns, (Float32(a[1]), Float32(a[2]), Float32(b[1]), Float32(b[2])))
        end
        lines_n[] = length(lns) ÷ 2
        upload!(lines_vao, lines_vbo, lns)
    end

    status = Ref("Click to add a point, drag to add a segment, right-click to clear.")
    hover = Ref(describe_winners(GFeature{2}[]))
    println(status[])

    function set_status!(s)
        status[] = s
        println(s)
    end

    # `insert_entry!`/`insert_features!` mutate `cx` in place, node by node,
    # as they go -- if a later cell in the same call hits one of the
    # library's known not-yet-implemented scope limits and throws, the cells
    # already processed stay clipped/relabeled while `feats` never gets the
    # new entry appended (that only happens at the very end of a
    # successful call), leaving `cx` referencing a face `feats` doesn't
    # know about. Left alone, that corruption poisons every later add for
    # the rest of the session (a `cur_feat = only(...)` lookup elsewhere
    # finds nothing and throws too) -- so a failed attempt here is rolled
    # back to a snapshot taken just before it, rather than left in place.
    function step!(entry)
        cx_backup = deepcopy(cx)
        feats_backup = copy(feats)
        try
            insert_entry!(cx, feats, entry_feats(entry, Val(2)))
            push!(entries, entry)
            rebuild!()
        catch e
            cx = cx_backup
            feats = feats_backup
            rethrow()
        end
    end

    function commit_point!(pos)
        i = nearest_vertex(coords, pos)
        if i !== nothing
            set_status!("Point $i already exists here -- ignored.")
            return
        end
        push!(coords, pos)
        idx = length(coords)
        try
            step!((:point, pos, idx))
            set_status!("Added point $idx.")
        catch e
            set_status!("Error: " * sprint(showerror, e))
        end
    end

    function commit_segment!(pos_a, pos_b)
        ia = nearest_vertex(coords, pos_a)
        if ia === nothing
            push!(coords, pos_a)
            ia = length(coords)
        end
        ib = nearest_vertex(coords, pos_b)
        if ib === nothing
            push!(coords, pos_b)
            ib = length(coords)
        end
        if ia == ib
            set_status!("Drag start/end resolved to the same vertex -- ignored.")
            return
        end
        try
            step!((:segment, coords[ia], coords[ib], ia, ib))
            set_status!("Added segment $ia-$ib.")
        catch e
            set_status!("Error: " * sprint(showerror, e))
        end
    end

    function do_clear!()
        empty!(coords)
        empty!(entries)
        cx = init_bbox_complex(Val(2), lo0, hi0)[1]
        empty!(feats)
        set_status!("Cleared.")
        hover[] = describe_winners(GFeature{2}[])
        rebuild!()
    end

    # Cuts every current segment in half at its midpoint and rebuilds the
    # whole complex from scratch (vertex indices necessarily change, so
    # this isn't a single incremental `step!` -- it's a fresh replay of a
    # new, larger entry list through the same `step!` used for interactive
    # adds, one entry at a time, so a mid-refine construction failure rolls
    # back only that one entry rather than corrupting the whole rebuild).
    function refine!()
        old_entries = copy(entries)
        empty!(coords)
        empty!(entries)
        cx = init_bbox_complex(Val(2), lo0, hi0)[1]
        empty!(feats)

        for e in old_entries
            if e[1] === :point
                p = e[2]
                push!(coords, p)
                step!((:point, p, length(coords)))
            else
                _, pa, pb, _, _ = e
                mid = (pa + pb) / 2
                push!(coords, pa)
                ia = length(coords)
                push!(coords, mid)
                im = length(coords)
                push!(coords, pb)
                ib = length(coords)
                step!((:segment, pa, mid, ia, im))
                step!((:segment, mid, pb, im, ib))
            end
        end
        set_status!("Refined: $(count(e -> e[1] === :segment, old_entries)) segment(s) cut in half.")
    end

    # Clears the canvas and replays `preset` (an index-free entry list, see
    # `entries_to_preset`) through the same `step!` used for interactive
    # adds -- so a preset is built exactly the way a user's own drawing
    # would be, just from a fixed list instead of mouse gestures.
    function load_preset!(preset, label::AbstractString)
        empty!(coords)
        empty!(entries)
        cx = init_bbox_complex(Val(2), lo0, hi0)[1]
        empty!(feats)
        for e in preset
            if e[1] === :point
                p = e[2]
                push!(coords, p)
                step!((:point, p, length(coords)))
            else
                pa, pb = e[2], e[3]
                push!(coords, pa)
                ia = length(coords)
                push!(coords, pb)
                ib = length(coords)
                step!((:segment, pa, pb, ia, ib))
            end
        end
        set_status!("Loaded preset \"$label\".")
    end

    function draw_frame!()
        w, h = GLFW.GetFramebufferSize(window)
        glViewport(0, 0, w, h)
        glClear(GL_COLOR_BUFFER_BIT)
        lo, hi = view_lo(), view_hi()

        # Computed once here and threaded through: which features are
        # hovered (see `hover_winner_indices`'s own docstring for why this
        # needs no `cx`/geometry lookup at all) drives both the shader's own
        # per-pixel cell/edge/vertex tinting (`set_hover_uniforms!`, right
        # below) and the remaining CPU-drawn UI chrome further down
        # (`draw_hover_highlights!`) -- computed exactly once per frame so
        # both agree on the identical answer.
        pos = world_pos_from_cursor()
        thickness = EDGE_HOVER_THICKNESS_FRACTION * view_half[]
        hover_idx = hover_winner_indices(feats, pos, thickness)
        hover_color = length(hover_idx) == 1 ? CELL_HOVER_COLOR : BOUNDARY_HOVER_COLOR

        # The same pair of features can tie along more than one disjoint
        # stored edge (e.g. two segments bordering each other in two
        # separate places, with a third feature's own territory wedged
        # between) -- `hover_idx` alone can't distinguish which one is
        # actually under the cursor, so the shader's own tie-band would
        # paint all of them at once. `find_hover_target` resolves the
        # specific `cx` edge (already computed below for the text label
        # anyway, just moved up here so both agree on one lookup instead of
        # two), and its own two endpoints bound the shader's highlight to
        # just that edge. Wrapped defensively for the same reason the label
        # lookup below already is: a still-open construction bug can make
        # this throw, and a bad cell under the mouse hard-crashing the
        # whole session every frame would be worse than an unbounded
        # highlight for one frame.
        hover_kind, hover_target_id = nothing, nothing
        try
            hover_kind, hover_target_id = find_hover_target(cx, pos, thickness)
        catch e
            e isa ErrorException || rethrow()
        end
        hover_bound = nothing
        if hover_kind === :edge
            v1, v2 = cx.nodes[hover_target_id].subcells
            hover_bound = (cx.nodes[v1].point, cx.nodes[v2].point)
        end

        glUseProgram(bg_prog)
        set_background_uniforms!(bg_prog, bg_feats[], lo, hi)
        set_hover_uniforms!(bg_prog, hover_idx, thickness, hover_color; bound=hover_bound)
        glBindVertexArray(bg_vao)
        glDrawArrays(GL_TRIANGLES, 0, 3)
        glBindVertexArray(0)

        glUseProgram(overlay_prog)
        oloc(name) = glGetUniformLocation(overlay_prog, name)
        glUniform2f(oloc("uCanvasLo"), Float32(lo[1]), Float32(lo[2]))
        glUniform2f(oloc("uCanvasHi"), Float32(hi[1]), Float32(hi[2]))
        glUniform1f(oloc("uPointSize"), Float32(w * POINT_SIZE_FRACTION))
        glLineWidth(Float32(w * LINE_WIDTH_FRACTION))
        if lines_n[] > 0
            glBindVertexArray(lines_vao)
            glDrawArrays(GL_LINES, 0, lines_n[])
            glBindVertexArray(0)
        end
        if points_n[] > 0
            glBindVertexArray(points_vao)
            glDrawArrays(GL_POINTS, 0, points_n[])
            glBindVertexArray(0)
        end

        draw_hover_highlights!(w, h, lo, hi, pos, hover_idx, thickness, hover_kind, hover_target_id)
    end

    # The cell/edge/vertex highlight itself is now drawn directly by the
    # background shader, per pixel, off the same feature list it already
    # renders the diagram from (`set_hover_uniforms!`, called from
    # `draw_frame!` right alongside `set_background_uniforms!`) -- not by
    # reconstructing the hovered cell's geometry from `cx` here. What's
    # left here is the UI chrome that's genuinely CPU-appropriate: the
    # yellow "which specific input feature is winning here" marker (drawn
    # at its own exact, already-known coordinates -- no geometry
    # reconstruction involved), the cursor box, and the on-screen text
    # label (built directly from `hover_idx`/`feats`, the same inputs the
    # shader used, so the label can never disagree with the highlight).
    function draw_hover_highlights!(w, h, lo, hi, pos, hover_idx, thickness, hover_kind, hover_target_id)
        glUseProgram(highlight_prog)
        hloc(name) = glGetUniformLocation(highlight_prog, name)
        glUniform2f(hloc("uCanvasLo"), Float32(lo[1]), Float32(lo[2]))
        glUniform2f(hloc("uCanvasHi"), Float32(hi[1]), Float32(hi[2]))

        glUniform4f(hloc("uColor"), INPUT_HIGHLIGHT_COLOR...)
        for f in winners_at(feats, pos)
            facevec = sort(collect(f.face))
            if length(facevec) == 1
                p = coords[facevec[1]]
                upload!(highlight_vao, highlight_vbo, Float32[Float32(p[1]), Float32(p[2])])
                glUniform1f(hloc("uPointSize"), Float32(w * POINT_SIZE_FRACTION * 2.4))
                glBindVertexArray(highlight_vao)
                glDrawArrays(GL_POINTS, 0, 1)
                glBindVertexArray(0)
            else
                pa, pb = coords[facevec[1]], coords[facevec[2]]
                upload!(highlight_vao, highlight_vbo, Float32[Float32(pa[1]), Float32(pa[2]), Float32(pb[1]), Float32(pb[2])])
                glLineWidth(Float32(w * LINE_WIDTH_FRACTION * 2.4))
                glBindVertexArray(highlight_vao)
                glDrawArrays(GL_LINES, 0, 2)
                glBindVertexArray(0)
            end
        end

        # The cursor box itself: a small semitransparent square, sized as a
        # fraction of the current view so it looks the same size on screen
        # at any zoom level (unlike the box's own world-space extent, which
        # shrinks/grows with `view_half`).
        box_half = CURSOR_BOX_HALF_FRACTION * view_half[]
        bx, by = pos[1], pos[2]
        box_data = Float32[
            bx-box_half, by-box_half,
            bx+box_half, by-box_half,
            bx+box_half, by+box_half,
            bx-box_half, by+box_half,
        ]
        upload!(highlight_vao, highlight_vbo, box_data)
        glUniform4f(hloc("uColor"), 0.05f0, 0.05f0, 0.05f0, 0.4f0)
        glBindVertexArray(highlight_vao)
        glDrawArrays(GL_TRIANGLE_FAN, 0, 4)
        glBindVertexArray(0)

        # Outside every feature's validity region (e.g. off the edge of the
        # canvas bbox): no label to show, so the box itself is hidden rather
        # than showing placeholder text -- the cursor square above already
        # indicates where the cursor is regardless.
        if !isempty(hover_idx)
            label = Label(Set(feats[i].face for i in hover_idx))
            text = compact_label(label)
            # The actual hovered subcell (vertex/edge/cell), found the same
            # thickness-aware way `hover_idx` was -- needed both for
            # `mindist` (a property of that subcell's own geometry, not of
            # wherever the cursor happens to sit within/near it -- see
            # `subcell_mindist`'s own docstring) and for its own raw `cx`
            # node id, shown as "C<id>=" ahead of the label purely as a
            # debugging reference back into `cx.nodes` (matches whatever
            # `cx.nodes[id]` you'd inspect at the REPL). Unlike `hover_idx`
            # itself (pure feats+pos, always safe), this walks `cx`'s own
            # boundary structure (`polygon_vertices_2d` et al.), which a
            # still-open construction bug (a malformed cell whose edges
            # don't form a simple cycle -- task #43) can make throw. That
            # bug existing is one thing; the mouse merely passing over the
            # bad cell hard-crashing the whole session every frame is
            # another -- this is a debugging aid, not load-bearing, so it
            # degrades to just the label (no id, no mindist) rather than
            # taking the render loop down with it. `hover_kind`/
            # `hover_target_id` are `draw_frame!`'s own already-computed
            # `find_hover_target` result (needed there too now, to bound the
            # shader's own tie highlight -- see its call site) rather than a
            # second, separate lookup here.
            try
                if hover_kind !== nothing
                    text = "C" * string(hover_target_id) * "=" * text
                    mindist = subcell_mindist(cx, feats, hover_kind, hover_target_id)
                    text *= "\nD=" * string(round(mindist, digits=3))
                end
            catch e
                e isa ErrorException || rethrow()
            end
            render_hover_box!(text)
        end
    end

    # Draws the on-screen hover text box: a small dark panel with white
    # glyphs (possibly multiple lines -- see `rasterize_text`), offset from
    # the cursor and clamped to stay fully within the window regardless of
    # where the cursor is (flipping to the other side near an edge isn't
    # enough on its own close to a corner, or in a small window). The glyph
    # texture is only re-rasterized/re-uploaded when the text actually
    # changes -- cheap enough to do every frame regardless, but there's no
    # reason to.
    function render_hover_box!(text::AbstractString)
        ww, wh = GLFW.GetWindowSize(window)
        if text != last_hover_text[]
            pixels, tw, th = rasterize_text(text)
            upload_text_texture!(text_tex, pixels, tw, th)
            last_hover_text[] = text
            hover_text_size[] = (tw, th)
        end
        tw, th = hover_text_size[]
        pad = 6
        cxp, cyp = cursor_pixel_pos()
        x0, y0 = cxp + 14, cyp + 14
        x0 + tw + 2pad > ww && (x0 = cxp - 14 - tw - 2pad)
        y0 + th + 2pad > wh && (y0 = cyp - 14 - th - 2pad)
        # The drawn panel extends `pad` beyond the text on every side (see
        # `box` below: `x0-pad .. x0+tw+pad`), so keeping *the panel*
        # in-window means clamping `x0` itself to `[pad, ww-tw-pad]`, not
        # `[0, ww-tw-2pad]` -- that would leave the left/top pad hanging off
        # the edge even though `x0` itself looked "clamped".
        x0 = clamp(x0, pad, max(pad, ww - tw - pad))
        y0 = clamp(y0, pad, max(pad, wh - th - pad))

        glUseProgram(screen_prog)
        sloc(name) = glGetUniformLocation(screen_prog, name)
        glUniform2f(sloc("uViewport"), Float32(ww), Float32(wh))

        glUniform1i(sloc("uUseTexture"), Int32(0))
        glUniform4f(sloc("uColor"), 0.05f0, 0.05f0, 0.05f0, 0.72f0)
        box = Float32[
            x0-pad, y0-pad, 0.0, 0.0,
            x0+tw+pad, y0-pad, 0.0, 0.0,
            x0+tw+pad, y0+th+pad, 0.0, 0.0,
            x0-pad, y0+th+pad, 0.0, 0.0,
        ]
        upload!(screen_vao, screen_vbo, box)
        glBindVertexArray(screen_vao)
        glDrawArrays(GL_TRIANGLE_FAN, 0, 4)
        glBindVertexArray(0)

        glUniform1i(sloc("uUseTexture"), Int32(1))
        glUniform4f(sloc("uColor"), 1.0f0, 1.0f0, 1.0f0, 1.0f0)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, text_tex)
        glUniform1i(sloc("uTex"), Int32(0))
        upload!(screen_vao, screen_vbo, text_quad_data(x0, y0, tw, th))
        glBindVertexArray(screen_vao)
        glDrawArrays(GL_TRIANGLE_FAN, 0, 4)
        glBindVertexArray(0)
        glBindTexture(GL_TEXTURE_2D, 0)
    end

    # `GLFW.GetCursorPos`/`SetCursorPos` are tied to real OS pointer
    # tracking, which a never-shown (`visible=false`) window doesn't have --
    # readback can come back essentially unrelated to whatever was last set
    # (confirmed directly: setting (895,5) on a hidden window read back as
    # (51,713)). `test_cursor_override`, when set, lets headless
    # verification (or any other scripted use) simulate "the cursor is at
    # this pixel" without touching GLFW's cursor APIs at all -- real
    # interactive use never sets it, so this is a no-op then.
    function cursor_pixel_pos()
        test_cursor_override[] !== nothing && return test_cursor_override[]
        return GLFW.GetCursorPos(window)
    end

    function world_pos_from_cursor()
        wx, wy = GLFW.GetWindowSize(window)
        cx_, cy = cursor_pixel_pos()
        lo, hi = view_lo(), view_hi()
        x = lo[1] + (cx_ / wx) * (hi[1] - lo[1])
        y = hi[2] - (cy / wy) * (hi[2] - lo[2])   # window y grows downward; world y grows upward
        return SVector(x, y)
    end

    # Zooms by `factor` (< 1 zooms in, > 1 zooms out) around `anchor` (a
    # world position, typically the cursor's current position) -- `anchor`
    # stays under the same screen pixel after rescaling. Unbounded in both
    # directions; the shader evaluates the same feature quadratics exactly
    # regardless of view scale, so there's no inherent range limit (the
    # underlying `cx`/`feats` domain isn't touched by this at all).
    function zoom!(anchor::SVector{2,Float64}, factor::Float64)
        new_half = view_half[] * factor
        actual_factor = new_half / view_half[]
        view_center[] = anchor + (view_center[] - anchor) * actual_factor
        view_half[] = new_half
    end

    GLFW.SetScrollCallback(window, (_, xoffset, yoffset) -> begin
        zoom!(world_pos_from_cursor(), ZOOM_FACTOR_PER_SCROLL^yoffset)
    end)

    function do_refine!()
        try
            refine!()
        catch e
            set_status!("Error during refine: " * sprint(showerror, e))
        end
    end

    function do_save_prompt!()
        print("Save current drawing as preset name: ")
        flush(stdout)
        name = strip(readline())
        if isempty(name)
            set_status!("Save cancelled (empty name).")
            return
        end
        save_preset(name, entries_to_preset(entries))
        set_status!("Saved preset \"$name\".")
    end

    # One discoverable menu lists every available preset, rather than a
    # dedicated, easy-to-forget key binding per built-in (an earlier version
    # had `H` load "house" specifically and nothing else this way). The
    # built-ins (`BUILTIN_PRESETS`, e.g. "house", "benchmark") are lettered
    # a, b, c, ... since they're fixed and few; anything saved to disk is
    # numbered 1, 2, 3, ... (in the order `load_presets_file` returns them)
    # as before, since that list grows over a session and numbers read more
    # naturally as "the thing I just saved" than a letter would.
    function do_load_prompt!()
        saved = load_presets_file()
        saved_names = sort(collect(keys(saved)))
        if isempty(BUILTIN_PRESETS) && isempty(saved_names)
            set_status!("No presets available.")
            return
        end
        println("Presets:")
        for (i, (name, _)) in enumerate(BUILTIN_PRESETS)
            println("  $(Char('a' + i - 1)). $name")
        end
        for (i, name) in enumerate(saved_names)
            println("  $i. $name")
        end
        print("Load which? ")
        flush(stdout)
        choice = lowercase(strip(readline()))
        name, preset = if length(choice) == 1 && 'a' <= choice[1] < Char('a' + length(BUILTIN_PRESETS))
            BUILTIN_PRESETS[Int(only(choice)) - Int('a') + 1]
        else
            idx = tryparse(Int, choice)
            if idx === nothing || idx < 1 || idx > length(saved_names)
                set_status!("Load cancelled (invalid choice).")
                return
            end
            saved_names[idx] => saved[saved_names[idx]]
        end
        try
            load_preset!(preset, name)
        catch e
            set_status!("Error loading preset \"$name\": " * sprint(showerror, e))
        end
    end

    rebuild!()
    state = (; coords, entries, commit_point!, commit_segment!, do_clear!, status, hover,
        do_refine!, do_save_prompt!, do_load_prompt!,
        feats=(() -> feats), draw_frame!, world_pos_from_cursor, window, zoom!,
        view_lo, view_hi, view_center, view_half,
        bg_feats=(() -> bg_feats[]), cx=(() -> cx), test_cursor_override)
    return window, state
end

# Runs the interactive event/render loop: polls input each iteration
# (rather than registering GLFW callbacks), detects left-button
# press/release edges to distinguish a click (add a point) from a drag (add
# a segment), and a right-button press to clear. Hover info is shown
# on-screen only (the shader-tinted cell/boundary plus the hover text box --
# see `draw_hover_highlights!`), not printed to the terminal.
function run!(window, state)
    left_was_down = false
    drag_start = Ref{Union{Nothing,SVector{2,Float64}}}(nothing)
    right_was_down = false
    key_was_down = Dict(GLFW.KEY_R => false, GLFW.KEY_S => false, GLFW.KEY_L => false)

    while !GLFW.WindowShouldClose(window)
        GLFW.PollEvents()

        left_down = GLFW.GetMouseButton(window, GLFW.MOUSE_BUTTON_LEFT) == GLFW.PRESS
        if left_down && !left_was_down
            drag_start[] = state.world_pos_from_cursor()
        elseif !left_down && left_was_down && drag_start[] !== nothing
            start = drag_start[]
            drag_start[] = nothing
            stop = state.world_pos_from_cursor()
            if norm(stop - start) < SNAP_RADIUS
                state.commit_point!(start)
            else
                state.commit_segment!(start, stop)
            end
        end
        left_was_down = left_down

        right_down = GLFW.GetMouseButton(window, GLFW.MOUSE_BUTTON_RIGHT) == GLFW.PRESS
        right_down && !right_was_down && state.do_clear!()
        right_was_down = right_down

        for (key, action) in ((GLFW.KEY_R, state.do_refine!),
            (GLFW.KEY_S, state.do_save_prompt!), (GLFW.KEY_L, state.do_load_prompt!))
            down = GLFW.GetKey(window, key) == GLFW.PRESS
            down && !key_was_down[key] && action()
            key_was_down[key] = down
        end

        state.draw_frame!()
        GLFW.SwapBuffers(window)
    end
    GLFW.DestroyWindow(window)
end

if abspath(PROGRAM_FILE) == @__FILE__
    GLFW.Init()
    window, state = build_app()
    run!(window, state)
    GLFW.Terminate()
end
