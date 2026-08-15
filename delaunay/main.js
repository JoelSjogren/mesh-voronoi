import * as THREE from 'three';
import { Delaunay } from 'd3-delaunay';

// ---------- state ----------
// points: { id, kind: 'free'|'slider', x, y, segId, t }
//   free points store x,y directly.
//   slider points store segId + t (0..1 along the segment) instead.
//   a segment may have any number of slider points attached to it.
let points = [];
let segments = []; // { id, ax, ay, bx, by }
let nextId = 1;
let paused = false;
let showCircumcircles = false;
let lastMovedSliderId = null;

// hit-test thresholds are in screen pixels; divide by `zoom` to get world units
const POINT_HIT_R = 9;
const SEG_HIT_R = 7;
const MIN_DRAG = 4;

let dragState = null;

// ---------- three.js setup ----------
const container = document.getElementById('canvas-container');
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x111111);

let width = container.clientWidth;
let height = container.clientHeight;

// Orthographic camera over a pan/zoom-able frustum, y increasing downward.
// zoom=1 keeps the original 1:1 screen-pixel-to-world-unit mapping;
// viewCenter is the world point shown at the center of the canvas.
const ZOOM_MIN = 0.1;
const ZOOM_MAX = 25;
let zoom = 1;
let viewCenter = { x: width / 2, y: height / 2 };

const camera = new THREE.OrthographicCamera(0, width, 0, height, -1, 1);
camera.position.z = 1;

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(width, height);
container.appendChild(renderer.domElement);

function updateCameraFrustum() {
  const halfW = width / 2 / zoom;
  const halfH = height / 2 / zoom;
  camera.left = viewCenter.x - halfW;
  camera.right = viewCenter.x + halfW;
  camera.top = viewCenter.y - halfH;
  camera.bottom = viewCenter.y + halfH;
  camera.updateProjectionMatrix();
  updateHeatmapQuadBounds();
}

function pixelToWorld(sx, sy) {
  return {
    x: camera.left + (sx / width) * (camera.right - camera.left),
    y: camera.top + (sy / height) * (camera.bottom - camera.top),
  };
}

// Sets viewCenter so that world point `anchor` renders at screen pixel (sx, sy).
// Used for zoom-to-cursor and for right-drag panning (keep the grabbed point under the cursor).
function setViewCenterForAnchor(sx, sy, anchor) {
  const halfW = width / 2 / zoom;
  const halfH = height / 2 / zoom;
  viewCenter.x = anchor.x - (sx / width - 0.5) * 2 * halfW;
  viewCenter.y = anchor.y - (sy / height - 0.5) * 2 * halfH;
}

// ---------- circumcircle-containment heatmap (full-screen shader) ----------
const HEATMAP_MAX_CIRCLES = 256;

const heatmapGeom = new THREE.BufferGeometry();
const heatmapMat = new THREE.ShaderMaterial({
  uniforms: {
    uCircles: { value: Array.from({ length: HEATMAP_MAX_CIRCLES }, () => new THREE.Vector3()) },
    uCircleCount: { value: 0 },
  },
  vertexShader: `
    varying vec2 vPos;
    void main() {
      vPos = position.xy;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
  `,
  fragmentShader: `
    varying vec2 vPos;
    uniform vec3 uCircles[${HEATMAP_MAX_CIRCLES}];
    uniform int uCircleCount;
    void main() {
      int count = 0;
      for (int i = 0; i < ${HEATMAP_MAX_CIRCLES}; i++) {
        if (i >= uCircleCount) break;
        vec2 c = uCircles[i].xy;
        float r = uCircles[i].z;
        if (distance(vPos, c) < r) count++;
      }
      float v = 1.0 - exp(-float(count) * 0.35);
      gl_FragColor = vec4(vec3(v), 1.0);
    }
  `,
  depthTest: false,
});
const heatmapMesh = new THREE.Mesh(heatmapGeom, heatmapMat);
heatmapMesh.renderOrder = -1;
heatmapMesh.visible = false;
scene.add(heatmapMesh);

function updateHeatmapQuadBounds() {
  const l = camera.left;
  const r = camera.right;
  const t = camera.top;
  const b = camera.bottom;
  const arr = new Float32Array([l, t, 0, r, t, 0, r, b, 0, l, b, 0]);
  heatmapGeom.setAttribute('position', new THREE.BufferAttribute(arr, 3));
  heatmapGeom.setIndex([0, 1, 2, 0, 2, 3]);
}

function updateHeatmapUniforms() {
  const n = Math.min(lastCircles.length, HEATMAP_MAX_CIRCLES);
  const arr = heatmapMat.uniforms.uCircles.value;
  for (let i = 0; i < n; i++) {
    arr[i].set(lastCircles[i].cx, lastCircles[i].cy, lastCircles[i].r);
  }
  heatmapMat.uniforms.uCircleCount.value = n;
}

window.addEventListener('resize', () => {
  width = container.clientWidth;
  height = container.clientHeight;
  renderer.setSize(width, height);
  updateCameraFrustum();
});

// smooth zoom-to-cursor on any wheel/trackpad-scroll/trackpad-pinch gesture
const ZOOM_SENSITIVITY = 0.0015;
renderer.domElement.addEventListener(
  'wheel',
  (e) => {
    e.preventDefault();
    const rect = renderer.domElement.getBoundingClientRect();
    const sx = e.clientX - rect.left;
    const sy = e.clientY - rect.top;
    const before = pixelToWorld(sx, sy);

    zoom = Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, zoom * Math.exp(-e.deltaY * ZOOM_SENSITIVITY)));
    setViewCenterForAnchor(sx, sy, before);
    updateCameraFrustum();
    refreshSegmentMesh();
  },
  { passive: false },
);

updateCameraFrustum();

// triangulation edges
const triGeom = new THREE.BufferGeometry();
const triMat = new THREE.LineBasicMaterial({ color: 0x3399ff, transparent: true, opacity: 0.8, depthTest: false });
const triLines = new THREE.LineSegments(triGeom, triMat);
triLines.renderOrder = 1;
scene.add(triLines);

// per-triangle circumcircle outline, drawn in plain white
const circleGeom = new THREE.BufferGeometry();
const circleMat = new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.35, depthTest: false });
const circleLines = new THREE.LineSegments(circleGeom, circleMat);
circleLines.renderOrder = 2;
scene.add(circleLines);

// user-drawn segments, rendered as filled quads (not LineSegments) so they can
// actually be made thicker than 1px - WebGL ignores LineBasicMaterial.linewidth
// on virtually every platform. Drawn below the triangulation/circumcircle
// lines (lower renderOrder) so its extra width still peeks out on either
// side even where a thin blue Delaunay edge exactly overlaps it.
const SEGMENT_HALF_WIDTH_PX = 1.25; // screen pixels; kept constant across zoom levels
const segGeom = new THREE.BufferGeometry();
const segMat = new THREE.MeshBasicMaterial({ color: 0xff9933, depthTest: false });
const segMesh = new THREE.Mesh(segGeom, segMat);
segMesh.renderOrder = 0;
scene.add(segMesh);

// live preview while dragging out a new segment
const previewGeom = new THREE.BufferGeometry();
const previewMat = new THREE.LineBasicMaterial({ color: 0xff9933, transparent: true, opacity: 0.5, depthTest: false });
const previewLine = new THREE.LineSegments(previewGeom, previewMat);
previewLine.visible = false;
previewLine.renderOrder = 1;
scene.add(previewLine);

function circleTexture(color) {
  const size = 64;
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = size;
  const ctx = canvas.getContext('2d');
  ctx.beginPath();
  ctx.arc(size / 2, size / 2, size / 2 - 2, 0, Math.PI * 2);
  ctx.fillStyle = color;
  ctx.fill();
  return new THREE.CanvasTexture(canvas);
}

const freeTex = circleTexture('#ffffff');
const sliderTex = circleTexture('#ffdd33');
const generatedTex = circleTexture('#c9a0ff');

function makePointsObject(texture, size) {
  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(0), 3));
  const mat = new THREE.PointsMaterial({
    map: texture,
    size,
    transparent: true,
    depthTest: false,
    sizeAttenuation: false,
  });
  const pts = new THREE.Points(geom, mat);
  pts.renderOrder = 3;
  scene.add(pts);
  return pts;
}

const freePointsObj = makePointsObject(freeTex, 10);
const sliderPointsObj = makePointsObject(sliderTex, 12);
// derived from state, not part of it - not draggable/removable, so it's drawn
// smaller and dimmer than real points and sits below them in render order.
const generatedPointsObj = makePointsObject(generatedTex, 6);
generatedPointsObj.renderOrder = 2.5;
generatedPointsObj.material.opacity = 0.85;

// ---------- geometry helpers ----------
function screenToWorld(evt) {
  const rect = renderer.domElement.getBoundingClientRect();
  return pixelToWorld(evt.clientX - rect.left, evt.clientY - rect.top);
}

function pointPos(p) {
  if (p.kind === 'free') return { x: p.x, y: p.y };
  const seg = segments.find((s) => s.id === p.segId);
  return { x: seg.ax + (seg.bx - seg.ax) * p.t, y: seg.ay + (seg.by - seg.ay) * p.t };
}

function distPointToSegment(px, py, ax, ay, bx, by) {
  const dx = bx - ax;
  const dy = by - ay;
  const len2 = dx * dx + dy * dy;
  let t = len2 > 0 ? ((px - ax) * dx + (py - ay) * dy) / len2 : 0;
  t = Math.max(0, Math.min(1, t));
  const cx = ax + t * dx;
  const cy = ay + t * dy;
  return { dist: Math.hypot(px - cx, py - cy), t };
}

function findPointAt(x, y) {
  const r = POINT_HIT_R / zoom;
  for (let i = points.length - 1; i >= 0; i--) {
    const p = points[i];
    const pos = pointPos(p);
    if (Math.hypot(pos.x - x, pos.y - y) <= r) return p;
  }
  return null;
}

function findSegmentAt(x, y) {
  const r = SEG_HIT_R / zoom;
  for (let i = segments.length - 1; i >= 0; i--) {
    const s = segments[i];
    const { dist } = distPointToSegment(x, y, s.ax, s.ay, s.bx, s.by);
    if (dist <= r) return s;
  }
  return null;
}

// Dragging point C (at parameter t on segment A0B0) to (x, y) reshapes the
// segment via A' = A0 + w(t)*d, B' = B0 + w(1-t)*d, with w(t) = (1-t)(1+2t)
// and d = (x,y) - C0. This keeps C at the same parameter t on the new
// segment for any drag direction, so w(0)=1/w(1)=0 pins the far endpoint
// in place, and w(0.5)=1 makes a midpoint drag a pure translation.
function segmentDragWeight(t) {
  return (1 - t) * (1 + 2 * t);
}

function dragSegmentTo(state, x, y) {
  const seg = segments.find((s) => s.id === state.segId);
  if (!seg) return;
  const dx = x - state.C0.x;
  const dy = y - state.C0.y;
  const wA = segmentDragWeight(state.t);
  const wB = segmentDragWeight(1 - state.t);
  seg.ax = state.A0.x + wA * dx;
  seg.ay = state.A0.y + wA * dy;
  seg.bx = state.B0.x + wB * dx;
  seg.by = state.B0.y + wB * dy;
}

function addFreePoint(x, y) {
  const p = { id: nextId++, kind: 'free', x, y };
  points.push(p);
  update();
  return p;
}

function addSegment(ax, ay, bx, by) {
  const seg = { id: nextId++, ax, ay, bx, by };
  segments.push(seg);
  attachSliderPoint(seg, 0);
  attachSliderPoint(seg, 1);
  update();
  return seg;
}

function attachSliderPoint(seg, t) {
  const sp = { id: nextId++, kind: 'slider', segId: seg.id, t };
  points.push(sp);
  lastMovedSliderId = sp.id;
  return sp;
}

function removePoint(id) {
  const idx = points.findIndex((p) => p.id === id);
  if (idx === -1) return;
  points.splice(idx, 1);
  if (lastMovedSliderId === id) lastMovedSliderId = null;
  update();
}

function removeSegment(id) {
  const idx = segments.findIndex((s) => s.id === id);
  if (idx === -1) return;
  segments.splice(idx, 1);
  points = points.filter((p) => {
    if (p.kind === 'slider' && p.segId === id) {
      if (lastMovedSliderId === p.id) lastMovedSliderId = null;
      return false;
    }
    return true;
  });
  update();
}

function clearAll() {
  points = [];
  segments = [];
  lastMovedSliderId = null;
  update();
}

function randomGaussian() {
  let u = 0;
  let v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

// stddev as a fraction of the visible half-extent; smaller = a more visibly
// non-uniform cluster (large enough that clamping to the margin rarely kicks in)
const GAUSSIAN_SIGMA_FRACTION = 0.2;

function randomizePoints(n, dist) {
  points = [];
  segments = [];
  lastMovedSliderId = null;

  // always scoped to what's currently visible, so nothing spawns off-screen
  const marginPx = 40;
  const marginWorld = marginPx / zoom;
  const left = camera.left + marginWorld;
  const right = camera.right - marginWorld;
  const top = camera.top + marginWorld;
  const bottom = camera.bottom - marginWorld;
  const cx = (left + right) / 2;
  const cy = (top + bottom) / 2;
  const halfW = Math.max(0, (right - left) / 2);
  const halfH = Math.max(0, (bottom - top) / 2);

  for (let i = 0; i < n; i++) {
    let x;
    let y;
    if (dist === 'gaussian') {
      x = cx + randomGaussian() * halfW * GAUSSIAN_SIGMA_FRACTION;
      y = cy + randomGaussian() * halfH * GAUSSIAN_SIGMA_FRACTION;
      x = Math.min(right, Math.max(left, x));
      y = Math.min(bottom, Math.max(top, y));
    } else {
      x = left + Math.random() * (right - left);
      y = top + Math.random() * (bottom - top);
    }
    points.push({ id: nextId++, kind: 'free', x, y });
  }
  update();
}

// ---------- intersections of user segments with circumcircles ----------
function segmentCircleIntersections(seg, circle) {
  const dx = seg.bx - seg.ax;
  const dy = seg.by - seg.ay;
  const fx = seg.ax - circle.cx;
  const fy = seg.ay - circle.cy;
  const a = dx * dx + dy * dy;
  if (a < 1e-12) return [];
  const b = 2 * (fx * dx + fy * dy);
  const c = fx * fx + fy * fy - circle.r * circle.r;
  const disc = b * b - 4 * a * c;
  if (disc < 0) return [];
  const sqrtDisc = Math.sqrt(disc);
  const ts = disc < 1e-9 ? [-b / (2 * a)] : [(-b - sqrtDisc) / (2 * a), (-b + sqrtDisc) / (2 * a)];
  const pts = [];
  for (const t of ts) {
    if (t >= 0 && t <= 1) pts.push({ x: seg.ax + t * dx, y: seg.ay + t * dy });
  }
  return pts;
}

const INTERSECTION_DEDUP_EPS = 1e-6;

// How many rounds of "add every point where a segment crosses a circumcircle"
// to run. Purely derived, not part of `points` - each generation retriangulates
// the real points plus every point generated so far, so it reacts live to any
// edit instead of needing to be regenerated by hand. 0 disables it.
let intersectionGenerations = 0;

function circumcenter(ax, ay, bx, by, cx, cy) {
  const d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));
  if (Math.abs(d) < 1e-9) return null;
  const ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (ay - by)) / d;
  const uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (bx - ax)) / d;
  return [ux, uy];
}

function computeCircles(positions) {
  if (positions.length < 3) return [];
  const delaunay = Delaunay.from(positions, (d) => d.x, (d) => d.y);
  const tris = delaunay.triangles;
  const circles = [];
  for (let t = 0; t < tris.length; t += 3) {
    const A = positions[tris[t]];
    const B = positions[tris[t + 1]];
    const C = positions[tris[t + 2]];
    const cc = circumcenter(A.x, A.y, B.x, B.y, C.x, C.y);
    const [cx, cy] = cc || [(A.x + B.x + C.x) / 3, (A.y + B.y + C.y) / 3];
    circles.push({ cx, cy, r: Math.hypot(A.x - cx, A.y - cy) });
  }
  return circles;
}

function generateIntersectionPoints(basePositions, generations) {
  let current = basePositions;
  const generated = [];
  for (let g = 0; g < generations; g++) {
    const circles = computeCircles(current);
    if (circles.length === 0) break;

    const newPts = [];
    const isDuplicate = (x, y) =>
      current.some((p) => Math.hypot(p.x - x, p.y - y) <= INTERSECTION_DEDUP_EPS) ||
      newPts.some((p) => Math.hypot(p.x - x, p.y - y) <= INTERSECTION_DEDUP_EPS);

    for (const seg of segments) {
      for (const circle of circles) {
        for (const pt of segmentCircleIntersections(seg, circle)) {
          if (!isDuplicate(pt.x, pt.y)) newPts.push(pt);
        }
      }
    }
    if (newPts.length === 0) break; // fixed point reached
    generated.push(...newPts);
    current = current.concat(newPts);
  }
  return generated;
}

// ---------- delaunay computation ----------
const CIRCLE_SEGMENTS = 48;

let lastTriangleEdges = new Float32Array(0);
let lastCirclePositions = new Float32Array(0);
let lastCircles = []; // [{ cx, cy, r }, ...] one per Delaunay triangle
let lastGeneratedPositions = []; // derived, non-editable intersection points

function recomputeDelaunay(basePositions, generated) {
  const pos = basePositions.concat(generated);

  if (pos.length < 3) {
    lastTriangleEdges = new Float32Array(0);
    lastCirclePositions = new Float32Array(0);
    lastCircles = [];
    return;
  }

  const delaunay = Delaunay.from(pos, (d) => d.x, (d) => d.y);
  const tris = delaunay.triangles;
  const triCount = tris.length / 3;

  const edgeArr = new Float32Array(tris.length * 2 * 3);
  const circleArr = new Float32Array(triCount * CIRCLE_SEGMENTS * 2 * 3);
  const circles = [];

  let ei = 0;
  let cri = 0;

  for (let t = 0; t < tris.length; t += 3) {
    const A = pos[tris[t]];
    const B = pos[tris[t + 1]];
    const C = pos[tris[t + 2]];

    edgeArr[ei++] = A.x; edgeArr[ei++] = A.y; edgeArr[ei++] = 0;
    edgeArr[ei++] = B.x; edgeArr[ei++] = B.y; edgeArr[ei++] = 0;
    edgeArr[ei++] = B.x; edgeArr[ei++] = B.y; edgeArr[ei++] = 0;
    edgeArr[ei++] = C.x; edgeArr[ei++] = C.y; edgeArr[ei++] = 0;
    edgeArr[ei++] = C.x; edgeArr[ei++] = C.y; edgeArr[ei++] = 0;
    edgeArr[ei++] = A.x; edgeArr[ei++] = A.y; edgeArr[ei++] = 0;

    const cc = circumcenter(A.x, A.y, B.x, B.y, C.x, C.y);
    const [cx, cy] = cc || [(A.x + B.x + C.x) / 3, (A.y + B.y + C.y) / 3];
    const r = Math.hypot(A.x - cx, A.y - cy);
    circles.push({ cx, cy, r });

    for (let s = 0; s < CIRCLE_SEGMENTS; s++) {
      const a0 = (s / CIRCLE_SEGMENTS) * Math.PI * 2;
      const a1 = ((s + 1) / CIRCLE_SEGMENTS) * Math.PI * 2;
      circleArr[cri++] = cx + Math.cos(a0) * r; circleArr[cri++] = cy + Math.sin(a0) * r; circleArr[cri++] = 0;
      circleArr[cri++] = cx + Math.cos(a1) * r; circleArr[cri++] = cy + Math.sin(a1) * r; circleArr[cri++] = 0;
    }
  }

  lastTriangleEdges = edgeArr;
  lastCirclePositions = circleArr;
  lastCircles = circles;
}

// ---------- render sync ----------
function setPositionGeometry(lineObj, arr) {
  lineObj.geometry.dispose();
  lineObj.geometry = new THREE.BufferGeometry();
  lineObj.geometry.setAttribute('position', new THREE.BufferAttribute(arr, 3));
}

function setPointsGeometry(ptsObj, arr) {
  ptsObj.geometry.dispose();
  ptsObj.geometry = new THREE.BufferGeometry();
  ptsObj.geometry.setAttribute('position', new THREE.BufferAttribute(arr, 3));
}

function buildSegmentQuads(halfWidth) {
  const positions = new Float32Array(segments.length * 6 * 3); // 2 triangles per segment
  let pi = 0;
  for (const s of segments) {
    const dx = s.bx - s.ax;
    const dy = s.by - s.ay;
    const len = Math.hypot(dx, dy) || 1;
    const nx = (-dy / len) * halfWidth;
    const ny = (dx / len) * halfWidth;
    const p0x = s.ax + nx; const p0y = s.ay + ny;
    const p1x = s.bx + nx; const p1y = s.by + ny;
    const p2x = s.bx - nx; const p2y = s.by - ny;
    const p3x = s.ax - nx; const p3y = s.ay - ny;
    positions[pi++] = p0x; positions[pi++] = p0y; positions[pi++] = 0;
    positions[pi++] = p1x; positions[pi++] = p1y; positions[pi++] = 0;
    positions[pi++] = p2x; positions[pi++] = p2y; positions[pi++] = 0;
    positions[pi++] = p0x; positions[pi++] = p0y; positions[pi++] = 0;
    positions[pi++] = p2x; positions[pi++] = p2y; positions[pi++] = 0;
    positions[pi++] = p3x; positions[pi++] = p3y; positions[pi++] = 0;
  }
  return positions;
}

function refreshSegmentMesh() {
  setPositionGeometry(segMesh, buildSegmentQuads(SEGMENT_HALF_WIDTH_PX / zoom));
}

function update() {
  // intersection-generation points are purely derived and always reflect the
  // live state, even while the base triangulation below is paused
  const basePositions = points.map(pointPos);
  lastGeneratedPositions =
    intersectionGenerations > 0 ? generateIntersectionPoints(basePositions, intersectionGenerations) : [];

  if (!paused) recomputeDelaunay(basePositions, lastGeneratedPositions);
  setPositionGeometry(triLines, lastTriangleEdges);
  setPositionGeometry(circleLines, showCircumcircles ? lastCirclePositions : new Float32Array(0));
  heatmapMesh.visible = showCircumcircles;
  updateHeatmapUniforms();

  refreshSegmentMesh();

  const free = points.filter((p) => p.kind === 'free');
  const slider = points.filter((p) => p.kind === 'slider');

  const freeArr = new Float32Array(free.length * 3);
  free.forEach((p, i) => { freeArr[i * 3] = p.x; freeArr[i * 3 + 1] = p.y; freeArr[i * 3 + 2] = 0; });
  setPointsGeometry(freePointsObj, freeArr);

  const sliderArr = new Float32Array(slider.length * 3);
  slider.forEach((p, i) => {
    const pos = pointPos(p);
    sliderArr[i * 3] = pos.x; sliderArr[i * 3 + 1] = pos.y; sliderArr[i * 3 + 2] = 0;
  });
  setPointsGeometry(sliderPointsObj, sliderArr);

  const generatedArr = new Float32Array(lastGeneratedPositions.length * 3);
  lastGeneratedPositions.forEach((p, i) => {
    generatedArr[i * 3] = p.x; generatedArr[i * 3 + 1] = p.y; generatedArr[i * 3 + 2] = 0;
  });
  setPointsGeometry(generatedPointsObj, generatedArr);

  document.getElementById('info').textContent =
    `${points.length} points (${slider.length} on segments), ${lastGeneratedPositions.length} generated, ` +
    `${segments.length} segments${paused ? ' — paused' : ''}`;
}

// ---------- undo / redo ----------
// Snapshots the full editable state (not view/display settings like zoom or
// pause). One snapshot is pushed per discrete action (add/remove/clear/etc.)
// or once per whole drag gesture (on press, before any mutation), never per
// animation frame, so one undo step matches one thing the user did.
const UNDO_LIMIT = 200;
let undoStack = [];
let redoStack = [];

function snapshotState() {
  return {
    points: structuredClone(points),
    segments: structuredClone(segments),
    nextId,
    lastMovedSliderId,
    intersectionGenerations,
  };
}

function restoreState(state) {
  points = structuredClone(state.points);
  segments = structuredClone(state.segments);
  nextId = state.nextId;
  lastMovedSliderId = state.lastMovedSliderId;
  intersectionGenerations = state.intersectionGenerations;
  intersectionGensSlider.value = intersectionGenerations;
  intersectionGensLabel.textContent = intersectionGenerations;
  update();
}

function updateUndoRedoButtons() {
  undoBtn.disabled = undoStack.length === 0;
  redoBtn.disabled = redoStack.length === 0;
}

function pushUndo() {
  undoStack.push(snapshotState());
  if (undoStack.length > UNDO_LIMIT) undoStack.shift();
  redoStack = [];
  updateUndoRedoButtons();
}

function undo() {
  if (undoStack.length === 0) return;
  redoStack.push(snapshotState());
  restoreState(undoStack.pop());
  updateUndoRedoButtons();
}

function redo() {
  if (redoStack.length === 0) return;
  undoStack.push(snapshotState());
  restoreState(redoStack.pop());
  updateUndoRedoButtons();
}

// ---------- pointer interaction ----------
const dom = renderer.domElement;

dom.addEventListener('pointerdown', (e) => {
  const { x, y } = screenToWorld(e);

  if (e.button === 0 && e.ctrlKey) {
    // ctrl+left-drag always pans, regardless of what's under the cursor
    const rect = dom.getBoundingClientRect();
    dragState = {
      mode: 'panPending',
      startSX: e.clientX - rect.left,
      startSY: e.clientY - rect.top,
      anchor: { x, y },
    };
    dom.setPointerCapture(e.pointerId);
    return;
  }

  if (e.button === 0 && e.shiftKey) {
    // shift+left-drag always starts a segment drag from empty-space semantics,
    // ignoring anything under the cursor - it never moves existing points/segments
    dragState = { mode: 'pending', startX: x, startY: y };
    dom.setPointerCapture(e.pointerId);
    return;
  }

  if (e.button !== 0) return;

  const hitPoint = findPointAt(x, y);
  if (hitPoint) {
    pushUndo();
    dragState = { mode: 'point', pointId: hitPoint.id };
    dom.setPointerCapture(e.pointerId);
    return;
  }

  const hitSeg = findSegmentAt(x, y);
  if (hitSeg) {
    const { t } = distPointToSegment(x, y, hitSeg.ax, hitSeg.ay, hitSeg.bx, hitSeg.by);
    dragState = {
      mode: 'segPending',
      segId: hitSeg.id,
      startX: x,
      startY: y,
      t,
      A0: { x: hitSeg.ax, y: hitSeg.ay },
      B0: { x: hitSeg.bx, y: hitSeg.by },
      C0: { x: hitSeg.ax + t * (hitSeg.bx - hitSeg.ax), y: hitSeg.ay + t * (hitSeg.by - hitSeg.ay) },
    };
    dom.setPointerCapture(e.pointerId);
    return;
  }

  dragState = { mode: 'pending', startX: x, startY: y };
  dom.setPointerCapture(e.pointerId);
});

dom.addEventListener('contextmenu', (e) => {
  e.preventDefault();
  const { x, y } = screenToWorld(e);

  const hitPoint = findPointAt(x, y);
  if (hitPoint) {
    pushUndo();
    removePoint(hitPoint.id);
    return;
  }

  const hitSeg = findSegmentAt(x, y);
  if (hitSeg) {
    pushUndo();
    removeSegment(hitSeg.id);
    return;
  }

  pushUndo();
  clearAll();
});

dom.addEventListener('pointermove', (e) => {
  if (!dragState) return;
  const { x, y } = screenToWorld(e);

  if (dragState.mode === 'point') {
    const p = points.find((pp) => pp.id === dragState.pointId);
    if (!p) return;
    if (p.kind === 'free') {
      p.x = x;
      p.y = y;
    } else {
      const seg = segments.find((s) => s.id === p.segId);
      const { t } = distPointToSegment(x, y, seg.ax, seg.ay, seg.bx, seg.by);
      p.t = t;
      lastMovedSliderId = p.id;
    }
    update();
  } else if (dragState.mode === 'pending') {
    const dx = x - dragState.startX;
    const dy = y - dragState.startY;
    if (Math.hypot(dx, dy) > MIN_DRAG / zoom) {
      dragState.mode = 'creating';
      previewLine.visible = true;
    }
  } else if (dragState.mode === 'creating') {
    const arr = new Float32Array([dragState.startX, dragState.startY, 0, x, y, 0]);
    setPositionGeometry(previewLine, arr);
  } else if (dragState.mode === 'segPending') {
    const dx = x - dragState.startX;
    const dy = y - dragState.startY;
    if (Math.hypot(dx, dy) > MIN_DRAG / zoom) {
      pushUndo();
      dragState.mode = 'draggingSegment';
    }
  } else if (dragState.mode === 'draggingSegment') {
    dragSegmentTo(dragState, x, y);
    update();
  } else if (dragState.mode === 'panPending') {
    const rect = dom.getBoundingClientRect();
    const sx = e.clientX - rect.left;
    const sy = e.clientY - rect.top;
    if (Math.hypot(sx - dragState.startSX, sy - dragState.startSY) > MIN_DRAG) {
      dragState.mode = 'panning';
    }
  } else if (dragState.mode === 'panning') {
    const rect = dom.getBoundingClientRect();
    setViewCenterForAnchor(e.clientX - rect.left, e.clientY - rect.top, dragState.anchor);
    updateCameraFrustum();
  }
});

dom.addEventListener('pointerup', (e) => {
  if (!dragState) return;
  const { x, y } = screenToWorld(e);

  if (dragState.mode === 'pending') {
    pushUndo();
    addFreePoint(x, y);
  } else if (dragState.mode === 'creating') {
    pushUndo();
    addSegment(dragState.startX, dragState.startY, x, y);
    previewLine.visible = false;
  } else if (dragState.mode === 'segPending') {
    // plain click on a segment (no real drag): attach a new slider point there
    const seg = segments.find((s) => s.id === dragState.segId);
    if (seg) {
      pushUndo();
      attachSliderPoint(seg, dragState.t);
      update();
    }
  }
  dragState = null;
});

window.addEventListener('keydown', (e) => {
  if (e.key !== 'Enter' || lastMovedSliderId == null) return;
  const sp = points.find((p) => p.id === lastMovedSliderId);
  if (!sp) return;
  const pos = pointPos(sp);
  pushUndo();
  addFreePoint(pos.x, pos.y);
});

// ---------- panel wiring ----------
const countInput = document.getElementById('count');
const distSelect = document.getElementById('dist');
const randomizeBtn = document.getElementById('randomize');
const pauseCheck = document.getElementById('pause');
const circlesCheck = document.getElementById('circumcircles');
const intersectionGensSlider = document.getElementById('intersectionGens');
const intersectionGensLabel = document.getElementById('intersectionGensLabel');
const undoBtn = document.getElementById('undo');
const redoBtn = document.getElementById('redo');

randomizeBtn.addEventListener('click', () => {
  pushUndo();
  randomizePoints(parseInt(countInput.value, 10) || 0, distSelect.value);
});

pauseCheck.addEventListener('change', () => {
  paused = pauseCheck.checked;
  update();
});

circlesCheck.addEventListener('change', () => {
  showCircumcircles = circlesCheck.checked;
  update();
});

// snapshot once before the gesture begins (mouse drag or a key press), not per
// 'input' tick, so one slider drag is one undo step like everything else
intersectionGensSlider.addEventListener('pointerdown', pushUndo);
intersectionGensSlider.addEventListener('keydown', pushUndo);
intersectionGensSlider.addEventListener('input', () => {
  intersectionGenerations = parseInt(intersectionGensSlider.value, 10);
  intersectionGensLabel.textContent = intersectionGenerations;
  update();
});

undoBtn.addEventListener('click', undo);
redoBtn.addEventListener('click', redo);
updateUndoRedoButtons();

// ---------- render loop ----------
function animate() {
  requestAnimationFrame(animate);
  renderer.render(scene, camera);
}
animate();

randomizePoints(30, 'uniform');
