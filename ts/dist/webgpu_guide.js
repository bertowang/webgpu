"use strict";
/**
 * WebGPU 学习指南 · 配套示例（TypeScript）
 *
 * 编译命令（在 ts/ 目录执行）：
 *   npx -p typescript tsc webgpu_guide.ts --target ES2020 --lib ES2020,DOM --outDir dist
 */
const gpuGlobal = globalThis;
const gpuApi = navigator.gpu;
// ── 矩阵与向量运算 ───────────────────────────────────────
/** 矩阵乘法：a × b，列主序 */
function mat4Multiply(a, b) {
    const out = new Float32Array(16);
    for (let c = 0; c < 4; c++) {
        for (let r = 0; r < 4; r++) {
            let sum = 0;
            for (let k = 0; k < 4; k++)
                sum += a[r + k * 4] * b[k + c * 4];
            out[r + c * 4] = sum;
        }
    }
    return out;
}
/** 绕 X 轴旋转 */
function mat4RotationX(angle) {
    const c = Math.cos(angle), s = Math.sin(angle);
    return new Float32Array([1, 0, 0, 0, 0, c, -s, 0, 0, s, c, 0, 0, 0, 0, 1]);
}
/** 绕 Y 轴旋转 */
function mat4RotationY(angle) {
    const c = Math.cos(angle), s = Math.sin(angle);
    return new Float32Array([c, 0, s, 0, 0, 1, 0, 0, -s, 0, c, 0, 0, 0, 0, 1]);
}
/** 透视投影：fovy 弧度、宽高比、近远平面 */
function mat4Perspective(fovy, aspect, near, far) {
    const f = 1 / Math.tan(fovy / 2);
    return new Float32Array([
        f / aspect, 0, 0, 0,
        0, f, 0, 0,
        0, 0, (far + near) / (near - far), -1,
        0, 0, (2 * far * near) / (near - far), 0
    ]);
}
function subtract(a, b) {
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}
function dot(a, b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
function cross(a, b) {
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}
function normalize(v) {
    const len = Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    return [v[0] / len, v[1] / len, v[2] / len];
}
/** LookAt 视角矩阵 */
function mat4LookAt(eye, target, up) {
    const z = normalize(subtract(eye, target));
    const x = normalize(cross(up, z));
    const y = cross(z, x);
    return new Float32Array([
        x[0], y[0], z[0], 0,
        x[1], y[1], z[1], 0,
        x[2], y[2], z[2], 0,
        -dot(x, eye), -dot(y, eye), -dot(z, eye), 1
    ]);
}
// ── 渲染器类 ─────────────────────────────────────────────
class WebGPURenderer {
    constructor(options, onStatus) {
        this.indexCount = 0;
        this.depthTexture = null;
        this.depthW = 0;
        this.depthH = 0;
        this.rotation = 0;
        this.firstFrame = { value: true };
        this.canvas = options.canvas;
        this.fovY = options.fovY ?? Math.PI / 4;
        this.near = options.near ?? 0.1;
        this.far = options.far ?? 100;
        this.onStatus = onStatus;
    }
    /** 初始化：设备 → 上下文 → 管线 → 几何 → 深度纹理 → 渲染循环 */
    async init() {
        if (!gpuApi)
            throw new Error('当前浏览器不支持 WebGPU');
        const adapter = await gpuApi.requestAdapter();
        if (!adapter)
            throw new Error('未找到 WebGPU 适配器');
        this.device = await adapter.requestDevice();
        this.resizeCanvas();
        const context = this.canvas.getContext('webgpu');
        this.context = context;
        const format = gpuApi.getPreferredCanvasFormat();
        this.context.configure({ device: this.device, format, alphaMode: 'opaque' });
        this.createPipeline(format);
        this.createGeometry();
        this.ensureDepthTexture();
        this.onStatus('WebGPU 已就绪，正在启动渲染…');
        requestAnimationFrame(() => this.render());
    }
    /** WGSL 着色器 + 渲染管线 */
    createPipeline(format) {
        const shader = this.device.createShaderModule({
            code: `
                struct Uniforms { mvp: mat4x4f };
                @group(0) @binding(0) var<uniform> uniforms: Uniforms;
                struct VertexOutput {
                    @builtin(position) position : vec4f,
                    @location(0) color : vec4f
                };
                @vertex
                fn vertex_main(@location(0) position : vec3f, @location(1) color : vec3f) -> VertexOutput {
                    var out: VertexOutput;
                    out.position = uniforms.mvp * vec4f(position, 1.0);
                    out.color = vec4f(color, 1.0);
                    return out;
                }
                @fragment
                fn fragment_main(input : VertexOutput) -> @location(0) vec4f {
                    return input.color;
                }
            `
        });
        this.pipeline = this.device.createRenderPipeline({
            layout: 'auto',
            vertex: {
                module: shader,
                entryPoint: 'vertex_main',
                buffers: [{
                        arrayStride: 6 * 4,
                        attributes: [
                            { shaderLocation: 0, offset: 0, format: 'float32x3' },
                            { shaderLocation: 1, offset: 12, format: 'float32x3' }
                        ]
                    }]
            },
            fragment: {
                module: shader,
                entryPoint: 'fragment_main',
                targets: [{ format }]
            },
            primitive: { topology: 'triangle-list', cullMode: 'none' },
            depthStencil: {
                format: 'depth24plus',
                depthWriteEnabled: true,
                depthCompare: 'less'
            }
        });
    }
    /** 立方体顶点/索引缓冲 + Uniform Buffer + BindGroup */
    createGeometry() {
        const faces = [
            { corners: [[-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5]], color: [1, 0.3, 0.3] },
            { corners: [[0.5, -0.5, -0.5], [-0.5, -0.5, -0.5], [-0.5, 0.5, -0.5], [0.5, 0.5, -0.5]], color: [0.3, 0.8, 0.3] },
            { corners: [[0.5, -0.5, 0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [0.5, 0.5, 0.5]], color: [0.3, 0.5, 1] },
            { corners: [[-0.5, -0.5, -0.5], [-0.5, -0.5, 0.5], [-0.5, 0.5, 0.5], [-0.5, 0.5, -0.5]], color: [1, 0.85, 0.2] },
            { corners: [[-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, -0.5, 0.5], [-0.5, -0.5, 0.5]], color: [1, 0.2, 1] },
            { corners: [[-0.5, 0.5, -0.5], [0.5, 0.5, -0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5]], color: [0.2, 1, 1] }
        ];
        const vertices = [];
        for (const face of faces) {
            for (const p of face.corners)
                vertices.push(...p, ...face.color);
        }
        const vertexData = new Float32Array(vertices);
        const indices = new Uint16Array([
            0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7, 8, 9, 10, 8, 10, 11,
            12, 13, 14, 12, 14, 15, 16, 17, 18, 16, 18, 19, 20, 21, 22, 20, 22, 23
        ]);
        const vertexBuffer = this.device.createBuffer({
            size: vertexData.byteLength,
            usage: gpuGlobal.GPUBufferUsage.VERTEX | gpuGlobal.GPUBufferUsage.COPY_DST,
            mappedAtCreation: true
        });
        new Float32Array(vertexBuffer.getMappedRange()).set(vertexData);
        vertexBuffer.unmap();
        const indexBuffer = this.device.createBuffer({
            size: indices.byteLength,
            usage: gpuGlobal.GPUBufferUsage.INDEX | gpuGlobal.GPUBufferUsage.COPY_DST,
            mappedAtCreation: true
        });
        new Uint16Array(indexBuffer.getMappedRange()).set(indices);
        indexBuffer.unmap();
        this.uniformBuffer = this.device.createBuffer({
            size: 64,
            usage: gpuGlobal.GPUBufferUsage.UNIFORM | gpuGlobal.GPUBufferUsage.COPY_DST
        });
        this.bindGroup = this.device.createBindGroup({
            layout: this.pipeline.getBindGroupLayout(0),
            entries: [{ binding: 0, resource: { buffer: this.uniformBuffer } }]
        });
        this.vertexBuffer = vertexBuffer;
        this.indexBuffer = indexBuffer;
        this.indexCount = indices.length;
    }
    /** 深度纹理：尺寸随画布变化而重建 */
    ensureDepthTexture() {
        if (this.depthW === this.canvas.width && this.depthH === this.canvas.height)
            return;
        if (this.depthTexture)
            this.depthTexture.destroy();
        this.depthTexture = null;
        this.depthW = this.canvas.width;
        this.depthH = this.canvas.height;
        if (!this.depthW || !this.depthH)
            return;
        this.depthTexture = this.device.createTexture({
            size: [this.depthW, this.depthH],
            format: 'depth24plus',
            usage: gpuGlobal.GPUTextureUsage.RENDER_ATTACHMENT
        });
    }
    /** 每帧：更新矩阵 → 编码命令 → 提交 */
    render() {
        try {
            this.resizeCanvas();
            this.ensureDepthTexture();
            if (!this.depthTexture || !this.canvas.width || !this.canvas.height) {
                requestAnimationFrame(() => this.render());
                return;
            }
            this.rotation += 0.01;
            const aspect = this.canvas.width / this.canvas.height;
            const projection = mat4Perspective(this.fovY, aspect, this.near, this.far);
            const view = mat4LookAt([0, 0, 3], [0, 0, 0], [0, 1, 0]);
            const model = mat4Multiply(mat4RotationY(this.rotation), mat4RotationX(this.rotation * 0.6));
            const mvp = mat4Multiply(mat4Multiply(projection, view), model);
            this.device.queue.writeBuffer(this.uniformBuffer, 0, mvp);
            const encoder = this.device.createCommandEncoder();
            const pass = encoder.beginRenderPass({
                colorAttachments: [{
                        view: this.context.getCurrentTexture().createView(),
                        clearValue: { r: 0.05, g: 0.06, b: 0.09, a: 1 },
                        loadOp: 'clear',
                        storeOp: 'store'
                    }],
                depthStencilAttachment: {
                    view: this.depthTexture.createView(),
                    depthClearValue: 1.0,
                    depthLoadOp: 'clear',
                    depthStoreOp: 'store'
                }
            });
            pass.setPipeline(this.pipeline);
            pass.setBindGroup(0, this.bindGroup);
            pass.setVertexBuffer(0, this.vertexBuffer);
            pass.setIndexBuffer(this.indexBuffer, 'uint16');
            pass.drawIndexed(this.indexCount);
            pass.end();
            this.device.queue.submit([encoder.finish()]);
            if (this.firstFrame.value) {
                this.firstFrame.value = false;
                this.onStatus('运行中：旋转立方体');
            }
            requestAnimationFrame(() => this.render());
        }
        catch (error) {
            this.onStatus('渲染错误：' + error.message);
        }
    }
    resizeCanvas() {
        const w = this.canvas.clientWidth;
        const h = this.canvas.clientHeight;
        if (w > 0 && h > 0 && (this.canvas.width !== w || this.canvas.height !== h)) {
            this.canvas.width = w;
            this.canvas.height = h;
        }
    }
}
// ── 启动 ─────────────────────────────────────────────────
const canvas = document.getElementById('webgpu-canvas');
const statusEl = document.getElementById('demo-status');
if (canvas && statusEl) {
    const renderer = new WebGPURenderer({ canvas }, (message) => {
        statusEl.textContent = message;
    });
    renderer.init().catch((error) => {
        statusEl.textContent = 'WebGPU 初始化失败：' + error.message;
    });
}
