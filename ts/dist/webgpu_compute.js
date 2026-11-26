"use strict";
/**
 * WebGPU Compute Shader 示例：矩阵乘法（AI 计算的最小内核）
 *
 * 编译命令（在 ts/ 目录执行）：
 *   npx -p typescript tsc webgpu_compute.ts --target ES2020 --lib ES2020,DOM --outDir dist
 */
const gpuGlobal = globalThis;
const gpuApi = navigator.gpu;
// ── compute shader 矩阵乘法 ────────────────────────────────
/** 4×4 矩阵乘法的 WGSL compute kernel */
const MATMUL_SHADER = `
    @group(0) @binding(0) var<storage, read> matrixA: array<f32>;
    @group(0) @binding(1) var<storage, read> matrixB: array<f32>;
    @group(0) @binding(2) var<storage, read_write> matrixC: array<f32>;

    @compute @workgroup_size(1)
    fn main(@builtin(global_invocation_id) gid: vec3u) {
        let row = gid.x;
        let col = gid.y;
        var sum = 0.0;
        for (var k = 0u; k < 4u; k = k + 1u) {
            sum = sum + matrixA[row * 4u + k] * matrixB[k * 4u + col];
        }
        matrixC[row * 4u + col] = sum;
    }
`;
/** 创建可映射读取的 staging buffer */
function createBuffer(device, data, usage) {
    const buffer = device.createBuffer({
        size: 64,
        usage: usage,
        mappedAtCreation: data !== null
    });
    if (data !== null) {
        new Float32Array(buffer.getMappedRange()).set(data);
        buffer.unmap();
    }
    return buffer;
}
/** 在 GPU 上计算 A × B，并异步读回结果 */
async function matmulOnGPU(a, b) {
    if (!gpuApi)
        throw new Error('当前浏览器不支持 WebGPU');
    const adapter = await gpuApi.requestAdapter();
    if (!adapter)
        throw new Error('未找到 WebGPU 适配器');
    const device = await adapter.requestDevice();
    // 1. 编译 compute shader 并创建管线（无顶点/片段阶段）
    const shader = device.createShaderModule({ code: MATMUL_SHADER });
    const pipeline = device.createComputePipeline({
        layout: 'auto',
        compute: { module: shader, entryPoint: 'main' }
    });
    // 2. 创建三个 storage buffer：两个输入、一个输出
    const usage = gpuGlobal.GPUBufferUsage;
    const bufferA = createBuffer(device, a, usage.STORAGE | usage.COPY_DST);
    const bufferB = createBuffer(device, b, usage.STORAGE | usage.COPY_DST);
    const bufferC = createBuffer(device, null, usage.STORAGE | usage.COPY_SRC);
    // 3. 绑定资源：@binding(0)=A, @binding(1)=B, @binding(2)=C
    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: bufferA } },
            { binding: 1, resource: { buffer: bufferB } },
            { binding: 2, resource: { buffer: bufferC } }
        ]
    });
    // 4. 编码 compute pass：启动 4×4=16 个线程，每个线程算一个输出元素
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(4, 4, 1);
    pass.end();
    // 5. 把结果拷贝到可映射的 staging buffer 并读回
    const staging = createBuffer(device, null, usage.COPY_DST | usage.MAP_READ);
    encoder.copyBufferToBuffer(bufferC, 0, staging, 0, 64);
    device.queue.submit([encoder.finish()]);
    await staging.mapAsync(gpuGlobal.GPUMapMode.READ);
    const result = new Float32Array(staging.getMappedRange()).slice();
    staging.unmap();
    return result;
}
/** 把矩阵格式化成字符串，便于展示 */
function formatMatrix(m, size) {
    const rows = [];
    for (let r = 0; r < size; r++) {
        const row = [];
        for (let c = 0; c < size; c++) {
            row.push(m[r * size + c].toFixed(0).padStart(4, ' '));
        }
        rows.push(row.join(' '));
    }
    return rows.join('\n');
}
/** 在页面上展示结果 */
async function runDemo() {
    const out = document.getElementById('matmul-result');
    if (!out)
        return;
    const A = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
    const B = new Float32Array([16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]);
    try {
        const C = await matmulOnGPU(A, B);
        out.textContent =
            'A =\n' + formatMatrix(A, 4) +
                '\n\nB =\n' + formatMatrix(B, 4) +
                '\n\nA × B =\n' + formatMatrix(C, 4);
    }
    catch (error) {
        out.textContent = '计算失败：' + error.message;
    }
}
runDemo();
