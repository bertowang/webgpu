"use strict";
/**
 * WebGPU Compute Shader 示例：Softmax（Transformer Attention 的核心算子）
 *
 * 编译命令（在 ts/ 目录执行）：
 *   npx -p typescript tsc webgpu_softmax.ts --target ES2020 --lib ES2020,DOM --outDir dist
 */
const gpuGlobal = globalThis;
const gpuApi = navigator.gpu;
/** Softmax：每行独立做「减最大值 → exp → 归一化」，避免数值溢出 */
const SOFTMAX_SHADER = `
    @group(0) @binding(0) var<storage, read> input: array<f32>;
    @group(0) @binding(1) var<storage, read_write> output: array<f32>;

    const N: u32 = 4u;

    @compute @workgroup_size(1)
    fn main(@builtin(global_invocation_id) gid: vec3u) {
        let row = gid.x;
        // 1) 求该行最大值（数值稳定技巧）
        var maxVal = input[row * N];
        for (var i = 1u; i < N; i = i + 1u) {
            maxVal = max(maxVal, input[row * N + i]);
        }
        // 2) 求 exp 之和
        var sum = 0.0;
        for (var i = 0u; i < N; i = i + 1u) {
            sum = sum + exp(input[row * N + i] - maxVal);
        }
        // 3) 归一化写出
        for (var i = 0u; i < N; i = i + 1u) {
            output[row * N + i] = exp(input[row * N + i] - maxVal) / sum;
        }
    }
`;
function createBuffer(device, data, usage) {
    const buffer = device.createBuffer({ size: 64, usage, mappedAtCreation: data !== null });
    if (data !== null) {
        new Float32Array(buffer.getMappedRange()).set(data);
        buffer.unmap();
    }
    return buffer;
}
async function softmaxOnGPU(input) {
    if (!gpuApi)
        throw new Error('当前浏览器不支持 WebGPU');
    const adapter = await gpuApi.requestAdapter();
    if (!adapter)
        throw new Error('未找到 WebGPU 适配器');
    const device = await adapter.requestDevice();
    const shader = device.createShaderModule({ code: SOFTMAX_SHADER });
    const pipeline = device.createComputePipeline({
        layout: 'auto',
        compute: { module: shader, entryPoint: 'main' }
    });
    const usage = gpuGlobal.GPUBufferUsage;
    const bufferIn = createBuffer(device, input, usage.STORAGE | usage.COPY_DST);
    const bufferOut = createBuffer(device, null, usage.STORAGE | usage.COPY_SRC);
    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: bufferIn } },
            { binding: 1, resource: { buffer: bufferOut } }
        ]
    });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(4, 1, 1); // 4 行，每行一个线程
    pass.end();
    const staging = createBuffer(device, null, usage.COPY_DST | usage.MAP_READ);
    encoder.copyBufferToBuffer(bufferOut, 0, staging, 0, 64);
    device.queue.submit([encoder.finish()]);
    await staging.mapAsync(gpuGlobal.GPUMapMode.READ);
    const result = new Float32Array(staging.getMappedRange()).slice();
    staging.unmap();
    return result;
}
function format(m) {
    const rows = [];
    for (let r = 0; r < 4; r++) {
        const row = [];
        for (let c = 0; c < 4; c++)
            row.push(m[r * 4 + c].toFixed(4).padStart(8, ' '));
        rows.push(row.join(' '));
    }
    return rows.join('\n');
}
async function runDemo() {
    const out = document.getElementById('softmax-result');
    if (!out)
        return;
    const X = new Float32Array([
        1, 2, 3, 4,
        0, 0, 0, 0,
        -1, 0, 1, 2,
        10, 20, 30, 40
    ]);
    try {
        const Y = await softmaxOnGPU(X);
        out.textContent = '输入（Attention scores）=\n' + format(X) + '\n\nSoftmax 输出=\n' + format(Y);
    }
    catch (error) {
        out.textContent = '计算失败：' + error.message;
    }
}
runDemo();
