# WebGPU 学习示例

本目录包含用 JavaScript 和 TypeScript 两种语言实现的 WebGPU 学习示例，核心示例是一个**旋转的彩色立方体**。

## 目录结构

```
webgpu_example/
├── README.md                                    # 本说明文件
├── js/                                          # JavaScript 版本
│   ├── webgpu_learning_example.html              # JS 版主文档（推荐阅读）
│   ├── optimized_webgpu_learning_example.html    # 早期优化版（简化三角形示例）
│   └── index.html                               # 早期 WebGPU 旋转立方体页面
└── ts/                                          # TypeScript 版本
    ├── webgpu_learning_example.ts                # TS 源文件（带类型注解、接口、类）
    ├── webgpu_learning_example_ts.html           # TS 版文档（内嵌 TS 源码展示与编译后 JS）
    └── dist/
        └── webgpu_learning_example.js            # tsc 编译产物
```

## 示例内容

两个主文档（`js/webgpu_learning_example.html` 与 `ts/webgpu_learning_example_ts.html`）都是**自包含的 HTML 页面**，用 Chrome 直接打开即可阅读文档并运行实时演示，无需启动服务器或安装依赖。

文档内容主要包括：

- **数据处理流程节点图**：从 CPU 侧准备数据 → 创建 GPUBuffer → 编译 WGSL → BindGroup → RenderPipeline → 编码命令 → 提交队列 → 呈现画布，共 8 个节点，每个节点说明对应 API 与设计原因。
- **函数与注释**：按函数逐一展示对应注释和实现，包括矩阵乘法、X/Y 轴旋转矩阵、透视投影矩阵、LookAt 视角矩阵、向量运算（减/点/叉/归一化）以及 `initWebGPU` 主流程。
- **底层原理说明**：9 个小节解释 WebGPU API 的设计动机，例如 Adapter 与 Device 分工、Canvas 配置与像素格式、WGSL 编译模型、Buffer usage 标志、BindGroup 布局解耦、RenderPipeline 不可变性、深度纹理同步、命令编码与批处理、矩阵列主序等。
- **实时演示**：页面侧栏展示旋转立方体。若浏览器不支持 WebGPU，演示会跳过并在状态栏提示，文档阅读不受影响。

## 各文件说明

### JavaScript 版本（`js/`）

- `webgpu_learning_example.html`（推荐）：完整学习文档，含流程节点图、函数注释、底层原理与实时演示。
- `optimized_webgpu_learning_example.html`：早期优化版，演示为简单彩色三角形，保留作历史参考。
- `index.html`：早期 WebGPU 旋转立方体页面，结构简单。

### TypeScript 版本（`ts/`）

- `webgpu_learning_example.ts`：用 TypeScript 编写的 WebGPU 功能实现，包含类型别名、接口、`WebGPURenderer` 类、类型断言等，不依赖 `@webgpu/types`。
- `webgpu_learning_example_ts.html`：TS 版文档，内嵌展示 TS 源码并内嵌编译后的 JS demo。
- `dist/webgpu_learning_example.js`：由 `webgpu_learning_example.ts` 编译生成的 JS。

修改 TS 版本时，编辑 `webgpu_learning_example.ts`，重新编译并把产物同步到 HTML 底部的内嵌脚本中：

```bash
cd ts
npx -p typescript tsc webgpu_learning_example.ts --target ES2020 --lib ES2020,DOM --outDir dist
```

（也可使用 Homebrew 安装的 `tsc`，命令相同。）

## 运行环境

- 浏览器：支持 WebGPU 的 Chrome 113+ 或 Edge 113+。
- 打开方式：直接用 Chrome 打开对应的 `.html` 文件即可（`file://` 协议下可用），无需本地服务器。
- 若浏览器不支持 WebGPU，页面文档仍可正常阅读，仅实时演示会显示提示。

## 参考文档

- [WebGPU 规范](https://gpuweb.github.io/gpuweb/)
- [MDN WebGPU 文档](https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API)
