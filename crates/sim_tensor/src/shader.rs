use crate::{CpuBackend, ExecutionContext, ImageTensor, TensorError};
use naga::{ShaderStage, front::glsl};
use std::{
    fmt,
    sync::{Arc, Mutex, mpsc},
    time::Duration,
};
use thiserror::Error;
use wgpu::util::DeviceExt;

pub const MAX_SHADER_SOURCE_BYTES: usize = 256 * 1024;
pub const MAX_SHADER_IMAGES: usize = 5;
pub const MAX_SHADER_FLOATS: usize = 20;
pub const MAX_SHADER_INTS: usize = 20;
pub const MAX_SHADER_BOOLS: usize = 10;
pub const MAX_SHADER_CURVES: usize = 4;
pub const MAX_SHADER_OUTPUTS: usize = 4;
pub const MAX_SHADER_PASSES: u32 = 32;
pub const MAX_SHADER_DIMENSION: u32 = 8192;
pub const MAX_SHADER_PIXELS: u64 = 67_108_864;
pub const MAX_SHADER_BATCH: u64 = 64;
pub const MAX_SHADER_CURVE_SAMPLES: usize = 65_536;
pub const MAX_SHADER_TOTAL_CURVE_SAMPLES: usize = 262_144;

const SHADER_TEXTURE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Float;
const UNIFORM_BINDING: u32 = 18;
#[cfg(target_os = "linux")]
static SHADER_INSTANCE_LOCK: Mutex<Option<std::sync::Weak<wgpu::Instance>>> = Mutex::new(None);
// GLSL matches fragment interpolation. The fixed GL triangle is mapped into
// WGPU's top-origin texture rows and zero-to-one depth range for every pass.
const VERTEX_SHADER: &str = r#"#version 450
layout(location = 0) out vec2 v_texCoord;

void main() {
    vec2 positions[3] = vec2[3](
        vec2(-1.0, -1.0),
        vec2(3.0, -1.0),
        vec2(-1.0, 3.0)
    );
    vec2 position = positions[gl_VertexIndex];
    gl_Position = vec4(position.x, -position.y, 0.5, 1.0);
    v_texCoord = position * 0.5 + vec2(0.5, 0.5);
}
"#;

#[derive(Clone, Debug)]
pub struct NativeShaderRequest {
    pub fragment_source: String,
    pub images: Vec<ImageTensor>,
    pub floats: Vec<f32>,
    pub ints: Vec<i32>,
    pub bools: Vec<bool>,
    pub curves: Vec<Vec<f32>>,
    pub width: u32,
    pub height: u32,
}

#[derive(Clone, Debug)]
pub struct NativeShaderResult {
    pub outputs: Vec<ImageTensor>,
    pub pass_count: u32,
}

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum NativeShaderError {
    #[error("shader execution was cancelled")]
    Cancelled,
    #[error("shader request exceeds the bounded native contract: {0}")]
    Bounds(String),
    #[error("GLSL ES compilation failed: {0}")]
    Compilation(String),
    #[error("shader backend is unavailable: {0}")]
    BackendUnavailable(String),
    #[error("shader device was lost: {0}")]
    DeviceLost(String),
    #[error("shader tensor projection failed: {0}")]
    Tensor(String),
}

impl From<TensorError> for NativeShaderError {
    fn from(error: TensorError) -> Self {
        Self::Tensor(error.to_string())
    }
}

pub trait NativeShaderExecutor: fmt::Debug + Send + Sync {
    fn configuration_identity(&self) -> String;

    fn execute(
        &self,
        request: &NativeShaderRequest,
        backend: &CpuBackend,
        context: &ExecutionContext<'_>,
    ) -> Result<NativeShaderResult, NativeShaderError>;
}

#[derive(Debug)]
pub struct WgpuNativeShaderExecutor {
    state: WgpuNativeShaderState,
}

#[derive(Debug)]
enum WgpuNativeShaderState {
    Ready {
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        adapter_name: String,
        execution_lock: Mutex<()>,
        _instance: Arc<wgpu::Instance>,
    },
    Unavailable(String),
}

impl WgpuNativeShaderExecutor {
    pub fn new_or_unavailable() -> Self {
        match Self::new() {
            Ok(executor) => executor,
            Err(error) => Self {
                state: WgpuNativeShaderState::Unavailable(error.to_string()),
            },
        }
    }

    pub fn new() -> Result<Self, NativeShaderError> {
        // Unavailable NVIDIA entry-point discovery can race with live driver cleanup.
        // Share only discovery while executors are alive, retaining independent devices.
        #[cfg(target_os = "linux")]
        let mut instance_guard = SHADER_INSTANCE_LOCK.lock().map_err(|_| {
            NativeShaderError::BackendUnavailable(
                "shader initialization lock is poisoned".to_owned(),
            )
        })?;
        #[cfg(target_os = "linux")]
        let cached_instance = instance_guard.as_ref().and_then(std::sync::Weak::upgrade);
        let create_instance = || {
            Arc::new(wgpu::Instance::new(wgpu::InstanceDescriptor {
                backends: wgpu::Backends::PRIMARY,
                flags: wgpu::InstanceFlags::default(),
                backend_options: wgpu::BackendOptions::default(),
                memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
                display: None,
            }))
        };
        #[cfg(target_os = "linux")]
        let instance = cached_instance.unwrap_or_else(create_instance);
        #[cfg(not(target_os = "linux"))]
        let instance = create_instance();
        #[cfg(target_os = "linux")]
        {
            *instance_guard = Some(Arc::downgrade(&instance));
        }
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .map_err(|error| NativeShaderError::BackendUnavailable(error.to_string()))?;
        let adapter_name = adapter.get_info().name;
        let required_features = wgpu::Features::FLOAT32_FILTERABLE;
        if !adapter.features().contains(required_features) {
            return Err(NativeShaderError::BackendUnavailable(
                "adapter does not support filterable 32-bit float textures".to_owned(),
            ));
        }
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("sim-native-shader-device"),
            required_features,
            required_limits: wgpu::Limits {
                max_color_attachment_bytes_per_sample: 64,
                ..wgpu::Limits::default()
            },
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::MemoryUsage,
            trace: wgpu::Trace::Off,
        }))
        .map_err(|error| NativeShaderError::BackendUnavailable(error.to_string()))?;
        Ok(Self {
            state: WgpuNativeShaderState::Ready {
                device: Arc::new(device),
                queue: Arc::new(queue),
                adapter_name,
                execution_lock: Mutex::new(()),
                _instance: instance,
            },
        })
    }

    pub fn adapter_name(&self) -> Option<&str> {
        match &self.state {
            WgpuNativeShaderState::Ready { adapter_name, .. } => Some(adapter_name),
            WgpuNativeShaderState::Unavailable(_) => None,
        }
    }
}

#[cfg(target_os = "linux")]
impl Drop for WgpuNativeShaderExecutor {
    fn drop(&mut self) {
        let _initialization_guard = match SHADER_INSTANCE_LOCK.lock() {
            Ok(guard) => guard,
            Err(error) => {
                eprintln!("shader initialization lock is poisoned during device cleanup");
                error.into_inner()
            }
        };
        let state = std::mem::replace(
            &mut self.state,
            WgpuNativeShaderState::Unavailable(String::new()),
        );
        drop(state);
    }
}

impl NativeShaderExecutor for WgpuNativeShaderExecutor {
    fn configuration_identity(&self) -> String {
        match &self.state {
            WgpuNativeShaderState::Ready { adapter_name, .. } => {
                format!("wgpu-glsl-es300-v2:{adapter_name}")
            }
            WgpuNativeShaderState::Unavailable(_) => "wgpu-glsl-es300-v2:unavailable".to_owned(),
        }
    }

    fn execute(
        &self,
        request: &NativeShaderRequest,
        _backend: &CpuBackend,
        context: &ExecutionContext<'_>,
    ) -> Result<NativeShaderResult, NativeShaderError> {
        context
            .cancellation
            .check()
            .map_err(|_| NativeShaderError::Cancelled)?;
        validate_request(request)?;
        let compiled = compile_fragment_source(&request.fragment_source)?;
        let (device, queue, execution_lock) = match &self.state {
            WgpuNativeShaderState::Ready {
                device,
                queue,
                execution_lock,
                ..
            } => (device, queue, execution_lock),
            WgpuNativeShaderState::Unavailable(reason) => {
                return Err(NativeShaderError::BackendUnavailable(reason.clone()));
            }
        };
        let _guard = execution_lock.lock().map_err(|_| {
            NativeShaderError::DeviceLost("shader execution lock is poisoned".to_owned())
        })?;
        execute_wgpu(device, queue, request, &compiled, _backend, context)
    }
}

fn validate_request(request: &NativeShaderRequest) -> Result<(), NativeShaderError> {
    if request.fragment_source.is_empty() || request.fragment_source.len() > MAX_SHADER_SOURCE_BYTES
    {
        return Err(NativeShaderError::Bounds(
            "fragment source length is invalid".to_owned(),
        ));
    }
    if request.images.len() > MAX_SHADER_IMAGES
        || request.floats.len() > MAX_SHADER_FLOATS
        || request.ints.len() > MAX_SHADER_INTS
        || request.bools.len() > MAX_SHADER_BOOLS
        || request.curves.len() > MAX_SHADER_CURVES
    {
        return Err(NativeShaderError::Bounds(
            "uniform cardinality exceeds the source contract".to_owned(),
        ));
    }
    if request.width == 0
        || request.height == 0
        || request.width > MAX_SHADER_DIMENSION
        || request.height > MAX_SHADER_DIMENSION
        || u64::from(request.width) * u64::from(request.height) > MAX_SHADER_PIXELS
    {
        return Err(NativeShaderError::Bounds(
            "output dimensions exceed the source contract".to_owned(),
        ));
    }
    if request.floats.iter().any(|value| !value.is_finite())
        || request
            .curves
            .iter()
            .flatten()
            .any(|value| !value.is_finite())
    {
        return Err(NativeShaderError::Bounds(
            "shader numeric inputs must be finite".to_owned(),
        ));
    }
    if request.images.is_empty() {
        return Err(NativeShaderError::Bounds(
            "at least one shader image is required".to_owned(),
        ));
    }
    let mut batch_size = None;
    for image in &request.images {
        let (batch, _, _, channels) = image.dimensions()?;
        if batch == 0 || batch > MAX_SHADER_BATCH || !matches!(channels, 3 | 4) {
            return Err(NativeShaderError::Bounds(
                "shader images must be bounded RGB or RGBA batches".to_owned(),
            ));
        }
        if batch_size
            .replace(batch)
            .is_some_and(|expected| expected != batch)
        {
            return Err(NativeShaderError::Bounds(
                "shader image batch counts must match".to_owned(),
            ));
        }
    }
    let total_curve_samples = request.curves.iter().try_fold(0usize, |total, curve| {
        if curve.is_empty() || curve.len() > MAX_SHADER_CURVE_SAMPLES {
            return Err(NativeShaderError::Bounds(
                "curve sample count exceeds the source contract".to_owned(),
            ));
        }
        total
            .checked_add(curve.len())
            .ok_or_else(|| NativeShaderError::Bounds("curve sample count overflowed".to_owned()))
    })?;
    if total_curve_samples > MAX_SHADER_TOTAL_CURVE_SAMPLES {
        return Err(NativeShaderError::Bounds(
            "aggregate curve samples exceed the source contract".to_owned(),
        ));
    }
    Ok(())
}

fn execute_wgpu(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    request: &NativeShaderRequest,
    compiled: &CompiledNativeShader,
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
) -> Result<NativeShaderResult, NativeShaderError> {
    debug_assert_eq!(compiled.module.entry_points.len(), 1);
    let validation_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let fragment_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("sim-native-glsl-fragment"),
        source: wgpu::ShaderSource::Glsl {
            shader: compiled.source.clone().into(),
            stage: naga::ShaderStage::Fragment,
            defines: &[],
        },
    });
    let vertex_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("sim-native-shader-vertex"),
        source: wgpu::ShaderSource::Glsl {
            shader: VERTEX_SHADER.into(),
            stage: naga::ShaderStage::Vertex,
            defines: &[],
        },
    });
    let bind_group_layout = create_shader_bind_group_layout(device);
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("sim-native-shader-pipeline-layout"),
        bind_group_layouts: &[Some(&bind_group_layout)],
        immediate_size: 0,
    });
    let targets = (0..compiled.output_count)
        .map(|_| {
            Some(wgpu::ColorTargetState {
                format: SHADER_TEXTURE_FORMAT,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })
        })
        .collect::<Vec<_>>();
    let create_pipeline = |label: &'static str, targets: &[Option<wgpu::ColorTargetState>]| {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &vertex_module,
                entry_point: Some("main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &fragment_module,
                entry_point: Some("main"),
                targets,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                // The clip-space Y conversion reverses source winding.
                front_face: wgpu::FrontFace::Cw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        })
    };
    let pipeline = create_pipeline("sim-native-shader-pipeline", &targets);
    let intermediate_pipeline = (compiled.pass_count > 1).then(|| {
        // Source intermediate framebuffers retain only output zero.
        let mut intermediate_targets = targets.clone();
        for target in intermediate_targets.iter_mut().skip(1) {
            *target = None;
        }
        create_pipeline(
            "sim-native-shader-intermediate-pipeline",
            &intermediate_targets,
        )
    });
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("sim-native-shader-sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Nearest,
        ..Default::default()
    });
    let black_texture = upload_rgba_texture(device, queue, 1, 1, &[0.0, 0.0, 0.0, 0.0])?;
    let black_view = black_texture.create_view(&wgpu::TextureViewDescriptor::default());
    let curve_textures = request
        .curves
        .iter()
        .map(|curve| upload_curve_texture(device, queue, curve))
        .collect::<Result<Vec<_>, _>>()?;
    let curve_views = curve_textures
        .iter()
        .map(|texture| texture.create_view(&wgpu::TextureViewDescriptor::default()))
        .collect::<Vec<_>>();
    let (batch_size, _, _, _) = request.images[0].dimensions()?;
    let output_pixel_count = usize::try_from(
        u64::from(request.width)
            .checked_mul(u64::from(request.height))
            .and_then(|pixels| pixels.checked_mul(4))
            .ok_or_else(|| NativeShaderError::Bounds("output size overflowed".to_owned()))?,
    )
    .map_err(|_| NativeShaderError::Bounds("output size is not addressable".to_owned()))?;
    let mut output_values = (0..MAX_SHADER_OUTPUTS)
        .map(|_| {
            let total = output_pixel_count
                .checked_mul(batch_size as usize)
                .ok_or_else(|| {
                    NativeShaderError::Bounds("batch output size overflowed".to_owned())
                })?;
            let mut values = Vec::new();
            values.try_reserve_exact(total).map_err(|error| {
                NativeShaderError::Bounds(format!("output allocation failed: {error}"))
            })?;
            Ok(values)
        })
        .collect::<Result<Vec<Vec<f32>>, NativeShaderError>>()?;
    for batch_index in 0..batch_size {
        context
            .cancellation
            .check()
            .map_err(|_| NativeShaderError::Cancelled)?;
        let input_textures = request
            .images
            .iter()
            .map(|image| upload_image_batch(device, queue, image, batch_index))
            .collect::<Result<Vec<_>, _>>()?;
        let input_views = input_textures
            .iter()
            .map(|texture| texture.create_view(&wgpu::TextureViewDescriptor::default()))
            .collect::<Vec<_>>();
        let output_textures = (0..compiled.output_count)
            .map(|_| create_render_texture(device, request.width, request.height, true))
            .collect::<Vec<_>>();
        let output_views = output_textures
            .iter()
            .map(|texture| texture.create_view(&wgpu::TextureViewDescriptor::default()))
            .collect::<Vec<_>>();
        let ping_pong_textures = if compiled.pass_count > 1 {
            (0..2)
                .map(|_| create_render_texture(device, request.width, request.height, false))
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        let ping_pong_views = ping_pong_textures
            .iter()
            .map(|texture| texture.create_view(&wgpu::TextureViewDescriptor::default()))
            .collect::<Vec<_>>();
        let uniform_buffers = (0..compiled.pass_count)
            .map(|pass| {
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("sim-native-shader-uniforms"),
                    contents: &shader_uniform_bytes(request, pass),
                    usage: wgpu::BufferUsages::UNIFORM,
                })
            })
            .collect::<Vec<_>>();
        let bind_groups = (0..compiled.pass_count)
            .map(|pass| {
                let primary_view = if pass == 0 {
                    input_views.first().unwrap_or(&black_view)
                } else {
                    ping_pong_views
                        .get(((pass - 1) % 2) as usize)
                        .unwrap_or(&black_view)
                };
                create_shader_bind_group(
                    device,
                    &bind_group_layout,
                    &sampler,
                    primary_view,
                    &input_views,
                    &curve_views,
                    &black_view,
                    &uniform_buffers[pass as usize],
                )
            })
            .collect::<Vec<_>>();
        let row_bytes = request
            .width
            .checked_mul(16)
            .ok_or_else(|| NativeShaderError::Bounds("readback row size overflowed".to_owned()))?;
        let padded_row_bytes = align_copy_row_bytes(row_bytes)?;
        let readback_size = u64::from(padded_row_bytes)
            .checked_mul(u64::from(request.height))
            .ok_or_else(|| NativeShaderError::Bounds("readback size overflowed".to_owned()))?;
        let readback_buffers = (0..compiled.output_count)
            .map(|_| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("sim-native-shader-readback"),
                    size: readback_size,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                })
            })
            .collect::<Vec<_>>();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("sim-native-shader-encoder"),
        });
        for pass_index in 0..compiled.pass_count {
            let is_last = pass_index + 1 == compiled.pass_count;
            if is_last {
                let attachments = output_views
                    .iter()
                    .map(|view| {
                        Some(wgpu::RenderPassColorAttachment {
                            view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                                store: wgpu::StoreOp::Store,
                            },
                            depth_slice: None,
                        })
                    })
                    .collect::<Vec<_>>();
                let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("sim-native-shader-final-pass"),
                    color_attachments: &attachments,
                    depth_stencil_attachment: None,
                    ..Default::default()
                });
                render_pass.set_pipeline(&pipeline);
                render_pass.set_bind_group(0, &bind_groups[pass_index as usize], &[]);
                render_pass.draw(0..3, 0..1);
            } else {
                let target_view = &ping_pong_views[(pass_index % 2) as usize];
                let mut attachments = Vec::with_capacity(compiled.output_count);
                attachments.push(Some(wgpu::RenderPassColorAttachment {
                    view: target_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                }));
                attachments.resize_with(compiled.output_count, || None);
                let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("sim-native-shader-intermediate-pass"),
                    color_attachments: &attachments,
                    depth_stencil_attachment: None,
                    ..Default::default()
                });
                render_pass.set_pipeline(intermediate_pipeline.as_ref().ok_or_else(|| {
                    NativeShaderError::Compilation("intermediate pipeline is missing".to_owned())
                })?);
                render_pass.set_bind_group(0, &bind_groups[pass_index as usize], &[]);
                render_pass.draw(0..3, 0..1);
            }
        }
        for (texture, buffer) in output_textures.iter().zip(&readback_buffers) {
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(padded_row_bytes),
                        rows_per_image: Some(request.height),
                    },
                },
                wgpu::Extent3d {
                    width: request.width,
                    height: request.height,
                    depth_or_array_layers: 1,
                },
            );
        }
        let submission = queue.submit([encoder.finish()]);
        #[cfg(test)]
        tests::cancellation_rendezvous(tests::CancellationPhase::Submitted)?;
        if let Err(error) = wait_for_submission(device, submission, &context.cancellation) {
            if let Some(validation_error) = pollster::block_on(validation_scope.pop()) {
                return Err(NativeShaderError::Compilation(validation_error.to_string()));
            }
            return Err(error);
        }
        for (output_index, buffer) in readback_buffers.iter().enumerate() {
            let values = readback_f32(
                device,
                buffer,
                request.width,
                request.height,
                padded_row_bytes,
                &context.cancellation,
            )?;
            output_values[output_index].extend_from_slice(&values);
        }
        for output in output_values.iter_mut().skip(compiled.output_count) {
            output.resize(output.len() + output_pixel_count, 0.0);
        }
    }
    if let Some(error) = pollster::block_on(validation_scope.pop()) {
        return Err(NativeShaderError::Compilation(error.to_string()));
    }
    context
        .cancellation
        .check()
        .map_err(|_| NativeShaderError::Cancelled)?;
    let outputs = output_values
        .into_iter()
        .map(|values| {
            ImageTensor::from_f32(
                backend,
                context,
                batch_size,
                u64::from(request.height),
                u64::from(request.width),
                4,
                &values,
            )
            .map_err(NativeShaderError::from)
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(NativeShaderResult {
        outputs,
        pass_count: compiled.pass_count,
    })
}

fn create_shader_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let mut entries = Vec::with_capacity(19);
    for texture_index in 0..9u32 {
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: texture_index * 2,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        });
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: texture_index * 2 + 1,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        });
    }
    entries.push(wgpu::BindGroupLayoutEntry {
        binding: UNIFORM_BINDING,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    });
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("sim-native-shader-bindings"),
        entries: &entries,
    })
}

#[allow(clippy::too_many_arguments)]
fn create_shader_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    primary_view: &wgpu::TextureView,
    input_views: &[wgpu::TextureView],
    curve_views: &[wgpu::TextureView],
    black_view: &wgpu::TextureView,
    uniforms: &wgpu::Buffer,
) -> wgpu::BindGroup {
    let mut entries = Vec::with_capacity(19);
    for index in 0..MAX_SHADER_IMAGES {
        let view = if index == 0 {
            primary_view
        } else {
            input_views.get(index).unwrap_or(black_view)
        };
        entries.push(wgpu::BindGroupEntry {
            binding: index as u32 * 2,
            resource: wgpu::BindingResource::TextureView(view),
        });
        entries.push(wgpu::BindGroupEntry {
            binding: index as u32 * 2 + 1,
            resource: wgpu::BindingResource::Sampler(sampler),
        });
    }
    for index in 0..MAX_SHADER_CURVES {
        let view = curve_views.get(index).unwrap_or(black_view);
        let binding = 10 + index as u32 * 2;
        entries.push(wgpu::BindGroupEntry {
            binding,
            resource: wgpu::BindingResource::TextureView(view),
        });
        entries.push(wgpu::BindGroupEntry {
            binding: binding + 1,
            resource: wgpu::BindingResource::Sampler(sampler),
        });
    }
    entries.push(wgpu::BindGroupEntry {
        binding: UNIFORM_BINDING,
        resource: uniforms.as_entire_binding(),
    });
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("sim-native-shader-bind-group"),
        layout,
        entries: &entries,
    })
}

fn shader_uniform_bytes(request: &NativeShaderRequest, pass: u32) -> Vec<u8> {
    let mut words = vec![0u32; 56];
    words[0] = (request.width as f32).to_bits();
    words[1] = (request.height as f32).to_bits();
    words[2] = pass;
    for (index, value) in request.floats.iter().enumerate() {
        words[3 + index] = value.to_bits();
    }
    for (index, value) in request.ints.iter().enumerate() {
        words[23 + index] = *value as u32;
    }
    for (index, value) in request.bools.iter().enumerate() {
        words[43 + index] = u32::from(*value);
    }
    bytemuck::cast_slice(&words).to_vec()
}

fn upload_image_batch(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    image: &ImageTensor,
    batch_index: u64,
) -> Result<wgpu::Texture, NativeShaderError> {
    let (batch, height, width, channels) = image.dimensions()?;
    if batch_index >= batch {
        return Err(NativeShaderError::Bounds(
            "shader batch index is out of bounds".to_owned(),
        ));
    }
    let source = image.as_f32_slice()?;
    let frame_pixels = height
        .checked_mul(width)
        .ok_or_else(|| NativeShaderError::Bounds("input image size overflowed".to_owned()))?;
    let frame_values = frame_pixels
        .checked_mul(channels)
        .ok_or_else(|| NativeShaderError::Bounds("input image stride overflowed".to_owned()))?;
    let start =
        usize::try_from(batch_index.checked_mul(frame_values).ok_or_else(|| {
            NativeShaderError::Bounds("input batch offset overflowed".to_owned())
        })?)
        .map_err(|_| NativeShaderError::Bounds("input batch offset is invalid".to_owned()))?;
    let length = usize::try_from(frame_values)
        .map_err(|_| NativeShaderError::Bounds("input frame length is invalid".to_owned()))?;
    let frame = source.get(start..start + length).ok_or_else(|| {
        NativeShaderError::Tensor("input image storage changed after validation".to_owned())
    })?;
    let rgba_length = usize::try_from(
        frame_pixels
            .checked_mul(4)
            .ok_or_else(|| NativeShaderError::Bounds("RGBA input size overflowed".to_owned()))?,
    )
    .map_err(|_| NativeShaderError::Bounds("RGBA input size is invalid".to_owned()))?;
    let mut rgba = Vec::new();
    rgba.try_reserve_exact(rgba_length).map_err(|error| {
        NativeShaderError::Bounds(format!("input staging allocation failed: {error}"))
    })?;
    // Keep textures in the source GL row order throughout ping-pong rendering.
    for row in frame.chunks_exact(width as usize * channels as usize).rev() {
        for pixel in row.chunks_exact(channels as usize) {
            rgba.extend_from_slice(&pixel[..3]);
            rgba.push(if channels == 4 { pixel[3] } else { 1.0 });
        }
    }
    upload_rgba_texture(
        device,
        queue,
        u32::try_from(width)
            .map_err(|_| NativeShaderError::Bounds("input width is invalid".to_owned()))?,
        u32::try_from(height)
            .map_err(|_| NativeShaderError::Bounds("input height is invalid".to_owned()))?,
        &rgba,
    )
}

fn upload_curve_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    curve: &[f32],
) -> Result<wgpu::Texture, NativeShaderError> {
    let mut rgba = Vec::new();
    rgba.try_reserve_exact(curve.len() * 4).map_err(|error| {
        NativeShaderError::Bounds(format!("curve staging allocation failed: {error}"))
    })?;
    for value in curve {
        rgba.extend_from_slice(&[*value, 0.0, 0.0, 1.0]);
    }
    upload_rgba_texture(
        device,
        queue,
        u32::try_from(curve.len())
            .map_err(|_| NativeShaderError::Bounds("curve width is invalid".to_owned()))?,
        1,
        &rgba,
    )
}

fn upload_rgba_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    width: u32,
    height: u32,
    values: &[f32],
) -> Result<wgpu::Texture, NativeShaderError> {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("sim-native-shader-input"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: SHADER_TEXTURE_FORMAT,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let expected = usize::try_from(u64::from(width) * u64::from(height) * 4)
        .map_err(|_| NativeShaderError::Bounds("texture input size is invalid".to_owned()))?;
    if values.len() != expected {
        return Err(NativeShaderError::Bounds(
            "texture input length does not match its dimensions".to_owned(),
        ));
    }
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytemuck::cast_slice(values),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(width * 16),
            rows_per_image: Some(height),
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    Ok(texture)
}

fn create_render_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    copy_source: bool,
) -> wgpu::Texture {
    let mut usage = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;
    if copy_source {
        usage |= wgpu::TextureUsages::COPY_SRC;
    }
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("sim-native-shader-render-target"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: SHADER_TEXTURE_FORMAT,
        usage,
        view_formats: &[],
    })
}

fn align_copy_row_bytes(row_bytes: u32) -> Result<u32, NativeShaderError> {
    let alignment = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    row_bytes
        .checked_add(alignment - 1)
        .map(|value| value / alignment * alignment)
        .ok_or_else(|| NativeShaderError::Bounds("copy row alignment overflowed".to_owned()))
}

fn wait_for_submission(
    device: &wgpu::Device,
    submission: wgpu::SubmissionIndex,
    cancellation: &crate::CancellationToken,
) -> Result<(), NativeShaderError> {
    loop {
        cancellation
            .check()
            .map_err(|_| NativeShaderError::Cancelled)?;
        match device.poll(wgpu::PollType::Wait {
            submission_index: Some(submission.clone()),
            timeout: Some(Duration::from_millis(10)),
        }) {
            Ok(status) if status.is_queue_empty() => return Ok(()),
            Ok(_) | Err(wgpu::PollError::Timeout) => {}
            Err(error) => return Err(NativeShaderError::DeviceLost(error.to_string())),
        }
    }
}

fn readback_f32(
    device: &wgpu::Device,
    buffer: &wgpu::Buffer,
    width: u32,
    height: u32,
    padded_row_bytes: u32,
    cancellation: &crate::CancellationToken,
) -> Result<Vec<f32>, NativeShaderError> {
    let (sender, receiver) = mpsc::sync_channel(1);
    buffer
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            if sender.send(result).is_err() {
                return;
            }
        });
    #[cfg(test)]
    tests::cancellation_rendezvous(tests::CancellationPhase::MappingRequested)?;
    let mapped = loop {
        cancellation
            .check()
            .map_err(|_| NativeShaderError::Cancelled)?;
        match receiver.try_recv() {
            Ok(result) => break result,
            Err(mpsc::TryRecvError::Empty) => match device.poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(Duration::from_millis(10)),
            }) {
                Ok(_) | Err(wgpu::PollError::Timeout) => {}
                Err(error) => return Err(NativeShaderError::DeviceLost(error.to_string())),
            },
            Err(mpsc::TryRecvError::Disconnected) => {
                return Err(NativeShaderError::DeviceLost(
                    "shader readback callback disconnected".to_owned(),
                ));
            }
        }
    };
    mapped.map_err(|error| NativeShaderError::DeviceLost(error.to_string()))?;
    let mapped = buffer.slice(..).get_mapped_range();
    let row_values = width as usize * 4;
    let mut values = Vec::new();
    values
        .try_reserve_exact(row_values * height as usize)
        .map_err(|error| {
            NativeShaderError::Bounds(format!("readback allocation failed: {error}"))
        })?;
    for row in mapped
        .chunks_exact(padded_row_bytes as usize)
        .take(height as usize)
        .rev()
    {
        let bytes = row.get(..row_values * 4).ok_or_else(|| {
            NativeShaderError::DeviceLost("shader readback row is truncated".to_owned())
        })?;
        let row = bytemuck::try_cast_slice::<u8, f32>(bytes).map_err(|error| {
            NativeShaderError::DeviceLost(format!("shader readback is misaligned: {error}"))
        })?;
        values.extend_from_slice(row);
    }
    drop(mapped);
    buffer.unmap();
    Ok(values)
}

#[derive(Debug)]
struct CompiledNativeShader {
    source: String,
    module: naga::Module,
    output_count: usize,
    pass_count: u32,
}

fn compile_fragment_source(source: &str) -> Result<CompiledNativeShader, NativeShaderError> {
    let output_count = detect_output_count(source);
    let pass_count = detect_pass_count(source)?;
    let lowered = lower_es_300_source(source)?;
    let module = glsl::Frontend::default()
        .parse(&glsl::Options::from(ShaderStage::Fragment), &lowered)
        .map_err(|errors| NativeShaderError::Compilation(errors.emit_to_string(&lowered)))?;
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .map_err(|error| NativeShaderError::Compilation(format!("{error:?}")))?;
    Ok(CompiledNativeShader {
        source: lowered,
        module,
        output_count,
        pass_count,
    })
}

fn lower_es_300_source(source: &str) -> Result<String, NativeShaderError> {
    let mut lines = source.lines();
    let version = lines.next().unwrap_or_default().trim();
    if version != "#version 300 es" {
        return Err(NativeShaderError::Compilation(
            "source must begin with `#version 300 es`".to_owned(),
        ));
    }
    let mut lowered = String::from("#version 450\n");
    lowered.push_str(
        "layout(std140, set = 0, binding = 18) uniform SimShaderUniforms {\n\
         vec2 u_resolution;\n\
         int u_pass;\n\
         float u_float0; float u_float1; float u_float2; float u_float3;\n\
         float u_float4; float u_float5; float u_float6; float u_float7;\n\
         float u_float8; float u_float9; float u_float10; float u_float11;\n\
         float u_float12; float u_float13; float u_float14; float u_float15;\n\
         float u_float16; float u_float17; float u_float18; float u_float19;\n\
         int u_int0; int u_int1; int u_int2; int u_int3;\n\
         int u_int4; int u_int5; int u_int6; int u_int7;\n\
         int u_int8; int u_int9; int u_int10; int u_int11;\n\
         int u_int12; int u_int13; int u_int14; int u_int15;\n\
         int u_int16; int u_int17; int u_int18; int u_int19;\n\
         int zed_bool0; int zed_bool1; int zed_bool2; int zed_bool3; int zed_bool4;\n\
         int zed_bool5; int zed_bool6; int zed_bool7; int zed_bool8; int zed_bool9;\n\
         };\n",
    );
    for index in 0..MAX_SHADER_BOOLS {
        lowered.push_str(&format!("#define u_bool{index} (zed_bool{index} != 0)\n"));
    }
    for line in lines {
        let trimmed = line.trim();
        if trimmed.starts_with("#pragma passes ") {
            continue;
        }
        if trimmed.starts_with("precision ") && trimmed.ends_with(';') {
            continue;
        }
        if is_injected_scalar_uniform(trimmed) {
            continue;
        }
        if let Some((name, texture_binding)) = sampled_uniform_binding(trimmed) {
            let sampler_binding = texture_binding + 1;
            lowered.push_str(&format!(
                "layout(set = 0, binding = {texture_binding}) uniform texture2D zed_texture_{name};\n\
                 layout(set = 0, binding = {sampler_binding}) uniform sampler zed_sampler_{name};\n\
                 #define {name} sampler2D(zed_texture_{name}, zed_sampler_{name})\n"
            ));
            continue;
        }
        if trimmed == "in vec2 v_texCoord;" {
            lowered.push_str("layout(location = 0) in vec2 v_texCoord;\n");
            continue;
        }
        lowered.push_str(line);
        lowered.push('\n');
    }
    Ok(lowered)
}

fn detect_output_count(source: &str) -> usize {
    source
        .match_indices("fragColor")
        .filter_map(|(start, _)| {
            source
                .as_bytes()
                .get(start + "fragColor".len())
                .copied()
                .filter(u8::is_ascii_digit)
                .map(|digit| usize::from(digit - b'0') + 1)
        })
        .max()
        .unwrap_or(1)
        .min(MAX_SHADER_OUTPUTS)
}

fn detect_pass_count(source: &str) -> Result<u32, NativeShaderError> {
    let mut count = 1u32;
    for line in source.lines() {
        let trimmed = line.trim();
        let Some(value) = trimmed.strip_prefix("#pragma passes ") else {
            continue;
        };
        let parsed = value.parse::<u32>().map_err(|_| {
            NativeShaderError::Compilation("invalid `#pragma passes` value".to_owned())
        })?;
        count = parsed.max(1);
    }
    if count > MAX_SHADER_PASSES {
        return Err(NativeShaderError::Bounds(format!(
            "shader pass count {count} exceeds {MAX_SHADER_PASSES}"
        )));
    }
    Ok(count)
}

fn is_injected_scalar_uniform(line: &str) -> bool {
    let Some(declaration) = line.strip_prefix("uniform ") else {
        return false;
    };
    let Some(declaration) = declaration.strip_suffix(';') else {
        return false;
    };
    let mut parts = declaration.split_whitespace();
    let Some(value_type) = parts.next() else {
        return false;
    };
    let Some(name) = parts.next() else {
        return false;
    };
    if parts.next().is_some() {
        return false;
    }
    match value_type {
        "vec2" => name == "u_resolution",
        "float" => indexed_uniform(name, "u_float", MAX_SHADER_FLOATS),
        "int" => name == "u_pass" || indexed_uniform(name, "u_int", MAX_SHADER_INTS),
        "bool" => indexed_uniform(name, "u_bool", MAX_SHADER_BOOLS),
        _ => false,
    }
}

fn sampled_uniform_binding(line: &str) -> Option<(&str, u32)> {
    let declaration = line.strip_prefix("uniform sampler2D ")?.strip_suffix(';')?;
    if declaration.split_whitespace().count() != 1 {
        return None;
    }
    if let Some(index) = indexed_uniform_index(declaration, "u_image", MAX_SHADER_IMAGES) {
        return Some((declaration, index * 2));
    }
    indexed_uniform_index(declaration, "u_curve", MAX_SHADER_CURVES)
        .map(|index| (declaration, 10 + index * 2))
}

fn indexed_uniform(name: &str, prefix: &str, limit: usize) -> bool {
    indexed_uniform_index(name, prefix, limit).is_some()
}

fn indexed_uniform_index(name: &str, prefix: &str, limit: usize) -> Option<u32> {
    let index = name.strip_prefix(prefix)?.parse::<usize>().ok()?;
    (index < limit).then_some(index as u32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CancellationToken, CpuWorkspaceAuthority, StreamId};

    const TEST_MEMORY_LIMIT_BYTES: u64 = 4 * 1024 * 1024;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(super) enum CancellationPhase {
        Submitted,
        MappingRequested,
    }

    struct CancellationRendezvous {
        phase: CancellationPhase,
        reached: mpsc::SyncSender<CancellationPhase>,
        resume: mpsc::Receiver<()>,
    }

    thread_local! {
        static CANCELLATION_RENDEZVOUS: std::cell::RefCell<Option<CancellationRendezvous>> =
            const { std::cell::RefCell::new(None) };
    }

    pub(super) fn cancellation_rendezvous(
        phase: CancellationPhase,
    ) -> Result<(), NativeShaderError> {
        let rendezvous = CANCELLATION_RENDEZVOUS.with(|slot| {
            let mut slot = slot.borrow_mut();
            if slot
                .as_ref()
                .is_some_and(|rendezvous| rendezvous.phase == phase)
            {
                slot.take()
            } else {
                None
            }
        });
        if let Some(rendezvous) = rendezvous {
            rendezvous.reached.send(phase).map_err(|error| {
                NativeShaderError::DeviceLost(format!("test phase receiver disappeared: {error}"))
            })?;
            rendezvous
                .resume
                .recv_timeout(Duration::from_secs(30))
                .map_err(|error| {
                    NativeShaderError::DeviceLost(format!("test phase release failed: {error}"))
                })?;
        }
        Ok(())
    }

    fn tensor_context<'a>(
        authority: &CpuWorkspaceAuthority,
        cancellation: &'a CancellationToken,
    ) -> Result<ExecutionContext<'a>, TensorError> {
        Ok(ExecutionContext {
            stream: StreamId::DEFAULT,
            scratch: authority.authorize_workspace(0)?,
            rng_phase: None,
            cancellation,
        })
    }

    #[test]
    fn source_default_fragment_shader_parses_after_es_admission() -> Result<(), NativeShaderError> {
        let source = r#"#version 300 es
precision highp float;
precision highp int;
uniform sampler2D u_image0;
uniform vec2 u_resolution;
in vec2 v_texCoord;
layout(location = 0) out vec4 fragColor0;
void main() {
    fragColor0 = texture(u_image0, v_texCoord);
}
"#;
        let module = compile_fragment_source(source)?;
        assert_eq!(module.module.entry_points.len(), 1);
        let globals = module
            .module
            .global_variables
            .iter()
            .filter_map(|(_, variable)| variable.name.as_deref())
            .collect::<Vec<_>>();
        assert!(globals.contains(&"zed_texture_u_image0"));
        assert!(globals.contains(&"zed_sampler_u_image0"));
        assert!(module.output_count == 1);
        assert!(module.pass_count == 1);
        Ok(())
    }

    #[test]
    fn lowering_binds_all_uniform_families_mrt_and_multipass() -> Result<(), NativeShaderError> {
        let source = r#"#version 300 es
#pragma passes 3
precision highp float;
precision highp int;
uniform sampler2D u_image0;
uniform sampler2D u_image4;
uniform sampler2D u_curve3;
uniform vec2 u_resolution;
uniform float u_float19;
uniform int u_int19;
uniform bool u_bool9;
uniform int u_pass;
in vec2 v_texCoord;
layout(location = 0) out vec4 fragColor0;
layout(location = 1) out vec4 fragColor1;
layout(location = 2) out vec4 fragColor2;
layout(location = 3) out vec4 fragColor3;
void main() {
    vec4 image0 = texture(u_image0, v_texCoord);
    vec4 image4 = texture(u_image4, v_texCoord);
    float curve = texture(u_curve3, vec2(v_texCoord.x, 0.5)).r;
    float scalar = u_float19 + float(u_int19 + u_pass) + (u_bool9 ? 1.0 : 0.0);
    fragColor0 = image0;
    fragColor1 = image4;
    fragColor2 = vec4(curve);
    fragColor3 = vec4(u_resolution / max(u_resolution, vec2(1.0)), scalar, 1.0);
}
"#;
        let compiled = compile_fragment_source(source)?;
        assert_eq!(compiled.output_count, MAX_SHADER_OUTPUTS);
        assert_eq!(compiled.pass_count, 3);
        for required in [
            "binding = 0) uniform texture2D zed_texture_u_image0",
            "binding = 8) uniform texture2D zed_texture_u_image4",
            "binding = 16) uniform texture2D zed_texture_u_curve3",
            "layout(std140, set = 0, binding = 18)",
        ] {
            assert!(compiled.source.contains(required), "missing {required}");
        }
        assert!(!compiled.source.contains("#pragma passes"));
        let request = NativeShaderRequest {
            fragment_source: source.to_owned(),
            images: Vec::new(),
            floats: vec![0.0; MAX_SHADER_FLOATS],
            ints: vec![0; MAX_SHADER_INTS],
            bools: vec![false; MAX_SHADER_BOOLS],
            curves: Vec::new(),
            width: 1,
            height: 1,
        };
        assert_eq!(shader_uniform_bytes(&request, 0).len(), 224);
        Ok(())
    }

    #[test]
    fn invalid_source_and_excessive_passes_fail_closed() {
        assert!(matches!(
            compile_fragment_source("#version 300 es\nvoid main( {"),
            Err(NativeShaderError::Compilation(_))
        ));
        assert!(matches!(
            compile_fragment_source(
                "#version 300 es\n#pragma passes 33\nlayout(location = 0) out vec4 fragColor0;\nvoid main() { fragColor0 = vec4(1.0); }"
            ),
            Err(NativeShaderError::Bounds(_))
        ));
    }

    #[test]
    fn cancellation_precedes_backend_and_request_admission()
    -> Result<(), Box<dyn std::error::Error>> {
        let executor = WgpuNativeShaderExecutor::new_or_unavailable();
        let cancellation = CancellationToken::default();
        cancellation.cancel();
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(TEST_MEMORY_LIMIT_BYTES)?;
        let context = tensor_context(&authority, &cancellation)?;
        let request = NativeShaderRequest {
            fragment_source: String::new(),
            images: Vec::new(),
            floats: Vec::new(),
            ints: Vec::new(),
            bools: Vec::new(),
            curves: Vec::new(),
            width: 0,
            height: 0,
        };
        assert!(matches!(
            executor.execute(&request, &backend, &context),
            Err(NativeShaderError::Cancelled)
        ));
        Ok(())
    }

    #[test]
    fn headless_adapter_executes_identity_or_reports_typed_unavailability()
    -> Result<(), Box<dyn std::error::Error>> {
        let executor = match WgpuNativeShaderExecutor::new() {
            Ok(executor) => executor,
            Err(NativeShaderError::BackendUnavailable(reason)) => {
                assert!(!reason.is_empty());
                return Ok(());
            }
            Err(error) => return Err(error.into()),
        };
        let cancellation = CancellationToken::default();
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(TEST_MEMORY_LIMIT_BYTES)?;
        let context = tensor_context(&authority, &cancellation)?;
        let source = r#"#version 300 es
precision highp float;
precision highp int;
uniform sampler2D u_image0;
uniform vec2 u_resolution;
in vec2 v_texCoord;
layout(location = 0) out vec4 fragColor0;
void main() {
    fragColor0 = texture(u_image0, v_texCoord);
}
"#;
        let image = ImageTensor::from_f32(
            &backend,
            &context,
            1,
            2,
            2,
            3,
            &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        )?;
        let result = executor.execute(
            &NativeShaderRequest {
                fragment_source: source.to_owned(),
                images: vec![image],
                floats: Vec::new(),
                ints: Vec::new(),
                bools: Vec::new(),
                curves: Vec::new(),
                width: 2,
                height: 2,
            },
            &backend,
            &context,
        )?;
        assert_eq!(result.outputs.len(), MAX_SHADER_OUTPUTS);
        assert_eq!(
            result.outputs[0].to_f32_vec()?,
            vec![
                1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            ]
        );
        assert!(
            result.outputs[1]
                .to_f32_vec()?
                .iter()
                .all(|value| *value == 0.0)
        );
        assert_eq!(result.pass_count, 1);
        Ok(())
    }

    #[test]
    fn headless_adapter_executes_four_targets_and_multipass()
    -> Result<(), Box<dyn std::error::Error>> {
        let executor = match WgpuNativeShaderExecutor::new() {
            Ok(executor) => executor,
            Err(NativeShaderError::BackendUnavailable(reason)) => {
                eprintln!("headless shader adapter unavailable: {reason}");
                assert!(!reason.is_empty());
                return Ok(());
            }
            Err(error) => return Err(error.into()),
        };
        eprintln!("headless shader adapter: {:?}", executor.adapter_name());
        let cancellation = CancellationToken::default();
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(TEST_MEMORY_LIMIT_BYTES)?;
        let context = tensor_context(&authority, &cancellation)?;
        for pass_count in [1, 2, 3] {
            let source = format!(
                r#"#version 300 es
#pragma passes {pass_count}
precision highp float;
uniform sampler2D u_image0;
uniform int u_pass;
uniform vec2 u_resolution;
in vec2 v_texCoord;
layout(location = 0) out vec4 fragColor0;
layout(location = 1) out vec4 fragColor1;
layout(location = 2) out vec4 fragColor2;
layout(location = 3) out vec4 fragColor3;
void main() {{
    fragColor0 = texture(u_image0, v_texCoord) + vec4(0.125, 0.0, 0.0, 0.0);
    fragColor1 = vec4(v_texCoord, 0.0, 1.0);
    fragColor2 = vec4(gl_FragCoord.xy / u_resolution, gl_FragCoord.z, gl_FragCoord.w);
    fragColor3 = vec4(float(u_pass) * 0.25, 0.5, 0.75, gl_FrontFacing ? 1.0 : 0.0);
}}
"#
            );
            let input = [
                0.25, 0.5, 0.75, 0.5, 0.75, 0.25, 0.0, 0.25, 0.5, 0.125, 0.0, 0.75,
            ];
            let image = ImageTensor::from_f32(&backend, &context, 1, 2, 2, 3, &input)?;
            let result = executor.execute(
                &NativeShaderRequest {
                    fragment_source: source,
                    images: vec![image],
                    floats: Vec::new(),
                    ints: Vec::new(),
                    bools: Vec::new(),
                    curves: Vec::new(),
                    width: 2,
                    height: 2,
                },
                &backend,
                &context,
            )?;
            assert_eq!(result.pass_count, pass_count);
            let coordinates = vec![
                0.25, 0.75, 0.0, 1.0, 0.75, 0.75, 0.0, 1.0, 0.25, 0.25, 0.0, 1.0, 0.75, 0.25, 0.0,
                1.0,
            ];
            let fragment_coordinates = vec![
                0.25, 0.75, 0.5, 1.0, 0.75, 0.75, 0.5, 1.0, 0.25, 0.25, 0.5, 1.0, 0.75, 0.25, 0.5,
                1.0,
            ];
            let expected = [
                input
                    .chunks_exact(3)
                    .flat_map(|pixel| {
                        [
                            pixel[0] + 0.125 * pass_count as f32,
                            pixel[1],
                            pixel[2],
                            1.0,
                        ]
                    })
                    .collect::<Vec<_>>(),
                coordinates,
                fragment_coordinates,
                [(pass_count - 1) as f32 * 0.25, 0.5, 0.75, 1.0].repeat(4),
            ];
            assert_eq!(result.outputs.len(), expected.len());
            let actual = result
                .outputs
                .iter()
                .map(ImageTensor::to_f32_vec)
                .collect::<Result<Vec<_>, _>>()?;
            eprintln!("shader outputs after {pass_count} passes: {actual:?}");
            for (actual, expected) in actual.iter().zip(expected) {
                assert_eq!(actual, &expected);
            }
        }
        Ok(())
    }

    #[test]
    fn concurrent_initialization_preserves_execution_and_unavailable_results()
    -> Result<(), NativeShaderError> {
        let barrier = std::sync::Barrier::new(4);
        for _ in 0..4 {
            std::thread::scope(|scope| -> Result<(), NativeShaderError> {
                let threads = (0..4)
                .map(|_| {
                    scope.spawn(|| -> Result<(), NativeShaderError> {
                        barrier.wait();
                        let executor = WgpuNativeShaderExecutor::new();
                        barrier.wait();
                        let executor = match executor {
                            Ok(executor) => executor,
                            Err(NativeShaderError::BackendUnavailable(reason)) => {
                                assert!(!reason.is_empty());
                                return Ok(());
                            }
                            Err(error) => return Err(error),
                        };
                        let cancellation = CancellationToken::default();
                        let (backend, authority) =
                            CpuWorkspaceAuthority::create_backend(TEST_MEMORY_LIMIT_BYTES)?;
                        let context = tensor_context(&authority, &cancellation)?;
                        let image =
                            ImageTensor::from_f32(&backend, &context, 1, 1, 1, 3, &[0.0; 3])?;
                        let result = executor.execute(
                            &NativeShaderRequest {
                                fragment_source: "#version 300 es\nprecision highp float;\nlayout(location = 0) out vec4 fragColor0;\nvoid main() { fragColor0 = vec4(0.25, 0.5, 0.75, 1.0); }".to_owned(),
                                images: vec![image],
                                floats: Vec::new(),
                                ints: Vec::new(),
                                bools: Vec::new(),
                                curves: Vec::new(),
                                width: 1,
                                height: 1,
                            },
                            &backend,
                            &context,
                        )?;
                        assert_eq!(result.pass_count, 1);
                        assert_eq!(result.outputs.len(), MAX_SHADER_OUTPUTS);
                        let mut outputs = result.outputs.iter();
                        assert_eq!(
                            outputs
                                .next()
                                .ok_or_else(|| NativeShaderError::Tensor(
                                    "shader did not publish its first target".to_owned(),
                                ))?
                                .to_f32_vec()?,
                            vec![0.25, 0.5, 0.75, 1.0]
                        );
                        for output in outputs {
                            assert_eq!(output.to_f32_vec()?, vec![0.0; 4]);
                        }
                        drop(result);
                        drop(context);
                        assert_eq!(backend.memory_snapshot().current_bytes, 0);
                        Ok(())
                    })
                })
                .collect::<Vec<_>>();
                for thread in threads {
                    thread.join().map_err(|_| {
                        NativeShaderError::DeviceLost(
                            "shader initialization thread panicked".to_owned(),
                        )
                    })??;
                }
                Ok(())
            })?;
        }
        Ok(())
    }

    #[test]
    fn submitted_and_mapping_cancellation_recreate_exact_shader_outputs()
    -> Result<(), NativeShaderError> {
        for phase in [
            CancellationPhase::Submitted,
            CancellationPhase::MappingRequested,
        ] {
            let executor = match WgpuNativeShaderExecutor::new() {
                Ok(executor) => executor,
                Err(NativeShaderError::BackendUnavailable(reason)) => {
                    eprintln!("in-flight cancellation adapter unavailable: {reason}");
                    assert!(!reason.is_empty());
                    return Ok(());
                }
                Err(error) => return Err(error),
            };
            eprintln!(
                "in-flight cancellation adapter: {:?}, phase: {phase:?}",
                executor.adapter_name()
            );
            let cancellation = CancellationToken::default();
            let (reached_sender, reached_receiver) = mpsc::sync_channel(1);
            let (resume_sender, resume_receiver) = mpsc::sync_channel(1);
            std::thread::scope(|scope| -> Result<(), NativeShaderError> {
                let cancellation = &cancellation;
                let worker = scope.spawn(move || -> Result<(), NativeShaderError> {
                    let (backend, authority) =
                        CpuWorkspaceAuthority::create_backend(TEST_MEMORY_LIMIT_BYTES)?;
                    let context = tensor_context(&authority, cancellation)?;
                    let image = ImageTensor::from_f32(
                        &backend, &context, 1, 1, 1, 3, &[0.25, 0.5, 0.75],
                    )?;
                    let request = NativeShaderRequest {
                        fragment_source: "#version 300 es\nprecision highp float;\nuniform sampler2D u_image0;\nin vec2 v_texCoord;\nlayout(location = 0) out vec4 fragColor0;\nvoid main() { fragColor0 = texture(u_image0, v_texCoord); }".to_owned(),
                        images: vec![image],
                        floats: Vec::new(),
                        ints: Vec::new(),
                        bools: Vec::new(),
                        curves: Vec::new(),
                        width: 1,
                        height: 1,
                    };
                    let input_bytes = backend.memory_snapshot().current_bytes;
                    CANCELLATION_RENDEZVOUS.with(|slot| {
                        *slot.borrow_mut() = Some(CancellationRendezvous {
                            phase,
                            reached: reached_sender,
                            resume: resume_receiver,
                        });
                    });
                    let result = executor.execute(&request, &backend, &context);
                    assert!(matches!(result, Err(NativeShaderError::Cancelled)), "{result:?}");
                    assert_eq!(backend.memory_snapshot().current_bytes, input_bytes);
                    drop(result);
                    drop(request);
                    drop(context);
                    drop(executor);
                    assert_eq!(backend.memory_snapshot().current_bytes, 0);
                    Ok(())
                });
                let reached = reached_receiver.recv_timeout(Duration::from_secs(30));
                assert!(cancellation.cancel());
                let resumed = resume_sender.send(());
                let result = worker.join().map_err(|_| {
                    NativeShaderError::DeviceLost("cancellation worker panicked".to_owned())
                });
                let reached = reached.map_err(|error| {
                    NativeShaderError::DeviceLost(format!("test phase was not reached: {error}"))
                })?;
                assert_eq!(reached, phase);
                resumed.map_err(|error| {
                    NativeShaderError::DeviceLost(format!("test phase could not resume: {error}"))
                })?;
                result??;
                Ok(())
            })?;

            let executor = WgpuNativeShaderExecutor::new()?;
            let cancellation = CancellationToken::default();
            let (backend, authority) =
                CpuWorkspaceAuthority::create_backend(TEST_MEMORY_LIMIT_BYTES)?;
            let context = tensor_context(&authority, &cancellation)?;
            for pass_count in [1, 2, 3] {
                let image =
                    ImageTensor::from_f32(&backend, &context, 1, 1, 1, 3, &[0.25, 0.5, 0.75])?;
                let result = executor.execute(
                    &NativeShaderRequest {
                        fragment_source: format!(
                            "#version 300 es\n#pragma passes {pass_count}\nprecision highp float;\nuniform sampler2D u_image0;\nuniform int u_pass;\nin vec2 v_texCoord;\nlayout(location = 0) out vec4 fragColor0;\nlayout(location = 1) out vec4 fragColor1;\nlayout(location = 2) out vec4 fragColor2;\nlayout(location = 3) out vec4 fragColor3;\nvoid main() {{ fragColor0 = texture(u_image0, v_texCoord) + vec4(0.125, 0.0, 0.0, 0.0); fragColor1 = vec4(0.0, 0.25, 0.5, 1.0); fragColor2 = vec4(0.5, 0.0, 0.75, 1.0); fragColor3 = vec4(float(u_pass) * 0.25, 0.5, 0.75, 1.0); }}"
                        ),
                        images: vec![image],
                        floats: Vec::new(),
                        ints: Vec::new(),
                        bools: Vec::new(),
                        curves: Vec::new(),
                        width: 1,
                        height: 1,
                    },
                    &backend,
                    &context,
                )?;
                assert_eq!(result.pass_count, pass_count);
                let actual = result
                    .outputs
                    .iter()
                    .map(ImageTensor::to_f32_vec)
                    .collect::<Result<Vec<_>, _>>()?;
                assert_eq!(
                    actual,
                    vec![
                        vec![0.25 + 0.125 * pass_count as f32, 0.5, 0.75, 1.0],
                        vec![0.0, 0.25, 0.5, 1.0],
                        vec![0.5, 0.0, 0.75, 1.0],
                        vec![(pass_count - 1) as f32 * 0.25, 0.5, 0.75, 1.0],
                    ]
                );
                drop(result);
                assert_eq!(backend.memory_snapshot().current_bytes, 0);
            }
            drop(context);
            drop(executor);
            assert_eq!(backend.memory_snapshot().current_bytes, 0);
        }
        Ok(())
    }

    #[test]
    fn request_bounds_are_fail_closed() {
        let request = NativeShaderRequest {
            fragment_source: "#version 300 es\nvoid main() {}".to_owned(),
            images: Vec::new(),
            floats: Vec::new(),
            ints: Vec::new(),
            bools: Vec::new(),
            curves: Vec::new(),
            width: 0,
            height: 1,
        };
        assert!(matches!(
            validate_request(&request),
            Err(NativeShaderError::Bounds(_))
        ));
    }
}
