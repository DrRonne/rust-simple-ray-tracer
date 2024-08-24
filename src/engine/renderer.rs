
extern crate ocl;
use ocl::{ProQue, Buffer, MemFlags};
use crate::engine::error::RendererError;
use crate::engine::render::RenderObject;
use crate::engine::camera::Camera;

const render_src: &str = r#"
    void cframe_multiply_vector(__constant float *cframe,
                                __private float *pos,
                                __private float *out)
    {
        out[0] = cframe[3] * pos[0] + cframe[4] * pos[1] + cframe[5] * pos[2] + cframe[0];
        out[1] = cframe[6] * pos[0] + cframe[7] * pos[1] + cframe[8] * pos[2] + cframe[1];
        out[2] = cframe[9] * pos[0] + cframe[10] * pos[1] + cframe[11] * pos[2] + cframe[2];
    }

    void matrix_multiplication(__constant float* A,
                               __private float* B,
                               __private float* C,
                               int w)
    {
        for (int i = 0; i < w * w; i++) {
            C[i] = 0.0f;
            int r = i % w;
            int c = i / w;
            for (int j = 0; j < w; j++) {
                C[i] += A[(j * w) + c] * B[(r * w) + j];
            }
        }
    }

    void setup_rotation_from_angles(float alpha,
                                    float beta,
                                    float gamma,
                                    __private float* out)
    {
        float sa = sin(alpha);
        float ca = cos(alpha);
        float sb = sin(beta);
        float cb = cos(beta);
        float sg = sin(gamma);
        float cg = cos(gamma);
        out[0] = cb * cg;
        out[1] = sa * sb * cg - ca * sg;
        out[2] = ca * sb * cg + sa * sg;
        out[3] = cb * sg;
        out[4] = sa * sb * sg + ca * cg;
        out[5] = ca * sb * sg - sa * cg;
        out[6] = -sb;
        out[7] = sa * cb;
        out[8] = ca * cb;
    }

    bool solveQuadratic(const float a, const float b, const float c, 
					    float *x0, float *x1) {
        float discr = b * b - 4 * a * c;
        if (discr < 0) return false;
        else if (discr == 0) *x0 = *x1 = -0.5 * b / a;
        else {
            float q = (b > 0) ?
                -0.5 * (b + sqrt(discr)) :
                -0.5 * (b - sqrt(discr));
            *x0 = q / a;
            *x1 = c / q;
        }
        // if (x0 > x1) std::swap(x0, x1);
        
        return true;
    }

    void intersect_sphere(__constant float *sphere_cframe,
                          float sphere_radius,
                          float *ray_cframe,
                          float *t)
    {
        float Lx = sphere_cframe[0] - ray_cframe[0];
        float Ly = sphere_cframe[1] - ray_cframe[1];
        float Lz = sphere_cframe[2] - ray_cframe[2];

        float a = ray_cframe[5] * ray_cframe[5] + ray_cframe[8] * ray_cframe[8] + ray_cframe[11] * ray_cframe[11];
        float b = 2 * (ray_cframe[5] * Lx + ray_cframe[8] * Ly + ray_cframe[11] * Lz);
        float c = Lx * Lx + Ly * Ly + Lz * Lz - sphere_radius * sphere_radius;
        float t0, t1;
        if (solveQuadratic(a, b, c, &t0, &t1)) {
            if (t0 > 0 && t1 > 0){
                *t = min(t0, t1);
            }
            else if (t0 > 0) {
                *t = t0;
            }
            else if (t1 > 0) {
                *t = t1;
            }
            else {
                *t = -1;
            }
        } else {
            *t = -1;
        }
    }

    int intersect_objects(__constant float* object_cframe,
                          unsigned int object_amnt,
                          float *ray_cframe,
                          __constant float *object_props,
                          uchar prop_size,
                          float *out_t)
    {
        float t = 9999999;
        int index_found = -1;
        for (int i = 0; i < object_amnt; i++)
        {
            float local_t;
            intersect_sphere(&object_cframe[i * 12], object_props[i * prop_size], ray_cframe, &local_t);
            if (local_t > 0 && local_t < t) {
                t = local_t;
                index_found = i;
            }
        }
        *out_t = t;
        return index_found;
    }

    void calculate_normal_vector(__constant float* object_cframe,
                                 int object_index,
                                 __constant float *object_props,
                                 uchar prop_size,
                                 float *edge_pos,
                                 float *out_normal)
    {
        float normal[3] = { edge_pos[0] - object_cframe[object_index * 12], edge_pos[1] - object_cframe[(object_index * 12) + 1], edge_pos[2] - object_cframe[(object_index * 12) + 2] };
        float normal_size = sqrt((normal[0] * normal[0]) + (normal[1] * normal[1]) + (normal[2] * normal[2]));
        out_normal[0] = normal[0] / normal_size;
        out_normal[1] = normal[1] / normal_size;
        out_normal[2] = normal[2] / normal_size;
    }

    void render_pixel(__global uchar *output_buffer,
                      __constant float* object_cframe,
                      unsigned int object_amnt,
                      __constant float *camera_cframe,
                      float *ray_rotation_matrix,
                      __constant float *object_props,
                      uchar prop_size,
                      __constant uchar *color,
                      __constant float *directionlight_direction,
                      __constant uchar *directionlight_color)
    {
        float ray_cframe[12] = { camera_cframe[0], camera_cframe[1], camera_cframe[2],
                                 ray_rotation_matrix[0], ray_rotation_matrix[1], ray_rotation_matrix[2],
                                 ray_rotation_matrix[3], ray_rotation_matrix[4], ray_rotation_matrix[5],
                                 ray_rotation_matrix[6], ray_rotation_matrix[7], ray_rotation_matrix[8] };
        
        float t;
        int intersection_index = intersect_objects(object_cframe,
                                                   object_amnt,
                                                   ray_cframe,
                                                   object_props,
                                                   prop_size,
                                                   &t);

        if (intersection_index >= 0)
        {
            float edge_pos[3] = { ray_cframe[0] - (ray_cframe[5] * t), ray_cframe[1] - (ray_cframe[8] * t), ray_cframe[2] - (ray_cframe[11] * t) };
            float normal[3] = { 0.0f, 0.0f, 0.0f };
            calculate_normal_vector(object_cframe,
                                    intersection_index,
                                    object_props,
                                    prop_size,
                                    edge_pos,
                                    normal);
            // The calculated edge_pos can be slightly inside inside the object, causing the ray to calculate the shadow to collide with the object itself.
            // This is due to floating point precision.
            // To combat this, take the starting point of the ray at a distance of "correction_factor" more outwards of the object.
            float correction_factor = 0.01;
            float corrected_edge_pos[3] = { edge_pos[0] + (normal[0] * correction_factor), edge_pos[1] + (normal[1] * correction_factor), edge_pos[2] + (normal[2] * correction_factor) };
            float edge_to_dir_light[12] = { corrected_edge_pos[0], corrected_edge_pos[1], corrected_edge_pos[2],
                                            0.0, 0.0, directionlight_direction[0],
                                            0.0, 0.0, directionlight_direction[1],
                                            0.0, 0.0, directionlight_direction[2] };
            float dl_t;
            int dl_int_index = intersect_objects(object_cframe,
                                                 object_amnt,
                                                 edge_to_dir_light,
                                                 object_props,
                                                 prop_size,
                                                 &dl_t);
            if (dl_int_index < 0 || (dl_int_index == intersection_index))
            {
                float diffuseFactor = fmax(normal[0] * (-directionlight_direction[0]) + normal[1] * (-directionlight_direction[1]) + normal[2] * (-directionlight_direction[2]), 0.0f);
                float directional_diffuse_light_color[3] = { directionlight_color[0] * diffuseFactor / 0xff, directionlight_color[1] * diffuseFactor / 0xff, directionlight_color[2] * diffuseFactor / 0xff };

                output_buffer[get_global_id(0) * 4] = (uchar) (((float) color[intersection_index * 3]) * directional_diffuse_light_color[0]);
                output_buffer[get_global_id(0) * 4 + 1] = (uchar) (((float) color[intersection_index * 3 + 1]) * directional_diffuse_light_color[1]);
                output_buffer[get_global_id(0) * 4 + 2] = (uchar) (((float) color[intersection_index * 3 + 2]) * directional_diffuse_light_color[2]);
                output_buffer[get_global_id(0) * 4 + 3] = 0xff;
            } else {
                output_buffer[get_global_id(0) * 4] = 0x00;
                output_buffer[get_global_id(0) * 4 + 1] = 0x00;
                output_buffer[get_global_id(0) * 4 + 2] = 0x00;
                output_buffer[get_global_id(0) * 4 + 3] = 0xff;
            }
        } else {
            output_buffer[get_global_id(0) * 4] = 0x00;
            output_buffer[get_global_id(0) * 4 + 1] = 0x00;
            output_buffer[get_global_id(0) * 4 + 2] = 0x00;
            output_buffer[get_global_id(0) * 4 + 3] = 0xff;
        }
    }
    
    __kernel void render(__constant uchar *buffer,
                         __global uchar *output_buffer,
                         ushort width,
                         ushort height,
                         __constant float *camera,
                         float camera_width,
                         float camera_height,
                         float focal_length,
                         __constant float *object_cframe,
                         unsigned int object_amnt,
                         __constant float *object_props,
                         uchar prop_size,
                         __constant uchar *color,
                         __constant float *directionlight_direction,
                         __constant uchar *directionlight_color) {
        int x = get_global_id(0) % width;
        int y = get_global_id(0) / width;
        float cam_x = - (camera_width / 2) + (((float) x / (float) width) * camera_width);
        float cam_y = - (camera_height / 2) + (((float) y / (float) height) * camera_height);
        float alpha_dist = sqrt((focal_length * focal_length) + (cam_y * cam_y));
        float alpha = asin(cam_y / alpha_dist);
        float beta_dist = sqrt((focal_length * focal_length) + (cam_x * cam_x));
        float beta = asin(cam_x / beta_dist);
        float cam_ray_rotation[] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        setup_rotation_from_angles(alpha, beta, 0.0f, cam_ray_rotation);
        float cam_ray[] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        matrix_multiplication(&camera[3], cam_ray_rotation, cam_ray, 3);
        render_pixel(output_buffer, object_cframe, object_amnt, camera, cam_ray, object_props, prop_size, color, directionlight_direction, directionlight_color);
    }
"#;

pub struct Renderer {
    width: u32,
    height: u32,
    pro_que: Option<ProQue>,
    buffer: Option<Buffer<u8>>,
    output_buffer: Option<Buffer<u8>>,
}

impl Renderer {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            pro_que: None,
            buffer: None,
            output_buffer: None,
         }
    }

    pub fn init(&mut self) -> Result<(), RendererError> {
        self.pro_que = Some(ProQue::builder()
            .src(render_src)
            .dims(self.width * self.height)
            .build().map_err(|e| RendererError::KernelBuildError(e))?);
        
        self.buffer = Some(self.pro_que.as_mut().ok_or(RendererError::RendererNotInitializedError)?.create_buffer::<u8>().map_err(|e| RendererError::CreateBufferError(e))?);
        self.output_buffer = Some(Buffer::builder().queue(self.pro_que.as_mut().ok_or(RendererError::RendererNotInitializedError)?.queue().clone())
            .flags(MemFlags::new().read_write())
            .len(self.width * self.height * 4)
            .build().map_err(|e| RendererError::CreateBufferError(e))?);
        
        Ok(())
    }

    pub fn render_frame(&mut self, mut camera: Camera, mut render_objects: Vec<RenderObject>, directionlight_direction: Vec<f32>, directionlight_color: Vec<u8>) -> Result<Vec::<u8>, RendererError> {
        let c_width = u16::try_from(self.width).map_err(|_| RendererError::DimensionsTooBigError)?;
        let c_height = u16::try_from(self.height).map_err(|_| RendererError::DimensionsTooBigError)?;

        let mut cframe_vec = Vec::<f32>::new();
        let mut object_props_vec = Vec::<f32>::new();
        let mut color_vec = Vec::<u8>::new();
        let prop_size: u8 = render_objects[0].get_object_props_vec().len() as u8;
        for obj in render_objects.iter_mut() {
            cframe_vec.extend(obj.convert_to_cframe_buffer());
            object_props_vec.extend(obj.get_object_props_vec());
            color_vec.extend(obj.get_color_vec());
        }

        let cframe_buffer = Buffer::builder().queue(self.pro_que.as_mut().ok_or(RendererError::RendererNotInitializedError)?.queue().clone())
            .flags(MemFlags::new().read_write())
            .len(cframe_vec.len())
            .copy_host_slice(&cframe_vec)
            .build().map_err(|e| RendererError::CreateBufferError(e))?;

        let object_prop_buffer = Buffer::builder().queue(self.pro_que.as_mut().ok_or(RendererError::RendererNotInitializedError)?.queue().clone())
            .flags(MemFlags::new().read_write())
            .len(object_props_vec.len())
            .copy_host_slice(&object_props_vec)
            .build().map_err(|e| RendererError::CreateBufferError(e))?;

        let color_buffer = Buffer::builder().queue(self.pro_que.as_mut().ok_or(RendererError::RendererNotInitializedError)?.queue().clone())
            .flags(MemFlags::new().read_write())
            .len(color_vec.len())
            .copy_host_slice(&color_vec)
            .build().map_err(|e| RendererError::CreateBufferError(e))?;

        let camera_buffer = Buffer::builder().queue(self.pro_que.as_mut().ok_or(RendererError::RendererNotInitializedError)?.queue().clone())
            .flags(MemFlags::new().read_write())
            .len(13)
            .copy_host_slice(&camera.to_vec())
            .build().map_err(|e| RendererError::CreateBufferError(e))?;

        let directionlight_direction_buffer = Buffer::builder().queue(self.pro_que.as_mut().ok_or(RendererError::RendererNotInitializedError)?.queue().clone())
            .flags(MemFlags::new().read_write())
            .len(3)
            .copy_host_slice(&directionlight_direction)
            .build().map_err(|e| RendererError::CreateBufferError(e))?;

        let directionlight_color_buffer = Buffer::builder().queue(self.pro_que.as_mut().ok_or(RendererError::RendererNotInitializedError)?.queue().clone())
            .flags(MemFlags::new().read_write())
            .len(3)
            .copy_host_slice(&directionlight_color)
            .build().map_err(|e| RendererError::CreateBufferError(e))?;

        let focal_length = camera.get_focal_length();
        let horizontal_fov = camera.get_fov();
        let horizontal_fov_rad = horizontal_fov / 180.0 * std::f32::consts::PI;
        // law of sines
        let camera_width = (focal_length * (horizontal_fov_rad / 2f32).sin()) / (((std::f32::consts::PI / 2.0) - (horizontal_fov_rad / 2f32)).sin());
        let camera_height = camera_width * ((self.height as f32) / (self.width as f32));

        let kernel = self.pro_que.as_mut().ok_or(RendererError::RendererNotInitializedError)?.kernel_builder("render")
            .arg(self.buffer.as_ref().ok_or(RendererError::RendererNotInitializedError)?)
            .arg(self.output_buffer.as_ref().ok_or(RendererError::RendererNotInitializedError)?)
            .arg(c_width)
            .arg(c_height)
            .arg(camera_buffer)
            .arg(camera_width)
            .arg(camera_height)
            .arg(focal_length)
            .arg(cframe_buffer)
            .arg((cframe_vec.len() / 12) as u32)
            .arg(object_prop_buffer)
            .arg(prop_size)
            .arg(color_buffer)
            .arg(directionlight_direction_buffer)
            .arg(directionlight_color_buffer)
            .build().map_err(|e| RendererError::AddArgumentsError(e))?;

        unsafe { kernel.enq().map_err(|e| RendererError::ExecuteKernelError(e))?; }

        let mut vec = vec![0u8; self.output_buffer.as_ref().ok_or(RendererError::RendererNotInitializedError)?.len()];
        self.output_buffer.as_ref().ok_or(RendererError::RendererNotInitializedError)?.read(&mut vec).enq().map_err(|e| RendererError::ReadBufferError(e))?;

        return Ok(vec);
    }
}