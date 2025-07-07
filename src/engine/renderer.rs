
extern crate ocl;
use ocl::{ProQue, Buffer, MemFlags};
use crate::engine::error::RendererError;
use crate::engine::render::RenderObject;
use crate::engine::camera::Camera;
use crate::engine::sphere_merge::SphereMerge;

const render_src: &str = r#"
    #pragma OPENCL EXTENSION cl_amd_printf : enable

    #define MAXIMUM_BOUNCES 10
    #define MAXIMUM_TRANSPARENCY_BOUNCES 5 // Maximum amount of transparent objects traversed to calculate shadow color.
    #define CORRECTION_FACTOR 0.01 // Correction factor to prevent ray from colliding with the object itself due to floating point precision issues.
    #define AIR_REFRACTIVE_INDEX 1.0 // Refractive index of air, used for calculating the refracted ray.
    #define MAXIMUM_MERGES 6
    #define MAXIMUM_MARCHING_STEPS 100 // Maximum amount of steps to take when marching the ray.
    #define STEP_MARGIN 0.01f // How close the ray needs to get for it to count as an intersection
    #define STEP_MULTIPLIER 0.8f // When the ray closes in on an object, this multiplier will be used to have smaller steps (or bigger if larger than 1.0f)

    typedef struct {
        float cframe[12];    // 48 bytes
        float contribution;  // 4 bytes
        int surface_index;   // 4 bytes
        uchar color[4];      // 4 bytes
    } RayStackEntry;

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

    float calculate_squared_distance_point_to_line(__constant float *point,
                                                   float *ray)
    {
        float origin_point_diff[3] = { ray[0] - point[0],
                                       ray[1] - point[1],
                                       ray[2] - point[2] };
        float dot = (origin_point_diff[0] * -ray[5]) + (origin_point_diff[1] * -ray[8]) + (origin_point_diff[2] * -ray[11]);
        float projected_length[3] = { -ray[5] * dot,
                                      -ray[8] * dot,
                                      -ray[11] * dot };
        float diff[3] = { origin_point_diff[0] - projected_length[0],
                          origin_point_diff[1] - projected_length[1],
                          origin_point_diff[2] - projected_length[2] };
        return (diff[0] * diff[0]) + (diff[1] * diff[1]) + (diff[2] * diff[2]);
    }

    void find_point_on_line_at_distance_from_point(__constant float *point,
                                                   float *ray,
                                                   float distance,
                                                   float *out_t)
    {
        float point_origin_diff[3] = { ray[0] - point[0],
                                       ray[1] - point[1],
                                       ray[2] - point[2] };
        float a = (ray[5] * ray[5]) + (ray[8] * ray[8]) + (ray[11] * ray[11]);
        float b = -2.0f * (ray[5] * point_origin_diff[0] + ray[8] * point_origin_diff[1] + ray[11] * point_origin_diff[2]);
        float c = (point_origin_diff[0] * point_origin_diff[0]) + 
                  (point_origin_diff[1] * point_origin_diff[1]) + 
                  (point_origin_diff[2] * point_origin_diff[2]) - (distance * distance);
        float t0, t1;
        if (solveQuadratic(a, b, c, &t0, &t1)) {
            if (t0 > 0 && t1 > 0) {
                *out_t = min(t0, t1);
            } else if (t0 > 0) {
                *out_t = t0;
            } else if (t1 > 0) {
                *out_t = t1;
            } else {
                *out_t = -1.0f; // No solution
            }
        } else {
            *out_t = -1.0f; // No solution
        }
    }

    float smooth_min(float a,
                     float b,
                     float k)
    {
        float h = fmax(k - fabs(a - b), 0.0f);
        return fmin(a, b) - (h * h * 0.25f / k);
    }

    float sphere_sdf(__constant float *sphere_cframe,
                     float sphere_radius,
                     float *position)
    {
        float diff[3] = { sphere_cframe[0] - position[0],
                          sphere_cframe[1] - position[1],
                          sphere_cframe[2] - position[2] };
        return sqrt((diff[0] * diff[0]) + (diff[1] * diff[1]) + (diff[2] * diff[2])) - sphere_radius;
    }

    float merge_sphere_sdf(__constant float *object_cframe,
                           int index,
                           __constant float *object_props,
                           uchar prop_size,
                           __constant uint *merge_indices,
                           __constant float *merge_radii,
                           float *position)
    {
        float sdf = sphere_sdf(&object_cframe[index * 12], object_props[index * prop_size], position);
        for (int m = 0; m < MAXIMUM_MERGES; m++)
        {
            int local_index = index * MAXIMUM_MERGES + m;
            uint m_index = merge_indices[local_index];
            if (merge_radii[local_index] < 0.01f)
                break;
            float other_sdf = sphere_sdf(&object_cframe[m_index * 12], object_props[m_index * prop_size], position);
            sdf = smooth_min(other_sdf, sdf, merge_radii[local_index]);
        }
        return sdf;
    }

    void calculate_sdf_normal(
        float *position,
        __constant float *object_cframe,
        int index,
        __constant float *object_props,
        uchar prop_size,
        __constant uint *merge_indices,
        __constant float *merge_radii,
        float *out_normal
    ) {
        float epsilon = 0.001f;
        float dx[3] = {epsilon, 0.0f, 0.0f};
        float dy[3] = {0.0f, epsilon, 0.0f};
        float dz[3] = {0.0f, 0.0f, epsilon};
        float p[3];

        // X
        p[0] = position[0] + epsilon; p[1] = position[1]; p[2] = position[2];
        float sdf_x1 = merge_sphere_sdf(object_cframe, index, object_props, prop_size, merge_indices, merge_radii, p);
        p[0] = position[0] - epsilon;
        float sdf_x0 = merge_sphere_sdf(object_cframe, index, object_props, prop_size, merge_indices, merge_radii, p);

        // Y
        p[0] = position[0]; p[1] = position[1] + epsilon; p[2] = position[2];
        float sdf_y1 = merge_sphere_sdf(object_cframe, index, object_props, prop_size, merge_indices, merge_radii, p);
        p[1] = position[1] - epsilon;
        float sdf_y0 = merge_sphere_sdf(object_cframe, index, object_props, prop_size, merge_indices, merge_radii, p);

        // Z
        p[1] = position[1]; p[2] = position[2] + epsilon;
        float sdf_z1 = merge_sphere_sdf(object_cframe, index, object_props, prop_size, merge_indices, merge_radii, p);
        p[2] = position[2] - epsilon;
        float sdf_z0 = merge_sphere_sdf(object_cframe, index, object_props, prop_size, merge_indices, merge_radii, p);

        out_normal[0] = (sdf_x1 - sdf_x0) / (2.0f * epsilon);
        out_normal[1] = (sdf_y1 - sdf_y0) / (2.0f * epsilon);
        out_normal[2] = (sdf_z1 - sdf_z0) / (2.0f * epsilon);

        // Normalize
        float len = sqrt(out_normal[0]*out_normal[0] + out_normal[1]*out_normal[1] + out_normal[2]*out_normal[2]);
        if (len > 1e-6f) {
            out_normal[0] /= len;
            out_normal[1] /= len;
            out_normal[2] /= len;
        } else {
            out_normal[0] = 0.0f;
            out_normal[1] = 0.0f;
            out_normal[2] = 1.0f;
        }
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

    void calculate_metaball_normal(float *position,
                                   __constant float *object_cframe,
                                   __constant float *object_props,
                                   uchar prop_size,
                                   int main_index,
                                   __constant uint *merge_indices,
                                   __constant float *merge_radii,
                                   float *out_normal) 
    {
        // Start with zero normal
        out_normal[0] = 0.0f;
        out_normal[1] = 0.0f;
        out_normal[2] = 0.0f;

        // Main sphere
        float dx = position[0] - object_cframe[main_index * 12];
        float dy = position[1] - object_cframe[main_index * 12 + 1];
        float dz = position[2] - object_cframe[main_index * 12 + 2];
        float r = object_props[main_index * prop_size];
        float dist2 = dx*dx + dy*dy + dz*dz + 1e-6f; // avoid div by zero
        float dist4 = dist2 * dist2;
        out_normal[0] -= -2.0f * r * r * dx / dist4;
        out_normal[1] -= -2.0f * r * r * dy / dist4;
        out_normal[2] -= -2.0f * r * r * dz / dist4;

        // Merged spheres
        for (int m = 0; m < MAXIMUM_MERGES; m++) {
            if (merge_radii[m] < 0.01f)
                break;
            int merge_idx = merge_indices[m];
            float mdx = position[0] - object_cframe[merge_idx * 12];
            float mdy = position[1] - object_cframe[merge_idx * 12 + 1];
            float mdz = position[2] - object_cframe[merge_idx * 12 + 2];
            float mr = object_props[merge_idx * prop_size];
            float mdist2 = mdx*mdx + mdy*mdy + mdz*mdz + 1e-6f;
            float mdist4 = mdist2 * mdist2;
            out_normal[0] -= -2.0f * mr * mr * mdx / mdist4;
            out_normal[1] -= -2.0f * mr * mr * mdy / mdist4;
            out_normal[2] -= -2.0f * mr * mr * mdz / mdist4;
        }

        // Normalize the normal
        float len = sqrt(out_normal[0]*out_normal[0] + out_normal[1]*out_normal[1] + out_normal[2]*out_normal[2]);
        if (len > 1e-6f) {
            out_normal[0] /= len;
            out_normal[1] /= len;
            out_normal[2] /= len;
        } else {
            out_normal[0] = 0.0f;
            out_normal[1] = 0.0f;
            out_normal[2] = 1.0f;
        }
    }

    float calculate_squared_euclidean_distance(__constant float *a,
                                               float *b)
    {
        float dx = a[0] - b[0];
        float dy = a[1] - b[1];
        float dz = a[2] - b[2];
        return (dx * dx) + (dy * dy) + (dz * dz);
    }

    float squared_sphere_sdf(__constant float *sphere_cframe,
                             float sphere_radius,
                             float *point)
    {
        return calculate_squared_euclidean_distance(sphere_cframe, point) - (sphere_radius * sphere_radius);
    }

    float sphere_field_function(float sphere_radius,
                                float squared_distance)
    {
        return (sphere_radius * sphere_radius) / squared_distance;
    }

    float calculate_manhattan_distance(__constant float *a,
                                       float *b)
    {
        return fabs(a[0] - b[0]) + fabs(a[1] - b[1]) + fabs(a[2] - b[2]);
    }

    float intersect_object(__constant float* object_cframe,
                           __constant float *object_props,
                           int index,
                           uchar prop_size,
                           float *ray_cframe,
                           __constant uint *merge_indices,
                           __constant float *merge_radii,
                           float *step_size)
    {
        float local_t = 0.0f;
        float step_position[3] = { ray_cframe[0],
                                   ray_cframe[1],
                                   ray_cframe[2] };
        *step_size = merge_sphere_sdf(object_cframe, index, object_props, prop_size, merge_indices, merge_radii, step_position) * STEP_MULTIPLIER;
        if (*step_size < 0)
        {
            // inside sphere, stop
            return 99999.0f;
        }
        int steps = 0;
        for (int i = 0; i < MAXIMUM_MARCHING_STEPS; i++)
        {
            steps++;
            if (*step_size < 0)
                printf("step size is %f", *step_size);
            if (fabs(*step_size) <= STEP_MARGIN)
            {
                break;
            }
            if (*step_size < 0)
                printf("step size is %f", *step_size);
            local_t += *step_size;
            step_position[0] = ray_cframe[0] - (local_t * ray_cframe[5]);
            step_position[1] = ray_cframe[1] - (local_t * ray_cframe[8]);
            step_position[2] = ray_cframe[2] - (local_t * ray_cframe[11]);
            *step_size = merge_sphere_sdf(object_cframe, index, object_props, prop_size, merge_indices, merge_radii, step_position) * STEP_MULTIPLIER;
        }
        return local_t;
    }

    int intersect_objects(__constant float* object_cframe,
                          unsigned int object_amnt,
                          __constant uint *merge_indices,
                          __constant float *merge_radii,
                          float *ray_cframe,
                          __constant float *object_props,
                          uchar prop_size,
                          float *out_t,
                          float *out_position,
                          float *out_normal)
    {
        float t = 9999999;
        int index_found = -1;
        for (int i = 0; i < object_amnt - 1; i++)
        {
            float local_t = 0.0f;
            float squared_distance = calculate_squared_distance_point_to_line(&object_cframe[i * 12], ray_cframe);
            // Use squared distance to avoid unnecessary square root calculation
            if (squared_distance > (object_props[i * prop_size] + 0.01f) * (object_props[i * prop_size] + 0.01f) * 2.0f)
            {
                // Check if the merged spheres are not too close either
                int close_merge_found = 0;
                for (int m = 0; m < MAXIMUM_MERGES; m++)
                {
                    int radius_index = (i * MAXIMUM_MERGES) + m;
                    int m_index = merge_indices[radius_index];
                    if (merge_radii[radius_index] < 0.01f)
                        break;
                    float merge_squared_distance = calculate_squared_distance_point_to_line(&object_cframe[m_index * 12], ray_cframe);
                    if (merge_squared_distance < (object_props[m_index * prop_size] + 0.01f) * (object_props[m_index * prop_size] + 0.01f) * 2.0f)
                    {
                        close_merge_found = 1;
                        break;
                    }
                }
                if (close_merge_found == 0)
                {
                    // No merges close enough
                    continue;
                }
                // continue; // Skip this object if the ray is too far away
            }
            float step_size = 99999.0f;
            local_t = intersect_object(object_cframe,
                                       object_props,
                                       i,
                                       prop_size,
                                       ray_cframe,
                                       merge_indices,
                                       merge_radii,
                                       &step_size);
            if (local_t < t && step_size < STEP_MARGIN)
            {
                index_found = i;
                t = local_t;
            }
        }
        if (index_found >= 0)
        {
            out_position[0] = ray_cframe[0] - (ray_cframe[5] * t);
            out_position[1] = ray_cframe[1] - (ray_cframe[8] * t);
            out_position[2] = ray_cframe[2] - (ray_cframe[11] * t);
            calculate_sdf_normal(out_position,
                                 object_cframe,
                                 index_found,
                                 object_props,
                                 prop_size,
                                 merge_indices,
                                 merge_radii,
                                 out_normal);
            *out_t = t;
        }
        return index_found;
    }

    void get_intersection_from_ray(uchar *out_color,
                                   int *intersection_index,
                                   float *out_normal,
                                   float *out_position,
                                   __constant float* object_cframe,
                                   __constant uint *merge_indices,
                                   __constant float *merge_radii,
                                   unsigned int object_amnt,
                                   float *ray_cframe,
                                   __constant float *object_props,
                                   uchar prop_size,
                                   __constant uchar *color,
                                   __constant float *reflectance,
                                   __constant float *transparency,
                                   __constant float *refractive_index,
                                   __constant float *directionlight_direction,
                                   __constant uchar *directionlight_color)
    {
        float t;
        float edge_pos[3];
        *intersection_index = intersect_objects(object_cframe,
                                                object_amnt,
                                                merge_indices,
                                                merge_radii,
                                                ray_cframe,
                                                object_props,
                                                prop_size,
                                                &t,
                                                edge_pos,
                                                out_normal);

        if (*intersection_index >= 0)
        {
            // The calculated edge_pos can be slightly inside inside the object, causing the ray to calculate the shadow to collide with the object itself.
            // This is due to floating point precision.
            // To combat this, take the starting point of the ray at a distance of "CORRECTION_FACTOR" more outwards of the object.
            out_position[0] = edge_pos[0] + out_normal[0] * CORRECTION_FACTOR;
            out_position[1] = edge_pos[1] + out_normal[1] * CORRECTION_FACTOR;
            out_position[2] = edge_pos[2] + out_normal[2] * CORRECTION_FACTOR;
            float diffuseFactor;
            if (transparency[*intersection_index] >= 0.01f)
            {
                diffuseFactor = 1.0f; // Transparent objects don't have a shade
            }
            else
            {
                diffuseFactor = fmax(out_normal[0] * (-directionlight_direction[0]) + out_normal[1] * (-directionlight_direction[1]) + out_normal[2] * (-directionlight_direction[2]), 0.0f);
            }
            float directional_diffuse_light_color[3] = { directionlight_color[0] * diffuseFactor / 0xff, directionlight_color[1] * diffuseFactor / 0xff, directionlight_color[2] * diffuseFactor / 0xff };
            out_color[0] = (uchar) (((float) color[*intersection_index * 3]) * directional_diffuse_light_color[0]);
            out_color[1] = (uchar) (((float) color[*intersection_index * 3 + 1]) * directional_diffuse_light_color[1]);
            out_color[2] = (uchar) (((float) color[*intersection_index * 3 + 2]) * directional_diffuse_light_color[2]);
            float edge_to_dir_light[12] = { out_position[0], out_position[1], out_position[2],
                                            0.0, 0.0, directionlight_direction[0],
                                            0.0, 0.0, directionlight_direction[1],
                                            0.0, 0.0, directionlight_direction[2] };
            int shadow_intersections = 0;
            while (shadow_intersections < MAXIMUM_TRANSPARENCY_BOUNCES)
            {
                float dl_t;
                float light_edge_pos[3];
                float light_normal[3];
                int dl_int_index = intersect_objects(object_cframe,
                                                     object_amnt,
                                                     merge_indices,
                                                     merge_radii,
                                                     edge_to_dir_light,
                                                     object_props,
                                                     prop_size,
                                                     &dl_t,
                                                     light_edge_pos,
                                                     light_normal);
                if (dl_int_index < 0 || (dl_int_index == *intersection_index))
                {
                    break;
                }
                else if (transparency[dl_int_index] < 0.01f)
                {
                    out_color[0] = 0x00;
                    out_color[1] = 0x00;
                    out_color[2] = 0x00;
                    out_color[3] = 0xff;
                    break;
                }
                else
                {
                    uchar filtered_light_r = directionlight_color[0] * (transparency[dl_int_index] * 1.0f + (1.0f - transparency[dl_int_index]) * (color[dl_int_index * 3] / 255.0f));
                    uchar filtered_light_g = directionlight_color[1] * (transparency[dl_int_index] * 1.0f + (1.0f - transparency[dl_int_index]) * (color[dl_int_index * 3 + 1] / 255.0f));
                    uchar filtered_light_b = directionlight_color[2] * (transparency[dl_int_index] * 1.0f + (1.0f - transparency[dl_int_index]) * (color[dl_int_index * 3 + 2] / 255.0f));
                    out_color[0] = (uchar) (((float) out_color[0]) * ((float) filtered_light_r / 255.0f));
                    out_color[1] = (uchar) (((float) out_color[1]) * ((float) filtered_light_g / 255.0f));
                    out_color[2] = (uchar) (((float) out_color[2]) * ((float) filtered_light_b / 255.0f));
                    edge_to_dir_light[0] = edge_to_dir_light[0] - (directionlight_direction[0] * (dl_t + (CORRECTION_FACTOR * 2)));
                    edge_to_dir_light[1] = edge_to_dir_light[1] - (directionlight_direction[1] * (dl_t + (CORRECTION_FACTOR * 2)));
                    edge_to_dir_light[2] = edge_to_dir_light[2] - (directionlight_direction[2] * (dl_t + (CORRECTION_FACTOR * 2)));
                    float step_size = 99999.0f;
                    float light_edge_pos[3];
                    float light_normal[3];
                    float other_side_ray[12] = { edge_to_dir_light[0] - (directionlight_direction[0] * object_props[dl_int_index * prop_size] * 3),
                                                edge_to_dir_light[1] - (directionlight_direction[1] * object_props[dl_int_index * prop_size] * 3),
                                                edge_to_dir_light[2] - (directionlight_direction[2] * object_props[dl_int_index * prop_size] * 3),
                                                0.0f, 0.0f, -directionlight_direction[0],
                                                0.0f, 0.0f, -directionlight_direction[1],
                                                0.0f, 0.0f, -directionlight_direction[2] };
                    float dl_t2 = intersect_object(object_cframe,
                                                   object_props,
                                                   dl_int_index,
                                                   prop_size,
                                                   other_side_ray,
                                                   merge_indices,
                                                   merge_radii,
                                                   &step_size);
                    if (step_size < STEP_MARGIN)
                    {
                        edge_to_dir_light[0] = other_side_ray[0] - (other_side_ray[5] * (dl_t2 - CORRECTION_FACTOR * 5));
                        edge_to_dir_light[1] = other_side_ray[1] - (other_side_ray[8] * (dl_t2 - CORRECTION_FACTOR * 5));
                        edge_to_dir_light[2] = other_side_ray[2] - (other_side_ray[11] * (dl_t2 - CORRECTION_FACTOR * 5));
                    }
                    else
                    {
                        break;
                    }
                }
            }
        } else {
            out_color[0] = 0x00;
            out_color[1] = 0x00;
            out_color[2] = 0x00;
            out_color[3] = 0xff;
        }
    }

    void calculate_reflected_ray(float *incoming_ray,
                                 float *normal,
                                 float *position,
                                 float *reflected_ray)
    {
        float double_normal = 2.0f * (normal[0] * incoming_ray[5] + normal[1] * incoming_ray[8] + normal[2] * incoming_ray[11]);
        float reflected_dir_x = incoming_ray[5] - (normal[0] * double_normal);
        float reflected_dir_y = incoming_ray[8] - (normal[1] * double_normal);
        float reflected_dir_z = incoming_ray[11] - (normal[2] * double_normal);
        float len = sqrt(reflected_dir_x * reflected_dir_x +
            reflected_dir_y * reflected_dir_y +
            reflected_dir_z * reflected_dir_z);
        if (len > 1e-6f) {
            reflected_dir_x /= len;
            reflected_dir_y /= len;
            reflected_dir_z /= len;
        } else {
            reflected_dir_x = 0.0f;
            reflected_dir_y = 0.0f;
            reflected_dir_z = 1.0f;
        }
        reflected_ray[0] = position[0];
        reflected_ray[1] = position[1];
        reflected_ray[2] = position[2];
        reflected_ray[3] = incoming_ray[3];
        reflected_ray[4] = incoming_ray[4];
        reflected_ray[5] = reflected_dir_x;
        reflected_ray[6] = incoming_ray[6];
        reflected_ray[7] = incoming_ray[7];
        reflected_ray[8] = reflected_dir_y;
        reflected_ray[9] = incoming_ray[9];
        reflected_ray[10] = incoming_ray[10];
        reflected_ray[11] = reflected_dir_z;
    }

    void calculate_refracted_ray(float *normal,
                                 float *incoming_ray,
                                 float *position,
                                 float *outgoing_ray,
                                 float n1,
                                 float n2)
    {
        float n = n1 / n2;
        // Use the correct indices for the direction vector
        float dir_x = incoming_ray[5];
        float dir_y = incoming_ray[8];
        float dir_z = incoming_ray[11];

        float cosI = (normal[0] * dir_x + normal[1] * dir_y + normal[2] * dir_z);
        float sinT2 = n * n * (1.0 - cosI * cosI);
        if (sinT2 > 1.0)
        {
            // Total internal reflection, no refraction
            return;
        }
        float cosT = sqrt(1.0 - sinT2);

        float refracted_x = n * dir_x - (n * cosI - cosT) * normal[0];
        float refracted_y = n * dir_y - (n * cosI - cosT) * normal[1];
        float refracted_z = n * dir_z - (n * cosI - cosT) * normal[2];

        // Normalize the refracted direction
        float len = sqrt(refracted_x * refracted_x + refracted_y * refracted_y + refracted_z * refracted_z);
        if (len > 1e-6f) {
            refracted_x /= len;
            refracted_y /= len;
            refracted_z /= len;
        } else {
            refracted_x = 0.0f;
            refracted_y = 0.0f;
            refracted_z = 1.0f;
        }

        outgoing_ray[0] = position[0];
        outgoing_ray[1] = position[1];
        outgoing_ray[2] = position[2];
        outgoing_ray[3] = incoming_ray[3];
        outgoing_ray[4] = incoming_ray[4];
        outgoing_ray[5] = refracted_x;
        outgoing_ray[6] = incoming_ray[6];
        outgoing_ray[7] = incoming_ray[7];
        outgoing_ray[8] = refracted_y;
        outgoing_ray[9] = incoming_ray[9];
        outgoing_ray[10] = incoming_ray[10];
        outgoing_ray[11] = refracted_z;
    }

    void calculate_transparency_ray(float *normal,
                                    float *incoming_ray,
                                    float *position,
                                    float *outgoing_ray,
                                    float n1,
                                    __constant float *object_cframe,
                                    __constant float* object_props,
                                    uchar prop_size,
                                    int intersection_index)
    {
        float radius = object_props[intersection_index * prop_size];
        float internal_ray[12];
        float corrected_position[3] = { position[0] - (normal[0] * CORRECTION_FACTOR * 5),
                                        position[1] - (normal[1] * CORRECTION_FACTOR * 5),
                                        position[2] - (normal[2] * CORRECTION_FACTOR * 5) };
        calculate_refracted_ray(normal, incoming_ray, corrected_position, internal_ray, AIR_REFRACTIVE_INDEX, n1);
        float local_t = 0.0f;
        intersect_sphere(&object_cframe[intersection_index * 12], radius, internal_ray, &local_t);
        float internal_edge_pos[3] = { internal_ray[0] - (internal_ray[5] * local_t), internal_ray[1] - (internal_ray[8] * local_t), internal_ray[2] - (internal_ray[11] * local_t) };
        float internal_normal[3] = { 0.0f, 0.0f, 0.0f };
        calculate_normal_vector(object_cframe,
                                intersection_index,
                                object_props,
                                prop_size,
                                internal_edge_pos,
                                internal_normal);
        internal_edge_pos[0] += (internal_normal[0] * CORRECTION_FACTOR * 5);
        internal_edge_pos[1] += (internal_normal[1] * CORRECTION_FACTOR * 5);
        internal_edge_pos[2] += (internal_normal[2] * CORRECTION_FACTOR * 5);
        if (local_t > 0.0f)
        {
            float inversed_internal_normal[3] = { -internal_normal[0], -internal_normal[1], -internal_normal[2] };
            calculate_refracted_ray(inversed_internal_normal, internal_ray, internal_edge_pos, outgoing_ray, n1, AIR_REFRACTIVE_INDEX);
        }
    }

    void render_pixel(__global uchar *output_buffer,
                      __constant float* object_cframe,
                      unsigned int object_amnt,
                      __constant uint *merge_indices,
                      __constant float *merge_radii,
                      __constant float *camera_cframe,
                      float *ray_rotation_matrix,
                      __constant float *object_props,
                      uchar prop_size,
                      __constant uchar *color,
                      __constant float *reflectance,
                      __constant float *transparency,
                      __constant float *refractive_index,
                      __constant float *directionlight_direction,
                      __constant uchar *directionlight_color)
    {
        float ray_cframe[12] = { camera_cframe[0], camera_cframe[1], camera_cframe[2],
                                 ray_rotation_matrix[0], ray_rotation_matrix[1], ray_rotation_matrix[2],
                                 ray_rotation_matrix[3], ray_rotation_matrix[4], ray_rotation_matrix[5],
                                 ray_rotation_matrix[6], ray_rotation_matrix[7], ray_rotation_matrix[8] };
        RayStackEntry ray_stack[MAXIMUM_BOUNCES];
        int stack_ptr = 0;
        int maximum_surface_index = 0;
        for (int i = 0; i < MAXIMUM_BOUNCES; i++) {
            ray_stack[i].color[0] = 0x00;
            ray_stack[i].color[1] = 0x00;
            ray_stack[i].color[2] = 0x00;
            ray_stack[i].color[3] = 0xff;
            ray_stack[i].contribution = 0.0f;
            for (int j = 0; j < 12; j++) {
                ray_stack[i].cframe[j] = 0.0f;
            }
            ray_stack[i].surface_index = -1;
            // 0 means the ray is from the camera and should be added up to the pixel.
            // If it's anything else, then that is the color from the perspective of some surface.
        }

        for (int i = 0; i < 12; i++) {
            ray_stack[stack_ptr].cframe[i] = ray_cframe[i];
        }
        ray_stack[stack_ptr].color[0] = 0x00;
        ray_stack[stack_ptr].color[1] = 0x00;
        ray_stack[stack_ptr].color[2] = 0x00;
        ray_stack[stack_ptr].color[3] = 0xff;
        ray_stack[stack_ptr].contribution = 1.0f;
        ray_stack[stack_ptr].surface_index = 0;

        while (stack_ptr < MAXIMUM_BOUNCES && ray_stack[stack_ptr].surface_index != -1) {
            float result_normal[3] = { 0.0f, 0.0f, 0.0f };
            float result_position[3] = { 0.0f, 0.0f, 0.0f };
            int intersection_index = -1;
            int rays_inserted = 0;
            get_intersection_from_ray(ray_stack[stack_ptr].color,
                                      &intersection_index,
                                      result_normal,
                                      result_position,
                                      object_cframe,
                                      merge_indices,
                                      merge_radii,
                                      object_amnt,
                                      ray_stack[stack_ptr].cframe,
                                      object_props,
                                      prop_size,
                                      color,
                                      reflectance,
                                      transparency,
                                      refractive_index,
                                      directionlight_direction,
                                      directionlight_color);
            if (intersection_index < 0) {
                // No intersection found, just continue to the next ray in the stack
                ray_stack[stack_ptr].surface_index = -1;
                stack_ptr++;
                continue;
            }

            if (reflectance[intersection_index] > 0.01f && ray_stack[stack_ptr].surface_index == 0) {
                calculate_reflected_ray(ray_stack[stack_ptr].cframe,
                                        result_normal,
                                        result_position,
                                        ray_stack[stack_ptr + rays_inserted + 1].cframe);
                ray_stack[stack_ptr + rays_inserted + 1].contribution = ray_stack[stack_ptr].contribution * reflectance[intersection_index];
                ray_stack[stack_ptr + rays_inserted + 1].surface_index = ray_stack[stack_ptr].surface_index;
                rays_inserted++;
            }

            if (transparency[intersection_index] > 0.01f) {
                calculate_transparency_ray(result_normal,
                                           ray_stack[stack_ptr].cframe,
                                           result_position,
                                           ray_stack[stack_ptr + rays_inserted + 1].cframe,
                                           refractive_index[intersection_index],
                                           object_cframe,
                                           object_props,
                                           prop_size,
                                           intersection_index);
                if (ray_stack[stack_ptr].surface_index == 0) {
                    ray_stack[stack_ptr + rays_inserted + 1].contribution = ray_stack[stack_ptr].contribution * (1.0f - reflectance[intersection_index]) * transparency[intersection_index];
                }
                else {
                    ray_stack[stack_ptr + rays_inserted].contribution = (1.0f - reflectance[intersection_index]) * transparency[intersection_index];
                }
                ray_stack[stack_ptr + rays_inserted + 1].surface_index = ray_stack[stack_ptr].surface_index;
                rays_inserted++;
            }

            // Add the color contribution of the current intersection
            ray_stack[stack_ptr].contribution *= (1.0f - reflectance[intersection_index]) * (1.0f - transparency[intersection_index]);
            stack_ptr++;
        }
        
        // Now add up the colors from the ray stack with their respective contributions
        output_buffer[get_global_id(0) * 4] = 0x00;
        output_buffer[get_global_id(0) * 4 + 1] = 0x00;
        output_buffer[get_global_id(0) * 4 + 2] = 0x00;
        output_buffer[get_global_id(0) * 4 + 3] = 0xff; // Alpha channel is always fully opaque
        for (int si = maximum_surface_index; si >=0; si--)
        {
            for (int i = 0; i < stack_ptr; i++) {
                if (ray_stack[i].surface_index == 0) {
                    output_buffer[get_global_id(0) * 4] += (uchar) ((float) ray_stack[i].color[0] * ray_stack[i].contribution);
                    output_buffer[get_global_id(0) * 4 + 1] += (uchar) ((float) ray_stack[i].color[1] * ray_stack[i].contribution);
                    output_buffer[get_global_id(0) * 4 + 2] += (uchar) ((float) ray_stack[i].color[2] * ray_stack[i].contribution);
                }
                else
                {
                    ray_stack[si].color[0] = (uchar) ((float) ray_stack[si].color[0] * ((float) ray_stack[i].color[0] / 255) * ray_stack[i].contribution);
                    ray_stack[si].color[1] = (uchar) ((float) ray_stack[si].color[1] * ((float) ray_stack[i].color[1] / 255) * ray_stack[i].contribution);
                    ray_stack[si].color[2] = (uchar) ((float) ray_stack[si].color[2] * ((float) ray_stack[i].color[2] / 255) * ray_stack[i].contribution);
                }
            }
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
                         __constant uint *merge_indices,
                         __constant float *merge_radii,
                         unsigned int object_amnt,
                         __constant float *object_props,
                         uchar prop_size,
                         __constant uchar *color,
                         __constant float *reflectance,
                         __constant float *transparency,
                         __constant float *refractive_index,
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
        render_pixel(output_buffer, object_cframe, object_amnt, merge_indices, merge_radii, camera, cam_ray, object_props, prop_size, color, reflectance, transparency, refractive_index, directionlight_direction, directionlight_color);
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

    pub fn render_frame(&mut self, mut camera: Camera, mut render_objects: Vec<RenderObject>, mut merge_indices: Vec<u32>, mut merge_radii: Vec<f32>, directionlight_direction: Vec<f32>, directionlight_color: Vec<u8>) -> Result<Vec::<u8>, RendererError> {
        let c_width = u16::try_from(self.width).map_err(|_| RendererError::DimensionsTooBigError)?;
        let c_height = u16::try_from(self.height).map_err(|_| RendererError::DimensionsTooBigError)?;

        let mut cframe_vec = Vec::<f32>::new();
        let mut object_props_vec = Vec::<f32>::new();
        let mut color_vec = Vec::<u8>::new();
        let mut reflectance_vec = Vec::<f32>::new();
        let mut transparency_vec = Vec::<f32>::new();
        let mut refractive_index_vec = Vec::<f32>::new();
        let prop_size: u8 = render_objects[0].get_object_props_vec().len() as u8;
        for obj in render_objects.iter_mut() {
            cframe_vec.extend(obj.convert_to_cframe_buffer());
            object_props_vec.extend(obj.get_object_props_vec());
            color_vec.extend(obj.get_color_vec());
            reflectance_vec.push(obj.get_reflectance());
            transparency_vec.push(obj.get_transparency());
            refractive_index_vec.push(obj.get_refractive_index());
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

        let reflectance_buffer = Buffer::builder().queue(self.pro_que.as_mut().ok_or(RendererError::RendererNotInitializedError)?.queue().clone())
            .flags(MemFlags::new().read_write())
            .len(reflectance_vec.len())
            .copy_host_slice(&reflectance_vec)
            .build().map_err(|e| RendererError::CreateBufferError(e))?;

        let transparency_buffer = Buffer::builder().queue(self.pro_que.as_mut().ok_or(RendererError::RendererNotInitializedError)?.queue().clone())
            .flags(MemFlags::new().read_write())
            .len(transparency_vec.len())
            .copy_host_slice(&transparency_vec)
            .build().map_err(|e| RendererError::CreateBufferError(e))?;

        let refractive_index_buffer = Buffer::builder().queue(self.pro_que.as_mut().ok_or(RendererError::RendererNotInitializedError)?.queue().clone())
            .flags(MemFlags::new().read_write())
            .len(refractive_index_vec.len())
            .copy_host_slice(&refractive_index_vec)
            .build().map_err(|e| RendererError::CreateBufferError(e))?;

        let camera_buffer = Buffer::builder().queue(self.pro_que.as_mut().ok_or(RendererError::RendererNotInitializedError)?.queue().clone())
            .flags(MemFlags::new().read_write())
            .len(13)
            .copy_host_slice(&camera.to_vec())
            .build().map_err(|e| RendererError::CreateBufferError(e))?;

        let merge_indices_buffer = Buffer::builder().queue(self.pro_que.as_mut().ok_or(RendererError::RendererNotInitializedError)?.queue().clone())
            .flags(MemFlags::new().read_write())
            .len(merge_indices.len())
            .copy_host_slice(&merge_indices)
            .build().map_err(|e| RendererError::CreateBufferError(e))?;
        
        let merge_radii_buffer = Buffer::builder().queue(self.pro_que.as_mut().ok_or(RendererError::RendererNotInitializedError)?.queue().clone())
            .flags(MemFlags::new().read_write())
            .len(merge_radii.len())
            .copy_host_slice(&merge_radii)
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
            .arg(merge_indices_buffer)
            .arg(merge_radii_buffer)
            .arg((cframe_vec.len() / 12) as u32)
            .arg(object_prop_buffer)
            .arg(prop_size)
            .arg(color_buffer)
            .arg(reflectance_buffer)
            .arg(transparency_buffer)
            .arg(refractive_index_buffer)
            .arg(directionlight_direction_buffer)
            .arg(directionlight_color_buffer)
            .build().map_err(|e| RendererError::AddArgumentsError(e))?;

        unsafe { kernel.enq().map_err(|e| RendererError::ExecuteKernelError(e))?; }

        let mut vec = vec![0u8; self.output_buffer.as_ref().ok_or(RendererError::RendererNotInitializedError)?.len()];
        self.output_buffer.as_ref().ok_or(RendererError::RendererNotInitializedError)?.read(&mut vec).enq().map_err(|e| RendererError::ReadBufferError(e))?;

        return Ok(vec);
    }
}