
extern crate ocl;
use ocl::{ProQue, Buffer, MemFlags};
use crate::engine::util::error::RendererError;
use crate::engine::camera::Camera;

use crate::engine::util::octree::octree::Octree;
use crate::engine::primitives::primitive::Primitive;

const RENDER_SRC: &str = r#"
    #pragma OPENCL EXTENSION cl_amd_printf : enable

    #define MAXIMUM_BOUNCES 10
    #define MAXIMUM_TRANSPARENCY_BOUNCES 5 // Maximum amount of transparent objects traversed to calculate shadow color.
    #define CORRECTION_FACTOR 0.01 // Correction factor to prevent ray from colliding with the object itself due to floating point precision issues.
    #define AIR_REFRACTIVE_INDEX 1.0 // Refractive index of air, used for calculating the refracted ray.
    #define MAXIMUM_MERGES 6
    #define MAXIMUM_MARCHING_STEPS 100 // Maximum amount of steps to take when marching the ray.
    #define STEP_MARGIN 0.01f // How close the ray needs to get for it to count as an intersection
    #define STEP_MULTIPLIER 0.8f // When the ray closes in on an object, this multiplier will be used to have smaller steps (or bigger if larger than 1.0f)
    #define MAXIMUM_BLENDS 30 // Maximum amount of objects to be used to blend properties

    typedef struct {
        float cframe[12];    // 48 bytes
        float contribution;  // 4 bytes
        int surface_index;   // 4 bytes
        uchar color[4];      // 4 bytes
    } RayStackEntry;

    typedef struct {
        uint subnode_indices[8];
        uint indices[8];
    } OctreeNode;

    typedef enum {
        SPHERE = 0,
    } Kind; 

    typedef struct {
        float x;
        float y;
        float z;
        float r00;
        float r01;
        float r02;
        float r10;
        float r11;
        float r12;
        float r20;
        float r21;
        float r22;
    } CFrame;

    typedef struct {
        float radius;
        float _pad[0];
    } SphereData;

    typedef union {
        SphereData sphere_data;
        float raw[1];
    } PrimitivePayload;

    typedef struct {
        Kind kind;
        float transparency;
        float refractive_index;
        float reflectance;
        uchar color[3];
        CFrame cframe;
        float render_radius;
        unsigned int merge_indices[6];
        float merge_radii[6];
        float _pad_common;
        PrimitivePayload payload;
    } Primitive;

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

    float calculate_squared_distance_point_to_line(CFrame point,
                                                   float *ray)
    {
        float origin_point_diff[3] = { ray[0] - point.x,
                                       ray[1] - point.y,
                                       ray[2] - point.z };
        float dot = (origin_point_diff[0] * -ray[5]) + (origin_point_diff[1] * -ray[8]) + (origin_point_diff[2] * -ray[11]);
        float projected_length[3] = { -ray[5] * dot,
                                      -ray[8] * dot,
                                      -ray[11] * dot };
        float diff[3] = { origin_point_diff[0] - projected_length[0],
                          origin_point_diff[1] - projected_length[1],
                          origin_point_diff[2] - projected_length[2] };
        return (diff[0] * diff[0]) + (diff[1] * diff[1]) + (diff[2] * diff[2]);
    }

    float smooth_min(float a,
                     float b,
                     float k)
    {
        float h = fmax(k - fabs(a - b), 0.0f);
        return fmin(a, b) - (h * h * 0.25f / k);
    }

    float sphere_sdf(CFrame sphere_cframe,
                     float sphere_radius,
                     float *position)
    {
        float diff[3] = { sphere_cframe.x - position[0],
                          sphere_cframe.y - position[1],
                          sphere_cframe.z - position[2] };
        return sqrt((diff[0] * diff[0]) + (diff[1] * diff[1]) + (diff[2] * diff[2])) - sphere_radius;
    }

    float merge_sphere_sdf(__constant Primitive* primitives,
                           int index,
                           float *position)
    {
        Primitive p = primitives[index];
        float sdf = sphere_sdf(p.cframe, p.payload.sphere_data.radius, position);
        for (int m = 0; m < MAXIMUM_MERGES; m++)
        {
            uint m_index = p.merge_indices[m];
            if (p.merge_radii[m] < 0.01f)
                break;
            Primitive p2 = primitives[m_index];
            float other_sdf = sphere_sdf(p2.cframe, p2.payload.sphere_data.radius, position);
            sdf = smooth_min(other_sdf, sdf, p.merge_radii[m]);
        }
        return sdf;
    }

    void calculate_sdf_normal(
        float *position,
        __constant Primitive* primitives,
        int index,
        float *out_normal
    ) {
        float epsilon = 0.001f;
        float dx[3] = {epsilon, 0.0f, 0.0f};
        float dy[3] = {0.0f, epsilon, 0.0f};
        float dz[3] = {0.0f, 0.0f, epsilon};
        float p[3];

        // X
        p[0] = position[0] + epsilon; p[1] = position[1]; p[2] = position[2];
        float sdf_x1 = merge_sphere_sdf(primitives, index, p);
        p[0] = position[0] - epsilon;
        float sdf_x0 = merge_sphere_sdf(primitives, index, p);

        // Y
        p[0] = position[0]; p[1] = position[1] + epsilon; p[2] = position[2];
        float sdf_y1 = merge_sphere_sdf(primitives, index, p);
        p[1] = position[1] - epsilon;
        float sdf_y0 = merge_sphere_sdf(primitives, index, p);

        // Z
        p[1] = position[1]; p[2] = position[2] + epsilon;
        float sdf_z1 = merge_sphere_sdf(primitives, index, p);
        p[2] = position[2] - epsilon;
        float sdf_z0 = merge_sphere_sdf(primitives, index, p);

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

    float calculate_squared_euclidean_distance(__constant float *a,
                                               float *b)
    {
        float dx = a[0] - b[0];
        float dy = a[1] - b[1];
        float dz = a[2] - b[2];
        return (dx * dx) + (dy * dy) + (dz * dz);
    }

    void gather_blend_objects(__constant Primitive *primitives,
                              int index,
                              int *out_blend_indices)
    {
        Primitive p = primitives[index];
        // Initialize output
        for (int i = 0; i < MAXIMUM_BLENDS; i++) out_blend_indices[i] = -1;
        int blend_count = 1;
        out_blend_indices[0] = index;

        int scan_ptr = 0;
        while (scan_ptr < blend_count && blend_count < MAXIMUM_BLENDS) {
            int current = out_blend_indices[scan_ptr];
            for (int m = 0; m < MAXIMUM_MERGES; m++) {
                uint merge_idx = p.merge_indices[m];
                if (p.merge_radii[m] < 0.01f)
                    break;

                // Check for duplicates in out_blend_indices
                int is_duplicate = 0;
                for (int i = 0; i < blend_count; i++) {
                    if (out_blend_indices[i] == merge_idx) {
                        is_duplicate = 1;
                        break;
                    }
                }
                if (!is_duplicate && blend_count < MAXIMUM_BLENDS) {
                    out_blend_indices[blend_count] = merge_idx;
                    blend_count++;
                }
            }
            scan_ptr++;
        }
    }

    void calculate_params_at_intersection(__constant Primitive* primitives,
                                          int index,
                                          float *position,
                                          uchar *out_color,
                                          float *out_transparency,
                                          float *out_reflectance,
                                          float *out_refractive_index)
    {
        int blend_objects[MAXIMUM_BLENDS];
        gather_blend_objects(primitives, index, blend_objects);
        float weights[MAXIMUM_BLENDS];
        float sum = 0.0f;
        int total_blends = 0;
        for (int i = 0; i < MAXIMUM_BLENDS; i++)
        {
            if (blend_objects[i] < 0)
                break;
            total_blends++;
            Primitive p = primitives[blend_objects[i]];
            float sdf = sphere_sdf(p.cframe, p.payload.sphere_data.radius, position);
            weights[i] = fmax(exp(-sdf / p.payload.sphere_data.radius), 0.0f);
            sum += weights[i];
        }

        // Normalize weights
        for (int i = 0; i < total_blends + 1; i++) {
            weights[i] /= sum;
        }

        // Initialize color
        out_color[0] = 0;
        out_color[1] = 0;
        out_color[2] = 0;
        *out_transparency = 0.0f;
        *out_reflectance = 0.0f;
        *out_refractive_index = 0.0f;

        // Blend
        for (int m = 0; m < total_blends; m++) {
            int m_index = blend_objects[m];
            Primitive pm = primitives[m_index];
            out_color[0] += (uchar)(pm.color[0] * weights[m]);
            out_color[1] += (uchar)(pm.color[1] * weights[m]);
            out_color[2] += (uchar)(pm.color[2] * weights[m]);
            *out_transparency += pm.transparency * weights[m];
            *out_reflectance += pm.reflectance * weights[m];
            *out_refractive_index += pm.refractive_index * weights[m];
        }
    }

    float intersect_object(__constant Primitive* primitives,
                           int index,
                           float *ray_cframe,
                           float *step_size)
    {
        float local_t = 0.0f;
        float step_position[3] = { ray_cframe[0],
                                   ray_cframe[1],
                                   ray_cframe[2] };
        *step_size = merge_sphere_sdf(primitives, index, step_position) * STEP_MULTIPLIER;
        if (*step_size < 0)
        {
            // inside sphere, stop
            return 99999.0f;
        }
        int steps = 0;
        for (int i = 0; i < MAXIMUM_MARCHING_STEPS; i++)
        {
            steps++;
            if (fabs(*step_size) <= STEP_MARGIN)
            {
                break;
            }
            local_t += *step_size;
            step_position[0] = ray_cframe[0] - (local_t * ray_cframe[5]);
            step_position[1] = ray_cframe[1] - (local_t * ray_cframe[8]);
            step_position[2] = ray_cframe[2] - (local_t * ray_cframe[11]);
            *step_size = merge_sphere_sdf(primitives, index, step_position) * STEP_MULTIPLIER;
        }
        return local_t;
    }

    int intersect_objects(__constant Primitive* primitives,
                          unsigned int primitive_amount,
                          float *ray_cframe,
                          float *out_t,
                          float *out_position,
                          float *out_normal,
                          uchar *out_color,
                          float *out_transparency,
                          float *out_reflectance,
                          float *out_refractive_index)
    {
        float t = 9999999;
        int index_found = -1;
        for (int i = 0; i < primitive_amount; i++)
        {
            Primitive p = primitives[i];
            float local_t = 0.0f;
            float squared_distance = calculate_squared_distance_point_to_line(p.cframe, ray_cframe);
            // Use squared distance to avoid unnecessary square root calculation
            if (squared_distance > p.render_radius * p.render_radius)
            {
                // Check if the merged spheres are not too close either
                int close_merge_found = 0;
                for (int m = 0; m < MAXIMUM_MERGES; m++)
                {
                    int m_index = p.merge_indices[m];
                    if (p.merge_radii[m] < 0.01f)
                        break;
                    float merge_squared_distance = calculate_squared_distance_point_to_line(p.cframe, ray_cframe);
                    if (merge_squared_distance < p.render_radius * p.render_radius)
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
            local_t = intersect_object(primitives,
                                       i,
                                       ray_cframe,
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
                                 primitives,
                                 index_found,
                                 out_normal);
            calculate_params_at_intersection(primitives,
                                             index_found,
                                             out_position,
                                             out_color,
                                             out_transparency,
                                             out_reflectance,
                                             out_refractive_index);
            *out_t = t;
        }
        return index_found;
    }

    void get_intersection_from_ray(uchar *out_color,
                                   int *intersection_index,
                                   float *out_normal,
                                   float *out_position,
                                   __constant Primitive* primitives,
                                   unsigned int primitive_amount,
                                   float *ray_cframe,
                                   __constant float *directionlight_direction,
                                   __constant uchar *directionlight_color,
                                   float *out_transparency,
                                   float *out_reflectance,
                                   float *out_refractive_index)
    {
        float t;
        float edge_pos[3];
        *intersection_index = intersect_objects(primitives,
                                                primitive_amount,
                                                ray_cframe,
                                                &t,
                                                edge_pos,
                                                out_normal,
                                                out_color,
                                                out_transparency,
                                                out_reflectance,
                                                out_refractive_index);

        if (*intersection_index >= 0)
        {
            // The calculated edge_pos can be slightly inside inside the object, causing the ray to calculate the shadow to collide with the object itself.
            // This is due to floating point precision.
            // To combat this, take the starting point of the ray at a distance of "CORRECTION_FACTOR" more outwards of the object.
            out_position[0] = edge_pos[0] + out_normal[0] * CORRECTION_FACTOR;
            out_position[1] = edge_pos[1] + out_normal[1] * CORRECTION_FACTOR;
            out_position[2] = edge_pos[2] + out_normal[2] * CORRECTION_FACTOR;
            float diffuseFactor;
            if (*out_transparency >= 0.01f)
            {
                diffuseFactor = 1.0f; // Transparent objects don't have a shade
            }
            else
            {
                diffuseFactor = fmax(out_normal[0] * (-directionlight_direction[0]) + out_normal[1] * (-directionlight_direction[1]) + out_normal[2] * (-directionlight_direction[2]), 0.0f);
            }
            float directional_diffuse_light_color[3] = { directionlight_color[0] * diffuseFactor / 0xff, directionlight_color[1] * diffuseFactor / 0xff, directionlight_color[2] * diffuseFactor / 0xff };
            out_color[0] = (uchar) (((float) out_color[0]) * directional_diffuse_light_color[0]);
            out_color[1] = (uchar) (((float) out_color[1]) * directional_diffuse_light_color[1]);
            out_color[2] = (uchar) (((float) out_color[2]) * directional_diffuse_light_color[2]);
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
                uchar shadow_color[3];
                float shadow_transparency;
                float shadow_reflectance;
                float shadow_refractive_index;
                int dl_int_index = intersect_objects(primitives,
                                                     primitive_amount,
                                                     edge_to_dir_light,
                                                     &dl_t,
                                                     light_edge_pos,
                                                     light_normal,
                                                     shadow_color,
                                                     &shadow_transparency,
                                                     &shadow_reflectance,
                                                     &shadow_refractive_index);
                if (dl_int_index < 0 || (dl_int_index == *intersection_index))
                {
                    break;
                }
                else if (shadow_transparency < 0.01f)
                {
                    out_color[0] = 0x00;
                    out_color[1] = 0x00;
                    out_color[2] = 0x00;
                    out_color[3] = 0xff;
                    break;
                }
                else
                {
                    Primitive p = primitives[dl_int_index];
                    uchar filtered_light_r = directionlight_color[0] * (shadow_transparency * 1.0f + (1.0f - shadow_transparency) * (shadow_color[0] / 255.0f));
                    uchar filtered_light_g = directionlight_color[1] * (shadow_transparency * 1.0f + (1.0f - shadow_transparency) * (shadow_color[1] / 255.0f));
                    uchar filtered_light_b = directionlight_color[2] * (shadow_transparency * 1.0f + (1.0f - shadow_transparency) * (shadow_color[2] / 255.0f));
                    out_color[0] = (uchar) (((float) out_color[0]) * ((float) filtered_light_r / 255.0f));
                    out_color[1] = (uchar) (((float) out_color[1]) * ((float) filtered_light_g / 255.0f));
                    out_color[2] = (uchar) (((float) out_color[2]) * ((float) filtered_light_b / 255.0f));
                    edge_to_dir_light[0] = edge_to_dir_light[0] - (directionlight_direction[0] * (dl_t + (CORRECTION_FACTOR * 2)));
                    edge_to_dir_light[1] = edge_to_dir_light[1] - (directionlight_direction[1] * (dl_t + (CORRECTION_FACTOR * 2)));
                    edge_to_dir_light[2] = edge_to_dir_light[2] - (directionlight_direction[2] * (dl_t + (CORRECTION_FACTOR * 2)));
                    float step_size = 99999.0f;
                    float light_edge_pos[3];
                    float light_normal[3];
                    float other_side_ray[12] = { edge_to_dir_light[0] - (directionlight_direction[0] * p.render_radius * 3),
                                                edge_to_dir_light[1] - (directionlight_direction[1] * p.render_radius * 3),
                                                edge_to_dir_light[2] - (directionlight_direction[2] * p.render_radius * 3),
                                                0.0f, 0.0f, -directionlight_direction[0],
                                                0.0f, 0.0f, -directionlight_direction[1],
                                                0.0f, 0.0f, -directionlight_direction[2] };
                    float dl_t2 = intersect_object(primitives,
                                                   dl_int_index,
                                                   other_side_ray,
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
                                    __constant Primitive *primitives,
                                    int intersection_index)
    {
        float radius = primitives[intersection_index].render_radius;
        float internal_ray[12];
        calculate_refracted_ray(normal, incoming_ray, position, internal_ray, AIR_REFRACTIVE_INDEX, n1);
        internal_ray[0] = internal_ray[0] - internal_ray[5] * radius * 4;
        internal_ray[1] = internal_ray[1] - internal_ray[8] * radius * 4;
        internal_ray[2] = internal_ray[2] - internal_ray[11] * radius * 4;
        internal_ray[5] = -internal_ray[5];
        internal_ray[8] = -internal_ray[8];
        internal_ray[11] = -internal_ray[11];
        float step_size;
        float local_t = intersect_object(primitives,
                                         intersection_index,
                                         internal_ray,
                                         &step_size);
        float internal_edge_pos[3] = { internal_ray[0] - (internal_ray[5] * local_t), internal_ray[1] - (internal_ray[8] * local_t), internal_ray[2] - (internal_ray[11] * local_t) };
        float internal_normal[3] = { 0.0f, 0.0f, 0.0f };
        calculate_sdf_normal(internal_edge_pos,
                             primitives,
                             intersection_index,
                             internal_normal);
        internal_edge_pos[0] += (internal_normal[0] * CORRECTION_FACTOR * 5);
        internal_edge_pos[1] += (internal_normal[1] * CORRECTION_FACTOR * 5);
        internal_edge_pos[2] += (internal_normal[2] * CORRECTION_FACTOR * 5);
        if (local_t > 0.0f)
        {
            internal_ray[5] = -internal_ray[5];
            internal_ray[8] = -internal_ray[8];
            internal_ray[11] = -internal_ray[11];
            float inversed_internal_normal[3] = { -internal_normal[0], -internal_normal[1], -internal_normal[2] };
            calculate_refracted_ray(inversed_internal_normal, internal_ray, internal_edge_pos, outgoing_ray, n1, AIR_REFRACTIVE_INDEX);
        }
    }

    void render_pixel(__global uchar *output_buffer,
                      __constant Primitive* primitives,
                      unsigned int primitive_amount,
                      __constant float *camera_cframe,
                      float *ray_rotation_matrix,
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
            float calc_transparency;
            float calc_reflectance;
            float calc_refractive_index;
            get_intersection_from_ray(ray_stack[stack_ptr].color,
                                      &intersection_index,
                                      result_normal,
                                      result_position,
                                      primitives,
                                      primitive_amount,
                                      ray_stack[stack_ptr].cframe,
                                      directionlight_direction,
                                      directionlight_color,
                                      &calc_transparency,
                                      &calc_reflectance,
                                      &calc_refractive_index);
            // if (calc_transparency > 0.1f && calc_transparency < 0.5f)
            if (intersection_index < 0) {
                // No intersection found, just continue to the next ray in the stack
                ray_stack[stack_ptr].surface_index = -1;
                stack_ptr++;
                continue;
            }

            if (calc_reflectance > 0.01f && ray_stack[stack_ptr].surface_index == 0) {
                calculate_reflected_ray(ray_stack[stack_ptr].cframe,
                                        result_normal,
                                        result_position,
                                        ray_stack[stack_ptr + rays_inserted + 1].cframe);
                ray_stack[stack_ptr + rays_inserted + 1].contribution = ray_stack[stack_ptr].contribution * calc_reflectance;
                ray_stack[stack_ptr + rays_inserted + 1].surface_index = ray_stack[stack_ptr].surface_index;
                rays_inserted++;
            }

            if (calc_transparency > 0.01f) {
                calculate_transparency_ray(result_normal,
                                           ray_stack[stack_ptr].cframe,
                                           result_position,
                                           ray_stack[stack_ptr + rays_inserted + 1].cframe,
                                           calc_refractive_index,
                                           primitives,
                                           intersection_index);
                if (ray_stack[stack_ptr].surface_index == 0) {
                    ray_stack[stack_ptr + rays_inserted + 1].contribution = ray_stack[stack_ptr].contribution * (1.0f - calc_reflectance) * calc_transparency;
                }
                else {
                    ray_stack[stack_ptr + rays_inserted].contribution = (1.0f - calc_reflectance) * calc_transparency;
                }
                ray_stack[stack_ptr + rays_inserted + 1].surface_index = ray_stack[stack_ptr].surface_index;
                rays_inserted++;
            }

            // Add the color contribution of the current intersection
            ray_stack[stack_ptr].contribution *= (1.0f - calc_reflectance) * (1.0f - calc_transparency);
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
                         __constant float *directionlight_direction,
                         __constant uchar *directionlight_color,
                         __constant OctreeNode* nodes,
                         __constant Primitive* primitives,
                         unsigned int primitive_amount) {
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
        render_pixel(output_buffer, primitives, primitive_amount, camera, cam_ray, directionlight_direction, directionlight_color);
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
            .src(RENDER_SRC)
            .dims(self.width * self.height)
            .build().map_err(|e| RendererError::KernelBuildError(e))?);
        
        self.buffer = Some(self.pro_que.as_mut().ok_or(RendererError::RendererNotInitializedError)?.create_buffer::<u8>().map_err(|e| RendererError::CreateBufferError(e))?);
        self.output_buffer = Some(Buffer::builder().queue(self.pro_que.as_mut().ok_or(RendererError::RendererNotInitializedError)?.queue().clone())
            .flags(MemFlags::new().read_write())
            .len(self.width * self.height * 4)
            .build().map_err(|e| RendererError::CreateBufferError(e))?);

        Ok(())
    }

    pub fn render_frame(&mut self, mut camera: Camera, directionlight_direction: Vec<f32>, directionlight_color: Vec<u8>, primitives: &Vec<Primitive>) -> Result<Vec::<u8>, RendererError> {
        let c_width = u16::try_from(self.width).map_err(|_| RendererError::DimensionsTooBigError)?;
        let c_height = u16::try_from(self.height).map_err(|_| RendererError::DimensionsTooBigError)?;

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

        let mut octree = Octree::new(10f32, 10u32, 10u32, 10u32, (0f32, 0f32, 0f32));
        let octree_buffer = Buffer::builder().queue(self.pro_que.as_mut().ok_or(RendererError::RendererNotInitializedError)?.queue().clone())
            .flags(MemFlags::new().read_write())
            .len(octree.get_nodes().full_len())
            .copy_host_slice(octree.get_nodes().get_items())
            .build().map_err(|e| RendererError::CreateBufferError(e))?;

        let primitive_buffer = Buffer::builder().queue(self.pro_que.as_mut().ok_or(RendererError::RendererNotInitializedError)?.queue().clone())
            .flags(MemFlags::new().read_write())
            .len(primitives.len())
            .copy_host_slice(primitives)
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
            .arg(directionlight_direction_buffer)
            .arg(directionlight_color_buffer)
            .arg(octree_buffer)
            .arg(primitive_buffer)
            .arg(primitives.len() as u32)
            .build().map_err(|e| RendererError::AddArgumentsError(e))?;

        unsafe { kernel.enq().map_err(|e| RendererError::ExecuteKernelError(e))?; }

        let mut vec = vec![0u8; self.output_buffer.as_ref().ok_or(RendererError::RendererNotInitializedError)?.len()];
        self.output_buffer.as_ref().ok_or(RendererError::RendererNotInitializedError)?.read(&mut vec).enq().map_err(|e| RendererError::ReadBufferError(e))?;

        return Ok(vec);
    }
}