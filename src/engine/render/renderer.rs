
extern crate ocl;
use ocl::{ProQue, Buffer, MemFlags};
use crate::engine::util::error::RendererError;
use crate::engine::camera::Camera;

use crate::engine::util::octree::octree_node::OctreeNode;
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
    #define MAX_OBJECTS_PER_NODE 8 // Maximum amount of objects allowed per octree node
    #define MAXIMUM_OCTREE_NODE_CHECKS 16 // No matter if there's an object deeper in the octree or further away, it will only do 16 node traversals
    __constant float OCTREE_SUBNODE_OFFSETS[8][3] = {
        {0.0, 0.0, 0.0}, // bottom-front-left
        {1.0, 0.0, 0.0}, // bottom-front-right
        {0.0, 1.0, 0.0}, // top-front-left
        {1.0, 1.0, 0.0}, // top-front-right
        {0.0, 0.0, 1.0}, // bottom-back-left
        {1.0, 0.0, 1.0}, // bottom-back-right
        {0.0, 1.0, 1.0}, // top-back-left
        {1.0, 1.0, 1.0}, // top-back-right
    };

    typedef struct {
        uint node_index;
        float tmin, tmax;
        float node_center[3];
        float node_size;
    } OctreeStackEntry;

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
        // Initialize output
        for (int i = 0; i < MAXIMUM_BLENDS; i++) out_blend_indices[i] = -1;
        int blend_count = 1;
        out_blend_indices[0] = index;

        int scan_ptr = 0;
        while (scan_ptr < blend_count && blend_count < MAXIMUM_BLENDS) {
            int current = out_blend_indices[scan_ptr];
            Primitive p = primitives[current];
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
        // printf("intersection object at index %d with render radius %f and position (%f, %f, %f)\n", index, primitives[index].payload.sphere_data.radius, primitives[index].cframe.x, primitives[index].cframe.y, primitives[index].cframe.z);
        // printf("ray cframe origin is (%f, %f, %f) with direction (%f, %f, %f)\n", ray_cframe[0], ray_cframe[1], ray_cframe[2], ray_cframe[5], ray_cframe[8], ray_cframe[11]);
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
            // printf("step size is %f\n", *step_size);
        }
        return local_t;
    }

    uint select_octree_root_node(__constant uint *octree_root_node_indices,
                                 __constant float *octree_root_node_position,
                                 __constant uint *octree_dimensions,
                                 float octree_root_node_size,
                                 float *position,
                                 float *out_position,
                                 int *out_chunk_indices)
    {
        // printf("in select octree root node\n");
        uint index_x = (uint) ((position[0] - octree_root_node_position[0]) / octree_root_node_size);
        uint index_y = (uint) ((position[1] - octree_root_node_position[1]) / octree_root_node_size);
        uint index_z = (uint) ((position[2] - octree_root_node_position[2]) / octree_root_node_size);
        // printf("index_x: %d, index_y: %d, index_z: %d\n", index_x, index_y, index_z);
        // printf("octree dimensions: %d, %d, %d\n", octree_dimensions[0], octree_dimensions[1], octree_dimensions[2]);
        // printf("octree root node position and size: (%f, %f, %f), %f\n",
        //        octree_root_node_position[0],
        //        octree_root_node_position[1],
        //        octree_root_node_position[2],
        //        octree_root_node_size);
        // printf("start position is (%f, %f, %f)\n",
        //        position[0],
        //        position[1],
        //        position[2]);
        float test_difference = position[0] - octree_root_node_position[0];
        // printf("try again %f %f %f %f\n", position[0], octree_root_node_position[0], test_difference, ((position[0] - octree_root_node_position[0]) / octree_root_node_size));

        if (index_x < octree_dimensions[0] && index_y < octree_dimensions[1] && index_z < octree_dimensions[2])
        {
            // printf("Selected root node at (%d, %d, %d)\n", index_x, index_y, index_z);
            out_position[0] = octree_root_node_position[0] + index_x * octree_root_node_size;
            out_position[1] = octree_root_node_position[1] + index_y * octree_root_node_size;
            out_position[2] = octree_root_node_position[2] + index_z * octree_root_node_size;
            out_chunk_indices[0] = index_x;
            out_chunk_indices[1] = index_y;
            out_chunk_indices[2] = index_z;
            return index_x + index_y * octree_dimensions[0] + index_z * octree_dimensions[0] * octree_dimensions[1];
        }
        // printf("try again %f %f %f %f\n", position[0], octree_root_node_position[0], test_difference, ((position[0] - octree_root_node_position[0]) / octree_root_node_size));
        // printf("Selected root node is out of bounds\n");
        return 0xFFFFFFFF;
    }

    inline bool intersect_objects_in_node(OctreeNode node,
                                          __constant Primitive* primitives,
                                          float *ray_cframe,
                                          float *t,
                                          int *index_found)
    {
        bool found = false;
        for (int i = 0; i < MAX_OBJECTS_PER_NODE; i++)
        {
            if (node.indices[i] == 0xFFFFFFFF)
                break;
            // printf("object in node y'all\n");
            Primitive p = primitives[node.indices[i]];
            float step_size = 99999.0f;
            float local_t = intersect_object(primitives, node.indices[i], ray_cframe, &step_size);
            // printf("intersected object at index %d, local_t = %f, step_size = %f\n", node.indices[i], local_t, step_size);
            if (local_t < *t && step_size < STEP_MARGIN)
            {
                *index_found = node.indices[i];
                *t = local_t;
                found = true;
            }
        }
        return found;
    }

    bool ray_intersects_aabb(float *ray_cframe,
                             float *aabb_min,
                             float *aabb_max,
                             float *out_tmin,
                             float *out_tmax)
    {
        float tmin = -INFINITY, tmax = INFINITY;
        for (int i = 0; i < 3; i++) {
            if (fabs(ray_cframe[5 + i*3]) < 1e-8f) {
                if (-ray_cframe[i] < aabb_min[i] || -ray_cframe[i] > aabb_max[i])
                    return false; // ray parallel and outside slab
                // parallel but inside → leave t0=-inf, t1=+inf
                continue;
            }
            float invD = 1.0f / -ray_cframe[5 + (i * 3)];
            float t0 = (aabb_min[i] - ray_cframe[i]) * invD;
            float t1 = (aabb_max[i] - ray_cframe[i]) * invD;
            if (invD < 0.0f) {
                float tmp = t0; t0 = t1; t1 = tmp;
            }
            tmin = fmax(tmin, t0);
            tmax = fmin(tmax, t1);
            if (tmax + 1e-6f < tmin) return false;
        }
        *out_tmin = tmin;
        *out_tmax = tmax;
        return true;
    }

    int walk_octree(__constant OctreeNode* nodes,
                    float octree_root_node_size,
                    uint root_node_index,
                    float* root_node_position,
                    __constant Primitive* primitives,
                    float *ray_cframe,
                    float *out_t,
                    bool debug_mode)
    {
        // Ray: origin o and (forward) direction d
        const float o0 = ray_cframe[0], o1 = ray_cframe[1], o2 = ray_cframe[2];
        const float d0 = -ray_cframe[5], d1 = -ray_cframe[8], d2 = -ray_cframe[11];

        // inv_dir (guard zero -> +/-INF)
        float inv_dir0 = 1.0f / d0; inv_dir0 = isfinite(inv_dir0) ? inv_dir0 : copysign(INFINITY, d0);
        float inv_dir1 = 1.0f / d1; inv_dir1 = isfinite(inv_dir1) ? inv_dir1 : copysign(INFINITY, d1);
        float inv_dir2 = 1.0f / d2; inv_dir2 = isfinite(inv_dir2) ? inv_dir2 : copysign(INFINITY, d2);

        // root AABB test
        float t_root_min, t_root_max;
        float root_aabb_max[3] = {
            root_node_position[0] + octree_root_node_size,
            root_node_position[1] + octree_root_node_size,
            root_node_position[2] + octree_root_node_size
        };
        bool hits = ray_intersects_aabb(ray_cframe, root_node_position, root_aabb_max, &t_root_min, &t_root_max);
        if (!hits) return -1;

        // stack
        OctreeStackEntry stack[MAXIMUM_OCTREE_NODE_CHECKS];
        int sp = 0;

        // push root
        stack[sp].node_index = root_node_index;
        stack[sp].tmin = t_root_min;
        stack[sp].tmax = t_root_max;
        stack[sp].node_size = octree_root_node_size;
        stack[sp].node_center[0] = root_node_position[0] + 0.5f * octree_root_node_size;
        stack[sp].node_center[1] = root_node_position[1] + 0.5f * octree_root_node_size;
        stack[sp].node_center[2] = root_node_position[2] + 0.5f * octree_root_node_size;
        sp++;

        float bestT = INFINITY;
        int bestPrim = -1;
        int iter = 0;

        while (sp > 0) {
            iter++;
            // pop nearest candidate (your invariant: top is nearest)
            OctreeStackEntry entry = stack[--sp];

            // quick prune
            if (entry.tmin > bestT) continue;          // no better hit in this node
            if (entry.tmax < 0.0f) continue;           // node is behind ray start

            OctreeNode node = nodes[entry.node_index];

            int node_index_found = -1;
            OctreeNode current_node = nodes[entry.node_index];
            bool object_in_node = intersect_objects_in_node(current_node,
                                                            primitives,
                                                            ray_cframe,
                                                            &bestT,
                                                            &node_index_found);
            if (object_in_node)
            {
                // printf("let's gooooooooo");
                bestPrim = node_index_found;
                break; // This break may break certain cases...
            }

            // compute entry point P (use entry.tmin, clamp to 0)
            float tentry = fmax(entry.tmin, 0.0f);
            float Px = o0 + d0 * tentry;
            float Py = o1 + d1 * tentry;
            float Pz = o2 + d2 * tentry;

            // starting child bits (0 or 1)
            int bx = (Px >= entry.node_center[0]) ? 1 : 0;
            int by = (Py >= entry.node_center[1]) ? 1 : 0;
            int bz = (Pz >= entry.node_center[2]) ? 1 : 0;
            int childBits = bx | (by << 1) | (bz << 2);

            // plane times to center planes (one division emulated by inv_dir multiply)
            float tx = (entry.node_center[0] - o0) * inv_dir0;
            float ty = (entry.node_center[1] - o1) * inv_dir1;
            float tz = (entry.node_center[2] - o2) * inv_dir2;

            // clamp plane times so those > entry.tmax are ignored (set to +INF if you use min-based stepping,
            // or -INF if you use max-based). Here we step forward (nearest-first), so invalid plane -> +INF.
            tx = select(INFINITY, tx, islessequal(tx, entry.tmax));
            ty = select(INFINITY, ty, islessequal(ty, entry.tmax));
            tz = select(INFINITY, tz, islessequal(tz, entry.tmax));

            // We'll step at most 4 children in ray order, collect valid children into local nearest-first array
            OctreeStackEntry local_stack[4];
            int lp = 0;
            int bits = childBits;
            float prev_t = entry.tmin;

            for (int step = 0; step < 4; ++step) {
                // current child index (0..7)
                int cidx = bits;

                // compute child center and size:
                float child_size = entry.node_size * 0.5f;         // child side length
                float child_half = child_size * 0.5f;             // offset from parent center to child center
                float child_cx = entry.node_center[0] + ((bits & 1) ?  child_half : -child_half);
                float child_cy = entry.node_center[1] + ((bits & 2) ?  child_half : -child_half);
                float child_cz = entry.node_center[2] + ((bits & 4) ?  child_half : -child_half);

                // check child exists BEFORE doing full slab math
                uint childNodeIdx = node.subnode_indices[cidx];
                if (childNodeIdx != 0xFFFFFFFFu) {
                    // compute child slab (using inv_dir)
                    float xmin = child_cx - 0.5f * child_size;
                    float xmax = child_cx + 0.5f * child_size;
                    float ymin = child_cy - 0.5f * child_size;
                    float ymax = child_cy + 0.5f * child_size;
                    float zmin = child_cz - 0.5f * child_size;
                    float zmax = child_cz + 0.5f * child_size;

                    float tx1 = (xmin - o0) * inv_dir0; float tx2 = (xmax - o0) * inv_dir0;
                    float ch_tmin = fmin(tx1, tx2), ch_tmax = fmax(tx1, tx2);

                    float ty1 = (ymin - o1) * inv_dir1; float ty2 = (ymax - o1) * inv_dir1;
                    ch_tmin = fmax(ch_tmin, fmin(ty1, ty2)); ch_tmax = fmin(ch_tmax, fmax(ty1, ty2));

                    float tz1 = (zmin - o2) * inv_dir2; float tz2 = (zmax - o2) * inv_dir2;
                    ch_tmin = fmax(ch_tmin, fmin(tz1, tz2)); ch_tmax = fmin(ch_tmax, fmax(tz1, tz2));

                    // clamp to parent interval
                    ch_tmin = fmax(ch_tmin, entry.tmin);
                    ch_tmax = fmin(ch_tmax, entry.tmax);

                    // If valid interval, record it (nearest-first)
                    if (ch_tmax >= ch_tmin) {
                        local_stack[lp].node_index = childNodeIdx;
                        local_stack[lp].tmin = ch_tmin;
                        local_stack[lp].tmax = ch_tmax;
                        local_stack[lp].node_center[0] = child_cx;
                        local_stack[lp].node_center[1] = child_cy;
                        local_stack[lp].node_center[2] = child_cz;
                        local_stack[lp].node_size = child_size;
                        // Optionally compute child's plane times later when popped
                        lp++;
                    }
                }

                // find the next plane crossing -> nearest plane time among tx,ty,tz
                float tnext = fmin(tx, fmin(ty, tz));

                // if the next crossing is outside node interval, break stepping
                if (tnext > entry.tmax) break;
                if (tnext <= prev_t) {
                    // avoid infinite loop (shouldn't normally happen)
                    // mark the chosen plane as invalid and continue
                    if (tx <= ty && tx <= tz) { tx = INFINITY; continue; }
                    if (ty <= tx && ty <= tz) { ty = INFINITY; continue; }
                    tz = INFINITY; continue;
                }

                // flip the bit for the axis we crossed, and invalidate that plane so it won't be chosen again
                bool takeX = (tnext == tx);
                bool takeY = (tnext == ty);
                bool takeZ = !takeX && !takeY;

                bits ^= (takeX ? 1 : 0) | (takeY ? 2 : 0) | (takeZ ? 4 : 0);

                tx = takeX ? INFINITY : tx;
                ty = takeY ? INFINITY : ty;
                tz = takeZ ? INFINITY : tz;

                prev_t = tnext;
                // continue stepping to next child (up to 4)
            } // end for up to 4 children

            // push children into global stack in reverse order (so nearest ends up on top)
            for (int i = lp - 1; i >= 0; --i) {
                // optional early prune by bestT
                if (local_stack[i].tmin > bestT) continue;
                // push
                stack[sp++] = local_stack[i];
                if (sp >= MAXIMUM_OCTREE_NODE_CHECKS) break; // safety
            }

            if (iter > MAXIMUM_OCTREE_NODE_CHECKS) break;
        } // end while stack

        // if (iter > 8)
        //     printf("iter is pretty big: %d\n", iter);

        *out_t = bestT;
        return bestPrim;
    }

    int walk_chunks(__constant OctreeNode* nodes,
                    __constant uint *octree_root_node_indices,
                    __constant float *octree_root_node_position,
                    __constant uint *octree_dimensions,
                    float octree_root_node_size,
                    __constant Primitive* primitives,
                    float *ray_cframe,
                    float *out_t,
                    bool debug_mode)
    {
        // printf("walk chunks\n");
        // 1. Compute starting chunk indices (ix, iy, iz)
        float start_chunk_pos[3];
        int start_chunk_indices[3];
        uint root_node_index = select_octree_root_node(octree_root_node_indices,
                                                       octree_root_node_position,
                                                       octree_dimensions,
                                                       octree_root_node_size,
                                                       ray_cframe,
                                                       start_chunk_pos,
                                                       start_chunk_indices);
        if (debug_mode)
            printf("Selected root node %d with position (%f, %f, %f) with cframe pos (%f, %f, %f)\n", root_node_index, start_chunk_pos[0], start_chunk_pos[1], start_chunk_pos[2], ray_cframe[0], ray_cframe[1], ray_cframe[2]);
        int ix = start_chunk_indices[0];
        int iy = start_chunk_indices[1];
        int iz = start_chunk_indices[2];

        // 2. Compute step direction for each axis
        float dir[3]    = { -ray_cframe[5], -ray_cframe[8], -ray_cframe[11] };
        float invdir[3]  = { 1.0f / dir[0], 1.0f / dir[1], 1.0f / dir[2] };
        int step_x = (dir[0] > 0) ? 1 : -1;
        int step_y = (dir[1] > 0) ? 1 : -1;
        int step_z = (dir[2] > 0) ? 1 : -1;
        if (debug_mode)
            printf("ray direction is %f %f %f\n", -ray_cframe[5], -ray_cframe[8], -ray_cframe[11]);

        // 3. Compute tMax and tDelta for each axis
        float next_boundary_x = octree_root_node_position[0] + (ix + (step_x > 0 ? 1 : 0)) * octree_root_node_size;
        float next_boundary_y = octree_root_node_position[1] + (iy + (step_y > 0 ? 1 : 0)) * octree_root_node_size;
        float next_boundary_z = octree_root_node_position[2] + (iz + (step_z > 0 ? 1 : 0)) * octree_root_node_size;

        float tMaxX = (dir[0] != 0) ? (next_boundary_x - ray_cframe[0]) * invdir[0] : INFINITY;
        float tMaxY = (dir[1] != 0) ? (next_boundary_y - ray_cframe[1]) * invdir[1] : INFINITY;
        float tMaxZ = (dir[2] != 0) ? (next_boundary_z - ray_cframe[2]) * invdir[2] : INFINITY;

        float tDeltaX = (dir[0] != 0) ? octree_root_node_size * fabs(invdir[0]) : INFINITY;
        float tDeltaY = (dir[1] != 0) ? octree_root_node_size * fabs(invdir[1]) : INFINITY;
        float tDeltaZ = (dir[2] != 0) ? octree_root_node_size * fabs(invdir[2]) : INFINITY;

        // 4. Traverse chunks
        int index_found = -1;
        float t = 999999.0f;
        while (index_found < 0 &&
               ix >= 0 && ix < octree_dimensions[0] &&
               iy >= 0 && iy < octree_dimensions[1] &&
               iz >= 0 && iz < octree_dimensions[2]) {

            // Traverse the octree in chunk (ix, iy, iz)
            int chunk_index = octree_root_node_indices[ix + iy * octree_dimensions[0] + iz * octree_dimensions[0] * octree_dimensions[1]];
            float chunk_min[3] = {
                octree_root_node_position[0] + ix * octree_root_node_size,
                octree_root_node_position[1] + iy * octree_root_node_size,
                octree_root_node_position[2] + iz * octree_root_node_size
            };
            if (debug_mode)
                printf("Traversing chunk %d at (%d, %d, %d) with position (%f, %f, %f)\n", chunk_index, ix, iy, iz, chunk_min[0], chunk_min[1], chunk_min[2]);
            index_found = walk_octree(nodes,
                                    octree_root_node_size,
                                    chunk_index,
                                    chunk_min,
                                    primitives,
                                    ray_cframe,
                                    &t,
                                    debug_mode);

            // Step to next chunk
            if (tMaxX < tMaxY && tMaxX < tMaxZ) {
                tMaxX += tDeltaX;
                ix += step_x;
            } else if (tMaxY < tMaxZ) {
                tMaxY += tDeltaY;
                iy += step_y;
            } else {
                tMaxZ += tDeltaZ;
                iz += step_z;
            }
        }
        *out_t = t;
        return index_found;
    }

    int intersect_objects(__constant Primitive* primitives,
                          unsigned int primitive_amount,
                          __constant OctreeNode* nodes,
                          __constant uint *octree_root_node_indices,
                          __constant float *octree_root_node_position,
                          __constant uint *octree_dimensions,
                          float octree_root_node_size,
                          float *ray_cframe,
                          float *out_t,
                          float *out_position,
                          float *out_normal,
                          uchar *out_color,
                          float *out_transparency,
                          float *out_reflectance,
                          float *out_refractive_index,
                          bool debug_mode)
    {
        // printf("START: ray cframe origin is (%f, %f, %f) with direction (%f, %f, %f)\n", ray_cframe[0], ray_cframe[1], ray_cframe[2], ray_cframe[5], ray_cframe[8], ray_cframe[11]);
        float t = 9999999;
        int index_found = walk_chunks(nodes,
                                      octree_root_node_indices,
                                      octree_root_node_position,
                                      octree_dimensions,
                                      octree_root_node_size,
                                      primitives,
                                      ray_cframe,
                                      &t,
                                      debug_mode);
        // float t = 9999999;
        // int index_found = -1;
        // for (int i = 0; i < primitive_amount; i++)
        // {
        //     Primitive p = primitives[i];
        //     float local_t = 0.0f;
        //     float squared_distance = calculate_squared_distance_point_to_line(p.cframe, ray_cframe);
        //     // Use squared distance to avoid unnecessary square root calculation
        //     if (squared_distance > p.render_radius * p.render_radius)
        //     {
        //         // Check if the merged spheres are not too close either
        //         int close_merge_found = 0;
        //         for (int m = 0; m < MAXIMUM_MERGES; m++)
        //         {
        //             int m_index = p.merge_indices[m];
        //             if (p.merge_radii[m] < 0.01f)
        //                 break;
        //             float merge_squared_distance = calculate_squared_distance_point_to_line(p.cframe, ray_cframe);
        //             if (merge_squared_distance < p.render_radius * p.render_radius)
        //             {
        //                 close_merge_found = 1;
        //                 break;
        //             }
        //         }
        //         if (close_merge_found == 0)
        //         {
        //             // No merges close enough
        //             continue;
        //         }
        //         // continue; // Skip this object if the ray is too far away
        //     }
        //     float step_size = 99999.0f;
        //     local_t = intersect_object(primitives,
        //                                i,
        //                                ray_cframe,
        //                                &step_size);
        //     if (local_t < t && step_size < STEP_MARGIN)
        //     {
        //         index_found = i;
        //         t = local_t;
        //     }
        // }
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
                                   __constant OctreeNode* nodes,
                                   __constant uint *octree_root_node_indices,
                                   __constant float *octree_root_node_position,
                                   __constant uint *octree_dimensions,
                                   float octree_root_node_size,
                                   float *ray_cframe,
                                   __constant float *directionlight_direction,
                                   __constant uchar *directionlight_color,
                                   float *out_transparency,
                                   float *out_reflectance,
                                   float *out_refractive_index,
                                   bool debug_mode)
    {
        float t;
        float edge_pos[3];
        *intersection_index = intersect_objects(primitives,
                                                primitive_amount,
                                                nodes,
                                                octree_root_node_indices,
                                                octree_root_node_position,
                                                octree_dimensions,
                                                octree_root_node_size,
                                                ray_cframe,
                                                &t,
                                                edge_pos,
                                                out_normal,
                                                out_color,
                                                out_transparency,
                                                out_reflectance,
                                                out_refractive_index,
                                                debug_mode);

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
                if (debug_mode)
                    printf("TRYING SHADOW");
                float dl_t;
                float light_edge_pos[3];
                float light_normal[3];
                uchar shadow_color[3];
                float shadow_transparency;
                float shadow_reflectance;
                float shadow_refractive_index;
                int dl_int_index = intersect_objects(primitives,
                                                     primitive_amount,
                                                     nodes,
                                                     octree_root_node_indices,
                                                     octree_root_node_position,
                                                     octree_dimensions,
                                                     octree_root_node_size,
                                                     edge_to_dir_light,
                                                     &dl_t,
                                                     light_edge_pos,
                                                     light_normal,
                                                     shadow_color,
                                                     &shadow_transparency,
                                                     &shadow_reflectance,
                                                     &shadow_refractive_index,
                                                     debug_mode);
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
                      __constant OctreeNode* nodes,
                      __constant uint *octree_root_node_indices,
                      __constant float *octree_root_node_position,
                      __constant uint *octree_dimensions,
                      float octree_root_node_size,
                      __constant float *camera_cframe,
                      float *ray_rotation_matrix,
                      __constant float *directionlight_direction,
                      __constant uchar *directionlight_color,
                      bool debug_mode)
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
                                      nodes,
                                      octree_root_node_indices,
                                      octree_root_node_position,
                                      octree_dimensions,
                                      octree_root_node_size,
                                      ray_stack[stack_ptr].cframe,
                                      directionlight_direction,
                                      directionlight_color,
                                      &calc_transparency,
                                      &calc_reflectance,
                                      &calc_refractive_index,
                                      debug_mode);
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
                         __constant uint *octree_root_node_indices,
                         __constant float *octree_root_node_position,
                         __constant uint *octree_dimensions,
                         float octree_root_node_size,
                         __constant Primitive* primitives,
                         unsigned int primitive_amount) {
        int debug_x = (int)(12800.0f * 0.5f);
        int debug_y = (int)(7200.0f * 0.5f);
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
        if (x == debug_x && y == debug_y)
            render_pixel(output_buffer, primitives, primitive_amount, nodes, octree_root_node_indices, octree_root_node_position, octree_dimensions,octree_root_node_size, camera, cam_ray, directionlight_direction, directionlight_color, true);
        else
            render_pixel(output_buffer, primitives, primitive_amount, nodes, octree_root_node_indices, octree_root_node_position, octree_dimensions,octree_root_node_size, camera, cam_ray, directionlight_direction, directionlight_color, false);
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

    pub fn render_frame(
        &mut self,
        mut camera: Camera,
        directionlight_direction: [f32; 3],
        directionlight_color: [u8; 3],
        primitives: &Vec<Primitive>,
        octree_nodes: &Vec<OctreeNode>,
        octree_root_node_size: f32,
        octree_root_position: (f32, f32, f32),
        octree_root_node_indices: &Vec<u32>,
        octree_dimensions: (u32, u32, u32)
    ) -> Result<Vec::<u8>, RendererError> {
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

        // println!("octree content: {:?}", octree_nodes);
        let octree_buffer = Buffer::builder().queue(self.pro_que.as_mut().ok_or(RendererError::RendererNotInitializedError)?.queue().clone())
            .flags(MemFlags::new().read_write())
            .len(octree_nodes.len())
            .copy_host_slice(octree_nodes)
            .build().map_err(|e| RendererError::CreateBufferError(e))?;

        let octree_root_node_indices_buffer = Buffer::builder().queue(self.pro_que.as_mut().ok_or(RendererError::RendererNotInitializedError)?.queue().clone())
            .flags(MemFlags::new().read_write())
            .len(octree_root_node_indices.len())
            .copy_host_slice(octree_root_node_indices)
            .build().map_err(|e| RendererError::CreateBufferError(e))?;

        let octree_root_node_position_buffer = Buffer::builder().queue(self.pro_que.as_mut().ok_or(RendererError::RendererNotInitializedError)?.queue().clone())
            .flags(MemFlags::new().read_write())
            .len(3)
            .copy_host_slice(&[octree_root_position.0, octree_root_position.1, octree_root_position.2])
            .build().map_err(|e| RendererError::CreateBufferError(e))?;

        let octree_dimensions_buffer = Buffer::builder().queue(self.pro_que.as_mut().ok_or(RendererError::RendererNotInitializedError)?.queue().clone())
            .flags(MemFlags::new().read_write())
            .len(3)
            .copy_host_slice(&[octree_dimensions.0, octree_dimensions.1, octree_dimensions.2])
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
            .arg(octree_root_node_indices_buffer)
            .arg(octree_root_node_position_buffer)
            .arg(octree_dimensions_buffer)
            .arg(octree_root_node_size)
            .arg(primitive_buffer)
            .arg(primitives.len() as u32)
            .build().map_err(|e| RendererError::AddArgumentsError(e))?;

        unsafe { kernel.enq().map_err(|e| RendererError::ExecuteKernelError(e))?; }

        let mut vec = vec![0u8; self.output_buffer.as_ref().ok_or(RendererError::RendererNotInitializedError)?.len()];
        self.output_buffer.as_ref().ok_or(RendererError::RendererNotInitializedError)?.read(&mut vec).enq().map_err(|e| RendererError::ReadBufferError(e))?;

        return Ok(vec);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::world::World;
    use crate::engine::util::cframe::Positionable;
    use image::{ImageBuffer, Rgba};
    const DEFAULT_WIDTH: u32 = 1280;
    const DEFAULT_HEIGHT: u32 = 720;
    const DEFAULT_FOV: f32 = 90.0;
    const DEFAULT_FOCAL_LENGTH: f32 = 0.1;

    fn create_world() -> World {
        let mut world = World::new();
        world
    }

    fn save_image(
        data: &[u8],
        width: u32,
        height: u32,
        path: &str,
    ) {
        // Assumes RGBA8 format
        let img = ImageBuffer::<Rgba<u8>, _>::from_raw(width, height, data)
            .expect("Failed to create image buffer from raw data");
        img.save(path).expect("Failed to save image");
    }

    #[test]
    fn test_render_frame() {
        // Just testing that renderer doesn't go into error
        let mut renderer: Renderer = Renderer::new(DEFAULT_WIDTH, DEFAULT_HEIGHT);
        renderer.init().expect("Failed to initialize renderer");
        let mut world = create_world();
        let camera = Camera::new(DEFAULT_FOV, DEFAULT_FOCAL_LENGTH);
        let directionlight_direction = world.get_direction_light_direction_vec();
        let directionlight_color = world.get_direction_light_color_vec();
        world.push_primitive(Primitive::new_sphere(1.0));
        let result = renderer.render_frame(camera, directionlight_direction, directionlight_color, world.get_primitives(), world.get_octree_nodes(), world.get_octree_root_node_size(), world.get_octree_root_position(), world.get_octree_root_node_indices(), world.get_octree_dimensions());
        assert!(result.is_ok());
    }

    #[test]
    fn test_render_sphere() {
        // Test that the rendered sphere results in a circle on the screen
        let mut renderer: Renderer = Renderer::new(DEFAULT_WIDTH, DEFAULT_HEIGHT);
        renderer.init().expect("Failed to initialize renderer");
        let mut world = create_world();
        world.set_direction_light_direction([0.0, 0.0, -1.0]);
        let camera = Camera::new(DEFAULT_FOV, DEFAULT_FOCAL_LENGTH);
        let directionlight_direction = world.get_direction_light_direction_vec();
        let directionlight_color = world.get_direction_light_color_vec();
        let mut sphere = Primitive::new_sphere(1.0);
        let sphere_offset = 5f32;
        sphere.set_position(0f32, 0f32, -sphere_offset);
        sphere.set_color([0xFF, 0x00, 0x00]);
        world.push_primitive(sphere);
        let colors = renderer.render_frame(camera, directionlight_direction, directionlight_color, world.get_primitives(), world.get_octree_nodes(), world.get_octree_root_node_size(), world.get_octree_root_position(), world.get_octree_root_node_indices(), world.get_octree_dimensions()).expect("failed to render frame");
        // assert_eq!(1, 2);
        save_image(&colors, DEFAULT_WIDTH, DEFAULT_HEIGHT, "artifacts/render_sphere.png");
        let center_x = (DEFAULT_WIDTH / 2) as i32;
        let center_y = (DEFAULT_HEIGHT / 2) as i32;

        let mut found_coordinate = 0;
        for x in 0..DEFAULT_WIDTH as i32 {
            let offset = ((DEFAULT_HEIGHT / 2 * DEFAULT_WIDTH + x as u32) * 4) as usize;
            let pixel = &colors[offset..offset + 4];
            if pixel[0] > 0 {
                found_coordinate = x;
                break;
            }
        }
        let expected_radius = (DEFAULT_WIDTH as f32 / 2.0) as i32 - found_coordinate as i32;
        let margin = 1;
        for y in 0..DEFAULT_HEIGHT as i32 {
            for x in 0..DEFAULT_WIDTH as i32 {
                let dx = x - center_x;
                let dy = y - center_y;
                let dist2 = dx * dx + dy * dy;
                let offset = ((y as u32 * DEFAULT_WIDTH + x as u32) * 4) as usize;
                let pixel = &colors[offset..offset + 4];

                if dist2 < (expected_radius - margin) * (expected_radius - margin) {
                    // Should be sphere color (red)
                    assert!(pixel[0] > 0 && pixel[1] == 0x00 && pixel[2] == 0x00);
                } else if dist2 > (expected_radius + margin) * (expected_radius + margin) {
                    // Should be background (black)
                    assert!(pixel[0] == 0x00 && pixel[1] == 0x00 && pixel[2] == 0x00);
                }
            }
        }
    }

    #[test]
    fn test_render_multiple_spheres() {
        // Test a scene with multiple spheres
        let mut renderer: Renderer = Renderer::new(DEFAULT_WIDTH, DEFAULT_HEIGHT);
        renderer.init().expect("Failed to initialize renderer");
        let mut world = create_world();
        world.set_direction_light_direction([0.0, 0.0, -1.0]);
        let camera = Camera::new(DEFAULT_FOV, DEFAULT_FOCAL_LENGTH);
        let directionlight_direction = world.get_direction_light_direction_vec();
        let directionlight_color = world.get_direction_light_color_vec();
        let mut red_sphere = Primitive::new_sphere(1.0);
        red_sphere.set_position(0f32, 0f32, -10f32);
        red_sphere.set_color([0xFF, 0x00, 0x00]);
        let mut blue_sphere = Primitive::new_sphere(1.0);
        blue_sphere.set_position(5f32, 5f32, -25f32);
        blue_sphere.set_color([0x00, 0x00, 0xFF]);
        let mut green_sphere = Primitive::new_sphere(1.0);
        green_sphere.set_position(-5f32, -5f32, -37f32);
        green_sphere.set_color([0x00, 0xFF, 0x00]);
        let mut yellow_sphere = Primitive::new_sphere(1.0);
        yellow_sphere.set_position(-3f32, 2f32, -17f32);
        yellow_sphere.set_color([0xFF, 0xFF, 0x00]);
        let mut pink_sphere = Primitive::new_sphere(1.0);
        pink_sphere.set_position(8f32, -2f32, -20f32);
        pink_sphere.set_color([0xFF, 0x00, 0xFF]);
        world.push_primitive(red_sphere);
        world.push_primitive(blue_sphere);
        world.push_primitive(green_sphere);
        world.push_primitive(yellow_sphere);
        world.push_primitive(pink_sphere);
        let colors = renderer.render_frame(camera, directionlight_direction, directionlight_color, world.get_primitives(), world.get_octree_nodes(), world.get_octree_root_node_size(), world.get_octree_root_position(), world.get_octree_root_node_indices(), world.get_octree_dimensions()).expect("failed to render frame");
        save_image(&colors, DEFAULT_WIDTH, DEFAULT_HEIGHT, "artifacts/render_multiple_sphere.png");

        let mut red_found = false;
        let mut green_found = false;
        let mut blue_found = false;
        let mut yellow_found = false;
        let mut pink_found = false;
        let margin = 10;
        for y in 0..DEFAULT_HEIGHT as i32 {
            for x in 0..DEFAULT_WIDTH as i32 {
                let offset = ((y as u32 * DEFAULT_WIDTH + x as u32) * 4) as usize;
                let pixel = &colors[offset..offset + 4];

                if pixel[0] >= 0xFF - margin {
                    if pixel[1] < 0x10 && pixel[2] < 0x10 {
                        red_found = true;
                        continue;
                    } else if pixel[2] >= 0xFF - margin && pixel[1] < 0x10 {
                        pink_found = true;
                        continue;
                    } else if pixel[1] >= 0xFF - margin && pixel[2] < 0x10 {
                        yellow_found = true;
                        continue;
                    }
                }
                if pixel[1] >= 0xFF - margin && pixel[2] < 0x10 && pixel[0] < 0x10 {
                    green_found = true;
                    continue;
                } else if pixel[2] >= 0xFF - margin && pixel[0] < 0x10 && pixel[1] < 0x10 {
                    blue_found = true;
                    continue;
                }
            }
        }
        assert!(red_found, "Red sphere not found in rendered image");
        assert!(green_found, "Green sphere not found in rendered image");
        assert!(blue_found, "Blue sphere not found in rendered image");
        assert!(yellow_found, "Yellow sphere not found in rendered image");
        assert!(pink_found, "Pink sphere not found in rendered image");
    }

    #[test]
    fn test_sphere_merge_3D_and_color() {
        // Test that two spheres that are close together merge into a sort of metaball
        let mut renderer: Renderer = Renderer::new(DEFAULT_WIDTH, DEFAULT_HEIGHT);
        renderer.init().expect("Failed to initialize renderer");
        let mut world = create_world();
        world.set_direction_light_direction([0.0, 0.0, -1.0]);
        let camera = Camera::new(DEFAULT_FOV, DEFAULT_FOCAL_LENGTH);
        let directionlight_direction = world.get_direction_light_direction_vec();
        let directionlight_color = world.get_direction_light_color_vec();
        let mut sphere1 = Primitive::new_sphere(1.0);
        sphere1.set_position(-0.8f32, 0f32, -5f32);
        sphere1.set_color([0xFF, 0x00, 0x00]);
        sphere1.set_render_radius(2.0f32);
        let mut sphere2 = Primitive::new_sphere(1.0);
        sphere2.set_position(0.8f32, 0f32, -5f32);
        sphere2.set_color([0x00, 0x00, 0xFF]);
        sphere2.set_render_radius(2.0f32);
        let s1 = world.push_primitive(sphere1);
        let s2 = world.push_primitive(sphere2);
        world.merge_primitives(s1, s2, 1.0f32);
        let colors = renderer.render_frame(camera, directionlight_direction, directionlight_color, world.get_primitives(), world.get_octree_nodes(), world.get_octree_root_node_size(), world.get_octree_root_position(), world.get_octree_root_node_indices(), world.get_octree_dimensions()).expect("failed to render frame");
        save_image(&colors, DEFAULT_WIDTH, DEFAULT_HEIGHT, "artifacts/render_sphere_merge.png");

        let mut found_coordinate = 0;
        for x in 0..DEFAULT_WIDTH as i32 {
            let offset = ((DEFAULT_HEIGHT / 2 * DEFAULT_WIDTH + x as u32) * 4) as usize;
            let pixel = &colors[offset..offset + 4];
            if pixel[0] > 0 {
                found_coordinate = x;
                break;
            }
        }

        // Need to use a window because the blending can be a bit noisy
        let window = 10; // should be even
        for x in found_coordinate + (window / 2)..(DEFAULT_WIDTH as i32 - found_coordinate - (window / 2)) as i32 {
            let offset = ((DEFAULT_HEIGHT / 2 * DEFAULT_WIDTH + x as u32) * 4) as usize;
            let min_offset = offset - ((window / 2) * 4) as usize;
            let max_offset = offset + ((window / 2) * 4) as usize;
            let min_pixel = &colors[min_offset..min_offset + 4];
            let max_pixel = &colors[max_offset..max_offset + 4];
            let min_n_red = min_pixel[0] as f32 / (min_pixel[0] as f32 + min_pixel[2] as f32);
            let max_n_red = max_pixel[0] as f32 / (max_pixel[0] as f32 + max_pixel[2] as f32);
            let min_n_blue = min_pixel[2] as f32 / (min_pixel[0] as f32 + min_pixel[2] as f32);
            let max_n_blue = max_pixel[2] as f32 / (max_pixel[0] as f32 + max_pixel[2] as f32);
            assert!(min_n_red > max_n_red, "Red should be decreasing");
            assert!(min_n_blue < max_n_blue, "Blue should be increasing");
        }

        // Check that there is not just color blending, but also 3D blending
        let row = 120; // This is hardcoded at the moment, didn't really feel like doing math on this
        // Basically, 2 separate spheres would have some empty space in between them
        // The merge removes this, making it look like 1 object, we can check that there is no empty space on this row of pixels to verify that
        let mut at_start = false;
        let mut at_end = false;
        for x in found_coordinate..(DEFAULT_WIDTH as i32 - found_coordinate) as i32 {
            let offset = ((row * DEFAULT_WIDTH + x as u32) * 4) as usize;
            let pixel = &colors[offset..offset + 4];
            if pixel[0] > 0 || pixel[2] > 0 {
                if !at_start {
                    assert!(!at_end, "Found colored pixel after empty space in the middle of the merged sphere");
                    at_start = true;
                }
            } else {
                if at_start {
                    at_end = true;
                }
            }
        }
    }

    #[test]
    fn test_sphere_merge_transparency() {
        // Test that two spheres that are close together merge into a sort of metaball
        // Test that the transparency is blended between the two spheres
        let mut renderer: Renderer = Renderer::new(DEFAULT_WIDTH, DEFAULT_HEIGHT);
        renderer.init().expect("Failed to initialize renderer");
        let mut world = create_world();
        world.set_direction_light_direction([0.0, 0.0, -1.0]);
        let camera = Camera::new(DEFAULT_FOV, DEFAULT_FOCAL_LENGTH);
        let directionlight_direction = world.get_direction_light_direction_vec();
        let directionlight_color = world.get_direction_light_color_vec();
        let mut sphere1 = Primitive::new_sphere(1.0);
        sphere1.set_position(-0.8f32, 0f32, -5f32);
        sphere1.set_color([0xFF, 0x00, 0x00]);
        sphere1.set_render_radius(2.0f32);
        let mut sphere2 = Primitive::new_sphere(1.0);
        sphere2.set_position(0.8f32, 0f32, -5f32);
        sphere2.set_color([0xFF, 0x00, 0x00]);
        sphere2.set_render_radius(2.0f32);
        sphere2.set_transparency(1.0f32);
        let s1 = world.push_primitive(sphere1);
        let s2 = world.push_primitive(sphere2);
        world.merge_primitives(s1, s2, 1.0f32);
        let colors = renderer.render_frame(camera, directionlight_direction, directionlight_color, world.get_primitives(), world.get_octree_nodes(), world.get_octree_root_node_size(), world.get_octree_root_position(), world.get_octree_root_node_indices(), world.get_octree_dimensions()).expect("failed to render frame");
        save_image(&colors, DEFAULT_WIDTH, DEFAULT_HEIGHT, "artifacts/render_sphere_merge_transparency.png");

        let mut found_coordinate = 0;
        for x in 0..DEFAULT_WIDTH as i32 {
            let offset = ((DEFAULT_HEIGHT / 2 * DEFAULT_WIDTH + x as u32) * 4) as usize;
            let pixel = &colors[offset..offset + 4];
            if pixel[0] > 0 {
                found_coordinate = x;
                break;
            }
        }

        // Need to use a window because the blending can be a bit noisy
        let window = 10; // should be even
        for x in found_coordinate + (window / 2)..(DEFAULT_WIDTH as i32 - found_coordinate - (window / 2)) as i32 {
            let offset = ((DEFAULT_HEIGHT / 2 * DEFAULT_WIDTH + x as u32) * 4) as usize;
            let min_offset = offset - ((window / 2) * 4) as usize;
            let max_offset = offset + ((window / 2) * 4) as usize;
            let min_pixel = &colors[min_offset..min_offset + 4];
            let max_pixel = &colors[max_offset..max_offset + 4];
            assert!(min_pixel[0] > max_pixel[0], "Red should be decreasing");
        }
    }

    #[test]
    fn test_sphere_merge_reflectance() {
        // Test that two spheres that are close together merge into a sort of metaball
        // Test that the reflectance is blended between the two spheres
        // TODO: when nothing renders, this also passes
        let mut renderer: Renderer = Renderer::new(DEFAULT_WIDTH, DEFAULT_HEIGHT);
        renderer.init().expect("Failed to initialize renderer");
        let mut world = create_world();
        world.set_direction_light_direction([0.0, 0.0, -1.0]);
        let camera = Camera::new(DEFAULT_FOV, DEFAULT_FOCAL_LENGTH);
        let directionlight_direction = world.get_direction_light_direction_vec();
        let directionlight_color = world.get_direction_light_color_vec();
        let mut sphere1 = Primitive::new_sphere(1.0);
        sphere1.set_position(-0.8f32, 0f32, -5f32);
        sphere1.set_color([0xFF, 0x00, 0x00]);
        sphere1.set_render_radius(2.0f32);
        let mut sphere2 = Primitive::new_sphere(1.0);
        sphere2.set_position(0.8f32, 0f32, -5f32);
        sphere2.set_color([0xFF, 0x00, 0x00]);
        sphere2.set_render_radius(2.0f32);
        sphere2.set_reflectance(1.0f32);
        let s1 = world.push_primitive(sphere1);
        let s2 = world.push_primitive(sphere2);
        world.merge_primitives(s1, s2, 1.0f32);
        let colors = renderer.render_frame(camera, directionlight_direction, directionlight_color, world.get_primitives(), world.get_octree_nodes(), world.get_octree_root_node_size(), world.get_octree_root_position(), world.get_octree_root_node_indices(), world.get_octree_dimensions()).expect("failed to render frame");
        save_image(&colors, DEFAULT_WIDTH, DEFAULT_HEIGHT, "artifacts/render_sphere_merge_reflectance.png");

        let mut found_coordinate = 0;
        for x in 0..DEFAULT_WIDTH as i32 {
            let offset = ((DEFAULT_HEIGHT / 2 * DEFAULT_WIDTH + x as u32) * 4) as usize;
            let pixel = &colors[offset..offset + 4];
            if pixel[0] > 0 {
                found_coordinate = x;
                break;
            }
        }
        // Actually start from the middle of the right sphere to eliminate the diffusing of the light on the left side
        found_coordinate += (DEFAULT_WIDTH as i32 / 2) - found_coordinate;

        // Need to use a window because the blending can be a bit noisy
        let window = 10; // should be even
        for x in found_coordinate + (window / 2)..(DEFAULT_WIDTH as i32 - found_coordinate - (window / 2)) as i32 {
            let offset = ((DEFAULT_HEIGHT / 2 * DEFAULT_WIDTH + x as u32) * 4) as usize;
            let min_offset = offset - ((window / 2) * 4) as usize;
            let max_offset = offset + ((window / 2) * 4) as usize;
            let min_pixel = &colors[min_offset..min_offset + 4];
            let max_pixel = &colors[max_offset..max_offset + 4];
            assert!(min_pixel[0] > max_pixel[0], "Red should be decreasing");
        }
    }

    #[test]
    fn test_solid_shadow() {
        // Test that a solid object casts a shadow
        // TODO: when nothing renders, this also passes...
        let mut renderer: Renderer = Renderer::new(DEFAULT_WIDTH, DEFAULT_HEIGHT);
        renderer.init().expect("Failed to initialize renderer");
        let mut world = create_world();
        world.set_direction_light_direction([0.0, 0.0, -1.0]);
        let camera = Camera::new(DEFAULT_FOV, DEFAULT_FOCAL_LENGTH);
        let directionlight_direction = world.get_direction_light_direction_vec();
        let directionlight_color = world.get_direction_light_color_vec();
        let mut sphere = Primitive::new_sphere(1.0);
        sphere.set_position(0.0f32, 0f32, -5f32);
        sphere.set_color([0xFF, 0x00, 0x00]);
        world.push_primitive(sphere);
        // Put a second sphere behind the camera, a shadow should be cast on the first sphere
        let mut sphere2 = Primitive::new_sphere(0.5);
        sphere2.set_position(0.0f32, 0f32, 5f32);
        world.push_primitive(sphere2);
        let colors = renderer.render_frame(camera, directionlight_direction, directionlight_color, world.get_primitives(), world.get_octree_nodes(), world.get_octree_root_node_size(), world.get_octree_root_position(), world.get_octree_root_node_indices(), world.get_octree_dimensions()).expect("failed to render frame");
        save_image(&colors, DEFAULT_WIDTH, DEFAULT_HEIGHT, "artifacts/render_solid_shadow.png");

        let mut found_coordinate = 0;
        for x in 0..DEFAULT_WIDTH as i32 {
            let offset = ((DEFAULT_HEIGHT / 2 * DEFAULT_WIDTH + x as u32) * 4) as usize;
            let pixel = &colors[offset..offset + 4];
            if pixel[0] > 0 {
                found_coordinate = x;
                break;
            }
        }
        let mut inner_edge_found = 0;
        for x in found_coordinate..(DEFAULT_WIDTH / 2) as i32 {
            let offset = ((DEFAULT_HEIGHT / 2 * DEFAULT_WIDTH + x as u32) * 4) as usize;
            let pixel = &colors[offset..offset + 4];
            if pixel[0] == 0x00 {
                inner_edge_found = x;
                break;
            }
        }

        let expected_radius = (DEFAULT_WIDTH as f32 / 2.0) as i32 - found_coordinate as i32;
        let expected_shadow_radius = (DEFAULT_WIDTH as f32 / 2.0) as i32 - inner_edge_found as i32;

        let center_x = (DEFAULT_WIDTH / 2) as i32;
        let center_y = (DEFAULT_HEIGHT / 2) as i32;

        let margin = 1;
        for y in 0..DEFAULT_HEIGHT as i32 {
            for x in 0..DEFAULT_WIDTH as i32 {
                let dx = x - center_x;
                let dy = y - center_y;
                let dist2 = dx * dx + dy * dy;
                let offset = ((y as u32 * DEFAULT_WIDTH + x as u32) * 4) as usize;
                let pixel = &colors[offset..offset + 4];

                if dist2 < (expected_shadow_radius - margin) * (expected_shadow_radius - margin) {
                    // Should be shadow color black
                    assert!(pixel[0] == 0x00 && pixel[1] == 0x00 && pixel[2] == 0x00);
                } else if dist2 > (expected_shadow_radius + margin) * (expected_shadow_radius + margin) &&
                          dist2 < (expected_radius - margin) * (expected_radius - margin) {
                    // Should be sphere color (red)
                    assert!(pixel[0] > 0 && pixel[1] == 0x00 && pixel[2] == 0x00);
                } else if dist2 > (expected_radius + margin) * (expected_radius + margin) {
                    // Should be background (black)
                    assert!(pixel[0] == 0x00 && pixel[1] == 0x00 && pixel[2] == 0x00);
                }
            }
        }
    }

    #[test]
    fn test_transparent_shadow() {
        // Test that a transparent object casts a shadow
        // Stack 2 transparent shadows to check blending
        let mut renderer: Renderer = Renderer::new(DEFAULT_WIDTH, DEFAULT_HEIGHT);
        renderer.init().expect("Failed to initialize renderer");
        let mut world = create_world();
        world.set_direction_light_direction([0.01 / ((0.01 * 0.01 * 2.0 + 1.0) as f32).sqrt(), 0.01 / ((0.01 * 0.01 * 2.0 + 1.0) as f32).sqrt(), -1.0 / ((0.01 * 0.01 * 2.0 + 1.0) as f32).sqrt()]);
        let camera = Camera::new(DEFAULT_FOV, DEFAULT_FOCAL_LENGTH);
        let directionlight_direction = world.get_direction_light_direction_vec();
        let directionlight_color = world.get_direction_light_color_vec();
        let mut sphere = Primitive::new_sphere(1.0);
        sphere.set_position(0.0f32, 0f32, -5f32);
        sphere.set_color([0xFF, 0xFF, 0xFF]);
        world.push_primitive(sphere);
        // Put a second sphere behind the camera, a shadow should be cast on the first sphere
        let mut sphere2 = Primitive::new_sphere(0.5);
        sphere2.set_position(0.0f32, 0f32, 5f32);
        sphere2.set_transparency(0.5f32);
        sphere2.set_color([0x00, 0xFF, 0x00]);
        world.push_primitive(sphere2);
        let mut sphere3 = Primitive::new_sphere(0.25);
        sphere3.set_position(0.0f32, 0f32, 10f32);
        sphere3.set_transparency(0.5f32);
        sphere3.set_color([0x00, 0x00, 0xFF]);
        world.push_primitive(sphere3);
        let colors = renderer.render_frame(camera, directionlight_direction, directionlight_color, world.get_primitives(), world.get_octree_nodes(), world.get_octree_root_node_size(), world.get_octree_root_position(), world.get_octree_root_node_indices(), world.get_octree_dimensions()).expect("failed to render frame");
        save_image(&colors, DEFAULT_WIDTH, DEFAULT_HEIGHT, "artifacts/render_transparent_shadow.png");

        let mut found_coordinate = 0;
        for x in 0..DEFAULT_WIDTH as i32 {
            let offset = ((DEFAULT_HEIGHT / 2 * DEFAULT_WIDTH + x as u32) * 4) as usize;
            let pixel = &colors[offset..offset + 4];
            if pixel[0] > 0 {
                found_coordinate = x;
                break;
            }
        }
        let mut inner_edge_found = 0;
        for x in (found_coordinate + 10)..(DEFAULT_WIDTH / 2) as i32 {
            let offset = ((DEFAULT_HEIGHT / 2 * DEFAULT_WIDTH + x as u32) * 4) as usize;
            let pixel = &colors[offset..offset + 4];
            if pixel[0] < 120 {
                inner_edge_found = x;
                break;
            }
        }
        let mut inner_edge_found2 = 0;
        for x in inner_edge_found..(DEFAULT_WIDTH / 2) as i32 {
            let offset = ((DEFAULT_HEIGHT / 2 * DEFAULT_WIDTH + x as u32) * 4) as usize;
            let pixel = &colors[offset..offset + 4];
            if pixel[0] < 80 {
                inner_edge_found2 = x;
                break;
            }
        }

        let expected_radius = (DEFAULT_WIDTH as f32 / 2.0) as i32 - found_coordinate as i32;
        let expected_shadow_radius = (DEFAULT_WIDTH as f32 / 2.0) as i32 - inner_edge_found as i32;
        let expected_shadow_radius2 = (DEFAULT_WIDTH as f32 / 2.0) as i32 - inner_edge_found2 as i32;

        let center_x = (DEFAULT_WIDTH / 2) as i32;
        let center_y = (DEFAULT_HEIGHT / 2) as i32;

        let margin = 5;
        for y in 0..DEFAULT_HEIGHT as i32 {
            for x in 0..DEFAULT_WIDTH as i32 {
                let dx = x - center_x;
                let dy = y - center_y;
                let dist2 = dx * dx + dy * dy;
                let offset = ((y as u32 * DEFAULT_WIDTH + x as u32) * 4) as usize;
                let pixel = &colors[offset..offset + 4];

                if dist2 < (expected_shadow_radius2 - margin) * (expected_shadow_radius2 - margin) {
                    // 2 overlapping transparent shadows, should contain very little red, but a bit of green and blue
                    assert!(pixel[0] < 80 && pixel[1] > 100 && pixel[2] > 100);
                } else if dist2 > (expected_shadow_radius2 + margin) * (expected_shadow_radius2 + margin) &&
                          dist2 < (expected_shadow_radius - margin) * (expected_shadow_radius - margin) {
                    // Just 1 shadow, should contain more red and a lot more green
                    assert!(pixel[0] > 100 && pixel[1] > 200 && pixel[2] > 100);
                } else if dist2 > (expected_shadow_radius + margin) * (expected_shadow_radius + margin) &&
                          dist2 < (expected_radius - margin) * (expected_radius - margin) {
                    // No shadow, object is white, so all colors should be equal and present
                    assert!(pixel[0] > 0x00 && pixel[1] > 0x00 && pixel[2] > 0x00);
                    assert!(pixel[0] == pixel[1] && pixel[1] == pixel[2], "Colors should be equal for white object");
                } else if dist2 > (expected_radius + margin) * (expected_radius + margin) {
                    // Outside object, should be black
                    assert!(pixel[0] == 0x00 && pixel[1] == 0x00 && pixel[2] == 0x00);
                }
            }
        }
    }
}