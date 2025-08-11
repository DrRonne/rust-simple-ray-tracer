use criterion::{criterion_group, criterion_main, Criterion, BatchSize};
use first_test::engine::util::octree::octree::Octree;

const AMOUNT_OF_OBJECTS: usize = 10000;
const ROOT_SIZE: f32 = 100.0;
const OBJECT_SIZE: f32 = 1.0;
const CHUNK_WIDTH: u32 = 10;
const CHUNK_HEIGHT: u32 = 10;
const CHUNK_DEPTH: u32 = 10;
const ROOT_POSITION: (f32, f32, f32) = (0.0f32, 0.0f32, 0.0f32);

fn bench_move_objects(c: &mut Criterion) {
    c.bench_function(&format!("insert {} objects", AMOUNT_OF_OBJECTS), |b| {
        b.iter(|| {
            let mut tree = Octree::new(ROOT_SIZE, CHUNK_WIDTH, CHUNK_HEIGHT, CHUNK_DEPTH, ROOT_POSITION);
            for i in 0..AMOUNT_OF_OBJECTS {
                tree.move_item(i as u32, rand::random::<f32>() * ROOT_SIZE * CHUNK_WIDTH as f32, rand::random::<f32>() * ROOT_SIZE * CHUNK_HEIGHT as f32, rand::random::<f32>() * ROOT_SIZE * CHUNK_DEPTH as f32, OBJECT_SIZE, None);
            }
        });
    });

    c.bench_function(&format!("move {} objects", AMOUNT_OF_OBJECTS), |b| {
        b.iter_batched(
            || {
                // Setup: create and fill the octree
                let mut tree = Octree::new(ROOT_SIZE, CHUNK_WIDTH, CHUNK_HEIGHT, CHUNK_DEPTH, ROOT_POSITION);
                let mut positions = (0..AMOUNT_OF_OBJECTS)
                    .map(|_| (rand::random::<f32>() * ROOT_SIZE, rand::random::<f32>() * ROOT_SIZE, rand::random::<f32>() * ROOT_SIZE))
                    .collect::<Vec<_>>();
                for i in 0..AMOUNT_OF_OBJECTS {
                    tree.move_item(i as u32, positions[i].0, positions[i].1, positions[i].2, OBJECT_SIZE, None);
                }
                (tree, positions)
            },
            |(mut tree, positions)| {
                // Set position of all objects to some other random position within the octree
                for i in 0..AMOUNT_OF_OBJECTS {
                    tree.move_item(i as u32, rand::random::<f32>() * ROOT_SIZE, rand::random::<f32>() * ROOT_SIZE, rand::random::<f32>() * ROOT_SIZE, OBJECT_SIZE, Some(positions[i]));
                }
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function(&format!("remove {} objects", AMOUNT_OF_OBJECTS), |b| {
        b.iter_batched(
            || {
                // Setup: create and fill the octree
                let mut tree = Octree::new(ROOT_SIZE, CHUNK_WIDTH, CHUNK_HEIGHT, CHUNK_DEPTH, ROOT_POSITION);
                let mut positions = (0..AMOUNT_OF_OBJECTS)
                    .map(|_| (rand::random::<f32>() * ROOT_SIZE, rand::random::<f32>() * ROOT_SIZE, rand::random::<f32>() * ROOT_SIZE))
                    .collect::<Vec<_>>();
                for i in 0..AMOUNT_OF_OBJECTS {
                    tree.move_item(i as u32, positions[i].0, positions[i].1, positions[i].2, OBJECT_SIZE, None);
                }
                (tree, positions)
            },
            |(mut tree, positions)| {
                // Set position of all objects to some position outside the octree
                // This simulates removing them from the octree
                for i in 0..AMOUNT_OF_OBJECTS {
                    tree.move_item(i as u32, ROOT_SIZE * 2.0, ROOT_SIZE * 2.0, ROOT_SIZE * 2.0, OBJECT_SIZE, Some(positions[i]));
                }
            },
            BatchSize::SmallInput,
        );
    });
}

criterion_group!(benches, bench_move_objects);
criterion_main!(benches);
