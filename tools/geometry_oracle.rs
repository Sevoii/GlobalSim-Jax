// Destination-only oracle; imports pinned read-only source without modifying it.
#[path = "C:/Users/a/RustroverProjects/GlobalSim/src/replay_frames.rs"]
mod replay_frames;
#[path = "C:/Users/a/RustroverProjects/GlobalSim/src/json.rs"]
mod json;
#[path = "C:/Users/a/RustroverProjects/GlobalSim/src/retail_geometry.rs"]
mod retail_geometry;

fn main() {
    let text = std::fs::read_to_string("C:/Users/a/RustroverProjects/GlobalSim/data/course_geometry/10101.json").unwrap();
    let data = json::parse(&text).unwrap();
    let geometry = retail_geometry::CourseGeometry::from_value(&data, 10101, 1200.).unwrap();
    // Frozen source-boundary observations for the supported 10101/1200m asset.
    // Reference revision: 58da7bc4693d9aa8f6340e348cba1c5b5fb5a856.
    print!("{{\"revision\":\"58da7bc4693d9aa8f6340e348cba1c5b5fb5a856\",\"position_bits\":[");
    for (index, distance) in [0.0_f32, 1.2, 600.0, 1198.8, 1200.0].iter().enumerate() {
        if index > 0 { print!(","); }
        let p = geometry.position(*distance, 0.0).unwrap();
        print!("[{},{}]", p[0].to_bits(), p[1].to_bits());
    }
    let (next, world) = geometry.advance(468.12298583984375, 0.3888888955116272, 1.5, [0., 0.]).unwrap();
    println!("],\"advance_bits\":[{},{},{},{}]}}", next.to_bits(), world[0].to_bits(), world[1].to_bits(), geometry.lane_max_at(0.).to_bits());
}
