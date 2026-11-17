// M3b source oracle.  It imports the pinned retail introsort unchanged; the
// tiny overlap adapter below is the verbatim `resolve_overlaps_with_order`
// body over its only two required Row fields.  It writes JSON without serde.
#[path = "C:/Users/a/RustroverProjects/GlobalSim/src/retail_sort.rs"]
mod retail_sort;

use retail_sort::comparison_sort;

#[derive(Clone, Copy)]
struct Row { position: f32, lane: f32 }

fn overlap(rows: &mut [Row], force: &mut [bool]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..rows.len()).collect();
    comparison_sort(&mut order, |&a, &b| (rows[a].lane as i32).cmp(&(rows[b].lane as i32)));
    if let Some(&first) = order.first() { force[first] = false; }
    for pair in order.windows(2) {
        let previous = rows[pair[0]];
        let current = &mut rows[pair[1]];
        if (current.position - previous.position).abs() < 0.4
            && (current.lane - previous.lane).abs() < (0.4_f64 / 18.) as f32 {
            current.lane = previous.lane + (0.4_f64 / 18.) as f32;
            force[pair[1]] = true;
        }
    }
    order
}

fn list_u(v: &[usize]) -> String { v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",") }
fn list_b(v: &[bool]) -> String { v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",") }
fn list_bits(v: &[Row]) -> String { v.iter().map(|x| x.lane.to_bits().to_string()).collect::<Vec<_>>().join(",") }

fn main() {
    let sizes = [0usize, 1, 2, 3, 16, 17, 18, 32];
    let mut sorts = Vec::new();
    for &size in &sizes {
        let mut rows: Vec<(i32, usize)> = (0..size).map(|i| (((i * 7 + 3) % 5) as i32, i)).collect();
        comparison_sort(&mut rows, |a, b| a.0.cmp(&b.0));
        let input = (0..size).map(|i| format!("[{},{}]", (i * 7 + 3) % 5, i)).collect::<Vec<_>>().join(",");
        let order: Vec<usize> = rows.iter().map(|x| x.1).collect();
        sorts.push(format!("{{\"size\":{size},\"input\":[{input}],\"order\":[{}]}}", list_u(&order)));
    }
    let special_inputs = [vec![2i32, 2, 1], vec![2, 1, 2], vec![1, 2, 2]];
    let mut special_sorts = Vec::new();
    for input in special_inputs {
        let mut rows: Vec<(i32, usize)> = input.iter().copied().enumerate().map(|(i, key)| (key, i)).collect();
        comparison_sort(&mut rows, |a, b| a.0.cmp(&b.0));
        let input_text = input.iter().enumerate().map(|(i, key)| format!("[{key},{i}]")).collect::<Vec<_>>().join(",");
        special_sorts.push(format!("{{\"input\":[{input_text}],\"order\":[{}]}}", list_u(&rows.iter().map(|x| x.1).collect::<Vec<_>>())));
    }
    let overlap_sizes = [0usize, 1, 2, 3, 9, 16, 17, 18, 24, 32];
    let mut overlaps = Vec::new();
    for &size in &overlap_sizes {
        // Equal integer lane keys exercise the source equal-key permutation;
        // alternating positions make both strict and non-overlap outcomes.
        let mut rows: Vec<Row> = (0..size).map(|i| Row {
            position: if i % 5 == 4 { 0.40000004 } else { (i % 3) as f32 * 0.01 },
            lane: (i % 4) as f32 * 0.004,
        }).collect();
        let mut force: Vec<bool> = (0..size).map(|i| i % 2 == 0).collect();
        let input_pos = rows.iter().map(|x| x.position.to_bits().to_string()).collect::<Vec<_>>().join(",");
        let input_lane = list_bits(&rows);
        let input_force = list_b(&force);
        let order = overlap(&mut rows, &mut force);
        overlaps.push(format!("{{\"size\":{size},\"position_bits\":[{input_pos}],\"lane_bits\":[{input_lane}],\"force\":[{input_force}],\"order\":[{}],\"out_lane_bits\":[{}],\"out_force\":[{}]}}", list_u(&order), list_bits(&rows), list_b(&force)));
    }
    println!("{{\"provenance\":\"pinned 58da7bc4693d9aa8f6340e348cba1c5b5fb5a856 retail_sort.rs plus verbatim retail_lane overlap adapter\",\"sorts\":[{}],\"special_sorts\":[{}],\"overlaps\":[{}]}}", sorts.join(","), special_sorts.join(","), overlaps.join(","));
}
