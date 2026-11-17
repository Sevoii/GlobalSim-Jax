// AGPL-3.0-only. Destination-only inputs, diagnostics and bit serialization.
fn h(position: f32, lane: f32, speed: f32, target: f32) -> Horse {
    Horse {
        point: Row { position, lane },
        speed,
        target,
    }
}
fn horse_json(h: Horse) -> String {
    format!(
        "[{},{},{},{}]",
        h.point.position.to_bits(),
        h.point.lane.to_bits(),
        h.speed.to_bits(),
        h.target.to_bits()
    )
}
fn horses_json(horses: &[Horse]) -> String {
    format!(
        "[{}]",
        horses
            .iter()
            .map(|&h| horse_json(h))
            .collect::<Vec<_>>()
            .join(",")
    )
}
fn parameters() -> TargetParameters {
    TargetParameters {
        block: BlockParameters {
            front_distance: 2.0,
            front_lane: 0.0416666716337204,
            side_distance: 1.0499999523162842,
            side_lane: 0.1111110970377922,
        },
        force_in_threshold: 0.2,
        cooldown: 0.5,
        outward_normalize: 1.0,
        outward_coefficient: 0.0,
        outward_random: 0.0,
    }
}
fn context(owner: Horse, vision: f32) -> TargetContext {
    TargetContext {
        neutral_pace: true,
        lane_preference: 0,
        owner,
        phase: 0,
        max_lane: 0.8,
        vision,
        first_move_point: 0.0,
        final_corner_start: None,
        empty_in_move: true,
        previous_target: 0.0,
        fixed_goal: 0.0,
        has_hp: true,
        pace_down: false,
    }
}
fn cache_json(s: &TargetState, original_owner: Horse) -> String {
    let side = s.side.unwrap();
    let (_, distance, lane, front) = s.front_cache.unwrap();
    format!(
        r#"{{"owner":{},"peers":{},"count":{},"visible":{:?},"side":[{},{},{},{},{},{}],"front":{},"front_distance":{},"front_lane":{},"force_in":{}}}"#,
        horse_json(original_owner),
        horses_json(&s.cached),
        s.cached.len(),
        s.visible,
        side.can_in,
        side.can_out,
        side.in_space.to_bits(),
        side.out_space.to_bits(),
        side.inner_blocker.unwrap_or(f32::NAN).to_bits(),
        side.inner_blocker_index.map_or(-1, |i| i as i32),
        front.map_or(-1, |i| i as i32),
        distance.to_bits(),
        lane.to_bits(),
        s.force_in
    )
}
fn state_json(s: &TargetState, original_owner: Horse) -> String {
    format!(
        r#"{{"targets":{:?},"cooldown":{},"force_check":{},"force_in":{},"extra_outward":[{},{}],"previous_can_in":{},"skip_passing":{},"cache":{}}}"#,
        s.targets,
        s.cooldown.to_bits(),
        s.force_check,
        s.force_in,
        s.extra_outward.is_some(),
        s.extra_outward.unwrap_or(f32::NAN).to_bits(),
        s.previous_can_in.map_or(-1, |x| x as i32),
        s.skip_passing,
        cache_json(s, original_owner)
    )
}
fn trace(family: usize) -> String {
    let mut owner = h(0.0, 0.2, 24.0, 25.0);
    let mut vision = 20.0;
    let mut initial_targets = vec![];
    let mut initial_cooldown = 0.0;
    let mut initial_force = false;
    let mut extra = Some(0.7);
    let mut previous = Some(false);
    let name;
    let cached;
    let mut live;
    match family {
        0 => {
            name = "ordered_old_new_front";
            cached = vec![
                h(10.0, 0.35, 20.0, 26.0),
                h(11.0, 0.4, 20.0, 26.0),
                h(12.0, 0.5, 20.0, 26.0),
                h(0.5, 0.2, 30.0, 30.0),
            ];
            live = vec![
                h(10.0, 0.35, 20.0, 24.0),
                h(11.0, 0.4, 20.0, 24.0),
                h(12.0, 0.5, 20.0, 24.0),
                h(0.5, 0.2, 30.0, 30.0),
            ];
            initial_targets = vec![2, 99, 0];
        }
        1 => {
            name = "empty_cooldown_overshoot";
            cached = vec![];
            live = vec![];
            initial_cooldown = 0.15;
        }
        2 => {
            name = "cached_live_split";
            cached = vec![h(10.0, 0.4, 20.0, 26.0)];
            live = vec![h(100.0, 0.4, 26.0, 24.0)];
        }
        3 => {
            name = "cached_peer_blocking";
            cached = vec![h(10.0, 0.4, 20.0, 26.0), h(10.5, 0.4, 20.0, 26.0)];
            live = vec![h(100.0, 0.4, 26.0, 24.0), h(100.0, 0.4, 26.0, 26.0)];
        }
        4 => {
            name = "bitwise_parameter_and_owner_fallback";
            cached = vec![h(2.0, 0.2, 30.0, 30.0)];
            live = cached.clone();
        }
        5 => {
            name = "full_capacity_retained_invisible";
            owner.point.lane = 0.4;
            vision = 0.0;
            cached = (0..32)
                .map(|i| h(3.0 + i as f32 * 0.1, 0.4, 20.0, 26.0))
                .collect::<Vec<_>>();
            live = cached
                .iter()
                .map(|v| h(v.point.position, v.point.lane, 20.0, 24.0))
                .collect();
            initial_targets = (0..32).rev().collect();
            initial_force = true;
        }
        6 => {
            name = "force_in_optional_and_exact_zero";
            cached = vec![h(0.0, 0.15, 20.0, 26.0)];
            live = cached.clone();
            initial_cooldown = 0.1;
            extra = None;
            previous = None;
        }
        7 => {
            name = "signed_zero_preservation";
            owner.point.position = -0.0;
            cached = vec![];
            live = vec![];
            initial_cooldown = -0.0;
            extra = Some(-0.0);
            previous = None;
            initial_force = true;
        }
        8 => {
            name = "UNSUPPORTED_pairwise_subnormal";
            owner.point.position = -10.0;
            let a = f32::MIN_POSITIVE;
            let b = f32::from_bits(a.to_bits() + 1);
            cached = vec![h(a, 0.4, 20.0, 26.0), h(b, 0.4, 20.0, 26.0)];
            live = vec![h(100.0, 0.4, 26.0, 24.0), h(100.0, 0.4, 26.0, 26.0)];
        }
        _ => unreachable!(),
    }
    let original_owner = owner;
    let p = parameters();
    let c = context(owner, vision);
    let mut s = TargetState::default();
    s.collect(&c, &cached, p);
    s.targets = initial_targets;
    s.cooldown = initial_cooldown;
    s.force_check = initial_force;
    s.extra_outward = extra;
    s.previous_can_in = previous;
    s.skip_passing = true;
    let initial = state_json(&s, original_owner);
    let mut updates = vec![];
    let ticks = if family == 8 { 1 } else { 4 };
    for tick in 0..ticks {
        let mut current_owner = owner;
        let mut current = p;
        let mut dt = 0.1;
        match family {
            0 => {
                if tick == 1 {
                    live[0].speed = 30.0;
                    live[0].target = 30.0;
                }
                if tick == 2 {
                    live[0].speed = 20.0;
                    live[0].target = 24.0;
                }
                if tick == 3 {
                    live[2].speed = 30.0;
                    live[2].target = 30.0;
                }
            }
            2 => {
                live[0] = match tick {
                    0 => h(100.0, 0.4, 26.0, 24.0),
                    1 => h(100.0, 0.4, 24.0, 26.0),
                    _ => h(-100.0, 0.4, 26.0, 24.0),
                };
                if tick == 3 {
                    current_owner.speed = 19.0;
                }
            }
            3 => {
                if tick % 2 == 1 {
                    current.block.front_distance = 0.25;
                }
            }
            4 => {
                if tick == 1 {
                    current.block.front_distance = f32::from_bits(2.0_f32.to_bits() + 1);
                }
                if tick >= 2 {
                    current_owner.point.position = 0.5;
                }
                if tick == 3 {
                    current_owner.point.lane = 0.5;
                }
            }
            6 => {
                dt = [0.0, 0.1, 0.1, 0.0][tick];
            }
            7 => {
                dt = 0.0;
                if tick % 2 == 1 {
                    current_owner.point.position = 0.0;
                }
            }
            _ => {}
        }
        let ctx = context(current_owner, vision);
        s.update_targets(&ctx, &live, dt, current);
        updates.push(format!(r#"{{"owner":{},"live":{},"dt":{},"cooldown_param":{},"front_distance":{},"front_lane":{},"out":{}}}"#,
            horse_json(current_owner),horses_json(&live),dt.to_bits(),current.cooldown.to_bits(),current.block.front_distance.to_bits(),current.block.front_lane.to_bits(),state_json(&s,original_owner)));
    }
    if family == 8 {
        assert!(
            s.targets.is_empty(),
            "original-source subnormal diagnostic changed"
        );
    }
    format!(
        r#"{{"name":"{}","supported":{},"collect_owner":{},"cached":{},"collect_config":{{"max_lane":{},"vision":{},"phase":0,"empty_in_move":true,"force_in_threshold":{},"front_distance":{},"front_lane":{},"side_distance":{},"side_lane":{}}},"initial":{},"updates":[{}]}}"#,
        name,
        family != 8,
        horse_json(original_owner),
        horses_json(&cached),
        0.8_f32.to_bits(),
        vision.to_bits(),
        p.force_in_threshold.to_bits(),
        p.block.front_distance.to_bits(),
        p.block.front_lane.to_bits(),
        p.block.side_distance.to_bits(),
        p.block.side_lane.to_bits(),
        initial,
        updates.join(",")
    )
}
fn main() {
    let cases = (0..8).map(trace).collect::<Vec<_>>().join(",");
    println!(
        r#"{{"reference_revision":"58da7bc4693d9aa8f6340e348cba1c5b5fb5a856","scope":"default-feature original-source update_targets subset only; no target search or RNG","cases":[{}],"diagnostics":[{}]}}"#,
        cases,
        trace(8)
    );
}
