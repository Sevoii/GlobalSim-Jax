// AGPL-3.0-only. Generated original-source subset; see TARGET_UPDATE_ORACLE.md.
#[path="lane_query_oracle.rs"] mod query;
use query::*;
mod overtaking {
//! CheckOvertakeTarget, RVA 0x340ffb0. Uses a shared pre-movement snapshot.
pub fn is_target(gap:f64, speed_gap:f64, owner_target:f64, other_target:f64,
                 other_speed:f64, directly_blocking:bool, other_blocked:bool)->bool {
    is_target_with_block_check(gap, speed_gap, owner_target, other_target,
        other_speed, directly_blocking, || other_blocked)
}

// Blocking only matters after the gap, relative-speed and direct-blocker checks.
// Let callers defer the field scan until this decision actually needs it.
pub(crate) fn is_target_with_block_check(
    gap: f64, speed_gap: f64, owner_target: f64, other_target: f64,
    other_speed: f64, directly_blocking: bool, other_blocked: impl FnOnce() -> bool,
) -> bool {
    let gap=gap as f32;let speed_gap=speed_gap as f32;
    if gap < -1.0 {return false;}
    if directly_blocking {return (other_speed as f32)<owner_target as f32;}
    if !(1.0..=20.0).contains(&gap) || speed_gap<=0.0 || gap/speed_gap>15.0 {return false;}
    let blocked_result = (other_speed as f32) < owner_target as f32;
    let unblocked_result = (other_target as f32) < owner_target as f32;
    // The blocker scan only chooses between these two comparisons. If they
    // agree, knowing whether the peer is blocked cannot change eligibility.
    if blocked_result == unblocked_result { return blocked_result; }
    if other_blocked() { blocked_result } else { unblocked_result }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn irrelevant_blocker_queries_are_skipped() {
        for (gap, speed_gap, direct) in [(-2., 1., false), (21., 1., false),
            (10., 0., false), (20., 1., false), (1., 1., true)] {
            let actual = is_target_with_block_check(gap, speed_gap, 22., 24., 20., direct,
                || panic!("irrelevant blocker query"));
            assert_eq!(actual, is_target(gap, speed_gap, 22., 24., 20., direct, false));
        }
    }

    #[test]
    fn equal_speed_comparisons_skip_blocker_queries() {
        for (owner, target, speed, expected) in [
            (22., 21., 20., true),
            (22., 24., 23., false),
            (22., 22., 22., false),
            (22., f64::NAN, 23., false),
            (22., 24., f64::NAN, false),
            (f64::NAN, 24., 20., false),
        ] {
            assert_eq!(is_target_with_block_check(10., 1., owner, target, speed,
                false, || panic!("equivalent speed comparisons do not need blockers")), expected);
        }
    }

    #[test]
    fn differing_speed_comparisons_query_blockers_once() {
        for (target, speed) in [(24., 20.), (20., 24.)] {
            for blocked in [false, true] {
                let calls = std::cell::Cell::new(0);
                let actual = is_target_with_block_check(10., 1., 22., target, speed, false, || {
                    calls.set(calls.get() + 1);
                    blocked
                });
                assert_eq!(calls.get(), 1);
                assert_eq!(actual, if blocked { speed < 22. } else { target < 22. });
            }
        }
    }

    #[test]
    fn relevant_blocker_state_selects_the_same_speed() {
        for blocked in [false, true] {
            let calls = std::cell::Cell::new(0);
            let result = is_target_with_block_check(10., 1., 22., 24., 20., false, || {
                calls.set(calls.get() + 1);
                blocked
            });
            assert_eq!(calls.get(), 1);
            assert_eq!(result, blocked);
        }
    }
}

}
#[derive(Clone, Copy, Debug)]
pub struct Horse {
    pub point: Row,
    pub speed: f32,
    /// Live target speed including active current-speed receivers.
    pub target: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct TargetParameters {
    pub block: BlockParameters,
    pub force_in_threshold: f32,
    pub cooldown: f32,
    pub outward_normalize: f32,
    pub outward_coefficient: f32,
    pub outward_random: f32,
}

pub struct TargetContext {
    pub neutral_pace: bool,
    pub lane_preference: u8,
    pub owner: Horse,
    pub phase: i32,
    pub max_lane: f32,
    pub vision: f32,
    pub first_move_point: f32,
    pub final_corner_start: Option<f32>,
    pub empty_in_move: bool,
    pub previous_target: f32,
    pub fixed_goal: f32,
    pub has_hp: bool,
    pub pace_down: bool,
}

/// Roster indices must remain stable for the lifetime of this state, even
/// after peers finish. `collect` runs before any runner moves; subsequent
/// calls deliberately also accept the separately sampled live peer values.
#[derive(Default)]
pub struct TargetState {
    pub targets: Vec<usize>,
    pub cached: Vec<Horse>,
    pub visible: Vec<usize>,
    pub side: Option<SideRoom>,
    pub cooldown: f32,
    pub force_check: bool,
    pub force_in: bool,
    pub extra_outward: Option<f32>,
    previous_can_in: Option<bool>,
    skip_passing: bool,
    front_cache: Option<(Row, f32, f32, Option<usize>)>,
    /// Owner index when `cached` is exactly the position-sorted roster's rows.
    sorted_owner: Option<usize>,
}

impl TargetState {
    pub fn collect(&mut self, c: &TargetContext, others: &[Horse], p: TargetParameters) {
        self.cached.clear();
        self.cached.extend_from_slice(others);
        self.sorted_owner = None;
        self.collect_cached(c, p);
    }

    fn collect_cached(&mut self, c: &TargetContext, p: TargetParameters) {
        let owner = c.owner.point;
        let mut side = SideRoom {
            can_in: owner.lane > 0.0001,
            can_out: owner.lane < c.max_lane,
            in_space: owner.lane,
            out_space: c.max_lane - owner.lane,
            inner_blocker: None,
            inner_blocker_index: None,
        };
        let mut front = None;
        let mut nearest = f32::INFINITY;
        let mut inner_nearby = false;
        self.visible.clear();
        for (i, horse) in self.cached.iter().enumerate() {
            let row = horse.point;
            let position_gap = row.position - owner.position;
            let lane_gap = row.lane - owner.lane;
            if is_visible(position_gap, lane_gap, c.vision) { self.visible.push(i); }
            if position_gap > 0. && position_gap < p.block.front_distance
                && position_gap < nearest
                && lane_gap.abs() <= front_lane_limit(position_gap, p.block) {
                front = Some(i);
                nearest = position_gap;
            }
            if !(position_gap.abs() >= p.block.side_distance) {
                if lane_gap < 0. {
                    side.in_space = side.in_space.min(-lane_gap);
                    if -lane_gap < p.block.side_lane {
                        side.can_in = false;
                        side.inner_blocker = Some(row.lane);
                        side.inner_blocker_index = Some(i);
                    }
                } else if lane_gap > 0. {
                    side.out_space = side.out_space.min(lane_gap);
                    if lane_gap < p.block.side_lane { side.can_out = false; }
                }
            }
            let inward_gap = owner.lane - row.lane;
            inner_nearby |= position_gap.abs() < p.block.side_distance
                && inward_gap > 0. && inward_gap < p.block.side_lane;
        }
        self.side = Some(side);
        self.front_cache = Some((owner, p.block.front_distance, p.block.front_lane, front));
        self.force_in = c.empty_in_move && c.phase < 1
            && owner.lane >= p.force_in_threshold && inner_nearby;
    }

    pub fn update_targets(
        &mut self,
        c: &TargetContext,
        live: &[Horse],
        dt: f32,
        p: TargetParameters,
    ) {
        assert_eq!(live.len(), self.cached.len());
        // Reuse the pre-movement query only while its owner and parameters
        // still match. Immediate skill callbacks may have changed the owner.
        let front = match self.front_cache {
            Some((owner, distance, lane, front))
                if owner.position.to_bits() == c.owner.point.position.to_bits()
                    && owner.lane.to_bits() == c.owner.point.lane.to_bits()
                    && distance.to_bits() == p.block.front_distance.to_bits()
                    && lane.to_bits() == p.block.front_lane.to_bits() => front,
            _ => front_index_iter(c.owner.point, self.cached.iter().map(|r| r.point), p.block),
        };
        self.skip_passing = false;
        let can_in = self.side.is_some_and(|side| side.can_in);
        let inside_opened = self.previous_can_in == Some(false) && can_in;
        self.previous_can_in = Some(can_in);
        let eligible = |i: usize| {
            let row = self.cached[i];
            crate::overtaking::is_target_with_block_check(
                (row.point.position - c.owner.point.position) as f64,
                (c.owner.speed - row.speed) as f64,
                c.owner.target as f64,
                live[i].target as f64,
                live[i].speed as f64,
                front == Some(i),
                || has_front_blocker(row.point, self.cached.iter().map(|r| r.point)
                    .chain(std::iter::once(c.owner.point)), p.block),
            )
        };
        self.targets.retain(|&i| i < self.cached.len());
        let existing_count = self.targets.len();
        for &i in &self.visible {
            if !self.targets.contains(&i) && eligible(i) {
                self.targets.push(i);
            }
        }
        // New entries already passed the identical immutable eligibility
        // check above. Revalidate only entries retained from the prior tick.
        let mut ordinal = 0;
        self.targets.retain(|&i| {
            ordinal += 1;
            ordinal > existing_count || eligible(i)
        });
        if let Some(i) = front {
            if !self.targets.contains(&i) {
                self.targets.push(i);
            }
        }
        if !self.targets.is_empty() {
            self.cooldown = p.cooldown;
        } else if self.cooldown > 0. {
            self.cooldown -= dt;
            if self.cooldown <= 0. {
                self.force_check = true;
            }
        }
        // A newly opened inward gap can request one normal lane search while
        // retaining the passing cooldown for subsequent updates.
        if cfg!(feature = "experimental-gap-refresh") && inside_opened && self.targets.is_empty()
            && c.previous_target == c.owner.point.lane && c.phase == 0
            && self.side.is_some_and(|side| !side.can_out)
            && !c.pace_down && c.neutral_pace {
            self.skip_passing = true;
            self.force_check = true;
        }
    }

}
pub(crate) fn front_index_iter(owner: Row, rows: impl Iterator<Item = Row>, p: BlockParameters) -> Option<usize> {
    let mut found = None;
    let mut nearest = f32::INFINITY;
    for (i, row) in rows.enumerate() {
        let gap = row.position - owner.position;
        if gap > 0.
            && gap < p.front_distance
            && gap < nearest
            && (row.lane - owner.lane).abs()
                <= front_lane_limit(gap, p)
        {
            found = Some(i);
            nearest = gap;
        }
    }
    found
}

// Overtaking only needs to know whether a peer is blocked. Stop at the
// first qualifying blocker; nearest-distance tie order is irrelevant here.
fn has_front_blocker(owner: Row, mut rows: impl Iterator<Item = Row>, p: BlockParameters) -> bool {
    rows.any(|row| {
        let gap = row.position - owner.position;
        gap > 0. && gap < p.front_distance
            && (row.lane - owner.lane).abs()
                <= front_lane_limit(gap, p)
    })
}

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
