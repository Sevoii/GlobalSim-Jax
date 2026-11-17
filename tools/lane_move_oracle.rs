// Destination-only mechanical extraction from pinned retail_lane.rs.
// Included: SideRoom struct; MoveParameters struct (loader omitted); MoveState through impl.
// Source revision: 58da7bc4693d9aa8f6340e348cba1c5b5fb5a856.
#[path = "C:/Users/a/RustroverProjects/GlobalSim/src/replay_frames.rs"] mod replay_frames;
#[path = "C:/Users/a/RustroverProjects/GlobalSim/src/json.rs"] mod json;
#[path = "C:/Users/a/RustroverProjects/GlobalSim/src/retail_physics.rs"] mod retail_physics;
#[derive(Clone, Copy, Debug)]
pub struct SideRoom {
    pub can_in: bool,
    pub can_out: bool,
    pub in_space: f32,
    pub out_space: f32,
    pub inner_blocker: Option<f32>,
    pub inner_blocker_index: Option<usize>,
}


#[derive(Clone, Copy, Debug)]
pub struct MoveParameters {
    pub power_coefficient: f32,
    pub speed_add: f32,
    pub base_speed: f32,
    pub outer_coefficient: f32,
    pub acceleration: f32,
}


#[derive(Clone, Copy, Debug)]
pub struct MoveState {
    pub lane: f32,
    pub target: f32,
    pub speed: f32,
    pub continue_time: f32,
    pub continue_direction: i32,
}

pub struct MoveContext<M> {
    /// Switch to the post-finish rule after the one-second grace period.
    pub after_goal: bool,
    pub position: f32,
    pub power: f32,
    pub phase: i32,
    pub rank: usize,
    pub first_move_point: f32,
    pub max_lane: f32,
    pub side: SideRoom,
    pub modifiers: M,
}

fn approximately(a: f32, b: f32) -> bool {
    (a - b).abs() < (1e-6_f32 * a.abs().max(b.abs())).max(1.1210387714598537e-44_f32)
}

impl MoveState {
    /// The separate field-overlap pass must run after every runner moves.
    /// Returns whether movement occurred; blocked moves may retain momentum.
    pub fn step(&mut self, context: MoveContext<impl IntoIterator<Item = f32>>, dt: f32, p: MoveParameters) -> bool {
        use crate::retail_physics::{apply_additive_modifiers, bounded_stat};
        self.target = self.target.clamp(0., context.max_lane);
        if approximately(self.target, self.lane) {
            self.continue_time = 0.;
            return false;
        }
        let previous_direction = self.continue_direction;
        self.continue_direction = (self.target >= self.lane) as i32;
        let speed = if context.after_goal {
            // Finished runners use signed dt as their lateral speed. The
            // displacement below therefore uses dt twice, with no normal
            // acceleration, skill modifiers or inward lane multiplier.
            if self.target < self.lane { -dt } else { dt }
        } else {
            // v20: lateral movement follows the backend's x87-style evaluation:
            // the base rate local is computed at double and rounded once; the
            // factor is an unrounded double temporary of `rate * factor`, whose
            // product is rounded once. Inward speed and the lane update are
            // likewise double expressions rounded on the store (full corpus:
            // exact distance samples +135k, never-drifting runners +2079).
            let rate = ((bounded_stat(context.power) as f64 * p.power_coefficient as f64
                + p.speed_add as f64)
                * p.base_speed as f64) as f32;
            let factor = if context.phase >= 2 {
                // v10: official late-race lateral speed uses the 0-based order (leader x1.00).
                1. + (context.rank as f64 - 1.) * 0.01f32 as f64
            } else if context.position <= context.first_move_point {
                // v69: the lane ratio is a float quotient (doc 13.186).
                1. + (self.lane / context.max_lane) as f64 * p.outer_coefficient as f64
            } else {
                1.
            };
            let rate = (rate as f64 * factor) as f32;
            // v69: the lane-change acceleration is an unrounded double temporary
            // added to the stored lateral speed (doc 13.186).
            let acceleration = (p.acceleration as f64 * p.base_speed as f64) * dt as f64;
            self.speed = if self.speed < rate {
                ((self.speed as f64 + acceleration) as f32).min(rate)
            } else {
                ((self.speed as f64 - acceleration) as f32).max(rate)
            }
            .max(0.);
            let mut speed =
                apply_additive_modifiers(self.speed, context.modifiers).clamp(0., 0.6);
            if approximately(speed, 0.) {
                return false;
            }
            if self.target < self.lane {
                speed = (-(speed as f64) * (1. + self.lane as f64)) as f32;
            }
            speed
        };
        let mut next = (self.lane as f64 + dt as f64 * speed as f64) as f32;
        next = if self.target < self.lane {
            next.max(self.target)
        } else {
            next.min(self.target)
        };
        next = next.clamp(0., context.max_lane);
        if (next < self.lane && !context.side.can_in) || (next > self.lane && !context.side.can_out)
        {
            return false;
        }
        let moved = next != self.lane;
        self.lane = next;
        if previous_direction == self.continue_direction {
            self.continue_time += dt;
        }
        if approximately(next, self.target) {
            self.speed = 0.;
        }
        moved
    }
}



fn emit(name: &str, mut state: MoveState, after_goal: bool, position: f32, power: f32, phase: i32, rank: usize, first: f32, max_lane: f32, dt: f32, can_in: bool, can_out: bool, modifiers: Vec<f32>, p: MoveParameters) {
    let initial = state;
    let moved = state.step(MoveContext { after_goal, position, power, phase, rank, first_move_point: first, max_lane, side: SideRoom { can_in, can_out, in_space: 0.0, out_space: 0.0, inner_blocker: None, inner_blocker_index: None }, modifiers: modifiers.clone() }, dt, p);
    print!("{{\"name\":\"{}\",\"input\":[{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},[{}]],\"out\":[{},{},{},{},{},{}]}}", name, initial.lane.to_bits(), initial.target.to_bits(), initial.speed.to_bits(), initial.continue_time.to_bits(), initial.continue_direction, after_goal as u8, position.to_bits(), power.to_bits(), phase, rank, first.to_bits(), max_lane.to_bits(), dt.to_bits(), can_in as u8, can_out as u8, modifiers.iter().map(|x| x.to_bits().to_string()).collect::<Vec<_>>().join(","), state.lane.to_bits(), state.target.to_bits(), state.speed.to_bits(), state.continue_time.to_bits(), state.continue_direction, moved as u8);
}
fn main() {
    let p = MoveParameters { power_coefficient: 0.0010000000474974513, speed_add: 0.30000001192092896, base_speed: 0.019999999552965164, outer_coefficient: 0.05000000074505806, acceleration: 1.5 };
    print!("{{\"provenance\":\"pinned retail_lane MoveState exact extraction\",\"rows\":[");
    let cases = [
        ("blocked_out", MoveState { lane: 0.1, target: 0.5, speed: 0.0, continue_time: 0.2, continue_direction: 1 }, false, 100.0, 900.0, 1, 2usize, 375.0, 1.0, 0.0666, true, false),
        ("blocked_in", MoveState { lane: 0.5, target: 0.1, speed: 0.3, continue_time: 0.2, continue_direction: 0 }, false, 100.0, 900.0, 1, 2usize, 375.0, 1.0, 0.0666, false, true),
        ("reversal", MoveState { lane: 0.5, target: 0.1, speed: 0.3, continue_time: 0.2, continue_direction: 1 }, false, 100.0, 900.0, 1, 2usize, 375.0, 1.0, 0.0666, true, true),
        ("blocked_reversal", MoveState { lane: 0.5, target: 0.1, speed: 0.3, continue_time: 0.2, continue_direction: 1 }, false, 100.0, 900.0, 1, 2usize, 375.0, 1.0, 0.0666, false, true),
        ("continuation", MoveState { lane: 0.1, target: 0.5, speed: 0.0, continue_time: 0.2, continue_direction: 1 }, false, 100.0, 900.0, 1, 2usize, 375.0, 1.0, 0.0666, true, true),
        ("target_equal", MoveState { lane: 0.3, target: 0.3, speed: 0.2, continue_time: 0.2, continue_direction: 1 }, false, 100.0, 900.0, 1, 2usize, 375.0, 1.0, 0.0666, true, true),
        ("target_equal_zero", MoveState { lane: 0.0, target: 0.0, speed: 0.2, continue_time: 0.2, continue_direction: 1 }, false, 100.0, 900.0, 1, 2usize, 375.0, 1.0, 0.0666, true, true),
        ("modified_zero", MoveState { lane: 0.5, target: 0.1, speed: 0.0, continue_time: 0.2, continue_direction: 1 }, false, 100.0, 900.0, 1, 2usize, 375.0, 1.0, 0.0666, true, true),
        ("dt_zero", MoveState { lane: 0.1, target: 0.5, speed: 0.0, continue_time: 0.2, continue_direction: 1 }, false, 100.0, 900.0, 1, 2usize, 375.0, 1.0, 0.0, true, true),
        ("after_out", MoveState { lane: 0.1, target: 0.5, speed: 0.3, continue_time: 0.2, continue_direction: 1 }, true, 100.0, 900.0, 1, 2usize, 375.0, 1.0, 0.0666, true, true),
        ("after_in", MoveState { lane: 0.5, target: 0.1, speed: 0.3, continue_time: 0.2, continue_direction: 1 }, true, 100.0, 900.0, 1, 2usize, 375.0, 1.0, 0.0666, true, true),
        ("after_blocked", MoveState { lane: 0.1, target: 0.5, speed: 0.3, continue_time: 0.2, continue_direction: 1 }, true, 100.0, 900.0, 1, 2usize, 375.0, 1.0, 0.0666, true, false),
        ("late_rank", MoveState { lane: 0.1, target: 0.5, speed: 0.0, continue_time: 0.2, continue_direction: 1 }, false, 400.0, 900.0, 3, 120usize, 375.0, 1.5, 0.0666, true, true),
    ];
    for (i, (name, state, after, position, power, phase, rank, first, max_lane, dt, can_in, can_out)) in cases.into_iter().enumerate() {
        if i > 0 { print!(","); }
        let modifiers = if name == "modified_zero" { vec![-1.0] } else if name == "continuation" { vec![0.1] } else { Vec::new() };
        emit(name, state, after, position, power, phase, rank, first, max_lane, dt, can_in, can_out, modifiers, p);
    }
    println!("]}}");
}
