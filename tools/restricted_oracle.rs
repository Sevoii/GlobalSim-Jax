// Destination-only restricted oracle. Pinned source remains read-only.
#[path = "C:/Users/a/RustroverProjects/GlobalSim/src/replay_frames.rs"] mod replay_frames;
#[path = "C:/Users/a/RustroverProjects/GlobalSim/src/json.rs"] mod json;
#[path = "C:/Users/a/RustroverProjects/GlobalSim/src/retail_physics.rs"] mod retail_physics;

use retail_physics::{Parameters, TargetSpeed, Acceleration, advance_start_delay, finish_time, phase_boundaries};

#[derive(Clone, Copy)] struct S { time:f32,pos:f32,speed:f32,hp:f32,delay:f32,phase:usize,dash:bool,finished:bool,finish:f32 }
fn tick(p:&Parameters, mut s:S, speed:f32, guts:f32, power:f32, strategy:usize, aptitude:f32, ground:f32, distance:f32, dt:f32)->S {
    // M2 wrapper adopts the field scheduling mask: position == goal is no longer stepped.
    if s.finished || s.pos >= distance { return s; }
    let (delay, gate)=advance_start_delay(s.delay,dt); s.delay=delay;
    let Some(fraction)=gate else { s.time+=dt; return s };
    let (_, target_phase)=retail_physics::target_speed_indices(s.pos,distance);
    let base=p.base_target_speed(speed,strategy,target_phase,aptitude,distance);
    let ordinary=p.minimum_speed(guts,distance); let minimum=if s.dash {3.} else {ordinary};
    let target=p.target_speed(TargetSpeed{base,section:0.,pace:1.,force_in:0.,power,moving_lane_skill:false,slope:0.,downhill:false,hp_remaining:s.hp>0.,minimum},&[]);
    let accel=if s.hp<=0. {-1.2} else if s.speed>target {[-1.2,-0.8,-1.0][s.phase]} else {p.acceleration(Acceleration{power,strategy,phase:s.phase,ground,distance:aptitude,uphill:false,start_dash:s.dash},&[])};
    let mut next=s.speed+accel*dt; next=if s.speed<=target {next.min(target)} else {next.max(target)};
    let dash_limit=p.start_dash_speed(distance); if s.dash {next=next.min(dash_limit)}; next=next.max(minimum); s.speed=next;
    let drain=if s.dash {20.*ground} else {p.hp_per_second(next,p.course_speed(distance),ground,guts,s.phase,1.)}; s.hp=(s.hp-drain*dt).max(0.);
    s.phase=phase_boundaries(distance).iter().filter(|&&b|s.pos>=b).count().min(2);
    let moved=((next*dt) as f64*fraction) as f32; let previous=s.pos; s.pos+=moved; s.time+=dt;
    if previous < distance && s.pos>=distance {s.finished=true;s.finish=finish_time(s.time,dt,previous,s.pos,distance)}
    if s.dash && next>=dash_limit{s.dash=false}; s
}
fn main(){
 let root=json::parse(&std::fs::read_to_string("C:/Users/a/PycharmProjects/GlobalSim-Jax/globalsim_jax/global_parameters.json").unwrap()).unwrap(); let p=Parameters::from_value(root.get("parameters").unwrap()).unwrap();
 let dash_limit=p.start_dash_speed(1200.);
 let cases=[
 ("delayed",S{time:0.,pos:0.,speed:0.,hp:p.initial_hp(1000.,1,1200.),delay:0.04,phase:0,dash:true,finished:false,finish:f32::NAN},1000.,900.,1000.,1,1.,1.1,1200.),
 ("phase",S{time:0.,pos:199.99,speed:18.,hp:1000.,delay:0.,phase:0,dash:false,finished:false,finish:f32::NAN},1000.,900.,1000.,1,1.,1.,1200.),
 ("finish_overshoot",S{time:0.,pos:1199.8,speed:20.,hp:1000.,delay:0.,phase:2,dash:false,finished:false,finish:f32::NAN},1000.,900.,1000.,1,1.,1.,1200.),
 ("exact_goal_field_stop",S{time:0.,pos:1200.,speed:20.,hp:1000.,delay:0.,phase:2,dash:false,finished:false,finish:f32::NAN},1000.,900.,1000.,1,1.,1.,1200.),
 ("dash_exit",S{time:0.,pos:1.,speed:dash_limit,hp:1000.,delay:0.,phase:0,dash:true,finished:false,finish:f32::NAN},1000.,900.,1000.,1,1.,1.,1200.),
 ("hp_zero_decel",S{time:0.,pos:1.,speed:20.,hp:0.,delay:0.,phase:0,dash:false,finished:false,finish:f32::NAN},1000.,900.,1000.,1,1.,1.,1200.)
 ];
 print!("{{\"provenance\":\"pinned Rust restricted extraction retail_physics.rs\",\"dt\":0.066600002348423,\"cases\":[");
 for (ci,(name,mut s,sp,gu,pw,st,ap,gr,di)) in cases.iter().enumerate(){if ci>0{print!(",")} let initial=s; print!("{{\"name\":\"{}\",\"inputs\":[{},{},{},{},{},{},{}],\"initial\":[{},{},{},{},{},{},{},{}],\"rows\":[",name,sp,gu,pw,st,ap,gr,di,initial.time.to_bits(),initial.pos.to_bits(),initial.speed.to_bits(),initial.hp.to_bits(),initial.delay.to_bits(),initial.phase,initial.dash as u8,initial.finished as u8); for i in 0..8{if i>0{print!(",")} s=tick(&p,s,*sp,*gu,*pw,*st,*ap,*gr,*di,0.066600002348423); print!("[{}, {}, {}, {}, {}, {}, {}, {}, {}]",s.time.to_bits(),s.pos.to_bits(),s.speed.to_bits(),s.hp.to_bits(),s.delay.to_bits(),s.phase,s.dash as u8,s.finished as u8,s.finish.to_bits());} print!("]}}");} println!("]}}");
}
