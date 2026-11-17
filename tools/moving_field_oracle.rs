// M3b restricted integration adapter. M3a longitudinal body retained; original MoveState inserted after gate.
// Source lane primitives come from exact accepted oracle modules; overlap is mechanically copied.
// Original Rust repository is read-only; wrapper first-finish/timeouts are explicit adaptations.
// Destination-only M3a fixed-lane extraction. Reference modules are pinned
// read-only source; this adapter is a bounded no-skills/no-pace profile.
#[path = "C:/Users/a/RustroverProjects/GlobalSim/src/replay_frames.rs"] mod replay_frames;
#[path = "C:/Users/a/RustroverProjects/GlobalSim/src/json.rs"] mod json;
#[path = "C:/Users/a/RustroverProjects/GlobalSim/src/retail_physics.rs"] mod retail_physics;
use retail_physics::{Parameters, TargetSpeed, Acceleration, advance_start_delay, finish_time, phase_boundaries};
const DT:f32=0.066600002348423;
#[derive(Clone,Copy)] struct R{time:f32,pos:f32,lane:f32,speed:f32,hp:f32,delay:f32,phase:usize,dash:bool,finished:bool,finish:f32,min:f32,front:i32,gap:f32,side:bool,rank:i32,ft:f64,st:f64,at:f64}
fn limit(g:f32)->f32{let n=(g/2.)*0.6_f32;((1_f64-n as f64)*0.0416666716337204_f64)as f32}
fn snapshot(a:&mut[R;3]){for o in 0..3{let mut best=f32::INFINITY;let mut who=-1;let mut rank=1;let mut side=false;for i in 0..3{if a[i].pos>a[o].pos||(a[i].pos==a[o].pos&&i<o){rank+=1}if i==o{continue}let g=a[i].pos-a[o].pos;let lane=(a[i].lane-a[o].lane).abs();if g>0.&&g<2.&&(g<best||(g==best&&(i as i32)<who))&&lane<=limit(g){best=g;who=i as i32}if !a[i].finished&&g.abs()<1.0499999523162842&&lane<0.1111110970377922&&lane!=0.{side=true}}a[o].front=who;a[o].gap=if who<0{f32::NAN}else{best};a[o].side=side;a[o].rank=rank}}
fn tick_one(p:&Parameters,a:&mut[R;3],i:usize,truncated:bool,e:&mut[E;3],sides:&[mq::SideRoom;3]){if truncated||a[i].finished||e[i].timeout||a[i].pos>=1200.{return}let mut min=if a[i].dash{3.}else{p.minimum_speed(900.,1200.)};if !a[i].dash&&a[i].front>=0{min=min.min(a[a[i].front as usize].min)}a[i].min=min;let(delay,gate)=advance_start_delay(a[i].delay,DT);a[i].delay=delay;let Some(frac)=gate else{a[i].time+=DT;return};
let mut movement=mv::MoveState {lane:a[i].lane,target:e[i].target,speed:e[i].speed,continue_time:e[i].time,continue_direction:e[i].direction};
let side=sides[i];
e[i].moved=movement.step(mv::MoveContext {after_goal:false,position:a[i].pos,power:1000.,phase:a[i].phase as i32,rank:a[i].rank as usize,first_move_point:375.,max_lane:1.,side:mv::SideRoom {can_in:side.can_in,can_out:side.can_out,in_space:side.in_space,out_space:side.out_space,inner_blocker:side.inner_blocker,inner_blocker_index:side.inner_blocker_index},modifiers:Vec::<f32>::new()},DT,mp());
a[i].lane=movement.lane;e[i].target=movement.target;e[i].speed=movement.speed;e[i].time=movement.continue_time;e[i].direction=movement.continue_direction;
let(_,tp)=retail_physics::target_speed_indices(a[i].pos,1200.);let base=p.base_target_speed(1000.,1,tp,1.,1200.);let target=p.target_speed(TargetSpeed{base,section:0.,pace:1.,force_in:0.,power:1000.,moving_lane_skill:false,slope:0.,downhill:false,hp_remaining:a[i].hp>0.,minimum:min},&[]);let acc=if a[i].hp<=0.{-1.2}else if a[i].speed>target{[-1.2,-0.8,-1.][a[i].phase]}else{p.acceleration(Acceleration{power:1000.,strategy:1,phase:a[i].phase,ground:1.,distance:1.,uphill:false,start_dash:a[i].dash},&[])};let mut s=a[i].speed+acc*DT;s=if a[i].speed<=target{s.min(target)}else{s.max(target)};let dash=p.start_dash_speed(1200.);if a[i].dash{s=s.min(dash)}if a[i].front>=0{s=s.min(p.blocking_speed_cap(min,a[a[i].front as usize].speed,a[i].gap))}s=s.max(min);a[i].speed=s;let drain=if a[i].dash{20.}else{p.hp_per_second(s,p.course_speed(1200.),1.,900.,a[i].phase,1.)};a[i].hp=(a[i].hp-drain*DT).max(0.);a[i].phase=phase_boundaries(1200.).iter().filter(|&&b|a[i].pos>=b).count().min(2);let old=a[i].pos;a[i].pos+=((s*DT)as f64*frac)as f32;a[i].time+=DT;if old<1200.&&a[i].pos>=1200.{a[i].finished=true;a[i].finish=finish_time(a[i].time,DT,old,a[i].pos,1200.)}if a[i].dash&&s>=dash{a[i].dash=false}}

fn runner(pos:f32,lane:f32,speed:f32,delay:f32)->R{R{time:0.,pos,lane,speed,hp:1000.,delay,phase:0,dash:false,finished:false,finish:f32::NAN,min:3.,front:-1,gap:f32::NAN,side:false,rank:0,ft:0.,st:0.,at:0.}}
pub fn resolve_overlaps(rows: &mut [Row], force_checks: &mut [bool]) {
    resolve_overlaps_with_order(rows, force_checks, &mut Vec::new());
}

pub fn resolve_overlaps_with_order(rows: &mut [Row], force_checks: &mut [bool], order: &mut Vec<usize>) {
    assert_eq!(rows.len(), force_checks.len());
    order.clear();
    order.extend(0..rows.len());
    comparison_sort(order, |&a, &b| {
        (rows[a].lane as i32).cmp(&(rows[b].lane as i32))
    });
    if let Some(&first) = order.first() {
        force_checks[first] = false;
    }
    for pair in order.windows(2) {
        let previous = rows[pair[0]];
        let current = &mut rows[pair[1]];
        if (current.position - previous.position).abs() < 0.4
            && (current.lane - previous.lane).abs() < (0.4_f64 / 18.) as f32
        {
            current.lane = previous.lane + (0.4_f64 / 18.) as f32;
            force_checks[pair[1]] = true;
        }
    }
}



#[path="lane_move_oracle.rs"] mod mv;
#[path="lane_query_oracle.rs"] mod mq;
#[path="C:/Users/a/RustroverProjects/GlobalSim/src/retail_sort.rs"] mod retail_sort;
use retail_sort::comparison_sort;
use mq::Row;
#[derive(Clone,Copy)] struct E {target:f32,speed:f32,time:f32,direction:i32,force:bool,moved:bool,ticks:u32,timeout:bool}
fn mp()->mv::MoveParameters {mv::MoveParameters {power_coefficient:0.001,speed_add:0.3,base_speed:0.02,outer_coefficient:0.05,acceleration:1.5}}
fn room(a:&[R;3],i:usize)->mq::SideRoom {
 let others:Vec<Row>=(0..3).filter(|&j|j!=i).map(|j|Row{position:a[j].pos,lane:a[j].lane}).collect();
 mq::side_room(Row{position:a[i].pos,lane:a[i].lane},1.,&others,mq::BlockParameters{front_distance:2.,front_lane:0.0416666716337204,side_distance:1.0499999523162842,side_lane:0.1111110970377922})
}
fn state(a:&[R;3],e:&[E;3],tr:bool) {
 print!("[");for i in 0..3 {if i>0 {print!(",")} let r=a[i];let x=e[i];
 print!("[{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}]",r.time.to_bits(),r.pos.to_bits(),r.speed.to_bits(),r.hp.to_bits(),r.delay.to_bits(),r.phase,r.dash as u8,r.finished as u8,r.finish.to_bits(),r.min.to_bits(),r.ft.to_bits(),r.st.to_bits(),r.at.to_bits(),r.lane.to_bits(),tr as u8,x.target.to_bits(),x.speed.to_bits(),x.time.to_bits(),x.direction,x.force as u8,x.moved as u8,x.ticks,x.timeout as u8);
 }print!("]");
}
fn observations(a:&[R;3],s:&[mq::SideRoom;3]) {
 print!("[");for i in 0..3 {if i>0{print!(",")}let (_,g)=advance_start_delay(a[i].delay,DT);print!("[{},{},{},{},{},{},{},{}]",a[i].front,a[i].gap.to_bits(),a[i].side as u8,a[i].rank,s[i].can_in as u8,s[i].can_out as u8,g.is_some() as u8,g.unwrap_or(f64::NAN).to_bits())}print!("]");
}

fn counter_move(a:&[R;3],e:&[E;3],i:usize,side:mq::SideRoom,rank:usize)->u32 {
 let mut m=mv::MoveState {lane:a[i].lane,target:e[i].target,speed:e[i].speed,continue_time:e[i].time,continue_direction:e[i].direction};
 m.step(mv::MoveContext {after_goal:false,position:a[i].pos,power:1000.,phase:a[i].phase as i32,rank,first_move_point:375.,max_lane:1.,side:mv::SideRoom {can_in:side.can_in,can_out:side.can_out,in_space:side.in_space,out_space:side.out_space,inner_blocker:side.inner_blocker,inner_blocker_index:side.inner_blocker_index},modifiers:Vec::<f32>::new()},DT,mp());m.lane.to_bits()
}

fn case(p:&Parameters,name:&str,mut a:[R;3],mut e:[E;3],cap:u32) {
 print!("{{\"name\":\"{}\",\"max_ticks\":{},\"initial\":",name,cap);state(&a,&e,false);print!(",\"rows\":[");let mut tr=false;
 for t in 0..8 {if t>0{print!(",")} if !tr {snapshot(&mut a)}let sides=std::array::from_fn(|i|room(&a,i));
 let entry_positions=a.map(|r|r.pos);let mut witness:Option<(&str,u32,u32)>=None;
 print!("{{\"observation\":");observations(&a,&sides);
 if !tr {
  for i in 0..3 {let f=a[i].front>=0;a[i].ft=if f{a[i].ft+DT as f64}else{0.};a[i].st=if a[i].side{a[i].st+DT as f64}else{0.};a[i].at=if f&&a[i].side{a[i].at+DT as f64}else{0.};}
  for i in 0..3 {
 if t==0 && i==1 && name=="cached_side" {witness=Some(("cached_side",counter_move(&a,&e,i,sides[i],a[i].rank as usize),counter_move(&a,&e,i,room(&a,i),a[i].rank as usize)));}
 if t==0 && i==1 && name=="cached_rank" {let live_rank=1+(0..3).filter(|&j|a[j].pos>a[i].pos||(a[j].pos==a[i].pos&&j<i)).count();witness=Some(("cached_rank",counter_move(&a,&e,i,sides[i],a[i].rank as usize),counter_move(&a,&e,i,sides[i],live_rank)));}
 e[i].moved=false;if !a[i].finished && !e[i].timeout && a[i].pos<1200. {e[i].ticks+=1;tick_one(p,&mut a,i,tr,&mut e,&sides);e[i].timeout=!a[i].finished && e[i].ticks>=cap;}}
  let wrong_post=if t==0 && name=="post_position_overlap" {let mut wrong:Vec<Row>=(0..3).map(|i|Row{position:entry_positions[i],lane:a[i].lane}).collect();let mut force:Vec<bool>=e.iter().map(|x|x.force).collect();resolve_overlaps(&mut wrong,&mut force);Some(wrong[1].lane.to_bits())}else{None};
 let mut rows:Vec<Row>=a.iter().map(|r|Row{position:r.pos,lane:r.lane}).collect();let mut flags:Vec<bool>=e.iter().map(|x|x.force).collect();resolve_overlaps(&mut rows,&mut flags);
  for i in 0..3 {a[i].lane=rows[i].lane;e[i].force=flags[i];}
 if let Some(wrong)=wrong_post {witness=Some(("post_position_overlap",a[1].lane.to_bits(),wrong));}
  tr=a.iter().any(|r|r.finished)||e.iter().all(|x|x.timeout);
 }
 print!(",\"state\":");state(&a,&e,tr);
 match witness {Some((kind,right,wrong))=>print!(",\"witness\":{{\"kind\":\"{}\",\"correct_lane_bits\":{},\"wrong_lane_bits\":{}}}",kind,right,wrong),None=>print!(",\"witness\":null")};print!("}}");
 }print!("]}}");
}
fn main() {
 let root=json::parse(&std::fs::read_to_string("C:/Users/a/PycharmProjects/GlobalSim-Jax/globalsim_jax/global_parameters.json").unwrap()).unwrap();let p=Parameters::from_value(root.get("parameters").unwrap()).unwrap();
 let make=|target:f32,speed:f32|E{target,speed,time:0.25,direction:1,force:true,moved:false,ticks:0,timeout:false};
 print!("{{\"revision\":\"58da7bc4693d9aa8f6340e348cba1c5b5fb5a856\",\"scope\":\"restricted supplied-target schedule extraction; no engine/target-policy/GPU claim\",\"dt_bits\":{},\"columns\":[\"time\",\"position\",\"speed\",\"hp\",\"delay\",\"phase\",\"dash\",\"finished\",\"finish_time\",\"minimum\",\"front_time_f64\",\"side_time_f64\",\"all_time_f64\",\"lane\",\"truncated\",\"target\",\"lane_speed\",\"continue_time\",\"direction\",\"force\",\"moved\",\"ticks\",\"timeout\"],\"cases\":[",DT.to_bits());
 case(&p,"cached_side",[runner(20.,0.2,13.,0.),runner(20.,0.1,13.,0.),runner(60.,1.,13.,0.)],[make(0.6,0.6),make(0.6,0.6),make(1.,0.)],100);print!(",");
 case(&p,"post_position_overlap",[runner(11.1,0.2,13.,0.),runner(10.,0.26,30.,0.),runner(60.,1.,13.,0.)],[make(0.2,0.),make(0.2,0.6),make(1.,0.)],100);print!(",");
 let mut rank=[runner(900.,0.1,30.,0.),runner(900.5,0.3,13.,0.),runner(60.,1.,13.,0.)];for r in rank.iter_mut().take(2){r.phase=2;}
 case(&p,"cached_rank",rank,[make(0.6,0.026),make(0.6,0.026),make(1.,0.)],100);print!(",");
 case(&p,"chained",[runner(20.,0.,13.,0.),runner(20.,0.001,13.,0.),runner(20.,0.002,13.,0.)],[make(0.,0.),make(0.001,0.),make(0.002,0.)],100);print!(",");
 case(&p,"closed_gate",[runner(20.,0.,13.,1.),runner(20.,0.001,13.,1.),runner(20.,0.002,13.,1.)],[make(1.3,0.3),make(-0.1,0.3),make(0.5,0.3)],100);print!(",");
 case(&p,"partial_gate",[runner(20.,0.1,13.,0.02),runner(60.,0.5,13.,0.),runner(100.,1.,13.,0.)],[make(0.6,0.6),make(0.5,0.),make(1.,0.)],100);print!(",");
 case(&p,"first_finish",[runner(1199.5,0.2,13.,0.),runner(1100.,1.,13.,0.),runner(100.,0.5,13.,0.)],[make(0.6,0.6),make(1.,0.),make(0.6,0.6)],100);print!(",");
 let mut timeout=[runner(10.,0.,13.,1.),runner(11.,0.02,13.,1.),runner(12.,0.04,13.,1.)];for r in &mut timeout {r.ft=0.125;r.st=0.125;r.at=0.125;}
 case(&p,"all_timeout",timeout,[make(1.3,0.3),make(-0.1,0.3),make(0.5,0.3)],3);print!(",");
 case(&p,"live_blocker_cap",[runner(11.,0.02,28.,0.),runner(10.,0.,30.,0.02),runner(30.,1.,20.,0.)],[make(0.02,0.),make(0.,0.),make(1.,0.)],100);println!("]}}");
}
