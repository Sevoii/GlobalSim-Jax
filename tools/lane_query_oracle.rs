// mechanical query extraction from pinned retail_lane.rs
#[derive(Clone, Copy, Debug)]
pub struct Row { pub position:f32, pub lane:f32 }
#[derive(Clone, Copy, Debug)]
pub struct BlockParameters { pub front_distance:f32,pub front_lane:f32,pub side_distance:f32,pub side_lane:f32 }
#[derive(Clone, Copy, Debug)]
pub struct SideRoom {
    pub can_in: bool,
    pub can_out: bool,
    pub in_space: f32,
    pub out_space: f32,
    pub inner_blocker: Option<f32>,
    pub inner_blocker_index: Option<usize>,
}

impl SideRoom {
    pub fn reachable(&self, lane: f32, candidate: f32) -> bool {
        let delta = candidate - lane;
        delta == 0.
            || if delta > 0. {
                self.can_out && delta <= self.out_space
            } else {
                self.can_in && -delta <= self.in_space
            }
    }
}

pub fn is_visible(gap: f32, lane_gap: f32, vision: f32) -> bool {
    if vision <= 0. || !(0. ..=vision).contains(&gap) {
        return false;
    }
    // Store the distance ratio, then keep the width expression wide until
    // its final float32 store. Intermediate float rounding can admit a
    // runner just outside the cone and close an otherwise clear corridor.
    // The width grows with the ratio, which is at most one here. Rows wider
    // than the full-distance width cannot qualify; skip their division.
    let widest = ((1_f64 * 0.6388888955116272_f32 as f64 + 0.1111111119389534_f32 as f64) * 0.5) as f32;
    if lane_gap.abs() > widest { return false; }
    let ratio = gap / vision;
    let width = ((ratio as f64 * 0.6388888955116272_f32 as f64
        + 0.1111111119389534_f32 as f64) * 0.5) as f32;
    lane_gap.abs() <= width
}

pub fn front_index(owner: Row, rows: &[Row], p: BlockParameters) -> Option<usize> {
    front_index_iter(owner, rows.iter().copied(), p)
}

// Store the narrowing coefficient before interpolating the lane limit.
// This operation order distinguishes captured edges on both sides of the cone.
#[inline]
pub(crate) fn front_lane_limit(gap: f32, p: BlockParameters) -> f32 {
    let narrowing = (gap / p.front_distance) * 0.6_f32;
    ((1.0 - narrowing as f64) * p.front_lane as f64) as f32
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

pub fn side_room(owner: Row, max_lane: f32, rows: &[Row], p: BlockParameters) -> SideRoom {
    side_room_iter(owner, max_lane, rows.iter().copied(), p)
}

fn side_room_iter(owner: Row, max_lane: f32, rows: impl Iterator<Item = Row>, p: BlockParameters) -> SideRoom {
    let mut result = SideRoom {
        can_in: owner.lane > 0.0001,
        can_out: owner.lane < max_lane,
        in_space: owner.lane,
        out_space: max_lane - owner.lane,
        inner_blocker: None,
        inner_blocker_index: None,
    };
    for (index, row) in rows.enumerate() {
        if (row.position - owner.position).abs() >= p.side_distance {
            continue;
        }
        let gap = row.lane - owner.lane;
        if gap < 0. {
            result.in_space = result.in_space.min(-gap);
            if -gap < p.side_lane {
                result.can_in = false;
                result.inner_blocker = Some(row.lane);
                result.inner_blocker_index = Some(index);
            }
        } else if gap > 0. {
            result.out_space = result.out_space.min(gap);
            if gap < p.side_lane {
                result.can_out = false;
            }
        }
    }
    result
}

#[path = "C:/Users/a/RustroverProjects/GlobalSim/src/overtaking.rs"] mod overtaking;
fn side(name:&str, owner:Row, max:f32, rows:&[Row], p:BlockParameters){let s=side_room(owner,max,rows,p);print!("{{\"name\":\"{}\",\"owner\":[{},{}],\"max\":{},\"rows\":[",name,owner.position.to_bits(),owner.lane.to_bits(),max.to_bits());for(i,r)in rows.iter().enumerate(){if i>0{print!(",")}print!("[{},{}]",r.position.to_bits(),r.lane.to_bits())}println!("],\"out\":[{},{},{},{},{},{}]}}",s.can_in as u8,s.can_out as u8,s.in_space.to_bits(),s.out_space.to_bits(),s.inner_blocker.unwrap_or(f32::NAN).to_bits(),s.inner_blocker_index.map(|x|x as i32).unwrap_or(-1));}
fn main(){let p=BlockParameters{front_distance:2.,front_lane:0.0416666716337204,side_distance:1.0499999523162842,side_lane:0.1111110970377922};let o=Row{position:0.,lane:0.};let d=p.side_distance;let dl=p.side_lane;let dn=f32::from_bits(d.to_bits()-1);let dp=f32::from_bits(d.to_bits()+1);let ln=f32::from_bits(dl.to_bits()-1);let lp=f32::from_bits(dl.to_bits()+1);let full:Vec<Row>=(0..32).map(|i|Row{position:i as f32*0.01,lane:if i%2==0{-0.2}else{0.2}}).collect();print!("{{\"side\":[");let cases:[(&str,Row,f32,&[Row]);20]=[("empty",o,1.,&[]),("full32",o,1.,&full),("rail_exact",Row{position:0.,lane:0.0001},1.,&[]),("rail_next",Row{position:0.,lane:f32::from_bits(0.0001f32.to_bits()+1)},1.,&[]),("outside",Row{position:0.,lane:1.2},1.,&[]),("dist_exact",o,1.,&[Row{position:d,lane:-0.01}]),("dist_nextin",o,1.,&[Row{position:dn,lane:-0.01}]),("dist_nextout",o,1.,&[Row{position:dp,lane:-0.01}]),("lane_exact",o,1.,&[Row{position:0.,lane:-dl}]),("lane_nextin",o,1.,&[Row{position:0.,lane:-ln}]),("lane_nextout",o,1.,&[Row{position:0.,lane:-lp}]),("last_reordered",o,1.,&[Row{position:0.,lane:-0.05},Row{position:0.,lane:-0.08},Row{position:0.,lane:0.}]),("tighten_no_block",Row{position:0.,lane:0.2},1.,&[Row{position:0.,lane:-0.2}]),("last_blocker_reversed",o,1.,&[Row{position:0.,lane:-0.08},Row{position:0.,lane:-0.05}]),("inner_true_last",Row{position:0.,lane:0.2},1.,&[Row{position:0.,lane:0.15},Row{position:0.,lane:0.12}]),("inner_true_reversed",Row{position:0.,lane:0.2},1.,&[Row{position:0.,lane:0.12},Row{position:0.,lane:0.15}]),("outer_block",o,1.,&[Row{position:0.,lane:0.01}]),("outer_exact",o,1.,&[Row{position:0.,lane:dl}]),("outer_nextin",o,1.,&[Row{position:0.,lane:ln}]),("owner_max",Row{position:0.,lane:1.0},1.,&[])];for(i,(n,a,m,r))in cases.iter().enumerate(){if i>0{print!(",")}side(n,*a,*m,r,p)}print!("],\"visible\":[");let gap=1193.378173828125_f32-1191.0203857421875_f32;let outside=0.2024788111448288_f32-0.10926413536071777_f32;let lower=f32::from_bits(outside.to_bits()-1);let vs:[(f32,f32,f32);12]=[(0.,0.,20.),(20.,0.,20.),(-1.,0.,20.),(0.,0.,0.),(gap,outside,20.),(gap,lower,20.),(gap,-lower,20.),(gap,-outside,20.),(10.,0.,20.),(10.,1.,20.),(20.,0.375,20.),(20.,-0.375,20.)];for(i,(g,l,v))in vs.iter().enumerate(){if i>0{print!(",")}print!("{{\"in\":[{},{},{}],\"out\":{}}}",g.to_bits(),l.to_bits(),v.to_bits(),is_visible(*g,*l,*v)as u8)}print!("],\"overtake\":[");let os:[(f64,f64,f64,f64,f64,bool,bool);26]=[(-1.0,1.,22.,24.,20.,false,false),(-10.0001,1.,22.,24.,20.,false,true),(1.,1.,22.,24.,20.,false,false),(20.,1.,22.,24.,20.,false,true),(200.0001,1.,22.,24.,20.,false,true),(15.,1.,22.,24.,20.,false,true),(150.0001,1.,22.,24.,20.,false,true),(10.,1.,22.,24.,20.,true,false),(10.,1.,22.,24.,20.,true,true),(10.,1.,22.,24.,20.,false,false),(10.,1.,22.,24.,20.,false,true),(10.,1.,22.,20.,24.,false,false),(10.,1.,22.,20.,24.,false,true),(10.,0.,22.,24.,20.,false,true),(10.,-1.,22.,24.,20.,false,true),(1.00000001,1.,22.,24.,20.,false,true),(20.00000001,1.,22.,24.,20.,false,true),(10.,1.,22.,24.,20.,false,false),(10.,1.,22.,24.,20.,false,true),(10.,1.,22.,24.,20.,true,false),(20.00000001,2.,22.,24.,20.,false,true),(0.99999999,1.,22.,24.,20.,false,true),(-1.00000001,1.,22.,24.,20.,true,true),(10.,1.,22.00000001,24.,22.,true,false),(10.,1.,22.00000002,22.00000001,24.,false,false),(15.,0.99999999,22.,24.,20.,false,true)];for(i,x)in os.iter().enumerate(){if i>0{print!(",")}print!("{{\"in\":[{},{},{},{},{},{},{}],\"out\":{}}}",x.0.to_bits(),x.1.to_bits(),x.2.to_bits(),x.3.to_bits(),x.4.to_bits(),x.5 as u8,x.6 as u8,overtaking::is_target(x.0,x.1,x.2,x.3,x.4,x.5,x.6)as u8)}println!("],\"unsupported\":[{{\"in\":[0,0,{}],\"visible\":{}}},{{\"in\":[{},{}],\"direct\":{}}}]}}", (1e-40f32).to_bits(), is_visible(0.,0.,1e-40) as u8, (2e-40f32).to_bits(), (1e-40f32).to_bits(), overtaking::is_target(1.,1.,2e-40,5.,1e-40,true,false) as u8)}
