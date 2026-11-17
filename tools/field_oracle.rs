// Destination-only M3a query oracle. Reference revision is documented in
// docs/ORACLE_PROVENANCE.json; read-only source is never modified.
fn limit(gap:f32)->f32 { let narrowing=(gap/2.0)*0.6_f32; ((1.0_f64-narrowing as f64)*0.0416666716337204_f64) as f32 }
fn sample(rows:&[(f32,f32,bool)]) -> (Vec<i32>,Vec<i32>,Vec<u32>,Vec<u8>,Vec<i32>) {
 let n=rows.len(); let mut front=vec![-1;n]; let mut gaps=vec![f32::NAN;n]; let mut side=vec![0;n];
 for o in 0..n { let (p,l,_)=rows[o]; let mut nearest=f32::INFINITY; for i in 0..n {if i==o{continue} let (q,ol,done)=rows[i]; let g=q-p; let lg=ol-l;
   if g>0. && g<2. && (g<nearest || (g==nearest && i < front[o] as usize)) && lg.abs()<=limit(g) {front[o]=i as i32;gaps[o]=g;nearest=g;}
   if !done && g.abs()<1.0499999523162842 && lg.abs()<0.1111110970377922 && lg!=0. {side[o]=1;}
 }}
 let mut rank=vec![0;n]; for o in 0..n {rank[o]=1+(0..n).filter(|&i|rows[i].0>rows[o].0 || (rows[i].0==rows[o].0 && i<o)).count() as i32;}
 (front,rank,gaps.iter().map(|x|x.to_bits()).collect(),side,vec![])
}
fn main(){let cases=[("ties",vec![(10.,0.,false),(11.,0.,true),(11.,0.,false),(9.5,0.05,false)]),("bounds",vec![(0.,0.,false),(2.,0.,false),(1.,0.02916667,false),(1.,0.,false)])]; let raw=0.123456789_f64;let lane=raw as f32;let gap=0.01_f32;let narrowing=(gap/2.0)*0.6_f32;let width=((1.0_f64-narrowing as f64)*lane as f64)as f32; print!("{{\"revision\":\"58da7bc4693d9aa8f6340e348cba1c5b5fb5a856\",\"cases\":["); for (k,(name,rows)) in cases.iter().enumerate(){if k>0{print!(",")}let(f,r,g,s,_)=sample(rows);print!("{{\"name\":\"{}\",\"front\":{:?},\"rank\":{:?},\"gap_bits\":{:?},\"side\":{:?}}}",name,f,r,g,s);}println!("],\"custom_vectors\":[{{\"front_lane_raw\":{},\"front_lane_f32_bits\":{},\"gap_bits\":{},\"expected_bits\":{}}}]}}",raw,lane.to_bits(),gap.to_bits(),width.to_bits());}
