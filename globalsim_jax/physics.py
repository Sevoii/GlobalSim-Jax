"""Narrow stateless retail physics port from Rust `retail_physics.rs`."""
from dataclasses import dataclass
import json
from importlib.resources import files
import jax
import jax.numpy as jnp
from .precision import require_x64

f32 = lambda x: jnp.asarray(x, jnp.float32)


def bounded_stat(value):
    return jnp.clip(f32(value), 1., 2000.)


def _root(value):
    return jnp.sqrt(f32(value).astype(jnp.float64)).astype(jnp.float32)
def phase_boundaries(distance):
    distance = f32(distance)
    section = distance / f32(24.)
    return jnp.stack((section * 4, section * 16, section * 20, distance))
def target_speed_indices(position, distance):
    quotient = f32(position).astype(jnp.float64) / (f32(distance) / 24).astype(jnp.float64)
    section = jnp.clip(quotient.astype(jnp.int32), 0, 23)
    return section, jnp.where(section < 4, 0, jnp.where(section < 16, 1, 2))
def advance_start_delay(remaining, dt):
    remaining, dt = f32(remaining), f32(dt)
    after = jnp.where(remaining > 0, remaining - dt, remaining)
    active = (dt != 0) & (after <= 0)
    next_remaining = jnp.where(dt == 0, remaining, jnp.where(active, 0., after))
    # `active=False` is the Option::None equivalent; use finite zero payload.
    fraction = jnp.where(active, jnp.where(after < 0, -after.astype(jnp.float64) / dt.astype(jnp.float64), 1.), 0.)
    return next_remaining, active, fraction
def finish_time(frame_time, dt, previous, distance, course):
    fraction=jnp.clip((f32(distance)-f32(course))/(f32(distance)-f32(previous)),0.,1.)
    return (f32(frame_time).astype(jnp.float64)-fraction.astype(jnp.float64)*f32(dt).astype(jnp.float64)).astype(jnp.float32)

@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Parameters:
    course: jax.Array; phase_speed: jax.Array; phase_accel: jax.Array; end_speed_sqrt: jax.Array; add_speed: jax.Array
    minimum: jax.Array; blocking: jax.Array; spurt: jax.Array; slope: jax.Array; target_min: jax.Array; accel: jax.Array
    hp_initial: jax.Array; hp_style: jax.Array; hp_rates: jax.Array; hp_drain: jax.Array
    @classmethod
    def default(cls):
        require_x64(); raw=json.loads(files('globalsim_jax').joinpath('global_parameters.json').read_text(encoding='utf-8'))['parameters']
        rows=lambda parent,a,b: [[row[k] for k in ('Start','Middle','End','Last')] for row in parent[a]+parent[b]]
        return cls(f32([raw[k] for k in ('raceBaseSpeedBaseSpeed','raceBaseSpeedBaseDistance','raceBaseSpeedDistanceRate')]), f32(rows(raw['BasetTargetSpeed'],'PhaseBaseTargetSpeedPerArray','PhaseBaseTargetSpeedPerExArray')), f32(rows(raw['Speed'],'PhaseAccelCoefArray','PhaseAccelCoefExArray')), f32(raw['BasetTargetSpeed']['PhaseEndBaseTargetSpeedCoef']), f32(raw['addSpeedParamCoef']), f32([raw['Speed'][k] for k in ('MinSpeedRate','MinSpeedGutsCoefSqrt','MinSpeedGutsCoef')]), f32([raw['Block'][k] for k in ('FrontBlockDistanceGap','FrontBlockMaxSpeedRateMin','FrontBlockMaxSpeedRateMax')]), f32([raw[k] for k in ('lastSpurtBaseTargetSpeedAddCoef','LastSpurtTargetSpeedCoefSqrt','LastSpurtTargetSpeedGutsCoef1','LastSpurtTargetSpeedGutsPow','LastSpurtTargetSpeedGutsCoef2','baseTargetSpeedCoef')]), f32([raw['Slope'][k] for k in ('UpSlopeAddSpeedVal1','DownSlopeAddSpeedVal1','DownSlopeAddSpeedVal2')]), f32(raw['Speed']['TargetSpeedMin']), f32([raw[k] for k in ('AccelPowCoefSqrt','accelPowCoef','accelPowCoefUpSlope','StartAccelAdd')]), f32(raw['Hp']['HpInitialVal1']), f32([raw['Hp'][k] for k in ('HpMaxCoefNige','HpMaxCoefSenko','HpMaxCoefSashi','HpMaxCoefOikomi','HpMaxCoefOoNige')]), f32([raw['Hp'][k] for k in ('HpDecRateBaseNormal','HpDecRateBasePositionKeepPaseDown','HpDecRateBaseTemptation','HpDecRateBaseCompeteTopNige','HpDecRateBaseCompeteTopOonige','HpDecRateBaseTemptationAndCompeteTopNige','HpDecRateBaseTemptationAndCompeteTopOonige','HpDecRateMultiplyDownSlopeAccelMode')]), f32([raw['Hp'][k] for k in ('SpeedGapParam1','SpeedGapParam1Pow','HpDecBase','HpGutsCoefSqrt','HpGutsCoef')]))
    def course_speed(self,distance): return self.course[0]-(f32(distance)-self.course[1])/self.course[2]
    def base_target_speed(self,speed,strategy,phase,aptitude,distance):
        base=(self.phase_speed[jnp.asarray(strategy)-1,jnp.asarray(phase)].astype(jnp.float64)*f32(.01).astype(jnp.float64)*self.course_speed(distance).astype(jnp.float64))
        end=(_root(bounded_stat(speed)*self.end_speed_sqrt).astype(jnp.float64)*f32(aptitude).astype(jnp.float64)*self.add_speed.astype(jnp.float64)).astype(jnp.float32)
        return (base+jnp.where(jnp.asarray(phase)>=2,end.astype(jnp.float64),0.)).astype(jnp.float32)
    def start_dash_speed(self,distance): return self.course_speed(distance)*self.minimum[0]
    def minimum_speed(self,guts,distance): return (self.start_dash_speed(distance).astype(jnp.float64)+_root(bounded_stat(guts)*self.minimum[1]).astype(jnp.float64)*self.minimum[2].astype(jnp.float64)).astype(jnp.float32)
    def blocking_speed_cap(self,minimum,blocker,gap):
        ratio=jnp.clip(f32(gap)/self.blocking[0],0.,1.); rate=(self.blocking[1].astype(jnp.float64)+ratio.astype(jnp.float64)*(self.blocking[2].astype(jnp.float64)-self.blocking[1].astype(jnp.float64))).astype(jnp.float32); return jnp.maximum(f32(minimum),f32(blocker)*rate)
    def maximum_spurt_speed(self,speed,guts,strategy,aptitude,distance):
        base=(self.base_target_speed(speed,strategy,2,aptitude,distance).astype(jnp.float64)+self.course_speed(distance).astype(jnp.float64)*self.spurt[0].astype(jnp.float64)).astype(jnp.float32); addition=_root(bounded_stat(speed)*self.spurt[1]).astype(jnp.float64)*f32(aptitude).astype(jnp.float64)*self.add_speed.astype(jnp.float64); guts_add=(bounded_stat(guts)*self.spurt[2]).astype(jnp.float32)**self.spurt[3]; return ((base.astype(jnp.float64)*self.spurt[5].astype(jnp.float64)+addition)+guts_add.astype(jnp.float64)*self.spurt[4].astype(jnp.float64)).astype(jnp.float32)
    def target_speed(self,base,section,pace,force_in,power,moving_lane_skill,slope,downhill,hp_remaining,minimum,modifiers,spot_struggle=0.,bonus=0.):
        """Fixture-backed target-speed helper; modifiers are applied in input order."""
        speed=(f32(base)+f32(section))*f32(pace)+f32(force_in)
        modifiers = f32(modifiers)
        if modifiers.shape[0]:
            speed=jax.lax.fori_loop(0, modifiers.shape[0], lambda n, value: value + modifiers[n], speed)
        speed=speed+jnp.where(moving_lane_skill,_root(bounded_stat(power)*f32(.0002)),0.)
        correction=jnp.where(f32(slope)>0,-(f32(slope).astype(jnp.float64)*(self.slope[0].astype(jnp.float64)/bounded_stat(power).astype(jnp.float64))).astype(jnp.float32),jnp.where((f32(slope)<0)&downhill,(self.slope[1].astype(jnp.float64)+jnp.abs(f32(slope)).astype(jnp.float64)/self.slope[2].astype(jnp.float64)).astype(jnp.float32),0.))
        speed=speed+correction+f32(spot_struggle)+f32(bonus)
        speed=jnp.where(hp_remaining,speed,f32(minimum))
        return jnp.maximum(jnp.minimum(speed,30.),self.target_min)
    def initial_hp(self,stamina,strategy,distance): return ((bounded_stat(stamina).astype(jnp.float64)*self.hp_initial.astype(jnp.float64))*self.hp_style[jnp.asarray(strategy)-1].astype(jnp.float64)+f32(distance).astype(jnp.float64)).astype(jnp.float32)
    def hp_state_rate(self,rushing,competing,runaway,pace_down,downhill):
        index=jnp.where(competing,jnp.where(rushing,5,3)+jnp.asarray(runaway,jnp.int32),jnp.where(rushing,2,jnp.where(pace_down,1,0))); rate=self.hp_rates[index]; return jnp.where(downhill,rate*self.hp_rates[7],rate)
    def hp_per_second(self,speed,base_speed,ground,guts,phase,rate):
        # Inputs are field values in Rust; materialize their f32 ABI boundary
        # so a surrounding jitted graph cannot reuse an unrounded producer.
        store_f32=lambda value: jax.lax.bitcast_convert_type(jax.lax.bitcast_convert_type(f32(value),jnp.uint32),jnp.float32)
        speed=store_f32(speed); base_speed=store_f32(base_speed)
        ground=store_f32(ground); rate=store_f32(rate)
        gap=jax.lax.optimization_barrier((speed-base_speed)+self.hp_drain[0])
        # This is literally Rust's f32 `(gap * gap) / denominator` local.
        # The three barriers prevent CPU XLA from rewriting it as a reciprocal
        # multiply while retaining each required f32 rounding point.
        numerator = jax.lax.optimization_barrier(gap * gap)
        denominator = jax.lax.optimization_barrier(self.hp_drain[1])
        quotient = jax.lax.optimization_barrier(numerator / denominator)
        drained=jax.lax.optimization_barrier(quotient.astype(jnp.float64)*self.hp_drain[2].astype(jnp.float64))
        rated=jax.lax.optimization_barrier(drained*rate.astype(jnp.float64))
        value=jax.lax.optimization_barrier((rated*ground.astype(jnp.float64)).astype(jnp.float32))
        factor=jax.lax.optimization_barrier((1.+self.hp_drain[4].astype(jnp.float64)/_root(bounded_stat(guts)*self.hp_drain[3]).astype(jnp.float64)).astype(jnp.float32))
        return jnp.where(jnp.asarray(phase)>=2,value*factor,value)
    def acceleration_from_base(self,base,start_dash,modifiers):
        """Ordered additive modifiers plus start-dash/clamp, fixture-tested."""
        modifiers = f32(modifiers)
        value = f32(base) if not modifiers.shape[0] else jax.lax.fori_loop(0, modifiers.shape[0],lambda n,current: current+modifiers[n],f32(base))
        return jnp.maximum(value+jnp.where(start_dash,self.accel[3],0.),0.)
    def acceleration(self,power,strategy,phase,ground,distance,uphill,start_dash,modifiers):
        base=(_root(bounded_stat(power)*self.accel[0]).astype(jnp.float64)*self.accel[jnp.where(uphill,2,1)].astype(jnp.float64)*f32(ground).astype(jnp.float64)*f32(distance).astype(jnp.float64)*self.phase_accel[jnp.asarray(strategy)-1,jnp.asarray(phase)].astype(jnp.float64)).astype(jnp.float32); return self.acceleration_from_base(base,start_dash,modifiers)
