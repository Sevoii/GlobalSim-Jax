import json
from dataclasses import replace
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import pytest
from globalsim_jax.physics import Parameters, phase_boundaries, target_speed_indices, advance_start_delay, finish_time

ROOT=Path(__file__).parent/'fixtures'
def bits(x): return np.asarray(x,dtype=np.float32).view(np.uint32).item()
def test_physics_fixture_exact_bits_eager_and_jit():
    p=Parameters.default(); data=json.loads((ROOT/'retail_physics_port.json').read_text())
    @jax.jit
    def compiled(speed,guts,stamina,strategy,phase,aptitude,distance,ground,gap,base,section,pace,force_in,power,moving,slope,downhill,hp_remaining,minimum,rushing,competing,runaway,pace_down,start_dash,modifiers):
        course=p.course_speed(distance); base_speed=p.base_target_speed(speed,strategy,phase,aptitude,distance); rate=p.hp_state_rate(rushing,competing,runaway,pace_down,downhill)
        hp = p.hp_per_second(base_speed,course,ground,guts,phase,rate)
        return (course,base_speed,p.minimum_speed(guts,distance),p.maximum_spurt_speed(speed,guts,strategy,aptitude,distance),p.target_speed(base,section,pace,force_in,power,moving,slope,downhill,hp_remaining,minimum,modifiers),p.initial_hp(stamina,strategy,distance),rate,hp,p.acceleration(power,strategy,phase,ground,aptitude,slope>0,start_dash,modifiers),p.blocking_speed_cap(p.minimum_speed(guts,distance),base_speed,gap))
    for row in data['vectors']:
        i=row['input']; t=i['target']; f=i['flags']; expected=row['expected_bits']
        values={
          'course_speed':p.course_speed(i['distance']), 'base_target_speed':p.base_target_speed(i['speed'],i['strategy'],i['phase'],i['aptitude'],i['distance']), 'minimum_speed':p.minimum_speed(i['guts'],i['distance']), 'maximum_spurt_speed':p.maximum_spurt_speed(i['speed'],i['guts'],i['strategy'],i['aptitude'],i['distance']),
          'target_speed':p.target_speed(t['base'],t['section'],t['pace_coef'],t['force_in'],i['power'],t['moving_lane_skill'],t['slope'],f['downhill'],t['hp_remaining'],t['minimum'],jnp.array(i['modifiers'])),
          'initial_hp':p.initial_hp(i['stamina'],i['strategy'],i['distance']), 'hp_state_rate':p.hp_state_rate(f['rushing'],f['competing'],f['runaway'],f['pace_down'],f['downhill']), 'hp_per_second':p.hp_per_second(p.base_target_speed(i['speed'],i['strategy'],i['phase'],i['aptitude'],i['distance']),p.course_speed(i['distance']),i['ground'],i['guts'],i['phase'],p.hp_state_rate(f['rushing'],f['competing'],f['runaway'],f['pace_down'],f['downhill'])),
          'acceleration':p.acceleration(i['power'],i['strategy'],i['phase'],i['ground'],i['aptitude'],t['slope']>0,i['start_dash'],jnp.array(i['modifiers'])), 'blocking_speed_cap':p.blocking_speed_cap(p.minimum_speed(i['guts'],i['distance']),p.base_target_speed(i['speed'],i['strategy'],i['phase'],i['aptitude'],i['distance']),i['gap'])}
        for key,value in values.items(): assert bits(value)==expected[key], (key,bits(value),expected[key])
        compiled_values=compiled(i['speed'],i['guts'],i['stamina'],i['strategy'],i['phase'],i['aptitude'],i['distance'],i['ground'],i['gap'],t['base'],t['section'],t['pace_coef'],t['force_in'],i['power'],t['moving_lane_skill'],t['slope'],f['downhill'],t['hp_remaining'],t['minimum'],f['rushing'],f['competing'],f['runaway'],f['pace_down'],i['start_dash'],jnp.array(i['modifiers']))
        compiled_keys=list(values)
        for key,value in zip(compiled_keys,compiled_values):
            assert bits(value)==expected[key], ('jit',key,bits(value),expected[key])
def test_hp_per_second_jit_exact_regression():
    p=Parameters.default(); fn=jax.jit(p.hp_per_second)
    for row in json.loads((ROOT/'retail_physics_port.json').read_text())['vectors']:
        i=row['input']; f=i['flags']; base=p.base_target_speed(i['speed'],i['strategy'],i['phase'],i['aptitude'],i['distance']); course=p.course_speed(i['distance']); rate=p.hp_state_rate(f['rushing'],f['competing'],f['runaway'],f['pace_down'],f['downhill'])
        assert bits(fn(base,course,i['ground'],i['guts'],i['phase'],rate))==row['expected_bits']['hp_per_second']
def test_ordered_acceleration_modifier_fold_eager_and_jit():
    p=Parameters.default(); p=replace(p,accel=jnp.array([1.,-2.,-3.,24.],jnp.float32),phase_accel=jnp.ones((5,4),jnp.float32))
    def calc(modifiers): return p.acceleration_from_base(1.,False,jnp.array(modifiers,jnp.float32))
    assert float(calc([16777216.,-16777216.]))==0.; assert float(calc([-16777216.,16777216.]))==1.
    jit_calc=jax.jit(calc); assert float(jit_calc([16777216.,-16777216.]))==0.; assert float(jit_calc([-16777216.,16777216.]))==1.
def test_boundary_and_finish_fixtures():
    assert bits(phase_boundaries(1600.)[2])==np.float32(1333.333251953125).view(np.uint32)
    section,phase=target_speed_indices(1600.,1600.); assert (int(section),int(phase))==(23,2)
    remaining,active,fraction=advance_start_delay(.0666,.0666); assert float(remaining)==0 and bool(active) and float(fraction)==1.
    remaining,active,_=advance_start_delay(.04,0.); assert float(remaining)==np.float32(.04) and not bool(active)
    remaining,active,fraction=advance_start_delay(-.02,.1); assert float(remaining)==0 and bool(active) and np.isclose(float(fraction),.2)
    _,active,fraction=advance_start_delay(-.02,0.); assert not bool(active) and float(fraction)==0.
    rows=json.loads((ROOT/'retail_finish_crossings.json').read_text())['rows']
    for row in rows:
        if row['adjacent_ticks']:
            actual=finish_time(row['frame_time'],row['dt'],row['previous_distance'],row['distance'],row['course_distance'])
            assert np.isclose(float(actual),row['expected'],rtol=0,atol=1e-5)
