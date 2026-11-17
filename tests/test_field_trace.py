import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np

from globalsim_jax.field import FieldState, initialize_field, step as field_step
from globalsim_jax.physics import Parameters
from globalsim_jax.restricted import RestrictedState

TRACE = json.loads((Path(__file__).parent / "fixtures" / "field_tick_trace.json").read_text())

def f32(bits): return np.asarray(bits, np.uint32).view(np.float32)
def f64(bits): return np.asarray(bits, np.uint64).view(np.float64)
def bits32(x): return np.asarray(x, np.float32).view(np.uint32).item()
def bits64(x): return np.asarray(x, np.float64).view(np.uint64).item()

def initial(case):
    rows=case["initial"]
    matrix=lambda channel,dtype=jnp.float32:jnp.asarray([[f32(row[channel]) for row in rows]],dtype)
    runners=RestrictedState(matrix(0),matrix(1),matrix(2),matrix(3),matrix(4),jnp.zeros((1,3),bool),jnp.zeros((1,3),bool),jnp.full((1,3),jnp.nan,jnp.float32),jnp.zeros((1,3),jnp.int32),jnp.asarray([[row[5] for row in rows]],jnp.int32),jnp.asarray([[bool(row[6]) for row in rows]],bool))
    lane=jnp.asarray([[f32(value) for value in case["lanes_bits"]]],jnp.float32)
    base=initialize_field(runners,lane,prior_minimum=matrix(8))
    return FieldState(runners,lane,matrix(8),jnp.asarray([[f64(row[13]) for row in rows]],jnp.float64),jnp.asarray([[f64(row[14]) for row in rows]],jnp.float64),jnp.asarray([[f64(row[15]) for row in rows]],jnp.float64),jnp.zeros((1,),bool))

def args(batch=1):
    inp=TRACE["inputs"]
    row=lambda bits,dtype:jnp.full((batch,3),f32(bits),dtype)
    return dict(speed_stat=row(inp["speed_stat_bits"],jnp.float32),power=row(inp["power_bits"],jnp.float32),guts=row(inp["guts_bits"],jnp.float32),strategy=jnp.ones((batch,3),jnp.int32),aptitude=row(inp["aptitude_bits"],jnp.float32),ground=row(inp["ground_bits"],jnp.float32),distance=float(f32(inp["distance_bits"])),dt=float(f32(TRACE["dt_bits"])),max_ticks=99)

def assert_row(state,front,gap,side,rank,truncated,expected,batch=0):
    fields=[state.runners.time,state.runners.position,state.runners.speed,state.runners.hp,state.runners.delay,state.runners.phase,state.runners.start_dash,state.runners.finish_time,state.minimum,front,gap,side,rank,state.blocked_front_time,state.blocked_side_time,state.blocked_all_time,state.runners.finished]
    for runner,row in enumerate(expected):
        for channel,value in enumerate(fields):
            actual=np.asarray(value)[batch,runner]
            if channel in (5,9,12): assert int(actual)==row[channel]
            elif channel in (6,11,16): assert int(actual)==row[channel]
            elif channel in (13,14,15): assert bits64(actual)==row[channel]
            else: assert bits32(actual)==row[channel]
        assert int(np.asarray(truncated)[batch])==row[17]

def scalar_trace(parameters,case):
    state=initial(case); kw=args()
    eager=[]
    for row in case["rows"]:
        state,front,gap,side,rank=field_step(parameters,state,**kw)
        assert_row(state,front,gap,side,rank,state.truncated,row)
        eager.append((state,front,gap,side,rank))
    def body(carry,_):
        out=field_step(parameters,carry,**kw)
        return out[0],out
    _,compiled=jax.jit(lambda value:jax.lax.scan(body,value,None,length=8))(initial(case))
    for tick,row in enumerate(case["rows"]):
        states,fronts,gaps,sides,ranks=compiled
        runner_trace=states.runners
        state=FieldState(RestrictedState(*[getattr(runner_trace,n)[tick] for n in runner_trace.__dataclass_fields__]),*[getattr(states,n)[tick] for n in states.__dataclass_fields__ if n!='runners'])
        assert_row(state,fronts[tick],gaps[tick],sides[tick],ranks[tick],state.truncated,row)
    return eager,compiled

def test_frozen_field_cases_eager_and_jitted_scan():
    assert TRACE["rank_label"]=="pre_movement"
    parameters=Parameters.default()
    for case in TRACE["cases"]: scalar_trace(parameters,case)

def test_two_batch_field_trace_matches_scalar_every_field():
    parameters=Parameters.default(); cases=TRACE["cases"][:2]
    scalar=[scalar_trace(parameters,case)[1] for case in cases]
    states=[initial(case) for case in cases]
    batch=FieldState(RestrictedState(*[jnp.concatenate((getattr(states[0].runners,n),getattr(states[1].runners,n)),axis=0) for n in states[0].runners.__dataclass_fields__]),*[jnp.concatenate((getattr(states[0],n),getattr(states[1],n)),axis=0) for n in states[0].__dataclass_fields__ if n!='runners'])
    kw=args(batch=2)
    def body(carry,_):
        out=field_step(parameters,carry,**kw); return out[0],out
    _,trace=jax.jit(lambda value:jax.lax.scan(body,value,None,length=8))(batch)
    states_trace,fronts,gaps,sides,ranks=trace
    for tick in range(8):
        for field in states_trace.__dataclass_fields__:
            if field=='runners':
                for leaf in states_trace.runners.__dataclass_fields__:
                    actual=np.asarray(getattr(states_trace.runners,leaf)[tick])
                    for b in range(2):
                        expected=np.asarray(getattr(scalar[b][0].runners,leaf)[tick,0])
                        if actual.dtype==np.float32: assert np.array_equal(actual[b].view(np.uint32),expected.view(np.uint32))
                        else: assert np.array_equal(actual[b],expected)
            else:
                actual=np.asarray(getattr(states_trace,field)[tick])
                for b in range(2):
                    expected=np.asarray(getattr(scalar[b][0],field)[tick,0])
                    if actual.dtype==np.float64: assert np.array_equal(actual[b].view(np.uint64),expected.view(np.uint64))
                    elif actual.dtype==np.float32: assert np.array_equal(actual[b].view(np.uint32),expected.view(np.uint32))
                    else: assert np.array_equal(actual[b],expected)
        for b,case in enumerate(cases):
            st=FieldState(RestrictedState(*[getattr(states_trace.runners,n)[tick] for n in states_trace.runners.__dataclass_fields__]),*[getattr(states_trace,n)[tick] for n in states_trace.__dataclass_fields__ if n!='runners'])
            assert_row(st,fronts[tick],gaps[tick],sides[tick],ranks[tick],st.truncated,case["rows"][tick],b)

def test_lower_blocker_live_speed_is_not_simultaneous_prepass_speed():
    case=TRACE["cases"][1]; row=case["rows"][0]
    parameters=Parameters.default()
    minimum=jnp.asarray(f32(row[1][8]),jnp.float32)
    gap=jnp.asarray(f32(row[1][10]),jnp.float32)
    frozen_cap=parameters.blocking_speed_cap(minimum,jnp.asarray(f32(case["initial"][0][2]),jnp.float32),gap)
    live_cap=parameters.blocking_speed_cap(minimum,jnp.asarray(f32(row[0][2]),jnp.float32),gap)
    actual=jnp.asarray(f32(row[1][2]),jnp.float32)
    assert bits32(actual)==bits32(live_cap)
    assert bits32(actual)!=bits32(frozen_cap)

def test_partial_gate_refreshes_minimum_and_timers_before_movement():
    case=TRACE["cases"][1]; initial_row=case["initial"][1]; first=case["rows"][0][1]
    assert first[1]!=initial_row[1] and first[4]==0
    assert first[8]!=initial_row[8]
    assert first[13]!=0 and first[14]!=0 and first[15]!=0

def test_closed_gate_frozen_case_advances_clock_and_refreshes_field_state():
    case=next(case for case in TRACE["cases"] if case["name"]=="closed_gate")
    initial_row=case["initial"][1]; first=case["rows"][0][1]
    assert first[0]!=initial_row[0]
    assert first[1]==initial_row[1] and first[2]==initial_row[2] and first[3]==initial_row[3]
    assert first[4]!=initial_row[4] and first[8]!=initial_row[8]
    assert first[13]!=0 and first[14]!=0 and first[15]!=0
