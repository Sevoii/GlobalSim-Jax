import jax
import jax.numpy as jnp
import numpy as np
import json
from pathlib import Path
import pytest

from globalsim_jax.field import BlockParameters, FieldState, blocking_snapshot, front_lane_limit, initialize_field, prepare_blocking, ranks, step as field_step, run_checked
from globalsim_jax.restricted import RestrictedConfig
from globalsim_jax.physics import Parameters
from globalsim_jax.restricted import initialize


def test_rank_ties_and_blocking_boundary_rules_eager_and_jit():
    # Equal positions: lower runner index wins; retail `RaceField.update` is one-based.
    position = jnp.asarray([[10.0, 10.0, 11.0]], jnp.float32)
    assert np.array_equal(np.asarray(ranks(position)), [[2, 3, 1]])
    # Front ties retain lowest index; finished front rows remain blockers.
    position = jnp.asarray([[10.0, 11.0, 11.0, 9.5]], jnp.float32)
    lane = jnp.asarray([[0.0, 0.0, 0.0, 0.05]], jnp.float32)
    finished = jnp.asarray([[False, True, False, False]])
    front, gap, side = blocking_snapshot(position, lane, finished)
    assert np.array_equal(np.asarray(front), [[1, -1, -1, -1]])
    assert np.asarray(gap, np.float32).view(np.uint32)[0, 0] == np.asarray(1.0, np.float32).view(np.uint32)
    assert np.array_equal(np.asarray(side), [[True, False, False, True]])
    compiled = jax.jit(blocking_snapshot)(position, lane, finished)
    for eager, jit in zip((front, gap, side), compiled):
        if eager.dtype == jnp.float32:
            assert np.array_equal(np.asarray(eager).view(np.uint32), np.asarray(jit).view(np.uint32))
        else:
            assert np.array_equal(np.asarray(eager), np.asarray(jit))


def test_frozen_rust_query_cases():
    fixture = json.loads((Path(__file__).parent / "fixtures" / "field_query_trace.json").read_text())
    rows = {
        "ties": ([[10., 11., 11., 9.5]], [[0., 0., 0., .05]], [[False, True, False, False]]),
        "bounds": ([[0., 2., 1., 1.]], [[0., 0., .02916667, 0.]], [[False, False, False, False]]),
    }
    for case in fixture["cases"]:
        position, lane, finished = rows[case["name"]]
        front, gap, side = blocking_snapshot(jnp.asarray(position), jnp.asarray(lane), jnp.asarray(finished))
        assert np.array_equal(np.asarray(front)[0], case["front"])
        assert np.array_equal(np.asarray(ranks(jnp.asarray(position)))[0], case["rank"])
        assert np.array_equal(np.asarray(gap)[0].view(np.uint32), case["gap_bits"])
        assert np.array_equal(np.asarray(side, np.uint8)[0], case["side"])


def test_block_strict_boundaries_and_lane_limit():
    parameters = BlockParameters()
    assert float(front_lane_limit(jnp.float32(0.0), parameters)) == np.float32(parameters.front_lane)
    # Exactly front_distance and exactly side_distance are excluded; front lane limit is inclusive.
    position = jnp.asarray([[0.0, 2.0, 1.0, 1.0]], jnp.float32)
    limit = front_lane_limit(jnp.float32(1.0), parameters)
    lane = jnp.asarray([[0.0, 0.0, limit, 0.0]], jnp.float32)
    front, _, side = blocking_snapshot(position, lane, jnp.zeros((1, 4), bool), parameters)
    assert int(front[0, 0]) == 2
    assert bool(side[0, 0])  # 1m row remains side-adjacent; only 2m is excluded.


def test_ordered_stored_minimum_and_f64_block_timers():
    params = Parameters.default()
    runners = initialize(params, speed=[[1000., 1000.]], stamina=[[1000., 1000.]], power=[[1000., 1000.]], guts=[[900., 900.]], strategy=[[1, 1]], aptitude=[[1., 1.]], ground=[[1., 1.]], delay=[[1., 1.]], distance=1200.)
    runners = runners.__class__(runners.time, jnp.asarray([[10., 11.]], jnp.float32), runners.speed, runners.hp, runners.delay, runners.finished, runners.timed_out, runners.finish_time, runners.ticks, runners.phase, jnp.asarray([[False, False]]))
    field = initialize_field(runners, [[0., 0.]], prior_minimum=[[3., 3.]])
    updated, front, _, side, rank = prepare_blocking(params, field, guts=jnp.asarray([[900., 900.]], jnp.float32), distance=1200., dt=0.1)
    assert np.array_equal(np.asarray(front), [[1, -1]])
    assert np.array_equal(np.asarray(rank), [[2, 1]])
    assert updated.blocked_front_time.dtype == jnp.float64
    assert float(updated.blocked_front_time[0, 0]) == float(jnp.float64(jnp.float32(0.1)))
    assert not bool(side[0, 0])


def test_ordered_field_step_truncates_after_crossing_pass():
    params = Parameters.default()
    runners = initialize(params, speed=[[1000., 1000.]], stamina=[[1000., 1000.]], power=[[1000., 1000.]], guts=[[900., 900.]], strategy=[[1, 1]], aptitude=[[1., 1.]], ground=[[1., 1.]], delay=[[0., 0.]], distance=1200.)
    runners = runners.__class__(runners.time, jnp.asarray([[1199.9, 1199.8]], jnp.float32), jnp.asarray([[20., 20.]], jnp.float32), runners.hp, runners.delay, runners.finished, runners.timed_out, runners.finish_time, runners.ticks, jnp.asarray([[2, 2]], jnp.int32), jnp.asarray([[False, False]]))
    field = initialize_field(runners, [[0., 0.]], prior_minimum=[[3., 3.]])
    args = dict(speed_stat=jnp.asarray([[1000., 1000.]], jnp.float32), power=jnp.asarray([[1000., 1000.]], jnp.float32), guts=jnp.asarray([[900., 900.]], jnp.float32), strategy=jnp.asarray([[1, 1]], jnp.int32), aptitude=jnp.ones((1, 2), jnp.float32), ground=jnp.ones((1, 2), jnp.float32), distance=1200., dt=.066600002348423)
    moved, *_ = field_step(params, field, max_ticks=99, **args)
    assert bool(moved.truncated[0])
    compiled, *_ = jax.jit(lambda value: field_step(params, value, max_ticks=99, **args))(field)
    assert np.array_equal(np.asarray(compiled.runners.position).view(np.uint32), np.asarray(moved.runners.position).view(np.uint32))
    frozen, *_ = field_step(params, moved, max_ticks=99, **args)
    assert np.array_equal(np.asarray(frozen.runners.position), np.asarray(moved.runners.position))

def test_terminal_field_state_freezes_all_leaves_and_scan_padding():
    params=Parameters.default()
    runners=initialize(params,speed=[[1000.,1000.]],stamina=[[1000.,1000.]],power=[[1000.,1000.]],guts=[[900.,900.]],strategy=[[1,1]],aptitude=[[1.,1.]],ground=[[1.,1.]],delay=[[0.,0.]],distance=1200.)
    runners=runners.__class__(runners.time,jnp.asarray([[1199.9,1199.8]],jnp.float32),jnp.asarray([[20.,20.]],jnp.float32),runners.hp,runners.delay,runners.finished,runners.timed_out,runners.finish_time,runners.ticks,jnp.asarray([[2,2]],jnp.int32),jnp.zeros((1,2),bool))
    field=initialize_field(runners,[[0.,.02]],prior_minimum=[[3.,3.]])
    kw=dict(speed_stat=jnp.full((1,2),1000.,jnp.float32),power=jnp.full((1,2),1000.,jnp.float32),guts=jnp.full((1,2),900.,jnp.float32),strategy=jnp.ones((1,2),jnp.int32),aptitude=jnp.ones((1,2),jnp.float32),ground=jnp.ones((1,2),jnp.float32),distance=1200.,dt=.066600002348423,max_ticks=99)
    terminal,*_=field_step(params,field,**kw)
    frozen,*_=field_step(params,terminal,**kw)
    for name in terminal.__dataclass_fields__:
        left,right=getattr(terminal,name),getattr(frozen,name)
        if name=="runners":
            for leaf in left.__dataclass_fields__:
                x,y=np.asarray(getattr(left,leaf)),np.asarray(getattr(right,leaf))
                assert np.array_equal(x.view(np.uint32),y.view(np.uint32)) if x.dtype==np.float32 else np.array_equal(x,y)
        else:
            x,y=np.asarray(left),np.asarray(right)
            assert np.array_equal(x.view(np.uint64),y.view(np.uint64)) if x.dtype==np.float64 else np.array_equal(x,y)
    _,trace=jax.lax.scan(lambda carry,_:(field_step(params,carry,**kw)[0],field_step(params,carry,**kw)[0]),terminal,None,length=3)
    for name in trace.__dataclass_fields__:
        if name=="runners":
            for leaf in trace.runners.__dataclass_fields__:
                x=np.asarray(getattr(trace.runners,leaf)); assert np.array_equal(x[0],x[1]) and np.array_equal(x[1],x[2])
        else:
            x=np.asarray(getattr(trace,name)); assert np.array_equal(x[0],x[1]) and np.array_equal(x[1],x[2])

def test_initialize_field_rejects_nonfresh_without_prior_minimum():
    params=Parameters.default()
    runners=initialize(params, speed=[[1000.]], stamina=[[1000.]], power=[[1000.]], guts=[[900.]], strategy=[[1]], aptitude=[[1.]], ground=[[1.]], delay=[[0.]], distance=1200.)
    moved=runners.__class__(runners.time, jnp.asarray([[1.]],jnp.float32), runners.speed, runners.hp, runners.delay, runners.finished, runners.timed_out, runners.finish_time, runners.ticks, runners.phase, runners.start_dash)
    with pytest.raises(ValueError,match="prior_minimum"):
        initialize_field(moved,[[0.]])
    assert initialize_field(moved,[[0.]],prior_minimum=[[3.]]).minimum.shape==(1,1)

def test_all_timeout_attachment_seeds_truncated_and_negative_minimum_rejects():
    params=Parameters.default()
    runners=initialize(params,speed=[[1000.]],stamina=[[1000.]],power=[[1000.]],guts=[[900.]],strategy=[[1]],aptitude=[[1.]],ground=[[1.]],delay=[[0.]],distance=1200.)
    timed=runners.__class__(runners.time,runners.position,runners.speed,runners.hp,runners.delay,runners.finished,jnp.ones((1,1),bool),runners.finish_time,runners.ticks,runners.phase,runners.start_dash)
    field=initialize_field(timed,[[0.]],prior_minimum=[[3.]])
    assert bool(field.truncated[0])
    with pytest.raises(ValueError,match="nonnegative"):
        initialize_field(timed,[[0.]],prior_minimum=[[-1.]])

def test_whole_field_state_is_bitwise_frozen_after_finish_and_alltimeout():
    params=Parameters.default()
    base=initialize(params,speed=[[1000.,1000.]],stamina=[[1000.,1000.]],power=[[1000.,1000.]],guts=[[900.,900.]],strategy=[[1,1]],aptitude=[[1.,1.]],ground=[[1.,1.]],delay=[[0.,0.]],distance=1200.)
    crossing=base.__class__(base.time,jnp.asarray([[1198.9,1199.9]],jnp.float32),jnp.asarray([[20.,20.]],jnp.float32),base.hp,base.delay,base.finished,base.timed_out,base.finish_time,base.ticks,jnp.asarray([[2,2]],jnp.int32),jnp.zeros((1,2),bool))
    field=initialize_field(crossing,[[0.,.02]],prior_minimum=[[3.,3.]])
    kw=dict(speed_stat=jnp.full((1,2),1000.,jnp.float32),power=jnp.full((1,2),1000.,jnp.float32),guts=jnp.full((1,2),900.,jnp.float32),strategy=jnp.ones((1,2),jnp.int32),aptitude=jnp.ones((1,2),jnp.float32),ground=jnp.ones((1,2),jnp.float32),distance=1200.,dt=.066600002348423,max_ticks=99)
    terminal=field_step(params,field,**kw)[0]
    assert np.any(np.asarray(terminal.blocked_front_time)>0) and np.any(np.asarray(terminal.blocked_side_time)>0) and np.any(np.asarray(terminal.blocked_all_time)>0)
    timed=base.__class__(base.time,jnp.asarray([[0.,1.]],jnp.float32),base.speed,base.hp,base.delay,base.finished,jnp.ones((1,2),bool),base.finish_time,base.ticks,base.phase,base.start_dash)
    timeout=initialize_field(timed,[[0.,.02]],prior_minimum=[[3.,3.]])
    timeout=FieldState(timeout.runners,timeout.lane,timeout.minimum,jnp.asarray([[.1,.2]],jnp.float64),jnp.asarray([[.3,.4]],jnp.float64),jnp.asarray([[.5,.6]],jnp.float64),timeout.truncated)
    for state in (terminal,timeout):
        expected=[(np.asarray(leaf).dtype,np.asarray(leaf).shape,np.asarray(leaf).tobytes()) for leaf in jax.tree_util.tree_leaves(state)]
        again=field_step(params,state,**kw)[0]
        actual=[(np.asarray(leaf).dtype,np.asarray(leaf).shape,np.asarray(leaf).tobytes()) for leaf in jax.tree_util.tree_leaves(again)]
        assert actual==expected
        _,trace=jax.lax.scan(lambda carry,_:(field_step(params,carry,**kw)[0],field_step(params,carry,**kw)[0]),state,None,length=3)
        for tick in range(3):
            row=jax.tree_util.tree_map(lambda value:value[tick],trace)
            got=[(np.asarray(leaf).dtype,np.asarray(leaf).shape,np.asarray(leaf).tobytes()) for leaf in jax.tree_util.tree_leaves(row)]
            assert got==expected

def test_custom_block_parameters_round_to_f32_before_widening_and_jit():
    # The host value has extra f64 digits; Rust's field parameter is f32 first.
    custom=BlockParameters(front_lane=0.0416666716337204 + 1e-10)
    assert custom.front_lane == float(np.float32(0.0416666716337204 + 1e-10))
    gap=jnp.asarray([0.25,1.0,1.75],jnp.float32)
    eager=front_lane_limit(gap,custom)
    compiled=jax.jit(lambda x:front_lane_limit(x,custom))(gap)
    assert np.array_equal(np.asarray(eager).view(np.uint32),np.asarray(compiled).view(np.uint32))
    with pytest.raises(ValueError): BlockParameters(front_lane=1e-50)

def test_custom_front_lane_matches_frozen_rust_vector_not_unrounded_plane():
    fixture=json.loads((Path(__file__).parent / "fixtures" / "field_query_trace.json").read_text())
    vector=fixture["custom_vectors"][0]
    parameters=BlockParameters(front_lane=vector["front_lane_raw"])
    gap=jnp.asarray(f32:=np.asarray(vector["gap_bits"],np.uint32).view(np.float32),jnp.float32)
    eager=front_lane_limit(gap,parameters)
    compiled=jax.jit(lambda value:front_lane_limit(value,parameters))(gap)
    assert np.asarray(eager,np.float32).view(np.uint32).item()==vector["expected_bits"]
    assert np.asarray(compiled,np.float32).view(np.uint32).item()==vector["expected_bits"]
    narrowing=(gap/jnp.float32(2.))*jnp.float32(.6)
    bad=((jnp.float64(1.)-narrowing.astype(jnp.float64))*jnp.float64(vector["front_lane_raw"])).astype(jnp.float32)
    assert np.asarray(bad,np.float32).view(np.uint32).item()!=vector["expected_bits"]

def test_run_checked_allows_truncated_postgoal_batch_alongside_active_batch():
    params=Parameters.default()
    runners=initialize(params,speed=[[1000.],[1000.]],stamina=[[1000.],[1000.]],power=[[1000.],[1000.]],guts=[[900.],[900.]],strategy=[[1],[1]],aptitude=[[1.],[1.]],ground=[[1.],[1.]],delay=[[0.],[0.]],distance=1200.)
    runners=runners.__class__(runners.time,jnp.asarray([[1200.],[0.]],jnp.float32),runners.speed,runners.hp,runners.delay,runners.finished,runners.timed_out,runners.finish_time,runners.ticks,runners.phase,runners.start_dash)
    field=initialize_field(runners,[[0.],[0.]],prior_minimum=[[3.],[3.]])
    field=FieldState(field.runners,field.lane,field.minimum,field.blocked_front_time,field.blocked_side_time,field.blocked_all_time,jnp.asarray([True,False]))
    _,trace=run_checked(params,field,config=RestrictedConfig(1200.,.1,1),speed_stat=jnp.asarray([[1000.],[1000.]],jnp.float32),power=jnp.asarray([[1000.],[1000.]],jnp.float32),guts=jnp.asarray([[900.],[900.]],jnp.float32),strategy=jnp.ones((2,1),jnp.int32),aptitude=jnp.ones((2,1),jnp.float32),ground=jnp.ones((2,1),jnp.float32))
    assert trace.runners.position.shape==(1,2,1)
