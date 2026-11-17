"""Restricted M2 demo and synchronized primitive benchmark."""
import argparse, time
import jax

def _enable_x64(): jax.config.update('jax_enable_x64', True)

def benchmark(batch: int, repeats: int):
    _enable_x64()
    import jax.numpy as jnp
    from .rng import seed, next_double
    seeds=jnp.arange(batch,dtype=jnp.int32)
    fn=jax.jit(jax.vmap(lambda x: next_double(seed(x))[1]))
    fn(seeds).block_until_ready()  # compilation/warmup excluded
    started=time.perf_counter()
    for _ in range(repeats): fn(seeds).block_until_ready()
    elapsed=time.perf_counter()-started
    device=jax.devices()[0]
    print(f"primitive=rng.next_double batch={batch} repeats={repeats} device={device.platform}:{device} elapsed_s={elapsed:.6f} samples_per_s={batch*repeats/elapsed:.0f}")
    print("Primitive-only synchronized result; not a full-race benchmark or GPU speedup claim.")


def restricted_demo(ticks: int):
    """Run the checked, single-lane flat/no-field M2 profile."""
    _enable_x64()
    import jax.numpy as jnp
    from .physics import Parameters
    from .restricted import RestrictedConfig, initialize, run_checked

    parameters = Parameters.default()
    runner = dict(
        speed=jnp.asarray([[1000.]], jnp.float32), stamina=jnp.asarray([[1000.]], jnp.float32),
        power=jnp.asarray([[1000.]], jnp.float32), guts=jnp.asarray([[900.]], jnp.float32),
        strategy=jnp.asarray([[1]], jnp.int32), aptitude=jnp.asarray([[1.]], jnp.float32),
        ground=jnp.asarray([[1.]], jnp.float32), delay=jnp.asarray([[0.]], jnp.float32),
    )
    state = initialize(parameters, **runner, distance=1200.)
    _, trace = run_checked(
        parameters, state, config=RestrictedConfig(1200., 0.066600002348423, ticks),
        speed_stat=runner["speed"], power=runner["power"], guts=runner["guts"],
        strategy=runner["strategy"], aptitude=runner["aptitude"], ground=runner["ground"],
    )
    final = trace.position[-1, 0, 0]
    print(f"restricted_profile=flat_no_field ticks={ticks} final_position_m={float(final):.6f}")
    print("Checked M2 subset only; no field interactions, skills, geometry integration, or full-race claim.")


def field_demo(ticks: int):
    _enable_x64()
    import jax.numpy as jnp
    from .physics import Parameters
    from .restricted import RestrictedConfig, initialize
    from .field import initialize_field, run_checked
    params=Parameters.default()
    runner=dict(speed=jnp.asarray([[1000.,1000.]],jnp.float32),stamina=jnp.asarray([[1000.,1000.]],jnp.float32),power=jnp.asarray([[1000.,1000.]],jnp.float32),guts=jnp.asarray([[900.,900.]],jnp.float32),strategy=jnp.asarray([[1,1]],jnp.int32),aptitude=jnp.ones((1,2),jnp.float32),ground=jnp.ones((1,2),jnp.float32),delay=jnp.zeros((1,2),jnp.float32))
    field=initialize_field(initialize(params,**runner,distance=1200.),jnp.asarray([[0.,.02]],jnp.float32))
    _,trace=run_checked(params,field,config=RestrictedConfig(1200.,.066600002348423,ticks),speed_stat=runner["speed"],power=runner["power"],guts=runner["guts"],strategy=runner["strategy"],aptitude=runner["aptitude"],ground=runner["ground"])
    print(f"m3a_fixed_lane positions={list(map(float,trace.runners.position[-1,0]))} truncated={bool(trace.truncated[-1,0])}")
    print("M3a bounded profile: no skills, pacing, lane selection, or post-goal retail parity.")


def moving_field_demo(ticks: int):
    _enable_x64()
    import jax.numpy as jnp
    from .physics import Parameters
    from .restricted import RestrictedConfig, initialize
    from .field import initialize_field
    from .lane_move import MoveParameters
    from .moving_field import initialize as attach, run_checked
    config = RestrictedConfig(distance=1200., dt=.0666, max_ticks=ticks)
    values = dict(
        speed=jnp.full((1, 2), 1000., jnp.float32),
        stamina=jnp.full((1, 2), 1000., jnp.float32),
        power=jnp.full((1, 2), 1000., jnp.float32),
        guts=jnp.full((1, 2), 900., jnp.float32),
        strategy=jnp.ones((1, 2), jnp.int32),
        aptitude=jnp.ones((1, 2), jnp.float32),
        ground=jnp.ones((1, 2), jnp.float32),
        delay=jnp.zeros((1, 2), jnp.float32),
    )
    params = Parameters.default()
    field = initialize_field(initialize(params, distance=config.distance, **values), [[.1, .5]])
    state = attach(field, [[.7, 0.]])
    movement = MoveParameters(*[jnp.float32(x) for x in (.001, .3, .02, .05, 1.5)])
    final, _ = run_checked(
        params, state, config=config, max_lane=1., first_move_point=375.,
        move_parameters=movement,
        speed_stat=values['speed'], power=values['power'], guts=values['guts'],
        strategy=values['strategy'], aptitude=values['aptitude'], ground=values['ground'],
    )
    final.field.lane.block_until_ready()
    print(f"profile=supplied_target_moving_field ticks={ticks} positions={final.field.runners.position.tolist()} lanes={final.field.lane.tolist()} truncated={final.field.truncated.tolist()}")
    print("Restricted flat profile; automatic targeting, pacing, skills and GPU validation remain unsupported.")


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    b=sub.add_parser('benchmark'); b.add_argument('--batch',type=int,default=8192); b.add_argument('--repeats',type=int,default=100)
    d=sub.add_parser('restricted-demo'); d.add_argument('--ticks',type=int,default=8)
    f=sub.add_parser('field-demo'); f.add_argument('--ticks',type=int,default=8)
    m=sub.add_parser('moving-field-demo'); m.add_argument('--ticks',type=int,default=8)
    args=parser.parse_args(argv)
    if args.command=='benchmark': benchmark(args.batch,args.repeats)
    if args.command=='restricted-demo': restricted_demo(args.ticks)
    if args.command=='field-demo': field_demo(args.ticks)
    if args.command=='moving-field-demo': moving_field_demo(args.ticks)

if __name__ == '__main__':
    main()
