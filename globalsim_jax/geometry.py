"""Captured-course geometry subset ported from Rust retail_geometry.rs.

Only bundled course 10101 is currently shipped; overrun positions are rejected.
"""
import json
from dataclasses import dataclass
from importlib.resources import files
import jax
import jax.numpy as jnp
import numpy as np
from .precision import require_x64


def _concrete(value):
    return not isinstance(value, jax.core.Tracer)


def _finite_scalar(value, name):
    if _concrete(value) and not np.isfinite(float(value)):
        raise ValueError(f"geometry {name} must be finite")


@dataclass(frozen=True)
class CourseGeometry:
    course_id: int
    distance: float
    x: jax.Array
    z: jax.Array
    rotation: jax.Array
    rotation_fix: jax.Array
    distance_to_world: jax.Array
    lane_max: jax.Array

    @classmethod
    def load(cls, course_id: int, distance: float):
        require_x64()
        if course_id != 10101 or distance != 1200.:
            raise ValueError("only captured course 10101 at 1200m is bundled; overrun/other courses unsupported")
        raw = json.loads(files("globalsim_jax").joinpath("course_10101.json").read_text())
        a = raw["animation"]
        identity = raw.get("course_identity", {})
        if raw.get("course_id") != course_id or identity.get("id") != course_id or identity.get("distance") != distance:
            raise ValueError("captured geometry identity does not match requested course")
        if any(len(a.get(key, ())) != 1001 for key in ("x", "y", "z", "rotation")):
            raise ValueError("captured animation must contain 1001 aligned samples")
        rotation = np.asarray(a["rotation"], np.float32)
        rotation_fix = np.asarray(raw.get("rotation_fix", ()), np.float32)
        tables = [np.asarray(a[key], np.float32) for key in ("x", "y", "z")] + [rotation, rotation_fix]
        if rotation.shape != (1001, 4) or rotation_fix.shape != (4,) or any(not np.all(np.isfinite(table)) for table in tables):
            raise ValueError("captured geometry tables have invalid shape or non-finite values")
        x, y, z = (jnp.asarray(a[k], jnp.float32) for k in ("x", "y", "z"))
        def at(d):
            sample = (jnp.float64(d) / jnp.float64(distance) * 1000.).astype(jnp.float32)
            i = jnp.minimum(sample.astype(jnp.int32), 1000)
            j = jnp.minimum(i + 1, 1000)
            t = sample - i.astype(jnp.float32)
            return tuple((v[i].astype(jnp.float64) + (v[j] - v[i]).astype(jnp.float64) * t.astype(jnp.float64)).astype(jnp.float32) for v in (x, y, z))
        p, q = at(0.), at(1.)
        # Rust subtracts samples in f32, then widens the three differences.
        dx, dy, dz = ((p[0] - q[0]).astype(jnp.float64), (p[1] - q[1]).astype(jnp.float64), (p[2] - q[2]).astype(jnp.float64))
        scale = jnp.sqrt((dy * dy + dx * dx) + dz * dz).astype(jnp.float32)
        lane_max = np.float32(identity.get("lane_max", np.nan))
        if not np.isfinite(lane_max):
            raise ValueError("captured geometry lane_max must be finite")
        return cls(course_id,distance,x,z,jnp.asarray(rotation,jnp.float32),jnp.asarray(rotation_fix,jnp.float32),scale,lane_max*jnp.float32(.0001))

    def position(self, distance, lane):
        require_x64()
        _finite_scalar(distance, "distance")
        _finite_scalar(lane, "lane")
        if _concrete(distance) and not 0 <= float(distance) <= self.distance:
            raise ValueError("geometry requires a nonnegative in-course distance")
        d,lane=jnp.asarray(distance,jnp.float32),jnp.asarray(lane,jnp.float32)
        sample=(d.astype(jnp.float64)/jnp.float64(self.distance)*1000.).astype(jnp.float32); i=jnp.minimum(sample.astype(jnp.int32),1000); j=jnp.minimum(i+1,1000); t=sample-i.astype(jnp.float32)
        # Segment deltas are widened-sample f64 subtraction in Rust.
        x=(self.x[i].astype(jnp.float64)+(self.x[j].astype(jnp.float64)-self.x[i].astype(jnp.float64))*t.astype(jnp.float64)).astype(jnp.float32); z=(self.z[i].astype(jnp.float64)+(self.z[j].astype(jnp.float64)-self.z[i].astype(jnp.float64))*t.astype(jnp.float64)).astype(jnp.float32)
        a=self.rotation[i]; b=self.rotation[j]
        dot=((a[0]*b[0]+a[1]*b[1])+a[2]*b[2])+a[3]*b[3]
        b=jnp.where(dot<0,-b,b); q=jax.lax.optimization_barrier(a+(b-a)*t)
        q2 = jax.lax.optimization_barrier(jnp.stack((q[0] * q[0], q[1] * q[1], q[2] * q[2], q[3] * q[3])))
        norm=jnp.sqrt(((q2[0]+q2[1]+q2[2])+q2[3]).astype(jnp.float64)).astype(jnp.float32)
        # Rust's terminal sample uses its raw quaternion, no normalization.
        normalized = jax.lax.optimization_barrier(jnp.stack((q[0]/norm, q[1]/norm, q[2]/norm, q[3]/norm)))
        q=jax.lax.optimization_barrier(jnp.where(i==1000,a,normalized))
        ax,ay,az,aw=q; bx,by,bz,bw=self.rotation_fix
        qx=((aw*bx+ax*bw)+ay*bz)-az*by; qy=((aw*by+ay*bw)+az*bx)-ax*bz; qz=((aw*bz+az*bw)+ax*by)-ay*bx; qw=((aw*bw-ax*bx)-ay*by)-az*bz
        qx,qy,qz,qw = jax.lax.optimization_barrier(jnp.stack((qx,qy,qz,qw)))
        n2,n3=qy*2,qz*2; n5,n6,n8,n11=qy*n2,qz*n3,qx*n3,qw*n2; offset=(lane*11.25).astype(jnp.float64)
        return jnp.stack((x+((1.-(n5.astype(jnp.float64)+n6.astype(jnp.float64)))*offset).astype(jnp.float32),z+((n8.astype(jnp.float64)-n11.astype(jnp.float64))*offset).astype(jnp.float32)))

    def advance(self, distance, lane, movement, previous_world):
        require_x64()
        _finite_scalar(movement, "movement")
        if _concrete(movement) and float(movement) < 0:
            raise ValueError("geometry movement must be finite and nonnegative")
        if _concrete(previous_world):
            previous = np.asarray(previous_world, np.float32)
            if previous.shape != (2,) or not np.all(np.isfinite(previous)):
                raise ValueError("geometry previous_world must be a finite length-2 vector")
        current=self.position(distance,lane); delta=current-jnp.asarray(previous_world,jnp.float32); norm=jnp.sqrt(jnp.sum(delta.astype(jnp.float64)**2)).astype(jnp.float32); allowed=jnp.asarray(movement,jnp.float32)*self.distance_to_world; move=jnp.where(norm>allowed,(jnp.asarray(movement,jnp.float32).astype(jnp.float64)*(allowed.astype(jnp.float64)/norm.astype(jnp.float64))).astype(jnp.float32),movement)
        return jnp.asarray(distance,jnp.float32)+move,current
