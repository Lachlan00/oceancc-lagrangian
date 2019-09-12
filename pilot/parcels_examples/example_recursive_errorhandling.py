from parcels import FieldSet, ParticleSet, JITParticle, ScipyParticle
from parcels import random, ErrorCode
import numpy as np
import pytest

ptype = {'scipy': ScipyParticle, 'jit': JITParticle}


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_recursive_errorhandling(mode, xdim=2, ydim=2):
    """Example script to show how recursaive error handling can work.

    In this example, a set of Particles is started at Longitude 0.5.
    These are run through a Kernel that throws an error if the
    Longitude is smaller than 0.7.
    The error Kernel then draws a new random number between 0 and 1

    Importantly, the 'normal' Kernel and Error Kernel keep iterating
    until a particle does have a longitude larger than 0.7.

    This behaviour can be useful if particles need to be 'pushed out'
    from e.g. land. Note however that current under-the-hood
    implementation is not extremely efficient, so code could be slow."""

    dimensions = {'lon': np.linspace(0., 1., xdim, dtype=np.float32),
                  'lat': np.linspace(0., 1., ydim, dtype=np.float32)}
    data = {'U': np.zeros((ydim, xdim), dtype=np.float32),
            'V': np.zeros((ydim, xdim), dtype=np.float32)}
    fieldset = FieldSet.from_data(data, dimensions, mesh='flat')

    # Set minimum value for valid longitudes (i.e. all longitudes < minlon are 'land')
    fieldset.add_constant('minlon', 0.7)

    # create a ParticleSet with all particles starting at centre of Field
    pset = ParticleSet.from_line(fieldset=fieldset, pclass=ptype[mode],
                                 start=(0.5, 0.5), finish=(0.5, 0.5), size=10)

    def TestLon(particle, fieldset, time):
        """Kernel to check whether a longitude is larger than fieldset.minlon.
        If not, the Kernel throws an error"""
        if particle.lon <= fieldset.minlon:
            return ErrorCode.Error

    def Error_RandomiseLon(particle, fieldset, time):
        """Error handling kernel that draws a new longitude.
        Note that this new longitude can be smaller than fieldset.minlon"""
        particle.lon = random.uniform(0., 1.)

    random.seed(123456)

    # The .execute below is only run for one timestep. Yet the
    # recovery={ErrorCode.Error: Error_RandomiseLon} assures Parcels keeps
    # attempting to move all particles beyond 0.7 longitude
    pset.execute(pset.Kernel(TestLon), runtime=1, dt=1,
                 recovery={ErrorCode.Error: Error_RandomiseLon})

    assert (np.array([p.lon for p in pset]) > fieldset.minlon).all()
