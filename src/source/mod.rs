use std::cell::RefCell;

use bit_set::BitSet;

use ndarray::prelude::*;

use rand::Rng;

use wave::State;


mod source2d;
mod source3d;


pub use self::source2d::OverlappingSource2;
pub use self::source2d::symmetry::*;


pub trait Source {
    type Dims;
    type Periodicity;
    type Pixel;

    fn wave_dims(&self, Self::Dims, Self::Periodicity) -> Self::Dims;

    fn initial_state(&self, Self::Dims) -> State<Self::Dims>;
    fn entropy(&self, &BitSet) -> f64;

    fn constrain(&self,
                 Array<RefCell<State<Self::Dims>>, Self::Dims>,
                 Self::Dims,
                 Self::Periodicity,
                 Self::Pixel)
                 -> Option<Array<RefCell<State<Self::Dims>>, Self::Dims>>;
    fn observe<R: Rng>(&self,
                       Array<RefCell<State<Self::Dims>>, Self::Dims>,
                       Self::Dims,
                       Self::Periodicity,
                       &mut R)
                       -> Option<Array<RefCell<State<Self::Dims>>, Self::Dims>>;
    fn propagate(&self,
                 Array<RefCell<State<Self::Dims>>, Self::Dims>,
                 Self::Dims,
                 Self::Periodicity)
                 -> Option<Array<RefCell<State<Self::Dims>>, Self::Dims>>;

    fn resolve<'a>(&self,
                   Self::Dims,
                   ArrayView<'a, RefCell<State<Self::Dims>>, Self::Dims>)
                   -> Array<Self::Pixel, Self::Dims>;
}
