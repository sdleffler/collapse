use std::cell::RefCell;
use std::collections::HashSet;
use std::fmt;
use std::hash::Hash;
use std::io;

use bit_set::BitSet;

use ndarray::prelude::*;
use ndarray::NdIndex;

use rand::Rng;
use rand::StdRng;

use source::{Source, Source2};


#[derive(Clone)]
pub struct State<P> {
    pub pos: P,
    pub entropy: f64,
    pub cfg: BitSet,
}


pub struct Wave<'a, R, S: ?Sized>
    where S: 'a + Source
{
    source: &'a S,

    states: Array<RefCell<State<S::Dims>>, S::Dims>,
    unobserved: HashSet<S::Dims>,

    dims: S::Dims,
    periodic: S::Periodicity,

    rng: R,
}


impl<'a, R, P: Eq + Hash + Copy> fmt::Debug for Wave<'a, R, Source2<P>> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut grid = String::new();
        for j in 0..self.states.dim().1 {
            for i in 0..self.states.dim().0 {
                grid.push_str(format!(" {:<4.2}", self.states[(i, j)].borrow().entropy).as_str());
            }
            grid.push('\n');
        }
        write!(f, "{}", grid)
    }
}


impl<'a, S: 'a + Source> Wave<'a, StdRng, S>
    where S::Dims: Copy + Eq + Hash + Dimension + NdIndex<Dim = S::Dims>,
          S::Periodicity: Copy
{
    pub fn new(dims: S::Dims,
               periodic: S::Periodicity,
               source: &'a S)
               -> io::Result<Wave<'a, StdRng, S>> {
        let mut unobserved = HashSet::new();
        let states = Array::from_shape_fn(source.wave_dims(dims, periodic), |pos| {
            unobserved.insert(pos);
            RefCell::new(source.initial_state(pos))
        });

        let rng = try!(StdRng::new());

        Ok(Wave {
            source: source,

            states: states,
            unobserved: unobserved,

            dims: dims,
            periodic: periodic,

            rng: rng,
        })
    }
}


impl<'a, R: Rng, S: 'a + Source> Wave<'a, R, S>
    where S::Dims: Copy + Eq + Ord + Hash + Dimension + NdIndex<Dim = S::Dims>,
          S::Periodicity: Copy
{
    pub fn observe(self) -> Option<Self> {
        let Wave { source, states, mut unobserved, dims, periodic, mut rng } = self;

        let observed = {
            let mut iter = unobserved.iter();
            let first: S::Dims = match iter.next() {
                Some(&p) => p,
                None => return None,
            };
            iter.fold(first,
                      |f, &p| if states[p].borrow().entropy < states[f].borrow().entropy {
                          p
                      } else {
                          f
                      })
        };

        unobserved.remove(&observed);

        let states = match source.observe(states, observed, periodic, &mut rng) {
            Some(states) => states,
            None => return None,
        };

        return Some(Wave {
            source: source,
            states: states,
            unobserved: unobserved,
            dims: dims,
            periodic: periodic,
            rng: rng,
        });
    }


    pub fn collapse(mut self) -> Option<Array<S::Pixel, S::Dims>> {
        while !self.is_collapsed() {
            self = match self.observe() {
                Some(wave) => wave,
                None => return None,
            };
        }

        Some(self.source.resolve(self.dims, self.states.view()))
    }


    pub fn resolve(&self) -> Array<S::Pixel, S::Dims> {
        assert!(self.is_collapsed());
        self.source.resolve(self.dims, self.states.view())
    }


    pub fn is_collapsed(&self) -> bool {
        self.unobserved.is_empty()
    }


    pub fn len(&self) -> usize {
        self.states.len()
    }


    pub fn uncollapsed(&self) -> usize {
        self.unobserved.len()
    }


    pub fn constrain(self, pos: S::Dims, val: S::Pixel) -> Option<Self> {
        let Wave { source, states, unobserved, dims, periodic, rng } = self;

        let states = match source.constrain(states, pos, periodic, val) {
            Some(states) => states,
            None => return None,
        };

        return Some(Wave {
            source: source,
            states: states,
            unobserved: unobserved,
            dims: dims,
            periodic: periodic,
            rng: rng,
        });
    }
}


#[cfg(test)]
mod tests {
    extern crate env_logger;
    extern crate image;

    use super::*;

    use std::cell::RefCell;
    use std::collections::HashSet;

    use image::{GenericImage, ImageBuffer};

    use ndarray::prelude::*;

    use rand::{StdRng, SeedableRng};

    use source::*;

    #[test]
    fn should_collapse_rooms() {
        let _ = env_logger::init();

        let img = image::open("resources/Rooms.png").expect("Failed to open source image");
        let src = Source2::from_image(img, (3, 3), (true, true), Symmetry2::all());

        let wave = Wave::new((64, 64), (true, true), &src).expect("IO error");

        let pixels = wave.collapse().expect("Wave contradiction!");

        let buffer = ImageBuffer::from_fn(pixels.dim().0 as u32,
                                          pixels.dim().1 as u32,
                                          |x, y| pixels[(x as usize, y as usize)]);
        buffer.save("output/CollapseTestRooms.png").expect("Error saving buffer");
    }


    #[test]
    fn should_collapse_flowers() {
        let _ = env_logger::init();

        let img = image::open("resources/Flowers.png").expect("Failed to open source image");
        let sky = img.get_pixel(0, 0);
        let gnd = img.get_pixel(0, 21);
        let src = Source2::from_image(img, (3, 3), (true, false), S2_IDENTITY | S2_REFLECT_Y);

        let mut wave = Wave::new((128, 128), (true, false), &src).expect("IO error");
        for i in 0..128 {
            wave = wave.constrain((i, 127), gnd)
                .expect("Constraint error")
                .constrain((i, 0), sky)
                .expect("Constraint error");
        }

        let pixels = wave.collapse().expect("Wave contradiction!");

        let buffer = ImageBuffer::from_fn(pixels.dim().0 as u32,
                                          pixels.dim().1 as u32,
                                          |x, y| pixels[(x as usize, y as usize)]);
        buffer.save("output/CollapseTestFlowers.png").expect("Error saving buffer");
    }


    #[test]
    fn should_collapse_sword() {
        let _ = env_logger::init();

        let img = image::open("resources/DitheringSword.png").expect("Failed to open source image");
        let empty = img.get_pixel(0, 0);
        let src = Source2::from_image(img, (3, 3), (true, true), S2_IDENTITY);

        let mut wave = Wave::new((128, 128), (true, true), &src).expect("IO error");
        for i in 0..128 {
            wave = wave.constrain((i, 127), empty)
                .expect("Constraint error")
                .constrain((i, 0), empty)
                .expect("Constraint error");
        }

        let pixels = wave.collapse().expect("Wave contradiction!");

        let buffer = ImageBuffer::from_fn(pixels.dim().0 as u32,
                                          pixels.dim().1 as u32,
                                          |x, y| pixels[(x as usize, y as usize)]);
        buffer.save("output/CollapseTestSword.png").expect("Error saving buffer");
    }
}
