use std::cell::RefCell;
use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::Hash;
use std::ops::{Neg, Sub};
use std::vec::Vec;

use bit_set::BitSet;

use bit_vec::BitVec;

use image::{GenericImage, Pixel};

use ndarray::prelude::*;
use ndarray::{Array3, ArrayView3, Ix3};

use vosealias::AliasTable;

use rand::Rng;

use source::Source;
use wave::State;


type RcArray3<A> = RcArray<A, Ix3>;


#[derive(Copy, Clone)]
pub struct Point3(u32, u32, u32);


#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Copy, Clone, Debug)]
struct Pixel3(u32);


#[derive(PartialEq, Eq, Hash, Clone, Copy, Debug)]
pub struct Offset3(isize, isize, isize);


impl Sub for Point3 {
    type Output = Offset3;

    fn sub(self, rhs: Point3) -> Offset3 {
        let x = self.0 as isize - rhs.0 as isize;
        let y = self.1 as isize - rhs.1 as isize;
        let z = self.2 as isize - rhs.2 as isize;

        Offset3(x, y, z)
    }
}


impl Neg for Offset3 {
    type Output = Self;

    fn neg(self) -> Self {
        Offset3(-self.0, -self.1, -self.2)
    }
}


pub struct OverlappingSource3<P> {
    palette: Vec<P>,
    inverse_palette: HashMap<P, Pixel3>,

    samples: Vec<Sample3<Pixel3>>,
    weights: Vec<(usize, f64)>,

    collide: HashMap<Offset3, Vec<BitSet>>,

    n: (usize, usize, usize),
}


pub mod symmetry {
    bitflags! {
        pub flags Symmetry3: u64 {
            const S3_0   = 0b1 << 0,
            const S3_1   = 0b1 << 1,
            const S3_2   = 0b1 << 2,
            const S3_3   = 0b1 << 3,
            const S3_4   = 0b1 << 4,
            const S3_5   = 0b1 << 5,
            const S3_6   = 0b1 << 6,
            const S3_7   = 0b1 << 7,
            const S3_8   = 0b1 << 8,
            const S3_9   = 0b1 << 9,
            const S3_10  = 0b1 << 10,
            const S3_11  = 0b1 << 11,

            const S3_16   = 0b1 << 16,
            const S3_17   = 0b1 << 17,
            const S3_18   = 0b1 << 18,
            const S3_19   = 0b1 << 19,
            const S3_20   = 0b1 << 20,
            const S3_21   = 0b1 << 21,
            const S3_22   = 0b1 << 22,
            const S3_23   = 0b1 << 23,
            const S3_24   = 0b1 << 24,
            const S3_25   = 0b1 << 25,
            const S3_26   = 0b1 << 26,
            const S3_27   = 0b1 << 27,

            const S3_32   = 0b1 << 32,
            const S3_33   = 0b1 << 33,
            const S3_34   = 0b1 << 34,
            const S3_35   = 0b1 << 35,
            const S3_36   = 0b1 << 36,
            const S3_37   = 0b1 << 37,
            const S3_38   = 0b1 << 38,
            const S3_39   = 0b1 << 39,
            const S3_40   = 0b1 << 40,
            const S3_41   = 0b1 << 41,
            const S3_42   = 0b1 << 42,
            const S3_43   = 0b1 << 43,

            const S3_48   = 0b1 << 48,
            const S3_49   = 0b1 << 49,
            const S3_50   = 0b1 << 50,
            const S3_51   = 0b1 << 51,
            const S3_52   = 0b1 << 52,
            const S3_53   = 0b1 << 53,
            const S3_54   = 0b1 << 54,
            const S3_55   = 0b1 << 55,
            const S3_56   = 0b1 << 56,
            const S3_57   = 0b1 << 57,
            const S3_58   = 0b1 << 58,
            const S3_59   = 0b1 << 59,

            const S3_REFLECT_X = 0b1 << (JS3_REFLECT | JS3_SWAP_TETRA | JS3_ROT180_Y),
            const S3_REFLECT_Y = 0b1 << (JS3_REFLECT | JS3_SWAP_TETRA | JS3_ROT180_X),
            const S3_REFLECT_Z = 0b1 << (JS3_REFLECT | JS3_SWAP_TETRA),
        }
    }

    pub const JS3_REFLECT: u64 = 0b100000;
    pub const JS3_SWAP_TETRA: u64 = 0b10000;

    pub const JS3_ROT120_ID: u64 = 0b0000;
    pub const JS3_ROT120_YZX: u64 = 0b0100;
    pub const JS3_ROT120_ZXY: u64 = 0b1000;

    pub const JS3_ROT120_MASK: u64 = 0b1100;

    pub const JS3_ROT180_ID: u64 = 0b00;
    pub const JS3_ROT180_Z: u64 = 0b01;
    pub const JS3_ROT180_X: u64 = 0b10;
    pub const JS3_ROT180_Y: u64 = 0b11;

    pub const JS3_ROT180_MASK: u64 = 0b11;

    impl Symmetry3 {
        #[inline]
        pub fn find_symmetry_dimensions(symmetry: u64,
                                        (mut x, mut y, mut z): (usize, usize, usize))
                                        -> (usize, usize, usize) {
            use std::mem::swap;

            if symmetry & JS3_REFLECT != 0 {
                swap(&mut x, &mut y);
            }

            if symmetry & JS3_SWAP_TETRA != 0 {
                swap(&mut x, &mut y);
            }

            match symmetry & JS3_ROT120_MASK {
                JS3_ROT120_ID => {}
                JS3_ROT120_YZX => {
                    swap(&mut y, &mut z);
                    swap(&mut x, &mut z);
                } // XYZ -> XZY -> YZX
                JS3_ROT120_ZXY => {
                    swap(&mut x, &mut y);
                    swap(&mut x, &mut z);
                } // XYZ -> YXZ -> ZXY
                _ => panic!("Invalid symmetry: {:?}", symmetry),
            }

            (x, y, z)
        }

        #[inline]
        pub fn apply_symmetry(symmetry: u64,
                              (bx, by, bz): (usize, usize, usize),
                              (mut x, mut y, mut z): (usize, usize, usize))
                              -> (usize, usize, usize) {
            use std::mem::swap;

            if symmetry & JS3_REFLECT != 0 {
                swap(&mut x, &mut y);
            }

            if symmetry & JS3_SWAP_TETRA != 0 {
                swap(&mut x, &mut y);
                z = bz - z;
            }

            match symmetry & JS3_ROT120_MASK {
                JS3_ROT120_ID => {}
                JS3_ROT120_YZX => {
                    swap(&mut y, &mut z);
                    swap(&mut x, &mut z);
                } // YZX -> YXZ -> XYZ
                JS3_ROT120_ZXY => {
                    swap(&mut x, &mut y);
                    swap(&mut x, &mut z);
                } // ZXY -> XZY -> XYZ
                _ => panic!("Invalid symmetry: {:?}", symmetry),
            }

            match symmetry & JS3_ROT180_MASK {
                JS3_ROT180_ID => {}
                JS3_ROT180_Z => {
                    x = bx - x;
                    y = by - y;
                }
                JS3_ROT180_X => {
                    y = by - y;
                    z = bz - z;
                }
                JS3_ROT180_Y => {
                    x = bx - x;
                    z = bz - z;
                }
                _ => unreachable!(),
            }

            (x, y, z)
        }


        #[inline]
        pub fn invert_symmetry(symmetry: u64,
                               (bx, by, bz): (usize, usize, usize),
                               (mut x, mut y, mut z): (usize, usize, usize))
                               -> (usize, usize, usize) {
            use std::mem::swap;

            match symmetry & JS3_ROT180_MASK {
                JS3_ROT180_ID => {}
                JS3_ROT180_Z => {
                    y = by - y;
                    x = bx - x;
                }
                JS3_ROT180_X => {
                    z = bz - z;
                    y = by - y;
                }
                JS3_ROT180_Y => {
                    z = bz - z;
                    x = bx - x;
                }
                _ => unreachable!(),
            }

            match symmetry & JS3_ROT120_MASK {
                JS3_ROT120_ID => {}
                JS3_ROT120_YZX => {
                    swap(&mut x, &mut z);
                    swap(&mut y, &mut z);
                } // XYZ -> XZY -> YZX
                JS3_ROT120_ZXY => {
                    swap(&mut x, &mut z);
                    swap(&mut x, &mut y);
                } // XYZ -> YXZ -> ZXY
                _ => panic!("Invalid symmetry: {:?}", symmetry),
            }

            if symmetry & JS3_SWAP_TETRA != 0 {
                z = bz - z;
                swap(&mut x, &mut y);
            }

            if symmetry & JS3_REFLECT != 0 {
                swap(&mut x, &mut y);
            }

            (x, y, z)
        }
    }
}
use self::symmetry::*;


#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Sample3<P>(RcArray3<P>);


impl<P> OverlappingSource3<P>
    where P: Copy
{
    pub fn from_image_stack<I: GenericImage<Pixel = P> + 'static>(imgs: Vec<I>,
                                                                  n: (usize, usize, usize),
                                                                  periodic: (bool, bool, bool),
                                                                  symmetry: Symmetry3)
                                                                  -> Self
        where P: Pixel + Eq + Hash + 'static
    {
        debug!("Generating palette map...");

        let palette_set: HashSet<P> =
            imgs.iter().flat_map(|img| img.pixels()).map(|(_, _, p)| p).collect();
        let palette_map: HashMap<P, Pixel3> = palette_set.into_iter()
            .enumerate()
            .map(|(i, p)| (p, Pixel3(i as u32)))
            .collect();

        debug!("Palette size: {}.", palette_map.len());

        debug!("Stacking images into Array3...");

        let symmetries: Vec<RcArray3<Pixel3>> = {
            let pixels = RcArray3::from_shape_fn(n, |(x, y, z)| {
                palette_map[&imgs[z].get_pixel(x as u32, y as u32)]
            });

            let mut symmetries = Vec::new();

            let mut symm_bits = symmetry.bits();

            let mut symm_n = 0;
            while symm_bits > 0 {
                if symm_bits & 0b1 != 0 {
                    let symm_dims = {
                        let (x, y, z) = Symmetry3::find_symmetry_dimensions(symm_n, n);
                        (if periodic.0 { x } else { x - (n.0 - 1) },
                         if periodic.1 { y } else { y - (n.1 - 1) },
                         if periodic.2 { z } else { z - (n.2 - 1) })
                    };
                    symmetries.push(RcArray3::from_shape_fn(symm_dims, |(x, y, z)| {
                        pixels[{
                            let (x, y, z) =
                                Symmetry3::invert_symmetry(symm_n, symm_dims, (x, y, z));
                            (x % n.0, y % n.1, z % n.2)
                        }]
                    }));
                }
                symm_bits >>= 1;
                symm_n += 1;
            }

            symmetries
        };


        let (samples, weights) = {
            let mut sample_set = HashMap::new();
            for symmetry in symmetries {
                for i in 0..symmetry.dim().0 - (n.0 - 1) {
                    for j in 0..symmetry.dim().1 - (n.1 - 1) {
                        for k in 0..symmetry.dim().2 - (n.2 - 1) {
                            let mut sample = symmetry.to_shared();
                            sample.islice(s![i as isize..(i + n.0) as isize,
                                             j as isize..(j + n.1) as isize,
                                             k as isize..(k + n.2) as isize]);
                            *sample_set.entry(Sample3(sample)).or_insert(0) += 1;
                        }
                    }
                }
            }

            debug!("Converting intermediate sample type into full sample type.");

            let (samples, weight_vec): (Vec<_>, Vec<_>) = sample_set.into_iter().unzip();

            let weights: Vec<_> = weight_vec.into_iter()
                .enumerate()
                .map(|(i, x)| (i, x as f64))
                .collect();

            (samples, weights)
        };

        // for (s, &Sample2(ref sample)) in samples.iter().enumerate() {
        //     let mut string = String::new();
        //     for j in 0..sample.dim().1 {
        //         for i in 0..sample.dim().0 {
        //             string.push_str(format!("{} ", sample[(i, j)].0).as_str());
        //         }
        //         string.push('\n');
        //     }
        //
        //     debug!("Sample {}, weighted {}: \n{}", s, weights[s].1, string);
        //     debug_assert_eq!(sample.dim(), (3, 3));
        // }

        debug!("Generating collision map. {} samples to collide.",
               samples.len());

        let collide = {
            let mut collide = HashMap::new();

            let n = (n.0 as isize, n.1 as isize, n.2 as isize);

            let check_at_offset = |dx, dy, dz, lx, ly, lz, rx, ry, rz| {
                let mut bitsets = Vec::new();
                for &Sample3(ref l) in samples.iter() {
                    let mut bs = BitSet::with_capacity(samples.len());
                    'rcheck: for (s, &Sample3(ref r)) in samples.iter().enumerate() {
                        for i in 0..dx {
                            for j in 0..dy {
                                for k in 0..dz {
                                    let p_l = l[((lx + i) as usize,
                                                 (ly + j) as usize,
                                                 (lz + k) as usize)];
                                    let p_r = r[((rx + i) as usize,
                                                 (ry + j) as usize,
                                                 (rz + k) as usize)];
                                    if p_l != p_r {
                                        continue 'rcheck;
                                    }
                                }
                            }
                        }
                        bs.insert(s);
                    }
                    bitsets.push(bs);
                }
                bitsets
            };

            for dx in 0..n.0 {
                for dy in 0..n.1 {
                    for dz in 0..n.2 {
                        collide.insert(Offset3(dx, dy, dz),
                                       check_at_offset(n.0 - dx,
                                                       n.1 - dy,
                                                       n.2 - dz,
                                                       dx,
                                                       dy,
                                                       dz,
                                                       0,
                                                       0,
                                                       0));
                        collide.insert(Offset3(-dx, dy, dz),
                                       check_at_offset(n.0 - dx,
                                                       n.1 - dy,
                                                       n.2 - dz,
                                                       0,
                                                       dy,
                                                       dz,
                                                       dx,
                                                       0,
                                                       0));
                        collide.insert(Offset3(dx, -dy, dz),
                                       check_at_offset(n.0 - dx,
                                                       n.1 - dy,
                                                       n.2 - dz,
                                                       dx,
                                                       0,
                                                       dz,
                                                       0,
                                                       dy,
                                                       0));
                        collide.insert(Offset3(-dx, -dy, dz),
                                       check_at_offset(n.0 - dx,
                                                       n.1 - dy,
                                                       n.2 - dz,
                                                       0,
                                                       0,
                                                       dz,
                                                       dx,
                                                       dy,
                                                       0));
                        collide.insert(Offset3(dx, dy, -dz),
                                       check_at_offset(n.0 - dx,
                                                       n.1 - dy,
                                                       n.2 - dz,
                                                       dx,
                                                       dy,
                                                       0,
                                                       0,
                                                       0,
                                                       dz));
                        collide.insert(Offset3(-dx, dy, -dz),
                                       check_at_offset(n.0 - dx,
                                                       n.1 - dy,
                                                       n.2 - dz,
                                                       0,
                                                       dy,
                                                       0,
                                                       dx,
                                                       0,
                                                       dz));
                        collide.insert(Offset3(dx, -dy, -dz),
                                       check_at_offset(n.0 - dx,
                                                       n.1 - dy,
                                                       n.2 - dz,
                                                       dx,
                                                       0,
                                                       0,
                                                       0,
                                                       dy,
                                                       dz));
                        collide.insert(Offset3(-dx, -dy, -dz),
                                       check_at_offset(n.0 - dx,
                                                       n.1 - dy,
                                                       n.2 - dz,
                                                       0,
                                                       0,
                                                       0,
                                                       dx,
                                                       dy,
                                                       dz));
                    }
                }
            }

            collide
        };

        debug!("Done.");

        OverlappingSource3 {
            palette: {
                let mut vec = palette_map.iter().map(|(&p, &px)| (p, px)).collect::<Vec<_>>();
                vec.sort_by_key(|x| x.1);
                vec.into_iter()
                    .map(|x| x.0)
                    .collect()
            },
            inverse_palette: palette_map,
            samples: samples,
            weights: weights,
            collide: collide,
            n: n,
        }
    }


    fn pick_sample<R: Rng>(&self, cfg: &mut BitSet, rng: &mut R) {
        let table: AliasTable<_, _> = cfg.iter().map(|i| self.weights[i]).collect();
        let chosen = table.pick(rng);
        cfg.clear();
        cfg.insert(*chosen);
    }
}


impl<P> Source for OverlappingSource3<P>
    where P: Eq + Hash + Copy
{
    type Dims = Ix3;
    type Periodicity = (bool, bool, bool);
    type Pixel = P;


    fn wave_dims(&self,
                 dims: (usize, usize, usize),
                 periodic: (bool, bool, bool))
                 -> (usize, usize, usize) {
        (if periodic.0 {
            dims.0
        } else {
            dims.0 - (self.n.0 - 1)
        },
         if periodic.1 {
            dims.1
        } else {
            dims.1 - (self.n.1 - 1)
        },
         if periodic.2 {
            dims.2
        } else {
            dims.2 - (self.n.2 - 1)
        })
    }


    fn initial_state(&self, pos: Ix3) -> State<Ix3> {
        let cfg = BitSet::from_bit_vec(BitVec::from_elem(self.samples.len(), true));
        let entropy = self.entropy(&cfg);
        State {
            pos: pos,
            entropy: entropy,
            cfg: cfg,
        }
    }


    fn constrain(&self,
                 states: Array3<RefCell<State<Ix3>>>,
                 pos: Ix3,
                 periodic: Self::Periodicity,
                 val: P)
                 -> Option<Array3<RefCell<State<Ix3>>>> {
        let n = (self.n.0 as isize, self.n.1 as isize, self.n.2 as isize);
        let pid = self.inverse_palette[&val];

        for i in 0..n.0 {
            let dim_adj_x = (states.dim().0 as isize - i) as usize;

            for j in 0..n.1 {
                let dim_adj_y = (states.dim().1 as isize - j) as usize;

                for k in 0..n.2 {
                    let dim_adj_z = (states.dim().2 as isize - k) as usize;

                    let subject_pos = ((pos.0 + dim_adj_x) % states.dim().0,
                                       (pos.1 + dim_adj_y) % states.dim().1,
                                       (pos.2 + dim_adj_z) % states.dim().2);

                    let mut subject = states[subject_pos].borrow_mut();

                    if !periodic.0 && (pos.0 as isize - subject.pos.0 as isize).abs() >= n.0 {
                        continue;
                    }

                    if !periodic.1 && (pos.1 as isize - subject.pos.1 as isize).abs() >= n.1 {
                        continue;
                    }

                    if !periodic.2 && (pos.2 as isize - subject.pos.2 as isize).abs() >= n.2 {
                        continue;
                    }

                    subject.cfg = subject.cfg
                        .iter()
                        .filter(|&idx| {
                            self.samples[idx].0[(i as usize, j as usize, k as usize)] == pid
                        })
                        .collect();

                    subject.entropy = self.entropy(&subject.cfg);

                    if !(subject.entropy >= 0.0) {
                        debug!("Destroyed wave position {:?}'s hopes and dreams.",
                               subject_pos);
                    }
                }
            }
        }

        self.propagate(states, pos, periodic)
    }


    fn propagate(&self,
                 states: Array3<RefCell<State<Ix3>>>,
                 observe: Ix3,
                 periodic: Self::Periodicity)
                 -> Option<Array3<RefCell<State<Ix3>>>> {
        let n = (self.n.0 as isize, self.n.1 as isize, self.n.2 as isize);

        let mut queue = VecDeque::new();
        queue.push_back(observe);

        while let Some(focus) = queue.pop_front() {
            let mut focus = match states.get(focus) {
                    Some(state) => state,
                    None => continue,
                }
                .borrow_mut();

            focus.entropy = self.entropy(&focus.cfg);

            if !(focus.entropy >= 0.0) {
                return None;
            }

            let mut focus_dirty = false;
            {
                for i in -n.0 + 1..n.0 {
                    let dim_adj_x = (states.dim().0 as isize + i) as usize;

                    for j in -n.1 + 1..n.1 {
                        let dim_adj_y = (states.dim().1 as isize + j) as usize;

                        for k in -n.2 + 1..n.2 {
                            if i == 0 && j == 0 && k == 0 {
                                continue;
                            }

                            let dim_adj_z = (states.dim().2 as isize + k) as usize;

                            let subject_pos = ((focus.pos.0 + dim_adj_x) % states.dim().0,
                                               (focus.pos.1 + dim_adj_y) % states.dim().1,
                                               (focus.pos.2 + dim_adj_z) % states.dim().2);

                            let mut subject = states[subject_pos].borrow_mut();

                            if !periodic.0 &&
                               (focus.pos.0 as isize - subject.pos.0 as isize).abs() >= n.0 {
                                continue;
                            }

                            if !periodic.1 &&
                               (focus.pos.1 as isize - subject.pos.1 as isize).abs() >= n.1 {
                                continue;
                            }

                            if !periodic.2 &&
                               (focus.pos.2 as isize - subject.pos.2 as isize).abs() >= n.2 {
                                continue;
                            }

                            let mut subject_dirty = false;

                            let mut focus_allowed = BitSet::new();
                            let mut subject_allowed = BitSet::new();

                            loop {
                                focus_allowed.clear();
                                subject_allowed.clear();

                                for focus_cfg in focus.cfg.iter() {
                                    subject_allowed.union_with(&self.collide[&Offset3(i, j, k)][focus_cfg]);
                                }

                                for subject_cfg in subject.cfg.iter() {
                                    focus_allowed.union_with(&self.collide[&Offset3(-i, -j, -k)][subject_cfg]);
                                }

                                let focus_len = focus.cfg.len();
                                let subject_len = subject.cfg.len();

                                focus.cfg.intersect_with(&focus_allowed);
                                subject.cfg.intersect_with(&subject_allowed);

                                let focus_modified = focus_len > focus.cfg.len();
                                let subject_modified = subject_len > subject.cfg.len();

                                if focus_modified {
                                    focus_dirty = true;
                                }

                                if subject_modified {
                                    subject_dirty = true;
                                }

                                if !(focus_modified || subject_modified) {
                                    break;
                                }
                            }

                            if subject_dirty {
                                queue.push_back(subject.pos);
                            }
                        }
                    }
                }
            }

            if focus_dirty {
                queue.push_back(focus.pos);
            }
        }

        return Some(states);
    }


    fn observe<R: Rng>(&self,
                       mut states: Array3<RefCell<State<Ix3>>>,
                       observe: Ix3,
                       periodic: Self::Periodicity,
                       rng: &mut R)
                       -> Option<Array3<RefCell<State<Ix3>>>> {
        self.pick_sample(&mut states[observe].borrow_mut().cfg, rng);
        self.propagate(states, observe, periodic)
    }


    fn entropy(&self, cfg: &BitSet) -> f64 {
        use std::f64;

        if cfg.is_empty() {
            return f64::NAN;
        }
        let weights: Vec<f64> = cfg.iter().map(|i| self.weights[i].1).collect();
        let sum: f64 = weights.iter().sum();
        weights.into_iter()
            .map(|w| {
                let p = w / sum;
                -(p * p.ln())
            })
            .sum()
    }


    fn resolve<'a>(&self, dim: Self::Dims, wave: ArrayView3<'a, RefCell<State<Ix3>>>) -> Array3<P> {
        Array::from_shape_fn(dim, |(x, y, z)| {
            let (wx, dx) = if x < wave.dim().0 {
                (x, 0)
            } else {
                (wave.dim().0 - 1, x - (wave.dim().0 - 1))
            };
            let (wy, dy) = if y < wave.dim().1 {
                (y, 0)
            } else {
                (wave.dim().1 - 1, y - (wave.dim().1 - 1))
            };
            let (wz, dz) = if z < wave.dim().2 {
                (z, 0)
            } else {
                (wave.dim().2 - 1, z - (wave.dim().2 - 1))
            };
            self.palette[self.samples[wave[(wx, wy, wz)]
                        .borrow()
                        .cfg
                        .iter()
                        .next()
                        .unwrap()]
                    .0[(dx, dy, dz)]
                .0 as usize]
        })
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use super::symmetry::*;

    #[test]
    fn schmidt_symmetries() {
        // Ensure that black magic symmetry code *actually works.*

        assert_eq!(Symmetry3::apply_symmetry(JS3_REFLECT | JS3_SWAP_TETRA | JS3_ROT180_Y,
                                             (1, 1, 1),
                                             (0, 0, 0)),
                   (1, 0, 0));
        assert_eq!(Symmetry3::apply_symmetry(JS3_REFLECT | JS3_SWAP_TETRA | JS3_ROT180_X,
                                             (1, 1, 1),
                                             (0, 0, 0)),
                   (0, 1, 0));
        assert_eq!(Symmetry3::apply_symmetry(JS3_REFLECT | JS3_SWAP_TETRA, (1, 1, 1), (0, 0, 0)),
                   (0, 0, 1));
        assert_eq!(Symmetry3::apply_symmetry(JS3_REFLECT | JS3_SWAP_TETRA | JS3_ROT180_Y,
                                             (1, 1, 1),
                                             (1, 0, 0)),
                   (0, 0, 0));
        assert_eq!(Symmetry3::apply_symmetry(JS3_REFLECT | JS3_SWAP_TETRA | JS3_ROT180_X,
                                             (1, 1, 1),
                                             (0, 1, 0)),
                   (0, 0, 0));
        assert_eq!(Symmetry3::apply_symmetry(JS3_REFLECT | JS3_SWAP_TETRA, (1, 1, 1), (0, 0, 1)),
                   (0, 0, 0));
        assert_eq!(Symmetry3::apply_symmetry(JS3_REFLECT | JS3_SWAP_TETRA | JS3_ROT180_Y,
                                             (1, 1, 1),
                                             (0, 0, 0)),
                   (1, 0, 0));
        assert_eq!(Symmetry3::apply_symmetry(JS3_REFLECT | JS3_SWAP_TETRA | JS3_ROT180_X,
                                             (1, 1, 1),
                                             (0, 0, 0)),
                   (0, 1, 0));
        assert_eq!(Symmetry3::apply_symmetry(JS3_REFLECT | JS3_SWAP_TETRA, (1, 1, 1), (0, 0, 0)),
                   (0, 0, 1));

        for x in 0..2 {
            for y in 0..2 {
                for z in 0..2 {
                    for &s in [0u64, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 16, 17, 18, 19, 20, 21,
                               22, 23, 24, 25, 26, 27, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
                               42, 43, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59]
                        .into_iter() {
                        assert_eq!(Symmetry3::invert_symmetry(s,
                                                              (1, 1, 1),
                                                              Symmetry3::apply_symmetry(s,
                                                                                        (1,
                                                                                         1,
                                                                                         1),
                                                                                        (x,
                                                                                         y,
                                                                                         z))),
                                   (x, y, z));
                    }
                }
            }
        }
    }
}
