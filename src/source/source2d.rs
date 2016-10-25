use std::cell::RefCell;
use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::Hash;
use std::ops::{Neg, Sub};
use std::vec::Vec;

use bit_set::BitSet;

use bit_vec::BitVec;

use image::{GenericImage, ImageBuffer, Pixel};
use image::imageops::{flip_horizontal, flip_vertical, rotate90, rotate180, rotate270};

use ndarray::prelude::*;
use ndarray::{Array2, ArrayView2, Ix2};

use pbr::{PbIter, ProgressBar};

use vosealias::AliasTable;

use rand::Rng;

use source::Source;
use wave::State;


type RcArray2<A> = RcArray<A, Ix2>;


#[derive(Copy, Clone)]
pub struct Point2(u32, u32);


#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Copy, Clone, Debug)]
struct Pixel2(u32);


#[derive(PartialEq, Eq, Hash, Clone, Copy, Debug)]
pub struct Offset2(isize, isize);


impl Sub for Point2 {
    type Output = Offset2;

    fn sub(self, rhs: Point2) -> Offset2 {
        let x = self.0 as isize - rhs.0 as isize;
        let y = self.1 as isize - rhs.1 as isize;

        Offset2(x, y)
    }
}


impl Neg for Offset2 {
    type Output = Self;

    fn neg(self) -> Self {
        Offset2(-self.0, -self.1)
    }
}


pub struct Source2<P> {
    palette: Vec<P>,
    inverse_palette: HashMap<P, Pixel2>,

    samples: Vec<Sample2<Pixel2>>,
    weights: Vec<(usize, f64)>,

    collide: HashMap<Offset2, Vec<BitSet>>,

    n: (usize, usize),
}


pub mod symmetry {
    bitflags! {
        pub flags Symmetry2: u8 {
            const S2_IDENTITY        = 0b00000001,
            const S2_ROTATE_90       = 0b00000010,
            const S2_ROTATE_180      = 0b00000100,
            const S2_ROTATE_270      = 0b00001000,
            const S2_REFLECT_Y       = 0b00010000,
            const S2_REFLECT_X       = 0b00100000,
            const S2_REFLECT_Y_ROT90 = 0b01000000,
            const S2_REFLECT_X_ROT90 = 0b10000000,
        }
    }
}
use self::symmetry::*;


#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Sample2<P>(RcArray2<P>);


impl<P> Source2<P>
    where P: Copy
{
    pub fn from_image<I: GenericImage<Pixel = P> + 'static>(img: &I,
                                                            n: (usize, usize),
                                                            periodic: (bool, bool),
                                                            symmetry: Symmetry2)
                                                            -> Self
        where P: Pixel + Eq + Hash + 'static
    {
        debug!("Generating palette map...");

        let palette_set: HashSet<P> = img.pixels().map(|(_, _, p)| p).collect();
        let palette_map: HashMap<P, Pixel2> = palette_set.into_iter()
            .enumerate()
            .map(|(i, p)| (p, Pixel2(i as u32)))
            .collect();

        debug!("Palette size: {}.", palette_map.len());

        debug!("Allocating ImageBuffer and copying source image...");

        let mut buf = ImageBuffer::<P, _>::new(img.width(), img.height());
        buf.copy_from(img, 0, 0);

        let sample_buf_size: usize = (img.width() * img.height()) as usize;

        debug!("Generating symmetry buffers...");

        let symmetries: Vec<RcArray2<Pixel2>> = {
            let symm_bufs = {
                let mut symm_bufs = Vec::new();

                if symmetry.contains(S2_IDENTITY) {
                    symm_bufs.push(buf.clone())
                }
                if symmetry.contains(S2_ROTATE_90) {
                    symm_bufs.push(rotate90(&buf))
                }
                if symmetry.contains(S2_ROTATE_180) {
                    symm_bufs.push(rotate180(&buf))
                }
                if symmetry.contains(S2_ROTATE_270) {
                    symm_bufs.push(rotate270(&buf))
                }
                if symmetry.contains(S2_REFLECT_X) {
                    symm_bufs.push(flip_horizontal(&buf))
                }
                if symmetry.contains(S2_REFLECT_Y) {
                    symm_bufs.push(flip_vertical(&buf))
                }
                if symmetry.contains(S2_REFLECT_Y_ROT90) {
                    symm_bufs.push(flip_vertical(&rotate90(&buf)))
                }
                if symmetry.contains(S2_REFLECT_X_ROT90) {
                    symm_bufs.push(flip_horizontal(&rotate90(&buf)))
                }

                symm_bufs
            };

            debug!("Allocating and filling symmetry 2d-arrays...");

            symm_bufs.into_iter()
                .map(|symm| {
                    RcArray::from_shape_fn((symm.width() as usize +
                                            if periodic.0 { n.0 - 1 } else { 0 },
                                            symm.height() as usize +
                                            if periodic.1 { n.1 - 1 } else { 0 }),
                                           |(x, y)| {
                                               palette_map[symm.get_pixel(x as u32 % symm.width(),
                                                                          y as u32 % symm.height())]
                                           })
                })
                .collect()
        };

        let mut symm0str = String::new();
        for j in 0..symmetries[0].dim().1 {
            for i in 0..symmetries[0].dim().0 {
                symm0str.push_str(format!("{} ", symmetries[0][(i, j)].0).as_str());
            }
            symm0str.push('\n');
        }

        debug!("Symmetry 1: \n{}", symm0str);

        debug!("Generating and deduplicating samples... 8 symmetries, {} samples per symmetry: \
                {} samples undeduplicated.",
               sample_buf_size,
               8 * sample_buf_size);


        let (samples, weights) = {
            let mut sample_set = HashMap::new();
            for (s, symmetry) in symmetries.iter().enumerate() {
                debug!("Processing symmetry {}.", s);
                for i in 0..symmetry.dim().0 - (n.0 - 1) {
                    for j in 0..symmetry.dim().1 - (n.1 - 1) {
                        let mut sample = symmetry.to_shared();
                        sample.islice(s![i as isize..(i + n.0 as usize) as isize,
                                         j as isize..(j + n.1 as usize) as isize]);
                        *sample_set.entry(Sample2(sample)).or_insert(0) += 1;
                    }
                }
            }

            debug!("Converting intermediate sample type into full sample type.");

            let (sample_vec, weight_vec): (Vec<_>, Vec<_>) = sample_set.into_iter().unzip();

            let weights: Vec<_> = weight_vec.into_iter()
                .enumerate()
                .map(|(i, x)| (i, x as f64))
                .collect();

            (sample_vec.clone(), weights)
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

            let n = (n.0 as isize, n.1 as isize);

            let check_at_offset = |dx, dy, lx, ly, rx, ry| {
                let mut bitsets = Vec::new();
                for &Sample2(ref l) in samples.iter() {
                    let mut bs = BitSet::with_capacity(samples.len());
                    'rcheck: for (s, &Sample2(ref r)) in samples.iter().enumerate() {
                        for i in 0..dx {
                            for j in 0..dy {
                                let p_l = l[((lx + i) as usize, (ly + j) as usize)];
                                let p_r = r[((rx + i) as usize, (ry + j) as usize)];
                                if p_l != p_r {
                                    continue 'rcheck;
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
                    collide.insert(Offset2(dx, dy),
                                   check_at_offset(n.0 - dx, n.1 - dy, dx, dy, 0, 0));
                    collide.insert(Offset2(-dx, dy),
                                   check_at_offset(n.0 - dx, n.1 - dy, 0, dy, dx, 0));
                    collide.insert(Offset2(dx, -dy),
                                   check_at_offset(n.0 - dx, n.1 - dy, dx, 0, 0, dy));
                    collide.insert(Offset2(-dx, -dy),
                                   check_at_offset(n.0 - dx, n.1 - dy, 0, 0, dx, dy));
                }
            }

            collide
        };

        debug!("Done.");

        Source2 {
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


    pub fn from_image_cli<I: GenericImage<Pixel = P> + 'static>(img: &I,
                                                                n: (usize, usize),
                                                                periodic: (bool, bool),
                                                                symmetry: Symmetry2)
                                                                -> Self
        where P: Pixel + Eq + Hash + 'static
    {
        let mut progress = ProgressBar::new(2);
        progress.message("Deduplicating palette...");
        progress.tick();

        let palette_set: HashSet<P> = img.pixels().map(|(_, _, p)| p).collect();
        progress.message("Building palette map...");
        progress.inc();

        let palette_map: HashMap<P, Pixel2> = PbIter::new(palette_set.into_iter())
            .enumerate()
            .map(|(i, p)| (p, Pixel2(i as u32)))
            .collect();
        progress.inc();

        let mut progress = ProgressBar::new(symmetry.bits().count_ones() as u64 * 2 + 1);
        progress.message("Copying image into buffer...");
        progress.tick();

        let mut buf = ImageBuffer::<P, _>::new(img.width(), img.height());
        buf.copy_from(img, 0, 0);

        progress.message("Processing symmetries...");
        progress.inc();

        let symmetries: Vec<RcArray2<Pixel2>> = {
            let symm_bufs = {
                let mut symm_bufs = Vec::new();

                if symmetry.contains(S2_ROTATE_90) {
                    symm_bufs.push(rotate90(&buf));
                    progress.inc();
                }
                if symmetry.contains(S2_ROTATE_180) {
                    symm_bufs.push(rotate180(&buf));
                    progress.inc();
                }
                if symmetry.contains(S2_ROTATE_270) {
                    symm_bufs.push(rotate270(&buf));
                    progress.inc();
                }
                if symmetry.contains(S2_REFLECT_X) {
                    symm_bufs.push(flip_horizontal(&buf));
                    progress.inc();
                }
                if symmetry.contains(S2_REFLECT_Y) {
                    symm_bufs.push(flip_vertical(&buf));
                    progress.inc();
                }
                if symmetry.contains(S2_REFLECT_Y_ROT90) {
                    symm_bufs.push(flip_vertical(&rotate90(&buf)));
                    progress.inc();
                }
                if symmetry.contains(S2_REFLECT_X_ROT90) {
                    symm_bufs.push(flip_horizontal(&rotate90(&buf)));
                    progress.inc();
                }
                if symmetry.contains(S2_IDENTITY) {
                    symm_bufs.push(buf);
                    progress.inc();
                }

                symm_bufs
            };

            progress.message("Copying into array...");
            progress.tick();

            symm_bufs.into_iter()
                .map(|symm| {
                    let array = RcArray::from_shape_fn((symm.width() as usize +
                                            if periodic.0 { n.0 - 1 } else { 0 },
                                            symm.height() as usize +
                                            if periodic.1 { n.1 - 1 } else { 0 }),
                                           |(x, y)| {
                                               palette_map[symm.get_pixel(x as u32 % symm.width(),
                                                                          y as u32 % symm.height())]
                                           });
                    progress.inc();
                    array
                })
                .collect()
        };

        let mut progress = ProgressBar::new(symmetries.len() as u64);

        let (samples, weights) = {
            let mut sample_set = HashMap::new();

            progress.message("Sampling...");
            progress.tick();

            for symmetry in symmetries.iter() {
                for i in 0..symmetry.dim().0 - (n.0 - 1) {
                    for j in 0..symmetry.dim().1 - (n.1 - 1) {
                        let mut sample = symmetry.to_shared();
                        sample.islice(s![i as isize..(i + n.0 as usize) as isize,
                                         j as isize..(j + n.1 as usize) as isize]);
                        *sample_set.entry(Sample2(sample)).or_insert(0) += 1;
                    }
                }
                progress.inc();
            }

            let (sample_vec, weight_vec): (Vec<_>, Vec<_>) = sample_set.into_iter().unzip();

            let weights: Vec<_> = weight_vec.into_iter()
                .enumerate()
                .map(|(i, x)| (i, x as f64))
                .collect();

            (sample_vec, weights)
        };

        let mut progress = ProgressBar::new((n.0 * n.1 * 4) as u64);
        progress.message(format!("Colliding {} samples...", samples.len()).as_str());
        progress.tick();

        let collide = {
            let mut collide = HashMap::new();

            let n = (n.0 as isize, n.1 as isize);

            let mut check_at_offset = |dx, dy, lx, ly, rx, ry| {
                let mut bitsets = Vec::new();
                for &Sample2(ref l) in samples.iter() {
                    let mut bs = BitSet::with_capacity(samples.len());
                    'rcheck: for (s, &Sample2(ref r)) in samples.iter().enumerate() {
                        for i in 0..dx {
                            for j in 0..dy {
                                let p_l = l[((lx + i) as usize, (ly + j) as usize)];
                                let p_r = r[((rx + i) as usize, (ry + j) as usize)];
                                if p_l != p_r {
                                    continue 'rcheck;
                                }
                            }
                        }
                        bs.insert(s);
                    }
                    bitsets.push(bs);
                }
                progress.inc();
                bitsets
            };

            for dx in 0..n.0 {
                for dy in 0..n.1 {
                    collide.insert(Offset2(dx, dy),
                                   check_at_offset(n.0 - dx, n.1 - dy, dx, dy, 0, 0));
                    collide.insert(Offset2(-dx, dy),
                                   check_at_offset(n.0 - dx, n.1 - dy, 0, dy, dx, 0));
                    collide.insert(Offset2(dx, -dy),
                                   check_at_offset(n.0 - dx, n.1 - dy, dx, 0, 0, dy));
                    collide.insert(Offset2(-dx, -dy),
                                   check_at_offset(n.0 - dx, n.1 - dy, 0, 0, dx, dy));
                }
            }

            collide
        };

        Source2 {
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
        assert_eq!(cfg.len(), 1);
    }
}


impl<P> Source for Source2<P>
    where P: Eq + Hash + Copy
{
    type Dims = Ix2;
    type Periodicity = (bool, bool);
    type Pixel = P;


    fn wave_dims(&self, dims: (usize, usize), periodic: (bool, bool)) -> (usize, usize) {
        (if periodic.0 {
            dims.0
        } else {
            dims.0 - (self.n.0 - 1)
        },
         if periodic.1 {
            dims.1
        } else {
            dims.1 - (self.n.1 - 1)
        })
    }


    fn initial_state(&self, pos: Ix2) -> State<Ix2> {
        let cfg = BitSet::from_bit_vec(BitVec::from_elem(self.samples.len(), true));
        let entropy = self.entropy(&cfg);
        State {
            pos: pos,
            entropy: entropy,
            cfg: cfg,
        }
    }


    fn constrain(&self,
                 states: Array2<RefCell<State<Ix2>>>,
                 pos: Ix2,
                 periodic: Self::Periodicity,
                 val: P)
                 -> Option<Array2<RefCell<State<Ix2>>>> {
        let n = (self.n.0 as isize, self.n.1 as isize);
        let pid = self.inverse_palette[&val];

        for i in 0..n.0 {
            let dim_adj_x = (states.dim().0 as isize - i) as usize;

            for j in 0..n.1 {
                let dim_adj_y = (states.dim().1 as isize - j) as usize;

                let subject_pos = ((pos.0 + dim_adj_x) % states.dim().0,
                                   (pos.1 + dim_adj_y) % states.dim().1);

                let mut subject = states[subject_pos].borrow_mut();

                if !periodic.0 && (pos.0 as isize - subject.pos.0 as isize).abs() >= n.0 {
                    continue;
                }

                if !periodic.1 && (pos.1 as isize - subject.pos.1 as isize).abs() >= n.1 {
                    continue;
                }

                subject.cfg = subject.cfg
                    .iter()
                    .filter(|&idx| self.samples[idx].0[(i as usize, j as usize)] == pid)
                    .collect();

                subject.entropy = self.entropy(&subject.cfg);

                if !(subject.entropy >= 0.0) {
                    debug!("Destroyed wave position {:?}'s hopes and dreams.",
                           subject_pos);
                }
            }
        }

        self.propagate(states, pos, periodic)
    }


    fn propagate(&self,
                 states: Array2<RefCell<State<Ix2>>>,
                 observe: Ix2,
                 periodic: Self::Periodicity)
                 -> Option<Array2<RefCell<State<Ix2>>>> {
        let n = (self.n.0 as isize, self.n.1 as isize);

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
                        if i == 0 && j == 0 {
                            continue;
                        }

                        let dim_adj_y = (states.dim().1 as isize + j) as usize;

                        let subject_pos = ((focus.pos.0 + dim_adj_x) % states.dim().0,
                                           (focus.pos.1 + dim_adj_y) % states.dim().1);

                        let mut subject = states[subject_pos].borrow_mut();

                        if !periodic.0 &&
                           (focus.pos.0 as isize - subject.pos.0 as isize).abs() >= n.0 {
                            continue;
                        }

                        if !periodic.1 &&
                           (focus.pos.1 as isize - subject.pos.1 as isize).abs() >= n.1 {
                            continue;
                        }

                        let mut subject_dirty = false;

                        let mut focus_allowed = BitSet::new();
                        let mut subject_allowed = BitSet::new();

                        loop {
                            focus_allowed.clear();
                            subject_allowed.clear();

                            for focus_cfg in focus.cfg.iter() {
                                subject_allowed.union_with(&self.collide[&Offset2(i, j)][focus_cfg]);
                            }

                            for subject_cfg in subject.cfg.iter() {
                                focus_allowed.union_with(&self.collide[&Offset2(-i, -j)][subject_cfg]);
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

            if focus_dirty {
                queue.push_back(focus.pos);
            }
        }

        return Some(states);
    }


    fn observe<R: Rng>(&self,
                       mut states: Array2<RefCell<State<Ix2>>>,
                       observe: Ix2,
                       periodic: (bool, bool),
                       rng: &mut R)
                       -> Option<Array2<RefCell<State<Ix2>>>> {
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


    fn resolve(&self, dim: Self::Dims, wave: ArrayView2<RefCell<State<Ix2>>>) -> Array2<P> {
        Array::from_shape_fn(dim, |(x, y)| {
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
            self.palette[self.samples[wave[(wx, wy)]
                        .borrow()
                        .cfg
                        .iter()
                        .next()
                        .unwrap()]
                    .0[(dx, dy)]
                .0 as usize]
        })
    }
}
